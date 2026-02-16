# Scheduler 重构设计文档

> **文档版本**: v1.0
> **更新日期**: 2026-02-15
> **状态**: 待讨论

---

## 一、问题分析

### 1.1 核心问题：调度与资源管理耦合

`ResourceManagerV1` 类承担了两个本应分离的职责：

```
┌─────────────────────────────────────────────────────────────┐
│                    ResourceManagerV1                         │
│                      (1472 行)                               │
├─────────────────────────────────────────────────────────────┤
│  职责1: 资源管理 (Block 分配/回收/前缀缓存)                    │
│  职责2: 调度决策 (waiting/running 队列、prefill/decode/preempt)│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 schedule() 方法分析 (315 行)

`ResourceManagerV1.schedule()` 是系统中最复杂的方法，混合了：

| 逻辑类型 | 行数(约) | 说明 |
|---------|---------|------|
| RUNNING 队列遍历 | 150 | 判断 decode/prefill/preempt |
| WAITING 队列遍历 | 130 | 新请求调度 |
| Block 分配 | 50 | 分散在各处 |
| 抢占触发 | 30 | `_trigger_preempt` 调用 |
| 前缀缓存 | 20 | `get_prefix_cached_blocks` 调用 |

### 1.3 当前调度层次混乱

```
LocalScheduler          →  只是请求队列，不做调度决策
       ↓
EngineService           →  调度协调 + IPC (270行 _schedule_request_to_worker_v1)
       ↓
ResourceManagerV1       →  真正的调度决策 + 资源分配 (315行 schedule)
```

**问题**：
1. `LocalScheduler` 名不副实
2. `EngineService` 与 `ResourceManagerV1` 职责边界模糊
3. 调度逻辑分散在三层，难以理解和维护

### 1.4 对比 vLLM 架构

| 组件 | vLLM | FastDeploy (当前) |
|------|------|-------------------|
| Scheduler | 纯调度决策 | 不存在独立组件 |
| BlockManager | 纯 Block 管理 | ResourceManagerV1 混合 |
| Engine | 协调者 | EngineService 承担过多 |

---

## 二、目标架构

### 2.1 职责分离

```
┌─────────────────────────────────────────────────────────────┐
│                      目标架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  RequestQueue (原 LocalScheduler)                            │
│  ├── put_requests()     # 请求入队                           │
│  ├── get_requests()     # 请求出队                           │
│  └── 不做调度决策                                             │
│           ↓                                                  │
│  Scheduler (新增)                                            │
│  ├── add_request()      # 添加到 waiting                     │
│  ├── schedule()         # 返回调度决策                        │
│  ├── finish_request()   # 完成请求                           │
│  └── 不执行 Block 分配                                        │
│           ↓                                                  │
│  BlockManager (从 ResourceManagerV1 抽取)                    │
│  ├── allocate()         # 分配 Block                         │
│  ├── free()             # 释放 Block                         │
│  ├── get_prefix_cache() # 前缀缓存匹配                        │
│  └── 不管请求队列                                             │
│           ↓                                                  │
│  EngineService (简化)                                        │
│  └── 只做协调，不含调度逻辑                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
请求到达
    ↓
RequestQueue.put_requests()
    ↓
EngineService 主循环:
    ↓
    requests = RequestQueue.get_requests()
    ↓
    for req in requests:
        Scheduler.add_request(req)
    ↓
    schedule_output = Scheduler.schedule()  ← 只返回决策
    ↓
    for req in schedule_output.prefills:
        BlockManager.allocate(req)          ← 执行分配
    ↓
    WorkerQueue.put_tasks(tasks)
```

---

## 三、接口定义

### 3.1 ScheduleOutput (调度输出)

```python
from dataclasses import dataclass, field
from typing import List
from fastdeploy.engine.request import Request

@dataclass
class ScheduledPrefillRequest:
    """Prefill 调度结果"""
    request: Request
    num_new_tokens: int           # 本次 prefill 的 token 数
    num_new_blocks: int           # 需要分配的 block 数
    cached_block_ids: List[int] = field(default_factory=list)  # 前缀缓存命中的 block

@dataclass
class ScheduledDecodeRequest:
    """Decode 调度结果"""
    request: Request
    num_new_blocks: int           # 需要分配的 block 数 (通常是 enc_dec_block_num)

@dataclass
class PreemptedRequest:
    """被抢占的请求"""
    request: Request
    reason: str                   # 抢占原因

@dataclass
class ScheduleOutput:
    """Scheduler.schedule() 的返回值"""

    # 需要 prefill 的请求 (包含 WAITING 新调度的 + RUNNING 中继续 prefill 的)
    prefills: List[ScheduledPrefillRequest] = field(default_factory=list)

    # 需要 decode 的请求
    decodes: List[ScheduledDecodeRequest] = field(default_factory=list)

    # 被抢占的请求
    preempted: List[PreemptedRequest] = field(default_factory=list)

    # 出错的请求 (异步预处理失败等)
    errors: List[tuple[str, str]] = field(default_factory=list)  # (request_id, error_msg)

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes and not self.preempted

    @property
    def num_scheduled(self) -> int:
        return len(self.prefills) + len(self.decodes)
```

### 3.2 Scheduler 接口

```python
from typing import Optional, List
from dataclasses import dataclass
from collections import deque

class Scheduler:
    """
    纯调度决策组件，不执行资源分配。

    职责：
    - 维护 waiting/running 队列
    - 决定哪些请求 prefill/decode/preempt
    - 计算需要分配多少 block (但不执行分配)

    不负责：
    - Block 分配/回收 (由 BlockManager 负责)
    - 前缀缓存匹配 (由 BlockManager 负责，Scheduler 只使用结果)
    - IPC 通信 (由 EngineService 负责)
    """

    def __init__(
        self,
        config: 'EngineConfig',
        block_manager: 'BlockManager',
    ):
        """
        初始化 Scheduler。

        Args:
            config: 引擎配置，包含 max_num_seqs, max_num_batched_tokens 等
            block_manager: Block 管理器，用于查询可用资源
        """
        self.config = config
        self.block_manager = block_manager

        # 请求队列
        self.waiting: deque[Request] = deque()
        self.running: List[Request] = []
        self.requests: dict[str, Request] = {}  # request_id -> Request

    # ==================== 请求管理 ====================

    def add_request(self, request: 'Request') -> None:
        """
        添加新请求到 waiting 队列。

        Args:
            request: 新请求

        Note:
            - 异步预处理在此触发
            - 请求状态设为 WAITING
        """
        ...

    def abort_request(self, request_id: str) -> Optional['Request']:
        """
        中止请求。

        Args:
            request_id: 要中止的请求 ID

        Returns:
            被中止的请求，如果不存在则返回 None
        """
        ...

    def finish_request(self, request_id: str) -> Optional['Request']:
        """
        标记请求完成，从 running 队列移除。

        Args:
            request_id: 完成的请求 ID

        Returns:
            完成的请求，如果不存在则返回 None

        Note:
            - 不负责释放 Block，由调用方通过 BlockManager 释放
        """
        ...

    # ==================== 核心调度 ====================

    def schedule(self) -> ScheduleOutput:
        """
        执行一轮调度，返回调度决策。

        Returns:
            ScheduleOutput 包含：
            - prefills: 需要 prefill 的请求及其 token/block 需求
            - decodes: 需要 decode 的请求及其 block 需求
            - preempted: 被抢占的请求
            - errors: 出错的请求

        调度逻辑：
        1. 遍历 RUNNING 队列:
           - 已完成 prefill 的 → 判断是否需要 decode (分配新 block)
           - 未完成 prefill 的 → 继续 prefill
           - 资源不足时 → 触发抢占
        2. 遍历 WAITING 队列:
           - 检查资源是否充足
           - 充足则移入 RUNNING 并加入 prefills

        Note:
            - 此方法只返回决策，不执行 Block 分配
            - 调用方根据返回的 num_new_blocks 通过 BlockManager 分配
        """
        ...

    # ==================== 状态查询 ====================

    @property
    def num_waiting(self) -> int:
        """等待队列长度"""
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        """运行队列长度"""
        return len(self.running)

    def has_pending_requests(self) -> bool:
        """是否有待处理的请求"""
        return bool(self.waiting or self.running)

    def get_request(self, request_id: str) -> Optional['Request']:
        """获取请求"""
        return self.requests.get(request_id)
```

### 3.3 BlockManager 接口

```python
from typing import List, Optional, Tuple

class BlockManager:
    """
    纯资源管理组件，不涉及调度决策。

    职责：
    - Block 分配/回收
    - 前缀缓存匹配
    - 资源可用性查询

    不负责：
    - 请求队列管理 (由 Scheduler 负责)
    - 调度决策 (由 Scheduler 负责)
    """

    def __init__(
        self,
        config: 'EngineConfig',
        cache_manager: 'PrefixCacheManager',
    ):
        """
        初始化 BlockManager。

        Args:
            config: 引擎配置
            cache_manager: 底层缓存管理器
        """
        ...

    # ==================== 资源查询 ====================

    def can_allocate(self, num_blocks: int) -> bool:
        """
        检查是否能分配指定数量的 block。

        Args:
            num_blocks: 需要的 block 数量

        Returns:
            True 如果可以分配
        """
        ...

    def get_available_blocks(self) -> int:
        """返回可用 block 数量"""
        ...

    def get_available_slots(self) -> int:
        """返回可用 batch slot 数量"""
        ...

    # ==================== Block 分配 ====================

    def allocate(
        self,
        request: 'Request',
        num_blocks: int,
    ) -> List[int]:
        """
        为请求分配 block。

        Args:
            request: 请求对象
            num_blocks: 需要分配的 block 数

        Returns:
            分配的 block ID 列表

        Raises:
            RuntimeError: 如果资源不足
        """
        ...

    def free(self, request: 'Request') -> None:
        """
        释放请求占用的所有 block。

        Args:
            request: 请求对象
        """
        ...

    # ==================== 前缀缓存 ====================

    def match_prefix_cache(
        self,
        request: 'Request',
    ) -> Tuple[List[int], int]:
        """
        匹配前缀缓存。

        Args:
            request: 请求对象

        Returns:
            (cached_block_ids, matched_token_num)
            - cached_block_ids: 命中的缓存 block ID
            - matched_token_num: 命中的 token 数量
        """
        ...

    # ==================== 批次槽位管理 ====================

    def allocate_slot(self) -> int:
        """
        分配一个 batch slot。

        Returns:
            分配的 slot 索引

        Raises:
            RuntimeError: 如果没有可用 slot
        """
        ...

    def free_slot(self, slot_idx: int) -> None:
        """释放 batch slot"""
        ...
```

---

## 四、迁移计划

### 4.1 Phase 1: 抽取 ScheduleOutput (低风险)

**目标**: 定义调度输出数据结构

**改动**:
```
新增文件: fastdeploy/engine/sched/schedule_output.py
```

**内容**: 只包含 dataclass 定义，不改现有逻辑

---

### 4.2 Phase 2: 抽取 BlockManager (中风险)

**目标**: 从 ResourceManagerV1 抽取纯资源管理逻辑

**涉及方法**:

| 方法 | 行号 | 迁移到 |
|------|------|--------|
| `can_allocate_gpu_blocks` 调用 | 多处 | BlockManager.can_allocate() |
| `allocate_gpu_blocks` 调用 | 多处 | BlockManager.allocate() |
| `_free_blocks` | 249-266 | BlockManager.free() |
| `get_prefix_cached_blocks` | 1089-1151 | BlockManager.match_prefix_cache() |
| `get_available_position` | 继承自父类 | BlockManager.allocate_slot() |

**策略**:
- 创建 BlockManager 类，内部委托给现有 cache_manager
- ResourceManagerV1 改为持有 BlockManager 实例
- 逐步将直接调用改为通过 BlockManager

---

### 4.3 Phase 3: 抽取 Scheduler (高风险)

**目标**: 从 ResourceManagerV1 抽取调度决策逻辑

**核心改动**: 重构 `schedule()` 方法

```python
# 原来 (ResourceManagerV1.schedule)
def schedule(self):
    # 1. 遍历 running，判断 decode/prefill
    # 2. 分配 block (直接调用 cache_manager)
    # 3. 遍历 waiting，调度新请求
    # 4. 分配 block (直接调用 cache_manager)
    return scheduled_reqs, error_reqs

# 重构后 (Scheduler.schedule)
def schedule(self) -> ScheduleOutput:
    output = ScheduleOutput()

    # 1. 遍历 running，判断 decode/prefill
    for req in self.running:
        if self._should_decode(req):
            num_blocks = self._calc_decode_blocks(req)
            output.decodes.append(ScheduledDecodeRequest(req, num_blocks))
        elif self._should_prefill(req):
            num_tokens, num_blocks = self._calc_prefill_needs(req)
            output.prefills.append(ScheduledPrefillRequest(req, num_tokens, num_blocks))

    # 2. 遍历 waiting，调度新请求 (不在这里分配 block)
    while self._can_schedule_new():
        req = self.waiting.popleft()
        num_tokens, num_blocks = self._calc_prefill_needs(req)
        output.prefills.append(ScheduledPrefillRequest(req, num_tokens, num_blocks))
        self.running.append(req)

    return output  # 只返回决策，不执行分配
```

**EngineService 使用方式**:

```python
# EngineService._schedule_loop (简化后)
def _schedule_loop(self):
    # 1. 获取调度决策
    output = self.scheduler.schedule()

    # 2. 根据决策分配资源
    tasks = []
    for prefill in output.prefills:
        # 先匹配前缀缓存
        if self.config.cache_config.enable_prefix_caching:
            cached_blocks, _ = self.block_manager.match_prefix_cache(prefill.request)
            prefill.request.block_tables.extend(cached_blocks)
        # 分配剩余 block
        new_blocks = self.block_manager.allocate(prefill.request, prefill.num_new_blocks)
        prefill.request.block_tables.extend(new_blocks)
        tasks.append(self._prepare_prefill_task(prefill))

    for decode in output.decodes:
        new_blocks = self.block_manager.allocate(decode.request, decode.num_new_blocks)
        decode.request.block_tables.extend(new_blocks)
        tasks.append(self._prepare_decode_task(decode))

    for preempted in output.preempted:
        self.block_manager.free(preempted.request)
        tasks.append(self._prepare_preempt_task(preempted))

    # 3. 发送到 Worker
    self.worker_queue.put_tasks(tasks)
```

---

### 4.4 Phase 4: 清理与优化

**目标**: 删除旧代码，统一接口

**改动**:
- 删除 ResourceManagerV1 中的调度逻辑
- ResourceManagerV1 改名为 BlockManager 或标记废弃
- 更新所有调用点

---

## 五、兼容性设计

### 5.1 环境变量切换

```bash
# 使用新 Scheduler (默认关闭)
export FD_USE_NEW_SCHEDULER=1

# 回退到旧逻辑
export FD_USE_NEW_SCHEDULER=0
```

### 5.2 代码中的切换逻辑

```python
class EngineService:
    def __init__(self, cfg):
        if envs.FD_USE_NEW_SCHEDULER:
            self.block_manager = BlockManager(cfg, cache_manager)
            self.scheduler = Scheduler(cfg, self.block_manager)
        else:
            self.resource_manager = ResourceManagerV1(...)  # 旧逻辑

    def _schedule_request_to_worker(self):
        if envs.FD_USE_NEW_SCHEDULER:
            return self._schedule_with_new_scheduler()
        else:
            return self._schedule_request_to_worker_v1()  # 旧逻辑
```

---

## 六、测试策略

### 6.1 单元测试

```python
# tests/engine/sched/test_scheduler.py

class TestScheduler:
    def test_add_request_to_waiting(self):
        """测试请求添加到 waiting 队列"""

    def test_schedule_prefill_from_waiting(self):
        """测试从 waiting 调度 prefill"""

    def test_schedule_decode_from_running(self):
        """测试 running 中的 decode 调度"""

    def test_preemption_when_no_blocks(self):
        """测试资源不足时的抢占"""

    def test_schedule_output_correctness(self):
        """测试 ScheduleOutput 内容正确性"""
```

### 6.2 集成测试

```python
# tests/engine/test_scheduler_integration.py

class TestSchedulerIntegration:
    def test_prefill_decode_e2e(self):
        """端到端测试: prefill -> decode -> finish"""

    def test_preemption_and_reschedule(self):
        """测试抢占后重新调度"""

    def test_prefix_cache_hit(self):
        """测试前缀缓存命中"""
```

### 6.3 性能测试

- 对比新旧 Scheduler 的调度延迟
- 验证吞吐量不下降

---

## 七、文件结构

```
fastdeploy/engine/sched/
├── __init__.py
├── scheduler.py              # 新增: Scheduler 类
├── block_manager.py          # 新增: BlockManager 类
├── schedule_output.py        # 新增: ScheduleOutput 数据类
├── resource_manager_v1.py    # 保留: 逐步迁移
└── scheduler_metrics_logger.py
```

---

## 八、待讨论问题

1. **P/D 分离模式**: `preallocate_resource_in_p/d` 等方法如何迁移？
2. **多模态支持**: `_get_num_new_tokens` 中的多模态逻辑如何处理？
3. **Extend Tables**: 这个特性是否需要在新 Scheduler 中支持？
4. **抢占策略**: 当前是 LIFO，是否需要支持其他策略？
5. **Metrics**: 调度指标收集逻辑如何迁移？

---

## 九、参考资料

- [vLLM Scheduler 实现](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)
- [FastDeploy EngineService 重构计划](./engine_service_phased_refactor.md)
