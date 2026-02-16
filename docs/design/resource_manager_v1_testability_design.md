# ResourceManagerV1 可测试性设计方案

## 背景

ResourceManagerV1 是 FastDeploy 中负责 GPU 内存块管理和请求调度的核心组件，位于 [resource_manager_v1.py](../../fastdeploy/engine/sched/resource_manager_v1.py)。

---

## 调用场景与时机

### 整体架构位置

```
┌─────────────────────────────────────────────────────────────┐
│                      EngineService                          │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Scheduler  │  │ ResourceManagerV1│  │ TokenProcessor│  │
│  │  (HTTP/gRPC)│  │  (本文档重点)     │  │  (输出处理)   │  │
│  └──────┬──────┘  └────────┬─────────┘  └───────┬───────┘  │
│         │                  │                    │           │
│         ▼                  ▼                    ▼           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              EngineWorkerQueue (任务队列)             │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Worker (GPU 推理)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 调用时机与伪代码

#### 1. 初始化阶段（EngineService.__init__）

```python
# File: fastdeploy/engine/common_engine.py:117-125

class EngineService:
    def __init__(self, cfg):
        # 根据环境变量选择 V0 或 V1 调度器
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.resource_manager = ResourceManagerV1(
                max_num_seqs=cfg.scheduler_config.max_num_seqs,
                config=cfg,
                tensor_parallel_size=cfg.parallel_config.tensor_parallel_size,
                splitwise_role=cfg.scheduler_config.splitwise_role,
                local_data_parallel_id=cfg.parallel_config.local_data_parallel_id,
            )
        else:
            self.resource_manager = ResourceManager(...)
```

**时机**：Engine 启动时，仅调用一次

---

#### 2. 调度循环（核心循环，持续运行）

```python
# File: fastdeploy/engine/common_engine.py:940-1060
# 伪代码简化版

def _schedule_request_to_worker_v1(self):
    """Scheduler thread main loop"""
    while not self.stop_event.is_set():

        # ========== Step 1: 获取新请求 ==========
        # 时机：waiting 队列为空时，从 HTTP/gRPC 获取新请求
        if len(self.resource_manager.waiting) == 0:
            tasks = self._fetch_requests_from_scheduler()
            for task in tasks:
                self.resource_manager.add_request(task)  # ← 调用点

        # ========== Step 2: 调度请求 ==========
        # 时机：每次循环都调用，决定哪些请求可以执行
        scheduled_tasks, error_tasks = self.resource_manager.schedule()  # ← 核心调用点

        # ========== Step 3: 发送到 Worker ==========
        # 时机：有任务被调度成功时
        if scheduled_tasks:
            self.engine_worker_queue.put_tasks(scheduled_tasks)

        # ========== Step 4: 处理错误 ==========
        if error_tasks:
            for request_id, error in error_tasks:
                self._send_error_response(request_id, error)

        # 无任务时短暂休眠
        if not scheduled_tasks and not error_tasks:
            time.sleep(0.005)
```

**关键调用点**：
- `add_request(task)` - 新请求到达时
- `schedule()` - 每次调度循环（约 200 次/秒）

---

#### 3. 请求完成处理（TokenProcessor 回调）

```python
# File: fastdeploy/output/token_processor.py:536-552
# 伪代码简化版

class TokenProcessor:
    def process_output(self, task, result):
        """Process inference output"""

        # 检查是否完成（EOS 或达到 max_tokens）
        if result.finished:
            task_id = task.request_id

            # ========== 调用 finish_requests ==========
            # 时机：请求推理完成时，回收 GPU 内存块
            if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                self.resource_manager.finish_requests_async(task_id)  # ← 调用点
            else:
                # V0 版本直接操作标志位
                self.resource_manager.stop_flags[index] = True
                self.resource_manager.tasks_list[index] = None
```

**时机**：每个请求推理完成时调用

---

#### 4. P/D 分离场景（Prefill/Decode 分离部署）

```python
# 伪代码 - Prefill 实例

def handle_prefill_instance():
    # Prefill 实例收到请求后直接加入 running 队列
    self.resource_manager.add_request_in_p(tasks)  # ← P 实例专用

    # 预分配资源
    self.resource_manager.preallocate_resource_in_p(request)

    # Prefill 完成后发送到 Decode 实例
    send_to_decode_instance(request)


# 伪代码 - Decode 实例

def handle_decode_instance():
    # 收到 Prefill 实例发来的请求
    prefilled_request = receive_from_prefill_instance()

    # 检查资源是否可用
    if self.resource_manager.has_resource_for_prefilled_req(request_id):
        # 预分配 Decode 资源
        self.resource_manager.preallocate_resource_in_d(request)
        # 加入调度
        self.resource_manager.add_prefilled_request(request)
```

---

#### 5. 抢占场景（资源不足时）

```python
# 伪代码 - 资源不足触发抢占

def schedule(self):
    # 当新请求无法分配资源时
    if not self.cache_manager.can_allocate_gpu_blocks(needed_blocks):
        # 检查是否可以抢占
        if self._can_preempt():
            # 抢占最后加入的请求（LIFO）
            preempted = self._trigger_preempt()
            # 被抢占的请求加入待重新调度集合
            self.to_be_rescheduled_request_id_set.add(preempted.request_id)


# Worker 完成抢占后回调
def on_preemption_complete(request_id):
    # 重新调度被抢占的请求
    self.resource_manager.reschedule_preempt_task(request_id)  # ← 调用点
```

---

### 调用频率总结

| 方法 | 调用时机 | 频率 |
|------|----------|------|
| `__init__()` | Engine 启动 | 1 次 |
| `add_request()` | 新请求到达 | ~100-1000 次/秒 |
| `schedule()` | 调度循环 | ~200 次/秒 |
| `finish_requests()` | 请求完成 | ~100-1000 次/秒 |
| `preempted_all()` | 资源不足/RL 训练 | 偶发 |
| `reschedule_preempt_task()` | 抢占后重调度 | 偶发 |
| `update_metrics()` | 每次 schedule 后 | ~200 次/秒 |

---

### 当前问题

| 问题 | 影响 |
|------|------|
| 依赖硬编码 | PrefixCacheManager、IPCSignal、ThreadPoolExecutor 在构造函数中直接创建，无法替换 |
| 全局状态耦合 | 直接访问 `main_process_metrics` 单例，测试间相互影响 |
| 测试样板代码 | 每个测试需要 30-40 行配置构建代码 |
| 清理负担 | IPCSignal 和 ThreadPoolExecutor 需要显式 `addCleanup()` |

**目标**：
1. 通过依赖注入实现依赖隔离，便于 Mock 和测试
2. 减少测试样板代码（30行 → 2行）
3. 明确职责边界，分离公开接口与内部实现
4. 保持向后兼容

---

## 重构策略：先建立 Gold Base，再重构

### 核心原则

**Mock 的正确性依赖于对真实类行为的准确理解**。因此：

```
┌─────────────────────────────────────────────────────────────────┐
│                      重构顺序                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 0          Phase 1           Phase 2          Phase 3   │
│  ┌─────────┐     ┌──────────┐      ┌─────────┐     ┌─────────┐│
│  │确定依赖类│ →  │建立依赖类│  →   │设计 Mock│  →  │重构     ││
│  │的行为   │     │Gold Base │      │（基于GB）│     │Manager  ││
│  └─────────┘     └──────────┘      └─────────┘     └─────────┘│
│       ↓               ↓                 ↓               ↓      │
│   输入/输出        特征测试          Mock 实现      依赖注入   │
│   状态变化       (真实对象)        (模拟契约)      (验证GB)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 0: 确定依赖类的行为契约

**目标**：明确 ResourceManagerV1 依赖的每个类的输入、输出、状态变化。

#### 需要分析的依赖类

| 依赖类 | 位置 | 核心职责 |
|--------|------|----------|
| PrefixCacheManager | `cache_manager/prefix_cache_manager.py` | GPU 块分配、回收、前缀匹配 |
| IPCSignal | `inter_communicator/ipc_signal.py` | 进程间共享数组 |
| EncoderCacheManager | `cache_manager/multimodal_cache_manager.py` | 多模态特征缓存 |
| main_process_metrics | `metrics/metrics.py` | Prometheus 指标 |

#### PrefixCacheManager 行为契约（示例）

```python
# 需要确定的行为：

class PrefixCacheManager:
    """
    输入/输出/状态变化 契约
    """

    # === 状态查询 ===
    @property
    def num_gpu_blocks(self) -> int:
        """输出：总 GPU 块数（初始化后不变）"""

    @property
    def gpu_free_block_list(self) -> list[int]:
        """输出：当前空闲块 ID 列表"""

    # === 分配操作 ===
    def can_allocate_gpu_blocks(self, num_blocks: int) -> bool:
        """
        输入：需要的块数
        输出：True/False
        状态：不变
        """

    def allocate_gpu_blocks(self, num_blocks: int, request_id: str = None) -> list[int]:
        """
        输入：需要的块数, 可选的请求ID
        输出：分配的块 ID 列表（可能为空）
        状态：gpu_free_block_list 减少
        错误：块不足时返回空列表（不抛异常）
        """

    # === 回收操作 ===
    def recycle_gpu_blocks(self, block_ids: list[int], request_id: str = None) -> None:
        """
        输入：要回收的块 ID 列表
        输出：None
        状态：gpu_free_block_list 增加
        """

    # === 前缀匹配 ===
    def request_match_blocks(self, request, block_size: int) -> tuple[list, int, dict]:
        """
        输入：请求对象, 块大小
        输出：(匹配的块列表, 匹配的token数, 统计信息dict)
        状态：可能更新内部 radix tree
        """
```

### Phase 1: 建立依赖类的 Gold Base（特征测试）

**目标**：用真实对象建立输入输出基准，确保我们理解真实行为。

```python
# tests/gold_base/test_prefix_cache_manager_behavior.py

class TestPrefixCacheManagerBehavior:
    """
    特征测试（Characterization Tests）
    目的：记录 PrefixCacheManager 的真实行为，作为 Mock 设计的依据
    """

    def test_allocate_reduces_free_blocks(self):
        """验证：分配后 free_blocks 减少"""
        # Arrange
        manager = create_real_cache_manager(num_gpu_blocks=100)
        initial_free = len(manager.gpu_free_block_list)

        # Act
        allocated = manager.allocate_gpu_blocks(10, "req-1")

        # Assert - 记录真实行为
        assert len(allocated) == 10
        assert len(manager.gpu_free_block_list) == initial_free - 10

    def test_allocate_when_insufficient_blocks(self):
        """验证：块不足时的行为"""
        manager = create_real_cache_manager(num_gpu_blocks=5)

        # Act - 请求超过可用数量
        allocated = manager.allocate_gpu_blocks(10, "req-1")

        # Assert - 记录真实行为：返回空列表？部分分配？抛异常？
        # 这里记录的就是 Gold Base
        assert allocated == []  # 或其他真实行为

    def test_recycle_increases_free_blocks(self):
        """验证：回收后 free_blocks 增加"""
        manager = create_real_cache_manager(num_gpu_blocks=100)
        allocated = manager.allocate_gpu_blocks(10, "req-1")
        free_after_alloc = len(manager.gpu_free_block_list)

        # Act
        manager.recycle_gpu_blocks(allocated, "req-1")

        # Assert
        assert len(manager.gpu_free_block_list) == free_after_alloc + 10

    def test_request_match_blocks_return_structure(self):
        """验证：request_match_blocks 的返回结构"""
        manager = create_real_cache_manager(num_gpu_blocks=100)
        request = create_test_request(prompt_token_ids=[1,2,3,4])

        # Act
        result = manager.request_match_blocks(request, block_size=16)

        # Assert - 记录返回结构
        assert isinstance(result, tuple)
        assert len(result) == 3
        blocks, token_num, info = result
        assert isinstance(blocks, list)
        assert isinstance(token_num, int)
        assert isinstance(info, dict)
        # 记录 info dict 包含哪些 key
        assert "gpu_match_token_num" in info or info == {}  # 根据真实行为记录
```

### Phase 2: 基于 Gold Base 设计 Mock

**原则**：Mock 的行为必须与 Gold Base 记录的真实行为一致。

```python
class MockCacheManager:
    """
    基于 Gold Base 设计的 Mock
    每个方法的行为都有 Gold Base 测试作为依据
    """

    def allocate_gpu_blocks(self, num_blocks: int, request_id=None) -> list[int]:
        """
        行为依据：test_allocate_reduces_free_blocks
                 test_allocate_when_insufficient_blocks
        """
        self.allocate_calls.append((num_blocks, request_id))

        # 基于 Gold Base：块不足时返回空列表
        if not self.can_allocate_gpu_blocks(num_blocks):
            return []

        # 基于 Gold Base：分配后 free_blocks 减少
        allocated = self._free_blocks[:num_blocks]
        self._free_blocks = self._free_blocks[num_blocks:]
        return allocated

    def request_match_blocks(self, request, block_size):
        """
        行为依据：test_request_match_blocks_return_structure
        返回结构必须与真实对象一致
        """
        return (
            self._match_blocks,      # list
            self._match_token_num,   # int
            self._match_info         # dict with specific keys
        )
```

### Phase 3: 重构 ResourceManagerV1 + 用 Gold Base 验证

**验证**：重构后，运行 Gold Base 测试确保行为不变。

```bash
# 重构后验证
pytest tests/gold_base/ -v  # Gold Base 测试必须全部通过
pytest tests/v1/test_resource_manager_v1.py -v  # 现有测试必须通过
```

---

## 实施计划调整

| 阶段 | 内容 | 产出 |
|------|------|------|
| **Phase 0** | 分析依赖类，文档化输入/输出/状态契约 | 行为契约文档 |
| **Phase 1** | 编写特征测试，建立 Gold Base | `tests/gold_base/` |
| **Phase 2** | 基于 Gold Base 设计 Mock 实现 | `interfaces.py`, `testing.py` |
| **Phase 3** | 重构 ResourceManagerV1，添加依赖注入 | 修改 `resource_manager_v1.py` |
| **Phase 4** | 用 Gold Base 验证重构正确性 | 测试全绿 |

---

## 1. 职责边界划分

### 1.1 公开接口（External Interface）

供 Scheduler 和 Engine 调用的稳定 API：

| 方法 | 输入 | 输出 | 职责 |
|------|------|------|------|
| `add_request(request)` | Request | None | 将请求加入等待队列 |
| `schedule()` | 无 | `(scheduled, errors)` | 调度请求执行 |
| `finish_requests(ids)` | 请求 ID 列表 | None | 完成请求并回收资源 |
| `preempted_all()` | 无 | 被抢占任务列表 | 抢占所有运行中请求 |
| `reschedule_preempt_task(id, fn)` | 请求 ID, 回调 | None | 重新调度被抢占请求 |
| `update_metrics()` | 无 | None | 更新 Prometheus 指标 |
| `available_batch()` | 无 | int | 可用批次槽位数 |
| `available_block_num()` | 无 | int | 可用 GPU 块数 |

### 1.2 内部接口（Internal Interface）

实现细节，以 `_` 开头，不应被外部直接调用：

```
_prepare_prefill_task()    # 创建预填充任务
_prepare_decode_task()     # 创建解码任务
_prepare_preempt_task()    # 创建抢占任务
_free_blocks()             # 释放请求的块
_trigger_preempt()         # 触发抢占
_can_preempt()             # 判断是否可抢占
_get_num_new_tokens()      # 计算待处理 token 数
_update_mm_hashes()        # 更新多模态哈希
_download_features()       # 下载 BOS 特征
```

---

## 2. 依赖接口定义（Protocols）

### 2.1 ICacheManager

```python
from typing import Protocol, Optional, Any, Sequence

class ICacheManager(Protocol):
    """
    Interface for GPU block cache management.

    Implementations:
    - PrefixCacheManager (production)
    - MockCacheManager (test)
    """

    @property
    def num_gpu_blocks(self) -> int:
        """Total number of GPU blocks available."""
        ...

    @property
    def gpu_free_block_list(self) -> Sequence[int]:
        """List of free GPU block IDs."""
        ...

    def can_allocate_gpu_blocks(self, num_blocks: int) -> bool:
        """Check if allocation is possible."""
        ...

    def allocate_gpu_blocks(
        self, num_blocks: int, request_id: Optional[str] = None
    ) -> list[int]:
        """Allocate blocks, return allocated block IDs."""
        ...

    def recycle_gpu_blocks(
        self, block_ids: Sequence[int], request_id: Optional[str] = None
    ) -> None:
        """Return blocks to free pool."""
        ...

    def release_block_ids(self, request: Any) -> None:
        """Release all blocks associated with a request."""
        ...

    def request_match_blocks(
        self, request: Any, block_size: int
    ) -> tuple[list[int], int, dict]:
        """Match and return prefix cached blocks."""
        ...

    def update_cache_blocks(
        self, request: Any, block_size: int, computed_tokens: int
    ) -> None:
        """Update cache state after computation."""
        ...
```

### 2.2 IIPCSignal

```python
class IIPCSignal(Protocol):
    """
    Interface for inter-process signal communication.

    Implementations:
    - IPCSignal (production) - SharedMemory based
    - MockIPCSignal (test) - in-memory numpy array
    """

    @property
    def value(self) -> np.ndarray:
        """Get signal value array."""
        ...

    def clear(self) -> None:
        """Release shared memory resources."""
        ...
```

### 2.3 IMetricsRecorder

```python
class IMetricsRecorder(Protocol):
    """
    Interface for metrics recording, decouples from Prometheus.

    Implementations:
    - PrometheusMetricsRecorder (production)
    - NoOpMetricsRecorder (test) - discards all metrics
    - RecordingMetricsRecorder (test) - records for assertions
    """

    def set_max_batch_size(self, value: int) -> None: ...
    def set_batch_size(self, value: int) -> None: ...
    def set_gpu_cache_usage_perc(self, value: float) -> None: ...
    def set_available_gpu_block_num(self, value: int) -> None: ...
    def inc_prefix_cache_token_num(self, value: int) -> None: ...
```

### 2.4 IExecutorPool

```python
class IExecutorPool(Protocol):
    """
    Interface for async task execution.

    Implementations:
    - ThreadPoolExecutor (production)
    - MockExecutorPool (test) - synchronous execution
    """

    def submit(self, fn: Callable, *args, **kwargs) -> Future: ...
    def shutdown(self, wait: bool = True) -> None: ...
```

---

## 3. 依赖容器

```python
@dataclass
class ResourceManagerDependencies:
    """
    Container for injectable dependencies.

    Usage:
        # Production - all None, created internally
        deps = ResourceManagerDependencies()

        # Testing - inject mocks
        deps = ResourceManagerDependencies(
            cache_manager=MockCacheManager(),
            metrics=NoOpMetricsRecorder(),
        )
    """
    cache_manager: Optional[ICacheManager] = None
    encoder_cache: Optional[IMultimodalCacheManager] = None
    processor_cache: Optional[IMultimodalCacheManager] = None
    ipc_signal: Optional[IIPCSignal] = None
    finish_executor: Optional[IExecutorPool] = None
    preprocess_executor: Optional[IExecutorPool] = None
    metrics: Optional[IMetricsRecorder] = None
```

---

## 4. 重构后的构造函数

```python
class ResourceManagerV1(ResourceManager):
    def __init__(
        self,
        max_num_seqs: int,
        config: FDConfig,
        tensor_parallel_size: int,
        splitwise_role: str,
        local_data_parallel_id: int = 0,
        dependencies: Optional[ResourceManagerDependencies] = None,  # NEW
    ):
        deps = dependencies or ResourceManagerDependencies()

        # Cache manager: injected or created
        if deps.cache_manager is not None:
            self._init_base_state(max_num_seqs, config, deps.cache_manager)
        else:
            super().__init__(...)  # Original behavior

        # Metrics: injected or default Prometheus
        self._metrics = deps.metrics or PrometheusMetricsRecorder()

        # IPC signal: injected or created
        self.need_block_num_signal = deps.ipc_signal or self._create_ipc_signal()

        # Executors: injected or created
        self.finish_execution_pool = deps.finish_executor or ThreadPoolExecutor(1)
        self.async_preprocess_pool = deps.preprocess_executor or ThreadPoolExecutor(4)

        # ... rest unchanged
```

**向后兼容**：当 `dependencies=None` 时，行为与当前完全一致。

---

## 5. Mock 设计原则

### 5.1 Mock 粒度选择

**核心问题**：Mock 太细则测试变成"验证调用顺序"；Mock 太粗则失去测试价值。

#### 粒度选择矩阵

| 依赖类型 | Mock 策略 | 原因 |
|----------|-----------|------|
| **CacheManager** | 状态型 Mock（有内部状态） | 核心依赖，需模拟分配/回收的状态变化 |
| **IPCSignal** | 简单替身（仅数据容器） | 只是共享内存的包装，无复杂逻辑 |
| **Metrics** | NoOp 或 Recording | 副作用型依赖，测试时通常不关心 |
| **Executor** | 同步执行 Mock | 消除异步不确定性 |

#### Mock 粒度原则

```
┌─────────────────────────────────────────────────────────────┐
│                     Mock 粒度光谱                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stub ◄──────────────────────────────────────────► Fake    │
│  (固定返回值)                                    (简化实现)  │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │ NoOp    │    │ Stub    │    │ Mock    │    │ Fake    │ │
│  │ 空实现  │    │ 固定值  │    │ 有状态  │    │ 简化版  │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│       ↑              ↑              ↑              ↑       │
│    Metrics      IPCSignal    CacheManager    (不推荐)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 "够用"的标准

Mock 需要满足以下条件才算"够用"：

#### 1. 能触发被测对象的关键代码路径

```python
# CacheManager Mock 需要支持的场景：
class MockCacheManager:
    # 场景 1: 正常分配
    def can_allocate_gpu_blocks(self, n) -> bool:
        return len(self._free_blocks) >= n

    # 场景 2: 分配失败（触发抢占逻辑）
    def set_can_allocate(self, value: bool):
        self._can_allocate = value

    # 场景 3: 部分分配（边界情况）
    def set_available_blocks(self, n: int):
        self._free_blocks = list(range(n))
```

#### 2. 保持状态一致性（状态型 Mock）

```python
class MockCacheManager:
    """
    关键：分配后 free_blocks 要减少，回收后要增加。
    这不是"过度实现"，而是维护契约一致性。
    """
    def allocate_gpu_blocks(self, num_blocks, request_id=None):
        # 状态变更：分配后可用块减少
        allocated = self._free_blocks[:num_blocks]
        self._free_blocks = self._free_blocks[num_blocks:]
        return allocated

    def recycle_gpu_blocks(self, block_ids, request_id=None):
        # 状态变更：回收后可用块增加
        self._free_blocks.extend(block_ids)
```

#### 3. 不实现不需要的复杂逻辑

```python
class MockCacheManager:
    """
    不需要实现的：
    - Radix Tree 前缀匹配（prefix caching 的核心算法）
    - LRU 淘汰策略
    - 跨进程共享内存

    为什么不需要：这些是 CacheManager 的内部实现，
    ResourceManagerV1 只关心"能否分配"和"分配了哪些块"
    """

    def request_match_blocks(self, request, block_size):
        # 简化：直接返回空匹配，不实现真正的前缀匹配
        return ([], 0, {"gpu_match_token_num": 0, ...})
```

### 5.3 MockCacheManager 完整设计

```python
class MockCacheManager:
    """
    Mock 设计原则：
    1. 模拟接口契约，不模拟内部算法
    2. 维护必要的状态一致性
    3. 提供测试控制点（set_xxx 方法）
    4. 记录调用用于断言（xxx_calls 列表）
    """

    def __init__(self, num_gpu_blocks: int = 100):
        # === 状态 ===
        self._num_gpu_blocks = num_gpu_blocks
        self._free_blocks = list(range(num_gpu_blocks))

        # === 控制点（让测试能触发不同路径）===
        self._can_allocate = True
        self._match_result = ([], 0, {})  # prefix cache 匹配结果

        # === 调用记录（用于断言）===
        self.allocate_calls: list[tuple] = []
        self.recycle_calls: list[tuple] = []
        self.release_calls: list[str] = []

    # --- 核心接口：需要维护状态 ---

    @property
    def num_gpu_blocks(self) -> int:
        return self._num_gpu_blocks

    @property
    def gpu_free_block_list(self) -> list[int]:
        return self._free_blocks.copy()

    def can_allocate_gpu_blocks(self, num_blocks: int) -> bool:
        """契约：检查是否可分配"""
        if not self._can_allocate:
            return False
        return len(self._free_blocks) >= num_blocks

    def allocate_gpu_blocks(self, num_blocks: int, request_id=None) -> list[int]:
        """契约：分配后可用块减少"""
        self.allocate_calls.append((num_blocks, request_id))

        if not self.can_allocate_gpu_blocks(num_blocks):
            return []

        allocated = self._free_blocks[:num_blocks]
        self._free_blocks = self._free_blocks[num_blocks:]
        return allocated

    def recycle_gpu_blocks(self, block_ids, request_id=None) -> None:
        """契约：回收后可用块增加"""
        self.recycle_calls.append((list(block_ids), request_id))
        self._free_blocks.extend(block_ids)

    def release_block_ids(self, request) -> None:
        """契约：释放请求的所有块"""
        self.release_calls.append(request.request_id)
        # 简化：不追踪每个请求的块，由测试自行管理

    # --- 简化接口：返回固定值或可配置值 ---

    def request_match_blocks(self, request, block_size):
        """
        简化：不实现真正的 prefix matching 算法
        测试可通过 set_match_result() 控制返回值
        """
        return self._match_result

    def update_cache_blocks(self, request, block_size, computed_tokens):
        """简化：空实现，ResourceManagerV1 不依赖其返回值"""
        pass

    def write_cache_to_storage(self, request):
        """简化：空实现"""
        pass

    # --- 测试控制点 ---

    def set_can_allocate(self, value: bool) -> None:
        """控制分配是否成功（触发抢占路径）"""
        self._can_allocate = value

    def set_available_blocks(self, n: int) -> None:
        """设置可用块数量（边界测试）"""
        self._free_blocks = list(range(n))

    def set_match_result(self, blocks: list, tokens: int, info: dict) -> None:
        """设置 prefix cache 匹配结果"""
        self._match_result = (blocks, tokens, info)

    # --- 断言辅助 ---

    def assert_allocated(self, expected_blocks: int, request_id: str = None):
        """断言分配了指定数量的块"""
        total = sum(n for n, _ in self.allocate_calls)
        assert total == expected_blocks, f"Expected {expected_blocks}, got {total}"

    def reset(self) -> None:
        """重置状态用于下一个测试"""
        self._free_blocks = list(range(self._num_gpu_blocks))
        self._can_allocate = True
        self.allocate_calls.clear()
        self.recycle_calls.clear()
        self.release_calls.clear()
```

### 5.4 Mock vs 真实对象的选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 单元测试 schedule() 逻辑 | Mock CacheManager | 隔离调度逻辑，快速执行 |
| 测试 prefix caching 效果 | 真实 CacheManager | prefix matching 是核心功能 |
| 测试抢占触发条件 | Mock（设置 can_allocate=False） | 精确控制触发条件 |
| 测试完整请求生命周期 | 真实对象（集成测试） | 验证端到端行为 |
| 测试 metrics 是否正确 | RecordingMetrics | 需要验证具体值 |
| 普通功能测试 | NoOpMetrics | metrics 不影响逻辑 |

### 5.5 "不够用"的信号

当出现以下情况时，说明 Mock 设计需要调整：

```python
# 信号 1: Mock 比真实实现更复杂
class MockCacheManager:
    def allocate_gpu_blocks(self, n, req_id):
        # 如果这里需要实现 LRU、Radix Tree...
        # 说明：粒度太细，应该用真实对象或 Fake

# 信号 2: 测试只验证调用顺序，不验证结果
def test_schedule():
    manager.schedule()
    # 如果只断言 "cache_manager.allocate 被调用了"
    # 而不断言 "请求状态变为 RUNNING"
    # 说明：测试价值低，应重新设计

# 信号 3: 每个测试都要大量配置 Mock
def test_xxx():
    mock.method1.return_value = ...
    mock.method2.return_value = ...
    mock.method3.side_effect = ...
    # 说明：Mock 粒度太细，应该提供更高层抽象
```

---

## 6. Mock 实现

### 5.1 MockCacheManager

```python
class MockCacheManager:
    """
    Test double for ICacheManager.
    Provides controllable behavior for isolated unit testing.
    """

    def __init__(self, num_gpu_blocks: int = 100):
        self._num_gpu_blocks = num_gpu_blocks
        self._free_blocks = list(range(num_gpu_blocks))
        self._can_allocate = True

        # Call tracking for assertions
        self.allocate_calls: list[tuple] = []
        self.recycle_calls: list[tuple] = []

    def can_allocate_gpu_blocks(self, num_blocks: int) -> bool:
        return self._can_allocate and len(self._free_blocks) >= num_blocks

    def allocate_gpu_blocks(self, num_blocks: int, request_id=None) -> list[int]:
        self.allocate_calls.append((num_blocks, request_id))
        if not self.can_allocate_gpu_blocks(num_blocks):
            return []
        allocated = self._free_blocks[:num_blocks]
        self._free_blocks = self._free_blocks[num_blocks:]
        return allocated

    def recycle_gpu_blocks(self, block_ids, request_id=None) -> None:
        self.recycle_calls.append((list(block_ids), request_id))
        self._free_blocks.extend(block_ids)

    # Test helper
    def set_can_allocate(self, value: bool) -> None:
        """Control allocation behavior for testing."""
        self._can_allocate = value
```

### 5.2 MockIPCSignal

```python
class MockIPCSignal:
    """In-memory signal without SharedMemory."""

    def __init__(self, size: int):
        self._value = np.zeros(size, dtype=np.int32)

    @property
    def value(self) -> np.ndarray:
        return self._value

    def clear(self) -> None:
        pass  # No cleanup needed
```

### 5.3 MockExecutorPool

```python
class MockExecutorPool:
    """Synchronous execution for deterministic tests."""

    def __init__(self):
        self.submitted_tasks: list[tuple] = []

    def submit(self, fn, *args, **kwargs):
        self.submitted_tasks.append((fn, args, kwargs))
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait=True):
        pass
```

### 5.4 NoOpMetricsRecorder

```python
class NoOpMetricsRecorder:
    """Discards all metrics - use when not testing metrics."""

    def set_max_batch_size(self, v): pass
    def set_batch_size(self, v): pass
    def set_gpu_cache_usage_perc(self, v): pass
    def set_available_gpu_block_num(self, v): pass
    def inc_prefix_cache_token_num(self, v): pass
```

---

## 6. 测试夹具工厂

```python
@dataclass
class TestConfig:
    """Minimal configuration for testing."""
    max_num_seqs: int = 4
    block_size: int = 16
    num_gpu_blocks: int = 100
    enable_prefix_caching: bool = False
    enable_mm: bool = False
    splitwise_role: str = "mixed"


class ResourceManagerTestFixture:
    """
    Factory for creating test-ready ResourceManagerV1.

    Usage:
        fixture = ResourceManagerTestFixture()
        manager, mocks = fixture.create()

        mocks.cache_manager.set_can_allocate(False)
        manager.schedule()
        assert len(mocks.cache_manager.allocate_calls) == 0
    """

    @dataclass
    class Mocks:
        cache_manager: MockCacheManager
        ipc_signal: MockIPCSignal
        metrics: RecordingMetricsRecorder
        finish_executor: MockExecutorPool
        preprocess_executor: MockExecutorPool

    def create(self, config: TestConfig = None) -> tuple[ResourceManagerV1, Mocks]:
        config = config or TestConfig()

        mocks = self.Mocks(
            cache_manager=MockCacheManager(config.num_gpu_blocks),
            ipc_signal=MockIPCSignal(config.max_num_seqs),
            metrics=RecordingMetricsRecorder(),
            finish_executor=MockExecutorPool(),
            preprocess_executor=MockExecutorPool(),
        )

        deps = ResourceManagerDependencies(
            cache_manager=mocks.cache_manager,
            ipc_signal=mocks.ipc_signal,
            metrics=mocks.metrics,
            finish_executor=mocks.finish_executor,
            preprocess_executor=mocks.preprocess_executor,
        )

        fd_config = self._build_config(config)
        manager = ResourceManagerV1(
            max_num_seqs=config.max_num_seqs,
            config=fd_config,
            tensor_parallel_size=1,
            splitwise_role=config.splitwise_role,
            dependencies=deps,
        )

        return manager, mocks
```

---

## 7. 测试对比

### Before（当前 30-40 行样板代码）

```python
class TestResourceManagerV1(unittest.TestCase):
    def setUp(self):
        engine_args = EngineArgs(max_num_seqs=2, num_gpu_blocks_override=102, ...)
        args = asdict(engine_args)
        cache_cfg = CacheConfig(args)
        model_cfg = SimpleNamespace(enable_mm=True)
        speculative_cfg = SimpleNamespace(method=None)
        parallel_cfg = SimpleNamespace(local_engine_worker_queue_port=12345)
        scheduler_cfg = SimpleNamespace(splitwise_role="mixed", ...)
        # ... 20+ more lines

        self.manager = ResourceManagerV1(...)
        self.manager.cache_manager = Mock()  # Manual mock

        # Manual cleanup
        self.addCleanup(self.manager.need_block_num_signal.clear)
        self.addCleanup(self.manager.finish_execution_pool.shutdown, wait=True)
```

### After（2 行）

```python
class TestResourceManagerV1:
    @pytest.fixture
    def fixture(self):
        return ResourceManagerTestFixture()

    def test_schedule_allocates_blocks(self, fixture):
        # Arrange
        manager, mocks = fixture.create()
        request = create_test_request(request_id="req-001")
        manager.add_request(request)

        # Act
        scheduled, errors = manager.schedule()

        # Assert
        assert len(scheduled) == 1
        assert len(mocks.cache_manager.allocate_calls) > 0
        # No cleanup needed
```

---

## 8. 文件结构

```
fastdeploy/engine/sched/
├── resource_manager_v1.py    # 添加 dependencies 参数
├── interfaces.py             # 新建 - Protocol 定义
└── testing.py                # 新建 - Mock 实现和测试夹具

fastdeploy/engine/
└── resource_manager.py       # 添加 _init_base_state() 方法

tests/v1/
└── test_resource_manager_v1.py  # 迁移到新测试模式
```

---

## 9. 实施步骤

| 阶段 | 内容 | 风险 |
|------|------|------|
| **Phase 1** | 创建 `interfaces.py` 和 `testing.py`，定义 Protocol 和 Mock | 无破坏性 |
| **Phase 2** | 在 `ResourceManagerV1.__init__` 添加可选 `dependencies` 参数 | 向后兼容 |
| **Phase 3** | 迁移现有测试到新模式，删除 force coverage hack | 测试改进 |

---

## 10. 验证方案

```bash
# 1. 现有测试通过（向后兼容验证）
pytest tests/v1/test_resource_manager_v1.py -v

# 2. 新单元测试通过
pytest tests/v1/test_resource_manager_v1_unit.py -v

# 3. 真实覆盖率检查
pytest tests/v1/ --cov=fastdeploy/engine/sched --cov-report=term-missing
```

---

## 设计原则

| 原则 | 应用 |
|------|------|
| **职责清晰** | 公开接口 vs 内部实现明确分离 |
| **层次清楚** | Protocol → Mock → Fixture → Test |
| **可测试性** | 依赖注入实现隔离，便于 Mock |
| **向后兼容** | `dependencies=None` 时行为不变 |
| **最小变更** | 只修改必要代码，不过度设计 |
