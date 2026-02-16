# PrefixCacheManager 重构设计文档

> **文档版本**: v1.0
> **创建日期**: 2026-02-16
> **状态**: 设计中
> **关联分析**: [God Class 分析报告](god_class_analysis.md)

---

## 一、背景与目标

### 1.1 当前问题

PrefixCacheManager 是 FastDeploy 中职责混杂最严重的类，承担了 9 种职责，且管理多种外部资源：

| 指标 | 值 | 评估 |
|------|-----|------|
| 代码行数 | 2148 | 大型类 |
| 方法数量 | 46 | 🔴 过多（>40） |
| 实例变量 | 43 | 🔴 复杂（>30） |
| 平均圈复杂度 | B (6.70) | 🟢 较好 |
| 最高圈复杂度 | D (28) `free_block_ids_async` | 🟡 偏高 |

**核心问题**：PrefixCacheManager 管理多种外部资源（子进程、线程池、IPC 信号），违反单一职责原则。

### 1.2 重构目标

1. **分离外部资源管理**：进程启动、IPC 通信独立为专门组件
2. **提高可测试性**：缓存逻辑可不依赖进程/IPC 直接测试
3. **保持接口稳定**：对外接口签名不变
4. **支持策略可插拔**：驱逐策略、块分配策略可配置

---

## 二、职责分析

### 2.1 现有职责清单

```
PrefixCacheManager 承担的职责（9 种）：
├── GPU 块管理
│     └── allocate_gpu_blocks(), recycle_gpu_blocks(), can_allocate_gpu_blocks()
├── CPU 块管理
│     └── allocate_cpu_blocks(), recycle_cpu_blocks()
├── 前缀树管理
│     └── match_block(), build_path(), cache_output_blocks()
├── GPU/CPU 交换
│     └── issue_swap_task(), sync_swap_task(), _handle_swap_result()
├── 存储后端交互
│     └── write_cache_to_storage(), issue_prefetch_storage_task()
├── 驱逐策略
│     └── _evict_cache_async(), free_block_ids_async()
├── 进程启动与管理 🔴 (违反 SRP)
│     └── launch_cache_manager(), launch_cache_messager()
├── IPC 信号管理 🔴 (违反 SRP)
│     └── 9 个 IPCSignal 实例
└── 多模态缓存处理
      └── mm_match_block(), mm_build_path(), get_block_hash_extra_keys()
```

### 2.2 高复杂度方法（CC ≥ 10）

| 方法 | CC 等级 | 复杂度值 | 建议 |
|------|---------|---------|------|
| `free_block_ids_async()` | **D** | 28 | 🔴 考虑拆分 |
| `launch_cache_manager()` | **D** | 22 | 🔴 必须抽取（外部资源） |
| `free_cpu_block_ids()` | **C** | 15 | 🟡 关注 |
| `get_block_hash_extra_keys()` | **C** | 15 | 🟡 关注 |
| `cache_output_blocks()` | **C** | 15 | 🟡 关注 |
| `request_match_blocks()` | **C** | 14 | 🟡 关注 |
| `free_nodes_directly()` | **C** | 14 | 🟡 关注 |

### 2.3 外部资源管理分析

> 🔴 **核心问题**：PrefixCacheManager 管理多种外部资源，这是需要拆分的根本原因。

| 外部资源类型 | 具体实例 | 管理方式 | 问题 |
|-------------|---------|---------|------|
| **子进程** | cache_transfer_manager, cache_messager | subprocess.Popen | 缓存管理器不应负责进程生命周期 |
| **线程池** | executor_pool, free_gpu/cpu_executor_pool | ThreadPoolExecutor | 内部创建，应外部注入 |
| **后台线程** | recv_data_transfer_result, clear_prefix_cache | threading.Thread | 生命周期管理混乱 |
| **IPC 信号** | 9 个 IPCSignal 实例 | 共享内存 | 应由专门组件管理 |
| **IPC 队列** | cache_task_queue | EngineCacheQueue | 通信职责混入 |

---

## 三、状态耦合分析

### 3.1 共享状态清单

> ⚠️ **关键发现**：PrefixCacheManager 维护了 43 个实例变量，多个职责通过共享状态紧密耦合。

| 共享变量 | 涉及职责 | 严重程度 | 说明 |
|---------|---------|---------|------|
| `radix_tree_root` | 树管理/匹配/构建/释放 | 🔴 严重 | 前缀树根节点，核心数据结构 |
| `gpu_free_block_list` | GPU分配/驱逐/回收 | 🔴 严重 | GPU 空闲块堆，多处读写 |
| `req_leaf_map` / `leaf_req_map` | 请求跟踪/释放/树管理 | 🔴 严重 | 请求-叶节点双向映射 |
| `cache_task_queue` | 交换/存储/进程通信 | 🟡 中等 | 异步任务队列 |
| `node_map` | 树管理/交换结果处理 | 🟡 中等 | 节点 ID 映射 |
| `gpu_lru_leaf_heap/set` | 驱逐/释放 | 🟡 中等 | LRU 驱逐数据结构 |
| `request_release_lock` | 几乎所有公共方法 | 🟡 中等 | 全局锁，性能瓶颈 |

### 3.2 状态共享关系图

```
                    ┌───────────────────────────────────────────┐
                    │             radix_tree_root               │
                    │       (radix tree core data structure)    │
                    └─────────────────────┬─────────────────────┘
                                          │
     ┌────────────┬────────────┬──────────┼──────────┬────────────┐
     ▼            ▼            ▼          ▼          ▼            ▼
 match_block  build_path  release_    cache_      mm_match_
 mm_match_    mm_build_   block_ids   output_     block
 block        path                    blocks

                    ┌───────────────────────────────────────────┐
                    │       gpu_free_block_list (heap)          │
                    └─────────────────────┬─────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          ▼                               ▼                               ▼
     allocate_*_gpu_blocks         recycle_*_gpu_blocks           _evict_cache_async

                    ┌───────────────────────────────────────────┐
                    │       cache_task_queue (IPC queue)        │
                    └─────────────────────┬─────────────────────┘
                                          │
     ┌────────────┬────────────┬──────────┼──────────┬────────────┐
     ▼            ▼            ▼          ▼          ▼            ▼
 issue_swap_  issue_write_  issue_     recv_data_   process
 task         back_storage  prefetch_  transfer_    launch
              _task         storage    result       (init)
```

### 3.3 方法调用依赖

```
launch_cache_manager (初始化入口)
├── _get_kv_cache_shape
├── launch_cache_messager ← 启动 cache messager 进程
├── subprocess.Popen ← 启动 cache transfer manager 进程
└── threading.Thread ← 启动后台线程

request_match_blocks (请求匹配入口)
├── mm_match_block ← 多模态匹配
│   └── get_block_hash_extra_keys
├── _update_matched_node_info
├── can_allocate_gpu_blocks
│   └── free_block_ids → free_block_ids_async → _evict_cache_async
├── allocate_gpu_blocks
└── _prepare_cpu_cache → issue_swap_task

request_block_ids (块分配入口)
├── match_block
├── _check_validity
├── _update_matched_node_info
├── _prepare_cache
│   ├── allocate_gpu_blocks
│   └── _prepare_cpu_cache → issue_swap_task
└── build_path

release_block_ids (释放入口)
├── 更新 req_leaf_map / leaf_req_map
├── 更新 gpu_lru_leaf_heap / gpu_lru_leaf_set
└── recycle_gpu_blocks (条件)
```

---

## 四、拆分方案

### 4.1 拆分可行性评估

| 组件 | 可行性 | 风险点 | 现有模式参考 |
|------|--------|--------|-------------|
| `CacheProcessLauncher` | ✅ 高 | 逻辑完全独立 | `EngineService.WorkerManager` |
| `GPUBlockAllocator` | ✅ 高 | 接口清晰 | - |
| `CPUBlockAllocator` | ✅ 高 | 接口清晰 | - |
| `CacheTreeManager` | ⚠️ 中 | 与 req_*_map 耦合 | - |
| `SwapManager` | ⚠️ 中 | 依赖 cache_task_queue | - |
| `StorageBackendClient` | ✅ 高 | 逻辑相对独立 | - |
| `EvictionPolicy` | ⚠️ 中 | 与 LRU 堆耦合 | SGLang EvictionStrategy |
| `MultimodalCacheMatcher` | ✅ 高 | mm_* 方法内聚 | - |

### 4.2 建议拆分组件

| 组件 | 职责 | 预估行数 | 优先级 | Phase |
|------|------|----------|--------|-------|
| `CacheProcessLauncher` | 进程启动和管理 | ~300 | **P0** | Phase 1 |
| `GPUBlockAllocator` | GPU 块分配、回收、驱逐 | ~350 | **P1** | Phase 1 |
| `CPUBlockAllocator` | CPU 块分配和回收 | ~250 | P2 | Phase 2 |
| `CacheTreeManager` | 前缀树构建、匹配、维护 | ~550 | P2 | Phase 2 |
| `SwapManager` | GPU/CPU 数据交换 | ~300 | P2 | Phase 2 |
| `StorageBackendClient` | 存储后端读写 | ~200 | P3 | Phase 3 |
| `MultimodalCacheMatcher` | 多模态缓存匹配 | ~250 | P3 | Phase 3 |
| PrefixCacheManager（精简后） | 缓存协调 | ~400 | - | - |

### 4.3 重构后架构

```
PrefixCacheManager (精简后，~400行，协调者)
├── CacheProcessLauncher (~300行)
│     └── launch_cache_manager(), launch_cache_messager(), IPC 信号管理
├── GPUBlockAllocator (~350行)
│     └── allocate_gpu_blocks(), recycle_gpu_blocks(), _evict_cache_async()
├── CPUBlockAllocator (~250行)
│     └── allocate_cpu_blocks(), recycle_cpu_blocks()
├── CacheTreeManager (~550行)
│     └── match_block(), build_path(), radix_tree_root 管理
├── SwapManager (~300行)
│     └── issue_swap_task(), sync_swap_task()
├── StorageBackendClient (~200行)
│     └── write_cache_to_storage(), issue_prefetch_storage_task()
└── MultimodalCacheMatcher (~250行)
      └── mm_match_block(), mm_build_path()
```

---

## 五、接口设计

### 5.1 CacheProcessLauncher

```python
class CacheProcessLauncher:
    """
    缓存进程启动器 - 从 PrefixCacheManager 拆分

    职责:
    1. 启动 cache_transfer_manager 进程
    2. 启动 cache_messager 进程
    3. 管理 IPC 信号的初始化和生命周期
    4. 管理后台线程 (recv_data_transfer_result, clear_prefix_cache)
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._transfer_manager_proc: Optional[subprocess.Popen] = None
        self._messager_proc: Optional[subprocess.Popen] = None
        self._ipc_signals: Dict[str, IPCSignal] = {}
        self._threads: List[threading.Thread] = []

    def launch(self) -> None:
        """启动所有缓存相关进程和线程"""
        self._init_ipc_signals()
        self._launch_transfer_manager()
        self._launch_messager()
        self._start_background_threads()

    def shutdown(self) -> None:
        """关闭所有进程和线程"""
        self._stop_threads()
        self._terminate_processes()
        self._cleanup_ipc_signals()

    def get_ipc_signal(self, name: str) -> IPCSignal:
        """获取指定的 IPC 信号"""
        return self._ipc_signals[name]

    def get_cache_task_queue(self) -> EngineCacheQueue:
        """获取缓存任务队列"""
        return self._cache_task_queue

    def _init_ipc_signals(self) -> None:
        """初始化 9 个 IPC 信号"""
        signal_names = [
            "cache_ready_signal",
            "swap_space_ready_signal",
            # ... 其他信号
        ]
        for name in signal_names:
            self._ipc_signals[name] = IPCSignal(...)

    def _launch_transfer_manager(self) -> None:
        """启动 cache_transfer_manager 进程"""
        pass

    def _launch_messager(self) -> None:
        """启动 cache_messager 进程"""
        pass
```

### 5.2 GPUBlockAllocator

```python
class GPUBlockAllocator:
    """
    GPU 块分配器 - 从 PrefixCacheManager 拆分

    职责:
    1. 管理 GPU 空闲块列表
    2. 分配和回收 GPU 块
    3. 执行驱逐策略

    参考: vLLM BlockPool
    """

    def __init__(
        self,
        total_blocks: int,
        eviction_policy: Optional[EvictionPolicy] = None
    ):
        self.total_blocks = total_blocks
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self._free_block_list: List[int] = list(range(total_blocks))
        self._allocated_blocks: Set[int] = set()

    @property
    def free_count(self) -> int:
        """当前空闲块数量"""
        return len(self._free_block_list)

    def can_allocate(self, num_blocks: int) -> bool:
        """
        检查是否可以分配指定数量的块

        Args:
            num_blocks: 需要的块数量

        Returns:
            是否可以分配
        """
        return self.free_count >= num_blocks

    def allocate(self, num_blocks: int) -> List[int]:
        """
        分配指定数量的块

        Args:
            num_blocks: 需要的块数量

        Returns:
            分配的块 ID 列表

        Raises:
            AllocationError: 如果没有足够的块
        """
        if not self.can_allocate(num_blocks):
            raise AllocationError(f"需要 {num_blocks} 块，但只有 {self.free_count} 块可用")

        blocks = [self._free_block_list.pop() for _ in range(num_blocks)]
        self._allocated_blocks.update(blocks)
        return blocks

    def recycle(self, block_ids: List[int]) -> None:
        """
        回收块

        Args:
            block_ids: 要回收的块 ID 列表
        """
        for block_id in block_ids:
            if block_id in self._allocated_blocks:
                self._allocated_blocks.remove(block_id)
                self._free_block_list.append(block_id)

    def evict_if_needed(self, required_blocks: int) -> List[int]:
        """
        如果需要，执行驱逐以获得足够的块

        Args:
            required_blocks: 需要的块数量

        Returns:
            被驱逐的块 ID 列表
        """
        if self.can_allocate(required_blocks):
            return []

        return self.eviction_policy.evict(
            required=required_blocks - self.free_count,
            allocator=self
        )
```

### 5.3 EvictionPolicy (策略接口)

```python
class EvictionPolicy(ABC):
    """
    驱逐策略接口 - 可插拔设计

    参考: SGLang EvictionStrategy
    """

    @abstractmethod
    def evict(self, required: int, allocator: GPUBlockAllocator) -> List[int]:
        """
        执行驱逐

        Args:
            required: 需要驱逐的块数量
            allocator: 块分配器

        Returns:
            被驱逐的块 ID 列表
        """
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """LRU 驱逐策略"""

    def __init__(self):
        self._lru_heap: List[Tuple[float, int]] = []  # (timestamp, block_id)
        self._lru_set: Set[int] = set()

    def record_access(self, block_id: int) -> None:
        """记录块访问"""
        timestamp = time.time()
        heapq.heappush(self._lru_heap, (timestamp, block_id))
        self._lru_set.add(block_id)

    def evict(self, required: int, allocator: GPUBlockAllocator) -> List[int]:
        """按 LRU 顺序驱逐"""
        evicted = []
        while len(evicted) < required and self._lru_heap:
            _, block_id = heapq.heappop(self._lru_heap)
            if block_id in self._lru_set:
                self._lru_set.remove(block_id)
                evicted.append(block_id)
                allocator.recycle([block_id])
        return evicted


class LFUEvictionPolicy(EvictionPolicy):
    """LFU 驱逐策略（备选）"""
    pass
```

### 5.4 CacheConnector Public API

```python
class CacheConnector(ABC):
    """
    Public API - 缓存连接器抽象基类

    参考: vLLM KVConnector

    重构后 PrefixCacheManager 实现此接口，保证签名稳定
    """

    @abstractmethod
    def allocate_blocks(self, num_blocks: int) -> List[int]:
        """
        分配缓存块

        Args:
            num_blocks: 需要的块数量

        Returns:
            分配的块 ID 列表
        """
        pass

    @abstractmethod
    def free_blocks(self, block_ids: List[int]) -> None:
        """
        释放缓存块

        Args:
            block_ids: 要释放的块 ID 列表
        """
        pass

    @abstractmethod
    def match_prefix(self, token_ids: List[int]) -> MatchResult:
        """
        前缀匹配

        Args:
            token_ids: token ID 序列

        Returns:
            匹配结果 (matched_length, block_ids)
        """
        pass

    @abstractmethod
    def build_path(self, token_ids: List[int], block_ids: List[int]) -> None:
        """
        构建前缀树路径

        Args:
            token_ids: token ID 序列
            block_ids: 对应的块 ID 列表
        """
        pass
```

---

## 六、业界参考

### 6.1 vLLM / SGLang 对比

> ⚠️ 以下数据为参考值，实际实现时请参考最新版本源码。

| 对比维度 | FastDeploy | vLLM | SGLang |
|---------|-----------|---------|--------|
| **总行数** | ~2148 (单文件) | ~900 (多文件组合) | ~890 + ~2025 |
| **块管理** | ❌ 内嵌 | ✅ `BlockPool` 独立类 | ✅ `TokenToKVPoolAllocator` |
| **前缀树** | ❌ 内嵌 | ✅ `BlockHashToBlockMap` | ✅ `RadixCache` 独立类 |
| **协调器** | ❌ 无 | ✅ `KVCacheCoordinator` | ✅ `BasePrefixCache` 抽象 |
| **驱逐策略** | ❌ 内嵌 | ✅ 可插拔 | ✅ `EvictionStrategy` 策略模式 |
| **进程管理** | ❌ 内嵌（违反 SRP） | ✅ 外部 | ✅ 外部 |
| **IPC 通信** | ❌ 内嵌 | ✅ `KVCacheEvent` 独立 | ✅ 事件驱动 |

### 6.2 vLLM 架构参考

```
KVCacheManager (入口，~400行)
├── KVCacheCoordinator (协调器)
│   ├── SingleTypeKVCacheManager (单类型管理)
│   └── CrossAttentionManager (交叉注意力)
├── BlockPool (块池，~250行)
│   ├── FreeKVCacheBlockQueue (空闲队列)
│   └── BlockHashToBlockMap (哈希映射)
└── KVCacheMetricsCollector (指标收集)
```

### 6.3 SGLang 架构参考

```
RadixCache (前缀树，~890行)
├── BasePrefixCache (抽象基类)
├── EvictionStrategy (驱逐策略接口)
│   ├── LRUStrategy
│   ├── LFUStrategy
│   └── ...
└── TreeNode (节点数据结构)

memory_pool.py (~2025行)
├── ReqToTokenPool (请求->Token映射)
├── TokenToKVPoolAllocator (Token->KV分配)
└── KVCache (物理缓存)
```

### 6.4 设计启示

1. **职责隔离**：vLLM/SGLang 的缓存管理器**不负责进程启动**
2. **抽象层次**：都有 Coordinator/BasePrefixCache 作为抽象层
3. **策略模式**：驱逐策略可插拔，FastDeploy 的 LRU 逻辑硬编码在类中
4. **数据结构分离**：BlockPool/RadixCache 与业务逻辑分离，可独立测试

---

## 七、实施计划

### 7.1 Phase 1: CacheProcessLauncher + GPUBlockAllocator 抽取

**目标**：分离外部资源管理，建立块分配独立组件

**CacheProcessLauncher 范围**：
- `launch_cache_manager()`
- `launch_cache_messager()`
- 9 个 IPCSignal 初始化逻辑
- 后台线程启动逻辑

**GPUBlockAllocator 范围**：
- `allocate_gpu_blocks()`
- `recycle_gpu_blocks()`
- `can_allocate_gpu_blocks()`
- `_evict_cache_async()`
- `gpu_free_block_list` 管理

**预期收益**：
- 减少 ~650 行代码
- 进程管理职责分离，符合 SRP
- 块分配逻辑可独立测试

### 7.2 Phase 2: CPUBlockAllocator + CacheTreeManager 抽取

**目标**：进一步分离块管理和树管理

**CPUBlockAllocator 范围**：
- `allocate_cpu_blocks()`
- `recycle_cpu_blocks()`
- `free_cpu_block_ids()`

**CacheTreeManager 范围**：
- `match_block()`
- `build_path()`
- `cache_output_blocks()`
- `radix_tree_root` 管理
- `req_leaf_map` / `leaf_req_map` 管理

**预期收益**：
- 减少 ~800 行代码
- 前缀树逻辑独立，可独立测试

### 7.3 Phase 3: 其他组件抽取

**待评估组件**：
- SwapManager：GPU/CPU 数据交换
- StorageBackendClient：存储后端交互
- MultimodalCacheMatcher：多模态缓存匹配

---

## 八、测试策略

### 8.1 单元测试

```python
# GPUBlockAllocator 单元测试 - 无需 mock
def test_allocate_blocks():
    allocator = GPUBlockAllocator(total_blocks=100)
    blocks = allocator.allocate(10)
    assert len(blocks) == 10
    assert allocator.free_count == 90

def test_eviction_policy():
    policy = LRUEvictionPolicy()
    allocator = GPUBlockAllocator(total_blocks=10, eviction_policy=policy)

    # 分配所有块
    blocks = allocator.allocate(10)

    # 记录访问（模拟使用）
    for block in blocks[:5]:
        policy.record_access(block)

    # 回收一些块
    allocator.recycle(blocks[5:])

    # 需要更多块时触发驱逐
    evicted = allocator.evict_if_needed(3)
    assert len(evicted) <= 3

# CacheProcessLauncher 需要 mock（外部资源）
@patch('subprocess.Popen')
def test_launch_processes(mock_popen):
    launcher = CacheProcessLauncher(config)
    launcher.launch()
    assert mock_popen.call_count == 2  # transfer_manager + messager
```

### 8.2 集成测试

```python
# 确保拆分后 PrefixCacheManager 行为不变
def test_prefix_cache_manager_integration():
    manager = PrefixCacheManager(config)
    # 使用真实输入测试完整流程
    result = manager.request_block_ids(request)
    assert result.matched_length == expected_length
```

### 8.3 性能基准

重构前后需对比以下指标：

| 指标 | 容忍阈值 |
|------|----------|
| 块分配延迟 | ≤ 5% 退化 |
| 前缀匹配延迟 | ≤ 5% 退化 |
| 缓存命中率 | 无退化 |
| 进程启动时间 | ≤ 10% 退化 |

---

## 九、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 回归风险 | 重构前建立集成测试覆盖；每 PR 独立验证 |
| 并发安全 | 拆分后明确锁的归属；考虑细粒度锁优化 |
| 性能影响 | 热路径保持内联；拆分后性能基准测试 |
| 回滚需求 | 保留旧代码路径，环境变量 `FD_USE_LEGACY_PREFIX_CACHE_MANAGER=1` 切换 |

---

## 十、Hackathon 协同

### 10.1 相关任务

| Issue | 模块 | 建议 |
|-------|------|------|
| [#6219](https://github.com/PaddlePaddle/FastDeploy/pull/6219) | prefix_cache_manager.py | **结合重构一起推进** |
| [#6220](https://github.com/PaddlePaddle/FastDeploy/pull/6220) | cache_transfer_manager.py | 与 CacheProcessLauncher 相关 |

### 10.2 协同建议

建议先完成重构再补充单测，避免测试代码也需要大改。
