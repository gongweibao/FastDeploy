# 调度器统一详细设计

> **文档版本**: v1.0
> **创建日期**: 2026-02-14
> **关联问题**: V0/V1 调度器代码混杂、锁粒度过大

---

## 一、背景与目标

### 1.1 当前问题

| 问题 | 现状 | 影响 |
|------|------|------|
| 两套代码 | `resource_manager.py` (V0) + `resource_manager_v1.py` (V1) | 维护成本翻倍 |
| 环境变量切换 | `ENABLE_V1_KVCACHE_SCHEDULER` 控制 | 代码中到处是 if/else |
| 锁粒度过大 | schedule() 全程持锁 200+ 行 | 并发度受限 |
| 线性查找 | O(n) 查找空闲槽位 | 性能瓶颈 |

### 1.2 重构目标

- 定义统一的 ResourceManager 接口
- V0/V1 作为接口的不同实现
- 使用工厂模式创建实例
- 优化锁粒度
- 逐步废弃 V0

---

## 二、架构设计

### 2.1 统一接口

```python
# fastdeploy/engine/sched/base.py
"""
调度器基础接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass

from fastdeploy.engine.request import Request
from fastdeploy.config import FDConfig


@dataclass
class ScheduleResult:
    """调度结果"""
    scheduled_tasks: List[Request]      # 已调度的任务
    preempted_tasks: List[Request]      # 被抢占的任务
    waiting_count: int                   # 等待队列长度
    running_count: int                   # 运行队列长度


class ResourceManagerBase(ABC):
    """
    资源管理器基础接口

    定义所有资源管理器必须实现的方法
    """

    @abstractmethod
    def allocate_resources(self, tasks: List[Request]) -> List[Request]:
        """
        为任务分配资源

        Args:
            tasks: 待分配资源的任务列表

        Returns:
            成功分配资源的任务列表
        """
        pass

    @abstractmethod
    def release_resources(self, request_id: str) -> bool:
        """
        释放请求占用的资源

        Args:
            request_id: 请求 ID

        Returns:
            是否成功释放
        """
        pass

    @abstractmethod
    def schedule(self) -> ScheduleResult:
        """
        执行调度

        Returns:
            调度结果
        """
        pass

    @abstractmethod
    def available_batch(self) -> int:
        """
        获取可用批处理槽位数

        Returns:
            可用槽位数
        """
        pass

    @abstractmethod
    def get_running_requests(self) -> List[Request]:
        """
        获取运行中的请求列表

        Returns:
            运行中的请求
        """
        pass

    @abstractmethod
    def get_waiting_requests(self) -> List[Request]:
        """
        获取等待中的请求列表

        Returns:
            等待中的请求
        """
        pass
```

### 2.2 工厂模式

```python
# fastdeploy/engine/sched/factory.py
"""
资源管理器工厂
"""

from typing import Literal
from fastdeploy.config import FDConfig
from fastdeploy.engine.sched.base import ResourceManagerBase


class ResourceManagerFactory:
    """
    资源管理器工厂

    根据配置创建相应的资源管理器实例
    """

    _registry = {}

    @classmethod
    def register(cls, version: str):
        """
        注册资源管理器实现

        用法:
            @ResourceManagerFactory.register("v1")
            class ResourceManagerV1(ResourceManagerBase):
                ...
        """
        def decorator(manager_cls):
            cls._registry[version] = manager_cls
            return manager_cls
        return decorator

    @classmethod
    def create(
        cls,
        version: Literal["v0", "v1"] = "v1",
        config: FDConfig = None,
        **kwargs
    ) -> ResourceManagerBase:
        """
        创建资源管理器实例

        Args:
            version: 版本号 ("v0" 或 "v1")
            config: 引擎配置
            **kwargs: 其他参数

        Returns:
            资源管理器实例

        Raises:
            ValueError: 未知的版本号
        """
        if version not in cls._registry:
            raise ValueError(
                f"Unknown resource manager version: {version}. "
                f"Available: {list(cls._registry.keys())}"
            )

        manager_cls = cls._registry[version]
        return manager_cls(config, **kwargs)

    @classmethod
    def available_versions(cls) -> List[str]:
        """获取可用的版本列表"""
        return list(cls._registry.keys())
```

### 2.3 V0 实现（保持兼容）

```python
# fastdeploy/engine/sched/resource_manager_v0.py
"""
V0 版本资源管理器（兼容旧代码）
"""

from fastdeploy.engine.sched.base import ResourceManagerBase, ScheduleResult
from fastdeploy.engine.sched.factory import ResourceManagerFactory


@ResourceManagerFactory.register("v0")
class ResourceManagerV0(ResourceManagerBase):
    """
    V0 版本资源管理器

    保持与原 resource_manager.py 相同的行为
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.max_num_seqs = config.scheduler_config.max_num_seqs

        # 槽位管理
        self.stop_flags = [True] * self.max_num_seqs
        self.tasks_list = [None] * self.max_num_seqs

        # 缓存管理器
        self.cache_manager = self._create_cache_manager()

    def allocate_resources(self, tasks):
        """分配资源（原 allocate_resources_for_new_tasks）"""
        allocated = []
        for task in tasks:
            slot = self._find_free_slot()
            if slot is None:
                break

            blocks = self._allocate_blocks(task)
            if blocks is None:
                continue

            task.idx = slot
            task.block_tables = blocks
            self.stop_flags[slot] = False
            self.tasks_list[slot] = task
            allocated.append(task)

        return allocated

    def release_resources(self, request_id):
        """释放资源"""
        for i, task in enumerate(self.tasks_list):
            if task and task.request_id == request_id:
                self._recycle_blocks(task)
                self.stop_flags[i] = True
                self.tasks_list[i] = None
                return True
        return False

    def schedule(self):
        """执行调度"""
        # V0 的调度逻辑相对简单
        return ScheduleResult(
            scheduled_tasks=[t for t in self.tasks_list if t],
            preempted_tasks=[],
            waiting_count=0,
            running_count=sum(1 for f in self.stop_flags if not f),
        )

    def available_batch(self):
        """获取可用槽位"""
        return sum(self.stop_flags)

    def _find_free_slot(self):
        """查找空闲槽位（原 O(n) 实现）"""
        for i, flag in enumerate(self.stop_flags):
            if flag:
                return i
        return None

    # ... 其他方法保持不变
```

### 2.4 V1 实现（优化版）

```python
# fastdeploy/engine/sched/resource_manager_v1.py
"""
V1 版本资源管理器（优化版）
"""

import threading
from collections import deque
from typing import Set

from fastdeploy.engine.sched.base import ResourceManagerBase, ScheduleResult
from fastdeploy.engine.sched.factory import ResourceManagerFactory


@ResourceManagerFactory.register("v1")
class ResourceManagerV1(ResourceManagerBase):
    """
    V1 版本资源管理器

    优化点：
    - O(1) 槽位查找
    - 细粒度锁
    - 支持 Chunked Prefill
    - 支持 Preemption
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.max_num_seqs = config.scheduler_config.max_num_seqs

        # 队列管理
        self.waiting: deque = deque()
        self.running: list = []
        self.requests: dict = {}

        # 槽位管理（O(1) 查找）
        self.slot_manager = SlotManager(self.max_num_seqs)

        # 细粒度锁
        self._waiting_lock = threading.Lock()
        self._running_lock = threading.Lock()
        self._slot_lock = threading.Lock()

        # 缓存管理器
        self.cache_manager = self._create_cache_manager()

    def allocate_resources(self, tasks):
        """分配资源"""
        allocated = []

        for task in tasks:
            # O(1) 槽位分配
            with self._slot_lock:
                slot = self.slot_manager.allocate()

            if slot is None:
                break

            # 分配缓存块
            blocks = self._allocate_blocks(task)
            if blocks is None:
                with self._slot_lock:
                    self.slot_manager.release(slot)
                continue

            task.idx = slot
            task.block_tables = blocks

            with self._running_lock:
                self.running.append(task)
                self.requests[task.request_id] = task

            allocated.append(task)

        return allocated

    def release_resources(self, request_id):
        """释放资源"""
        with self._running_lock:
            if request_id not in self.requests:
                return False

            task = self.requests.pop(request_id)
            self.running.remove(task)

        # 释放槽位
        with self._slot_lock:
            self.slot_manager.release(task.idx)

        # 释放缓存块
        self._recycle_blocks(task)

        return True

    def schedule(self) -> ScheduleResult:
        """执行调度"""
        scheduled = []
        preempted = []

        # 调度运行中的请求（读锁即可）
        with self._running_lock:
            running_snapshot = self.running.copy()

        for request in running_snapshot:
            if self._should_preempt(request):
                preempted.append(request)
            else:
                scheduled.append(request)

        # 调度等待中的请求
        with self._waiting_lock:
            while self.waiting and self._has_capacity():
                request = self.waiting.popleft()
                if self._can_schedule(request):
                    scheduled.append(request)
                else:
                    self.waiting.appendleft(request)
                    break

        return ScheduleResult(
            scheduled_tasks=scheduled,
            preempted_tasks=preempted,
            waiting_count=len(self.waiting),
            running_count=len(self.running),
        )

    def available_batch(self):
        """获取可用槽位"""
        with self._slot_lock:
            return self.slot_manager.available_count()

    def get_running_requests(self):
        """获取运行中的请求"""
        with self._running_lock:
            return self.running.copy()

    def get_waiting_requests(self):
        """获取等待中的请求"""
        with self._waiting_lock:
            return list(self.waiting)


class SlotManager:
    """
    槽位管理器

    使用 set 实现 O(1) 的分配和释放
    """

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.free_slots: Set[int] = set(range(max_slots))
        self.used_slots: Set[int] = set()

    def allocate(self) -> Optional[int]:
        """O(1) 分配槽位"""
        if not self.free_slots:
            return None
        slot = self.free_slots.pop()
        self.used_slots.add(slot)
        return slot

    def release(self, slot: int) -> bool:
        """O(1) 释放槽位"""
        if slot not in self.used_slots:
            return False
        self.used_slots.remove(slot)
        self.free_slots.add(slot)
        return True

    def available_count(self) -> int:
        """可用槽位数"""
        return len(self.free_slots)

    def used_count(self) -> int:
        """已用槽位数"""
        return len(self.used_slots)
```

---

## 三、锁优化详细设计

### 3.1 当前问题

```python
# 当前：整个 schedule() 持有全局锁
def schedule(self):
    with self.lock:  # 200+ 行代码全程持锁
        # 遍历 RUNNING 队列
        while req_index < len(self.running):
            ...
        # 遍历 WAITING 队列
        while self.waiting:
            ...
        # 分配块
        ...
        # 更新状态
        ...
```

### 3.2 细粒度锁设计

```python
class ResourceManagerV1:
    def __init__(self):
        # 分离的锁
        self._waiting_lock = threading.Lock()   # 保护 waiting 队列
        self._running_lock = threading.Lock()   # 保护 running 队列
        self._slot_lock = threading.Lock()      # 保护槽位管理
        self._block_lock = threading.Lock()     # 保护块分配

    def add_to_waiting(self, request):
        """添加到等待队列"""
        with self._waiting_lock:
            self.waiting.append(request)

    def get_from_waiting(self):
        """从等待队列获取"""
        with self._waiting_lock:
            if self.waiting:
                return self.waiting.popleft()
            return None

    def add_to_running(self, request):
        """添加到运行队列"""
        with self._running_lock:
            self.running.append(request)

    def allocate_blocks(self, num_blocks):
        """分配块"""
        with self._block_lock:
            return self.cache_manager.allocate(num_blocks)
```

### 3.3 读写锁设计

```python
from threading import RLock


class RWLock:
    """读写锁实现"""

    def __init__(self):
        self._read_ready = threading.Condition(RLock())
        self._readers = 0

    def read_acquire(self):
        """获取读锁"""
        self._read_ready.acquire()
        self._readers += 1
        self._read_ready.release()

    def read_release(self):
        """释放读锁"""
        self._read_ready.acquire()
        self._readers -= 1
        if self._readers == 0:
            self._read_ready.notify_all()
        self._read_ready.release()

    def write_acquire(self):
        """获取写锁"""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def write_release(self):
        """释放写锁"""
        self._read_ready.release()

    def read_lock(self):
        """读锁上下文管理器"""
        return _ReadLockContext(self)

    def write_lock(self):
        """写锁上下文管理器"""
        return _WriteLockContext(self)


# 使用示例
class ResourceManagerV1:
    def __init__(self):
        self._rw_lock = RWLock()

    def get_running_count(self):
        """读操作：获取运行中请求数"""
        with self._rw_lock.read_lock():  # 读锁，允许并发读
            return len(self.running)

    def add_request(self, request):
        """写操作：添加请求"""
        with self._rw_lock.write_lock():  # 写锁，独占
            self.running.append(request)
```

---

## 四、使用方式变更

### 4.1 重构前

```python
# common_engine.py 中的代码
from fastdeploy import envs

if envs.ENABLE_V1_KVCACHE_SCHEDULER:
    from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1
    self.resource_manager = ResourceManagerV1(...)
else:
    from fastdeploy.engine.resource_manager import ResourceManager
    self.resource_manager = ResourceManager(...)

# 调度逻辑也有两套
if envs.ENABLE_V1_KVCACHE_SCHEDULER:
    self._schedule_request_to_worker_v1()
else:
    self._schedule_request_to_worker()
```

### 4.2 重构后

```python
# common_engine.py 中的代码
from fastdeploy.engine.sched.factory import ResourceManagerFactory
from fastdeploy import envs

# 使用工厂创建，代码统一
version = envs.SCHEDULER_VERSION.get()  # "v0" 或 "v1"
self.resource_manager = ResourceManagerFactory.create(
    version=version,
    config=self.cfg,
)

# 调度逻辑统一（因为接口一致）
def _schedule_request_to_worker(self):
    result = self.resource_manager.schedule()
    if result.scheduled_tasks:
        self.ipc.send_tasks(result.scheduled_tasks)
```

---

## 五、迁移策略

### 5.1 阶段一：接口抽象（1周）

1. 创建 `ResourceManagerBase` 接口
2. 创建 `ResourceManagerFactory` 工厂
3. V1 实现接口（`ResourceManagerV1`）
4. V0 实现接口（`ResourceManagerV0`，包装原代码）

### 5.2 阶段二：代码统一（1周）

1. 修改 `common_engine.py` 使用工厂创建
2. 统一调度调用方式
3. 删除 if/else 分支

### 5.3 阶段三：锁优化（1周）

1. 在 V1 中实现细粒度锁
2. 实现 O(1) 槽位管理
3. 性能测试验证

### 5.4 阶段四：废弃 V0（2周）

1. 添加 V0 废弃警告
2. 迁移剩余使用 V0 的场景
3. 删除 V0 代码

---

## 六、测试策略

### 6.1 接口一致性测试

```python
# tests/engine/sched/test_interface_consistency.py
"""
测试 V0 和 V1 接口行为一致
"""

import pytest
from fastdeploy.engine.sched.factory import ResourceManagerFactory


class TestInterfaceConsistency:
    """验证 V0 和 V1 行为一致"""

    @pytest.fixture(params=["v0", "v1"])
    def resource_manager(self, request, config):
        return ResourceManagerFactory.create(request.param, config)

    def test_allocate_resources(self, resource_manager):
        """测试资源分配"""
        tasks = [create_test_request() for _ in range(3)]

        allocated = resource_manager.allocate_resources(tasks)

        assert len(allocated) <= len(tasks)
        for task in allocated:
            assert task.idx >= 0
            assert task.block_tables is not None

    def test_release_resources(self, resource_manager):
        """测试资源释放"""
        task = create_test_request()
        resource_manager.allocate_resources([task])

        result = resource_manager.release_resources(task.request_id)

        assert result is True
        assert resource_manager.available_batch() > 0

    def test_schedule(self, resource_manager):
        """测试调度"""
        result = resource_manager.schedule()

        assert isinstance(result.scheduled_tasks, list)
        assert isinstance(result.preempted_tasks, list)
        assert result.waiting_count >= 0
        assert result.running_count >= 0
```

### 6.2 性能对比测试

```python
# tests/engine/sched/test_performance.py
"""
性能对比测试
"""

import time
import pytest


class TestPerformance:
    """性能测试"""

    def test_slot_allocation_performance(self):
        """槽位分配性能对比"""
        v0 = ResourceManagerFactory.create("v0", config)
        v1 = ResourceManagerFactory.create("v1", config)

        # V0: O(n) 线性查找
        start = time.time()
        for _ in range(10000):
            v0._find_free_slot()
        v0_time = time.time() - start

        # V1: O(1) set 查找
        start = time.time()
        for _ in range(10000):
            v1.slot_manager.allocate()
            v1.slot_manager.release(0)
        v1_time = time.time() - start

        print(f"V0: {v0_time:.3f}s, V1: {v1_time:.3f}s")
        assert v1_time < v0_time  # V1 应该更快

    def test_concurrent_schedule_performance(self):
        """并发调度性能"""
        # 测试细粒度锁的并发性能提升
        ...
```

---

## 七、风险与回滚

### 7.1 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 接口不兼容 | 低 | 高 | 完善接口测试 |
| 锁优化引入死锁 | 中 | 高 | 代码审查 + 压测 |
| 性能回退 | 低 | 中 | 性能基准测试 |

### 7.2 回滚方案

```python
# 通过环境变量控制版本
export FD_SCHEDULER_VERSION=v0  # 回滚到 V0
export FD_SCHEDULER_VERSION=v1  # 使用 V1
```

---

## 八、目录结构

```
fastdeploy/engine/sched/
├── __init__.py
├── base.py                    # ResourceManagerBase 接口
├── factory.py                 # ResourceManagerFactory 工厂
├── resource_manager_v0.py     # V0 实现（兼容）
├── resource_manager_v1.py     # V1 实现（优化）
├── slot_manager.py            # O(1) 槽位管理
└── locks.py                   # 读写锁实现
```

---

*文档创建时间: 2026-02-14*
