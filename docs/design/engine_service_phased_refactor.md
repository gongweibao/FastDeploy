# EngineService 渐进式重构计划

> **文档版本**: v1.4
> **更新日期**: 2026-02-16
> **状态**: 已审视
> **代码基线**: commit `e2332a111` (branch: develop)

---

## 一、问题分析

### 1.1 核心问题：God Class 反模式

`EngineService` 类 2210 行，承担了 5 种不同职责：

```
┌────────────────────────────────────────────────────┐
│                   EngineService                    │
│                    (2210 行)                       │
├────────────────────────────────────────────────────┤
│  1. 进程管理    - Worker 启动/停止/重启/监控       │
│  2. IPC 通信    - ZMQ socket 管理、消息收发        │
│  3. 资源协调    - 显存分配、模型加载同步、调度     │
│  4. 健康检查    - 心跳、存活检测                   │
│  5. 控制处理    - pause/resume/update_weights     │
└────────────────────────────────────────────────────┘
```

### 1.2 职责分布统计

> **注**: 以下行号基于代码基线 commit `e2332a111`，后续代码变更可能导致行号偏移。

| 职责 | 代码行数 | time.sleep 数量 | 核心方法 |
|------|----------|-----------------|----------|
| 资源协调/调度 | ~450 | 14 | `_schedule_request_to_worker_v1` (268行!) |
| 进程管理 | ~350 | 6 | `_start_worker_service`, `check_worker_initialize_status` |
| IPC 通信 | ~180 | 2 | `start_zmq_service`, `_insert_zmq_task_to_scheduler` |
| 控制处理 | ~170 | 1 | `_control_pause`, `_control_resume` |
| 健康检查 | ~130 | 3 | `check_health`, `_register_to_router` |

### 1.3 27 处 time.sleep 分布

| 类别 | 数量 | 行号示例 | 问题 |
|------|------|----------|------|
| 资源协调忙等待 | 14 | 735, 738, 745, 748... | 大量 0.001s 轮询 |
| Worker 启动等待 | 6 | 228, 248, 2062, 2151... | 固定等待时间 |
| 健康检查/心跳 | 3 | 1701, 1703, 1725 | 定时行为，相对合理 |
| IPC 等待 | 2 | 1099, 1457 | 可优化为事件驱动 |
| 控制流等待 | 1 | 1294 | 轮询等待队列清空 |

---

## 二、重构原则

| 原则 | 说明 |
|------|------|
| **组件内聚** | 每个组件一个 PR，保证代码一致性 |
| **向后兼容** | 每步完成后现有功能不受影响 |
| **可独立验证** | 每步有明确的测试方法 |
| **可随时暂停** | 任意阶段可以停下来 |
| **可回滚** | 保留旧代码路径，环境变量切换 |

---

## 三、分阶段计划

### Phase 1: WorkerManager 抽取 ⬅️ 首先做

**目标**: 将进程管理逻辑（含健康检查）抽取到独立类

**接口定义**:

```python
from typing import Optional, Tuple
from subprocess import Popen

class WorkerManager:
    """
    Manages Worker process lifecycle including start, stop, health check.

    This class is responsible for:
    - Starting/stopping worker subprocess
    - Monitoring worker readiness and health
    - Managing IPC signals for worker communication
    """

    def __init__(
        self,
        cfg: 'EngineConfig',
        ipc_signal_suffix: str,
        logger: Optional['Logger'] = None,
    ):
        """
        Initialize WorkerManager.

        Args:
            cfg: Engine configuration containing worker settings
            ipc_signal_suffix: Suffix for IPC signal names (for multi-instance isolation)
            logger: Optional logger instance
        """
        ...

    # ==================== Public Interface ====================

    def init_signals(self) -> None:
        """
        Initialize all IPC signals for worker communication.
        Must be called before start().

        Raises:
            RuntimeError: If signals already initialized
        """

    def start(self, command: str) -> None:
        """
        Start worker process with given command.

        Args:
            command: Shell command to launch worker process

        Raises:
            RuntimeError: If worker already running
            subprocess.SubprocessError: If failed to start process
        """

    def stop(self, timeout: float = 10.0) -> None:
        """
        Stop worker process gracefully.

        Args:
            timeout: Max seconds to wait for graceful shutdown

        Raises:
            TimeoutError: If worker doesn't stop within timeout (force kill applied)
        """

    def cleanup(self) -> None:
        """
        Cleanup all IPC signals. Should be called after stop().

        Raises:
            RuntimeError: If worker still running
        """

    def is_ready(self) -> bool:
        """
        Check if all workers are ready (non-blocking).

        Returns:
            True if all workers reported ready, False otherwise
        """

    def wait_for_ready(self, timeout: float = 60.0) -> bool:
        """
        Wait for all workers to be ready (blocking).

        Args:
            timeout: Max seconds to wait for all workers ready

        Returns:
            True if all workers ready within timeout, False otherwise
        """

    def check_health(self, timeout: float = 30.0) -> Tuple[bool, str]:
        """
        Check worker health status.

        Args:
            timeout: Max seconds since last heartbeat to consider healthy

        Returns:
            Tuple of (is_healthy, error_message)
            - (True, "") if healthy
            - (False, "reason") if unhealthy
        """

    # ==================== Properties (read-only) ====================

    @property
    def is_running(self) -> bool:
        """Returns True if worker process is running."""

    @property
    def pid(self) -> Optional[int]:
        """Returns worker process PID, or None if not running."""

    @property
    def is_model_loaded(self) -> bool:
        """Returns True if model is loaded."""

    @property
    def model_load_progress(self) -> float:
        """Returns model loading progress (0.0 to 1.0)."""
```

**内部状态**（不对外暴露）:

```python
# Private attributes (internal use only)
self._worker_proc: Optional[Popen]      # Worker subprocess handle
self._worker_init_status: Dict[str, float]  # Init progress {"weight": 0.5, "layer": 0.8}
self._signals_initialized: bool         # Whether init_signals() called

# IPC Signals (internal, accessed via public methods)
self._worker_ready_signal: IPCSignal    # Worker readiness flag
self._loaded_model_signal: IPCSignal    # Model loaded flag
self._worker_healthy_live_signal: IPCSignal  # Health heartbeat timestamp
```

**错误处理**:

| 场景 | 异常类型 | 说明 |
|------|----------|------|
| start() 时 worker 已运行 | `RuntimeError` | 防止重复启动 |
| start() 命令执行失败 | `subprocess.SubprocessError` | 进程启动失败 |
| stop() 超时 | `TimeoutError` | 优雅关闭失败，已强制 kill |
| cleanup() 时 worker 仍在运行 | `RuntimeError` | 必须先 stop() |

**IPC 信号归属**:

| 信号 | 归属 | 被谁使用 | 说明 |
|------|------|----------|------|
| `worker_ready_signal` | WorkerManager | WorkerManager | Worker 就绪状态 |
| `loaded_model_signal` | WorkerManager | EngineService(启动流程) | 模型加载完成 |
| `worker_healthy_live_signal` | WorkerManager | WorkerManager | 健康检查心跳 |
| `exist_task_signal` | **EngineService** | 调度逻辑 | 是否有新任务 |
| `exist_prefill_task_signal` | **EngineService** | 调度逻辑 | 是否进行 prefill |
| `exist_swapped_task_signal` | **EngineService** | 调度逻辑 | 是否有 swapped task |
| `cache_ready_signal` | **EngineService** | 调度逻辑 | Cache 就绪 |
| `model_weights_status_signal` | **EngineService** | 控制逻辑 | 权重更新状态 |
| `kv_cache_status_signal` | **EngineService** | 控制逻辑 | KV Cache 状态 |
| `prefix_tree_status_signal` | **EngineService** | 控制逻辑 | 前缀树状态 |
| `swap_space_ready_signal` | **EngineService** | 调度逻辑 | Swap 空间就绪 |
| `cache_transfer_inited_signal` | **EngineService** | 调度逻辑 | Cache 传输初始化 |

> **关键发现**: 大部分 IPC 信号被调度/控制逻辑使用，不应该放在 WorkerManager

**调整后的 WorkerManager 职责**:

```
WorkerManager 只管理:
├── worker_proc (进程句柄)
├── worker_ready_signal (进程就绪)
├── loaded_model_signal (模型加载)
└── worker_healthy_live_signal (健康心跳)

其他信号保留在 EngineService（供调度/控制逻辑使用）
```

**EngineService 使用 WorkerManager 伪代码**:

```python
class EngineService:

    def __init__(self, cfg):
        self.worker_manager: Optional[WorkerManager] = None
        self._finalizer = weakref.finalize(self, self._cleanup)

    # === 启动阶段 ===
    def _start_worker_service(self):
        # 1. 创建
        self.worker_manager = WorkerManager(cfg, ipc_suffix, logger)

        # 2. 初始化信号
        self.worker_manager.init_signals()

        # 3. 构建命令 (业务逻辑在 EngineService)
        command = self._build_worker_command()

        # 4. 启动
        self.worker_manager.start(command)

        # 5. 等待模型加载（编排逻辑在 EngineService）
        self._wait_for_model_loaded(timeout=600.0)

        # 6. 等待就绪（编排逻辑在 EngineService）
        self._wait_for_ready(timeout=60.0)

    def _wait_for_model_loaded(self, timeout: float):
        """
        Application orchestration logic - NOT in WorkerManager.
        WorkerManager only provides primitives (is_model_loaded, model_load_progress).
        """
        start = time.time()
        with progress_bar() as bar:
            while not self.worker_manager.is_model_loaded:
                if not self.worker_manager.is_running:
                    raise RuntimeError("Worker died during model loading")
                if time.time() - start > timeout:
                    raise TimeoutError("Model loading timeout")
                bar.update(self.worker_manager.model_load_progress)
                time.sleep(0.1)

    def _wait_for_ready(self, timeout: float):
        """
        Application orchestration logic - NOT in WorkerManager.
        """
        start = time.time()
        while not self.worker_manager.is_ready():
            if not self.worker_manager.is_running:
                raise RuntimeError("Worker died during initialization")
            if time.time() - start > timeout:
                raise TimeoutError("Worker ready timeout")
            time.sleep(0.1)

    # === 运行阶段 ===
    def check_health(self) -> Tuple[bool, str]:
        return self.worker_manager.check_health(timeout=30.0)

    # === 关闭阶段 ===
    def _cleanup(self):
        self.worker_manager.stop(timeout=10.0)
        self.worker_manager.cleanup()
```

> **关键设计决策**: `wait_for_model_loaded` 和 `wait_for_ready` 是**应用编排逻辑**，留在 EngineService。
> WorkerManager 只提供**原语**（`is_model_loaded`, `is_ready`, `model_load_progress`），不负责阻塞等待。

**调用时机**:

| 阶段 | 方法 | 触发时机 |
|------|------|----------|
| 启动 | `init_signals()` | `start()` 前 |
| 启动 | `start(command)` | 信号初始化后 |
| 启动 | `is_model_loaded` (属性) | EngineService 轮询 |
| 启动 | `is_ready()` | EngineService 轮询 |
| 运行 | `check_health()` | `/health` API 调用 |
| 关闭 | `stop()` | 服务销毁时 |
| 关闭 | `cleanup()` | stop() 之后 |

**为什么先做这个**:
- 边界相对清晰，主要是进程生命周期管理
- 对其他模块依赖较少

**涉及方法** (调整后):

| 方法 | 行号 | 说明 | 归属 |
|------|------|------|------|
| `_start_worker_service` | 1930-2051 | 构建启动命令 + 执行 | 拆分：构建→EngineService，执行→WorkerManager |
| `check_worker_initialize_status` | 2153-2210 | 检测初始化状态 | ✅ WorkerManager |
| `_worker_processes_ready` | 1811-1818 | 判断进程就绪 | ✅ WorkerManager |
| `_init_worker_signals` | 1820-1879 | 初始化 IPC 信号 | ✅ WorkerManager |
| `_setting_environ_variables` | 1881-1928 | 设置环境变量 | ✅ EngineService (构建命令用) |
| `check_health` | 2070-2080 | 健康检查 | ✅ WorkerManager |
| `_init_worker_monitor_signals` | 284-382 | 初始化监控信号 | ✅ WorkerManager |

> **关键变化**: `_start_worker_service` 拆分为命令构建（EngineService）和执行（WorkerManager）

**拆分为 1 个 PR**:

| PR | 内容 | 改动量 |
|----|------|--------|
| PR1 | 创建 WorkerManager 类，迁移所有进程管理方法 | ~500 行 |

> **说明**: 一次性迁移保证代码一致性，避免中间状态不稳定

**验证方式**:
- Worker 启停功能正常
- 模型加载正常
- 健康检查 API 正常
- 现有测试通过

---

### Phase 2: ZmqCommunicator 抽取

**目标**: 将 ZMQ 通信逻辑抽取到独立类

**涉及方法** (约 180 行):
- `start_zmq_service` (1082-1104)
- `_insert_zmq_task_to_scheduler` (1106-1222)
- `_zmq_send_generated_tokens` (1449-1521)
- `_send_error_response` (1418-1433)

**为什么需要抽取**：
- ZMQ 是**外部通信资源**，类似 WorkerManager 管理外部进程
- socket 生命周期管理是独立职责
- 可以独立测试通信逻辑

**接口定义**:

```python
from typing import Optional, Callable, Any
from dataclasses import dataclass

@dataclass
class ZmqConfig:
    """ZMQ 通信配置"""
    recv_address: str           # 接收地址，如 "tcp://*:8080"
    send_address: str           # 发送地址，如 "tcp://*:8081"
    recv_timeout: int = 1000    # 接收超时 (ms)
    send_timeout: int = 1000    # 发送超时 (ms)
    high_water_mark: int = 0    # 高水位标记 (0=无限制)


class ZmqCommunicator:
    """
    Manages ZMQ socket communication for request/response handling.

    This class is responsible for:
    - Creating and managing ZMQ sockets (PULL for recv, PUSH for send)
    - Receiving requests from external clients
    - Sending responses back to clients
    - Thread-safe message handling
    """

    def __init__(
        self,
        config: ZmqConfig,
        on_request: Callable[[dict], None],
        logger: Optional['Logger'] = None,
    ):
        """
        Initialize ZmqCommunicator.

        Args:
            config: ZMQ configuration
            on_request: Callback invoked when a request is received.
                        Signature: (request_dict) -> None
            logger: Optional logger instance
        """
        ...

    # ==================== Lifecycle ====================

    def start(self) -> None:
        """
        Start ZMQ service and begin listening for requests.

        Spawns a background thread for receiving messages.

        Raises:
            RuntimeError: If already started
            zmq.ZMQError: If failed to bind sockets
        """

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop ZMQ service and close all sockets.

        Args:
            timeout: Max seconds to wait for pending sends

        Raises:
            TimeoutError: If pending messages not sent within timeout
        """

    # ==================== Messaging ====================

    def send_response(
        self,
        request_id: str,
        response: dict,
        is_final: bool = False,
    ) -> None:
        """
        Send response for a request.

        Args:
            request_id: Original request identifier
            response: Response data dict
            is_final: Whether this is the final response for streaming

        Raises:
            RuntimeError: If communicator not started
            ValueError: If request_id unknown
        """

    def send_error(
        self,
        request_id: str,
        error_code: int,
        error_message: str,
    ) -> None:
        """
        Send error response for a request.

        Args:
            request_id: Original request identifier
            error_code: Error code
            error_message: Human-readable error description

        Raises:
            RuntimeError: If communicator not started
        """

    # ==================== Properties ====================

    @property
    def is_running(self) -> bool:
        """Returns True if communicator is running."""

    @property
    def pending_requests(self) -> int:
        """Returns number of requests awaiting response."""
```

**错误处理**:

| 场景 | 异常类型 | 说明 |
|------|----------|------|
| start() 时已启动 | `RuntimeError` | 防止重复启动 |
| start() 绑定失败 | `zmq.ZMQError` | 端口占用等 |
| stop() 超时 | `TimeoutError` | 待发送消息未完成 |
| send 时未启动 | `RuntimeError` | 需先调用 start() |
| send 未知 request_id | `ValueError` | 请求已过期或不存在 |

**EngineService 使用 ZmqCommunicator 伪代码**:

```python
class EngineService:

    def __init__(self, cfg):
        self.zmq_comm: Optional[ZmqCommunicator] = None

    def _start_zmq_service(self):
        config = ZmqConfig(
            recv_address=f"tcp://*:{self.cfg.zmq_recv_port}",
            send_address=f"tcp://*:{self.cfg.zmq_send_port}",
        )
        self.zmq_comm = ZmqCommunicator(
            config=config,
            on_request=self._handle_zmq_request,  # 回调处理请求
            logger=self.logger,
        )
        self.zmq_comm.start()

    def _handle_zmq_request(self, request: dict):
        """请求处理回调 - 业务逻辑保留在 EngineService"""
        # 验证请求、加入调度队列等
        self.insert_tasks(request)

    def _send_generated_tokens(self, request_id: str, tokens: list):
        """发送生成结果"""
        self.zmq_comm.send_response(
            request_id=request_id,
            response={"tokens": tokens},
            is_final=False,
        )

    def _cleanup(self):
        if self.zmq_comm:
            self.zmq_comm.stop()
```

**拆分为 1 个 PR**:

| PR | 内容 | 改动量 |
|----|------|--------|
| PR2 | 创建 ZmqCommunicator 类，迁移所有 ZMQ 通信方法 | ~200 行 |

**验证方式**:
- ZMQ 消息收发正常
- 流式响应正常
- 错误响应正常
- 现有测试通过

---

### Phase 3: SchedulerBridge 重构（待详细规划）

**目标**: 重构最复杂的调度逻辑

**涉及方法** (约 450 行):
- `_schedule_request_to_worker` (725-796)
- `_schedule_request_to_worker_v1` (798-1066) ← **268行的巨型方法**
- `insert_tasks` (435-529)
- `_insert_prefilled_requests` (531-581)

> **注意**: 这个阶段改动最大，风险最高。**不在当前范围内**，需要单独详细规划。
> 建议：先在 EngineService 内部将 268 行方法拆分为小方法，再考虑是否抽取。

---

### Phase 4: TimingConfig 抽取（可选，最后做）

**目标**: 将 27 处 sleep 的硬编码值抽取为可配置参数

> **注**: 这不是核心问题，sleep 硬编码不影响功能正确性。可作为后续优化，或在各组件拆分时顺便处理。

**实现方式**:
```python
@dataclass
class TimingConfig:
    worker_start_wait: float = 5.0
    scheduler_poll_interval: float = 0.001
    health_check_interval: float = 5.0
    # ...
```

---

## 四、目标架构

```
EngineService (~520行，协调者 + 控制者)
  │
  ├── 控制逻辑 (~170行) ← 不抽取，EngineService 的核心职责
  │     └── pause/resume/update_weights
  │
  ├── WorkerManager (进程管理+健康检查 ~480行)
  │     ├── start() / stop()
  │     ├── is_ready() / wait_for_ready()
  │     └── check_health()
  │
  ├── ZmqCommunicator (IPC通信 ~180行)
  │     ├── start()
  │     ├── send_response()
  │     └── receive_request()
  │
  └── (SchedulerBridge 待详细规划 ~450行)
```

> **设计决策**：
> - ControlHandler **不抽取**：控制自身状态是 EngineService 的核心职责
> - WorkerManager **抽取**：管理外部进程，独立职责
> - ZmqCommunicator **抽取**：管理外部通信资源，独立职责

---

## 五、关键文件

| 文件 | 说明 |
|------|------|
| `fastdeploy/engine/common_engine.py` | 现有代码，需要重构 |
| `fastdeploy/engine/components/` | 新建目录 |
| `fastdeploy/engine/components/__init__.py` | 组件导出 |
| `fastdeploy/engine/components/worker_manager.py` | Phase 1 |
| `fastdeploy/engine/components/zmq_communicator.py` | Phase 2 |
| `fastdeploy/engine/components/timing_config.py` | Phase 4 (可选) |

---

## 六、验证方法

每个 Phase 完成后:

1. **集成测试**（Gold Base）: 用真实对象验证行为正确
2. **单元测试**: 基于 Gold Base 设计 mock，测试边缘情况
3. **回归测试**: 现有功能不受影响
4. **性能测试**: time.sleep 行为不变（除非显式优化）

### 6.1 测试策略：先集成测试，再单元测试

```
┌─────────────────────────────────────────────────────────────┐
│                    测试策略                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: 集成测试（真实进程）                                │
│  ├── 验证 start/stop 真实行为                                │
│  ├── 验证 IPC 信号交互                                       │
│  └── 作为 Gold Base，确保理解真实行为                         │
│                                                             │
│  Phase 2: 单元测试（patch subprocess）                       │
│  ├── 基于 Gold Base 设计 mock 行为                           │
│  ├── 测试边缘情况（启动失败、超时等）                          │
│  └── 快速执行，CI 友好                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> **核心原则**：集成测试是验证 mock 正确性的依据，不能跳过。

### 6.2 WorkerManager 测试方案

**测试架构**:

```
tests/
├── mock_worker.py              # 轻量 Mock Worker（不依赖 GPU）
├── test_worker_manager.py      # WorkerManager 测试用例
└── conftest.py                 # pytest fixtures
```

**conftest.py 定义**:

```python
# tests/conftest.py
import pytest
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockEngineConfig:
    """用于测试的最小配置"""
    worker_num: int = 4
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    devices: str = "0,1,2,3"
    model_path: str = "/tmp/mock_model"


@pytest.fixture
def unique_ipc_suffix():
    """
    生成唯一的 IPC 后缀，避免测试并行时共享内存冲突。
    """
    return f"_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_engine_config():
    """提供默认的测试配置"""
    return MockEngineConfig()


def make_test_config(
    worker_num: int = 4,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    devices: Optional[str] = None,
) -> MockEngineConfig:
    """
    创建测试用的 EngineConfig。

    Args:
        worker_num: Worker 进程数量
        tensor_parallel_size: 张量并行度
        pipeline_parallel_size: 流水线并行度
        devices: GPU 设备列表，如 "0,1,2,3"

    Returns:
        MockEngineConfig 实例
    """
    if devices is None:
        devices = ",".join(str(i) for i in range(worker_num))
    return MockEngineConfig(
        worker_num=worker_num,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        devices=devices,
    )
```

**核心思路**: 使用真实的 `paddle.distributed.launch` 启动 Mock Worker 进程，验证真实的进程管理和 IPC 信号交互。

**Mock Worker 设计**:

```python
# tests/mock_worker.py
"""
轻量 Mock Worker - 只操作 IPCSignal，不依赖 Paddle GPU
paddle.distributed.launch 自动设置 PADDLE_TRAINER_ID 等环境变量
"""
import os
import time
import numpy as np
from fastdeploy.inter_communicator import IPCSignal

def main():
    suffix = os.environ['MOCK_IPC_SUFFIX']
    rank = int(os.environ.get('PADDLE_TRAINER_ID', '0'))
    worker_num = int(os.environ['MOCK_WORKER_NUM'])

    # 连接 WorkerManager 创建的共享内存 (create=False)
    # 参数: name, shape, dtype, suffix, create
    ready_signal = IPCSignal(
        name="worker_ready_signal",
        shape=(worker_num,),
        dtype=np.int32,
        suffix=suffix,
        create=False
    )
    health_signal = IPCSignal(
        name="worker_healthy_live_signal",
        shape=(worker_num,),
        dtype=np.int64,
        suffix=suffix,
        create=False
    )

    # 模拟启动延迟
    time.sleep(float(os.getenv('MOCK_WORKER_DELAY', '0.1')))

    # 写入 ready 信号
    if not os.getenv('MOCK_WORKER_SKIP_READY'):
        ready_signal.value[rank] = 1

    # 心跳循环
    start_time = time.time()
    while True:
        if not os.getenv('MOCK_WORKER_STOP_HEARTBEAT'):
            health_signal.value[rank] = int(time.time())
        time.sleep(0.5)

        # 模拟崩溃
        crash_after = os.getenv('MOCK_WORKER_CRASH_AFTER')
        if crash_after:
            elapsed = time.time() - start_time
            if elapsed > float(crash_after):
                os._exit(1)

if __name__ == "__main__":
    main()
```

**进程结构** (与生产环境一致):

```
测试进程
    │
    └── WorkerManager
           │
           └── paddle.distributed.launch (进程组组长)
                  ├── mock_worker.py (rank=0)
                  ├── mock_worker.py (rank=1)
                  └── ...
```

**测试用例覆盖**:

| 场景 | 环境变量 | 预期行为 |
|------|----------|----------|
| 正常启动 | 默认 | `wait_for_ready()` 返回 True |
| 启动超时 | `MOCK_WORKER_DELAY=999` | `wait_for_ready(timeout=1)` 返回 False |
| 启动失败 | `MOCK_WORKER_SKIP_READY=1` | ready 信号不完整 |
| 运行中崩溃 | `MOCK_WORKER_CRASH_AFTER=2` | `check_health()` 返回 (False, ...) |
| 心跳超时 | `MOCK_WORKER_STOP_HEARTBEAT=1` | `check_health()` 检测到超时 |
| 进程 hang | 无信号响应 | `stop(timeout=1)` 强制 kill 成功 |

**测试命令示例**:

```python
@pytest.mark.timeout(30)
def test_worker_start_and_ready(unique_ipc_suffix):
    cfg = make_test_config(worker_num=4)
    manager = WorkerManager(cfg, unique_ipc_suffix)
    manager.init_signals()

    cmd = (
        f"MOCK_IPC_SUFFIX={unique_ipc_suffix} MOCK_WORKER_NUM=4 "
        f"python -m paddle.distributed.launch --devices 0,1,2,3 "
        f"tests/mock_worker.py"
    )
    manager.start(cmd)

    assert manager.wait_for_ready(timeout=10.0)
    assert manager.is_ready()

    manager.stop()
    manager.cleanup()
```

**关键验证点**:

- ✅ 真实进程组创建和销毁 (`os.killpg`)
- ✅ 多 rank 信号写入 (`signal.value[rank] = 1`)
- ✅ 就绪判断 (`np.sum(signal.value) == worker_num`)
- ✅ 心跳超时检测
- ✅ 不依赖 GPU

**CI 配置**:

- 所有测试强制 30s 超时 (`@pytest.mark.timeout(30)`)
- 每个测试使用独立 IPC suffix (`unique_ipc_suffix` fixture)
- 进程类测试单独 job，限制并行度

---

## 七、回滚策略

```bash
# 方式 1: Git revert
git revert <commit-hash>

# 方式 2: 环境变量切换（保留旧代码路径时）
export FD_USE_LEGACY_ENGINE_SERVICE=1
```

---

## 八、已讨论结论

| 问题 | 结论 |
|------|------|
| PR 粒度 | ✅ 每个组件一个 PR，保证一致性 |
| 健康检查归属 | ✅ 归属 WorkerManager |
| TimingConfig 优先级 | ✅ 调整为最后（可选） |
| wait_for_model_loaded 归属 | ✅ 是应用编排，留在 EngineService；WorkerManager 只提供 `is_model_loaded` 属性 |
| ServiceRegistry | ✅ 不单独抽取，保留在 EngineService（服务注册是应用级编排逻辑） |
| ControlHandler | ✅ **不抽取**：控制自身状态是 EngineService 的核心职责 |
| Phase 3 (SchedulerBridge) | ✅ 待详细规划，不在当前范围 |
| 测试策略 | ✅ 先集成测试建立 Gold Base，再单元测试 |

---

## 九、待办事项

1. **Phase 1 实施**: 创建 WorkerManager 类
2. **Phase 2 实施**: 创建 ZmqCommunicator 类
3. **Phase 3 详细规划**: SchedulerBridge 268行方法拆分方案
