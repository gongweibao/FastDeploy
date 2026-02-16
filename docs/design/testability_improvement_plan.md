# FastDeploy 可测试性改进计划（全景图）

> **创建日期**: 2026-02-15
> **状态**: 草案讨论中

## Context

**问题**: FastDeploy 代码库存在可测试性问题，核心模块难以进行单元测试。

**根本原因**: 模块内部直接创建外部依赖（ZMQ、IPC Signal、Redis 等），无法 mock。

**目标**: 通过依赖注入，让核心模块可以脱离真实 ZMQ 和 GPU 独立测试。

**本次聚焦**: 依赖注入改造（time.sleep 问题后续单独处理）

---

## 设计原则（讨论达成的共识）

1. **EngineService 本身就是协调者** - 不需要抽 SchedulerCoordinator
2. **IPCManager 有价值** - 隔离 ZMQ 通信细节，可 mock
3. **测试边界清晰** - 输入来自 IPC，输出到 WorkerQueue
4. **依赖注入** - 构造函数接受依赖，而不是内部创建

---

## 可测试性问题全景图

### 按外部依赖分类

#### 1. ZMQ 依赖（9 个模块）

| 模块 | 文件 | 当前问题 | 改进方案 |
|------|------|----------|----------|
| EngineService | `engine/common_engine.py` | 直接创建 ZmqIpcServer | 注入 `IPCManager` |
| AsyncLLM | `engine/async_llm.py` | 直接创建 ZmqIpcClient | 注入 `IPCManager` |
| GPUModelRunner | `worker/gpu_model_runner.py` | 导入 zmq | 注入 `ZmqClientInterface` |
| XPUModelRunner | `worker/xpu_model_runner.py` | 导入 zmq | 注入 `ZmqClientInterface` |
| MetaXModelRunner | `worker/metax_model_runner.py` | 导入 zmq | 注入 `ZmqClientInterface` |
| ZmqIpcServer | `inter_communicator/zmq_server.py` | 直接创建 Context | 注入 `ZmqContextFactory` |
| ZmqIpcClient | `inter_communicator/zmq_client.py` | 直接创建 Context | 注入 `ZmqContextFactory` |
| api_server | `entrypoints/openai/api_server.py` | 导入 zmq | 通过 EngineClient 隔离 |
| SplitwiseConnector | `splitwise/splitwise_connector.py` | 直接使用 zmq | 注入 `ZmqClientInterface` |

#### 2. IPC Signal 依赖（6 个模块）

| 模块 | 文件 | 当前问题 | 改进方案 |
|------|------|----------|----------|
| ResourceManagerV1 | `engine/sched/resource_manager_v1.py` | 直接创建 IPCSignal | 注入 `SignalFactory` |
| EngineClient | `entrypoints/engine_client.py` | 直接创建多个 IPCSignal | 注入 `SignalFactory` |
| AsyncLLM | `engine/async_llm.py` | 直接创建 IPCSignal | 注入 `SignalFactory` |
| WorkerProcess | `worker/worker_process.py` | 直接创建 IPCSignal | 注入 `SignalFactory` |
| EngineWorkerQueue | `inter_communicator/engine_worker_queue.py` | 直接创建 Manager | 注入 `QueueManagerFactory` |
| TokenProcessor | `output/token_processor.py` | 可能依赖 Signal | 检查后确认 |

#### 3. Redis 依赖（2 个模块）

| 模块 | 文件 | 当前问题 | 改进方案 |
|------|------|----------|----------|
| GlobalScheduler | `scheduler/global_scheduler.py` | 直接创建 ConnectionPool | 注入 `RedisClientInterface` |
| SplitWiseScheduler | `scheduler/splitwise_scheduler.py` | 直接创建 Redis 连接 | 注入 `RedisClientInterface` |

#### 4. 进程/分布式依赖（3 个模块）

| 模块 | 文件 | 当前问题 | 改进方案 |
|------|------|----------|----------|
| AsyncLLM | `engine/async_llm.py` | 直接创建 multiprocessing.Process | 注入 `ProcessFactory` |
| WorkerProcess | `worker/worker_process.py` | 直接调用 fleet.init() | 注入 `DistributedEnvFactory` |
| LLM | `entrypoints/llm.py` | 直接创建 LLMEngine | 注入 `EngineFactory` |

#### 5. 全局状态问题（3 个模块）

| 模块 | 文件 | 当前问题 | 改进方案 |
|------|------|----------|----------|
| api_server | `entrypoints/openai/api_server.py` | 模块级 llm_engine = None | 封装到类实例 |
| tbo | `worker/tbo.py` | 模块级 GLOBAL_THREAD_INFO | 封装到类实例 |
| parser | `entrypoints/openai/api_server.py` | 导入时执行 parse_args() | 延迟初始化 |

#### 6. 平台特定代码（5 个模块）

| 模块 | 文件 | 当前问题 | 改进方案 |
|------|------|----------|----------|
| GPUModelRunner | `worker/gpu_model_runner.py` (3123行) | if platform.is_xxx() 条件导入 | `PlatformOpsInterface` |
| MetaXModelRunner | `worker/metax_model_runner.py` (3065行) | 平台特定代码 | `PlatformOpsInterface` |
| XPUModelRunner | `worker/xpu_model_runner.py` (1946行) | 平台特定代码 | `PlatformOpsInterface` |
| HPUModelRunner | `worker/hpu_model_runner.py` (1827行) | 平台特定代码 | `PlatformOpsInterface` |
| GCUModelRunner | `worker/gcu_model_runner.py` (1222行) | 平台特定代码 | `PlatformOpsInterface` |

---

## 需要新建的抽象接口

| 接口 | 职责 | 隔离的依赖 |
|------|------|-----------|
| `IPCManager` | 进程间通信 | ZMQ |
| `SignalFactory` | 创建 IPC Signal | 共享内存 |
| `RedisClientInterface` | Redis 操作 | Redis |
| `ProcessFactory` | 创建子进程 | multiprocessing |
| `QueueManagerFactory` | 创建队列管理器 | multiprocessing.Manager |
| `PlatformOpsInterface` | 平台特定操作 | GPU/XPU/HPU 等硬件 |
| `DistributedEnvFactory` | 分布式环境初始化 | paddle.distributed |

---

## 改进优先级

### P0: 核心引擎（影响最大）

| 模块 | 行数 | 需注入的依赖 |
|------|------|-------------|
| EngineService | 2210 | IPCManager |
| ResourceManagerV1 | 1472 | SignalFactory |
| AsyncLLM | ~500 | IPCManager, SignalFactory, ProcessFactory |

### P1: Worker 层

| 模块 | 行数 | 需注入的依赖 |
|------|------|-------------|
| GPUModelRunner | 3123 | PlatformOpsInterface, ZmqClientInterface |
| WorkerProcess | 1216 | SignalFactory, DistributedEnvFactory |
| EngineWorkerQueue | ~800 | QueueManagerFactory |

### P2: 调度层

| 模块 | 行数 | 需注入的依赖 |
|------|------|-------------|
| GlobalScheduler | ~500 | RedisClientInterface |
| SplitWiseScheduler | ~800 | RedisClientInterface |

### P3: 入口层

| 模块 | 需注入的依赖 |
|------|-------------|
| EngineClient | SignalFactory |
| api_server | 全局状态封装 |
| LLM | EngineFactory |

---

## IPCManager 接口设计（P0 详细方案）

### 接口定义

```python
# fastdeploy/engine/components/ipc_manager.py

class IPCManager:
    """进程间通信管理器 - 封装 ZMQ 细节"""

    def __init__(self, config: IPCConfig): ...

    # 生命周期
    def start(self) -> None: ...
    def stop(self) -> None: ...

    # 请求接收（来自 API Server）
    def receive_request(self, block: bool = True) -> Tuple[Optional[Exception], Optional[dict]]: ...

    # 响应发送（回 API Server）
    def send_response(self, request_id: str, response: List[Any]) -> bool: ...

    # 状态查询
    def is_connected(self) -> bool: ...
```

### EngineService 改造

```python
# 改造前
class EngineService:
    def __init__(self, cfg):
        self.recv_request_server = ZmqIpcServer(...)  # 直接创建
        self.send_response_server = ZmqIpcServer(...)

# 改造后
class EngineService:
    def __init__(self, cfg, ipc: IPCManager = None):
        self.ipc = ipc or IPCManager(cfg)  # 可注入
```

### 测试示例

```python
def test_request_scheduling():
    mock_ipc = MockIPCManager()
    mock_ipc.push_request({"request_id": "test", "prompt_token_ids": [1,2,3]})

    engine = EngineService(cfg, ipc=mock_ipc)
    engine.schedule_once()

    responses = mock_ipc.get_sent_responses()
    assert "test" in responses
```

---

## 统计汇总

| 分类 | 模块数 | 关键接口 |
|------|--------|----------|
| ZMQ 依赖 | 9 | IPCManager |
| IPC Signal 依赖 | 6 | SignalFactory |
| Redis 依赖 | 2 | RedisClientInterface |
| 进程/分布式依赖 | 3 | ProcessFactory |
| 全局状态问题 | 3 | 封装到类 |
| 平台特定代码 | 5 | PlatformOpsInterface |
| **总计** | **28 处** | **7 个接口** |

---

## 下一步行动建议

1. **先做 P0**：EngineService + IPCManager，验证依赖注入方案
2. **建立 Mock 基础设施**：创建 MockIPCManager 等测试替身
3. **逐步推进**：P0 验证后再扩展到 P1、P2、P3
