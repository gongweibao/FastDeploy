# EngineService 解耦详细设计

> **文档版本**: v1.0
> **创建日期**: 2026-02-14
> **关联问题**: EngineService 职责过重（2210行）、time.sleep 泛滥

---

## 一、背景与目标

### 1.1 当前问题

`fastdeploy/engine/common_engine.py` 中的 `EngineService` 类存在以下问题：

| 问题 | 现状 | 影响 |
|------|------|------|
| 代码规模 | 2210 行 | 难以阅读和维护 |
| 职责混杂 | 调度、IPC、进程管理、资源协调等 | 违反单一职责原则 |
| time.sleep | 27 处硬编码 | 性能不可调优 |
| 测试困难 | 需要启动整个系统 | 单测覆盖率低 |

### 1.2 重构目标

- 将 EngineService 拆分为 4-5 个独立组件
- 每个组件 < 500 行
- 支持独立单元测试
- sleep 间隔可配置

### 1.3 `use_async_llm` 双模式问题

#### 1.3.1 当前状况

当前 `EngineService` 通过 `use_async_llm` 参数支持两种运行模式：

| 模式 | 使用场景 | Worker 管理者 | 调用方 |
|------|---------|--------------|--------|
| **同步模式** (`use_async_llm=False`) | 离线批量推理 | `LLMEngine` | `LLMEngine` |
| **异步模式** (`use_async_llm=True`) | 在线 API 服务 | `EngineService` | `AsyncLLM` (通过 ZMQ) |

```
同步模式                                异步模式
========                               ========

┌─────────────────────┐               ┌─────────────────────┐
│     LLMEngine       │               │      AsyncLLM       │
│  ┌───────────────┐  │               │   (ZMQ Client)      │
│  │EngineService │  │               └──────────┬──────────┘
│  │(不启动Worker) │  │                          │ ZMQ
│  └───────────────┘  │                          ▼
│                     │               ┌─────────────────────┐
│  worker_proc ───────┼──┐            │   EngineService     │
└─────────────────────┘  │            │  (启动+管理Worker)   │
                         │            │                     │
                         ▼            │  worker_proc ───────┼──┐
                   ┌──────────┐       └─────────────────────┘  │
                   │ Workers  │                                │
                   │ (多卡TP) │        ┌───────────────────────┘
                   └──────────┘        ▼
                                 ┌──────────┐
                                 │ Workers  │
                                 │ (多卡TP) │
                                 └──────────┘
```

#### 1.3.2 存在的问题

这种设计存在以下问题：

| 问题 | 说明 |
|------|------|
| **职责不清** | Worker 管理逻辑分散在 `LLMEngine` 和 `EngineService` 两处 |
| **代码重复** | `_start_worker_service()` 在两处有类似实现 |
| **维护困难** | 修改 Worker 启动逻辑需要同时改两个地方 |
| **设计不一致** | 相同功能，两种模式实现方式不同 |
| **历史包袱** | 并非刻意设计，而是代码演进过程中逐渐形成 |

#### 1.3.3 推测的历史演进

```
阶段1：最初只有 LLMEngine
├── LLMEngine 负责一切（调度 + Worker 管理）
│
阶段2：需要支持异步 API 服务
├── 抽出 EngineService 作为共享逻辑
├── AsyncLLM 通过 ZMQ 调用 EngineService
├── EngineService 需要自己管理 Worker（因为运行在独立进程）
│
阶段3：为了兼容，加了 use_async_llm 参数
├── LLMEngine 继续用老方式（自己管 Worker）
├── AsyncLLM 用新方式（EngineService 管 Worker）
└── 两套逻辑并存 → 产生了现在的混乱
```

#### 1.3.4 重构方案：统一 Worker 管理

**核心原则**：Worker 管理统一由 `EngineService`（通过 `ProcessManager` 组件）负责，消除 `use_async_llm` 分支。

**重构后架构**：

```
┌─────────────────┐     ┌─────────────────┐
│    LLMEngine    │     │    AsyncLLM     │
│   (离线推理)     │     │   (在线服务)     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │  直接调用              │ ZMQ
         ▼                       ▼
┌─────────────────────────────────────────┐
│            EngineService                │
│  ┌─────────────────────────────────┐    │
│  │      ProcessManager             │    │
│  │   (统一管理 Worker 进程)         │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
                    │
                    ▼
             ┌──────────┐
             │ Workers  │
             │ (多卡TP) │
             └──────────┘
```

**变化点**：

| 变化 | 重构前 | 重构后 |
|------|-------|-------|
| Worker 启动 | 同步模式在 `LLMEngine`，异步模式在 `EngineService` | 统一在 `EngineService.ProcessManager` |
| `use_async_llm` 参数 | 控制是否启动 Worker | **删除**，不再需要 |
| `LLMEngine` 职责 | 调度 + Worker 管理 | 仅作为调用入口，委托给 `EngineService` |
| 两种模式差异 | Worker 管理 + 通信方式 | **仅通信方式不同**（直接调用 vs ZMQ） |

**重构后的门面类**：

```python
class EngineService:
    """
    引擎服务门面（重构后）

    不再需要 use_async_llm 参数，Worker 管理统一由 ProcessManager 负责
    """

    def __init__(self, cfg: FDConfig, start_queue: bool = True):
        self.cfg = cfg
        self.running = False

        # 初始化各组件
        self.ipc = IPCManager(cfg, start_queue)
        self.resource = ResourceCoordinator(cfg)
        self.scheduler = SchedulerCoordinator(cfg, self.resource, self.ipc)
        self.process = ProcessManager(cfg, self.ipc)  # 统一管理 Worker

    def start(self):
        """启动引擎服务（统一流程）"""
        self.running = True
        self.ipc.start()
        self.resource.start()
        self.scheduler.start()
        self.process.start_workers()  # 统一由 ProcessManager 启动
```

**重构后的 LLMEngine**：

```python
class LLMEngine:
    """
    离线推理引擎（重构后）

    不再自己管理 Worker，完全委托给 EngineService
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = EngineService(cfg)  # 不再传 use_async_llm

    def start(self):
        self.engine.start()  # EngineService 会启动 Worker
        # LLMEngine 不再调用 _start_worker_service()
```

#### 1.3.5 收益

| 收益 | 说明 |
|------|------|
| **职责清晰** | Worker 管理集中在 `ProcessManager`，`LLMEngine` 只是调用入口 |
| **消除重复** | `_start_worker_service()` 只有一处实现 |
| **易于维护** | 修改 Worker 逻辑只需改一个地方 |
| **设计一致** | 两种模式（离线/在线）使用相同的底层实现 |
| **便于测试** | `ProcessManager` 可以独立测试 |

### 1.4 接口兼容策略

#### 1.4.1 问题分析

当前 LLMEngine 直接访问 EngineService 的内部属性：

```python
# 当前 LLMEngine 的调用方式（暴露内部实现）
self.engine.scheduler.put_requests(...)
self.engine.resource_manager.available_block_num()
self.engine.data_processor.process_request(...)
self.engine.worker_healthy_live_signal.value[0]
```

这些属于"实现暴露"而非"接口设计"，但已被广泛使用。

#### 1.4.2 策略选择

| 策略 | 说明 | 风险 | 选择 |
|------|------|------|------|
| **策略 A：保持兼容** | 通过 `@property` 代理，LLMEngine 无需修改 | 低 | **✓ 采用** |
| 策略 B：清晰接口 | 提供新 API，需修改 LLMEngine | 高 | 未来考虑 |

**选择理由**：
1. **一次只做一件事**：本次重构目标是代码组织，接口重设计是另一个目标
2. **降低风险**：同时改内部实现和外部接口会增加回归风险
3. **渐进式演进**：先完成内部重构，验证稳定后再考虑接口优化

#### 1.4.3 实现方式

```python
class EngineService:
    """重构后的 EngineService，保持接口兼容"""

    def __init__(self, cfg: FDConfig, start_queue: bool = True):
        self.cfg = cfg

        # 内部使用新组件
        self._scheduler_coord = SchedulerCoordinator(cfg)
        self._resource_coord = ResourceCoordinator(cfg)
        self._ipc = IPCManager(cfg, start_queue)
        self._process = ProcessManager(cfg, self._ipc)

    # ========== 兼容属性（代理到新组件）==========

    @property
    def scheduler(self):
        """兼容：LLMEngine 仍可访问 engine.scheduler"""
        return self._scheduler_coord.scheduler

    @property
    def resource_manager(self):
        """兼容：LLMEngine 仍可访问 engine.resource_manager"""
        return self._resource_coord.resource_manager

    @property
    def data_processor(self):
        """兼容：LLMEngine 仍可访问 engine.data_processor"""
        return self._data_processor

    @property
    def worker_healthy_live_signal(self):
        """兼容：LLMEngine 仍可访问 engine.worker_healthy_live_signal"""
        return self._ipc.worker_healthy_live_signal

    @property
    def split_connector(self):
        """兼容：LLMEngine 仍可访问 engine.split_connector"""
        return self._split_connector
```

#### 1.4.4 演进路线

```
Phase 1（本次重构）
├── 内部拆分为 4-5 个组件
├── 通过 @property 保持接口兼容
└── LLMEngine / AsyncLLM 无需修改

Phase 2（未来优化）
├── 标记旧属性为 @deprecated
├── 提供新的公开 API
│   - add_request(request) -> str
│   - get_results() -> Dict
│   - available_blocks() -> int
└── 新代码使用新 API

Phase 3（清理）
├── 迁移所有调用方到新 API
└── 移除兼容属性
```

#### 1.4.5 兼容性保证

| 调用方 | 是否需要修改 | 说明 |
|-------|-------------|------|
| LLMEngine | **否** | 兼容属性保证无感知 |
| AsyncLLM | **否** | ZMQ 协议不变 |
| Worker | **否** | Queue/Signal 契约不变 |
| 外部用户代码 | **否** | 公开接口不变 |

### 1.5 与外部进程管理器的协调（RL 场景）

#### 1.5.1 问题背景

在 RL（强化学习）场景中，通常存在一个上层的 `InferenceManager`（由 RL 训练框架提供），负责管理推理实例的生命周期。此时 FastDeploy 内部的 `ProcessManager` 与外部 `InferenceManager` 会产生职责重叠：

```
┌─────────────────────────────────────────────────────────────┐
│              RL Framework (训练框架)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            InferenceManager                          │   │
│  │     (管理多个推理实例、跨节点调度、弹性伸缩)            │   │
│  └──────────────────────┬──────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │  ???  <-- 边界模糊点
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FastDeploy (推理引擎)                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            ProcessManager                            │   │
│  │         (管理节点内 Worker 进程)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**核心问题**：两个 Manager 如何融合共存？

#### 1.5.2 融合模式选择

| 模式 | 说明 | 适用场景 | 选择 |
|------|------|---------|------|
| **模式 A：层级委托** | InferenceManager 管实例，ProcessManager 管进程 | 通用场景 | **✓ 推荐** |
| 模式 B：ProcessManager 禁用 | 外部完全接管进程管理 | 深度集成 | 可选支持 |
| 模式 C：抽象统一接口 | 定义 IProcessLifecycle 接口 | 多框架适配 | 未来考虑 |

**选择模式 A 的理由**：
1. **职责清晰**：各管各的，边界明确
2. **松耦合**：FastDeploy 作为黑盒被调用，不暴露内部进程细节
3. **易于实现**：无需修改现有架构

#### 1.5.3 模式 A：层级委托（推荐方案）

**架构设计**：

```
┌────────────────────────────────────────────────────────────┐
│                   RL InferenceManager                       │
│  职责：                                                     │
│  - 实例级生命周期（启动/停止 FastDeploy 服务）               │
│  - 跨实例负载均衡                                          │
│  - 弹性伸缩决策                                            │
│  - 权重更新编排（调用 DynamicWeightManager）                │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           │ 接口: start()/stop()/health_check()
                           ▼
┌────────────────────────────────────────────────────────────┐
│                   FastDeploy EngineService                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ProcessManager                                       │  │
│  │  职责：                                               │  │
│  │  - 节点内 Worker 进程启动                             │  │
│  │  - 进程健康检查                                       │  │
│  │  - 进程崩溃重启                                       │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**职责划分**：

| 关注点 | InferenceManager (RL 框架) | ProcessManager (FastDeploy) |
|--------|---------------------------|----------------------------|
| 管理粒度 | 推理服务实例 | 节点内 Worker 进程 |
| 启动方式 | K8s Pod / Ray Actor / 裸进程 | `paddle.distributed.launch` |
| 健康检查 | 实例级（API 可达性） | 进程级（pid 存活） |
| 重启策略 | 跨节点迁移、弹性伸缩 | 原地重启 Worker |
| 权重更新 | 决定何时更新、哪些实例更新 | 执行具体的参数加载 |

**RL 框架调用示例**：
```python
# RL InferenceManager 视角：把 FastDeploy 当作黑盒
class InferenceManager:
    def start_inference_instance(self, config):
        """启动一个 FastDeploy 实例（不关心内部有几个进程）"""
        self.fd_service = AsyncLLM(config)  # 或 EngineService
        self.fd_service.start()

    def stop_inference_instance(self):
        """停止整个实例"""
        self.fd_service.stop()  # FastDeploy 内部自己管理进程清理

    def health_check(self):
        """实例级健康检查"""
        return self.fd_service.is_healthy()

    def trigger_weight_update(self):
        """触发权重更新（委托给 DynamicWeightManager）"""
        self.fd_service.update_weights()
```

#### 1.5.4 模式 B：ProcessManager 可选/禁用

如果 RL 框架需要完全接管进程管理，FastDeploy 应支持禁用内置的 ProcessManager：

```python
class EngineService:
    def __init__(
        self,
        cfg: FDConfig,
        start_queue: bool = True,
        managed_externally: bool = False,  # 新增：是否由外部管理进程
    ):
        self.cfg = cfg
        self.managed_externally = managed_externally

        # IPC 和调度器总是需要的
        self.ipc = IPCManager(cfg, start_queue)
        self.scheduler = SchedulerCoordinator(cfg, ...)

        # ProcessManager 根据模式决定
        if not managed_externally:
            self.process = ProcessManager(cfg, self.ipc)
        else:
            self.process = ExternalProcessAdapter(cfg, self.ipc)

    def start(self):
        self.ipc.start()
        self.scheduler.start()
        if not self.managed_externally:
            self.process.start_workers()
        # else: 等待外部启动的 Worker 连接
        self._wait_for_workers_ready()
```

**使用场景**：
```python
# RL 框架直接启动 Worker，FastDeploy 只做「引擎」
inference_manager.start_workers(...)  # RL 框架启动 Worker
engine = EngineService(cfg, managed_externally=True)  # FD 不启动 Worker
engine.connect_to_workers(worker_addresses)  # 只连接到已有 Worker
```

#### 1.5.5 设计原则总结

| 原则 | 说明 |
|------|------|
| **ProcessManager 是必要的** | 作为默认的进程管理实现，适用于独立部署场景 |
| **ProcessManager 应可选** | 支持 `managed_externally` 参数，允许外部接管 |
| **边界清晰** | FastDeploy 对外暴露「服务」接口，内部进程管理是实现细节 |
| **黑盒原则** | RL 框架只需和 FastDeploy 的服务层接口交互，不需要关心内部进程数量 |

#### 1.5.6 未来扩展：抽象进程管理接口

如需支持多种进程管理后端（K8s、Ray、Slurm 等），可定义抽象接口：

```python
class IProcessLifecycle(ABC):
    """进程生命周期管理接口"""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_healthy(self) -> bool: ...

    @abstractmethod
    def restart(self) -> None: ...

# FastDeploy 默认实现
class ProcessManager(IProcessLifecycle):
    def start(self):
        # 使用 paddle.distributed.launch 启动 Worker
        ...

# RL 框架自定义实现（如需要）
class RayProcessManager(IProcessLifecycle):
    def start(self):
        # 通过 Ray 启动 Worker
        ...
```

**注入方式**：
```python
class EngineService:
    def __init__(self, cfg, process_manager: IProcessLifecycle = None):
        self.process = process_manager or ProcessManager(cfg, self.ipc)
```

---

### 1.6 分阶段执行计划

本节详细描述重构的分阶段执行步骤，方便按计划逐步实现。

---

#### 阶段 0：准备工作（预计 1-2 天）

**目标**：建立基线，确保重构过程可验证。

| 步骤 | 任务 | 产出物 | 验证方式 |
|------|------|--------|---------|
| 0.1 | 运行现有测试套件，记录通过率 | `baseline_test_results.json` | 保存测试报告 |
| 0.2 | 部署基线捕获器（5.5节）到关键路径 | 捕获代码已注入 | 运行样例请求验证数据捕获 |
| 0.3 | 运行典型工作负载，捕获基线数据 | `baseline_data/*.json` | 检查各捕获点数据完整性 |
| 0.4 | 创建 `fastdeploy/engine/components/` 目录 | 目录结构 | `ls` 确认 |
| 0.5 | 创建各组件的空文件骨架 | 4个空 `.py` 文件 | 语法检查通过 |

**具体命令**：
```bash
# 步骤 0.1：运行现有测试
pytest tests/ --json-report --json-report-file=baseline_test_results.json

# 步骤 0.3：捕获基线数据
FD_CAPTURE_BASELINE=1 python examples/run_inference.py --model <model_path>

# 步骤 0.4-0.5：创建目录和文件
mkdir -p fastdeploy/engine/components
touch fastdeploy/engine/components/__init__.py
touch fastdeploy/engine/components/ipc_manager.py
touch fastdeploy/engine/components/process_manager.py
touch fastdeploy/engine/components/scheduler_coordinator.py
touch fastdeploy/engine/components/resource_coordinator.py
```

**完成标准**：
- [x] 基线测试报告已保存
- [x] 基线数据已捕获（至少 100 个请求）
- [x] 组件目录结构已创建

---

#### 阶段 1：IPCManager 抽取（预计 2-3 天）

**目标**：将 IPC 相关代码抽取到独立组件，不改变外部行为。

**依赖**：阶段 0 完成

| 步骤 | 任务 | 涉及代码 | 验证方式 |
|------|------|---------|---------|
| 1.1 | 分析 `common_engine.py` 中所有 IPC 相关代码 | 见下方代码清单 | 代码审查 |
| 1.2 | 实现 `IPCManager.__init__()` | 初始化 Queue/Signal | 单元测试 |
| 1.3 | 迁移 `start_worker_queue_service()` | Queue Server 启动逻辑 | 集成测试 |
| 1.4 | 迁移 `_init_ipc_signal()` | Signal 初始化 | 单元测试 |
| 1.5 | 迁移 `start_zmq_service()` | ZMQ 通信启动 | 集成测试 |
| 1.6 | 在 `EngineService` 中使用 `IPCManager` | 替换原有代码 | 端到端测试 |
| 1.7 | 运行完整测试套件 | - | 与基线对比 |

**步骤 1.1 代码清单**（需要迁移的方法/属性）：
```python
# 需要迁移到 IPCManager 的代码
from common_engine.py:
    - _init_ipc_signal()              # 行 ~450-550
    - start_worker_queue_service()    # 行 ~600-700
    - start_zmq_service()             # 行 ~750-850
    - _setup_zmq_sockets()            # 行 ~860-920
    - engine_worker_queue             # 属性
    - exist_task_signal               # 属性
    - worker_healthy_live_signal      # 属性
    - loaded_model_signal             # 属性
    - 其他 *_signal 属性
```

**步骤 1.2 实现模板**：
```python
# fastdeploy/engine/components/ipc_manager.py

class IPCManager:
    """IPC 通信管理器"""

    def __init__(self, cfg: FDConfig, start_queue: bool = True):
        self.cfg = cfg
        self.start_queue = start_queue

        # 延迟初始化
        self._engine_worker_queue: Optional[EngineWorkerQueue] = None
        self._signals: Dict[str, IPCSignal] = {}
        self._zmq_context: Optional[zmq.Context] = None

    def init_signals(self):
        """初始化所有 IPC Signal"""
        # 从 _init_ipc_signal() 迁移
        pass

    def start_queue_service(self):
        """启动 Queue Server"""
        # 从 start_worker_queue_service() 迁移
        pass

    def start_zmq(self, api_server_pid: int):
        """启动 ZMQ 服务"""
        # 从 start_zmq_service() 迁移
        pass

    # ========== 兼容属性 ==========
    @property
    def engine_worker_queue(self):
        return self._engine_worker_queue

    @property
    def exist_task_signal(self):
        return self._signals.get("exist_task")

    @property
    def worker_healthy_live_signal(self):
        return self._signals.get("worker_healthy_live")
```

**步骤 1.6 替换示例**：
```python
# EngineService 中的变更
class EngineService:
    def __init__(self, cfg, start_queue=True):
        # 旧代码（删除）：
        # self.engine_worker_queue = None
        # self.exist_task_signal = None
        # ...

        # 新代码：
        self._ipc = IPCManager(cfg, start_queue)

    # 兼容属性
    @property
    def engine_worker_queue(self):
        return self._ipc.engine_worker_queue

    @property
    def exist_task_signal(self):
        return self._ipc.exist_task_signal
```

**完成标准**：
- [x] `IPCManager` 类实现完成，包含所有迁移方法
- [x] `EngineService` 通过 `@property` 保持兼容
- [x] 单元测试覆盖 `IPCManager` 核心方法
- [x] 集成测试通过（同步模式 + 异步模式）
- [x] 基线对比无差异

---

#### 阶段 2：ProcessManager 抽取（预计 2-3 天）

**目标**：统一 Worker 进程管理，消除 `use_async_llm` 分支。

**依赖**：阶段 1 完成

| 步骤 | 任务 | 涉及代码 | 验证方式 |
|------|------|---------|---------|
| 2.1 | 分析 Worker 启动相关代码 | 见下方代码清单 | 代码审查 |
| 2.2 | 实现 `ProcessManager.__init__()` | 进程配置初始化 | 单元测试 |
| 2.3 | 迁移 `_start_worker_service()` | Worker 启动逻辑 | 集成测试 |
| 2.4 | 迁移 `start_cache_service()` | 缓存服务启动 | 集成测试 |
| 2.5 | 实现 `stop_workers()` | 进程清理 | 单元测试 |
| 2.6 | 实现心跳检测 `check_worker_health()` | 健康检查 | 单元测试 |
| 2.7 | 在 `EngineService` 中使用 `ProcessManager` | 替换原有代码 | 端到端测试 |
| 2.8 | 修改 `LLMEngine` 移除 Worker 管理代码 | 见下方说明 | 集成测试 |
| 2.9 | 删除 `use_async_llm` 参数 | 清理代码 | 全量测试 |

**步骤 2.1 代码清单**：
```python
# 需要迁移到 ProcessManager 的代码
from common_engine.py:
    - _start_worker_service()         # 行 ~1000-1100
    - start_cache_service()           # 行 ~1150-1250
    - _check_worker_status()          # 行 ~1300-1350
    - worker_proc                     # 属性（进程句柄）
    - cache_procs                     # 属性（缓存进程句柄）

from llm_engine.py:
    - _start_worker_service()         # 重复实现，需删除
```

**步骤 2.2 实现模板**：
```python
# fastdeploy/engine/components/process_manager.py

class ProcessManager:
    """Worker 进程管理器"""

    def __init__(self, cfg: FDConfig, ipc_manager: IPCManager):
        self.cfg = cfg
        self.ipc = ipc_manager

        self._worker_procs: List[Process] = []
        self._cache_procs: List[Process] = []
        self._running = False

    def start_workers(self):
        """启动所有 Worker 进程"""
        # 从 _start_worker_service() 迁移
        # 统一处理，不再区分 use_async_llm
        pass

    def start_cache_service(self, device_ids: List[str], suffix: int) -> List[Process]:
        """启动缓存服务进程"""
        # 从 start_cache_service() 迁移
        pass

    def stop_workers(self):
        """停止所有 Worker 进程"""
        pass

    def check_worker_health(self) -> bool:
        """检查 Worker 健康状态"""
        # 从 _check_worker_status() 迁移
        pass

    # ========== 兼容属性 ==========
    @property
    def worker_proc(self):
        """兼容：返回第一个 Worker 进程"""
        return self._worker_procs[0] if self._worker_procs else None
```

**步骤 2.8 LLMEngine 修改**：
```python
# 修改前：LLMEngine 自己管理 Worker
class LLMEngine:
    def start(self):
        self.engine.start()
        self._start_worker_service()  # 删除这行

# 修改后：委托给 EngineService
class LLMEngine:
    def start(self):
        self.engine.start()  # EngineService 内部会启动 Worker
        # 不再调用 _start_worker_service()
```

**完成标准**：
- [x] `ProcessManager` 类实现完成
- [x] `LLMEngine._start_worker_service()` 已删除
- [x] `use_async_llm` 参数已删除
- [x] 同步模式（LLMEngine）和异步模式（AsyncLLM）都正常工作
- [x] 基线对比无差异

---

#### 阶段 3：ResourceCoordinator 抽取（预计 1-2 天）

**目标**：抽取资源管理相关代码。

**依赖**：阶段 2 完成

| 步骤 | 任务 | 涉及代码 | 验证方式 |
|------|------|---------|---------|
| 3.1 | 分析资源管理相关代码 | 见下方代码清单 | 代码审查 |
| 3.2 | 实现 `ResourceCoordinator.__init__()` | 初始化 ResourceManager | 单元测试 |
| 3.3 | 迁移资源查询方法 | `available_block_num()` 等 | 单元测试 |
| 3.4 | 迁移资源释放方法 | `check_and_free_block_tables()` 等 | 单元测试 |
| 3.5 | 在 `EngineService` 中使用 `ResourceCoordinator` | 替换原有代码 | 端到端测试 |

**步骤 3.1 代码清单**：
```python
# 需要迁移到 ResourceCoordinator 的代码
from common_engine.py:
    - _init_resource_manager()        # 行 ~350-400
    - resource_manager                # 属性
    - cache_config                    # 属性（部分）
```

**步骤 3.2 实现模板**：
```python
# fastdeploy/engine/components/resource_coordinator.py

class ResourceCoordinator:
    """资源协调器"""

    def __init__(self, cfg: FDConfig):
        self.cfg = cfg
        self._resource_manager: Optional[ResourceManager] = None

    def init_resource_manager(self):
        """初始化 ResourceManager"""
        pass

    @property
    def resource_manager(self):
        return self._resource_manager

    def available_block_num(self) -> int:
        """查询可用 KV Cache 块数"""
        return self._resource_manager.available_block_num()
```

**完成标准**：
- [x] `ResourceCoordinator` 类实现完成
- [x] `EngineService` 通过 `@property` 保持兼容
- [x] 资源管理相关测试通过
- [x] 基线对比无差异

---

#### 阶段 4：SchedulerCoordinator 抽取（预计 2-3 天）

**目标**：抽取调度相关代码。

**依赖**：阶段 3 完成

| 步骤 | 任务 | 涉及代码 | 验证方式 |
|------|------|---------|---------|
| 4.1 | 分析调度相关代码 | 见下方代码清单 | 代码审查 |
| 4.2 | 实现 `SchedulerCoordinator.__init__()` | 初始化 Scheduler | 单元测试 |
| 4.3 | 迁移调度器启动逻辑 | `scheduler.start()` 封装 | 单元测试 |
| 4.4 | 迁移请求分发逻辑 | `_dispatch_tasks()` | 集成测试 |
| 4.5 | 迁移结果收集逻辑 | `_collect_results()` | 集成测试 |
| 4.6 | 在 `EngineService` 中使用 `SchedulerCoordinator` | 替换原有代码 | 端到端测试 |

**步骤 4.1 代码清单**：
```python
# 需要迁移到 SchedulerCoordinator 的代码
from common_engine.py:
    - _init_scheduler()               # 行 ~250-300
    - _start_schedule_loop()          # 行 ~1400-1600
    - _dispatch_tasks()               # 行 ~1650-1750
    - _collect_results()              # 行 ~1800-1900
    - scheduler                       # 属性
    - data_processor                  # 属性
```

**步骤 4.2 实现模板**：
```python
# fastdeploy/engine/components/scheduler_coordinator.py

class SchedulerCoordinator:
    """调度协调器"""

    def __init__(
        self,
        cfg: FDConfig,
        resource_coordinator: ResourceCoordinator,
        ipc_manager: IPCManager
    ):
        self.cfg = cfg
        self.resource = resource_coordinator
        self.ipc = ipc_manager

        self._scheduler: Optional[BaseScheduler] = None
        self._data_processor: Optional[InputPreprocessor] = None

    def init_scheduler(self):
        """初始化调度器"""
        pass

    def start(self):
        """启动调度循环"""
        pass

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def data_processor(self):
        return self._data_processor
```

**完成标准**：
- [x] `SchedulerCoordinator` 类实现完成
- [x] 调度循环正常运行
- [x] 请求处理流程完整
- [x] 基线对比无差异

---

#### 阶段 5：集成与清理（预计 2-3 天）

**目标**：集成所有组件，清理遗留代码，完成重构。

**依赖**：阶段 4 完成

| 步骤 | 任务 | 验证方式 |
|------|------|---------|
| 5.1 | 完成 `EngineService` 门面类整合 | 代码审查 |
| 5.2 | 删除 `common_engine.py` 中已迁移的代码 | 语法检查 |
| 5.3 | 更新 `__init__.py` 导出 | import 测试 |
| 5.4 | 运行完整测试套件 | 与基线对比 |
| 5.5 | 运行性能基准测试 | 性能无退化 |
| 5.6 | 代码审查和文档更新 | Review 通过 |
| 5.7 | 删除基线捕获代码（可选） | - |

**步骤 5.1 最终 EngineService 结构**：
```python
# fastdeploy/engine/engine_service.py（最终版本）

class EngineService:
    """引擎服务门面 - 重构后"""

    def __init__(self, cfg: FDConfig, start_queue: bool = True):
        self.cfg = cfg
        self.running = False

        # 内部组件
        self._ipc = IPCManager(cfg, start_queue)
        self._resource = ResourceCoordinator(cfg)
        self._scheduler_coord = SchedulerCoordinator(cfg, self._resource, self._ipc)
        self._process = ProcessManager(cfg, self._ipc)

    def start(self):
        """启动引擎"""
        self.running = True
        self._ipc.start()
        self._resource.start()
        self._scheduler_coord.start()
        self._process.start_workers()

    def stop(self):
        """停止引擎"""
        self.running = False
        self._process.stop_workers()
        self._scheduler_coord.stop()
        self._ipc.stop()

    # ========== 兼容属性（不修改外部调用） ==========

    @property
    def scheduler(self):
        return self._scheduler_coord.scheduler

    @property
    def resource_manager(self):
        return self._resource.resource_manager

    @property
    def data_processor(self):
        return self._scheduler_coord.data_processor

    @property
    def engine_worker_queue(self):
        return self._ipc.engine_worker_queue

    @property
    def worker_healthy_live_signal(self):
        return self._ipc.worker_healthy_live_signal

    # ... 其他兼容属性
```

**步骤 5.4 测试验证清单**：

| 测试类型 | 命令 | 通过标准 |
|---------|------|---------|
| 单元测试 | `pytest tests/unit/` | 100% 通过 |
| 集成测试 | `pytest tests/integration/` | 100% 通过 |
| 端到端测试 | `pytest tests/e2e/` | 100% 通过 |
| 基线对比 | `python scripts/verify_baseline.py` | 无差异 |
| 性能测试 | `python benchmarks/run_benchmark.py` | 吞吐量 ≥ 基线 95% |

**完成标准**：
- [x] `EngineService` 代码量 < 300 行
- [x] 各组件代码量 < 500 行
- [x] 所有测试通过
- [x] 基线对比无差异
- [x] 性能无退化（吞吐量在基线 95% 以上）
- [x] 代码审查通过

---

#### 阶段 6：后续优化（未来）

**目标**：接口优化和进一步解耦。

| 步骤 | 任务 | 优先级 |
|------|------|-------|
| 6.1 | 标记兼容属性为 `@deprecated` | 低 |
| 6.2 | 设计并实现新的公开 API | 低 |
| 6.3 | 迁移调用方到新 API | 低 |
| 6.4 | 移除兼容属性 | 低 |
| 6.5 | 配置化 sleep 间隔 | 中 |
| 6.6 | 添加更多单元测试 | 中 |

---

#### 执行计划总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        执行计划时间线                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  阶段 0          阶段 1          阶段 2          阶段 3           │
│  准备工作        IPCManager      ProcessManager  ResourceCoord   │
│  [██████]  →    [██████████]  → [██████████]  → [██████]         │
│  1-2 天          2-3 天          2-3 天          1-2 天           │
│                                                                 │
│                  阶段 4          阶段 5                           │
│                  SchedulerCoord  集成清理                         │
│             →   [██████████]  → [██████████]                    │
│                  2-3 天          2-3 天                          │
│                                                                 │
│  总计：约 11-16 天                                                │
└─────────────────────────────────────────────────────────────────┘
```

**风险控制**：
- 每个阶段完成后都进行基线对比验证
- 保持 `common_engine.py` 可回滚（通过 git）
- 阶段间可以暂停，不影响线上服务

---

## 二、架构设计

### 2.1 重构前后对比

```
重构前                                重构后
┌─────────────────────────┐          ┌─────────────────────────┐
│     EngineService       │          │     EngineService       │
│       2210 行            │          │       ~200 行           │
│                         │          │   (轻量协调层)           │
│  • 调度逻辑              │          └───────────┬─────────────┘
│  • 进程管理              │                      │
│  • IPC 通信              │     ┌────────┬───────┴───────┬────────┐
│  • 资源协调              │     ▼        ▼               ▼        ▼
│  • P/D 分离              │ ┌───────┐ ┌───────┐    ┌───────┐ ┌───────┐
│  • 监控信号              │ │Scheduler│ │Process│    │  IPC  │ │Resource│
│  • Token 处理            │ │  Coord  │ │Manager│    │Manager│ │ Coord │
│  • ...                  │ │ ~400行  │ │ ~300行│    │ ~300行│ │ ~400行│
└─────────────────────────┘ └───────┘ └───────┘    └───────┘ └───────┘
```

### 2.2 组件职责划分

| 组件 | 职责 | 预估行数 |
|------|------|---------|
| `EngineService` | 轻量门面，协调各组件 | ~200 |
| `SchedulerCoordinator` | 调度逻辑、任务分发 | ~400 |
| `ProcessManager` | Worker 进程生命周期管理 | ~300 |
| `IPCManager` | ZMQ/共享内存通信 | ~300 |
| `ResourceCoordinator` | 资源分配协调 | ~400 |

---

## 三、详细设计

### 3.1 EngineService（门面层）

```python
# fastdeploy/engine/engine_service.py
"""
EngineService - 轻量级门面类

只负责组件协调，不包含具体业务逻辑
"""

from fastdeploy.engine.components.scheduler_coordinator import SchedulerCoordinator
from fastdeploy.engine.components.process_manager import ProcessManager
from fastdeploy.engine.components.ipc_manager import IPCManager
from fastdeploy.engine.components.resource_coordinator import ResourceCoordinator


class EngineService:
    """
    引擎服务门面

    协调调度、进程、通信、资源等组件
    """

    def __init__(self, cfg: FDConfig, start_queue: bool = True, use_async_llm: bool = False):
        self.cfg = cfg
        self.running = False

        # 初始化各组件
        self.ipc = IPCManager(cfg, start_queue)
        self.resource = ResourceCoordinator(cfg)
        self.scheduler = SchedulerCoordinator(
            cfg=cfg,
            resource_coordinator=self.resource,
            ipc_manager=self.ipc,
        )
        self.process = ProcessManager(cfg, self.ipc)

    def start(self):
        """启动引擎服务"""
        self.running = True

        # 按顺序启动各组件
        self.ipc.start()
        self.resource.start()
        self.scheduler.start()
        self.process.start_workers()

    def stop(self):
        """停止引擎服务"""
        self.running = False

        # 按顺序停止各组件
        self.process.stop_workers()
        self.scheduler.stop()
        self.resource.stop()
        self.ipc.stop()

    def add_requests(self, task, sampling_params=None):
        """添加推理请求"""
        return self.scheduler.add_requests(task, sampling_params)

    def _exit_sub_services(self):
        """退出所有子服务"""
        self.stop()
```

### 3.2 SchedulerCoordinator（调度协调器）

```python
# fastdeploy/engine/components/scheduler_coordinator.py
"""
SchedulerCoordinator - 调度协调器

负责请求调度、任务分发
"""

import threading
from typing import List, Optional
from fastdeploy.engine.request import Request
from fastdeploy.config import FDConfig


class SchedulerCoordinator:
    """
    调度协调器

    职责：
    - 接收和验证请求
    - 调度请求到 Worker
    - 处理调度结果
    """

    def __init__(
        self,
        cfg: FDConfig,
        resource_coordinator: "ResourceCoordinator",
        ipc_manager: "IPCManager",
    ):
        self.cfg = cfg
        self.resource = resource_coordinator
        self.ipc = ipc_manager

        # 调度器实例
        self.scheduler = cfg.scheduler_config.scheduler()

        # 调度线程
        self._schedule_thread: Optional[threading.Thread] = None
        self._running = False

        # 配置参数（替代硬编码）
        self.poll_interval = SchedulerConfig.POLL_INTERVAL_MS / 1000
        self.wait_timeout = SchedulerConfig.WAIT_TIMEOUT_MS / 1000

    def start(self):
        """启动调度"""
        self._running = True
        self._schedule_thread = threading.Thread(
            target=self._schedule_loop,
            daemon=True,
            name="scheduler-coordinator"
        )
        self._schedule_thread.start()

    def stop(self):
        """停止调度"""
        self._running = False
        if self._schedule_thread:
            self._schedule_thread.join(timeout=5.0)

    def add_requests(self, task, sampling_params=None) -> List:
        """添加请求到调度队列"""
        # 1. 验证和预处理请求
        request = self._prepare_request(task, sampling_params)

        # 2. 放入调度队列
        self.scheduler.put_requests([request])

        return [request.request_id]

    def _schedule_loop(self):
        """调度主循环"""
        while self._running:
            # 获取待调度任务
            tasks = self.scheduler.get_requests(
                timeout=self.wait_timeout
            )

            if not tasks:
                continue

            # 分配资源
            allocated_tasks = self.resource.allocate_resources(tasks)

            # 发送到 Worker
            if allocated_tasks:
                self.ipc.send_tasks(allocated_tasks)

    def _prepare_request(self, task, sampling_params) -> Request:
        """准备请求"""
        if isinstance(task, dict):
            return Request.from_dict(task)
        return task


class SchedulerConfig:
    """调度器配置常量"""
    POLL_INTERVAL_MS = 1      # 轮询间隔（毫秒）
    WAIT_TIMEOUT_MS = 10      # 等待超时（毫秒）
    MAX_BATCH_SIZE = 64       # 最大批处理大小
```

### 3.3 ProcessManager（进程管理器）

```python
# fastdeploy/engine/components/process_manager.py
"""
ProcessManager - 进程管理器

负责 Worker 进程的生命周期管理
"""

import subprocess
import threading
from typing import Optional, List
from fastdeploy.config import FDConfig


class ProcessManager:
    """
    进程管理器

    职责：
    - 启动 Worker 进程
    - 监控进程健康状态
    - 处理进程异常和重启
    """

    def __init__(self, cfg: FDConfig, ipc_manager: "IPCManager"):
        self.cfg = cfg
        self.ipc = ipc_manager

        self.worker_procs: List[subprocess.Popen] = []
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False

        # 配置
        self.health_check_interval = ProcessConfig.HEALTH_CHECK_INTERVAL_S
        self.startup_timeout = ProcessConfig.STARTUP_TIMEOUT_S

    def start_workers(self):
        """启动所有 Worker 进程"""
        self._running = True

        # 构建启动命令
        cmd = self._build_worker_command()

        # 启动进程
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.worker_procs.append(proc)

        # 等待 Worker 就绪
        self._wait_for_workers_ready()

        # 启动健康检查
        self._start_health_check()

    def stop_workers(self):
        """停止所有 Worker 进程"""
        self._running = False

        for proc in self.worker_procs:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        self.worker_procs.clear()

    def _build_worker_command(self) -> str:
        """构建 Worker 启动命令"""
        # 使用 paddle.distributed.launch
        cmd = (
            f"python -m paddle.distributed.launch "
            f"--gpus {self.cfg.parallel_config.gpu_ids} "
            f"fastdeploy.worker.worker_process "
            f"--config {self.cfg.to_json()}"
        )
        return cmd

    def _wait_for_workers_ready(self):
        """等待 Worker 就绪"""
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            if self.ipc.workers_ready():
                return
            time.sleep(0.1)
        raise TimeoutError("Workers failed to start within timeout")

    def _start_health_check(self):
        """启动健康检查线程"""
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="worker-health-check"
        )
        self._health_check_thread.start()

    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            for proc in self.worker_procs:
                if proc.poll() is not None:
                    # 进程已退出
                    self._handle_worker_crash(proc)
            time.sleep(self.health_check_interval)

    def _handle_worker_crash(self, proc):
        """处理 Worker 崩溃"""
        # 记录日志、发送告警、尝试重启等
        pass


class ProcessConfig:
    """进程管理配置常量"""
    HEALTH_CHECK_INTERVAL_S = 5.0   # 健康检查间隔（秒）
    STARTUP_TIMEOUT_S = 300.0       # 启动超时（秒）
    RESTART_DELAY_S = 1.0           # 重启延迟（秒）
```

### 3.4 IPCManager（IPC 管理器）

```python
# fastdeploy/engine/components/ipc_manager.py
"""
IPCManager - 进程间通信管理器

负责 ZMQ、共享内存等通信机制
"""

import threading
from typing import List, Optional
from fastdeploy.config import FDConfig
from fastdeploy.inter_communicator import EngineWorkerQueue, EngineCacheQueue


class IPCManager:
    """
    IPC 管理器

    职责：
    - 管理通信队列
    - 发送任务到 Worker
    - 接收 Worker 结果
    - 信号同步
    """

    def __init__(self, cfg: FDConfig, start_queue: bool = True):
        self.cfg = cfg

        # 通信队列
        self.engine_worker_queue: Optional[EngineWorkerQueue] = None
        self.cache_queue: Optional[EngineCacheQueue] = None

        # 信号
        self.worker_ready_signal = None
        self.loaded_model_signal = None

        # 接收线程
        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

        if start_queue:
            self._init_queues()

    def _init_queues(self):
        """初始化通信队列"""
        self.engine_worker_queue = EngineWorkerQueue(
            address=f"tcp://*:{self.cfg.engine_worker_queue_port}"
        )
        self.cache_queue = EngineCacheQueue(
            address=f"tcp://*:{self.cfg.cache_queue_port}"
        )

    def start(self):
        """启动 IPC 服务"""
        self._running = True
        self._start_recv_thread()

    def stop(self):
        """停止 IPC 服务"""
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=5.0)

    def send_tasks(self, tasks: List) -> bool:
        """发送任务到 Worker"""
        if not self.engine_worker_queue:
            return False
        return self.engine_worker_queue.put_tasks(tasks)

    def recv_results(self, timeout: float = 0.001) -> List:
        """接收 Worker 结果"""
        if not self.engine_worker_queue:
            return []
        return self.engine_worker_queue.get_results(timeout=timeout)

    def workers_ready(self) -> bool:
        """检查 Worker 是否就绪"""
        if self.worker_ready_signal is None:
            return False
        return self.worker_ready_signal.value[0] == 1

    def _start_recv_thread(self):
        """启动接收线程"""
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name="ipc-recv"
        )
        self._recv_thread.start()

    def _recv_loop(self):
        """接收循环"""
        while self._running:
            results = self.recv_results(timeout=IPCConfig.RECV_TIMEOUT_MS / 1000)
            if results:
                self._handle_results(results)

    def _handle_results(self, results: List):
        """处理接收到的结果"""
        # 分发给相应的处理器
        pass


class IPCConfig:
    """IPC 配置常量"""
    RECV_TIMEOUT_MS = 1       # 接收超时（毫秒）
    SEND_TIMEOUT_MS = 1000    # 发送超时（毫秒）
    QUEUE_SIZE = 1000         # 队列大小
```

### 3.5 ResourceCoordinator（资源协调器）

```python
# fastdeploy/engine/components/resource_coordinator.py
"""
ResourceCoordinator - 资源协调器

负责资源分配和回收的协调
"""

from typing import List, Optional
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request


class ResourceCoordinator:
    """
    资源协调器

    职责：
    - 协调资源分配
    - 管理资源回收
    - 监控资源使用
    """

    def __init__(self, cfg: FDConfig):
        self.cfg = cfg

        # 资源管理器（V0 或 V1）
        self.resource_manager = self._create_resource_manager()

        self._running = False

    def _create_resource_manager(self):
        """创建资源管理器"""
        from fastdeploy import envs
        if envs.SCHEDULER_VERSION.get() == "v1":
            from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1
            return ResourceManagerV1(self.cfg)
        else:
            from fastdeploy.engine.resource_manager import ResourceManager
            return ResourceManager(self.cfg)

    def start(self):
        """启动资源协调"""
        self._running = True

    def stop(self):
        """停止资源协调"""
        self._running = False

    def allocate_resources(self, tasks: List[Request]) -> List[Request]:
        """
        为任务分配资源

        Args:
            tasks: 待分配资源的任务列表

        Returns:
            成功分配资源的任务列表
        """
        return self.resource_manager.allocate_resources_for_new_tasks(tasks)

    def release_resources(self, request_id: str):
        """释放请求占用的资源"""
        self.resource_manager.release_resources(request_id)

    def get_available_slots(self) -> int:
        """获取可用槽位数"""
        return self.resource_manager.available_batch()

    def get_resource_usage(self) -> dict:
        """获取资源使用情况"""
        return {
            "available_slots": self.get_available_slots(),
            "total_slots": self.resource_manager.max_num_seqs,
            "gpu_memory_used": self.resource_manager.get_gpu_memory_used(),
        }
```

---

## 四、sleep 参数化

### 4.1 当前问题

```python
# 代码中散布的硬编码 sleep
time.sleep(0.001)   # 27 处
time.sleep(0.005)   # 多处
time.sleep(0.01)    # 多处
time.sleep(0.1)     # 多处
```

### 4.2 解决方案：配置常量

```python
# fastdeploy/engine/constants.py
"""
引擎常量配置

所有 sleep 间隔和超时配置集中管理
"""


class SchedulerTiming:
    """调度器时序配置"""
    POLL_INTERVAL_MS = 1          # 资源轮询间隔
    WAIT_TIMEOUT_MS = 10          # 请求等待超时
    BATCH_INTERVAL_MS = 5         # 批处理间隔


class IPCTiming:
    """IPC 时序配置"""
    RECV_TIMEOUT_MS = 1           # 接收超时
    SEND_TIMEOUT_MS = 1000        # 发送超时
    HEARTBEAT_INTERVAL_MS = 100   # 心跳间隔


class ProcessTiming:
    """进程管理时序配置"""
    STARTUP_TIMEOUT_S = 300       # 启动超时
    HEALTH_CHECK_INTERVAL_S = 5   # 健康检查间隔
    SHUTDOWN_TIMEOUT_S = 10       # 关闭超时


# 使用方式
# time.sleep(SchedulerTiming.POLL_INTERVAL_MS / 1000)
```

### 4.3 进阶方案：自适应退避

```python
# fastdeploy/engine/utils/backoff.py
"""
自适应退避策略
"""

import time


class AdaptiveBackoff:
    """
    自适应退避

    根据负载情况动态调整等待间隔
    """

    def __init__(
        self,
        min_ms: float = 1,
        max_ms: float = 100,
        factor: float = 1.5,
    ):
        self.min_delay = min_ms / 1000
        self.max_delay = max_ms / 1000
        self.factor = factor
        self.current_delay = self.min_delay

    def wait(self):
        """等待并增加延迟"""
        time.sleep(self.current_delay)
        self.current_delay = min(
            self.current_delay * self.factor,
            self.max_delay
        )

    def reset(self):
        """重置延迟"""
        self.current_delay = self.min_delay

    def success(self):
        """成功时减少延迟"""
        self.current_delay = max(
            self.current_delay / self.factor,
            self.min_delay
        )


# 使用示例
backoff = AdaptiveBackoff(min_ms=1, max_ms=100)

while running:
    tasks = scheduler.get_requests()
    if tasks:
        backoff.reset()  # 有任务时重置
        process(tasks)
    else:
        backoff.wait()   # 无任务时等待（自适应）
```

### 4.4 进阶方案：条件变量

```python
# fastdeploy/engine/utils/sync.py
"""
同步原语
"""

import threading
from typing import Callable, Optional


class ResourceCondition:
    """
    资源条件变量

    用于替代轮询，实现高效等待
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._resource_available = False

    def wait_for_resource(self, timeout: Optional[float] = None) -> bool:
        """
        等待资源可用

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            True 如果资源可用，False 如果超时
        """
        with self._condition:
            if self._resource_available:
                self._resource_available = False
                return True
            return self._condition.wait(timeout=timeout)

    def notify_resource_available(self):
        """通知资源可用"""
        with self._condition:
            self._resource_available = True
            self._condition.notify_all()


# 使用示例
resource_cond = ResourceCondition()

# 等待方
while running:
    if resource_cond.wait_for_resource(timeout=0.1):
        tasks = scheduler.get_requests()
        process(tasks)

# 通知方
def on_resource_freed():
    resource_cond.notify_resource_available()
```

---

## 五、测试策略

### 5.1 测试分层

```
┌─────────────────────────────────────────────────────────────┐
│                    E2E Tests (端到端测试)                     │
│         验证完整的请求处理流程，从输入到输出                      │
├─────────────────────────────────────────────────────────────┤
│                Integration Tests (集成测试)                   │
│         验证拆分后的组件之间能正确协作                           │
├─────────────────────────────────────────────────────────────┤
│                  Unit Tests (单元测试)                        │
│      SchedulerCoordinator / ProcessManager / IPCManager      │
│                  每个新组件的独立测试                           │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 单元测试示例

```python
# tests/engine/components/test_scheduler_coordinator.py

import unittest
from unittest.mock import MagicMock, Mock

from fastdeploy.engine.components.scheduler_coordinator import SchedulerCoordinator


class TestSchedulerCoordinator(unittest.TestCase):

    def setUp(self):
        # Mock 依赖
        self.mock_resource = MagicMock()
        self.mock_ipc = MagicMock()
        self.mock_cfg = MagicMock()

        self.coordinator = SchedulerCoordinator(
            cfg=self.mock_cfg,
            resource_coordinator=self.mock_resource,
            ipc_manager=self.mock_ipc,
        )

    def test_add_requests_puts_to_scheduler(self):
        """验证添加请求会放入调度器"""
        task = {"request_id": "test_001", "prompt": "Hello"}

        self.coordinator.add_requests(task)

        # 验证调用了 scheduler.put_requests
        self.coordinator.scheduler.put_requests.assert_called_once()

    def test_schedule_loop_sends_tasks_to_worker(self):
        """验证调度循环会发送任务到 Worker"""
        mock_tasks = [Mock(), Mock()]
        self.coordinator.scheduler.get_requests.return_value = mock_tasks
        self.mock_resource.allocate_resources.return_value = mock_tasks

        # 执行一次调度
        self.coordinator._schedule_loop_once()

        # 验证发送了任务
        self.mock_ipc.send_tasks.assert_called_with(mock_tasks)


# tests/engine/components/test_process_manager.py

class TestProcessManager(unittest.TestCase):

    def test_start_workers_creates_process(self):
        """验证启动 Worker 会创建进程"""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.poll.return_value = None

            self.manager.start_workers()

            mock_popen.assert_called_once()

    def test_stop_workers_terminates_process(self):
        """验证停止 Worker 会终止进程"""
        mock_proc = MagicMock()
        self.manager.worker_procs = [mock_proc]

        self.manager.stop_workers()

        mock_proc.terminate.assert_called_once()
```

### 5.3 行为等价性测试

```python
# tests/engine/test_behavior_equivalence.py
"""
行为等价性测试：确保重构前后行为一致
"""

class TestBehaviorEquivalence(unittest.TestCase):
    """
    同时运行新旧两个实现，对比结果
    """

    @classmethod
    def setUpClass(cls):
        # 旧实现
        cls.old_engine = OldEngineService(cfg)

        # 新实现
        cls.new_engine = NewEngineService(cfg)

    def test_add_request_same_result(self):
        """验证 add_request 结果一致"""
        request = create_test_request()

        old_result = self.old_engine.add_requests(request)
        new_result = self.new_engine.add_requests(request)

        self.assertEqual(old_result, new_result)

    def test_scheduling_same_order(self):
        """验证调度顺序一致"""
        requests = [create_test_request() for _ in range(10)]

        old_order = self.old_engine.get_scheduling_order(requests)
        new_order = self.new_engine.get_scheduling_order(requests)

        self.assertEqual(old_order, new_order)
```

### 5.4 黄金测试（Golden Test）

```python
# tests/engine/test_golden.py
"""
黄金测试：记录重构前的行为作为基线
"""

import json


class GoldenTest:
    """
    在重构前运行，记录系统行为
    在重构后运行，对比行为是否一致
    """

    GOLDEN_FILE = "tests/engine/golden_baseline.json"

    @classmethod
    def capture_baseline(cls, engine):
        """捕获基线"""
        baseline = {
            "state": {
                "running": engine.running,
                "available_slots": engine.resource_manager.available_batch(),
            },
            "request_flow": cls._capture_request_flow(engine),
        }

        with open(cls.GOLDEN_FILE, "w") as f:
            json.dump(baseline, f, indent=2)

    @classmethod
    def verify_against_baseline(cls, engine):
        """验证是否符合基线"""
        with open(cls.GOLDEN_FILE) as f:
            baseline = json.load(f)

        current = {
            "state": {
                "running": engine.running,
                "available_slots": engine.resource_manager.available_batch(),
            },
        }

        assert current["state"] == baseline["state"]
```

### 5.5 基线捕获详细方案

#### 5.5.1 捕获点位置

在代码中的以下 **5 个关键位置** 添加捕获点：

| 捕获点 | 文件位置 | 函数/方法 | 说明 |
|--------|----------|-----------|------|
| **REQUEST_ENTRY** | `async_llm.py:356` | `add_request()` | 请求入口，捕获输入 |
| **SCHEDULE_DECISION** | `common_engine.py:829` | `_schedule_request_to_worker_v1()` | 调度决策点 |
| **RESOURCE_ALLOCATED** | `common_engine.py:484` | `insert_tasks()` | 资源分配完成 |
| **OUTPUT_RESULT** | `common_engine.py:1608` | `scheduler.put_results()` | 输出结果 |
| **METRICS_FINAL** | `common_engine.py:1608` | `put_results()` + `finished=True` | 最终指标 |

#### 5.5.2 各捕获点详细内容

**捕获点 1：REQUEST_ENTRY（请求入口）**

位置：`fastdeploy/engine/async_llm.py` 第 356 行 `add_request()` 方法

```python
# 在 request 构建完成后捕获
baseline_record = {
    "capture_point": "REQUEST_ENTRY",
    "request_id": request_id,
    "timestamp": time.time(),
    "input": {
        "prompt": prompt,                          # 原始 prompt
        "prompt_token_ids": request["prompt_token_ids"],  # tokenize 后的 token ids
        "prompt_token_ids_len": len(request["prompt_token_ids"]),
        "sampling_params": {
            "max_tokens": request.get("max_tokens"),
            "temperature": request.get("temperature"),
            "top_p": request.get("top_p"),
            "top_k": request.get("top_k"),
            "min_tokens": request.get("min_tokens"),
        },
    },
}
```

**捕获点 2：SCHEDULE_DECISION（调度决策）**

位置：`fastdeploy/engine/common_engine.py` 第 829 行 `scheduler.get_requests()` 之后

```python
# 在 tasks = self.scheduler.get_requests(...) 之后捕获
baseline_record = {
    "capture_point": "SCHEDULE_DECISION",
    "timestamp": time.time(),
    "schedule_batch": {
        "tasks_count": len(tasks),
        "task_ids": [t.request_id for t in tasks],
        "available_batch": self.resource_manager.available_batch(),
        "available_blocks": self.resource_manager.available_block_num(),
        "max_num_batched_tokens": self.cfg.scheduler_config.max_num_batched_tokens,
    },
}
```

**捕获点 3：RESOURCE_ALLOCATED（资源分配）**

位置：`fastdeploy/engine/common_engine.py` 第 484 行 `allocate_resources_for_new_tasks()` 之后

```python
# 在 tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks) 之后捕获
for task in tasks:
    baseline_record = {
        "capture_point": "RESOURCE_ALLOCATED",
        "request_id": task.request_id,
        "timestamp": time.time(),
        "resources": {
            "slot_idx": task.idx,                              # 分配的槽位
            "block_tables": list(task.block_tables) if task.block_tables else [],
            "num_blocks": len(task.block_tables) if task.block_tables else 0,
            "num_cached_tokens": getattr(task, "num_cached_tokens", 0),
        },
    }
```

**捕获点 4：OUTPUT_RESULT（输出结果）**

位置：`fastdeploy/engine/common_engine.py` 第 1608 行 `scheduler.put_results()` 调用处

```python
# 在 self.scheduler.put_results(ready_request_outputs) 之前捕获
for output in ready_request_outputs:
    baseline_record = {
        "capture_point": "OUTPUT_RESULT",
        "request_id": output.request_id,
        "timestamp": time.time(),
        "output": {
            "token_ids": list(output.outputs[0].token_ids) if output.outputs else [],
            "text": output.outputs[0].text if output.outputs else "",
            "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
            "finished": output.finished,
            "error_code": output.error_code,
            "error_msg": output.error_msg,
        },
    }
```

**捕获点 5：METRICS_FINAL（最终指标）**

位置：与捕获点 4 相同，但仅在 `finished=True` 时捕获

```python
# 在请求完成时捕获性能指标
if output.finished and output.metrics:
    baseline_record = {
        "capture_point": "METRICS_FINAL",
        "request_id": output.request_id,
        "timestamp": time.time(),
        "metrics": {
            "arrival_time": output.metrics.arrival_time,
            "first_scheduled_time": output.metrics.first_scheduled_time,
            "first_token_time": output.metrics.first_token_time,
            "finished_time": output.metrics.finished_time,
            "ttft_ms": (output.metrics.first_token_time - output.metrics.arrival_time) * 1000,
            "total_time_ms": (output.metrics.finished_time - output.metrics.arrival_time) * 1000,
            "prompt_tokens": output.metrics.prompt_tokens,
            "generated_tokens": output.metrics.generated_tokens,
        },
    }
```

#### 5.5.3 基线捕获器实现

```python
# fastdeploy/baseline/capture.py
"""
基线捕获器

用于重构前后的行为对比验证
"""

import json
import os
import time
from typing import Any, Dict, List, Optional


class BaselineCapture:
    """
    基线捕获器（单例模式）

    通过环境变量 FD_CAPTURE_BASELINE=1 启用
    """

    _instance: Optional["BaselineCapture"] = None

    def __init__(self, output_dir: str = "baseline_data"):
        self.output_dir = output_dir
        self.enabled = os.getenv("FD_CAPTURE_BASELINE", "0") == "1"
        self.records: List[Dict[str, Any]] = []

        if self.enabled:
            os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "BaselineCapture":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def capture(self, point: str, request_id: str, data: Dict[str, Any]):
        """
        捕获一条基线记录

        Args:
            point: 捕获点名称
            request_id: 请求 ID
            data: 捕获的数据
        """
        if not self.enabled:
            return

        record = {
            "capture_point": point,
            "request_id": request_id,
            "timestamp": time.time(),
            **data
        }
        self.records.append(record)

    def save(self, filename: Optional[str] = None) -> str:
        """
        保存基线数据到文件

        Args:
            filename: 文件名，默认使用时间戳

        Returns:
            保存的文件路径
        """
        if not filename:
            filename = f"baseline_{int(time.time())}.json"

        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

        return path

    def clear(self):
        """清空记录"""
        self.records.clear()


# 便捷函数
def capture_baseline(point: str, request_id: str, data: Dict[str, Any]):
    """捕获基线（便捷函数）"""
    BaselineCapture.get_instance().capture(point, request_id, data)
```

#### 5.5.4 基线对比验证器

```python
# fastdeploy/baseline/verify.py
"""
基线对比验证器
"""

import json
from typing import Dict, List, Tuple


class BaselineVerifier:
    """
    基线对比验证器

    对比重构前后的基线数据，找出差异
    """

    def __init__(self, baseline_path: str, new_path: str):
        self.baseline = self._load(baseline_path)
        self.new_data = self._load(new_path)

    def _load(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def verify(self) -> Tuple[bool, List[str]]:
        """
        执行验证

        Returns:
            (是否通过, 差异列表)
        """
        diffs = []

        # 按 request_id 分组对比
        baseline_by_req = self._group_by_request(self.baseline)
        new_by_req = self._group_by_request(self.new_data)

        # 检查请求数量
        if set(baseline_by_req.keys()) != set(new_by_req.keys()):
            diffs.append(f"Request IDs mismatch: baseline={len(baseline_by_req)}, new={len(new_by_req)}")

        # 逐请求对比
        for req_id in baseline_by_req:
            if req_id not in new_by_req:
                diffs.append(f"Request {req_id} missing in new data")
                continue

            req_diffs = self._compare_request(
                baseline_by_req[req_id],
                new_by_req[req_id],
                req_id
            )
            diffs.extend(req_diffs)

        return len(diffs) == 0, diffs

    def _group_by_request(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """按 request_id 分组"""
        grouped = {}
        for record in records:
            req_id = record.get("request_id")
            if req_id not in grouped:
                grouped[req_id] = []
            grouped[req_id].append(record)
        return grouped

    def _compare_request(
        self,
        baseline_records: List[Dict],
        new_records: List[Dict],
        req_id: str
    ) -> List[str]:
        """对比单个请求的记录"""
        diffs = []

        for base_rec in baseline_records:
            point = base_rec["capture_point"]
            new_rec = self._find_record(new_records, point)

            if new_rec is None:
                diffs.append(f"{req_id}: Missing capture point {point}")
                continue

            # 根据捕获点类型执行不同的对比策略
            if point == "OUTPUT_RESULT":
                # 输出必须完全一致
                if base_rec.get("output", {}).get("token_ids") != new_rec.get("output", {}).get("token_ids"):
                    diffs.append(f"{req_id}: OUTPUT_RESULT token_ids mismatch")

            elif point == "METRICS_FINAL":
                # 性能指标允许 10% 波动
                base_ttft = base_rec.get("metrics", {}).get("ttft_ms", 0)
                new_ttft = new_rec.get("metrics", {}).get("ttft_ms", 0)
                if base_ttft > 0 and abs(new_ttft - base_ttft) / base_ttft > 0.1:
                    diffs.append(
                        f"{req_id}: METRICS_FINAL ttft regression "
                        f"{base_ttft:.2f}ms → {new_ttft:.2f}ms"
                    )

            elif point == "SCHEDULE_DECISION":
                # 调度顺序必须一致
                base_ids = base_rec.get("schedule_batch", {}).get("task_ids", [])
                new_ids = new_rec.get("schedule_batch", {}).get("task_ids", [])
                if base_ids != new_ids:
                    diffs.append(f"{req_id}: SCHEDULE_DECISION order mismatch")

        return diffs

    def _find_record(self, records: List[Dict], point: str) -> Dict:
        """查找指定捕获点的记录"""
        for rec in records:
            if rec.get("capture_point") == point:
                return rec
        return None
```

#### 5.5.5 使用流程

```bash
# 1. 重构前：捕获基线
FD_CAPTURE_BASELINE=1 python -m pytest tests/engine/test_e2e.py -v
# 生成 baseline_data/baseline_1234567890.json

# 2. 执行重构
# ...

# 3. 重构后：捕获新数据
FD_CAPTURE_BASELINE=1 python -m pytest tests/engine/test_e2e.py -v
# 生成 baseline_data/baseline_1234567891.json

# 4. 对比验证
python -c "
from fastdeploy.baseline.verify import BaselineVerifier
verifier = BaselineVerifier(
    'baseline_data/baseline_1234567890.json',
    'baseline_data/baseline_1234567891.json'
)
passed, diffs = verifier.verify()
if passed:
    print('✅ 验证通过')
else:
    print('❌ 发现差异:')
    for d in diffs:
        print(f'  - {d}')
"
```

#### 5.5.6 各捕获点验证要求汇总

| 捕获点 | 验证要求 | 说明 |
|--------|---------|------|
| REQUEST_ENTRY | 输入一致性 | 确保输入没有被意外修改 |
| SCHEDULE_DECISION | 调度顺序一致 | 相同输入必须产生相同调度决策 |
| RESOURCE_ALLOCATED | 资源数量一致 | 分配的块数等资源量应一致 |
| OUTPUT_RESULT | **必须完全一致** | token_ids、text 必须完全匹配 |
| METRICS_FINAL | 允许 ±10% 波动 | 性能指标允许小幅波动 |

---

## 六、实施步骤

### 阶段 1：准备（1 周）

1. 补充现有 EngineService 的测试覆盖率
2. 建立黄金测试基线
3. 提取 sleep 配置常量

### 阶段 2：拆分（2 周）

1. 创建组件目录结构
2. 逐个抽取组件（按依赖顺序）：
   - IPCManager（无依赖）
   - ProcessManager（依赖 IPC）
   - ResourceCoordinator（依赖 IPC）
   - SchedulerCoordinator（依赖 Resource、IPC）
3. 重构 EngineService 为门面类
4. 每个组件配套单元测试

### 阶段 3：验证（1 周）

1. 运行行为等价性测试
2. 运行黄金测试
3. 运行全量集成测试
4. 性能对比测试

### 阶段 4：切换（1 周）

1. Feature Flag 控制新旧实现
2. 灰度发布
3. 监控告警
4. 清理旧代码

---

## 七、目录结构

```
fastdeploy/engine/
├── engine_service.py           # 重构后的门面类
├── common_engine.py            # 旧代码（保留用于回滚）
├── components/
│   ├── __init__.py
│   ├── scheduler_coordinator.py
│   ├── process_manager.py
│   ├── ipc_manager.py
│   └── resource_coordinator.py
├── constants.py                # 时序配置常量
└── utils/
    ├── __init__.py
    ├── backoff.py              # 自适应退避
    └── sync.py                 # 同步原语
```

---

## 八、风险与回滚

### 8.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 组件间通信异常 | 中 | 高 | 完善集成测试 |
| 性能回退 | 低 | 中 | 性能基准测试 |
| 行为不一致 | 低 | 高 | 黄金测试+等价性测试 |

### 8.2 回滚方案

```python
# 使用 Feature Flag 控制
class EngineService:
    def __init__(self, cfg):
        if envs.USE_NEW_ENGINE_ARCHITECTURE.get():
            self._init_new_architecture(cfg)
        else:
            self._init_old_architecture(cfg)
```

回滚只需设置环境变量：
```bash
export FD_USE_NEW_ENGINE_ARCHITECTURE=false
```

---

## 九、收益预估

| 指标 | 重构前 | 重构后 | 提升 |
|------|-------|-------|------|
| 单个组件行数 | 2210 | 300-400 | **5x 更小** |
| 单测启动时间 | 30-60秒 | <1秒 | **>30x 更快** |
| 单测覆盖率 | ~30% | ~80% | **2.5x** |
| Bug 定位时间 | 1-2小时 | 15-30分钟 | **4x 更快** |
| 新功能开发 | 需理解全部代码 | 只需理解相关组件 | **显著提升** |

---

## 附录：交互契约

> 本章节记录 EngineService 与外部组件的所有交互接口，作为重构的"不变量"约束。重构前后这些交互必须保持一致。

### A. LLMEngine ↔ EngineService 交互契约

LLMEngine 通过直接方法调用与 EngineService 交互（同步模式）。

#### A.1 方法调用列表

| 类别 | 方法/属性 | 参数 | 返回值 | 调用时机 |
|------|----------|------|--------|---------|
| **启动** | `start()` | 无 | 无 | 初始化后启动核心服务 |
| **启动** | `create_data_processor()` | 无 | 无 | 创建 InputPreprocessor |
| **启动** | `start_cache_service(device_ids, suffix)` | `List[str]`, `int` | `List[Process]` | 启动缓存管理进程 |
| **启动** | `start_zmq_service(api_server_pid)` | `int` | 无 | 启动 ZMQ 通信服务 |
| **调度** | `scheduler.put_requests(requests)` | `List[Request]` | `List[Tuple[str, str]]` | 添加请求到调度队列 |
| **调度** | `scheduler.get_requests(...)` | 见下方详细参数 | `List[Request]` | 获取可调度的请求 batch |
| **调度** | `scheduler.get_results()` | 无 | `Dict[str, List[RequestOutput]]` | 获取生成结果 |
| **调度** | `scheduler.start(...)` | 根据调度器类型不同 | 无 | 启动调度器 |
| **资源** | `resource_manager.available_block_num()` | 无 | `int` | 查询可用 KV Cache 块数 |
| **资源** | `resource_manager.reset_cache_config(cfg)` | `CacheConfig` | 无 | 重置缓存配置 |
| **资源** | `resource_manager.check_and_free_block_tables()` | 无 | 无 | 释放已完成请求的资源 |
| **数据** | `data_processor.process_request(...)` | 请求数据 | 处理后的 Request | 预处理输入 |
| **数据** | `data_processor.process_response(...)` | 响应数据 | 处理后的输出 | 后处理输出 |
| **健康** | `worker_healthy_live_signal.value[0]` | 无 | `int` (时间戳) | 检查 Worker 存活 |
| **连接** | `split_connector.start_receiver()` | 无 | 无 | 启动 Splitwise 接收器 |

#### A.2 关键方法参数详情

**`scheduler.get_requests()` 参数**：
```python
def get_requests(
    self,
    available_blocks: int,          # 可用 KV Cache 块数
    block_size: int,                # 每块大小
    reserved_output_blocks: int,    # 预留输出块数
    max_num_batched_tokens: int,    # 最大批处理 token 数
    batch: int = 1,                 # 最大请求数
) -> List[Request]
```

#### A.3 调用流程图

```
LLMEngine.__init__():
    self.engine = EngineService(cfg)

LLMEngine.start():
    self.engine.start()                     # 启动核心线程
    self.engine.create_data_processor()     # 创建数据处理器
    self.engine.start_cache_service(...)    # 启动缓存服务 (条件)
    self.engine.scheduler.start(...)        # 启动调度器
    self.engine.start_zmq_service(...)      # 启动 ZMQ (条件)

LLMEngine.add_requests():
    processed = self.engine.data_processor.process_request(...)
    self.engine.scheduler.put_requests([processed])

LLMEngine._get_generated_result():
    results = self.engine.scheduler.get_results()

LLMEngine.generate():
    output = self.engine.data_processor.process_response(...)
    self.engine.resource_manager.check_and_free_block_tables()
```

---

### B. AsyncLLM ↔ EngineService ZMQ 通信契约

AsyncLLM 通过 ZMQ 与 EngineService 进行进程间通信（异步模式）。

#### B.1 通信架构

```
AsyncLLM (客户端)                        EngineService (服务端)
       │                                        │
       │ ──── PUSH/PULL (请求通道) ──────────>  │
       │      ipc:///dev/shm/{pid}.socket       │
       │                                        │
       │ <─── DEALER/ROUTER (响应通道) ───────  │
       │      ipc:///dev/shm/router_{pid}.ipc   │
```

#### B.2 消息格式

**请求消息封装**（所有请求都包含元数据）：
```python
envelope = {
    "__meta": {
        "send_ts": float  # 发送时间戳，用于延迟计算
    },
    "data": {
        # 实际请求数据
    }
}
```

**推理请求格式**：
```python
{
    "request_id": str,              # 唯一请求 ID
    "prompt": str | List[int],      # 输入提示或 token IDs
    "prompt_token_ids": List[int],  # token IDs
    "prompt_token_ids_len": int,    # token 长度
    "max_tokens": int,              # 最大生成长度
    "min_tokens": int,              # 最小生成长度
    "sampling_params": {...},       # 采样参数
    "multimodal_inputs": {...},     # 多模态输入 (可选)
    "metrics": {...},               # 性能指标
    # ... 其他字段
}
```

**控制请求格式** (ControlRequest)：
```python
{
    "request_id": str,      # 控制请求 ID
    "method": str,          # 方法名: "pause" | "resume" | "reset_scheduler" | ...
    "args": Dict            # 方法参数
}
```

**中止请求格式**：
```python
{
    "request_id": str,      # 要中止的请求 ID
    "status": 4             # RequestStatus.ABORT.value
}
```

**推理响应格式** (RequestOutput)：
```python
{
    "request_id": str,
    "prompt": str,
    "prompt_token_ids": List[int],
    "outputs": {                    # CompletionOutput
        "index": int,
        "send_idx": int,
        "token_ids": List[int],
        "text": str,
        "logprob": float,
        "reasoning_content": str,   # 可选
    },
    "finished": bool,
    "error_code": int,              # 200=成功, 400=客户端错误, 500=服务端错误
    "error_msg": str,
    "zmq_send_time": float,
}
```

**控制响应格式** (ControlResponse)：
```python
{
    "request_id": str,
    "finished": bool,               # 通常为 True
    "error_code": int,              # 状态码
    "error_message": str,
    "result": Dict                  # 返回结果
}
```

#### B.3 序列化方式

| 消息类型 | 序列化方式 | 条件 |
|---------|-----------|------|
| 普通推理请求 | JSON (`jsonapi.dumps`) | 非多模态 |
| 多模态请求 | Pickle (`ForkingPickler.dumps`) | 包含图像/视频 |
| 所有响应 | Pickle (`ForkingPickler.dumps`) | - |

#### B.4 握手协议

AsyncLLM 发送请求后，需要通过 DEALER 通道注册 request_id：
```python
# 客户端
dealer.write([b"", request_id.encode("utf-8")])

# 服务端接收并记录
client, _, request_id = self.socket.recv_multipart()
self.req_dict[request_id.decode()] = client  # 保存用于响应路由
```

---

### C. EngineService ↔ Worker 通信契约

EngineService 通过 EngineWorkerQueue 和 IPC Signal 与 Worker 进程通信。

#### C.1 EngineWorkerQueue 消息格式

**队列消息格式**：
```python
(tasks: List[Request | ControlRequest | ScheduledTask], batch_size: int)
```

**任务类型 (RequestType)**：
```python
class RequestType(Enum):
    PREFILL = 0     # Prefill 任务（首次处理 prompt）
    DECODE = 1      # Decode 任务（生成 token）
    PREEMPTED = 2   # 抢占任务（终止推理并回收资源）
    EXTEND = 3      # 扩展任务（分配新 blocks）
```

**Request 对象关键字段**：
```python
class Request:
    request_id: str                    # 唯一请求 ID
    prompt_token_ids: List[int]        # 输入 token IDs
    prompt_token_ids_len: int          # 输入长度
    task_type: RequestType             # 任务类型
    block_tables: List[int]            # KV Cache block 表
    output_token_ids: List[int]        # 输出 token IDs
    num_computed_tokens: int           # 已计算 token 数
    prefill_start_index: int           # prefill 开始索引
    prefill_end_index: int             # prefill 结束索引
    sampling_params: SamplingParams    # 采样参数
    multimodal_inputs: Optional[dict]  # 多模态输入
    metrics: RequestMetrics            # 性能指标
```

**V1 调度器专用任务类型**：
```python
@dataclass
class ScheduledDecodeTask:      # 分配新 blocks 用于 decode
    idx: int
    request_id: str
    block_tables: List[int]
    task_type: RequestType = RequestType.DECODE

@dataclass
class ScheduledPreemptTask:     # 终止推理回收资源
    idx: int
    request_id: str
    task_type: RequestType = RequestType.PREEMPTED

@dataclass
class ScheduledExtendBlocksTask: # 分配新 blocks 用于扩展
    idx: int
    request_id: str
    extend_block_tables: List[int]
    task_type: RequestType = RequestType.EXTEND
```

#### C.2 IPC Signal 列表

| Signal 名称 | 数据类型 | 大小 | 写入方 | 读取方 | 用途 |
|------------|---------|------|-------|-------|------|
| `exist_task_signal` | int32 | [1] | Engine | Worker | 通知 Worker 有新任务 |
| `exist_swapped_task_signal` | int32 | [1] | Worker | Engine | 通知 Engine 有 swapped 任务 |
| `exist_prefill_task_signal` | int32 | [1] | Engine | Worker | 通知 Worker 进行 prefill |
| `worker_healthy_live_signal` | int32 | [tp_size] | Worker | Engine | Worker 心跳时间戳 |
| `cache_ready_signal` | int32 | [tp_size] | Worker | Engine | Cache 就绪通知 |
| `swap_space_ready_signal` | int32 | [tp_size] | Worker | Engine | Swap 空间就绪 |
| `cache_transfer_inited_signal` | int32 | [tp_size] | Worker | Engine | Cache 传输初始化完成 |
| `model_weights_status` | int32 | [1] | Both | Both | 模型权重状态同步 |
| `prefix_tree_status` | int32 | [1] | Both | Both | 前缀树状态同步 |
| `kv_cache_status` | int32 | [1] | Both | Both | KV Cache 状态同步 |
| `loaded_model_signal` | int32 | [1] | Worker | Engine | 模型加载完成通知 |
| `worker_ready_signal` | int32 | [tp_size] | Worker | Engine | Worker 就绪通知 |
| `get_profile_block_num_signal` | int32 | [1] | Worker | Engine | Profile block 数量 |

**状态常量**：
```python
class ExistTaskStatus:
    EMPTY = 0    # 无任务
    EXIST = 1    # 有任务

class ModelWeightsStatus:
    NORMAL = 0      # 正常
    UPDATING = 1    # 更新中
    CLEARING = -1   # 清理中
    CLEARED = -2    # 已清理
```

#### C.3 控制命令

| 命令 | 方向 | 处理方 | 说明 |
|------|------|-------|------|
| `pause` | Client → Engine | Engine | 暂停引擎，终止所有请求 |
| `resume` | Client → Engine | Engine | 恢复引擎 |
| `is_paused` | Client → Engine | Engine | 查询暂停状态 |
| `abort` | Client → Engine | Engine | 中止指定请求 |
| `update_weights` | Engine → Worker | Worker | 更新模型权重 |

**Pause 处理流程**：
```
1. 接收 ControlRequest(method="pause")
2. 设置 is_paused = True
3. 等待 engine_worker_queue 清空
4. 调用 resource_manager.preempted_all()
5. 发送 preempted tasks 到 worker
6. 等待 worker 完成 inflight 请求
7. 清理 token_processor 和 scheduler
8. 重置 cache_manager
9. 返回 ControlResponse
```

#### C.4 FMQ 控制响应队列

Worker 执行控制命令后，通过 FMQ 返回响应：
```python
# Worker 端
self._ctrl_output.put(ControlResponse(request_id, error_code, result))

# Engine 端
response = self._ctrl_worker_output_queues[rank].queue.get()
```

---

### D. 契约验证清单

重构时必须确保以下契约不变：

| 契约类型 | 验证方式 | 优先级 |
|---------|---------|-------|
| LLMEngine 方法签名 | 单元测试 Mock | 高 |
| ZMQ 消息格式 | 协议测试 | 高 |
| Queue 消息格式 | 协议测试 | 高 |
| IPC Signal 名称和语义 | 集成测试 | 高 |
| 控制命令处理流程 | 端到端测试 | 中 |
| 错误码定义 | 文档 + 测试 | 中 |

---

## 附录 E：基线捕获测试程序

> 本章节设计用于捕获通信数据的测试程序，覆盖 LLM（同步）和 AsyncLLM（异步）两种模式。

### E.1 设计目标

设计一个全面的测试程序，用于：
1. **覆盖两种运行模式**：LLM（同步）和 AsyncLLM（异步）
2. **展示各种参数组合**：SamplingParams 的各种配置
3. **覆盖各种生成场景**：单请求、批量、流式、Chat 等
4. **集成基线捕获**：方便捕获重构前后的通信数据

### E.2 测试程序结构

```
tests/baseline_capture/
├── __init__.py
├── test_scenarios.py          # 测试场景定义
├── test_llm_sync.py           # LLM 同步模式测试
├── test_async_llm.py          # AsyncLLM 异步模式测试
├── run_all_scenarios.py       # 运行所有场景的入口
└── README.md                  # 使用说明
```

### E.3 测试场景定义

```python
# tests/baseline_capture/test_scenarios.py
"""
基线捕获测试场景定义

定义各种测试场景，覆盖不同的参数组合和生成模式
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class GenerationMode(Enum):
    """生成模式"""
    SYNC = "sync"           # 同步模式（LLM）
    ASYNC = "async"         # 异步模式（AsyncLLM）


class OutputMode(Enum):
    """输出模式"""
    BLOCKING = "blocking"   # 阻塞式，等待完成
    STREAMING = "streaming" # 流式，逐 token 返回


@dataclass
class SamplingConfig:
    """采样参数配置"""
    name: str                                    # 配置名称
    temperature: float = 1.0                     # 温度
    top_p: float = 1.0                           # Top-P
    top_k: int = 0                               # Top-K (0=禁用)
    max_tokens: int = 100                        # 最大生成长度
    min_tokens: int = 1                          # 最小生成长度
    repetition_penalty: float = 1.0              # 重复惩罚
    presence_penalty: float = 0.0                # 存在惩罚
    frequency_penalty: float = 0.0               # 频率惩罚
    seed: Optional[int] = None                   # 随机种子
    stop: Optional[List[str]] = None             # 停止词
    n: int = 1                                   # 生成数量


@dataclass
class TestPrompt:
    """测试提示"""
    name: str                                    # 提示名称
    prompt: Union[str, List[int], Dict]          # 提示内容
    expected_min_tokens: int = 10                # 预期最小生成长度
    description: str = ""                        # 描述


@dataclass
class TestScenario:
    """测试场景"""
    name: str                                    # 场景名称
    description: str                             # 场景描述
    generation_mode: GenerationMode              # 生成模式
    output_mode: OutputMode                      # 输出模式
    prompts: List[TestPrompt] = field(default_factory=list)
    sampling_config: SamplingConfig = field(default_factory=lambda: SamplingConfig("default"))
    concurrent: bool = False                     # 是否并发请求
    chat_mode: bool = False                      # 是否为 Chat 模式
    tags: List[str] = field(default_factory=list)  # 标签，用于过滤


# ============================================================
# 预定义的采样配置
# ============================================================

SAMPLING_CONFIGS = {
    "greedy": SamplingConfig(name="greedy", temperature=0.0, top_p=1.0, max_tokens=100),
    "deterministic": SamplingConfig(name="deterministic", temperature=0.01, top_p=0.95, max_tokens=100, seed=42),
    "creative": SamplingConfig(name="creative", temperature=0.9, top_p=0.95, top_k=50, max_tokens=200),
    "balanced": SamplingConfig(name="balanced", temperature=0.7, top_p=0.9, max_tokens=150),
    "no_repeat": SamplingConfig(name="no_repeat", temperature=0.8, repetition_penalty=1.2, presence_penalty=0.5, max_tokens=150),
    "short": SamplingConfig(name="short", temperature=0.7, max_tokens=30, min_tokens=10),
    "long": SamplingConfig(name="long", temperature=0.7, max_tokens=500, min_tokens=100),
    "multi_choice": SamplingConfig(name="multi_choice", temperature=0.9, n=3, max_tokens=50),
    "with_stop": SamplingConfig(name="with_stop", temperature=0.7, stop=["。", ".", "\n\n"], max_tokens=200),
}

# ============================================================
# 预定义的测试提示
# ============================================================

TEST_PROMPTS = {
    "simple_en": TestPrompt(name="simple_en", prompt="Hello, my name is", expected_min_tokens=20, description="简单英文补全"),
    "simple_cn": TestPrompt(name="simple_cn", prompt="你好，请介绍一下", expected_min_tokens=30, description="简单中文补全"),
    "qa_capital": TestPrompt(name="qa_capital", prompt="中国的首都是哪里？", expected_min_tokens=10, description="简单问答"),
    "qa_math": TestPrompt(name="qa_math", prompt="请计算 123 + 456 = ?", expected_min_tokens=5, description="数学问答"),
    "long_prompt": TestPrompt(
        name="long_prompt",
        prompt="北京是中华人民共和国的首都，是全国政治中心、文化中心、国际交往中心、科技创新中心。" * 10 + "请总结上文的主要内容：",
        expected_min_tokens=50,
        description="长提示测试 Prefill",
    ),
}

# ============================================================
# Chat 测试用例
# ============================================================

CHAT_CASES = {
    "single_turn": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    "multi_turn": [
        {"role": "user", "content": "你知道地球到月球的距离是多少吗？"},
        {"role": "assistant", "content": "大约是38万公里左右。"},
        {"role": "user", "content": "那太阳到地球的距离是多少？"},
    ],
    "system_prompt": [
        {"role": "system", "content": "你是一个专业的科学顾问，请用简洁的语言回答问题。"},
        {"role": "user", "content": "什么是量子计算？"},
    ],
}


def get_all_scenarios() -> List[TestScenario]:
    """获取所有测试场景"""
    scenarios = []

    # 同步模式场景
    scenarios.append(TestScenario(
        name="sync_blocking_single_greedy",
        description="同步模式，阻塞输出，单请求，Greedy 采样",
        generation_mode=GenerationMode.SYNC,
        output_mode=OutputMode.BLOCKING,
        prompts=[TEST_PROMPTS["simple_en"]],
        sampling_config=SAMPLING_CONFIGS["greedy"],
        tags=["sync", "blocking", "single", "greedy"],
    ))

    # ... 更多场景见完整实现

    return scenarios


def get_scenarios_by_tags(tags: List[str]) -> List[TestScenario]:
    """根据标签过滤场景"""
    all_scenarios = get_all_scenarios()
    return [s for s in all_scenarios if any(t in s.tags for t in tags)]
```

### E.4 测试场景覆盖矩阵

| 场景 | 模式 | 输出 | 并发 | 采样 | 说明 |
|------|------|------|------|------|------|
| sync_blocking_single_greedy | 同步 | 阻塞 | 否 | Greedy | 最简单场景 |
| sync_blocking_single_creative | 同步 | 阻塞 | 否 | 高温度 | 创意生成 |
| sync_blocking_batch | 同步 | 阻塞 | 否 | 平衡 | 批量请求 |
| sync_streaming_single | 同步 | 流式 | 否 | 平衡 | 流式单请求 |
| sync_streaming_batch | 同步 | 流式 | 否 | 平衡 | 流式批量 |
| sync_long_prompt | 同步 | 阻塞 | 否 | 平衡 | 长 Prefill |
| sync_chat_single_turn | 同步 | 阻塞 | 否 | 平衡 | Chat 单轮 |
| sync_chat_multi_turn | 同步 | 阻塞 | 否 | 平衡 | Chat 多轮 |
| sync_multi_choice | 同步 | 阻塞 | 否 | n=3 | 多选项生成 |
| sync_with_stop | 同步 | 阻塞 | 否 | 停止词 | 停止词控制 |
| async_single | 异步 | 流式 | 否 | 平衡 | 异步单请求 |
| async_concurrent | 异步 | 流式 | **是** | 平衡 | 异步并发 |
| async_long_prompt | 异步 | 流式 | 否 | 平衡 | 异步长 Prefill |
| async_multi_choice | 异步 | 流式 | 否 | n=3 | 异步多选项 |
| async_high_concurrency | 异步 | 流式 | **是** | 短输出 | 高并发压力 |

### E.5 LLM 同步模式测试

```python
# tests/baseline_capture/test_llm_sync.py
"""LLM 同步模式基线捕获测试"""

import os
import time
import json
from typing import List, Dict, Any, Optional

os.environ["FD_CAPTURE_BASELINE"] = "1"  # 启用基线捕获

from fastdeploy import LLM, SamplingParams
from fastdeploy.baseline.capture import BaselineCapture
from test_scenarios import (
    TestScenario, GenerationMode, OutputMode,
    get_scenarios_by_tags, CHAT_CASES
)


class LLMSyncTester:
    """LLM 同步模式测试器"""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1, max_model_len: int = 8192, **kwargs):
        self.llm_kwargs = {"model": model_path, "tensor_parallel_size": tensor_parallel_size, "max_model_len": max_model_len, **kwargs}
        self.llm = None

    def setup(self):
        print(f"[LLMSyncTester] 初始化 LLM...")
        self.llm = LLM(**self.llm_kwargs)

    def teardown(self):
        if self.llm:
            del self.llm
            self.llm = None

    def _create_sampling_params(self, config) -> SamplingParams:
        return SamplingParams(
            temperature=config.temperature, top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            max_tokens=config.max_tokens, min_tokens=config.min_tokens,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            seed=config.seed, stop=config.stop, n=config.n,
        )

    def run_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """运行单个测试场景"""
        print(f"\n{'='*60}")
        print(f"[场景] {scenario.name}: {scenario.description}")

        sampling_params = self._create_sampling_params(scenario.sampling_config)
        result = {"scenario_name": scenario.name, "success": False, "outputs": [], "error": None}

        try:
            start_time = time.time()

            if scenario.chat_mode:
                result["outputs"] = self._run_chat(scenario, sampling_params)
            elif scenario.output_mode == OutputMode.STREAMING:
                result["outputs"] = self._run_streaming(scenario, sampling_params)
            else:
                result["outputs"] = self._run_blocking(scenario, sampling_params)

            result["timing_ms"] = (time.time() - start_time) * 1000
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
            import traceback; traceback.print_exc()

        return result

    def _run_blocking(self, scenario, sampling_params):
        prompts = [p.prompt for p in scenario.prompts]
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)
        return [{"request_id": o.request_id, "text": o.outputs.text} for o in outputs]

    def _run_streaming(self, scenario, sampling_params):
        prompts = [p.prompt for p in scenario.prompts]
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params, stream=True, use_tqdm=False)
        results = [{} for _ in prompts]
        for chunk in outputs:
            for i, c in enumerate(chunk):
                if c is not None and c.finished:
                    results[i] = {"request_id": c.request_id, "text": c.outputs.text}
        return results

    def _run_chat(self, scenario, sampling_params):
        chat_case = CHAT_CASES.get("multi_turn" if "multi_turn" in scenario.tags else "single_turn")
        outputs = self.llm.chat([chat_case], sampling_params, stream=True)
        results = [{"tokens": []}]
        for chunks in outputs:
            for i, chunk in enumerate(chunks):
                if chunk is not None:
                    results[i]["tokens"].append(chunk.outputs.text)
                    if chunk.finished:
                        results[i]["text"] = "".join(results[i]["tokens"])
        return results


def run_sync_tests(model_path: str, tags: Optional[List[str]] = None, output_file: str = "baseline_sync.json"):
    scenarios = get_scenarios_by_tags(tags or ["sync"])
    tester = LLMSyncTester(model_path=model_path)
    tester.setup()

    all_results = []
    try:
        for scenario in scenarios:
            if scenario.generation_mode == GenerationMode.SYNC:
                all_results.append(tester.run_scenario(scenario))
    finally:
        tester.teardown()
        BaselineCapture.get_instance().save(output_file)
        with open(output_file.replace(".json", "_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--output", type=str, default="baseline_sync.json")
    args = parser.parse_args()
    run_sync_tests(args.model, args.tags, args.output)
```

### E.6 AsyncLLM 异步模式测试

```python
# tests/baseline_capture/test_async_llm.py
"""AsyncLLM 异步模式基线捕获测试"""

import os
import asyncio
import uuid
import time
import json
from typing import List, Dict, Any, Optional

os.environ["FD_CAPTURE_BASELINE"] = "1"

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.baseline.capture import BaselineCapture
from test_scenarios import TestScenario, GenerationMode, get_scenarios_by_tags


class AsyncLLMTester:
    """AsyncLLM 异步模式测试器"""

    def __init__(self, model_path: str, tensor_parallel_size: int = 1, max_model_len: int = 8192,
                 engine_worker_queue_port: int = 6778, cache_queue_port: int = 6779, **kwargs):
        self.engine_args = EngineArgs(
            model=model_path, tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len, engine_worker_queue_port=engine_worker_queue_port,
            cache_queue_port=cache_queue_port, **kwargs
        )
        self.engine = None
        self.loop = None

    def setup(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.engine = AsyncLLM.from_engine_args(self.engine_args, pid=self.engine_args.engine_worker_queue_port)
        self.loop.run_until_complete(self.engine.start())
        self.loop.run_until_complete(self.engine.init_connections())

    def teardown(self):
        if self.engine:
            self.loop.run_until_complete(self.engine.shutdown())
        if self.loop:
            self.loop.close()

    def _create_sampling_params(self, config) -> SamplingParams:
        return SamplingParams(
            temperature=config.temperature, top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            max_tokens=config.max_tokens, min_tokens=config.min_tokens,
            repetition_penalty=config.repetition_penalty,
            seed=config.seed, stop=config.stop, n=config.n,
        )

    def run_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"[场景] {scenario.name}: {scenario.description}")

        sampling_params = self._create_sampling_params(scenario.sampling_config)
        result = {"scenario_name": scenario.name, "success": False, "outputs": []}

        try:
            start = time.time()
            if scenario.concurrent:
                result["outputs"] = self.loop.run_until_complete(self._run_concurrent(scenario, sampling_params))
            else:
                result["outputs"] = self.loop.run_until_complete(self._run_single(scenario, sampling_params))
            result["timing_ms"] = (time.time() - start) * 1000
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
            import traceback; traceback.print_exc()

        return result

    async def _run_single(self, scenario, sampling_params):
        prompt = scenario.prompts[0].prompt
        request_id = f"test_{scenario.name}_{uuid.uuid4()}"
        outputs = []
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.finished:
                outputs.append({"request_id": output.request_id, "text": output.outputs.text})
        return outputs

    async def _run_concurrent(self, scenario, sampling_params):
        async def gen_single(prompt_cfg, idx):
            request_id = f"test_{scenario.name}_{idx}_{uuid.uuid4()}"
            async for output in self.engine.generate(prompt_cfg.prompt, sampling_params, request_id):
                if output.finished:
                    return {"request_id": output.request_id, "text": output.outputs.text, "index": idx}
            return {"index": idx, "error": "No output"}

        tasks = [gen_single(p, i) for i, p in enumerate(scenario.prompts)]
        return await asyncio.gather(*tasks, return_exceptions=True)


def run_async_tests(model_path: str, tags: Optional[List[str]] = None, output_file: str = "baseline_async.json"):
    scenarios = get_scenarios_by_tags(tags or ["async"])
    tester = AsyncLLMTester(model_path=model_path)
    tester.setup()

    all_results = []
    try:
        for scenario in scenarios:
            if scenario.generation_mode == GenerationMode.ASYNC:
                all_results.append(tester.run_scenario(scenario))
    finally:
        tester.teardown()
        BaselineCapture.get_instance().save(output_file)
        with open(output_file.replace(".json", "_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--output", type=str, default="baseline_async.json")
    args = parser.parse_args()
    run_async_tests(args.model, args.tags, args.output)
```

### E.7 统一入口脚本

```python
# tests/baseline_capture/run_all_scenarios.py
"""运行所有基线捕获测试场景"""

import os
import argparse
from datetime import datetime
from test_scenarios import get_all_scenarios


def main():
    parser = argparse.ArgumentParser(description="基线捕获测试 - 统一入口")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--mode", choices=["sync", "async", "all"], default="all", help="测试模式")
    parser.add_argument("--tags", type=str, nargs="+", help="按标签过滤场景")
    parser.add_argument("--list", action="store_true", help="列出所有场景")
    parser.add_argument("--output-dir", type=str, default="baseline_data", help="输出目录")
    args = parser.parse_args()

    if args.list:
        for s in get_all_scenarios():
            print(f"[{s.name}] {s.description} | 标签: {s.tags}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ["sync", "all"]:
        from test_llm_sync import run_sync_tests
        run_sync_tests(args.model, args.tags, f"{args.output_dir}/baseline_sync_{timestamp}.json")

    if args.mode in ["async", "all"]:
        from test_async_llm import run_async_tests
        run_async_tests(args.model, args.tags, f"{args.output_dir}/baseline_async_{timestamp}.json")

    print(f"\n测试完成！基线数据保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
```

### E.8 使用说明

```bash
# 1. 列出所有可用场景
python tests/baseline_capture/run_all_scenarios.py --list

# 2. 运行所有同步模式测试
python tests/baseline_capture/run_all_scenarios.py --model /path/to/model --mode sync

# 3. 运行所有异步模式测试
python tests/baseline_capture/run_all_scenarios.py --model /path/to/model --mode async

# 4. 运行所有测试（同步 + 异步）
python tests/baseline_capture/run_all_scenarios.py --model /path/to/model --mode all

# 5. 按标签过滤运行
python tests/baseline_capture/run_all_scenarios.py --model /path/to/model --tags streaming concurrent

# 6. 重构前后对比验证
FD_CAPTURE_BASELINE=1 python tests/baseline_capture/run_all_scenarios.py --model /path/to/model --mode all

# 对比验证
python -c "
from fastdeploy.baseline.verify import BaselineVerifier
v = BaselineVerifier('baseline_data/baseline_sync_before.json', 'baseline_data/baseline_sync_after.json')
passed, diffs = v.verify()
print('通过' if passed else f'差异: {diffs}')
"
```

### E.9 与基线捕获集成

测试程序自动集成 5.5 节中设计的基线捕获功能：

| 集成点 | 说明 |
|--------|------|
| 环境变量 | `FD_CAPTURE_BASELINE=1` 启用捕获 |
| 自动捕获 | 在 5 个关键点自动记录数据 |
| 保存基线 | 测试结束后保存到指定文件 |
| 对比验证 | 使用 `BaselineVerifier` 进行重构前后对比 |

---

*文档创建时间: 2026-02-14*
