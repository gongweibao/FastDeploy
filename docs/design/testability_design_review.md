# FastDeploy 可测试性改进设计文档

> **版本**: v1.0
> **日期**: 2026-02-16
> **状态**: 评审中

---

## 1. Context (背景)

### 1.1 AI 编程时代的可测试性：从"最佳实践"到"生存刚需"

FastDeploy 是百度飞桨团队开发的大语言模型推理部署工具。在 AI 辅助编程（Copilot/Cursor/Claude Code）成为主流研发范式的今天，**代码可测试性已从软件工程的"最佳实践"升级为"生存刚需"**。

#### 为什么 AI 编程时代更需要可测试性？

```
传统研发循环（人工主导）:
  需求 → 人工编码 → 人工评审 → 手动测试 → 上线
  ↑___________________↓
      周期：天/周级

AI 辅助研发循环（人机协作）:
  需求 → AI 生成代码 → 快速验证 → 迭代修正 → 上线
  ↑_________↓    ↑_________↓
   秒级生成       必须秒级反馈！
```

| AI 编程特征 | 对可测试性的要求 | FastDeploy 现状 |
|------------|-----------------|----------------|
| **代码生成速度快** | 验证速度必须匹配生成速度，否则成为瓶颈 | ❌ 需 GPU 环境，反馈慢 |
| **迭代频率高** | 每次修改都需快速回归验证 | ❌ 依赖端到端测试 |
| **AI 缺乏业务上下文** | 需要测试用例作为"行为契约"指导 AI | ❌ 测试覆盖率低 |
| **重构更频繁** | 良好的测试是重构的安全网 | ❌ 改动风险高 |

> **核心洞察**: 在 AI 编程时代，**可测试性 = AI 编程效率的上限**。
> 一个无法快速验证的代码库，等于给 AI 助手戴上了镣铐。

#### 当前痛点量化

| 问题 | 影响 | AI 编程场景下的放大效应 |
|------|------|----------------------|
| 单元测试覆盖率低 | 核心模块难以单元测试 | AI 生成的代码无法快速验证正确性 |
| 测试执行缓慢 | 需真实 GPU/ZMQ，CI 周期长 | 人机交互循环被拉长，效率大打折扣 |
| 回归风险高 | 代码改动难快速验证 | AI 辅助重构变得不可行 |
| 缺乏行为契约 | 接口语义隐藏在实现中 | AI 难以理解模块边界和预期行为 |

### 1.2 核心目标：打造 AI-Ready 的代码库

| 目标层次 | 具体目标 | 度量指标 | AI 编程收益 |
|---------|---------|---------|------------|
| **L1: 可测试** | 核心模块可单元测试 | EngineService/Scheduler/BlockManager 无 GPU/ZMQ 可测 | AI 生成代码可即时验证 |
| **L2: 快反馈** | 测试执行提速 | 单元测试 < 10 秒 | 人机交互循环从分钟级→秒级 |
| **L3: 高覆盖** | 代码覆盖率提升 | 核心调度逻辑 > 80% | 测试即文档，AI 可理解行为契约 |
| **L4: 可演进** | 架构清晰解耦 | 单一职责，依赖注入 | AI 辅助重构安全可控 |

---

## 1.3 FastDeploy 数据流转全景图

理解数据流转是改进可测试性的基础。以下是完整的数据流转架构图：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              用户请求入口                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     HTTP Request (OpenAI API)                            │    │
│  │           POST /v1/chat/completions  |  POST /v1/completions             │    │
│  └────────────────────────────────┬────────────────────────────────────────┘    │
└───────────────────────────────────┼─────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 1: Entrypoints (入口层)                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  api_server.py                     serving_chat.py                         │ │
│  │  ┌──────────────────┐              ┌───────────────────────────────────┐   │ │
│  │  │ • 路由分发        │              │ • Chat Template 应用              │   │ │
│  │  │ • 并发控制        │──Request ───▶│ • Tokenization (分词)            │   │ │
│  │  │ • 请求验证        │              │ • 多模态预处理                    │   │ │
│  │  └──────────────────┘              └───────────────────────────────────┘   │ │
│  └────────────────────────────────────────────┬───────────────────────────────┘ │
└───────────────────────────────────────────────┼─────────────────────────────────┘
                                                │ Request 对象
                                                │ (request_id, prompt_token_ids,
                                                │  sampling_params, messages)
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 2: Engine (引擎层)                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  AsyncLLM / LLMEngine                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐    │ │
│  │  │ • 请求生命周期管理                                                   │    │ │
│  │  │ • 异步结果等待 (async generator)                                    │    │ │
│  │  │ • 多模态特征提取                                                     │    │ │
│  │  └────────────────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────┬───────────────────────────────┘ │
│                                               │                                  │
│  ┌────────────────────────────────────────────▼───────────────────────────────┐ │
│  │  EngineService (核心协调者) - 2210行，可测试性改进重点                       │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  职责 1: 调度协调                                                      │ │ │
│  │  │  ├─ _schedule_request_to_worker_v1() - 270行                          │ │ │
│  │  │  └─ 从 Scheduler 获取请求，分配资源，发送到 Worker                     │ │ │
│  │  │                                                                       │ │ │
│  │  │  职责 2: 资源管理                                                      │ │ │
│  │  │  └─ ResourceManager: KV Cache 块分配/回收                              │ │ │
│  │  │                                                                       │ │ │
│  │  │  职责 3: IPC 通信 ⚠️ 硬编码 ZMQ 依赖                                   │ │ │
│  │  │  └─ recv_server / send_server: ZmqIpcServer                           │ │ │
│  │  │                                                                       │ │ │
│  │  │  职责 4: Worker 生命周期                                               │ │ │
│  │  │  └─ Worker 进程启动/停止/健康检查                                      │ │ │
│  │  │                                                                       │ │ │
│  │  │  职责 5: 输出处理                                                      │ │ │
│  │  │  └─ TokenProcessor: 处理采样结果                                       │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────┬───────────────────────────────┘ │
└───────────────────────────────────────────────┼─────────────────────────────────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        │                       │                       │
                        ▼                       ▼                       ▼
┌─────────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────┐
│  LocalScheduler             │  │  ResourceManager        │  │  EngineWorkerQueue  │
│  ┌─────────────────────┐    │  │  ┌─────────────────┐    │  │  ┌───────────────┐  │
│  │ • 请求队列管理       │    │  │  │ • KV Cache 分配 │    │  │  │ • 任务队列    │  │
│  │ • 排队/出队          │    │  │  │ • Block Tables  │    │  │  │ • 批次大小    │  │
│  │ • 资源约束检查       │    │  │  │ • 前缀缓存匹配  │    │  │  │ • 同步信号    │  │
│  └─────────────────────┘    │  │  └─────────────────┘    │  │  └───────────────┘  │
└─────────────────────────────┘  └─────────────────────────┘  └─────────┬───────────┘
                                                                        │
                                    ┌───────────────────────────────────┘
                                    │ (tasks, batch_size)
                                    │ via Queue + ZMQ
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 3: Worker (工作层)                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  GPUWorker                                                                 │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐    │ │
│  │  │ 主循环:                                                             │    │ │
│  │  │   while True:                                                       │    │ │
│  │  │       tasks = queue.get()           # 获取任务                      │    │ │
│  │  │       outputs = model_runner.run()  # 执行推理                      │    │ │
│  │  │       output_queue.put(outputs)     # 返回结果                      │    │ │
│  │  └────────────────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────┬───────────────────────────────┘ │
│                                               │                                  │
│  ┌────────────────────────────────────────────▼───────────────────────────────┐ │
│  │  GPUModelRunner (3123行)                                                   │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  InputBatch (共享输入缓冲区)                                           │ │ │
│  │  │  ┌─────────────────────────────────────────────────────────────────┐  │ │ │
│  │  │  │  Token 数据:                                                     │  │ │ │
│  │  │  │    input_ids, pre_ids, prompt_ids                               │  │ │ │
│  │  │  │                                                                 │  │ │ │
│  │  │  │  序列长度:                                                       │  │ │ │
│  │  │  │    seq_lens_encoder, seq_lens_decoder, seq_lens_this_time       │  │ │ │
│  │  │  │                                                                 │  │ │ │
│  │  │  │  采样参数:                                                       │  │ │ │
│  │  │  │    top_p, top_k, temperature, repetition_penalty                │  │ │ │
│  │  │  │                                                                 │  │ │ │
│  │  │  │  KV Cache 管理:                                                  │  │ │ │
│  │  │  │    block_tables, free_list, cache_k, cache_v                    │  │ │ │
│  │  │  │                                                                 │  │ │ │
│  │  │  │  控制标志:                                                       │  │ │ │
│  │  │  │    stop_flags, not_need_stop, is_prefill                        │  │ │ │
│  │  │  └─────────────────────────────────────────────────────────────────┘  │ │ │
│  │  │                                                                       │ │ │
│  │  │  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐   │ │ │
│  │  │  │ insert_tasks_v1  │───▶│  execute_model   │───▶│   Sampler      │   │ │ │
│  │  │  │ 更新 InputBatch  │    │ 模型前向推理      │    │ Token 采样     │   │ │ │
│  │  │  └──────────────────┘    └──────────────────┘    └────────────────┘   │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────┬───────────────────────────────┘ │
└───────────────────────────────────────────────┼─────────────────────────────────┘
                                                │
                                                │ SamplerOutput
                                                │ (token_ids, logprobs)
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 4: Output Processing (输出处理层)                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  TokenProcessor                                                            │ │
│  │  ┌────────────────────────────────────────────────────────────────────┐    │ │
│  │  │ • 接收采样结果 (via ZMQ)                                            │    │ │
│  │  │ • 逐 Token 处理                                                     │    │ │
│  │  │ • 检测停止条件 (EOS/stop_token)                                     │    │ │
│  │  │ • 生成 RequestOutput                                                │    │ │
│  │  │ • 资源回收触发                                                       │    │ │
│  │  └────────────────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────┬───────────────────────────────┘ │
└───────────────────────────────────────────────┼─────────────────────────────────┘
                                                │
                                                │ RequestOutput
                                                │ (request_id, outputs, finished)
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Layer 5: Response (响应层)                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌────────────────────────┐         ┌─────────────────────────────────┐    │ │
│  │  │  Streaming 模式        │         │  Non-Streaming 模式              │    │ │
│  │  │  • Server-Sent Events  │         │  • 完整响应后返回                 │    │ │
│  │  │  • 逐 Token 返回       │         │  • JSON Response                │    │ │
│  │  └────────────────────────┘         └─────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 数据流转关键阶段

| 阶段 | 数据结构 | 核心处理 | 可测试性障碍 |
|------|---------|---------|-------------|
| **入口** | `ChatCompletionRequest` | 验证、分词、模板应用 | - |
| **调度** | `Request` → `ScheduledRequest` | 排队、资源检查、优先级 | 与 ResourceManager 耦合 |
| **资源分配** | `block_tables`, `free_list` | KV Cache 块分配 | 与调度混合在同一类 |
| **任务下发** | `(tasks, batch_size)` | IPC 队列通信 | ZMQ 硬编码依赖 |
| **模型推理** | `InputBatch` → `SamplerOutput` | 前向计算、采样 | 需要 GPU 环境 |
| **输出处理** | `SamplerOutput` → `RequestOutput` | Token 处理、停止检测 | ZMQ 通信依赖 |

### 1.5 Prefill/Decode 双阶段数据流

```
Request 进入系统
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    Prefill 阶段                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  输入: prompt_token_ids (完整提示词)                   │    │
│  │  处理: 一次性计算所有 token 的 KV Cache                │    │
│  │  输出: 第一个生成 token + 填充好的 KV Cache            │    │
│  │                                                       │    │
│  │  seq_lens_encoder = prompt_length                     │    │
│  │  seq_lens_decoder = 0                                 │    │
│  │  task_type = PREFILL                                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Decode 阶段 (循环)                         │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  输入: 上一步生成的 token (单个)                       │    │
│  │  处理: 基于已有 KV Cache，计算下一个 token             │    │
│  │  输出: 新生成的 token                                  │    │
│  │                                                       │    │
│  │  seq_lens_encoder = 0                                 │    │
│  │  seq_lens_decoder = current_decode_length             │    │
│  │  task_type = DECODE                                   │    │
│  │                                                       │    │
│  │  循环直到: EOS token 或达到 max_tokens                 │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 核心观察

### 2.1 当前架构问题（数据量化）

| 模块 | 行数 | 主要问题 |
|------|------|---------|
| EngineService | 2210 | 5 种职责混合、27 处 time.sleep |
| ResourceManagerV1 | 1472 | 调度决策 + 资源管理耦合 |
| GPUModelRunner | 3123 | 平台特定代码耦合 |
| WorkerProcess | 1216 | 分布式环境直接初始化 |

### 2.2 可测试性障碍分类

**障碍 1: 硬编码外部依赖 (28 处)**

```python
# 典型问题代码 (fastdeploy/engine/common_engine.py:1082)
class EngineService:
    def __init__(self, cfg):
        self.recv_server = ZmqIpcServer(...)  # Hardcoded dependency - cannot inject mock
        self.send_server = ZmqIpcServer(...)
```

| 依赖类型 | 影响模块数 | 核心抽象接口 |
|---------|-----------|-------------|
| ZMQ | 9 | `IPCManager` |
| IPC Signal | 6 | `SignalFactory` |
| Redis | 2 | `RedisClientInterface` |
| multiprocessing | 3 | `ProcessFactory` |

**障碍 2: 职责耦合**

```
当前调度层次（职责边界模糊）:
LocalScheduler      →  只是请求队列，名不副实
      ↓
EngineService       →  调度协调 + IPC (270行 _schedule_request_to_worker_v1)
      ↓
ResourceManagerV1   →  调度决策 + 资源分配 (315行 schedule 方法)
```

**障碍 3: 全局状态**

| 模块 | 问题 |
|------|------|
| `api_server.py` | 模块级 `llm_engine = None` |
| `tbo.py` | 模块级 `GLOBAL_THREAD_INFO` |

---

## 3. 设计思路

### 3.1 核心设计原则

| 原则 | 说明 | 应用示例 |
|------|------|---------|
| **职责分离 (SRP)** | 一个类只做一件事 | Scheduler 纯调度，BlockManager 纯资源管理 |
| **依赖注入 (DI)** | 构造函数接受依赖，而非内部创建 | `EngineService(ipc=ipc_manager)` |
| **接口隔离 (ISP)** | 依赖抽象接口，而非具体实现 | 依赖 `IPCManager` 而非 `ZmqIpcServer` |
| **向后兼容** | 默认参数保持原有行为 | `ipc=ipc or IPCManager(cfg)` |

### 3.2 关键设计决策

| 决策点 | 选项 | 决策及理由 |
|--------|------|-----------|
| EngineService 角色 | A: 抽取 SchedulerCoordinator<br>B: EngineService 本身是协调者 | **选 B**: 避免过度抽象，EngineService 本身就是协调者 |
| IPCManager 粒度 | A: 单一 IPCManager<br>B: 拆分 Request/Response Manager | **选 A**: 简化初期实现 |
| 迁移策略 | A: 一次性重构<br>B: 分阶段渐进式 | **选 B**: 小步快跑，降低风险 |

### 3.3 测试边界设计

```
                    测试边界
                       │
  ┌────────────────────┼────────────────────┐
  │   [可 Mock 区域]   │   [真实依赖]       │
  │                    │                    │
  │  MockIPCManager ───┤─── ZmqIpcServer    │
  │  MockBlockManager ─┤─── PrefixCacheManager
  │  MockWorkerQueue ──┤─── EngineWorkerQueue
  └────────────────────┼────────────────────┘
                       │
              EngineService (被测对象)
```

---

## 4. 核心实现方法

### 4.1 分层重构计划

| Phase | 目标 | 风险级别 | 关键文件 |
|-------|------|---------|---------|
| **Phase 1** | WorkerManager + IPCManager 抽取 | 低 | `engine/components/worker_manager.py` |
| **Phase 2** | Scheduler + BlockManager 分离 | 中-高 | `engine/sched/scheduler.py`, `block_manager.py` |
| **Phase 3** | ControlHandler + ZmqCommunicator | 中 | `engine/components/control_handler.py` |

### 4.2 关键接口定义

#### 4.2.1 IPCManager Interface

```python
# fastdeploy/engine/components/ipc_manager.py

class IPCManagerInterface(ABC):
    """
    Inter-Process Communication Manager interface.
    Abstracts ZMQ communication details for testability.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize and start IPC connections."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Gracefully shutdown IPC connections."""
        pass

    @abstractmethod
    def receive_request(self, block: bool = True) -> Tuple[Optional[Exception], Optional[dict]]:
        """
        Receive incoming request from API server.

        Returns:
            Tuple of (error, request_dict).
            - (None, dict) on success
            - (Exception, None) on error
        """
        pass

    @abstractmethod
    def send_response(self, request_id: str, response: List[Any]) -> bool:
        """
        Send response back to API server.

        Returns:
            True if sent successfully, False otherwise.
        """
        pass
```

#### 4.2.2 Scheduler Interface

```python
# fastdeploy/engine/sched/scheduler.py

@dataclass
class ScheduleOutput:
    """
    Output of Scheduler.schedule() - contains scheduling decisions only.
    Does NOT execute block allocation.
    """
    prefills: List['ScheduledPrefillRequest'] = field(default_factory=list)
    decodes: List['ScheduledDecodeRequest'] = field(default_factory=list)
    preempted: List['PreemptedRequest'] = field(default_factory=list)
    errors: List[tuple] = field(default_factory=list)

class Scheduler:
    """
    Pure scheduling decision component. Does NOT execute resource allocation.

    Responsibilities:
    - Maintain waiting/running queues
    - Decide prefill/decode/preempt for each request
    - Calculate required blocks (but NOT allocate)

    NOT responsible for:
    - Block allocation/deallocation (BlockManager's job)
    - IPC communication (EngineService's job)
    """

    def add_request(self, request: 'Request') -> None:
        """Add new request to waiting queue."""

    def schedule(self) -> ScheduleOutput:
        """Execute one scheduling round, return decisions ONLY."""

    def finish_request(self, request_id: str) -> Optional['Request']:
        """Mark request as finished, remove from running queue."""
```

#### 4.2.3 BlockManager Interface

```python
# fastdeploy/engine/sched/block_manager.py

class BlockManager:
    """
    Pure resource management component. Does NOT involve scheduling decisions.

    Responsibilities:
    - Block allocation/deallocation
    - Prefix cache matching
    - Resource availability queries
    """

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if num_blocks can be allocated."""

    def allocate(self, request: 'Request', num_blocks: int) -> List[int]:
        """Allocate blocks for request, returns block IDs."""

    def free(self, request: 'Request') -> None:
        """Free all blocks occupied by request."""

    def match_prefix_cache(self, request: 'Request') -> Tuple[List[int], int]:
        """Match prefix cache, returns (cached_block_ids, matched_token_num)."""
```

### 4.3 迁移策略

#### 环境变量切换机制

```python
# Backward compatible migration with feature flag
class EngineService:
    def __init__(self, cfg, ipc: IPCManager = None):
        if envs.FD_USE_NEW_SCHEDULER:
            self.scheduler = Scheduler(cfg, self.block_manager)
        else:
            self.resource_manager = ResourceManagerV1(...)  # Legacy path

        # Dependency injection with backward compatibility
        self.ipc = ipc or IPCManager(cfg)
```

---

## 5. 核心测试思路和方法

### 5.1 测试策略分层

```
                     测试金字塔
                        /\
                       /  \      E2E 测试 (现有)
                      / E2E\     - 真实 GPU + ZMQ
                     /------\
                    /  集成  \   集成测试 (新增)
                   /----------\  - Mock 外部依赖
                  /   单元测试  \ 单元测试 (新增)
                 /--------------\ - 纯逻辑测试，< 10s
```

### 5.2 Mock 设计原则

| 原则 | 说明 |
|------|------|
| **行为模拟** | Mock 应模拟真实行为，而非仅返回固定值 |
| **状态可控** | 可预设状态用于测试边界条件 |
| **调用验证** | 可验证方法调用次数和参数 |

### 5.3 关键测试用例示例

```python
# tests/engine/sched/test_scheduler.py

class TestScheduler:
    """Unit tests for Scheduler - NO GPU/ZMQ required."""

    def test_add_request_to_waiting(self, scheduler):
        """New request should be added to waiting queue."""
        request = create_mock_request(request_id="req-1")
        scheduler.add_request(request)
        assert scheduler.num_waiting == 1

    def test_schedule_prefill_from_waiting(self, scheduler):
        """Request in waiting queue should be scheduled for prefill."""
        scheduler.add_request(create_mock_request("req-1", num_tokens=100))
        output = scheduler.schedule()
        assert len(output.prefills) == 1
        assert scheduler.num_running == 1

    def test_preemption_when_no_blocks(self, scheduler, mock_block_manager):
        """Should preempt when no blocks available."""
        mock_block_manager.set_available_blocks(0)
        output = scheduler.schedule()
        assert len(output.preempted) >= 1
```

### 5.4 测试覆盖目标

| 模块 | 当前覆盖率 | 目标覆盖率 |
|------|-----------|-----------|
| Scheduler | 0% (新增) | > 90% |
| BlockManager | 0% (新增) | > 85% |
| IPCManager | 0% (新增) | > 80% |
| EngineService | ~20% | > 60% |

---

## 6. 竞品对比

### 6.1 架构对比表

| 组件/特性 | **FastDeploy (当前)** | **vLLM** | **SGLang** |
|----------|----------------------|----------|-----------|
| **Scheduler** | 不存在独立组件，逻辑在 ResourceManagerV1 | 纯调度决策，约 800 行 | Router 异步处理 |
| **BlockManager** | 与调度混合在 ResourceManagerV1 | 独立组件，约 600 行 | TokenPool 概念 |
| **Engine** | EngineService 2210 行，5 种职责 | LLMEngine 协调者角色 | Engine 解耦良好 |
| **可测试性** | 难以单元测试 | 良好的接口抽象 | 组件解耦便于测试 |

### 6.2 借鉴的设计点

| 借鉴来源 | 设计点 | 在本方案中的应用 |
|---------|--------|-----------------|
| vLLM | Scheduler 纯调度决策 | `Scheduler.schedule()` 返回 `ScheduleOutput`，不执行分配 |
| vLLM | BlockManager 纯资源管理 | 从 ResourceManagerV1 抽取独立 BlockManager |
| vLLM | SchedulerOutputs 数据类 | 定义 `ScheduledPrefillRequest` 等结构化输出 |
| SGLang | 组件解耦 | 依赖注入 + 接口隔离 |

---

## 7. 验证方案

### 7.1 成功指标

| 指标 | 基线 | 目标 | 验证方法 |
|------|------|------|---------|
| 单元测试执行时间 | N/A | < 10 秒 | CI 计时 |
| 核心模块覆盖率 | ~20% | > 70% | pytest-cov |
| 无 GPU 可测试模块数 | 0 | 4 | 测试标记 |
| 吞吐量 | 基线 | 下降 < 5% | benchmark |

### 7.2 回滚计划

```bash
# 环境变量切换回滚
export FD_USE_NEW_SCHEDULER=0
export FD_USE_LEGACY_ENGINE_SERVICE=1
```

---

## 8. 关键文件

| 文件 | 说明 |
|------|------|
| `fastdeploy/engine/common_engine.py` | EngineService 核心重构目标 (2210行) |
| `fastdeploy/engine/sched/resource_manager_v1.py` | 调度+资源分离源文件 (1472行) |
| `fastdeploy/engine/components/ipc_signal_manager.py` | 已有组件，可作为参考模式 |
| `fastdeploy/inter_communicator/zmq_server.py` | ZMQ 实现，IPCManager 需封装 |
| `tests/engine/test_common_engine.py` | 现有测试，展示 Mock 模式 |

---

## 9. 实施计划

| 阶段 | 时间 | 交付物 | 验收标准 |
|------|------|--------|---------|
| Phase 1 | Week 1-2 | WorkerManager + IPCManager | Mock 测试通过 |
| Phase 2 | Week 3-5 | Scheduler + BlockManager | 调度测试通过，性能无回归 |
| Phase 3 | Week 6-7 | ControlHandler + ZmqCommunicator | 控制命令测试通过 |
| 收尾 | Week 8 | 文档更新，CI 集成 | 覆盖率 > 70% |
