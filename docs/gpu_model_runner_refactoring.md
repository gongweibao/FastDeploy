# GPUModelRunner 重构方案

## 问题概述

`gpu_model_runner.py` 模块当前过于庞大，存在以下问题：

| 问题 | 说明 |
|------|------|
| 3123 行单一类 | 代码量过大，难以维护 |
| ~50 个方法 | 职责混杂，违反单一职责原则 |
| 流程和组件混合 | `execute_model` 既包含流程控制又包含具体逻辑 |
| 难以测试 | 各功能强耦合，难以独立测试 |
| 难以扩展 | 新增功能需要在原有类中添加代码 |

---

## 当前方法归类分析

```
┌─────────────────────────────────────────────────────────────┐
│                    GPUModelRunner (当前)                     │
├─────────────────────────────────────────────────────────────┤
│ 1. 生命周期/初始化 (10个方法)                                │
│    __init__, load_model, initialize_kv_cache,               │
│    _init_speculative_proposer, _init_logits_processor,      │
│    _init_image_preprocess, _initialize_attn_backend,        │
│    vision_encoder_compile, initialize_forward_meta           │
│                                                              │
│ 2. 输入/任务管理 (9个方法)                                   │
│    insert_tasks_v1, insert_prefill_inputs, _prepare_inputs,  │
│    get_input_length_list, _dummy_prefill_inputs,            │
│    _process_reorder, _process_mm_features,                  │
│    _preprocess_mm_task, _get_feature_positions              │
│                                                              │
│ 3. 模型执行/核心流程 (6个方法) ⭐ 核心流程                      │
│    execute_model, execute_model_normal,                     │
│    execute_model_overlap,                                   │
│    _preprocess_and_execute_model, _execute_empty_input       │
│                                                              │
│ 4. 输出/后处理 (6个方法)                                     │
│    _postprocess, _save_model_output, _pool,                 │
│    _get_prompt_logprobs_list,                               │
│    _get_p_done_idxs_gd, _async_output_busy_loop            │
│                                                              │
│ 5. 视觉/多模态 (6个方法) ⭐ 可独立组件                       │
│    extract_vision_features, extract_vision_features_ernie,  │
│    extract_vision_features_qwen, extract_vision_features_paddleocr, │
│    _dummy_run_extract_vision_features, prepare_rope3d       │
│                                                              │
│ 6. Profile/Dummy Runs (8个方法)                              │
│    profile_run, _dummy_run, _dummy_sampler_run,             │
│    _dummy_pooler_run, _dummy_pooler_run_task,              │
│    sot_warmup, capture_model, capture_model_prefill_and_mixed│
│                                                              │
│ 7. 状态/Cache 管理 (7个方法)                                 │
│    clear_cache, clear_parameters, clear_requests,            │
│    update_parameters, update_weights,                        │
│    update_share_input_block_num, cal_theortical_kvcache      │
│                                                              │
│ 8. 状态查询 (8个方法)                                        │
│    exist_prefill, exist_decode, only_prefill, only_decode,  │
│    not_need_stop, get_model, get_supported_pooling_tasks,   │
│    collect_distributed_status                                 │
│                                                              │
│ 9. 优化 (2个方法)                                            │
│    _update_chunked_prefill, padding_cudagraph_inputs        │
└─────────────────────────────────────────────────────────────┘
```

---

## 重构方案：组件化 + 流程分离

### 目标架构

```
fastdeploy/worker/
├── gpu_model_runner.py           # 精简后的协调器 (~300 行)
├── components/
│   ├── input_manager.py          # 输入管理组件
│   ├── vision_processor.py       # 视觉处理组件
│   ├── output_handler.py         # 输出处理组件
│   ├── cache_manager.py          # KV Cache 管理组件
│   └── profile_runner.py        # Profile 组件
└── flows/
    ├── inference_flow.py         # 推理流程编排
    └── initialization_flow.py   # 初始化流程编排
```

### 核心概念

| 层级 | 职责 | 特点 |
|------|------|------|
| **组件** | 负责具体的功能实现 | 持有状态，可独立测试 |
| **流程** | 负责编排，协调组件 | 无状态或轻量状态 |
| **Runner** | 作为协调器，管理生命周期 | 管理组件实例，提供统一接口 |

---

## 组件设计详情

### 1. InputManager - 输入管理组件

```python
# components/input_manager.py

class InputManager:
    """
    处理请求插入、输入准备、重排序

    职责：
    - 管理请求的插入和排队
    - 准备模型输入
    - 处理请求重排序
    """

    def __init__(self, fd_config: FDConfig):
        self.fd_config = fd_config
        self.share_inputs = {}

    def insert_tasks(self, req_dicts: List[Request], num_running_requests: int = None):
        """插入新的推理任务"""
        pass

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int = None):
        """插入 prefill 阶段的输入"""
        pass

    def prepare_inputs(self, last_token_num=-1, is_dummy_or_profile_run=False) -> None:
        """准备模型输入，处理 padding、offset 等"""
        pass

    def process_reorder(self) -> None:
        """处理请求重排序"""
        pass

    def get_input_length_list(self) -> List[int]:
        """获取输入长度列表"""
        pass

    def process_mm_features(self, request_list: List[Request]):
        """处理多模态特征"""
        pass
```

**迁移的方法：**
- `insert_tasks_v1`
- `insert_prefill_inputs`
- `_prepare_inputs`
- `_process_reorder`
- `_process_mm_features`
- `_preprocess_mm_task`
- `_get_feature_positions`
- `get_input_length_list`
- `_dummy_prefill_inputs`

#### insert_tasks_v1 vs insert_prefill_inputs

| 维度 | `insert_tasks_v1` | `insert_prefill_inputs` |
|------|-------------------|------------------------|
| **使用条件** | `ENABLE_V1_KVCACHE_SCHEDULER=1` | `ENABLE_V1_KVCACHE_SCHEDULER=0` (V0模式) |
| **调用的调度器** | V1 KV Cache Scheduler | V0 Scheduler |
| **支持的任务类型** | Prefill + Decode 两种任务 | 主要处理 Prefill 任务 |
| **任务类型判断** | 根据 `RequestType.PREFILL` 判断 | 根据 `disaggregate_info["role"]` 判断 |
| **分片逻辑** | 使用 `prefill_start_index/prefill_end_index` | 使用 `prefill_chunk_info` 或整体输入 |
| **支持 disaggregated** | 不支持 | 支持 prefill/decode 分离节点 |
| **代码位置** | L693-L902 | L904-L1123 |

**架构背景：**

```
                    Scheduler
                        │
         ┌──────────────┴──────────────┐
         │                             │
    V1 Scheduler                 V0 Scheduler
         │                             │
         ▼                             ▼
  insert_tasks_v1            insert_prefill_inputs
         │                             │
         └──────────────┬──────────────┘
                        ▼
                  share_inputs buffer
```

**关键差异说明：**

1. **调度器版本不同**：两个方法分别服务于不同版本的 KV Cache 调度器
2. **任务支持**：V1 版本可以同时处理 prefill 和 decode 任务，V0 主要处理 prefill
3. **分片策略**：V1 使用 start/end 索引，V0 使用 chunk_info 配合 enable_chunked_prefill 配置
4. **Disaggregated 支持**：V0 支持 disaggregated 场景（prefill/decode 分离节点），可以处理来自 decode 节点的请求

**重构建议：**

这两个方法职责相近但实现不同，在重构时可以考虑统一接口：

```python
class InputManager:
    def insert_tasks(self, req_dicts, use_v1_scheduler=False):
        """统一的任务插入接口"""
        if use_v1_scheduler:
            return self._insert_tasks_v1(req_dicts)
        else:
            return self._insert_tasks_v0(req_dicts)

    def _insert_tasks_v1(self, req_dicts):
        """V1 调度器的任务插入逻辑"""
        # 原有 insert_tasks_v1 的实现
        pass

    def _insert_tasks_v0(self, req_dicts):
        """V0 调度器的任务插入逻辑"""
        # 原有 insert_prefill_inputs 的实现
        pass
```

这样可以**统一接口，内部实现分离**，减少代码重复，同时保持与不同调度器的兼容性。

#### _prepare_inputs vs _process_mm_features

| 维度 | `_process_mm_features` | `_prepare_inputs` |
|------|----------------------|------------------|
| **调用时机** | 任务插入时（`insert_tasks_v1` 中） | 模型执行前（`_preprocess_and_execute_model` 中） |
| **主要职责** | 提取视觉特征并缓存 | 准备所有模型输入数据 |
| **是否调用模型** | ✅ 调用 vision encoder 提取特征 | ❌ 不调用模型，只是数据准备 |
| **处理内容** | 图像/视频输入 | 所有输入（tokens、vision features、采样参数等） |
| **输出位置** | `share_inputs["image_features_list"]` | `share_inputs` 的多个字段 |
| **依赖关系** | 独立执行 | 依赖 `_process_mm_features` 的结果 |

**调用流程：**

```
任务插入阶段:
    insert_tasks_v1 / insert_prefill_inputs
            │
            ▼
    _process_mm_features  ← 提取并缓存视觉特征
            │
            ▼
      (任务排队等待...)

模型执行前:
    _preprocess_and_execute_model
            │
            ▼
    _prepare_inputs  ← 准备所有模型输入
            │
            ├─> 使用缓存的 vision_features
            ├─> 移除 padding (pre_process)
            ├─> 初始化 forward meta
            └─> 获取 sampling metadata
            │
            ▼
       模型推理
```

**关键代码：**

```python
# insert_tasks_v1 中调用（L896）
self._process_mm_features(req_dicts)  # 先提取视觉特征

# _prepare_inputs 中使用（L1246-1249）
if self.enable_mm and self.share_inputs["image_features_list"] is not None:
    tensor_feats = [t for t in self.share_inputs["image_features_list"]
                   if isinstance(t, paddle.Tensor)]
    if tensor_feats:
        # 使用之前缓存的视觉特征
        self.share_inputs["image_features"] = paddle.concat(tensor_feats, axis=0)
```

**设计意图：**

1. **性能优化**：视觉特征提取是耗时操作，提前并行处理
2. **缓存复用**：不同请求可能有相同图像，提取一次后复用
3. **职责分离**：特征提取是模型相关操作，输入准备是数据整理操作

**组件归属：**

在重构时，这两个方法应该归属于不同的组件：

```python
# VisionProcessor 负责特征提取
class VisionProcessor:
    def process_mm_features(self, request_list):
        """提取并缓存视觉特征"""
        ...

# InputManager 负责输入准备（使用已缓存的特征）
class InputManager:
    def prepare_inputs(self, last_token_num=-1, is_dummy_or_profile_run=False):
        """准备所有模型输入（包括使用缓存的视觉特征）"""
        ...
```

---

### 2. VisionProcessor - 视觉处理组件

```python
# components/vision_processor.py

class VisionProcessor:
    """
    处理多模态视觉特征提取

    职责：
    - 处理图像输入
    - 提取视觉特征
    - 准备 3D RoPE 位置编码
    """

    def __init__(self, fd_config: FDConfig, model):
        self.fd_config = fd_config
        self.model = model
        self._init_image_preprocess()

    def process_mm_features(self, request_list: List[Request]):
        """处理并缓存视觉特征"""
        pass

    def extract_vision_features(self, multi_vision_inputs: Dict) -> paddle.Tensor:
        """提取视觉特征（统一入口）"""
        pass

    def extract_vision_features_ernie(self, vision_inputs) -> paddle.Tensor:
        """Ernie 模型的视觉特征提取"""
        pass

    def extract_vision_features_qwen(self, vision_inputs) -> paddle.Tensor:
        """Qwen 模型的视觉特征提取"""
        pass

    def extract_vision_features_paddleocr(self, inputs) -> paddle.Tensor:
        """PaddleOCR 的视觉特征提取"""
        pass

    def prepare_rope3d(self, vision_position_ids: Dict, batch_size: int):
        """准备 3D RoPE 位置编码"""
        pass

    def _preprocess_mm_task(self, one: dict) -> None:
        """预处理多模态任务"""
        pass

    def dummy_run_extract_vision_features(self):
        """视觉特征的 dummy run 用于 profiling"""
        pass
```

**迁移的方法：**
- `_process_mm_features`
- `extract_vision_features`
- `extract_vision_features_ernie`
- `extract_vision_features_qwen`
- `extract_vision_features_paddleocr`
- `prepare_rope3d`
- `_preprocess_mm_task`
- `_dummy_run_extract_vision_features`
- `_init_image_preprocess`
- `vision_encoder_compile`

---

### 3. OutputHandler - 输出处理组件

```python
# components/output_handler.py

class OutputHandler:
    """
    处理后处理和输出保存

    职责：
    - 后处理采样结果
    - 保存模型输出
    - 处理 pooling 输出
    - 异步输出循环
    """

    def __init__(self, fd_config: FDConfig):
        self.fd_config = fd_config
        self.async_output_queue = queue.Queue()
        self._start_async_output_thread()

    def postprocess(self, sampler_output, model_output, share_inputs, ...):
        """后处理步骤：stop flag、更新输入等"""
        pass

    def save_model_output(self, model_output, sampler_output, share_inputs, ...):
        """保存模型输出"""
        pass

    def pool(self, hidden_states: paddle.Tensor, num_running_requests: int) -> Optional[ModelRunnerOutput]:
        """处理 pooling 任务"""
        pass

    def get_prompt_logprobs_list(self, model_output: ModelOutputData):
        """获取 prompt logprobs"""
        pass

    def _async_output_busy_loop(self):
        """异步输出循环"""
        pass

    def _get_p_done_idxs_gd(self, model_forward_batch, num_running_requests):
        """获取已完成的请求索引（guided decoding）"""
        pass
```

**迁移的方法：**
- `_postprocess`
- `_save_model_output`
- `_pool`
- `_get_prompt_logprobs_list`
- `_get_p_done_idxs_gd`
- `_async_output_busy_loop`

---

### 4. CacheManager - KV Cache 管理组件

```python
# components/cache_manager.py

class CacheManager:
    """
    管理 KV Cache 初始化和状态

    职责：
    - 初始化 KV Cache
    - 管理 GPU blocks
    - 清理和重置 cache
    """

    def __init__(self, fd_config: FDConfig, model):
        self.fd_config = fd_config
        self.model = model
        self.total_block_num = 0

    def initialize_kv_cache(self, profile: bool = False) -> None:
        """初始化 KV Cache"""
        pass

    def clear_cache(self, profile=False):
        """清理 KV Cache"""
        pass

    def update_share_input_block_num(self, num_gpu_blocks: int):
        """更新共享输入的 block 数量"""
        pass

    def cal_theortical_kvcache(self):
        """计算理论 KV Cache 需求"""
        pass
```

**迁移的方法：**
- `initialize_kv_cache`
- `clear_cache`
- `update_share_input_block_num`
- `cal_theortical_kvcache`

---

### 5. ProfileRunner - Profile 组件

```python
# components/profile_runner.py

class ProfileRunner:
    """
    处理 profile 和 dummy run

    职责：
    - 执行 profile run
    - 执行 dummy run 用于 warmup
    - 模型捕获和编译
    """

    def __init__(self, fd_config: FDConfig, model):
        self.fd_config = fd_config
        self.model = model

    def profile_run(self) -> None:
        """执行 profile run"""
        pass

    def dummy_run(self):
        """执行 dummy run"""
        pass

    def dummy_sampler_run(self, ...):
        """Sampler 的 dummy run"""
        pass

    def dummy_pooler_run(self, ...):
        """Pooler 的 dummy run"""
        pass

    def sot_warmup(self) -> None:
        """SOT warmup"""
        pass

    def capture_model(self) -> None:
        """捕获模型用于 cudagraph"""
        pass

    def capture_model_prefill_and_mixed(self) -> None:
        """捕获 prefill 和 mixed 模式"""
        pass
```

**迁移的方法：**
- `profile_run`
- `_dummy_run`
- `_dummy_sampler_run`
- `_dummy_pooler_run`
- `_dummy_pooler_run_task`
- `sot_warmup`
- `capture_model`
- `capture_model_prefill_and_mixed`

---

### 6. InferenceFlow - 推理流程编排

```python
# flows/inference_flow.py

class InferenceFlow:
    """
    推理流程的编排层（纯流程，无状态）

    职责：
    - 编排推理的完整流程
    - 协调各个组件的调用顺序
    - 处理不同的执行模式
    """

    def __init__(self, model):
        self.model = model

    def execute(self, model_forward_batch: List[Request],
               input_mgr: InputManager,
               vision_proc: VisionProcessor,
               output_hdlr: OutputHandler,
               **kwargs) -> ModelRunnerOutput:
        """主执行入口"""
        # 1. 准备输入
        input_mgr.prepare_inputs(...)

        # 2. 处理多模态特征
        vision_proc.process_mm_features(...)

        # 3. 执行模型推理
        outputs = self._run_model(...)

        # 4. 后处理
        output_hdlr.postprocess(...)

        return outputs

    def execute_normal(self, ...):
        """正常模式执行"""
        pass

    def execute_overlap(self, ...):
        """Overlap 模式执行"""
        pass

    def _preprocess_and_execute_model(self, ...):
        """预处理并执行模型"""
        pass

    def _execute_empty_input(self, forward_meta):
        """执行空输入处理"""
        pass
```

**迁移的方法：**
- `execute_model`
- `execute_model_normal`
- `execute_model_overlap`
- `_preprocess_and_execute_model`
- `_execute_empty_input`

#### 为什么有了 InferenceFlow 还需要 GPUModelRunner？

| 维度 | GPUModelRunner | InferenceFlow |
|------|---------------|---------------|
| **拥有模型** | ✅ 持有 model 实例 | ❌ 只接受 model 参数 |
| **组件生命周期** | ✅ 创建和管理所有组件 | ❌ 不管理组件 |
| **配置管理** | ✅ 持有 FDConfig | ❌ 依赖传入的配置 |
| **初始化** | ✅ KV Cache、Attn Backend 等 | ❌ 不负责初始化 |
| **外部接口** | ✅ 提供给外部调用 | ❌ 内部使用 |
| **状态查询** | ✅ exist_prefill/decode 等 | ❌ 无状态 |
| **Profile/Warmup** | ✅ 负责 | ❌ 不负责 |
| **推理编排** | ⏸️ 委托给 InferenceFlow | ✅ 负责 |

**架构模式：Manager + Orchestrator**

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPUModelRunner                        │
│                    (Manager/Context)                         │
│                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 生命周期管理                                            │   │
│  │ • __init__ - 创建所有组件                               │   │
│  │ • load_model() - 加载模型并初始化依赖组件               │   │
│  │ • initialize_kv_cache() - 初始化 KV Cache                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 状态管理                                                │   │
│  │ • self.model - 持有模型实例                            │   │
│  │ • self.share_inputs - 共享输入缓冲区                     │   │
│  │ • self.encoder_cache - 编码器缓存                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 外部接口层                                              │   │
│  │ • execute_model() - 外部调用入口                          │   │
│  │ • exist_prefill() - 状态查询                            │   │
│  │ • clear_cache() - cache 管理                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼ 委托                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              InferenceFlow (Orchestrator)               │   │
│  │                                                            │
│  │  • execute() - 纯流程编排，单次推理的逻辑组织          │   │
│  │  • 无状态或极轻量状态                                   │   │
│  │  • 协调组件调用顺序                                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**关键区别：**

**GPUModelRunner（有状态的管理器）：**
- 拥有并管理所有组件
- 拥有模型实例和状态
- 提供稳定的对外接口
- 管理资源生命周期

**InferenceFlow（无状态的编排器）：**
- 只持有 model 引用，不拥有它
- 纯流程编排，每次调用都是独立的
- 不管理状态和资源

**为什么要保留 GPUModelRunner？**

1. **外部接口稳定性** - 外部调用者依赖 `GPUModelRunner` 的接口，不能直接暴露内部组件
2. **配置和状态管理** - 需要一个地方统一管理配置、状态和资源
3. **生命周期控制** - 模型和组件的创建、初始化、销毁需要协调
4. **向后兼容** - 保持现有 API 不变

**总结**：`GPUModelRunner` 是**上下文/管理器**，`InferenceFlow` 是**编排器**。前者负责"有什么"（资源、状态），后者负责"怎么做"（执行流程）。

---

### 7. GPUModelRunner - 精简后的协调器

```python
# gpu_model_runner.py (重构后 ~300 行)

class GPUModelRunner(ModelRunnerBase):
    """
    模型推理协调器

    职责：
    - 管理各个组件的生命周期
    - 提供统一的执行接口
    - 处理状态查询
    """

    def __init__(self, fd_config: FDConfig, device: str, ...):
        super().__init__(fd_config, device)

        # 初始化各个组件
        self.input_manager = InputManager(fd_config)
        self.vision_processor = None  # load_model 后初始化
        self.output_handler = OutputHandler(fd_config)
        self.cache_manager = None     # load_model 后初始化
        self.profile_runner = None    # load_model 后初始化
        self.inference_flow = None    # load_model 后初始化

        # 初始化其他组件
        self._init_speculative_proposer()
        self._init_logits_processor()

    def load_model(self) -> None:
        """加载模型并初始化依赖组件"""
        model = self._get_model_instance()

        # 初始化依赖 model 的组件
        self.vision_processor = VisionProcessor(self.fd_config, model)
        self.cache_manager = CacheManager(self.fd_config, model)
        self.profile_runner = ProfileRunner(self.fd_config, model)
        self.inference_flow = InferenceFlow(model)

        self.model = model

    def execute_model(self, model_forward_batch: Optional[List[Request]], **kwargs) -> ModelRunnerOutput:
        """执行模型推理（委托给 InferenceFlow）"""
        return self.inference_flow.execute(
            model_forward_batch,
            input_mgr=self.input_manager,
            vision_proc=self.vision_processor,
            output_hdlr=self.output_handler,
            **kwargs
        )

    # ========== 状态查询方法 ==========
    def exist_prefill(self) -> bool:
        """检查是否存在 prefill 阶段"""
        return self.input_manager.exist_prefill()

    def exist_decode(self) -> bool:
        """检查是否存在 decode 阶段"""
        return self.input_manager.exist_decode()

    def only_prefill(self) -> bool:
        """检查是否仅 prefill"""
        return self.input_manager.only_prefill()

    def only_decode(self) -> bool:
        """检查是否仅 decode"""
        return self.input_manager.only_decode()

    def not_need_stop(self) -> bool:
        """检查是否不需要停止"""
        return self.output_handler.not_need_stop()

    def get_model(self) -> nn.Layer:
        """获取模型"""
        return self.model

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """获取支持的 pooling 任务"""
        return self.output_handler.get_supported_pooling_tasks()

    # ========== Cache 管理方法 ==========
    def clear_cache(self, profile=False):
        """清理 cache（委托给 CacheManager）"""
        self.cache_manager.clear_cache(profile)

    def update_share_input_block_num(self, num_gpu_blocks: int):
        """更新 block 数量（委托给 CacheManager）"""
        self.cache_manager.update_share_input_block_num(num_gpu_blocks)

    # ========== Profile 方法 ==========
    def profile_run(self) -> None:
        """执行 profile（委托给 ProfileRunner）"""
        self.profile_runner.profile_run()

    def sot_warmup(self) -> None:
        """SOT warmup（委托给 ProfileRunner）"""
        self.profile_runner.sot_warmup()

    # ========== 其他组件委托方法 ==========
    def capture_model(self) -> None:
        """捕获模型"""
        self.profile_runner.capture_model()

    def clear_requests(self):
        """清理请求"""
        self.input_manager.clear_requests()

    def update_parameters(self, pid):
        """更新参数"""
        # 委托给相应的组件
        pass

    def update_weights(self, version: str = None, rsync_config: Dict[str, Any] = None):
        """更新权重"""
        # 委托给相应的组件
        pass

    # ========== 辅助方法（留在 Runner 中） ==========
    def _init_speculative_proposer(self):
        """初始化 speculative proposer"""
        pass

    def _init_logits_processor(self, request) -> tuple[Future[LogitsProcessorBase],]:
        """初始化 logits processor"""
        pass

    def _initialize_attn_backend(self) -> None:
        """初始化 attention backend"""
        pass

    def initialize_forward_meta(self, is_dummy_or_profile_run=False):
        """初始化 forward meta"""
        pass

    def collect_distributed_status(self):
        """收集分布式状态"""
        pass

    def padding_cudagraph_inputs(self) -> None:
        """Padding cudagraph 输入"""
        pass

    def _update_chunked_prefill(self, tasks):
        """更新 chunked prefill 状态"""
        pass
```

---

## 重构前后代码对比

### 重构前：3123 行

```python
class GPUModelRunner(ModelRunnerBase):
    # 所有方法都在一个类中
    def __init__(self, fd_config, device, ...):
        # 大量初始化代码
        ...

    def insert_tasks_v1(self, req_dicts, num_running_requests=None):
        # 输入管理逻辑
        ...

    def _process_mm_features(self, request_list):
        # 视觉处理逻辑
        ...

    def extract_vision_features(self, vision_inputs):
        # 具体的视觉特征提取
        ...

    def extract_vision_features_ernie(self, vision_inputs):
        # Ernie 特定逻辑
        ...

    def execute_model(self, model_forward_batch, ...):
        # 流程控制 + 具体实现
        self._prepare_inputs(...)
        self._process_mm_features(...)
        outputs = self._run_model(...)
        self._postprocess(...)
        self._save_model_output(...)
        ...

    def _postprocess(self, sampler_output, model_output, ...):
        # 后处理逻辑
        ...

    # ... 还有 40+ 个方法
```

### 重构后：~300 行 Runner + 各组件

```python
# gpu_model_runner.py
class GPUModelRunner(ModelRunnerBase):
    def __init__(self, fd_config, device, ...):
        # 初始化组件
        self.input_manager = InputManager(fd_config)
        self.output_handler = OutputHandler(fd_config)
        # ...

    def execute_model(self, model_forward_batch, ...):
        # 纯协调，委托给 InferenceFlow
        return self.inference_flow.execute(
            model_forward_batch,
            input_mgr=self.input_manager,
            vision_proc=self.vision_processor,
            output_hdlr=self.output_handler,
            ...
        )

# flows/inference_flow.py
class InferenceFlow:
    def execute(self, batch, input_mgr, vision_proc, output_hdlr, ...):
        # 纯流程编排，不包含具体实现
        input_mgr.prepare_inputs(batch)
        vision_proc.process_features(batch)
        outputs = self._run_model(batch)
        output_hdlr.postprocess(outputs)
        return outputs

# components/vision_processor.py
class VisionProcessor:
    def extract_vision_features(self, inputs):
        # 具体实现
        ...
```

---

## 重构步骤

### 第一阶段：提取独立组件（风险低）

**目标**：将职责明确、依赖简单的功能提取为独立组件

| 组件 | 文件 | 迁移方法 | 预估工作量 |
|------|------|----------|------------|
| VisionProcessor | `components/vision_processor.py` | 视觉处理相关方法 | 2-3 天 |
| OutputHandler | `components/output_handler.py` | 输出后处理方法 | 2-3 天 |
| ProfileRunner | `components/profile_runner.py` | Profile/dummy run 方法 | 2-3 天 |

**验收标准**：
- 组件可独立测试
- 原有功能保持不变
- 性能无明显下降

---

### 第二阶段：提取管理器（风险中等）

**目标**：将状态管理相关功能提取为独立管理器

| 组件 | 文件 | 迁移方法 | 预估工作量 |
|------|------|----------|------------|
| InputManager | `components/input_manager.py` | 输入/任务管理方法 | 3-4 天 |
| CacheManager | `components/cache_manager.py` | KV Cache 管理方法 | 2-3 天 |

**验收标准**：
- 组件间接口清晰
- 状态流转正确
- 并发安全性保证

---

### 第三阶段：流程编排层（风险高）

**目标**：将 execute_model 重构为流程编排

| 组件 | 文件 | 迁移方法 | 预估工作量 |
|------|------|----------|------------|
| InferenceFlow | `flows/inference_flow.py` | 模型执行流程方法 | 4-5 天 |
| InitializationFlow | `flows/initialization_flow.py` | 初始化流程方法 | 2-3 天 |

**验收标准**：
- 流程清晰，易于理解
- 组件调用顺序正确
- 错误处理完善

---

### 第四阶段：GPUModelRunner 精简

**目标**：Runner 作为协调器，只保留生命周期和状态查询

**操作**：
1. 删除已迁移的方法
2. 添加组件委托方法
3. 保留必要的状态查询方法
4. 更新文档

**预估工作量**：2-3 天

---

## 关键原则

### 1. 组件职责

- **负责具体的功能实现**
- **持有状态**
- **可独立测试**
- **接口稳定**

### 2. 流程职责

- **负责编排，协调组件**
- **无状态或轻量状态**
- **清晰的调用顺序**
- **易于理解和修改**

### 3. Runner 职责

- **管理组件生命周期**
- **提供统一的执行接口**
- **处理状态查询**
- **保持向后兼容**

---

## 注意事项

### 1. 性能考虑

- 组件间调用避免不必要的拷贝
- 共享数据结构（如 InputBatch、KV Cache）的访问需要线程安全
- 异步输出队列保持现有机制

### 2. 向后兼容

- Runner 对外接口保持不变
- 逐步迁移，避免大爆炸式改动
- 充分的单元测试和集成测试

### 3. 测试策略

- 每个组件独立测试
- 组件集成测试
- 端到端功能测试
- 性能基准测试

---

## 预期收益

| 维度 | 收益 |
|------|------|
| 可维护性 | 代码结构清晰，各组件职责单一 |
| 可测试性 | 组件可独立测试，测试覆盖率提高 |
| 可扩展性 | 新增功能只需修改对应组件 |
| 可读性 | 流程和组件分离，逻辑清晰 |
| 性能 | 组件化后可针对性优化 |

---

## 附录：完整方法迁移清单

### VisionProcessor (10 个方法)
- `_init_image_preprocess`
- `vision_encoder_compile`
- `_process_mm_features`
- `_preprocess_mm_task`
- `_get_feature_positions`
- `extract_vision_features`
- `extract_vision_features_ernie`
- `extract_vision_features_qwen`
- `extract_vision_features_paddleocr`
- `_dummy_run_extract_vision_features`

### OutputHandler (6 个方法)
- `_postprocess`
- `_save_model_output`
- `_pool`
- `_get_prompt_logprobs_list`
- `_get_p_done_idxs_gd`
- `_async_output_busy_loop`

### ProfileRunner (8 个方法)
- `profile_run`
- `_dummy_run`
- `_dummy_sampler_run`
- `_dummy_pooler_run`
- `_dummy_pooler_run_task`
- `_dummy_prefill_inputs`
- `sot_warmup`
- `capture_model`
- `capture_model_prefill_and_mixed`

### InputManager (9 个方法)
- `insert_tasks_v1`
- `insert_prefill_inputs`
- `_prepare_inputs`
- `get_input_length_list`
- `_process_reorder`
- `clear_requests`

### CacheManager (4 个方法)
- `initialize_kv_cache`
- `clear_cache`
- `update_share_input_block_num`
- `cal_theortical_kvcache`

### InferenceFlow (5 个方法)
- `execute_model`
- `execute_model_normal`
- `execute_model_overlap`
- `_preprocess_and_execute_model`
- `_execute_empty_input`

### GPUModelRunner (保留 ~15 个方法)
- `__init__`
- `load_model`
- `get_model`
- 状态查询方法（exist_prefill, exist_decode, only_prefill, only_decode）
- 组件委托方法（execute_model, clear_cache, profile_run 等）
- 辅助方法（_init_speculative_proposer, _init_logits_processor 等）
