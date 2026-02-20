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
    - 协调视觉特征提取（通过 VisionProcessor）
    """

    def __init__(self, fd_config: FDConfig, vision_processor: 'VisionProcessor' = None):
        self.fd_config = fd_config
        self.share_inputs = {}
        self.vision_processor = vision_processor  # VisionProcessor 引用，用于任务插入时提取视觉特征

    def set_vision_processor(self, vision_processor: 'VisionProcessor'):
        """设置 VisionProcessor（用于延迟初始化）"""
        self.vision_processor = vision_processor

    def insert_tasks(self, req_dicts: List[Request], num_running_requests: int = None):
        """插入新的推理任务（V1 模式）"""
        # ... 插入逻辑 ...
        # 任务插入时调用 VisionProcessor 提取视觉特征
        if self.vision_processor:
            self.vision_processor.process_mm_features(req_dicts)
        pass

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int = None):
        """插入 prefill 阶段的输入（V0 模式）"""
        # ... 插入逻辑 ...
        # 任务插入时调用 VisionProcessor 提取视觉特征
        if self.vision_processor:
            self.vision_processor.process_mm_features(req_dicts)
        pass

    def prepare_inputs(self, last_token_num=-1, is_dummy_or_profile_run=False) -> None:
        """准备模型输入，处理 padding、offset 等

        注意：使用 VisionProcessor 已缓存的视觉特征，不重新提取
        """
        pass

    def process_reorder(self) -> None:
        """处理请求重排序"""
        pass

    def get_input_length_list(self) -> List[int]:
        """获取输入长度列表"""
        pass
```

**迁移的方法：**
- `insert_tasks_v1`
- `insert_prefill_inputs`
- `_prepare_inputs`
- `_process_reorder`
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

**重构建议（V0/V1 兼容性处理）：**

这两个方法职责相近但实现不同，在重构时采用统一接口模式：

```python
class InputManager:
    def __init__(self, fd_config: FDConfig, vision_processor: 'VisionProcessor' = None):
        self.fd_config = fd_config
        self.share_inputs = {}
        self.vision_processor = vision_processor
        # 根据 envs.ENABLE_V1_KVCACHE_SCHEDULER 决定使用哪个调度器模式
        self.use_v1_scheduler = int(envs.get("ENABLE_V1_KVCACHE_SCHEDULER", 0)) == 1

    def insert_tasks(self, req_dicts: List[Request], num_running_requests: int = None):
        """
        统一的任务插入接口

        内部根据 use_v1_scheduler 自动选择对应的实现
        """
        if self.use_v1_scheduler:
            return self._insert_tasks_v1(req_dicts, num_running_requests)
        else:
            return self._insert_tasks_v0(req_dicts, num_running_requests)

        # 任务插入后调用 VisionProcessor 提取视觉特征
        if self.vision_processor:
            self.vision_processor.process_mm_features(req_dicts)

    def _insert_tasks_v1(self, req_dicts, num_running_requests):
        """V1 调度器的任务插入逻辑"""
        # 原有 insert_tasks_v1 的实现
        pass

    def _insert_tasks_v0(self, req_dicts, num_running_requests):
        """V0 调度器的任务插入逻辑"""
        # 原有 insert_prefill_inputs 的实现
        pass
```

**关键点：**
1. **统一接口**：对外只暴露 `insert_tasks()`，内部自动选择 V0/V1 实现
2. **配置驱动**：通过 `ENABLE_V1_KVCACHE_SCHEDULER` 环境变量控制
3. **视觉特征处理统一**：两种模式都在任务插入时调用 VisionProcessor
4. **向后兼容**：保留原有方法的逻辑，只改变接口层次

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
    - 使用 encoder_cache 进行跨请求特征复用

    注意：encoder_cache 由 GPUModelRunner 管理，通过引用访问
    """

    def __init__(self, fd_config: FDConfig, model, encoder_cache: dict = None):
        self.fd_config = fd_config
        self.model = model
        self.encoder_cache = encoder_cache  # 视觉特征缓存，由 GPUModelRunner 管理
        self._init_image_preprocess()

    def process_mm_features(self, request_list: List[Request]):
        """
        处理并缓存视觉特征

        工作流程：
        1. 检查 encoder_cache 中是否有缓存的特征
        2. 如果没有，提取新特征并缓存
        3. 将特征存储到 share_inputs["image_features_list"] 供后续使用
        """
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
- `_get_feature_positions`
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
    - 模型捕获和编译（CudaGraph 初始化阶段）
    - SOT warmup

    注意：CudaGraph 运行时的 padding 操作（padding_cudagraph_inputs）在 GPUModelRunner 中，
    因为它需要在每次推理前调用，且需要访问 share_inputs
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
        """
        捕获模型用于 cudagraph（初始化阶段）

        在模型加载完成后调用，用于捕获计算图并优化
        """
        pass

    def capture_model_prefill_and_mixed(self) -> None:
        """
        捕获 prefill 和 mixed 模式（初始化阶段）

        用于捕获包含 prefill 和 decode 的混合执行模式
        """
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
               output_hdlr: OutputHandler,
               **kwargs) -> ModelRunnerOutput:
        """主执行入口

        注意：视觉特征处理在任务插入阶段完成，不在此处调用
        """
        # 1. 处理重排序（如果需要）
        input_mgr.process_reorder()

        # 2. 准备输入（包括使用已缓存的 vision_features）
        input_mgr.prepare_inputs(...)

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
    - 管理 encoder_cache（视觉特征缓存）
    - 管理 share_inputs（共享输入缓冲区）
    """

    def __init__(self, fd_config: FDConfig, device: str, ...):
        super().__init__(fd_config, device)

        # 初始化共享状态
        self.share_inputs = InputBatch(self.fd_config)
        self.share_inputs.init_share_inputs()

        # 初始化 encoder_cache（视觉特征缓存）
        if self.cache_config.max_encoder_cache > 0:
            self.encoder_cache: dict[str, paddle.Tensor] = {}
        else:
            self.encoder_cache = None

        # 初始化各个组件
        self.input_manager = InputManager(fd_config)
        self.vision_processor = None  # load_model 后初始化，需要传入 encoder_cache
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
        self.vision_processor = VisionProcessor(self.fd_config, model, self.encoder_cache)
        self.cache_manager = CacheManager(self.fd_config, model)
        self.profile_runner = ProfileRunner(self.fd_config, model)
        self.inference_flow = InferenceFlow(model)

        # 设置 InputManager 的 VisionProcessor 引用
        self.input_manager.set_vision_processor(self.vision_processor)

        self.model = model

    def execute_model(self, model_forward_batch: Optional[List[Request]], **kwargs) -> ModelRunnerOutput:
        """执行模型推理（委托给 InferenceFlow）"""
        return self.inference_flow.execute(
            model_forward_batch,
            input_mgr=self.input_manager,
            output_hdlr=self.output_handler,
            **kwargs
        )

    # ========== 状态查询方法 ==========
    # 状态查询方法的委托原则：
    # - 所有状态查询接口由 GPUModelRunner 统一提供
    # - 内部实现委托给各个组件，但外部调用者不感知
    # - 这样可以保持组件接口稳定，同时便于状态管理

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
        """检查是否不需要停止（委托给 OutputHandler）"""
        return self.output_handler.not_need_stop()

    def get_model(self) -> nn.Layer:
        """获取模型"""
        return self.model

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """获取支持的 pooling 任务（委托给 OutputHandler）"""
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
    # 这些方法留在 GPUModelRunner 中是因为：
    # - 需要访问 share_inputs 或多个组件的状态
    # - 在每次推理循环中调用，不适合委托给独立的组件

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
        """
        Padding cudagraph 输入（运行时操作）

        注意：这个方法留在 GPUModelRunner 中，因为它：
        - 需要访问 share_inputs
        - 在每次推理前调用
        - 与 cudagraph 运行时紧密相关

        cudagraph 的初始化（capture_model）由 ProfileRunner 负责
        """
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
- **视觉特征提取时机**：在任务插入阶段完成，不在每次推理时重新提取

### 2. share_inputs 访问控制模式

`share_inputs`（InputBatch）是多个组件共用的共享数据结构，采用以下访问控制模式：

| 字段 | 写入者 | 读取者 | 说明 |
|------|--------|--------|------|
| `input_ids` | InputManager.prepare_inputs() | Model | 模型输入 ID |
| `image_features_list` | VisionProcessor.process_mm_features() | InputManager.prepare_inputs() | 视觉特征缓存（任务插入时写入） |
| `image_features` | InputManager.prepare_inputs() | Model | 拼接后的视觉特征 |
| `seq_lens_*` | InputManager.prepare_inputs() | OutputHandler | 序列长度信息 |
| `block_tables` | CacheManager | Model | KV Cache 块表 |
| `top_p`, `top_k` | InputManager.insert_tasks() | Sampler | 采样参数 |

**访问原则**：
1. **单一写入者原则**：每个字段只有一个组件负责写入
2. **延迟写入**：输入相关字段在 `prepare_inputs()` 时统一写入
3. **视觉特征例外**：`image_features_list` 在任务插入时由 VisionProcessor 写入，`prepare_inputs()` 只负责读取和拼接

**线程安全**：
- `share_inputs` 是单线程访问（Worker 进程内部）
- 异步输出线程只读取输出数据，不修改 `share_inputs`

### 3. 向后兼容

- Runner 对外接口保持不变
- 逐步迁移，避免大爆炸式改动
- 充分的单元测试和集成测试

### 4. 测试策略

- 每个组件独立测试
- 组件集成测试
- 端到端功能测试
- 性能基准测试

### 5. 组件初始化流程

各组件的初始化顺序和依赖关系：

```
GPUModelRunner.__init__()
    │
    ├─> 初始化共享状态 (share_inputs, encoder_cache)
    ├─> 初始化无依赖组件 (InputManager, OutputHandler)
    │
    └─> 延迟初始化（需要 model 的组件，在 load_model() 中）：
GPUModelRunner.load_model()
    │
    ├─> model = self._get_model_instance()
    │
    ├─> VisionProcessor(model, encoder_cache)  ← 依赖 encoder_cache
    ├─> CacheManager(model)                    ← 依赖 model
    ├─> ProfileRunner(model)                   ← 依赖 model
    ├─> InferenceFlow(model)                   ← 依赖 model
    │
    └─> input_manager.set_vision_processor(vision_processor)  ← 设置引用
```

**初始化依赖关系**：
1. GPUModelRunner: 无依赖，最先初始化
2. InputManager: 无依赖，可立即初始化
3. OutputHandler: 无依赖，可立即初始化
4. VisionProcessor: 依赖 model 和 encoder_cache
5. CacheManager: 依赖 model
6. ProfileRunner: 依赖 model
7. InferenceFlow: 依赖 model

### 6. 错误处理机制

组件化后的错误处理策略：

**组件内部错误处理**：
```python
class InputManager:
    def insert_tasks(self, req_dicts, ...):
        try:
            # 处理逻辑
            pass
        except Exception as e:
            logger.error(f"InputManager.insert_tasks failed: {e}")
            # 清理已修改的状态
            self._cleanup_partial_insert()
            raise
```

**跨组件错误传播**：
```python
class GPUModelRunner:
    def execute_model(self, batch, ...):
        try:
            return self.inference_flow.execute(batch, ...)
        except Exception as e:
            logger.error(f"Execute model failed: {e}")
            # 统一的错误处理和资源清理
            self._handle_execution_error(e)
            raise
```

**错误处理原则**：
1. **组件边界错误隔离**：每个组件负责处理自己的内部错误
2. **状态回滚**：组件出错时清理已修改的状态
3. **统一日志**：使用统一的日志格式便于问题追踪
4. **资源清理**：确保异常发生时释放已分配的资源
5. **错误信息传播**：关键错误信息需要传递给上层调用者

---

## 业界最佳实践参考

基于对 sglang 和 vLLM 两个成熟项目的分析，以下是可以参考的架构模式：

### 1. vLLM 的组件化模式

vLLM 的 GPUModelRunner 采用**高度组件化**的架构：

```python
class GPUModelRunner(LlamaModelRunnerMixin):
    def __init__(self, vllm_config, device):
        # 组件初始化，每个组件职责单一
        self.req_states = RequestState(...)          # 请求状态管理
        self.input_buffers = InputBuffers(...)        # 输入缓冲管理
        self.sampler = Sampler(...)                 # 采样逻辑
        self.encoder_runner = EncoderRunner(...)      # 多模态编码
        self.prompt_logprobs_worker = PromptLogprobsWorker(...)
        self.cudagraph_manager = CudaGraphManager(...)  # CUDA Graph 管理
        self.kv_connector: KVConnector = ...        # KV Cache 连接
```

**关键特点：**
- 每个组件有明确的生命周期方法：`add_request()`, `remove_request()`, `apply_staged_writes()`
- **Staged Writes 模式**：组件支持 `stage_write()` 收集变更，`apply_staged_writes()` 一次性提交，减少 kernel 启动
- 清晰的数据流：`EngineCoreRequest` → `SchedulerOutput` → `ForwardBatch` → `SamplerOutput`

### 2. SGLang 的进程分离模式

SGLang 将 CPU 密集型工作分离到独立进程：

```
TokenizerManager (tokenization)  →  ZMQ IPC  →  Scheduler (batching/scheduling)
         ↓                                              ↓
Scheduler                                           ModelWorker (GPU inference)
         ↓
DetokenizerManager (detokenization)
```

**优点：**
- 自然的功能边界
- 各组件可独立扩缩容
- 减少对 GPU 推理进程的干扰

### 3. 流程与组件的清晰分离

**vLLM 的分离方式：**

```
LLMEngine (流程编排层):
  - 调度决策
  - 请求生命周期管理
  - 统计信息收集
  - 多进程协调

ModelRunner (功能组件层):
  - Sampler - 纯采样逻辑
  - EncoderRunner - 多模态编码
  - BlockTables - KV Cache 块管理
  - KVConnector - Cache 传输
```

流程编排不依赖具体实现，只通过接口与组件交互。

### 4. 一致的生命周期方法模式

vLLM 组件遵循一致的生命周期模式：

```python
class Component:
    def add_request(self, req_idx, ...) -> None:
        """添加请求到组件"""

    def remove_request(self, req_idx) -> None:
        """从组件移除请求"""

    def reset_cache(self) -> None:
        """重置组件缓存"""

    def __call__(self, ...) -> Output:
        """执行组件功能"""
```

**建议：** FastDeploy 的组件也遵循类似的模式，便于理解和维护。

### 5. Registry 注册模式

vLLM 和 SGLang 都使用注册表实现可扩展性：

```python
# vLLM 示例
MULTIMODAL_REGISTRY = {
    "image": ImageProcessor,
    "audio": AudioProcessor,
    "video": VideoProcessor,
}

# 动态注册
@MULTIMODAL_REGISTRY.register("custom")
class CustomProcessor(MultimodalProcessorBase):
    ...
```

**建议：** 为 attention backend、sampling method、multimodal processor 等实现注册机制。

### 6. Staged Writes 模式（性能优化）

vLLM 使用 Staged Writes 来减少 GPU kernel 启动次数：

```python
# Stage 阶段：收集变更
sampler.stage_write(req_idx, "token", token)
sampler.stage_write(req_idx, "logprob", logprob)

# Commit 阶段：一次性提交
sampler.apply_staged_writes()  # 只启动一次 kernel
```

**建议：** 考虑在状态更新密集的地方（如 KV Cache 分配）引入此模式。

### 7. Dataclass/Struct 化的批数据

vLLM 使用结构化的批数据容器：

```python
@dataclass
class InputBuffers:
    """GPU 输入缓冲区的集中管理"""
    input_ids: paddle.Tensor
    positions: paddle.Tensor
    block_tables: paddle.Tensor
    ...
```

**建议：** 为 `InputBatch` 添加类型提示和数据验证。

---

## FastDeploy 设计 vs vLLM/SGLang 对比分析

### 架构对比表

| 维度 | FastDeploy | vLLM | SGLang |
|------|-------------|-------|--------|
| **架构模式** | Manager + Orchestrator | 高度组件化 | 进程分离 |
| **组件粒度** | 6 个组件 + 2 个流程 | ~10+ 组件 | 4+ 独立进程 |
| **生命周期方法** | ❌ 未统一 | ✅ add_request/remove_request/apply_staged_writes | - |
| **性能优化** | ❌ 无 Staged Writes | ✅ Staged Writes 模式 | - |
| **进程架构** | 单进程 | 单进程 | 多进程分离 |
| **可扩展性** | ⚠️ 部分支持 | ✅ Registry 注册模式 | ⚠️ 中等 |
| **向后兼容** | ✅ 强调兼容 | ⚠️ 逐步演进 | ⚠️ 中等 |

---

### 1. 我们好的地方

#### 1.1 向后兼容性

**FastDeploy**: 明确强调"Runner 对外接口保持不变，逐步迁移"

这比 vLLM/SGLang 更注重平滑演进，适合生产环境：

```python
# FastDeploy: 委托模式，外部接口不变
def execute_model(self, batch, ...):
    return self.inference_flow.execute(batch, ...)
```

#### 1.2 V0/V1 调度器统一抽象

**FastDeploy**:
```python
def insert_tasks(self, req_dicts, ...):
    if self.use_v1_scheduler:
        return self._insert_tasks_v1(req_dicts, ...)
    else:
        return self._insert_tasks_v0(req_dicts, ...)
```

这是一个实用的设计，统一了不同调度器版本的接口。

#### 1.3 encoder_cache 跨请求复用

**FastDeploy**: 明确支持视觉特征缓存和复用

这是 vLLM 和 SGLang 文档中未强调的优化，对于多模态场景很重要。

#### 1.4 share_inputs 访问控制明确

**FastDeploy**: 有完整的访问控制表和单一写入者原则

vLLM 虽然有类似概念，但文档中未如此明确说明访问模式。

---

### 2. 我们差的地方

#### 2.1 缺少一致的生命周期方法 ❌

**vLLM**:
```python
class Component:
    def add_request(self, req_idx, ...) -> None:
    def remove_request(self, req_idx) -> None:
    def reset_cache(self) -> None:
    def __call__(self, ...) -> Output:
```

**FastDeploy**: 组件方法名不统一，缺少标准模式

| 组件 | 添加请求 | 移除请求 | 重置缓存 |
|------|----------|----------|----------|
| InputManager | `insert_tasks` | `clear_requests` | - |
| VisionProcessor | `process_mm_features` | - | - |
| OutputHandler | - | - | - |
| CacheManager | - | `clear_cache` | - |

**建议**: 为所有组件引入统一的生命周期方法模式。

#### 2.2 缺少 Staged Writes 性能优化模式 ❌

**vLLM**:
```python
# Stage 阶段：收集变更
sampler.stage_write(req_idx, "token", token)
sampler.stage_write(req_idx, "logprob", logprob)

# Commit 阶段：一次性提交
sampler.apply_staged_writes()  # 只启动一次 kernel
```

**FastDeploy**: 每次操作立即写入，可能增加 kernel 启动次数

**影响**: 状态更新密集的地方（如 KV Cache 分配、采样参数更新）性能可能落后

**建议**: 在状态更新密集的操作中引入 Staged Writes 模式。

#### 2.3 没有进程分离架构 ❌

**SGLang**:
```
TokenizerManager → ZMQ IPC → Scheduler → ModelWorker
```

**优势**:
- CPU 密集型工作不影响 GPU 推理
- 各组件可独立扩缩容

**FastDeploy**: 所有组件在同一进程中，CPU 密集型操作可能影响推理性能

**建议**: 长期可考虑将 CPU 密集型组件（如 tokenizer/detokenizer）分离到独立进程。

#### 2.4 缺少 Registry 注册模式 ⚠️

**vLLM**:
```python
@MULTIMODAL_REGISTRY.register("custom")
class CustomProcessor(MultimodalProcessorBase):
    ...
```

**FastDeploy**: 扩展性依赖直接修改代码，缺少插件机制

**建议**: 为 attention backend、sampling method、multimodal processor 等实现注册机制。

#### 2.5 share_inputs 访问控制依赖约定而非强制 ⚠️

**问题**:
```python
# share_inputs 是字典风格，没有类型安全
self.share_inputs["image_features"] = tensor  # 谁都可以写
```

**vLLM**: 使用 Dataclass 封装
```python
@dataclass
class InputBuffers:
    input_ids: paddle.Tensor
    positions: paddle.Tensor
    # 编译时类型检查
```

**建议**: 将 `share_inputs` 从字典风格改为类型安全的 Dataclass 或 pydantic 模型。

---

### 3. 关键差距总结

| 类别 | 差距 | 影响 | 优先级 |
|------|------|------|--------|
| **性能** | Staged Writes | GPU kernel 启动次数 | 高 |
| **可维护性** | 统一生命周期方法 | 组件理解难度 | 高 |
| **可扩展性** | Registry 模式 | 插件生态 | 中 |
| **架构** | 进程分离 | CPU/GPU 干扰 | 中 |
| **类型安全** | Dataclass 封装 | 运行时错误 | 低 |

---

### 4. 建议的改进方向

#### 高优先级
1. **引入 Staged Writes 模式** - 性能影响直接
2. **统一组件生命周期方法** - 降低维护成本
3. **share_inputs 类型化** - 减少运行时错误

#### 中优先级
4. **Registry 注册模式** - 支持插件扩展
5. **考虑进程分离** - 长期架构优化

---

### 5. 总体评价

**FastDeploy 的设计在实用性和兼容性方面做得很好**，特别适合需要平滑迁移的生产场景。以下是其核心优势：

- ✅ 清晰的组件职责划分
- ✅ 明确的访问控制模式
- ✅ 强烈的向后兼容意识
- ✅ V0/V1 调度器统一抽象
- ✅ encoder_cache 跨请求复用优化

**但在性能优化模式和架构先进性方面落后于 vLLM/SGLang**：

- ❌ 缺少 Staged Writes 性能优化
- ❌ 组件生命周期方法不统一
- ❌ 没有进程分离架构
- ⚠️ 缺少 Registry 注册模式
- ⚠️ 类型安全性不足

**建议**: 在保持现有兼容性和实用性的前提下，逐步借鉴 vLLM/SGLang 的成熟实践，优先解决高优先级的性能和可维护性问题。

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

### VisionProcessor (11 个方法)
- `_init_image_preprocess`
- `vision_encoder_compile`
- `_process_mm_features`
- `_preprocess_mm_task`
- `_get_feature_positions`
- `extract_vision_features`
- `extract_vision_features_ernie`
- `extract_vision_features_qwen`
- `extract_vision_features_paddleocr`
- `prepare_rope3d`
- `_dummy_run_extract_vision_features`

### OutputHandler (6 个方法)
- `_postprocess`
- `_save_model_output`
- `_pool`
- `_get_prompt_logprobs_list`
- `_get_p_done_idxs_gd`
- `_async_output_busy_loop`

### ProfileRunner (9 个方法)
- `profile_run`
- `_dummy_run`
- `_dummy_sampler_run`
- `_dummy_pooler_run`
- `_dummy_pooler_run_task`
- `sot_warmup`
- `capture_model`
- `capture_model_prefill_and_mixed`

### InputManager (6 个方法)
- `insert_tasks_v1`
- `insert_prefill_inputs`
- `_prepare_inputs`
- `get_input_length_list`
- `_process_reorder`
- `clear_requests`
- `_dummy_prefill_inputs`（dummy run 时使用）

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
