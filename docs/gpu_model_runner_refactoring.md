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
# components/base_component.py

class BaseComponent:
    """
    组件基类，定义统一的生命周期方法
    """
    def add_request(self, req_idx: int, request: Request) -> None:
        """添加请求到组件"""
        pass

    def remove_request(self, req_idx: int) -> None:
        """从组件移除请求"""
        pass

    def reset_cache(self) -> None:
        """重置组件缓存"""
        pass

# components/input_manager.py

class InputManager(BaseComponent):
    """
    输入管理组件，遵循统一的生命周期方法

    职责：
    - 管理请求的插入和排队
    - 准备模型输入
    - 处理请求重排序
    - 协调视觉特征提取（通过 VisionProcessor 策略）

    设计模式：延迟加载 + 策略模式
    - VisionProcessor 通过策略模式注入，支持延迟初始化
    - 当 VisionProcessor 未设置时，使用 NullVisionProcessor（空实现）
    - 这样可以在 load_model 之前插入任务，多模态任务会被暂存并延迟处理
    """

    def __init__(self, fd_config: FDConfig, share_inputs: 'InputBatch', vision_processor: 'VisionProcessor' = None):
        self.fd_config = fd_config
        self.share_inputs = share_inputs  # 引用 GPUModelRunner 管理的共享输入
        self._vision_processor = vision_processor  # 可能为 None
        self._pending_mm_requests: List[Tuple[int, Request]] = []  # 暂存的多模态请求

        # 根据 envs.ENABLE_V1_KVCACHE_SCHEDULER 决定使用哪个调度器模式
        self.use_v1_scheduler = int(envs.get("ENABLE_V1_KVCACHE_SCHEDULER", 0)) == 1

    def set_vision_processor(self, vision_processor: 'VisionProcessor'):
        """
        设置 VisionProcessor（延迟初始化）

        注意：设置后会处理所有暂存的多模态请求
        """
        self._vision_processor = vision_processor
        if vision_processor and self._pending_mm_requests:
            self._process_pending_mm_requests()
            self._pending_mm_requests.clear()

    def get_vision_processor(self) -> 'VisionProcessor':
        """获取 VisionProcessor（总是返回有效的处理器）"""
        return self._vision_processor or NullVisionProcessor()

    def _process_pending_mm_requests(self):
        """处理暂存的多模态请求"""
        for req_idx, request in self._pending_mm_requests:
            self._vision_processor.process_request_features(req_idx, request)

    # ========== 统一生命周期方法 ==========
    def add_request(self, req_idx: int, request: Request) -> None:
        """添加请求到输入管理器（内部方法）"""
        self._insert_task_to_batch(req_idx, request)

    def remove_request(self, req_idx: int) -> None:
        """从输入管理器移除请求（内部方法）"""
        self._remove_task_from_batch(req_idx)

    def reset_cache(self) -> None:
        """重置输入缓存（调用 share_inputs.reset_cache()）"""
        self.share_inputs.reset_cache()

    # ========== 原有方法（向后兼容）==========
    def insert_tasks(self, req_dicts: List[Request], num_running_requests: int = None):
        """插入新的推理任务（统一接口）"""
        for idx, req in enumerate(req_dicts):
            self.add_request(idx, req)

    def insert_tasks_v1(self, req_dicts: List[Request], num_running_requests: int = None):
        """插入新的推理任务（V1 模式，兼容接口）"""
        # ... V1 插入逻辑 ...

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int = None):
        """插入 prefill 阶段的输入（V0 模式，兼容接口）"""
        # ... V0 插入逻辑 ...

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

    def exist_prefill(self) -> bool:
        """检查是否存在 prefill 阶段"""
        pass

    def exist_decode(self) -> bool:
        """检查是否存在 decode 阶段"""
        pass

    def only_prefill(self) -> bool:
        """检查是否仅 prefill"""
        pass

    def only_decode(self) -> bool:
        """检查是否仅 decode"""
        pass

    def not_need_stop(self) -> bool:
        """
        检查是否不需要停止

        职责说明：
        - 判断是否还有请求需要继续处理
        - 基于当前批次的请求状态（prefill/decode/完成）
        - 这是 InputManager 的职责，因为它管理所有待处理的请求
        """
        # 检查是否有待处理的请求（prefill 或 decode）
        return not (self.exist_prefill() or self.exist_decode())

    def initialize_forward_meta(self, is_dummy_or_profile_run=False):
        """初始化 forward meta（输入准备的一部分）

        此方法属于 InputManager 的职责，因为：
        - forward meta 是模型输入的一部分
        - 与 prepare_inputs 配合使用
        - 需要访问 share_inputs 中的输入信息
        """
        pass

    def get_input_length_list(self) -> List[int]:

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
```
                Scheduler
                    │
     ┌──────────────┴──────────────┐
     │                             │
V1 Scheduler                 V0 Scheduler
     │                             │
     ▼                             ▼
```
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

        工作流程：
        1. 根据 use_v1_scheduler 选择对应的实现（V0 或 V1）
        2. 将请求插入到 share_inputs 的相关字段
        3. 如果存在 VisionProcessor 且请求包含多模态输入，调用视觉特征处理
        4. 如果 VisionProcessor 未初始化，多模态请求会被暂存

        注意：
        - 视觉特征提取在任务插入阶段完成，不重新提取
        - 特征处理必须在任务插入之后，以便知道哪些请求需要处理
        - 支持在 load_model 之前调用：多模态请求会被暂存，load_model 后自动处理
        - 使用 NullVisionProcessor 模式确保代码路径一致
        """
        vision_proc = self.get_vision_processor()

        # 检查是否有需要 VisionProcessor 的多模态请求
        has_mm_requests = self._check_multimodal_requests(req_dicts)

        # 根据配置选择 V0 或 V1 调度器模式
        if self.use_v1_scheduler:
            self._insert_tasks_v1(req_dicts, num_running_requests)
        else:
            self._insert_tasks_v0(req_dicts, num_running_requests)

        # 处理视觉特征
        if has_mm_requests:
            if vision_proc is None:
                # VisionProcessor 未初始化，暂存多模态请求
                self._stash_mm_requests(req_dicts)
            else:
                # VisionProcessor 已初始化，直接处理
                vision_proc.process_mm_features(req_dicts)

    def _stash_mm_requests(self, req_dicts: List[Request]):
        """暂存多模态请求，等待 VisionProcessor 初始化后处理"""
        for idx, req in enumerate(req_dicts):
            if self._is_multimodal_request(req):
                self._pending_mm_requests.append((idx, req))

    def _check_multimodal_requests(self, req_dicts: List[Request]) -> bool:
        """检查请求列表中是否包含多模态输入"""
        for req in req_dicts:
            if hasattr(req, 'mm_inputs') and req.mm_inputs is not None:
                return True
        return False

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

# NullVisionProcessor - 空对象模式

class NullVisionProcessor(BaseComponent):
    """
    空对象模式的 VisionProcessor 实现

```
用于 VisionProcessor 未初始化时的默认处理器，确保代码路径一致
所有方法都返回空实现或无操作
"""
def process_request_features(self, req_idx: int, request: Request) -> None:
    """空实现：暂存请求特征处理"""
    pass

def process_mm_features(self, request_list: List[Request]) -> None:
    """空实现"""
    pass

def add_request(self, req_idx: int, request: Request) -> None:
    """空实现"""
    pass

def remove_request(self, req_idx: int) -> None:
    """空实现"""
    pass

def reset_cache(self) -> None:
    """空实现"""
    pass
```

## 2. VisionProcessor - 视觉处理组件

```python
# components/vision_processor.py

class VisionProcessor(BaseComponent):
    """
    视觉处理组件，遵循统一的生命周期方法

    职责：
    - 处理图像输入
    - 提取视觉特征并缓存到 encoder_cache
    - 将提取的特征写入 share_inputs["image_features_list"]
    - 准备 3D RoPE 位置编码

    注意：
    - encoder_cache 由 GPUModelRunner 管理，VisionProcessor 通过引用访问
    - share_inputs 由 GPUModelRunner 管理，VisionProcessor 通过引用访问
    - encoder_cache 用于跨请求复用（避免重复提取相同图像的特征）
    - share_inputs["image_features_list"] 用于当前批次的推理
    - process_request_features() 是统一的生命周期方法实现（add_request 的具体实现）
    - process_mm_features() 是批量处理方法，用于一次处理多个请求
    - 单一写入者原则：只有 VisionProcessor 写入 share_inputs["image_features_list"]
    """

    def __init__(self, fd_config: FDConfig, model, encoder_cache: dict = None, share_inputs: 'InputBatch' = None):
        self.fd_config = fd_config
        self.model = model
        self.encoder_cache = encoder_cache if encoder_cache is not None else {}  # 视觉特征缓存，由 GPUModelRunner 管理
        self.share_inputs = share_inputs  # 共享输入缓冲区，由 GPUModelRunner 管理
        self._init_image_preprocess()

    # ========== 统一生命周期方法实现 ==========

    def add_request(self, req_idx: int, request: Request) -> None:
        """
        添加请求（BaseComponent 生命周期方法）

        此方法为统一的生命周期接口，实际工作委托给 process_request_features

        Args:
            req_idx: 请求索引
            request: 请求对象
        """
        self.process_request_features(req_idx, request)

    def process_request_features(self, req_idx: int, request: Request) -> None:
        """
        处理单个请求的视觉特征

        工作流程：
        1. 检查 encoder_cache 中是否已有缓存
        2. 如果没有，提取新特征并缓存
        3. 将特征写入 share_inputs["image_features_list"]

        注意：
        - 这是 add_request 生命周期方法的具体实现
        - 同时负责缓存和写入 share_inputs，确保特征可用
        """
        mm_hash = self._get_mm_hash(request)

        # 确保特征被提取并缓存到 encoder_cache
        if mm_hash not in self.encoder_cache:
            features = self.extract_vision_features(request.mm_inputs)
            self.encoder_cache[mm_hash] = features.detach().cpu()

        # 将特征写入 share_inputs
        self._write_features_to_share_inputs(request)

    def remove_request(self, req_idx: int) -> None:
        """移除请求（清理相关状态）

        注意：
        - encoder_cache 不主动清理，保留跨请求复用能力
        - GPUModelRunner.reset_cache() 时统一清理
        """
        pass

    def reset_cache(self) -> None:
        """重置视觉特征缓存（清理 encoder_cache）"""
        self.encoder_cache.clear()

    # ========== 批量处理方法（业务方法）==========

    def process_mm_features(self, request_list: List[Request]) -> None:
        """
        批量处理并缓存视觉特征（业务方法）

        工作流程：
        1. 遍历请求列表，检查 encoder_cache 中是否有缓存的特征
        2. 如果没有，提取新特征并缓存到 encoder_cache
        3. 从 encoder_cache 读取特征（GPU 或 CPU）
        4. 将特征写入 share_inputs["image_features_list"] 供后续推理使用

        关键点：
        - 这是将特征写入 share_inputs 的唯一入口（单一写入者原则）
        - 从 encoder_cache 读取时需要判断是否需要转移到 GPU
        - share_inputs["image_features_list"] 在每次调用前会被重置或累加，由调用方控制
        - 内部调用 process_request_features 处理每个请求

        Args:
            request_list: 需要处理视觉特征的请求列表
        """
        # 清空之前的 image_features_list（或由调用方在调用前处理）
        # 这里假设调用方（InputManager.insert_tasks）负责管理 share_inputs 的生命周期

        for req in request_list:
            self.process_request_features(-1, req)  # req_idx 对于批量处理不重要

    def _write_features_to_share_inputs(self, request: Request) -> None:
        """从 encoder_cache 读取特征并写入 share_inputs"""
        mm_hash = self._get_mm_hash(request)
        if mm_hash in self.encoder_cache:
            # 确保 image_features_list 存在
            if self.share_inputs.image_features_list is None:
                self.share_inputs.image_features_list = []

            # 从 encoder_cache 读取特征
            cached_features = self.encoder_cache[mm_hash]

            # 如果特征在 CPU 上，转移到 GPU（如果需要）
            if cached_features.place.is_cpu():
                # 根据模型配置决定是否转移到 GPU
                device = self._get_device_for_features()
                features = cached_features.to(device)
            else:
                features = cached_features

            self.share_inputs.image_features_list.append(features)

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

class OutputHandler(BaseComponent):
    """
    输出处理组件，遵循统一的生命周期方法

    职责：
    - 后处理采样结果
    - 保存模型输出
    - 处理 pooling 输出
    - 异步输出循环
    - 处理输出相关的任务（如 prompt_logprobs、guided decoding）

    注意：
    - not_need_stop() 等状态查询方法不属于 OutputHandler 的职责
    - 这些状态查询应由 InputManager 或 Scheduler 负责
    - OutputHandler 专注于输出处理本身
    """

    def __init__(self, fd_config: FDConfig):
        self.fd_config = fd_config
        self.async_output_queue = queue.Queue()
        self._start_async_output_thread()

    # ========== 统一生命周期方法 ==========
    def add_request(self, req_idx: int, request: Request) -> None:
        """添加请求到输出处理"""
        # OutputHandler 可能不需要预初始化
        pass

    def remove_request(self, req_idx: int) -> None:
        """移除请求（清理输出状态）"""
        self._cleanup_output_state(req_idx)

    def reset_cache(self) -> None:
        """重置输出缓存"""
        # 清理异步输出队列等
        pass

    # ========== 状态查询方法（已移除，不属于 OutputHandler）==========
    # not_need_stop() 等方法已迁移到 InputManager 或 Scheduler

    # ========== 原有方法 ==========
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

    # ========== 内部辅助方法 ==========
    def _cleanup_output_state(self, req_idx: int):
        """清理请求的输出状态"""
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
# components/staged_write_mixin.py

class StagedWriteMixin:
    """
    Staged Writes 混入类

    用于需要批量更新状态的组件，减少 GPU kernel 启动次数

    设计原则：
    - stage_write(): 收集所有待写入的变更，暂存不执行
    - apply_staged_writes(): 一次性提交所有变更
    - 对于同一类型的写入操作（如 token），合并为一次调用
    - 支持立即写入模式（immediate write）用于依赖立即结果的场景

    注意：
    - 不同 key 的写入仍然需要分别调用 _batch_write（如 token 和 logprob）
    - 真正减少 kernel 启动次数需要 _batch_write 内部确保只启动一次 kernel
    - 这是一种按类型（key）分批的优化模式，而非全局合并

    适用场景：
    - 多个请求的状态更新（如更新多个请求的 token）
    - 采样参数的批量更新
    - 避免在循环中频繁触发 kernel

    立即写入模式（Immediate Write）：
    - 某些场景下需要立即写入（如 KV Cache 分配，后续操作依赖结果）
    - 使用 stage_write(..., immediate=True) 标记需要立即执行的写入
    - 立即写入不会进入 staged_writes，直接执行

    不适用场景：
    - 需要依赖其他组件写入结果的操作（应使用 staged 并协调执行顺序）
    - 跨组件的状态同步（应在更高层次协调）
    """

    def __init__(self):
        # 存储待写入的变更
        self._staged_writes: Dict[str, List[Tuple]] = {}

    def stage_write(self, key: str, value: Any, priority: int = 0, immediate: bool = False):
        """
        暂存写入操作

        Args:
            key: 写入目标（如 "token", "logprob"）
            value: 要写入的值
            priority: 写入优先级（用于控制写入顺序）
            immediate: 是否立即执行（用于需要依赖写入结果的场景）
        """
        if immediate:
            # 立即执行，不进入 staged_writes
            self._immediate_write(key, value)
        else:
            if key not in self._staged_writes:
                self._staged_writes[key] = []
            self._staged_writes[key].append((value, priority))

    def _immediate_write(self, key: str, value: Any):
        """立即写入操作（子类可覆盖以提供更高效的实现）"""
        self._batch_write(key, [value])

    def apply_staged_writes(self):
        """
        批量执行所有暂存的写入操作

        将多次写入合并为批量操作，减少 kernel 启动次数

        工作原理：
        - 对于每个 key，收集所有待写入的值
        - 按 priority 排序
        - 调用 _batch_write 一次性处理所有相同 key 的写入

        注意：
        - 不同 key（如 "token" 和 "logprob"）仍然分别调用 _batch_write
        - 每个 _batch_write 内部应该确保只启动一次 kernel
        - 立即写入不会在此处理，已在 stage_write 时完成
        """
        for key, writes in self._staged_writes.items():
            # 按优先级排序
            writes.sort(key=lambda x: x[1])
            values = [v for v, _ in writes]
            self._batch_write(key, values)
        self._staged_writes.clear()

    def has_pending_writes(self) -> bool:
        """检查是否有待处理的写入操作"""
        return len(self._staged_writes) > 0

    def _batch_write(self, key: str, values: List[Any]):
        """
        实际的批量写入操作（由子类实现）

        关键要求：此方法内部必须确保只启动一次 kernel

        Args:
            key: 写入目标
            values: 要写入的值列表

        示例实现：
            def _batch_write(self, key: str, values: List[Any]):
                if key == "token":
                    # 将所有 token 更新收集后，一次 kernel 写入
                    req_indices = [idx for idx, _ in values]
                    tokens = [token for _, token in values]
                    # 一次 scatter_assign 操作，只启动一次 kernel
                    paddle.scatter_assign(self.token_buffer, req_indices, tokens)
                elif key == "logprob":
                    req_indices = [idx for idx, _ in values]
                    logprobs = [logprob for _, logprob in values]
                    # 一次 kernel 更新所有 logprobs
                    paddle.scatter_assign(self.logprob_buffer, req_indices, logprobs)
        """
        raise NotImplementedError

# components/cache_manager.py

class CacheManager(BaseComponent, StagedWriteMixin):
    """
    Cache 管理组件，遵循统一的生命周期方法

    职责：
    - 初始化 KV Cache
    - 管理 GPU blocks
    - 清理和重置 cache

    注意：
    - KV Cache 分配使用立即写入模式（immediate=True），因为后续操作依赖分配结果
    - KV Cache 释放使用 staged 模式，批量释放以提高性能
    - 资源生命周期管理：分配和释放采用不同策略是合理的，因为依赖关系不同
    """

    def __init__(self, fd_config: FDConfig, model):
        super().__init__()
        self.fd_config = fd_config
        self.model = model
        self.total_block_num = 0

    # ========== 统一生命周期方法 ==========
    def add_request(self, req_idx: int, request: Request) -> None:
        """
        为请求分配 cache（立即执行，因为后续操作依赖分配结果）

        使用 immediate=True 确保分配立即生效，block_table 可立即使用
        """
        num_blocks = self._calculate_blocks_needed(request)
        self.stage_write("block_allocation", (req_idx, num_blocks), immediate=True)

    def remove_request(self, req_idx: int) -> None:
        """
        释放请求的 cache（staged 模式，批量执行以提高性能）

        释放操作可以延迟，因为释放后的 block 不需要立即被其他请求使用
        在 apply_staged_writes() 时批量释放，减少 kernel 启动次数
        """
        self.stage_write("block_free", (req_idx,))  # 暂存，稍后批量执行

    def reset_cache(self) -> None:
        """重置所有 cache"""
        self.clear_cache()

    # ========== 原有方法 ==========
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

    def _batch_write(self, key: str, values: List[Any]):
        """实际执行批量写入（内部确保只启动一次 kernel）"""
        if key == "block_allocation":
            # 批量分配 blocks，立即执行
            req_indices = [idx for idx, _ in values]
            num_blocks = [blocks for _, blocks in values]
            self._do_batch_allocate_blocks(req_indices, num_blocks)
        elif key == "block_free":
            # 批量释放 blocks，一次 kernel 操作
            req_indices = [idx for idx, _ in values]
            self._do_batch_free_blocks(req_indices)
        elif key == "block_table_update":
            # 批量更新 block_table，一次 kernel 操作
            req_indices = [idx for idx, _ in values]
            block_tables = [table for _, table in values]
            self._do_batch_update_block_tables(req_indices, block_tables)

    def _immediate_write(self, key: str, value: Any):
        """
        立即写入（单个请求的块分配）

        覆盖父类方法以提供优化的单个请求分配实现
        """
        if key == "block_allocation":
            req_idx, num_blocks = value
            self._allocate_block(req_idx, num_blocks)
        else:
            # 其他情况使用批量写入
            super()._immediate_write(key, value)
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
    推理流程的编排层（纯流程编排器）

    职责：
    - 编排推理的完整流程
    - 协调各个组件的调用顺序
    - 处理不同的执行模式

    设计原则：
    - 无状态编排器：不持有任何组件实例
    - 所有依赖通过参数传入
    - 每次调用完全独立，无副作用

    注意：
    - 视觉特征提取在任务插入阶段（insert_tasks）完成，不在推理流程中重复提取
    - prepare_inputs 只负责整理输入数据，包括拼接已缓存的 vision_features
    - model 通过参数传入，避免持有状态
    """

    def __init__(self):
        """无状态编排器，不需要初始化任何状态"""
        pass

    def execute(self, model_forward_batch: List[Request],
               model: nn.Layer,
               input_mgr: InputManager,
               output_hdlr: OutputHandler,
               **kwargs) -> ModelRunnerOutput:
        """
        主执行入口

        Args:
            model_forward_batch: 请求批次
            model: 模型实例（参数传入，不持有）
            input_mgr: 输入管理器
            output_hdlr: 输出处理器
            **kwargs: 其他参数

        工作流程：
        1. 检查是否为空输入
        2. 处理重排序（如果需要）
        3. 准备输入（包括处理视觉特征和拼接）
        4. 执行模型推理
        5. 后处理

        关键点：
        - 视觉特征提取在任务插入阶段（insert_tasks）完成
        - 此处 prepare_inputs 只负责数据整理（如移除 padding、初始化 forward meta）
        - 如果存在 vision_proc 且有视觉特征，负责拼接 image_features_list -> image_features
        """
        # 1. 检查空输入
        if self._is_empty_input(model_forward_batch):
            return self._execute_empty_input(kwargs.get('forward_meta'))

        # 2. 处理重排序（如果需要）
        input_mgr.process_reorder()

        # 3. 准备输入
        #    - 移除 padding
        #    - 初始化 forward meta
        #    - 获取 sampling metadata
        #    - 拼接视觉特征（如果存在）
        input_mgr.prepare_inputs(last_token_num=kwargs.get('last_token_num', -1),
                                is_dummy_or_profile_run=kwargs.get('is_dummy_or_profile_run', False))

        # 4. 执行模型推理
        outputs = self._preprocess_and_execute_model(
            model,
            model_forward_batch,
            input_mgr.share_inputs,
            kwargs.get('forward_meta')
        )

        # 5. 后处理
        output_hdlr.postprocess(outputs.sampler_output, outputs.model_output,
                                input_mgr.share_inputs, **kwargs)

        return outputs

    def _is_empty_input(self, model_forward_batch: List[Request]) -> bool:
        """检查是否为空输入"""
        return model_forward_batch is None or len(model_forward_batch) == 0

    def _execute_empty_input(self, forward_meta):
        """执行空输入处理"""
        # 处理无请求的情况
        pass

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
| **拥有模型** | ✅ 持有 model 实例 | ❌ model 作为参数传入 |
| **组件生命周期** | ✅ 创建和管理所有组件 | ❌ 不管理组件 |
| **配置管理** | ✅ 持有 FDConfig | ❌ 依赖传入的配置 |
| **初始化** | ✅ KV Cache、Attn Backend 等 | ❌ 不负责初始化 |
| **外部接口** | ✅ 提供给外部调用 | ❌ 内部使用 |
| **状态查询** | ✅ exist_prefill/decode 等 | ❌ 无状态 |
| **Profile/Warmup** | ✅ 负责 | ❌ 不负责 |
| **推理编排** | ⏸️ 委托给 InferenceFlow | ✅ 负责（纯流程）|

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
│  │  • execute() - 纯流程编排，无状态，model 作为参数传入  │   │
│  │  • 协调组件调用顺序                                      │   │
│  │  • 每次调用完全独立                                      │   │
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
- 不持有任何状态，所有依赖通过参数传入
- 纯流程编排，每次调用完全独立
- 不管理资源，只协调组件调用顺序
- model 通过 execute() 方法参数传入，不在 __init__ 中持有

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

        # 初始化共享状态（优先初始化，因为其他组件可能依赖）
        self.share_inputs = InputBatch(self.fd_config)
        self.share_inputs.init_share_inputs()

        # 初始化 encoder_cache（视觉特征缓存）
        if self.cache_config.max_encoder_cache > 0:
            self.encoder_cache: dict[str, paddle.Tensor] = {}
        else:
            self.encoder_cache = None

        # 初始化各个组件（传入 share_inputs 引用）
        self.input_manager = InputManager(fd_config, share_inputs=self.share_inputs)
        self.output_handler = OutputHandler(fd_config)
        self.cache_manager = None     # load_model 后初始化
        self.profile_runner = None    # load_model 后初始化
        self.inference_flow = None    # load_model 后初始化
        self.vision_processor = None  # load_model 后初始化，需要传入 encoder_cache 和 share_inputs

        # 初始化其他组件
        self._init_speculative_proposer()
        self._init_logits_processor()

    def load_model(self) -> None:
        """加载模型并初始化依赖组件"""
        model = self._get_model_instance()

        # 初始化依赖 model 的组件（传入 share_inputs 引用）
        self.vision_processor = VisionProcessor(
            self.fd_config,
            model,
            encoder_cache=self.encoder_cache,
            share_inputs=self.share_inputs
        )
        self.cache_manager = CacheManager(self.fd_config, model)
        self.profile_runner = ProfileRunner(self.fd_config, model)
        self.inference_flow = InferenceFlow()  # 无状态，不需要 model

        # 设置 InputManager 的 VisionProcessor 引用
        self.input_manager.set_vision_processor(self.vision_processor)

        self.model = model

    def execute_model(self, model_forward_batch: Optional[List[Request]], **kwargs) -> ModelRunnerOutput:
        """执行模型推理（委托给 InferenceFlow）"""
        return self.inference_flow.execute(
            model_forward_batch,
            model=self.model,  # model 作为参数传入
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
        """
        检查是否不需要停止

        职责说明：
        - 此方法判断是否还有请求需要继续处理
        - 状态由 InputManager 管理（它跟踪所有待处理的请求）
        - 委托给 InputManager，而非 OutputHandler
        """
        return self.input_manager.not_need_stop()

    def get_model(self) -> nn.Layer:
        """获取模型"""
        return self.model

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """
        获取支持的 pooling 任务

        职责说明：
        - pooling 任务类型由配置决定
        - 此方法直接从配置读取，不依赖组件状态
        - 放在 GPUModelRunner 中，因为它持有 FDConfig
        """
        return self._get_pooling_tasks_from_config()

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

    # initialize_forward_meta 已迁移到 InputManager，通过委托访问
    # GPUModelRunner 只保留与组件协调相关的方法

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

    def _get_pooling_tasks_from_config(self) -> list[PoolingTask]:
        """从配置获取支持的 pooling 任务"""
        # 从 FDConfig 中读取支持的 pooling 任务配置
        pooling_config = self.fd_config.get('pooling_tasks', [])
        return [PoolingTask.from_config(cfg) for cfg in pooling_config]
```

---

## 架构改进方案

基于与 vLLM/SGLang 的对比分析，以及 FastDeploy 现有代码库的探索结果，以下是对关键劣势的改进方案设计。

---

### 1. Staged Writes 性能优化模式

**目标**: 减少 GPU kernel 启动次数，优化状态更新密集操作（如 KV Cache 分配、采样参数更新）的性能。

**参考实现**: FastDeploy 代码库中暂无此模式，需要引入。vLLM 已实现类似模式。

#### 设计方案

```python
# components/staged_write_mixin.py

class StagedWriteMixin:
    """
    Staged Writes 混入类

    用于需要批量更新状态的组件，减少 GPU kernel 启动次数

    设计原则：
    - stage_write(): 收集所有待写入的变更，暂存不执行
    - apply_staged_writes(): 一次性提交所有变更，只启动一次 kernel
    - 对于同一类型的数据（如所有 token 更新），应该合并为一次写入操作

    注意：
    - 不同 key 的写入仍然需要分别调用 _batch_write
    - 但同一 key 的多次写入会合并为一次调用
    - 真正减少 kernel 启动需要确保 _batch_write 内部只启动一次 kernel
    """

    def __init__(self):
        # 存储待写入的变更
        self._staged_writes: Dict[str, List[Tuple]] = {}

    def stage_write(self, key: str, value: Any, priority: int = 0):
        """
        暂存写入操作

        Args:
            key: 写入目标（如 "token", "logprob"）
            value: 要写入的值
            priority: 写入优先级（用于控制写入顺序）
        """
        if key not in self._staged_writes:
            self._staged_writes[key] = []
        self._staged_writes[key].append((value, priority))

    def apply_staged_writes(self):
        """
        批量执行所有暂存的写入操作

        将多次写入合并为一次操作，减少 kernel 启动

        注意：对于不同的 key，会分别调用 _batch_write。
        要真正减少 kernel 启动次数，需要在 _batch_write 实现中确保
        所有相同类型的更新合并为一次 kernel 调用。
        """
        for key, writes in self._staged_writes.items():
            # 按优先级排序
            writes.sort(key=lambda x: x[1])
            values = [v for v, _ in writes]
            self._batch_write(key, values)
        self._staged_writes.clear()

    def _batch_write(self, key: str, values: List[Any]):
        """
        实际的批量写入操作（由子类实现）

        关键要求：此方法内部必须确保只启动一次 kernel

        Args:
            key: 写入目标
            values: 要写入的值列表

        示例实现：
            def _batch_write(self, key: str, values: List[Any]):
                if key == "token":
                    # 将所有 token 更新收集后，一次 kernel 写入
                    req_indices = [idx for idx, _ in values]
                    tokens = [token for _, token in values]
                    paddle.scatter_assign(self.token_buffer, req_indices, tokens)
        """
        raise NotImplementedError
```

#### 应用到 CacheManager

```python
# components/cache_manager.py

class CacheManager(StagedWriteMixin):
    def __init__(self, fd_config: FDConfig, model):
        super().__init__()
        self.fd_config = fd_config
        self.model = model
        self.total_block_num = 0

    def allocate_blocks(self, req_idx: int, num_blocks: int):
        """暂存 block 分配（不立即执行）"""
        self.stage_write("block_allocation", (req_idx, num_blocks))

    def update_block_table(self, req_idx: int, block_table: paddle.Tensor):
        """暂存 block table 更新（不立即执行）"""
        self.stage_write("block_table_update", (req_idx, block_table))

    def _batch_write(self, key: str, values: List[Any]):
        """实际执行批量写入（内部确保只启动一次 kernel）"""
        if key == "block_allocation":
            # 将所有 block 分配收集后，一次 kernel 写入
            req_indices = [idx for idx, _ in values]
            num_blocks = [blocks for _, blocks in values]
            self._do_batch_allocate_blocks(req_indices, num_blocks)
        elif key == "block_table_update":
            # 将所有 block table 更新收集后，一次 kernel 写入
            req_indices = [idx for idx, _ in values]
            block_tables = [table for _, table in values]
            self._do_batch_update_block_tables(req_indices, block_tables)
```

#### 应用到 OutputHandler（采样）

```python
# components/output_handler.py

class OutputHandler(StagedWriteMixin):
    def update_token(self, req_idx: int, token: int):
        """暂存 token 更新"""
        self.stage_write("token", (req_idx, token))

    def update_logprob(self, req_idx: int, logprob: float):
        """暂存 logprob 更新"""
        self.stage_write("logprob", (req_idx, logprob))

    def _batch_write(self, key: str, values: List[Any]):
        """实际执行批量写入（内部确保只启动一次 kernel）"""
        if key == "token":
            # 将所有 token 更新收集后，一次 kernel 写入
            req_indices = [idx for idx, _ in values]
            tokens = [t for _, t in values]
            self._do_batch_update_tokens(req_indices, tokens)
        elif key == "logprob":
            # 将所有 logprob 更新收集后，一次 kernel 写入
            req_indices = [idx for idx, _ in values]
            logprobs = [l for _, l in values]
            self._do_batch_update_logprobs(req_indices, logprobs)
```

#### 使用流程

```python
# Stage 阶段：暂存所有写入（不立即执行）
cache_manager.stage_write("block_allocation", (req_idx=1, num_blocks=5))
cache_manager.stage_write("block_allocation", (req_idx=2, num_blocks=3))
cache_manager.stage_write("block_table_update", (req_idx=1, block_table=...))

output_handler.stage_write("token", (req_idx=1, token=42))
output_handler.stage_write("token", (req_idx=2, token=128))

# Apply 阶段：一次性提交所有写入
# 注意：不同 key 会分别调用 _batch_write，但同一 key 的多次写入合并为一次调用
cache_manager.apply_staged_writes()
output_handler.apply_staged_writes()
```

#### 预期收益

| 场景 | 优化前 | 优化后 | 说明 |
|------|--------|--------|------|
| 同一类型的多次更新（如所有 token） | N 次 kernel 启动 | 1 次 kernel 启动 | 通过 _batch_write 合并 |
| 不同类型的更新 | 分别处理 | 仍需分别处理 | 不同 key 无法合并 |
| 混合状态更新 | 多次独立操作 | 可按类型分批处理 | 提升整体吞吐 |

---

### 2. 统一组件生命周期方法

**目标**: 为所有组件定义统一的生命周期方法模式，降低维护成本，提高代码一致性。

**参考实现**: FastDeploy 已有部分生命周期方法实现，但命名不统一。

#### 设计方案

```python
# components/base_component.py

class BaseComponent:
    """
    组件基类，定义统一的生命周期方法

    所有 GPUModelRunner 组件应继承此类或遵循此接口
    """

    # ========== 请求生命周期方法 ==========

    def add_request(self, req_idx: int, request: Request) -> None:
        """
        添加请求到组件

        Args:
            req_idx: 请求索引
            request: 请求对象
        """
        pass

    def remove_request(self, req_idx: int) -> None:
        """
        从组件移除请求

        Args:
            req_idx: 要移除的请求索引
        """
        pass

    def reset_cache(self) -> None:
        """重置组件缓存（如适用）"""
        pass

    # ========== 执行方法 ==========

    def __call__(self, **kwargs) -> Any:
        """
        执行组件功能（可选，用于纯函数式组件）

        默认实现：返回 None，子类可覆盖实现具体逻辑

        Args:
            **kwargs: 执行参数

        Returns:
            组件执行结果（默认 None）

        注意：
            - 此方法默认不抛出异常，允许子类选择性实现
            - 大多数有状态的组件不需要实现此方法
            - 主要用于纯函数式组件，如某些处理器或转换器
        """
        return None

    # ========== 状态查询方法（可选）==========

    def get_status(self) -> Dict[str, Any]:
        """
        获取组件状态（可选）

        Returns:
            组件状态字典
        """
        return {}
```

#### 应用到 InputManager

```python
# components/input_manager.py

class InputManager(BaseComponent):
    """
    输入管理组件，遵循统一的生命周期方法
    """

    def add_request(self, req_idx: int, request: Request) -> None:
        """添加请求到输入管理器"""
        # 原有 insert_tasks 的逻辑
        self._insert_task_to_batch(req_idx, request)

    def remove_request(self, req_idx: int) -> None:
        """从输入管理器移除请求"""
        self._remove_task_from_batch(req_idx)

    def reset_cache(self) -> None:
        """重置输入缓存"""
        self.share_inputs.reset_cache()

    # 保留原有方法（向后兼容）
    def insert_tasks(self, req_dicts: List[Request], ...):
        """兼容原有接口"""
        for idx, req in enumerate(req_dicts):
            self.add_request(idx, req)
```

#### 应用到 CacheManager

```python
# components/cache_manager.py

class CacheManager(BaseComponent):
    """
    Cache 管理组件，遵循统一的生命周期方法
    """

    def add_request(self, req_idx: int, request: Request) -> None:
        """为请求分配 cache"""
        num_blocks = self._calculate_blocks_needed(request)
        self.allocate_blocks_for_request(req_idx, num_blocks)

    def remove_request(self, req_idx: int) -> None:
        """释放请求的 cache"""
        self.free_blocks_for_request(req_idx)

    def reset_cache(self) -> None:
        """重置所有 cache"""
        self.clear_cache()
```

#### 应用到 VisionProcessor

```python
# components/vision_processor.py

class VisionProcessor(BaseComponent):
    """
    视觉处理组件，遵循统一的生命周期方法
    """

    def add_request(self, req_idx: int, request: Request) -> None:
        """添加请求（提取视觉特征）"""
        mm_hash = self._get_mm_hash(request)
        if mm_hash not in self.encoder_cache:
            features = self.extract_vision_features(request.mm_inputs)
            self.encoder_cache[mm_hash] = features.detach().cpu()

    def remove_request(self, req_idx: int) -> None:
        """移除请求（清理相关状态）"""
        # VisionProcessor 可能不需要做任何事
        pass

    def reset_cache(self) -> None:
        """重置视觉特征缓存"""
        self.encoder_cache.clear()
```

#### 应用到 OutputHandler

```python
# components/output_handler.py

class OutputHandler(BaseComponent):
    """
    输出处理组件，遵循统一的生命周期方法
    """

    def add_request(self, req_idx: int, request: Request) -> None:
        """添加请求到输出处理"""
        # OutputHandler 可能不需要预初始化
        pass

    def remove_request(self, req_idx: int) -> None:
        """移除请求（清理输出状态）"""
        self._cleanup_output_state(req_idx)

    def reset_cache(self) -> None:
        """重置输出缓存"""
        # 清理异步输出队列等
        pass
```

#### 更新后的生命周期方法对比

| 组件 | 添加请求 | 移除请求 | 重置缓存 |
|------|----------|----------|----------|
| InputManager | `add_request()` ✅ | `remove_request()` ✅ | `reset_cache()` ✅ |
| VisionProcessor | `add_request()` ✅ | `remove_request()` ✅ | `reset_cache()` ✅ |
| CacheManager | `add_request()` ✅ | `remove_request()` ✅ | `reset_cache()` ✅ |
| OutputHandler | `add_request()` ✅ | `remove_request()` ✅ | `reset_cache()` ✅ |

---

### 3. Registry 注册模式

**目标**: 为 attention backend、sampling method、multimodal processor 等实现注册机制，支持插件扩展。

**参考实现**: FastDeploy 已有完善的 Registry 实现，可直接复用：
- `ModelRegistry` - 模型注册 (`fastdeploy/model_executor/models/model_base.py`)
- `MultimodalRegistry` - 多模态模型注册 (`fastdeploy/multimodal/registry.py`)
- `ReasoningParserManager` - 推理解析器注册
- `ToolParserManager` - 工具解析器注册

#### 复用现有实现

```python
# 已有的 Registry 实现

# 1. ModelRegistry - 模型注册
from fastdeploy.model_executor.models.model_base import ModelRegistry

# 2. MultimodalRegistry - 多模态模型注册
from fastdeploy.multimodal.registry import MultimodalRegistry

# 3. ReasoningParserManager - 推理解析器注册
from fastdeploy.reasoning.abs_reasoning_parsers import ReasoningParserManager
```

#### 新增 Registry 设计

```python
# components/attention_registry.py

class AttentionBackendRegistry:
    """
    Attention Backend 注册表

    支持动态注册不同的 attention 实现
    """
    _backends: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """注册 attention backend（装饰器风格）"""
        def decorator(backend_class):
            cls._backends[name] = backend_class
            return backend_class
        return decorator

    @classmethod
    def register_module(cls, name: str, backend_class: Type):
        """直接注册 attention backend"""
        cls._backends[name] = backend_class

    @classmethod
    def get(cls, name: str) -> Type:
        """获取已注册的 backend"""
        if name not in cls._backends:
            raise ValueError(f"Attention backend '{name}' not found")
        return cls._backends[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的 backends"""
        return list(cls._backends.keys())
```

#### 使用示例

```python
# 注册 attention backend
@AttentionBackendRegistry.register("flash_attn")
class FlashAttentionBackend:
    def __init__(self, config):
        self.config = config

    def forward(self, *args, **kwargs):
        # Flash attention 实现
        pass

# 使用注册的 backend
backend = AttentionBackendRegistry.get("flash_attn")
attn_layer = backend(config)
```

---

### 4. share_inputs 类型安全化

**目标**: 将 `share_inputs` 从字典风格改为类型安全的 Dataclass，减少运行时错误。

**参考实现**: FastDeploy 已广泛使用 `@dataclass`，如：
- `SamplingParams` (`fastdeploy/engine/sampling_params.py`)
- `RequestMetrics` (`fastdeploy/engine/request.py`)
- `ModelInfo` (`fastdeploy/model_executor/models/model_base.py`)

#### 设计方案

```python
# worker/input_batch.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import paddle

@dataclass
class InputBatch:
    """
    类型安全的输入批次数据容器

    使用 dataclass 提供编译时类型检查和文档
    """

    # ========== 模型输入 ==========
    input_ids: paddle.Tensor = field(default_factory=lambda: paddle.zeros([0], dtype='int64'))
    positions: paddle.Tensor = field(default_factory=lambda: paddle.zeros([0], dtype='int64'))

    # ========== 多模态输入 ==========
    image_features: Optional[paddle.Tensor] = None
    image_features_list: Optional[List[paddle.Tensor]] = None
    rope_emb: Optional[paddle.Tensor] = None

    # ========== 序列信息 ==========
    seq_lens: paddle.Tensor = field(default_factory=lambda: paddle.zeros([0], dtype='int32'))
    seq_lens_encoder: paddle.Tensor = field(default_factory=lambda: paddle.zeros([0], dtype='int32'))
    seq_lens_decoder: paddle.Tensor = field(default_factory=lambda: paddle.zeros([0], dtype='int32'))

    # ========== KV Cache ==========
    block_tables: paddle.Tensor = field(default_factory=lambda: paddle.zeros([0, 0], dtype='int32'))
    caches: Optional[Dict[str, paddle.Tensor]] = None

    # ========== 采样参数 ==========
    top_p: Optional[List[float]] = None
    top_k: Optional[List[int]] = None
    temperature: Optional[List[float]] = None

    # ========== 其他状态 ==========
    stop_flags: paddle.Tensor = field(default_factory=lambda: paddle.zeros([0], dtype='int32'))

    # ========== 访问控制方法 ==========

    def set_input_ids(self, value: paddle.Tensor) -> None:
        """设置 input_ids（显式 setter，便于访问控制）"""
        self.input_ids = value

    def set_image_features(self, value: paddle.Tensor) -> None:
        """设置 image_features"""
        self.image_features = value

    def get_image_features_list(self) -> List[paddle.Tensor]:
        """获取 image_features_list（确保类型安全）"""
        if self.image_features_list is None:
            return []
        return self.image_features_list

    def condense(self) -> None:
        """压缩输入批次，保留运行的请求"""
        # 实现保持不变
        pass

    def reset_cache(self) -> None:
        """重置所有共享输入（重置所有字段为初始状态）"""
        # 重置所有字段
        pass
```

#### 更新访问控制表

| 字段 | 写入者 | 类型 | 访问控制 |
|------|--------|------|----------|
| `input_ids` | InputManager.set_input_ids() | `paddle.Tensor` | setter 方法控制 |
| `image_features_list` | VisionProcessor | `List[paddle.Tensor]` | getter 方法控制 |
| `image_features` | InputManager.set_image_features() | `paddle.Tensor` | setter 方法控制 |
| `block_tables` | CacheManager | `paddle.Tensor` | 直接赋值 |
| `top_p` | InputManager | `List[float]` | 直接赋值 |

---

### 5. 进程分离架构（长期规划）

**目标**: 将 CPU 密集型组件分离到独立进程，减少对 GPU 推理进程的干扰。

**参考实现**: SGLang 已实现进程分离架构，可作为参考。

#### 架构设计

```
┌─────────────────────────────────────────────────────┐
│                    Engine Service                    │
│                                                      │
│  ┌──────────────┐         ┌──────────────────┐      │
│  │ Tokenizer   │─────────▶│  Scheduler       │      │
│  │ Manager     │ ZMQ IPC │  (batching)     │      │
│  │ (CPU密集)   │         │                  │      │
│  └──────────────┘         └────────┬─────────┘      │
│                                   │                 │
│                          ┌──────────▼──────────┐      │
│                          │  ModelWorker      │      │
│                          │  (GPU推理)       │      │
│                          └───────────────────┘      │
│                                   │                 │
│                          ┌──────────▼──────────┐      │
│                          │ Detokenizer      │      │
│                          │ Manager          │      │
│                          │ (CPU密集)        │      │
│                          └───────────────────┘      │
└─────────────────────────────────────────────────────┘
```

#### 阶段规划

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| 阶段1 | 实现 TokenizerManager 独立进程 | 中 |
| 阶段2 | 实现 DetokenizerManager 独立进程 | 中 |
| 阶段3 | 优化 ZMQ IPC 通信 | 低 |

#### 通信协议

```python
# worker/tokenizer_ipc.py

class TokenizerManager:
    """
    Tokenizer 管理器，运行在独立进程中
    """
    def __init__(self):
        self.zmq_server = ZmqServer(mode=zmq.REP)

    def tokenize_batch(self, requests: List[str]) -> List[List[int]]:
        """批量 tokenize"""
        results = []
        for text in requests:
            tokens = self._do_tokenize(text)
            results.append(tokens)
        return results

# GPUModelRunner 中使用
class GPUModelRunner:
    def get_tokenized_input(self, requests: List[str]) -> List[List[int]]:
        """通过 IPC 调用 tokenizer"""
        return self.tokenizer_client.tokenize_batch(requests)
```

---

### 实施优先级

#### 第一阶段（立即实施）

1. **统一组件生命周期方法** - 降低维护成本
2. **share_inputs 类型安全化** - 减少运行时错误

#### 第二阶段（短期实施）

3. **Registry 注册模式** - 支持插件扩展
4. **Staged Writes 模式** - 性能优化

#### 第三阶段（长期规划）

5. **进程分离架构** - 长期架构优化

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

share_inputs 是多线程访问的共享数据结构，需要明确的线程安全机制：

```
┌─────────────────────────────────────────────────────────────────┐
│                     线程访问模式                                │
├─────────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                   │
│  │ 主推理线程    │         │ 异步输出线程  │                   │
│  │ (Worker)     │         │ (_async_...)  │                   │
│  │              │         │              │                   │
│  │  ┌───────┐   │  读/写  │   ┌───────┐   │   只读           │
│  │  │share  │◀──┼────────┼──▶│share  │   │                   │
│  │  │inputs │   │         │   │inputs │   │                   │
│  │  └───────┘   │         │   └───────┘   │                   │
│  │     ▲        │         │      ▲        │                   │
│  │     │        │         │      │        │                   │
│  │  写入操作     │         │   读取操作     │                   │
│  │  (各组件)    │         │  (输出处理)   │                   │
│  └──────────────┘         └──────────────┘                   │
│         │                          │                          │
│         │ 读写冲突保护              │                          │
│         └──────────────────────────┴──────────────────────────┘
│                              │                              │
│                              ▼                              │
│                    ┌─────────────────┐                      │
│                    │ ThreadSafeLock  │                      │
│                    │ (或读写锁)       │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

**线程安全设计选项**：

| 选项 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **无锁设计** | 使用 paddle.Tensor 的原子操作 | 性能最高 | 实现复杂，依赖框架支持 |
| **互斥锁** | 使用 threading.Lock 保护所有访问 | 简单可靠 | 性能影响大 |
| **读写锁** | 使用 threading.RWLock，支持多读单写 | 平衡性能和复杂度 | 需要谨慎使用 |
| **数据副本** | 异步线程读取时复制数据 | 避免锁开销 | 内存开销 |

**推荐方案：读写锁 + 约定访问模式**

```python
# worker/input_batch.py

import threading

class ThreadSafeInputBatch:
    """
    线程安全的输入批次数据容器

    使用读写锁模式：
    - 多个读操作可以并发执行
    - 写操作需要独占锁
    - 写操作优先级高于读操作（避免写饥饿）
    """

    def __init__(self, fd_config: FDConfig):
        self._data = InputBatch(fd_config)  # 实际数据容器
        self._lock = threading.RLock()  # 可重入锁，简化嵌套调用
        # 如果使用读写锁：
        # from fastdeploy.utils.rwlock import RWLock
        # self._rwlock = RWLock()

    def write(self, key: str, value: Any):
        """写入数据（需要独占锁）"""
        with self._lock:
            setattr(self._data, key, value)

    def read(self, key: str) -> Any:
        """读取数据（需要共享锁）"""
        with self._lock:
            return getattr(self._data, key)

    def snapshot(self) -> InputBatch:
        """创建数据快照（用于异步输出线程）"""
        with self._lock:
            # 返回数据的深拷贝或只读视图
            return self._data.copy()

# 使用示例
class OutputHandler(BaseComponent):
    def _async_output_busy_loop(self):
        """异步输出循环"""
        while True:
            output = self.async_output_queue.get()
            if output is None:  # 退出信号
                break

            # 使用快照而非直接访问 share_inputs，避免竞争
            input_snapshot = self.share_inputs.snapshot()
            self._process_output(output, input_snapshot)
```

**约定访问模式**：

1. **写操作**：只能由主推理线程执行，使用 `with input_batch._lock:` 保护
2. **读操作**：异步输出线程使用 `input_batch.snapshot()` 获取快照，避免竞争
3. **状态查询**：如 `exist_prefill()`、`not_need_stop()` 等由主线程调用，不需要锁

**关键原则**：
- `share_inputs` 主线程写，异步线程读（通过快照）
- 避免在异步线程中修改 `share_inputs`
- 避免在主线程持有锁时调用可能阻塞的异步操作

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
    ├─> 初始化共享状态 (share_inputs, encoder_cache)  ← 最先初始化
    ├─> 初始化 InputManager(share_inputs)              ← 依赖 share_inputs
    ├─> 初始化 OutputHandler()                         ← 无依赖
    │
    └─> 延迟初始化（需要 model 的组件，在 load_model() 中）：
GPUModelRunner.load_model()
    │
    ├─> model = self._get_model_instance()
    │
    ├─> VisionProcessor(model, encoder_cache, share_inputs)  ← 依赖两者
    ├─> CacheManager(model)                                  ← 依赖 model
    ├─> ProfileRunner(model)                                 ← 依赖 model
    ├─> InferenceFlow(model)                                 ← 依赖 model
    │
    └─> input_manager.set_vision_processor(vision_processor)  ← 设置引用
```

**初始化依赖关系**：
1. **share_inputs**: 无依赖，最先初始化
2. **encoder_cache**: 无依赖，与 share_inputs 同时初始化
3. **InputManager**: 依赖 share_inputs
4. **OutputHandler**: 无依赖
5. **VisionProcessor**: 依赖 model、encoder_cache、share_inputs
6. **CacheManager**: 依赖 model
7. **ProfileRunner**: 依赖 model
8. **InferenceFlow**: 依赖 model

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

### InputManager (8 个方法)
- `insert_tasks_v1`
- `insert_prefill_inputs`
- `insert_tasks` (统一接口，内部调用 v1/v0)
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

---

## 测试迁移策略

### 一、当前测试覆盖情况分析

#### 1.1 测试文件概览

| 文件 | 用例数 | 主要功能 |
|------|--------|----------|
| test_gpu_model_runner_e2e.py | 7 | 端到端 |
| test_gpu_model_runner_error_cases.py | 21 | 错误场景 |
| test_gpu_model_runner_p1_priority.py | 27 | P1功能 |
| test_gpu_model_runner_public.py | ~90 | 公开方法 |
| test_gpu_model_runner_public_init.py | ~3 | 初始化 |
| test_gpu_model_runner_public_simple.py | ~14 | 简单方法 |
| test_gpu_model_runner_public_vision_execute.py | ~30 | 视觉和执行 |

#### 1.2 公开方法覆盖统计

| 状态 | 数量 | 占比 | 说明 |
|------|------|------|------|
| ✅ 完整覆盖 | 18个方法 | 56% | 有实际测试用例，验证逻辑正确性 |
| ⚠️ 仅签名测试 | 14个方法 | 44% | 只验证方法能被调用，未验证逻辑 |
| ❌ 未覆盖 | 0个方法 | 0% | 无公开方法遗漏 |

**仅签名测试的方法列表**：
- `insert_prefill_inputs`
- `load_model`
- `initialize_forward_meta`
- `initialize_kv_cache`
- `capture_model`
- `capture_model_prefill_and_mixed`
- `vision_encoder_compile`
- `sot_warmup`
- `profile_run`
- `clear_parameters`
- `update_parameters`
- `padding_cudagraph_inputs`
- `extract_vision_features_ernie/qwen/paddleocr`
- `prepare_rope3d`

#### 1.3 已覆盖的典型场景

✅ **Speculative Decoding** - 7个测试用例
- 禁用状态、NgramProposer、MTPProposer 初始化
- execute_model 中的处理逻辑
- 不同 num_speculative_tokens 配置

✅ **Chunked Prefill** - 6个测试用例
- 大 prompt 分块处理
- 状态连续性
- max_chunked_prefill_len 约束

✅ **Prefix Caching** - 6个测试用例
- 启用/禁用状态
- cache_kvs_map 存储/检索/清空
- 与 prompt_logprobs 冲突检查

✅ **内存压力场景** - 4个测试用例
- KV cache 接近上限
- GPU↔CPU block 交换

✅ **视觉特征提取缓存** - 6个测试用例
- encoder_cache 命中/未命中
- 缓存驱逐
- Rope3D cache 准备

✅ **端到端流程** - 7个测试用例
- 完整 prefill → decode → stop 流程
- 多请求并发处理
- 采样参数流转

✅ **错误场景** - 21个测试用例
- 空批次、单 token 序列
- 无效参数、边界值

#### 1.4 测试质量问题

| 问题 | 严重性 | 说明 |
|------|--------|------|
| 仅签名测试 | 高 | 44%的方法只有 try-except 包裹的签名测试 |
| `__new__` 跳过初始化 | 中 | 手动设置属性，容易遗漏 |
| Mock 设置不完整 | 中 | 只验证返回值非 None，不验证正确性 |
| 代码错误 | 低 | 如 `is_pooling_mode` 应为 `is_pooling_model` |
| 断言过于宽泛 | 中 | 只验证 shape，不验证值 |
| 缺少真实执行测试 | 高 | 所有执行都是 mock |
| 测试文件过度分割 | 低 | 7个文件，难以维护 |

---

### 二、重构对测试的影响

#### 2.1 预期失败率分析

| 类别 | 预估失败数 | 占比 | 原因 |
|------|-----------|------|------|
| 方法直接调用 | ~40 | ~30% | 方法已迁移到组件，直接调用失败 |
| 属性访问 | ~20 | ~15% | 内部属性变为组件属性，访问路径变化 |
| Mock 依赖 | ~30 | ~25% | Mock 结构需要适配新组件结构 |
| 状态查询 | ~10 | ~8% | 状态查询从 Runner 委托到组件 |
| 其他 | ~5 | ~4% | 配置变化等 |

**总预期失败率：~70% 的测试用例需要修改**

#### 2.2 受影响的关键测试模式

**模式1：直接调用已迁移的方法**
```python
# 旧代码（会失败）
self.runner.extract_vision_features_ernie(vision_inputs)
self.runner._process_mm_features(request_list)
self.runner._postprocess(sampler_output, model_output, ...)

# 新代码（正确方式）
self.runner.vision_processor.extract_vision_features_ernie(vision_inputs)
self.runner.vision_processor.process_mm_features(request_list)
self.runner.output_handler.postprocess(sampler_output, model_output, ...)
```

**模式2：访问组件属性**
```python
# 旧代码（会失败）
self.runner.encoder_cache[mm_hash] = features
self.runner.share_inputs["image_features_list"] = [...]

# 新代码（正确方式）
self.runner.vision_processor.encoder_cache[mm_hash] = features
self.runner.input_manager.share_inputs["image_features_list"] = [...]
```

**模式3：Mock 组件方法**
```python
# 旧代码（需要调整）
self.runner.model.vision_encoder.return_value = paddle.zeros((10, 768))

# 新代码（正确方式）
self.runner.vision_processor.model.vision_encoder.return_value = paddle.zeros((10, 768))
# 或者创建完整的 VisionProcessor mock
```

**模式4：测试 execute_model**
```python
# 旧代码（测试 Runner 的 execute_model）
def test_execute_model_normal(self):
    self.runner.execute_model(batch, ...)

# 新代码（测试流程编排）
def test_execute_model_normal(self):
    # 需要确保所有组件都已正确初始化
    # InferenceFlow 作为无状态编排器，测试重点变为：
    # 1. 组件调用顺序是否正确
    # 2. 参数传递是否完整
    # 3. 错误处理是否完善
```

---

### 三、测试迁移策略

#### 3.1 方法迁移映射表

| 原方法（GPUModelRunner） | 新位置 | 测试修改方式 |
|-------------------------|--------|-------------|
| `insert_tasks_v1` | `input_manager.insert_tasks_v1` | 替换调用路径 |
| `insert_prefill_inputs` | `input_manager.insert_prefill_inputs` | 替换调用路径 |
| `_prepare_inputs` | `input_manager.prepare_inputs` | 替换调用路径 |
| `get_input_length_list` | `input_manager.get_input_length_list` | 替换调用路径，或委托测试 |
| `clear_requests` | `input_manager.clear_requests` | 替换调用路径 |
| `exist_prefill/decode` | `input_manager.exist_prefill/decode` | 保持委托调用（兼容） |
| `only_prefill/decode` | `input_manager.only_prefill/decode` | 保持委托调用（兼容） |
| `not_need_stop` | `input_manager.not_need_stop` | 保持委托调用（兼容） |
| `extract_vision_features*` | `vision_processor.*` | 替换调用路径 |
| `_process_mm_features` | `vision_processor.process_mm_features` | 替换调用路径 |
| `prepare_rope3d` | `vision_processor.prepare_rope3d` | 替换调用路径 |
| `_postprocess` | `output_handler.postprocess` | 替换调用路径 |
| `_save_model_output` | `output_handler.save_model_output` | 替换调用路径 |
| `_pool` | `output_handler.pool` | 替换调用路径 |
| `_get_prompt_logprobs_list` | `output_handler.get_prompt_logprobs_list` | 替换调用路径 |
| `clear_cache` | `cache_manager.clear_cache` | 保持委托调用（兼容） |
| `update_share_input_block_num` | `cache_manager.update_share_input_block_num` | 保持委托调用（兼容） |
| `initialize_kv_cache` | `cache_manager.initialize_kv_cache` | 替换调用路径 |
| `profile_run` | `profile_runner.profile_run` | 保持委托调用（兼容） |
| `sot_warmup` | `profile_runner.sot_warmup` | 保持委托调用（兼容） |
| `capture_model` | `profile_runner.capture_model` | 保持委托调用（兼容） |
| `execute_model` | `inference_flow.execute` | 保持委托调用（兼容） |

#### 3.2 测试用例分类与迁移

**类别 A：委托方法测试（无需修改）**

这些方法通过 Runner 委托到组件，外部接口保持不变：

```python
# 保持不变
self.runner.exist_prefill()
self.runner.exist_decode()
self.runner.clear_cache()
self.runner.profile_run()
self.runner.execute_model(batch, ...)
```

**类别 B：组件方法测试（修改调用路径）**

需要将调用路径更新为组件路径：

```python
# 迁移前
self.runner.extract_vision_features_ernie(vision_inputs)

# 迁移后
self.runner.vision_processor.extract_vision_features_ernie(vision_inputs)
```

**类别 C：状态查询测试（更新 Mock 结构）**

需要更新 Mock 的属性路径：

```python
# 迁移前
self.runner.encoder_cache[mm_hash] = features

# 迁移后
self.runner.vision_processor.encoder_cache[mm_hash] = features
```

**类别 D：集成测试（重构测试逻辑）**

需要重构为验证组件协作而非单一方法：

```python
# 迁移前：测试单一方法
def test_insert_tasks_v1(self):
    self.runner.insert_tasks_v1(req_dicts)
    self.assertEqual(len(self.runner.share_inputs["prompt_token_ids"]), 3)

# 迁移后：测试组件协作
def test_insert_tasks_v1(self):
    self.runner.input_manager.insert_tasks_v1(req_dicts)
    self.assertEqual(len(self.runner.input_manager.share_inputs.prompt_token_ids), 3)
    # 新增：验证 VisionProcessor 是否被正确调用
    if self._has_mm_requests(req_dicts):
        self.runner.vision_processor.process_mm_features.assert_called()
```

#### 3.3 测试文件重组建议

建议将 7 个测试文件按组件架构重组：

| 新文件 | 内容 | 说明 |
|--------|------|------|
| `test_gpu_model_runner_basic.py` | Runner 基础功能、状态查询、生命周期 | 合并 public 和 public_simple |
| `test_input_manager.py` | InputManager 所有方法 | 独立组件测试 |
| `test_vision_processor.py` | VisionProcessor 所有方法 | 独立组件测试 |
| `test_output_handler.py` | OutputHandler 所有方法 | 独立组件测试 |
| `test_cache_manager.py` | CacheManager 所有方法 | 独立组件测试 |
| `test_inference_flow.py` | InferenceFlow 流程编排测试 | 验证组件调用顺序 |
| `test_gpu_model_runner_integration.py` | 端到端集成测试 | 真实场景验证 |
| `test_gpu_model_runner_e2e.py` | 端到端真实执行测试 | 使用真实模型 |

---

### 四、测试迁移步骤

#### 第一阶段：准备迁移环境（1-2 天）

1. **创建新测试文件结构**
   ```bash
   mkdir -p tests/worker/components
   mkdir -p tests/worker/flows
   ```

2. **为每个组件创建测试基类**
   ```python
   # tests/worker/components/test_base_component.py
   class ComponentTestBase(unittest.TestCase):
       """组件测试基类，提供通用的 Mock 设置"""
       def setUp(self):
           self.mock_fd_config = self._create_mock_fd_config()
           self.mock_model = self._create_mock_model()

       def _create_mock_fd_config(self):
           # 统一的 Mock 设置
           pass

       def _create_mock_model(self):
           # 统一的 Mock 设置
           pass
   ```

#### 第二阶段：迁移组件测试（3-5 天）

按组件顺序迁移：

1. **InputManager 测试**
   - 迁移 `test_insert_tasks_v1` → `test_input_manager.py`
   - 迁移 `test_insert_prefill_inputs` → `test_input_manager.py`
   - 迁移 `test_get_input_length_list` → `test_input_manager.py`
   - 迁移 `test_clear_requests` → `test_input_manager.py`

2. **VisionProcessor 测试**
   - 迁移所有 vision 相关测试 → `test_vision_processor.py`
   - 补充真实的特征提取测试（当前仅签名测试）

3. **OutputHandler 测试**
   - 迁移所有输出处理测试 → `test_output_handler.py`
   - 包括 postprocess, save_model_output, pool 等

4. **CacheManager 测试**
   - 迁移 cache 相关测试 → `test_cache_manager.py`
   - 包括 initialize_kv_cache, clear_cache 等

#### 第三阶段：迁移流程测试（2-3 天）

1. **InferenceFlow 测试**
   - 创建 `test_inference_flow.py`
   - 重点验证：
     - 组件调用顺序
     - 参数传递完整性
     - 错误处理流程

2. **集成测试**
   - 更新 `test_gpu_model_runner_integration.py`
   - 验证组件间协作

#### 第四阶段：更新 Runner 测试（1-2 天）

1. **保留委托测试**
   - exist_prefill/decode
   - clear_cache
   - profile_run
   - execute_model

2. **移除已迁移的测试**
   - 刄件方法直接调用测试
   - 内部属性访问测试

3. **更新 Mock 结构**
   - 适配新的组件结构

#### 第五阶段：补充真实执行测试（3-5 天）

补充当前缺失的真实执行测试：

1. **真实模型前向传播**
   ```python
   def test_execute_model_normal_with_real_model(self):
       """使用真实模型前向传播"""
       runner = self._create_real_runner_with_small_model()
       requests = self._create_test_requests()
       output = runner.execute_model(requests)
       self._verify_output_correctness(output)
   ```

2. **Speculative decoding 真实测试**
   ```python
   def test_mtp_proposal_correctness(self):
       """验证 MTP 提案的正确性"""
       runner = self._create_runner_with_mtp()
       output = runner.execute_model(requests)
       self._verify_mtp_output(output)
   ```

3. **视觉特征提取完整测试**
   ```python
   def test_ernie_feature_extraction_with_cache(self):
       """测试 Ernie 特征提取和缓存"""
       runner = self._create_runner_with_vision()
       vision_inputs = self._create_vision_inputs()
       output = runner.execute_model(requests_with_vision)
       self._verify_features_extracted(output)
       self._verify_cache_hit_on_duplicate_input()
   ```

---

### 五、测试质量改进建议

#### 5.1 移除无效的 try-except 包装

```python
# 不推荐（当前代码）
def test_initialize_forward_meta_basic(self):
    try:
        self.runner.initialize_forward_meta()
        self.runner.forward_meta.assert_called_once()
    except Exception:
        pass  # 异常被吞掉，测试总是"通过"

# 推荐
def test_initialize_forward_meta_basic(self):
    with patch.object(self.runner, 'forward_meta') as mock_meta:
        self.runner.initialize_forward_meta()
        self.assertTrue(mock_meta.called)
```

#### 5.2 提供完整的 Mock 设置

```python
# 组件测试基类提供统一 Mock
class ComponentTestBase(unittest.TestCase):
    def _create_complete_mock_runner(self):
        """创建完整 mock 的 runner 实例"""
        mock_fd_config = Mock()
        mock_fd_config.model_config = self._create_mock_model_config()
        mock_fd_config.cache_config = self._create_mock_cache_config()

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = mock_fd_config
        runner.model = Mock()
        runner.share_inputs = InputBatch(mock_fd_config)
        runner.encoder_cache = {}

        # 创建组件 Mock
        runner.input_manager = Mock(spec=InputManager)
        runner.vision_processor = Mock(spec=VisionProcessor)
        runner.output_handler = Mock(spec=OutputHandler)
        runner.cache_manager = Mock(spec=CacheManager)
        runner.profile_runner = Mock(spec=ProfileRunner)
        runner.inference_flow = Mock(spec=InferenceFlow)

        return runner
```

#### 5.3 增强断言验证

```python
# 不推荐（当前代码）
self.assertIn(mm_hash, self.runner.encoder_cache)
self.assertEqual(
    self.runner.encoder_cache[mm_hash].shape, cached_features.shape
)  # 只验证 shape，不验证值

# 推荐
self.assertIn(mm_hash, self.runner.encoder_cache)
cached = self.runner.encoder_cache[mm_hash]
self.assertEqual(cached.shape, expected_shape)
# 验证特征提取的正确性
self.assertLess(paddle.mean(paddle.abs(cached - expected_features)), 1e-3)
```

---

### 六、迁移检查清单

#### 6.1 迁移前准备

- [ ] 创建新的测试文件结构
- [ ] 创建组件测试基类
- [ ] 备份现有测试

#### 6.2 组件测试迁移

- [ ] InputManager 测试完成
- [ ] VisionProcessor 测试完成
- [ ] OutputHandler 测试完成
- [ ] CacheManager 测试完成
- [ ] ProfileRunner 测试完成

#### 6.3 流程测试迁移

- [ ] InferenceFlow 测试完成
- [ ] 集成测试完成

#### 6.4 Runner 测试更新

- [ ] 委托方法测试更新
- [ ] Mock 结构更新
- [ ] 已迁移测试删除

#### 6.5 质量验证

- [ ] 所有测试通过
- [ ] 测试覆盖率不下降
- [ ] 性能无明显退化
- [ ] 真实执行测试补充完成

---

### 七、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 测试失败率高 | 迁移时间长 | 1. 分阶段迁移 2. 提供兼容层 3. 充分的 Mock 设置 |
| Mock 复杂度增加 | 维护成本上升 | 1. 统一 Mock 基类 2. 提供测试工具函数 |
| 组件集成测试缺失 | 组件间问题未发现 | 1. 优先完成集成测试 2. 补充真实执行测试 |
| 性能回归测试缺失 | 性能下降未发现 | 1. 建立性能基准 2. 每次重构后运行基准测试 |

---

### 八、总结

#### 8.1 当前测试状态

| 维度 | 状态 | 说明 |
|------|------|------|
| 方法覆盖 | ⚠️ 100%签名, 56%逻辑 | 所有方法有签名测试，56%有逻辑验证 |
| 场景覆盖 | ⚠️ 约40% | 基本场景有覆盖，复杂场景缺失 |
| Corner case | ⚠️ 部分覆盖 | 边界值有部分测试，错误场景较全 |
| 集成测试 | ❌ 不足 | e2e测试存在但全为mock |
| 性能测试 | ❌ 缺失 | 无基准测试 |
| 并发测试 | ❌ 缺失 | 无多线程/多进程测试 |

#### 8.2 迁移预期

- **预期失败率**: ~70% 的测试用例需要修改
- **迁移工作量**: 10-15 天（按阶段进行）
- **测试质量提升**:
  - 组件可独立测试
  - Mock 结构更清晰
  - 真实执行测试补充
  - 集成测试完善

#### 8.3 优先改进项

| 优先级 | 改进项 | 预期收益 |
|--------|---------|----------|
| P0 | 移除无效的try-except包装 | 提高测试有效性 |
| P0 | 补充真实执行测试 | 验证实际推理流程 |
| P0 | 完善视觉特征提取测试 | 确保多模态正确性 |
| P1 | 添加集成测试 | 发现组件间问题 |
| P1 | 补充分布式场景测试 | 确保TP/PP/EP正确性 |
| P2 | 建立性能基准 | 防止性能退化 |
| P2 | 添加并发测试 | 发现线程安全问题 |
