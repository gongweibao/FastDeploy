# GPUModelRunner 多模态数据处理流程

## 概述

GPUModelRunner 是 FastDeploy 中负责模型执行的核心类，支持纯文本和多模态 (MM) 数据的处理和推理。本文档聚焦于多模态数据的处理流程，包括图像预处理、视觉特征提取、编码器缓存和模型前向传播中的多模态路径。纯文本数据处理流程请参见 `gpu_model_runner_data_flow.md`。

> **相关文档**：[gpu_model_runner_data_flow.md](gpu_model_runner_data_flow.md)（完整数据流转）、[gpu_model_runner_refactoring.md](gpu_model_runner_refactoring.md)（重构方案）
>
> **术语约定**：本系列文档中，"提取"指视觉编码器对原始图像进行特征提取（在 insert_tasks 阶段完成）；"拼接"指将已缓存的 image_features_list 合并为 image_features（在 prepare_inputs 阶段完成）。

---

## 数据处理流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           GPUModelRunner 数据处理流程                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 初始化阶段 (init)                                                                                         │   │
│  │   ├── enable_mm: 检查是否启用多模态支持                                                                       │   │
│  │   ├── encoder_cache: 初始化编码器缓存 (可选)                                                                  │   │
│  │   └── _init_image_preprocess(): 初始化图像预处理器 (DataProcessor)                                          │   │
│  │       ├── 加载 image_preprocessor                                                                             │   │
│  │       ├── 预计算 image_mean/std_tensor                                                                       │   │
│  │       └── 配置 AMP black/white 列表                                                                           │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                                                           │
│                                      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 2. 插入任务 (insert_tasks_v1 / insert_prefill_inputs)                                                       │   │
│  │   遍历 Request 列表，为每个请求准备数据                                                                      │   │
│  │                                                                                                              │   │
│  │   ├── 获取 batch_idx = share_inputs.get_index_by_batch_id(request.idx)                                     │   │
│  │   │                                                                                                          │   │
│  │   ├── 【文本数据处理】                                                                                       │   │
│  │   │   ├── input_ids = request.prompt_token_ids                                                              │   │
│  │   │   ├── prompt_ids = request.prompt_token_ids                                                             │   │
│  │   │   ├── seq_lens_encoder/decoder                                                                         │   │
│  │   │   ├── block_tables: KV cache 块表                                                                       │   │
│  │   │   ├── sampling_params: temperature, top_p, top_k, penalties 等                                        │   │
│  │   │   └── stop_tokens/bad_words_token_ids                                                                   │   │
│  │   │                                                                                                          │   │
│  │   └── 【多模态数据处理】 (enable_mm=True 时)                                                                 │   │
│  │       ├── _preprocess_mm_task(): 转换 MM 输入格式                                                           │   │
│  │       │   ├── input_ids → paddle.tensor (int64)                                                             │   │
│  │       │   ├── images → paddle.tensor (uint8/bfloat16)                                                       │   │
│  │       │   ├── grid_thw → paddle.tensor (int64) [grid_t, grid_h, grid_w]                                    │   │
│  │       │   ├── position_ids → paddle.tensor (int64)                                                          │   │
│  │       │   └── image_type_ids/token_type_ids → paddle.tensor (int64)                                        │   │
│  │       │                                                                                                      │   │
│  │       ├── extract_vision_features(): 提取视觉特征                                                           │   │
│  │       │   ├── [Ernie-VL]: vision_model.extract_feature + resampler_model                                   │   │
│  │       │   ├── [Qwen-VL]: visual.extract_feature                                                            │   │
│  │       │   └── [PaddleOCR]: visual + projector                                                               │   │
│  │       │                                                                                                      │   │
│  │       ├── prepare_rope3d(): 准备 3D 旋转位置编码                                                            │   │
│  │       │   └── 调用 get_rope_3d() 生成 rope_emb                                                              │   │
│  │       │                                                                                                      │   │
│  │       └── share_inputs["image_features"] = 提取的视觉特征                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                                                           │
│                                      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 3. 多模态特征处理 (_process_mm_features)                                                                    │   │
│  │   【V0 和 V1 调度均会调用，在存在图像的 PREFILL 任务时执行】                                                │   │
│  │   【V0 调度：在 insert_prefill_inputs 中调用】                                                              │   │
│  │   【V1 调度：在 insert_tasks_v1 末尾调用（L896）】                                                          │   │
│  │                                                                                                              │   │
│  │   ├── 遍历 request_list (仅 PREFILL 任务)                                                                   │   │
│  │   │   ├── 构建 multi_vision_inputs 字典:                                                                    │   │
│  │   │   │   ├── images_lst: 批量图像数据                                                                      │   │
│  │   │   │   ├── grid_thw_lst: 网格尺寸列表                                                                    │   │
│  │   │   │   ├── vit_position_ids_lst: ViT 位置 ID                                                              │   │
│  │   │   │   ├── cu_seqlens: 累积序列长度                                                                      │   │
│  │   │   │   ├── encoder_cache_info: 编码器缓存信息                                                            │   │
│  │   │   │   ├── feature_position_list: 特征位置列表                                                           │   │
│  │   │   │   └── mm_hashes: 多模态哈希值 (用于缓存)                                                             │   │
│  │   │   │                                                                                                      │   │
│  │   │   └── 【编码器缓存】 (encoder_cache 存在时)                                                              │   │
│  │   │       ├── 检查 mm_hash 是否在 encoder_cache 中                                                           │   │
│  │   │       ├── 命中: 从缓存中加载特征                                                                        │   │
│  │   │       └── 未命中: 提取新特征并存入缓存                                                                   │   │
│  │   │                                                                                                          │   │
│  │   └── extract_vision_features(multi_vision_inputs)                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                                                           │
│                                      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 4. 输入重排序 (_process_reorder)                                                                           │   │
│  │   ├── enable_ids_reorder: 检查是否启用 ID 重排序                                                           │   │
│  │   ├── share_inputs.condense(): 压缩输入数据                                                                   │   │
│  │   └── reorder_split_prefill_and_decode(): 分离 prefill 和 decode token                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                                                           │
│                                      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 5. 准备模型输入 (_prepare_inputs)                                                                           │   │
│  │   ├── 【V1 调度时】                                                                                         │   │
│  │   │   ├── concat image_features_list → image_features                                                      │   │
│  │   │   └── recover_decode_task(): 恢复 decode 任务状态                                                       │   │
│  │   │                                                                                                          │   │
│  │   ├── pre_process(): 移除 padding，生成:                                                                     │   │
│  │   │   ├── ids_remove_padding: 移除 padding 的 token IDs                                                     │   │
│  │   │   ├── batch_id_per_token: 每个 token 对应的 batch ID                                                    │   │
│  │   │   ├── cu_seqlens_q/k: Query/Key 的累积序列长度                                                          │   │
│  │   │   └── cu_seqlens_q_output: 输出的累积序列长度                                                           │   │
│  │   │                                                                                                          │   │
│  │   ├── initialize_forward_meta(): 初始化 ForwardMeta                                                        │   │
│  │   │   ├── ids_remove_padding                                                                               │   │
│  │   │   ├── rotary_embs (rope_emb)                                                                           │   │
│  │   │   ├── seq_lens_encoder/decoder                                                                         │   │
│  │   │   ├── block_tables + caches (KV cache)                                                                │   │
│  │   │   └── attn_backend.init_attention_metadata()                                                           │   │
│  │   │                                                                                                          │   │
│  │   └── 创建 SamplingMetadata:                                                                               │   │
│  │       ├── temperature, top_p, top_k, min_p                                                                │   │
│  │       ├── penalties (frequency, presence, repetition)                                                      │   │
│  │       └── logits_processors                                                                                │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                                                           │
│                                      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 6. 模型执行 (_preprocess_and_execute_model)                                                                 │   │
│  │   ├── padding_cudagraph_inputs(): 为 CUDA Graph 准备 padding 输入                                            │   │
│  │   │                                                                                                          │   │
│  │   ├── 【多模态前向】 (enable_mm=True)                                                                       │   │
│  │   │   └── model(                                                                                           │   │
│  │   │       ids_remove_padding,                                                                               │   │
│  │   │       image_features,       ← 视觉特征嵌入                                                              │   │
│  │   │       forward_meta                                                                                       │   │
│  │   │   )                                                                                                      │   │
│  │   │                                                                                                          │   │
│  │   └── 【仅文本前向】 (enable_mm=False)                                                                       │   │
│  │       └── model(                                                                                           │   │
│  │           ids_remove_padding,                                                                               │   │
│  │           forward_meta                                                                                       │   │
│  │       )                                                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                                                           │
│                                      ▼                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ 7. 后处理 (_postprocess)                                                                                     │   │
│  │   ├── rebuild_padding(): 恢复 padding 格式                                                                  │   │
│  │   ├── compute_logits(): 计算 logits                                                                         │   │
│  │   ├── sampler.sample(): 采样下一个 token                                                                    │   │
│  │   └── post_process(): 后处理并保存输出                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 关键数据结构

| 数据结构 | 说明 | 类型 |
|---------|------|------|
| `share_inputs["input_ids"]` | 输入 token IDs | Tensor[int64] |
| `share_inputs["image_features"]` | 提取的视觉特征 | Tensor[bfloat16] |
| `share_inputs["rope_emb"]` | 3D 旋转位置编码 | List[Tensor] |
| `share_inputs["image_features_list"]` | 按请求存储的特征列表 | List[Tensor\|int] |
| `encoder_cache` | 编码器特征缓存 | Dict[mm_hash, Tensor] |
| `forward_meta` | 前向传播元数据 | ForwardMeta |
| `SamplingMetadata` | 采样元数据 | SamplingMetadata |

---

## 详细说明

### 1. 初始化阶段

类初始化时，如果启用多模态支持 (`enable_mm=True`)，会执行以下操作：

```python
# 位于 fastdeploy/worker/gpu_model_runner.py:152-177
if self.enable_mm:
    if "ernie" in self.fd_config.model_config.model_type:
        self._init_image_preprocess()

    # AMP 配置
    self.amp_black = ["reduce_sum", "c_softmax_with_cross_entropy", ...]
    self.amp_white = ["lookup_table", "flash_attn", "matmul", ...]

    # 编码器缓存
    if self.cache_config.max_encoder_cache > 0:
        self.encoder_cache: dict[str, paddle.Tensor] = {}
```

### 2. 任务插入与输入预处理

#### 文本数据处理
在 `insert_prefill_inputs()` 方法中处理文本数据：

```python
# 位于 fastdeploy/worker/gpu_model_runner.py:970-1019
self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
self.share_inputs["prompt_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(request.block_tables)
```

#### 多模态数据处理

##### `_preprocess_mm_task()` 输入输出说明

**输入**: `request.multimodal_inputs`，一个字典，其结构取决于模型类型：

| 字段 | 类型 | 说明 | 来源 |
|------|------|------|------|
| `images` | List[bytes] 或 List[np.ndarray] | 原始图像数据 | 用户请求 |
| `grid_thw` | List[List[int]] | 网格尺寸 [grid_t, grid_h, grid_w] | 图像预处理计算 |
| `position_ids` | List[int] | token 级别的位置 ID | tokenizer 生成 |
| `image_type_ids` | List[int] | 图像/文本类型标识 | tokenizer 生成 |
| `mm_hashes` | List[str] | 多模态内容的哈希值，用于缓存匹配 | 内容哈希计算 |

**输出**: 转换后的 paddle.Tensor 字典，供 `extract_vision_features()` 使用。

对于多模态请求，额外处理图像和位置信息：

```python
# 位于 fastdeploy/worker/gpu_model_runner.py:1008-1029
if self.enable_mm:
    inputs = self._preprocess_mm_task(request.multimodal_inputs)
    if inputs.get("images") is not None:
        self.share_inputs["image_features"] = self.extract_vision_features(inputs)

    # 准备 RoPE3D 位置编码
    self.share_inputs["rope_emb"][idx : idx + 1, :] = self.prepare_rope3d(
        position_ids, [request.get("max_tokens", 2048)], [0, position_ids.shape[0]]
    )[0]
```

### 3. 视觉特征提取

`extract_vision_features()` 根据模型类型调用相应的提取方法：

| 模型类型 | 提取方法 | 组件 |
|---------|---------|------|
| Ernie-VL | `extract_vision_features_ernie()` | vision_model + resampler_model |
| Qwen-VL | `extract_vision_features_qwen()` | visual |
| PaddleOCR | `extract_vision_features_paddleocr()` | visual + projector |

```python
# Ernie-VL 示例
# 位于 fastdeploy/worker/gpu_model_runner.py:2898-2929
images = self.image_preprocess.rescale_factor * images - self.image_preprocess.image_mean_tensor
images = images / self.image_preprocess.image_std_tensor

image_features = self.model.vision_model.extract_feature(images, grid_thw)
image_features = self.model.resampler_model(image_features, grid_thw)
```

### 4. 编码器缓存

当启用编码器缓存时，通过 `mm_hashes` 缓存已提取的视觉特征，避免重复计算：

```python
# 位于 fastdeploy/worker/gpu_model_runner.py:500-607
for i, mm_hash in enumerate(mm_hashes_list):
    if mm_hash in self.encoder_cache:
        # 缓存命中，直接使用
        mm_feature = self.encoder_cache[mm_hash].cuda()
    else:
        # 缓存未命中，提取新特征并存入缓存
        mm_feature = image_features_output[feature_idx : feature_idx + mm_token_lenght]
        self.encoder_cache[mm_hash] = mm_feature.detach().cpu()
```

#### encoder_cache 生命周期与驱逐策略

| 维度 | 说明 |
|------|------|
| **创建时机** | `GPUModelRunner.__init__()` 中，当 `cache_config.max_encoder_cache > 0` 时创建 |
| **容量上限** | 由 `cache_config.max_encoder_cache` 配置，单位为缓存条目数 |
| **写入时机** | 在 `_process_mm_features()` 中，首次提取视觉特征后写入 |
| **存储位置** | 特征以 `detach().cpu()` 形式存储在 CPU 内存，使用时通过 `.cuda()` 转移到 GPU |
| **缓存键** | `mm_hash`，基于多模态内容的哈希值，相同图像的哈希值相同 |
| **驱逐策略** | 当缓存条目数达到 `max_encoder_cache` 时，按 FIFO 顺序驱逐最早的条目 |
| **清理时机** | 调用 `clear_requests()` 或 `clear_cache()` 时统一清理 |
| **跨请求复用** | 不同请求引用相同图像时，通过 `mm_hash` 匹配复用已缓存的特征，避免重复视觉编码 |

### 5. 模型前向传播

根据是否启用多模态，使用不同的调用签名：

```python
# 位于 fastdeploy/worker/gpu_model_runner.py:2289-2299
if self.enable_mm:
    model_output = self.model(
        self.forward_meta.ids_remove_padding,
        self.share_inputs["image_features"],  # 视觉特征
        self.forward_meta,
    )
else:
    model_output = self.model(
        ids_remove_padding=self.forward_meta.ids_remove_padding,
        forward_meta=self.forward_meta,
    )
```

---

## 代码文件引用

| 功能 | 文件位置 |
|-----|---------|
| GPUModelRunner 类定义 | [fastdeploy/worker/gpu_model_runner.py:115](fastdeploy/worker/gpu_model_runner.py#L115) |
| 初始化图像预处理器 | [fastdeploy/worker/gpu_model_runner.py:2842](fastdeploy/worker/gpu_model_runner.py#L2842) |
| 多模态特征处理 | [fastdeploy/worker/gpu_model_runner.py:421](fastdeploy/worker/gpu_model_runner.py#L421) |
| 插入 prefill 输入 | [fastdeploy/worker/gpu_model_runner.py:904](fastdeploy/worker/gpu_model_runner.py#L904) |
| 提取视觉特征 | [fastdeploy/worker/gpu_model_runner.py:2991](fastdeploy/worker/gpu_model_runner.py#L2991) |
| 准备 RoPE3D | [fastdeploy/worker/gpu_model_runner.py:3037](fastdeploy/worker/gpu_model_runner.py#L3037) |
| 准备模型输入 | [fastdeploy/worker/gpu_model_runner.py:1243](fastdeploy/worker/gpu_model_runner.py#L1243) |
| 输入重排序 | [fastdeploy/worker/gpu_model_runner.py:1365](fastdeploy/worker/gpu_model_runner.py#L1365) |
| 模型执行 | [fastdeploy/worker/gpu_model_runner.py:2266](fastdeploy/worker/gpu_model_runner.py#L2266) |

---

## 支持的多模态模型

| 模型类型 | 特点 |
|---------|------|
| Ernie-VL | 百度文心视觉语言模型，使用 vision_model + resampler_model |
| Qwen-VL | 阿里通义千问视觉语言模型，使用 visual 提取器 |
| PaddleOCR | 百度文字识别模型，使用 visual + projector |

---

## 注意事项

1. **V0 vs V1 调度**：多模态特征处理在 V0 和 V1 调度中均会执行，但处理时机和方式有差异
   - **特征提取阶段**（两种调度器均在任务插入时完成）：
     - V0 调度：在 `insert_prefill_inputs()` 中调用 `_process_mm_features()` 批量提取并缓存视觉特征
     - V1 调度：在 `insert_tasks_v1()` 末尾调用 `_process_mm_features()` 提取并缓存视觉特征
   - **特征拼接阶段**（在模型执行前完成，只读取已缓存的特征，不重新提取）：
     - V0 调度：在 `_process_mm_features()` 中直接拼接 `image_features`
     - V1 调度：在 `_prepare_inputs()` 中从 `image_features_list` 合并为 `image_features`
   - **为什么处理时机不同**：V1 调度器同时处理 prefill 和 decode 任务，需要先收集所有特征再统一拼接；V0 调度器主要处理 prefill，可以在插入时直接处理

2. **编码器缓存**：当 `max_encoder_cache > 0` 时启用，可显著提升重复图像的处理效率。缓存存储在 CPU 内存，使用时转移到 GPU。详见上文"编码器缓存"章节

3. **RoPE3D 位置编码**：多模态模型需要 3D 旋转位置编码来处理图像 patch 的空间位置信息

4. **AMP 混合精度**：多模态模型使用自定义的 black/white 列表进行混合精度优化
