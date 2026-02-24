# GPU Model Runner 数据流转流程图

本文档描述了 FastDeploy 中 `GPUModelRunner` 组件的数据流转流程。

> **相关文档**：[gpu_model_runner_mm_data_flow.md](gpu_model_runner_mm_data_flow.md)（多模态数据处理流程）、[gpu_model_runner_refactoring.md](gpu_model_runner_refactoring.md)（重构方案）
>
> **术语约定**：本系列文档中，"提取"指视觉编码器对原始图像进行特征提取（在 insert_tasks 阶段完成）；"拼接"指将已缓存的 image_features_list 合并为 image_features（在 prepare_inputs 阶段完成）。

---

## 1. 完整数据流转流程图

```mermaid
flowchart TD
    subgraph Init["阶段1: 初始化"]
        A["GPUModelRunner.__init__()"] --> B["创建 share_inputs<br/>(InputBatch)"]
        B --> C["初始化 sampler"]
        C --> D["初始化 attn_backends"]
        D --> E["初始化 encoder_cache<br/>(视觉特征缓存)"]
    end

    subgraph LoadModel["阶段2: 加载模型"]
        F["load_model()"] --> G["加载模型实例"]
        G --> H["initialize_kv_cache()<br/>初始化KV Cache"]
        H --> I["vision_encoder_compile()"]
    end

    subgraph InsertTasks["阶段3: 插入任务"]
        J["GpuWorker.execute_model()"] --> K{调度器版本?}
        K -->|V1| L["insert_tasks_v1()<br/>支持 Prefill+Decode 任务"]
        K -->|V0| M["insert_prefill_inputs()<br/>主要处理 Prefill 任务<br/>支持 disaggregated 场景"]
        L --> N["写入 share_inputs<br/>- input_ids<br/>- prompt_ids<br/>- seq_lens<br/>- block_tables<br/>- 采样参数<br/>- stop_flags"]
        M --> N
        N --> N1{chunked prefill?}
        N1 -->|是| N2["分块处理<br/>prefill_start/end_index<br/>is_chunk_step = True"]
        N2 -->|还有剩余 chunk| N2
        N2 -->|所有 chunk 完成| O
        N1 -->|否| O{多模态任务?}
        O -->|是| P["_process_mm_features()<br/>- extract_vision_features()<br/>- 写入 encoder_cache<br/>- 写入 image_features_list<br/>(V0 和 V1 均会调用)"]
        O -->|否| Q["继续"]
        P --> R["prepare_rope3d()"]
        R --> Q
    end

    subgraph Preprocess["阶段4: 预处理"]
        S["_preprocess_and_execute_model()"] --> T["_process_reorder()<br/>- condense()<br/>- 分离prefill/decode"]
        T --> U["_prepare_inputs()<br/>pre_process()"]
        U --> V["生成 ids_remove_padding<br/>batch_id_per_token<br/>cu_seqlens_q/k"]
        V --> W["initialize_forward_meta()<br/>创建 ForwardMeta"]
        W --> X["拼接已缓存的 image_features<br/>(多模态模式，仅拼接，不重新提取)"]
        X --> Y["sampler.pre_process()"]
    end

    subgraph ModelExec["阶段5: 模型执行"]
        Z["model.forward()<br/>输入:<br/>- ids_remove_padding<br/>- image_features<br/>- forward_meta"] --> AA["返回 hidden_states<br/>或 full_hidden_states"]
    end

    subgraph Postprocess["阶段6: 后处理"]
        AB["_postprocess()"] --> AC["rebuild_padding()<br/>恢复padding格式"]
        AC --> AD["model.compute_logits()<br/>计算logits"]
        AD --> AE["sampler.sample()<br/>采样下一个token"]
        AE --> AF["返回 SamplerOutput<br/>- sampled_token_ids<br/>- logprobs"]
    end

    subgraph SaveOutput["阶段7: 保存输出"]
        AG["ModelOutputData<br/>构造输出对象"] --> AH{异步输出?}
        AH -->|是| AI["发送到 async_output_queue"]
        AH -->|否| AJ["发送到 ZMQ/共享内存"]
        AI --> AK["更新 share_inputs 状态"]
        AJ --> AK
        AK --> AL["返回给 Engine"]
    end

    Init --> LoadModel
    LoadModel --> InsertTasks
    InsertTasks --> Preprocess
    Preprocess --> ModelExec
    ModelExec --> Postprocess
    Postprocess --> SaveOutput

    style Init fill:#e1f5ff
    style LoadModel fill:#fff4e1
    style InsertTasks fill:#ffe1f5
    style Preprocess fill:#e1ffe1
    style ModelExec fill:#f5e1ff
    style Postprocess fill:#e1ffe1
    style SaveOutput fill:#fff4e1
```

---

## 2. 组件交互关系图

```mermaid
flowchart TB
    subgraph Upper["上层组件"]
        Engine["Engine<br/>推理引擎"]
        GpuWorker["GpuWorker<br/>设备封装"]
    end

    subgraph Core["GPUModelRunner 核心"]
        GPUModelRunner["GPUModelRunner<br/>模型执行协调器"]

        subgraph Components["核心子组件"]
            ShareInputs["share_inputs<br/>(InputBatch)<br/>共享输入缓冲区"]
            Sampler["sampler<br/>(Sampler)<br/>采样器"]
            ForwardMeta["forward_meta<br/>(ForwardMeta)<br/>前向元数据"]
            AttnBackends["attn_backends<br/>[]AttentionBackend<br/>注意力后端"]
            EncoderCache["encoder_cache<br/>(dict)<br/>视觉特征缓存"]
        end
    end

    subgraph Lower["下层组件"]
        Model["Model<br/>模型实例"]
        AttentionBackend["AttentionBackend<br/>flash_attn/pd_sep"]
        SamplerImpl["Sampler<br/>采样逻辑实现"]
    end

    subgraph Data["数据结构"]
        Request["Request<br/>请求对象"]
        ModelOutputData["ModelOutputData<br/>输出数据"]
        SamplerOutput["SamplerOutput<br/>采样输出"]
    end

    Engine -->|调用| GpuWorker
    GpuWorker -->|创建/调用| GPUModelRunner

    GPUModelRunner -->|包含| Components
    GPUModelRunner -->|使用| Model
    GPUModelRunner -->|使用| AttentionBackend
    GPUModelRunner -->|使用| SamplerImpl

    Request -->|插入| ShareInputs
    ShareInputs -->|预处理| ForwardMeta
    ForwardMeta -->|传入| Model
    Model -->|"返回 hidden_states"| ComputeLogits["compute_logits()<br/>计算logits"]
    ComputeLogits -->|"传入 logits"| SamplerImpl
    SamplerImpl -->|返回| SamplerOutput
    SamplerOutput -->|组合| ModelOutputData
    ModelOutputData -->|返回| Engine

    style Upper fill:#e8f4f8
    style Core fill:#f8e8f8
    style Lower fill:#f8f8e8
    style Data fill:#f4f8e8
```

---

## 3. 数据结构流转图

```mermaid
flowchart LR
    subgraph Input["输入阶段"]
        R["Request[]<br/>请求列表"]
        I1["input_ids<br/>token IDs"]
        I2["prompt_ids<br/>提示tokens"]
        I3["seq_lens<br/>序列长度"]
        I4["block_tables<br/>KV Cache块表"]
        I5["sampling_params<br/>采样参数"]
        I6["image_features_list<br/>视觉特征列表"]
    end

    subgraph ShareBuf["共享缓冲区"]
        S["share_inputs<br/>(InputBatch)"]
    end

    subgraph Forward["前向传播"]
        FM["forward_meta<br/>(ForwardMeta)"]
        F1["ids_remove_padding<br/>移除padding"]
        F2["batch_id_per_token<br/>token到batch映射"]
        F3["cu_seqlens_q/k<br/>累积序列长度"]
        F4["rotary_embs<br/>旋转位置编码"]
        F5["caches<br/>KV Cache"]
    end

    subgraph Output["输出阶段"]
        O1["hidden_states<br/>隐藏状态"]
        O2["logits<br/>logits"]
        O3["next_tokens<br/>采样的token"]
        O4["stop_flags<br/>停止标志"]
        O5["ModelOutputData<br/>输出数据"]
    end

    R -->|写入| S
    I1 --> S
    I2 --> S
    I3 --> S
    I4 --> S
    I5 --> S
    I6 --> S

    S -->|预处理| FM
    FM --> F1
    FM --> F2
    FM --> F3
    FM --> F4
    FM --> F5

    FM -->|传入模型| O1
    O1 -->|计算| O2
    O2 -->|采样| O3
    O3 -->|组合| O5
    O4 --> O5

    style Input fill:#ffe1e1
    style ShareBuf fill:#e1ffe1
    style Forward fill:#e1e1ff
    style Output fill:#ffffe1
```

---

## 4. 核心数据转换表

| 转换步骤 | 输入 | 输出 | 方法 |
|---------|------|------|------|
| 1 | Request[] | share_inputs | `insert_tasks_v1()` / `insert_prefill_inputs()` |
| 2 | share_inputs | forward_meta | `_prepare_inputs()` + `initialize_forward_meta()` |
| 3 | forward_meta + share_inputs | hidden_states | `model.forward()` |
| 4 | hidden_states | logits | `model.compute_logits()` |
| 5 | logits + share_inputs | SamplerOutput | `sampler.sample()` |
| 6 | SamplerOutput + share_inputs | ModelOutputData | `_postprocess()` + `_save_model_output()` |
| 7 | ModelOutputData | Engine/共享内存 | `post_process()` |

---

## 5. 核心类与方法

### GPUModelRunner
**文件**: `fastdeploy/worker/gpu_model_runner.py`

| 方法 | 描述 |
|------|------|
| `__init__()` | 初始化共享输入缓冲区、采样器、注意力后端 |
| `load_model()` | 加载模型实例、初始化KV Cache |
| `insert_tasks_v1()` | V1调度器：插入任务到共享输入 |
| `insert_prefill_inputs()` | V0调度器：插入prefill输入，支持disaggregated场景 |
| `_process_mm_features()` | 处理多模态特征 |
| `execute_model()` | 执行模型主入口 |
| `_preprocess_and_execute_model()` | 预处理并执行模型 |
| `_postprocess()` | 后处理模型输出 |

### InputBatch
**文件**: `fastdeploy/worker/input_batch.py`

| 属性 | 类型 | 描述 |
|------|------|------|
| input_ids | paddle.Tensor | 输入token IDs |
| prompt_ids | paddle.Tensor | 提示token IDs |
| seq_lens_encoder | paddle.Tensor | 编码器序列长度 |
| seq_lens_decoder | paddle.Tensor | 解码器序列长度 |
| block_tables | paddle.Tensor | KV Cache块表 |
| top_p, top_k | paddle.Tensor | 采样参数 |
| stop_flags | paddle.Tensor | 停止标志 |
| image_features | paddle.Tensor | 视觉特征 |

### ForwardMeta
**文件**: `fastdeploy/model_executor/forward_meta.py`

| 属性 | 类型 | 描述 |
|------|------|------|
| ids_remove_padding | paddle.Tensor | 移除padding的token IDs |
| rotary_embs | paddle.Tensor | 旋转位置编码 |
| attn_backend | AttentionBackend | 注意力后端 |
| block_tables | paddle.Tensor | 块表 |
| caches | list[paddle.Tensor] | KV缓存 |

### ModelOutputData
**文件**: `fastdeploy/worker/output.py`

| 属性 | 类型 | 描述 |
|------|------|------|
| next_tokens | paddle.Tensor | 下一个token |
| stop_flags | paddle.Tensor | 停止标志 |
| step_idx | int | 步骤索引 |
| seq_lens_this_time | paddle.Tensor | 当前步序列长度 |
| input_ids | paddle.Tensor | 输入IDs |
| full_hidden_states | paddle.Tensor | 完整隐藏状态 |

---

## 6. 文件路径索引

| 组件 | 文件路径 |
|------|----------|
| GPUModelRunner | `fastdeploy/worker/gpu_model_runner.py` |
| InputBatch | `fastdeploy/worker/input_batch.py` |
| ModelRunnerBase | `fastdeploy/worker/model_runner_base.py` |
| GpuWorker | `fastdeploy/worker/gpu_worker.py` |
| ModelOutputData | `fastdeploy/worker/output.py` |
| ForwardMeta | `fastdeploy/model_executor/forward_meta.py` |
| Sampler | `fastdeploy/model_executor/layers/sample/sampler.py` |
| Request | `fastdeploy/engine/request.py` |

---

## 7. 状态变化机制

### 7.1 状态流转流程图

```mermaid
flowchart TD
    subgraph InitState["初始状态"]
        S0["share_inputs 初始化<br/>- seq_lens_encoder = 0<br/>- seq_lens_decoder = 0<br/>- stop_flags = True<br/>- block_tables = -1<br/>- free_list = [num_blocks-1...0]"]
    end

    subgraph PrefillState["PREFILL 状态"]
        S1["插入 PREFILL 请求<br/>- stop_flags = False<br/>- seq_lens_encoder = prompt_len<br/>- seq_lens_decoder = 0<br/>- block_tables = [分配的块]<br/>- is_block_step = False"]
        S2["Chunked Prefill 继续<br/>- seq_lens_encoder = chunk_len<br/>- is_chunk_step = True"]
    end

    subgraph DecodeState["DECODE 状态"]
        S3["首次 DECODE<br/>- seq_lens_encoder = 0<br/>- seq_lens_decoder = 1<br/>- stop_flags = False"]
        S4["后续 DECODE<br/>- seq_lens_decoder += 1<br/>- seq_lens_this_time = 1"]
    end

    subgraph PreemptedState["PREEMPTED 状态"]
        S5["请求被抢占<br/>- stop_flags = True<br/>- seq_lens_this_time = 0<br/>- block_tables = -1"]
    end

    subgraph StopState["停止状态"]
        S6["达到停止条件<br/>- stop_flags = True<br/>- 释放 block_tables 到 free_list"]
    end

    subgraph MultiModalState["多模态特殊状态"]
        S7["多模态 PREFILL<br/>- 提取 vision_features<br/>- 写入 encoder_cache"]
        S8["视觉特征缓存命中<br/>- 重用 encoder_cache<br/>- 跳过视觉编码"]
    end

    S0 --> S1
    S1 -->|完成 prefill| S3
    S1 -->|chunked prefill| S2
    S2 -->|继续 prefill| S2
    S2 -->|完成| S3

    S3 -->|下一个 token| S4
    S4 -->|继续 decode| S4
    S4 -->|达到停止条件| S6
    S4 -->|内存不足被抢占| S5

    S5 -->|恢复请求| S1
    S6 --> S0

    S1 -->|多模态输入<br/>缓存未命中| S7
    S1 -->|多模态输入<br/>缓存命中| S8
    S7 -->|"特征提取完成<br/>继续当前 PREFILL 流程"| S3
    S8 -->|"直接使用缓存特征<br/>继续当前 PREFILL 流程"| S3

    style InitState fill:#e1f5ff
    style PrefillState fill:#fff4e1
    style DecodeState fill:#e1ffe1
    style PreemptedState fill:#ffe1e1
    style StopState fill:#f5e1ff
    style MultiModalState fill:#ffe1f5
```

### 7.2 关键状态字段变化表

| 状态字段 | 初始值 | PREFILL | DECODE | PREEMPTED | FINISHED |
|---------|--------|---------|--------|-----------|----------|
| `seq_lens_encoder` | 0 | prompt_len (或 chunk_len) | 0 | 0 | 0 |
| `seq_lens_decoder` | 0 | prefill_start_index | 每步+1 (在 insert_tasks_v1 中自增) | 0 | final_len |
| `seq_lens_this_time` | 0 | prompt_len (或 chunk_len) | 1 | 0 | 0 |
| `stop_flags` | True | False | False | True | True |
| `block_tables` | -1 | [分配的块] | [分配的块] | -1 | -1 |
| `is_block_step` | False | False | False | False | False |
| `is_chunk_step` | False | True (如果 chunked) | False | False | False |

### 7.3 KV Cache 块管理状态

```mermaid
stateDiagram-v2
    [*] --> FreeBlock: 初始化
    FreeBlock --> Allocated: 分配给请求
    Allocated --> InUse: PREFILL/DECODE
    InUse --> Allocated: 继续使用
    Allocated --> FreeBlock: 请求完成/抢占

    note right of FreeBlock
        free_list: [num_blocks-1...0]
        free_list_len: num_blocks * ratio
    end note

    note right of Allocated
        block_tables[batch_id] = [block_ids]
        block_len = len(block_tables)
    end note
```

### 7.4 关键状态更新代码位置

| 状态更新 | 方法 | 位置 |
|---------|------|------|
| 初始化状态 | `init_share_inputs()` | [input_batch.py:98](fastdeploy/worker/input_batch.py#L98) |
| PREFILL 状态 | `insert_tasks_v1()` | [gpu_model_runner.py:720](fastdeploy/worker/gpu_model_runner.py#L720) |
| DECODE 状态 | `insert_tasks_v1()` | [gpu_model_runner.py:808](fastdeploy/worker/gpu_model_runner.py#L808) |
| PREEMPTED 状态 | `insert_tasks_v1()` | [gpu_model_runner.py:820](fastdeploy/worker/gpu_model_runner.py#L820) |
| KV Cache 初始化 | `initialize_kv_cache()` | [gpu_model_runner.py:2690](fastdeploy/worker/gpu_model_runner.py#L2690) |
| 更新 stop_flags | `_postprocess()` | [gpu_model_runner.py:2354](fastdeploy/worker/gpu_model_runner.py#L2354) |
| 清除请求 | `clear_requests()` | [gpu_model_runner.py:2796](fastdeploy/worker/gpu_model_runner.py#L2796) |

### 7.5 请求状态枚举

```python
class RequestStatus(Enum):
    WAITING = 0      # 等待执行
    RUNNING = 1      # 正在执行
    PREEMPTED = 2    # 被抢占
    FINISHED = 3     # 已完成
    ABORT = 4        # 已中止

class RequestType(Enum):
    PREFILL = 0      # 预填充阶段
    DECODE = 1       # 解码阶段
    PREEMPTED = 2    # 抢占请求
    EXTEND = 3       # 扩展请求
```

### 7.6 状态更新关键代码片段

**PREFILL 状态设置** ([gpu_model_runner.py:720-783](fastdeploy/worker/gpu_model_runner.py#L720-L783)):
```python
if request.task_type.value == RequestType.PREFILL.value:
    self.share_inputs["stop_flags"][idx : idx + 1] = False
    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = prefill_start_index
    self.share_inputs["seq_lens_this_time_buffer"][idx : idx + 1] = length
    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
    self.exist_prefill_flag = True
    self.share_inputs["is_chunk_step"][idx : idx + 1] = prefill_end_index < len(input_ids)
```

**DECODE 状态设置** ([gpu_model_runner.py:808-815](fastdeploy/worker/gpu_model_runner.py#L808-L815)):
```python
elif request.task_type.value == RequestType.DECODE.value:
    encoder_block_num = len(request.block_tables)
    self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
    self.share_inputs["block_tables"][idx : idx + 1, :] = -1
    self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
        request.block_tables, dtype="int32"
    )
```

**PREEMPTED 状态设置** ([gpu_model_runner.py:820-827](fastdeploy/worker/gpu_model_runner.py#L820-L827)):
```python
else:  # preempted task
    self.share_inputs["preempted_idx"][idx : idx + 1, :] = 1
    self.share_inputs["block_tables"][idx : idx + 1, :] = -1
    self.share_inputs["stop_flags"][idx : idx + 1] = True
    self.share_inputs["seq_lens_this_time_buffer"][idx : idx + 1] = 0
    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
```
