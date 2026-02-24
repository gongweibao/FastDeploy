# GPUModelRunner 行为单元测试报告

## 1. 测试文件

`tests/worker/test_gpu_model_runner_behavior.py` — 64 个测试，全部通过。

## 2. 测试原则

| 原则 | 执行情况 |
|---|---|
| 行为测试，非实现细节 | 所有断言只检查返回值或可观察状态 |
| 不 mock 被测模块内部函数 | 未 patch GPUModelRunner 任何方法 |
| 仅 mock 真正外部系统 | GPU 算子、分布式通信、IPC |
| 不使用 assert_called / patch 内部函数 | 仅对外部 IPC 系统使用了一次 assert_called |
| 重构后仍成立 | 测试只依赖方法的输入输出契约 |

## 3. 测试覆盖

### 3.1 `_get_feature_positions` — 10 个测试，零 mock

过滤和调整 ImagePosition 在 prefill 范围内的位置，纯逻辑函数。

| 测试 | 类型 |
|---|---|
| 不修改原始对象 | 契约 |
| 单 token 位置在范围内 | 边界 |
| 位置在范围起始边界 | 边界 |
| 位置在范围结束边界（排除） | 边界 |
| 大量位置 | 边界 |
| 位置恰好结束于范围起点（排除） | 边界 |
| 位置恰好开始于范围终点（排除） | 边界 |
| 位置跨越整个范围 | 正常 |
| 空范围 | 边界 |
| 空位置列表 | 边界 |

### 3.2 `get_input_length_list` — 10 个测试，零 mock

为 CUDA Graph 捕获生成 dummy 输入长度分布。

| 测试 | 类型 |
|---|---|
| batch_size=0 返回空 | 边界 |
| 正常 decode 均匀分布 | 正常 |
| capture_prefill 集中 token | 正常 |
| capture_prefill 且 token 数 < batch_size | 边界 |
| max_model_len 约束 | 边界 |
| expert_parallel 限制为 32 | 正常 |
| enc_dec_block_num 累加到 block_num | 正常 |
| 负 decode_len 下限为 1 | 异常 |
| 单 batch | 正常 |
| 超大 num_tokens 被约束 | 边界 |

### 3.3 `cal_theortical_kvcache` — 8 个测试，零 mock

计算每个 KV Cache block 的字节数，纯算术。

| 测试 | 类型 |
|---|---|
| 默认 bf16 (2 bytes) | 正常 |
| int8 量化 (1 byte) | 正常 |
| fp8 量化 (1 byte) | 正常 |
| MLA cache 公式 | 正常 |
| MLA + int8 | 正常 |
| MTP 推测解码增加层数 | 正常 |
| ngram 推测不增加层数 | 正常 |
| 单层模型 | 边界 |

### 3.4 `exist_prefill` / `exist_decode` — 8 个测试

| 测试 | Mock | 类型 |
|---|---|---|
| V1 scheduler flag=True | `envs`（环境变量系统） | 正常 |
| V1 scheduler flag=False | `envs` | 正常 |
| 非 V1 有 encoder | `envs` | 正常 |
| 非 V1 无 encoder | `envs` | 正常 |
| 有 decoder | 无 | 正常 |
| 无 decoder | 无 | 正常 |
| 单元素 batch 有值 | 无 | 边界 |
| 单元素 batch 无值 | 无 | 边界 |

### 3.5 `only_prefill` / `only_decode` — 8 个测试

| 测试 | Mock | 类型 |
|---|---|---|
| 非 EP 无 decode → True | `envs` | 正常 |
| 非 EP 有 decode → False | `envs` | 正常 |
| EP 所有 rank 无 decode | `all_gather_object`（NCCL 通信） | 正常 |
| EP 部分 rank 有 decode | `all_gather_object` | 正常 |
| 非 EP 无 prefill → True | `envs` | 正常 |
| 非 EP 有 prefill → False | `envs` | 正常 |
| EP 所有 rank 无 prefill | `all_gather_object` | 正常 |
| EP 部分 rank 有 prefill | `all_gather_object` | 正常 |

### 3.6 `clear_requests` — 7 个测试

| 测试 | Mock | 类型 |
|---|---|---|
| stop_flags 全部置 True | 无 | 正常 |
| prompt_logprobs 清空 | 无 | 正常 |
| in_progress_logprobs 清空 | 无 | 正常 |
| forward_batch_list 重置为 None | 无 | 正常 |
| exist_prefill_flag 置 False | 无 | 正常 |
| 幂等调用无报错 | 无 | 边界 |
| routing_replay 通知外部存储 | `routing_replay_manager`（IPC 存储） | 正常 |

### 3.7 `insert_tasks_v1` — 13 个测试

| 测试 | Mock | 类型 |
|---|---|---|
| 单 PREFILL 设置状态 | `set_stop`（CUDA 算子） | 正常 |
| 单 DECODE 设置 block_tables | `set_stop` | 正常 |
| PREEMPTED 清除状态 | `set_stop` | 正常 |
| 混合批次 | `set_stop` | 正常 |
| sampling params 传播 | `set_stop` | 正常 |
| prompt_logprobs 跟踪 | `set_stop` | 正常 |
| 分块预填充状态 | `set_stop` | 边界 |
| num_running_requests 设置 | `set_stop` | 正常 |
| thinking 启用 | `set_stop` | 正常 |
| thinking 禁用 | `set_stop` | 正常 |
| 空请求列表 | `set_stop` | 边界 |
| EOS token 传播 | `set_stop` | 正常 |
| PREEMPTED 清除 logprobs | `set_stop` | 正常 |

## 4. Mock 使用说明

每个 mock 均为真正的外部系统，不涉及被测模块内部函数：

| Mock 对象 | 原因 |
|---|---|
| `set_stop` / `get_stop` | C++ CUDA 自定义算子，需物理 GPU |
| `paddle.distributed.all_gather_object` | NCCL 分布式通信，需多 GPU 集群 |
| `envs.ENABLE_V1_KVCACHE_SCHEDULER` | 全局环境变量系统 |
| `self.sampler` | GPU 采样组件（CUDA kernel） |
| `routing_replay_manager` | 外部 IPC/路由存储 |

## 5. 设计问题

### 5.1 God Class

`GPUModelRunner` 共 3131 行、60+ 方法，混合了输入准备、模型执行、输出管理、KV Cache、视觉特征、推测解码、CUDA Graph、IPC 通信。建议拆分为独立协作类，每个类独立可测。

### 5.2 构造函数副作用

`__init__` 执行：创建 GPU 张量（`InputBatch`）、连接 IPC（`ZmqIpcClient`、`IPCSignal`）、设置环境变量、启动守护线程。导致无法在无 GPU 环境正常实例化，测试必须通过 `__new__` 绕过构造函数。

### 5.3 业务逻辑与 GPU 操作耦合

`insert_tasks_v1` 末尾调用 `set_stop()`（CUDA 自定义算子），纯数据准备逻辑与 GPU 信号操作交织。建议将 `set_stop` 调用移至调用方（如 Worker），使 `insert_tasks_v1` 成为纯数据准备方法，无需 mock 即可测试。

### 5.4 无依赖注入

`Sampler`、`AttentionBackend`、`InputBatch`、`IPCSignal` 等在 `__init__` 内部创建而非注入。如果支持注入，测试可直接提供无操作实现，不需要绕过构造函数。

### 5.5 全局状态依赖

多个方法通过 `envs.ENABLE_V1_KVCACHE_SCHEDULER`、`envs.FD_USE_GET_SAVE_OUTPUT_V1` 等模块级变量分支行为。建议将这些配置纳入 `FDConfig`，消除对全局状态的依赖。

## 6. 不可测方法

以下方法因深度依赖 GPU 运行时而无法编写有意义的行为单元测试：

| 方法 | 原因 |
|---|---|
| `__init__` | 创建 GPU 张量、IPC 连接、守护线程 |
| `load_model` | 加载实际模型权重 |
| `execute_model` / `execute_model_normal` / `execute_model_overlap` | GPU forward pass |
| `_preprocess_and_execute_model` | 调用模型 forward |
| `_postprocess` | GPU 采样、rebuild_padding、step_cuda |
| `profile_run` | GPU 内存分析 |
| `capture_model` | CUDA Graph 捕获 |
| `initialize_kv_cache` | GPU 共享内存分配 |
| `_prepare_inputs` | 调用 `pre_process`（GPU 算子） |

这些方法需要集成测试或端到端测试环境（真实 GPU）来验证。
