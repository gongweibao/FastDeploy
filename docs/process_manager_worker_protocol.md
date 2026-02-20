# ProcessManager 和 Worker 交互协议

## 概述

本文档描述 FastDeploy 新架构中 ProcessManager 和 Worker 程序之间的交互协议。

ProcessManager 是引擎组件之一，负责管理 Worker 进程的生命周期，包括启动、健康监控和清理。

---

## 1. 启动阶段协议

### 1.1 启动命令格式

ProcessManager 通过 `paddle.distributed.launch` 启动 Worker 进程：

```bash
python -m paddle.distributed.launch \
    --log_dir log \
    --devices <device_ids> \
    --ips <ips> \
    --nnodes <nnode> \
    ../worker/worker_process.py \
    --max_num_seqs <value> \
    --max_model_len <value> \
    --gpu_memory_utilization <value> \
    --model <model_path> \
    --engine_worker_queue_port <port> \
    --pod_ip <ip> \
    --block_size <value> \
    --tensor_parallel_size <value> \
    --engine_pid <pid> \
    ... (约50+个参数)
```

### 1.2 环境变量设置

启动前设置以下环境变量：

| 环境变量 | 值 | 用途 |
|---------|---|------|
| `ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY` | `0` | 禁用模型加载并发 |
| `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` | `python` | Protobuf 实现 |
| `FLAGS_use_append_attn` | `1` | 启用 append attention |
| `NCCL_ALGO` | `Ring` | NCCL 算法 |

### 1.3 关键启动参数

| 参数 | 用途 |
|------|------|
| `--engine_worker_queue_port` | Engine-Worker 任务队列通信端口 |
| `--engine_pid` | Engine 进程 ID，用于 IPC 信号命名空间 |
| `--pod_ip` | Pod IP 地址，用于多节点通信 |
| `--ips` | 多节点部署的 IP 列表 |
| `--tensor_parallel_size` | 张量并行大小 |
| `--data_parallel_size` | 数据并行大小 |

### 1.4 进程管理

- 使用 `subprocess.Popen` 启动子进程
- `preexec_fn=os.setsid` 创建独立进程组，便于统一管理
- 标准输出重定向到 PIPE（用于解析加载进度）

---

## 2. IPC 通信机制

### 2.1 共享内存信号 (IPCSignal)

基于 `multiprocessing.shared_memory.SharedMemory` 的低延迟状态同步。

| 信号名称 | 方向 | 类型 | 用途 | 超时 |
|---------|------|------|------|------|
| `worker_ready_signal` | Worker → Engine | np.int32[] | Worker 完成初始化 | - |
| `worker_healthy_live_signal` | Worker → Engine | np.int32[] | Worker 存活心跳（时间戳） | 30秒 |
| `get_profile_block_num_signal` | Worker → Engine | np.int32[1] | Profiling 结果（块数） | 轮询等待 |
| `exist_task_signal` | Worker → Engine | np.int32[1] | 任务存在标志 | - |
| `exist_swapped_task_signal` | Worker → Engine | np.int32[1] | Swapped task 标志 | - |
| `exist_prefill_task_signal` | Worker → Engine | np.int32[1] | Prefill 任务标志 | - |
| `cache_ready_signal` | Worker → Engine | np.int32[] | Cache 准备就绪 | - |
| `swap_space_ready_signal` | Worker → Engine | np.int32[] | Swap 空间准备就绪 | - |
| `loaded_model_signal` | Worker → Engine | np.int32[1] | 模型加载完成 | - |

### 2.2 EngineWorkerQueue (FMQ) 任务队列

基于 `multiprocessing.managers.BaseManager` 的跨进程任务队列。

**主要方法**:
- `put_tasks(tasks)` - Engine 投递任务列表
- `get_tasks()` - Worker 获取任务列表
- `exist_tasks()` - 检查任务是否存在

**同步机制**:
- `client_read_flag` - 客户端读取完成标志数组
- TP barrier - Tensor Parallel 同步屏障

**共享资源**:
| 资源名称 | 类型 | 用途 |
|----------|------|------|
| `get_tasks` | ListProxy | 任务队列 |
| `get_client_read_flag` | ListProxy | 客户端读取标志 |
| `get_lock` | AcquirerProxy | 互斥锁 |
| `get_worker_process_tp_barrier` | Barrier | TP 屏障 |
| `get_cache_infos` | ListProxy | 缓存信息 |

### 2.3 通信地址

| 配置 | 地址格式 | 条件 |
|------|---------|------|
| TCP 模式 | `tcp://<pod_ip>:<port>` | `FD_ENGINE_TASK_QUEUE_WITH_SHM=0` |
| Unix Socket | `/dev/shm/fd_task_queue_*.sock` | `FD_ENGINE_TASK_QUEUE_WITH_SHM=1` |

---

## 3. Worker 初始化协议

### 3.1 初始化流程

```python
def run_worker_proc():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 初始化分布式环境
    ranks, local_rank = init_distributed_environment()

    # 3. 构建 FDConfig
    fd_config = initialize_fd_config(args, ranks, local_rank)

    # 4. 创建 Worker 实例
    worker_proc = PaddleDisWorkerProc(fd_config, ranks, local_rank)

    # 5. 初始化设备
    worker_proc.init_device()

    # 6. 加载模型权重
    worker_proc.load_model()

    # 7. 初始化 KV Cache
    worker_proc.initialize_kv_cache()

    # 8. CUDAGraph 优化和预热
    worker_proc.graph_optimize_and_warm_up_model()

    # 9. 初始化健康状态
    worker_proc.init_health_status()  # 设置 worker_ready_signal

    # 10. 启动任务队列服务
    worker_proc.start_task_queue_service()

    # 11. 进入主事件循环
    worker_proc.event_loop_normal()
```

### 3.2 加载进度报告

Worker 通过标准输出输出加载进度，ProcessManager 解析并显示：

| 正则表达式 | 解析内容 | 用途 |
|-----------|---------|------|
| `Loading (?:fastsafetensors |safetensors )?checkpoint shards:\s*(\d+)` | 分片数量 | 权重加载进度 |
| `Start load layer (\d+)` | 层数 | 层加载进度 |
| `set state for layer (\d+)` | 层数 | 层加载进度 |

---

## 4. 运行时协议

### 4.1 Worker 主事件循环

```python
while True:
    # 1. 更新健康心跳
    worker_healthy_live_signal.value[tp_rank] = int(time.time())

    # 2. 检测任务 (仅 tp_rank=0)
    if tp_rank == 0:
        if task_queue.exist_tasks():
            exist_task_signal.value[0] = ExistTaskStatus.EXIST

    # 3. TP 同步
    if tp_size > 1:
        barrier_wait()

    # 4. 获取并处理任务
    if exist_task_signal.value[0] == ExistTaskStatus.EXIST:
        tasks, read_finish = task_queue.get_tasks()
        # 分离控制请求和推理任务
        control_reqs, inference_tasks = separate_requests(tasks)

        # 处理控制请求
        for control_req in control_reqs:
            run_control_method(control_req)

        # 执行模型推理
        if inference_tasks:
            execute_model(inference_tasks)

        # 清理状态
        if read_finish:
            exist_task_signal.value[0] = ExistTaskStatus.EMPTY
```

### 4.2 任务状态常量

```python
class ExistTaskStatus:
    EMPTY = 0    # 无任务
    EXIST = 1    # 存在任务
    REFUSE = 2   # 拒绝任务
```

---

## 5. 生命周期管理协议

### 5.1 启动阶段

| ProcessManager | Worker |
|----------------|--------|
| `start_workers()` | 接收启动命令 |
| 解析日志显示加载进度 | 输出加载进度日志 |
| 等待 `worker_ready_signal` | 设置 `worker_ready_signal=1` |
| 等待 profiling 完成（如需要） | 执行 profile，设置 `get_profile_block_num_signal` |

### 5.2 运行阶段

| ProcessManager | Worker |
|----------------|--------|
| 周期性 `check_worker_health()` | 周期性更新 `worker_healthy_live_signal` |
| 检查超时 30 秒 | 每 loop 更新一次时间戳 |
| 检查 `worker_ready_signal` | 完成初始化后设置 |

### 5.3 停止阶段

| ProcessManager | Worker |
|----------------|--------|
| `stop_workers()` | 接收 `SIGTERM` 信号 |
| `os.killpg(pgid, SIGTERM)` | 清理资源并退出 |
| 清空进程列表 | - |

### 5.4 Profiling 协议

```
ProcessManager                         Worker
     |                                     |
     |--- do_profile=1 启动 worker ------>|
     |                                     |
     |<----- 输出 "Loading Weights" ------|
     |                                     |
     |<----- 输出 "Loading Layers" -------|
     |                                     |
     |--- 调用 stop_profile() ------------>|
     |    (do_profile=0)                   |
     |                                     |
     |    轮询 get_profile_block_num_signal|
     |<------ 设置 profiling 结果 ---------|
     |    get_profile_block_num_signal[0] = N |
     |                                     |
```

---

## 6. 关键文件路径

| 组件 | 文件路径 |
|------|---------|
| ProcessManager | `fastdeploy/engine/components/process_manager.py` |
| IPCManager | `fastdeploy/engine/components/ipc_manager.py` |
| IPCSignal | `fastdeploy/inter_communicator/ipc_signal.py` |
| IPCSignal 常量 | `fastdeploy/inter_communicator/ipc_signal_const.py` |
| EngineWorkerQueue | `fastdeploy/inter_communicator/engine_worker_queue.py` |
| Worker 主入口 | `fastdeploy/worker/worker_process.py` |
| Worker 基类 | `fastdeploy/worker/worker_base.py` |
| GPU Worker | `fastdeploy/worker/gpu_worker.py` |
| FMQ | `fastdeploy/inter_communicator/fmq.py` |
| ZMQ Server | `fastdeploy/inter_communicator/zmq_server.py` |

---

## 7. 配置传递

### 7.1 配置类映射

命令行参数通过以下配置类传递到 Worker：

| 配置类 | 主要参数 |
|--------|---------|
| `ModelConfig` | `--model`, `--dtype`, `--max_model_len`, `--quantization` |
| `ParallelConfig` | `--tensor_parallel_size`, `--expert_parallel_size`, `--data_parallel_size` |
| `CacheConfig` | `--block_size`, `--gpu_memory_utilization`, `--kv_cache_ratio` |
| `SchedulerConfig` | `--max_num_seqs`, `--max_num_batched_tokens`, `--splitwise_role` |
| `LoadConfig` | `--dynamic_load_weight`, `--load_strategy`, `--load_choices` |
| `SpeculativeConfig` | `--speculative_config` (JSON) |
| `GraphOptimizationConfig` | `--graph_optimization_config` (JSON) |
| `PlasAttentionConfig` | `--plas_attention_config` (JSON) |
| `StructuredOutputsConfig` | `--guided_decoding_backend`, `--reasoning_parser` |

### 7.2 完整参数列表

Worker 接收的命令行参数包括（但不限于）：

- `-m, --model`: 模型目录路径
- `--max_num_seqs`: 最大批处理大小
- `--max_model_len`: 模型最大长度
- `--block_size`: KV Cache 块大小
- `--gpu_memory_utilization`: GPU 内存利用率
- `--tensor_parallel_size`: 张量并行大小
- `--expert_parallel_size`: 专家并行大小
- `--data_parallel_size`: 数据并行大小
- `--enable_prefix_caching`: 启用前缀缓存
- `--enable_chunked_prefill`: 启用分块 prefill
- `--quantization`: 量化配置 (JSON)
- `--do_profile`: 是否进行 profiling

---

## 8. 错误处理

### 8.1 健康检查失败

```
ProcessManager 检测:
- worker_healthy_live_signal 超时 (30秒)
- 返回 False
```

### 8.2 初始化失败

```
ProcessManager 监控:
- Worker 进程异常退出 (poll() != None)
- 检查日志文件 log/launch_worker.log
- 返回 False
```

### 8.3 Profiling 超时

```
ProcessManager 等待:
- get_profile_block_num_signal.value[0] == 0 超时
- 抛出 RuntimeError: "Worker process failed to start during profiling"
```

---

## 9. 多节点支持

### 9.1 多节点启动

```bash
python -m paddle.distributed.launch \
    --ips <ip1>,<ip2>,<ip3> \
    --nnodes 3 \
    ...
```

### 9.2 跨节点通信

- `pod_ip`: 当前节点 IP
- `engine_worker_queue_port`: 任务队列端口
- `ips`: 所有节点 IP 列表

---

## 附录：时序图

```
┌─────────────────┐         ┌─────────────────┐
│  ProcessManager │         │     Worker      │
└────────┬────────┘         └────────┬────────┘
         │                          │
         │--- start_workers() ----->│
         │    (paddle.distributed.  │
         │     launch + args)       │
         │                          │
         │<----- log: Loading -----│
         │      Weights             │
         │                          │
         │<----- log: Loading -----│
         │      Layers              │
         │                          │
         │<-- worker_ready_signal =1│
         │                          │
         │--- put_tasks() --------->│
         │                          │
         │<-- exist_task_signal=1 --│
         │                          │
         │--- check_health() ------>│
         │                          │
         │<-- healthy_live_signal--│
         │                          │
         │--- stop_workers() ------>│
         │    (SIGTERM)             │
         │                          │
         │      exit                │
         │                          │
```
