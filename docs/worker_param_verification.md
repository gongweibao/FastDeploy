# Worker 参数验证流程

## 概述

本文档描述了如何验证老架构和新架构向 Worker 进程传递的参数是否完全一致。

## 验证原理

完整验证包括两个层次：

### 层次一：参数构造逻辑对比（端到端）

直接调用老架构和新架构的参数构造逻辑，对比提取的参数。

```
    配置对象 (FDConfig)
    ↓
    老架构: common_engine.py:_start_worker_service()
    新架构: process_manager.py:start_workers()
    ↓
    构造的命令行参数
    ↓
    解析并保存到文件
```

### 层次二：Worker 参数解析对比

模拟 Worker 命令行并触发参数转储，对比解析结果。

## 实现组件

### 1. 参数转储模块

**文件**: `fastdeploy/engine/components/param_dump.py`

主要类：

- `WorkerParamDumper`: 负责将参数转储到文件
  - `is_enabled()`: 检查是否启用参数转储
  - `_serialize_args()`: 将 args 对象序列化为字典
  - `_serialize_value()`: 递归序列化各种类型的值
  - `dump_params()`: 执行转储操作

- `ParamComparator`: 负责对比两组参数
  - `load_params()`: 加载指定架构的参数文件
  - `compare()`: 对比参数差异
  - `print_report()`: 打印对比报告

### 2. Worker 集成

**文件**: `fastdeploy/worker/worker_process.py:1047`

在 `parse_args()` 函数返回前添加参数转储调用：

```python
args = parser.parse_args()

# Dump parameters for verification if enabled
try:
    from fastdeploy.engine.components.param_dump import dump_worker_params
    dump_worker_params(args)
except ImportError:
    pass

return args
```

### 3. 验证脚本

#### 方式一：参数构造逻辑对比（推荐）

**文件**: `scripts/compare_params_direct.py`

直接调用老新架构的参数构造逻辑并对比结果。

```bash
python scripts/compare_params_direct.py
```

**优点**：
- 无需依赖真实模型文件
- 无需 GPU 环境
- 执行速度快
- 直接调用实际的参数构造代码逻辑

#### 方式二：Worker 参数解析对比

**文件**: `scripts/capture_and_compare_params.sh`

模拟 Worker 命令行并触发参数转储。

```bash
bash scripts/capture_and_compare_params.sh
```

**优点**：
- 验证 Worker 端参数解析流程
- 更接近真实运行场景

#### 方式三：单元测试

**文件**: `tests/engine/test_param_consistency.py`

```bash
pytest tests/engine/test_param_consistency.py -v
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FD_DUMP_WORKER_PARAMS` | `0` | 启用 Worker 参数转储（设置为 `1` 启用） |
| `FD_PARAM_DUMP_DIR` | `./param_dumps` | 参数转储目录 |
| `FD_USE_NEW_ENGINE_ARCHITECTURE` | `0` | 架构类型（`0`=老架构，`1`=新架构） |
| `FD_CAPTURE_LAUNCH_PARAMS` | `0` | 启用启动参数捕获（设置为 `1` 启用） |

## 输出文件格式

参数文件以 JSON 格式保存，文件名格式为：

```
worker_params_<arch>_<pid>.json
launch_command_<arch>.sh
```

示例：`worker_params_old_1002863.json`

### 示例内容

```json
{
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "gpu_memory_utilization": 0.9,
  "model": "/tmp/test_model",
  "block_size": 16,
  "tensor_parallel_size": 1,
  "dtype": "float16",
  "engine_worker_queue_port": "9923",
  "pod_ip": "127.0.0.1",
  ...
}
```

## 验证结果

### 层次一：参数构造逻辑对比

```
✓ No differences found! Parameters are IDENTICAL.
```

**结论**：老架构和新架构的参数构造逻辑完全一致。

## 架构对比

| 方面 | 老架构 | 新架构 | 是否一致 |
|------|--------|--------|----------|
| 启动工具 | `paddle.distributed.launch` | `paddle.distributed.launch` | ✅ 一致 |
| 进程创建 | `subprocess.Popen` | `subprocess.Popen` | ✅ 一致 |
| shell 参数 | `shell=True` | `shell=True` | ✅ 一致 |
| 进程组 | `preexec_fn=os.setsid` | `preexec_fn=os.setsid` | ✅ 一致 |
| 日志重定向 | `2>{log_dir}/launch_worker.log` | `2>{log_dir}/launch_worker.log` | ✅ 一致 |
| 参数构造位置 | `common_engine.py:_start_worker_service()` | `process_manager.py:start_workers()` | ✅ 逻辑一致 |
| 参数来源 | `self.cfg` (各 config 对象) | `self.cfg` (各 config 对象) | ✅ 一致 |

## 关键文件

| 文件 | 说明 |
|------|------|
| `fastdeploy/engine/components/param_dump.py` | Worker 参数转储和对比核心模块 |
| `fastdeploy/engine/components/launch_param_capture.py` | 启动参数捕获模块 |
| `fastdeploy/worker/worker_process.py` | Worker 进程入口（集成参数转储） |
| `scripts/compare_params_direct.py` | 参数构造逻辑对比 |
| `scripts/capture_and_compare_params.sh` | Worker 参数解析对比 |
| `tests/engine/test_param_consistency.py` | 单元测试 |
| `scripts/add_param_dump.py` | 辅助脚本：添加参数转储到已安装的包 |

## CI/CD 集成

可以在 CI 流程中添加验证步骤：

```yaml
- name: Verify Worker Parameters
  run: |
    export FD_DUMP_WORKER_PARAMS=1
    pytest tests/engine/test_param_consistency.py || exit 1
```

## 注意事项

1. 参数转储仅在 `FD_DUMP_WORKER_PARAMS=1` 时启用
2. 启动参数捕获仅在 `FD_CAPTURE_LAUNCH_PARAMS=1` 时启用
3. 转储不会影响正常功能，仅在开发/测试时使用
4. 对比时不考虑 `_metadata` 字段（包含 PID、时间戳等运行时信息）
5. 支持递归对比嵌套的字典和列表结构
