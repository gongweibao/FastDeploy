# GPUModelRunner 测试策略与原则

## 核心目标

为 `GPUModelRunner` 编写高质量的集成测试，遵循以下原则：

1. **真实性优先** - 使用真实组件和数据结构
2. **最小化 mock** - 仅 mock 外部不可控依赖
3. **快速可运行** - 通过小模型、小数据保证测试速度
4. **易于维护** - 清晰的 fixture 结构和测试分层

---

## Mock 使用策略

### ✅ 允许 Mock 的场景

| 场景 | 原因 | 示例 |
|------|------|------|
| 分布式通信 | 需要多进程，单元测试难以模拟 | `paddle.distributed.all_gather_object` |
| 外部输出 | ZMQ、共享内存等 IPC | `ZmqIpcClient.send_pyobj` |
| 文件系统 | 避免真实文件操作 | `open()`, `os.path` |
| 确定性控制 | 需要固定随机结果 | 随机数生成器（可选） |

### ❌ 禁止 Mock 的场景

| 场景 | 原因 | 替代方案 |
|------|------|----------|
| 核心数据结构 | `MagicMock` 丢失接口约束 | 使用真实类 + 小数据 |
| 模型组件（含 Vision Encoder） | mock 会隐藏真实 bug | 使用真实小模型 |
| KV cache | 真实交互才能验证 | 使用真实 Tensor |
| 采样器 | 需要验证采样逻辑 | 配置简单参数 |

### 🔄 建议使用 Fake 实现的场景

| 场景 | 策略 |
|------|------|
| Attention Backend | 继承接口，简化 forward 逻辑（可选） |
| Proposer (Speculative) | 继承接口，简单返回 draft tokens（可选） |
| Vision Encoder | 优先真实实现，无模型时可用假实现 |

---

## 测试数据策略

### 使用真实类，假数据内容

```python
# ✅ 正确：真实 Request 类，最小数据
request = Request(
    request_id="test_001",
    prompt_token_ids=[1, 2, 3],  # 短序列
    block_tables=[0, 1, 2],
    task_type=RequestType.PREFILL,
)

# ❌ 错误：mock Request 类
request = MagicMock(spec=Request)
request.request_id = "test_001"
```

### 数据大小控制

| 数据类型 | 测试时推荐大小 |
|---------|---------------|
| 序列长度 | 4-16 tokens |
| batch size | 1-8 个请求 |
| block 数量 | 4-10 blocks |
| KV cache shape | 小维度，但保持比例 |
| 图像 | 标准尺寸 (224x224) 或模型常用尺寸 |

---

## Fixture 分层设计

```
Session 级别 (所有测试共享一次)
├── fd_config              # 测试配置
└── model_runner           # 小模型 runner

Function 级别 (每个测试独立)
├── runner                 # 清理后的 runner
├── sample_*_request       # 单个请求样本
└── sample_batch_requests  # 批量请求样本

Mock 级别 (按需应用)
├── mock_distributed       # 分布式通信 mock
└── mock_external_output   # 外部输出 mock
```

### Fixture 生命周期

| 级别 | 用途 | 优势 |
|------|------|------|
| Session | 模型加载、KV cache 初始化 | 避免重复加载，加速测试 |
| Function | 每次测试前清理状态 | 保证测试隔离性 |
| Mock | 特定测试场景 | 按需启用 |

---

## 测试分层

### 1. 基础功能测试（无需模型）

**目标函数**：
- `exist_prefill()` / `exist_decode()`
- `only_prefill()` / `only_decode()`
- `collect_distributed_status()`
- `_get_feature_positions()`

**测试重点**：逻辑正确性，无需加载模型

### 2. 输入处理测试（轻量）

**目标函数**：
- `insert_tasks_v1()`
- `insert_prefill_inputs()`
- `_prepare_inputs()`

**测试重点**：share_inputs 状态正确性，使用真实 Request 对象

### 3. 多模态测试（真实实现优先）

**目标函数**：
- `_process_mm_features()`
- `extract_vision_features_*()`
- `prepare_rope3d()`

**测试重点**：完整的视觉特征提取流程，优先使用真实 vision encoder

**图片策略**：
- 使用标准尺寸（224x224）的主流测试图片
- 使用模型常用尺寸（如 Qwen 的 336x336）验证兼容性
- 使用边界尺寸（512x512）进行压力测试
- 图片内容固定（纯色或预设图案）保证输出确定性

**图片 Batch 场景**：
| 场景 | 描述 | 测试价值 |
|------|------|----------|
| 单请求单图片 | 基础场景 | 验证基本流程 |
| 单请求多图片 | 多图模式 | 验证多图特征合并逻辑 |
| 多请求各一张图片 | 并发场景 | 验证 batch 处理能力 |
| 多请求混合 | 有图+无图混合 | 验证边界条件处理 |

**确定性控制**：
- 设置固定随机种子：`np.random.seed(42)`, `paddle.seed(42)`
- 启用 deterministic 模式（如支持）：`paddle.set_cudnn_deterministic(True)`

**备选方案**：当没有多模态模型时，可使用假实现进行逻辑流程测试

### 4. 模型执行测试（真实模型）

**目标函数**：
- `execute_model()`
- `_preprocess_and_execute_model()`
- `_postprocess()`
- `_save_model_output()`

**测试重点**：端到端流程，使用真实小模型

### 5. 优化功能测试（可选）

**目标函数**：
- CUDA Graph 捕获
- Speculative Decoding
- Overlap Schedule

**测试重点**：优化路径，可能需要特定配置

---

## 配置控制而非 Mock

使用配置参数控制测试场景：

```python
# 通过配置控制
GRAPH_OPT_CONFIG = {
    "use_cudagraph": False,  # 关闭以测试基础逻辑
}

SCHEDULER_CONFIG = {
    "enable_overlap_schedule": False,  # 关闭复杂特性
}

# 而不是 mock
# ❌ runner.use_cudagraph = False  # 可能不一致
```

---

## 辅助函数

### 数据验证函数

```python
def assert_share_inputs_valid(runner: GPUModelRunner, num_running: int):
    """验证 share_inputs 的有效性"""
    # 检查基本信息、shape、边界条件

def assert_batch_consistency(runner: GPUModelRunner, requests: List[Request]):
    """验证 batch 与 requests 的一致性"""
    # 逐个检查 request 与 share_inputs 的对应关系
```

### 测试数据构建

```python
class TestDataBuilder:
    """测试数据构建工具"""

    @staticmethod
    def create_prefill_request(...) -> Request:
        """创建 Prefill 类型的真实 Request"""

    @staticmethod
    def create_decode_request(...) -> Request:
        """创建 Decode 类型的真实 Request"""

    @staticmethod
    def create_batch_requests(...) -> List[Request]:
        """创建一批 Request 对象"""
```

---

## 测试模型选择

### 推荐模型

| 用途 | 推荐模型 | 模型类型 | 原因 |
|------|---------|---------|------|
| **文本模型** | Qwen3-0.6B 或 Qwen2-0.5B | 文本生成 | 小而全，加载快，支持良好 |
| **多模态模型** | Ernie-4.5-VL-Lite | 视觉语言 | FastDeploy 原生支持，中文友好 |
| **备选多模态** | Qwen-VL-Chat | 视觉语言 | 开源易获取 |

### 模型获取方式

| 模型 | 获取方式 | 下载命令 |
|------|---------|---------|
| Qwen3-0.6B | Hugging Face / ModelScope | `pip install huggingface_hub; huggingface-cli download Qwen/Qwen3-0.6B` |
| Qwen2-0.5B | Hugging Face / ModelScope | `pip install huggingface_hub; huggingface-cli download Qwen/Qwen1.5-0.5B` |
| Ernie-4.5-VL-Lite | 百度文心内网 / 预编译 | 内网下载或联系模型团队 |
| Qwen-VL-Chat | Hugging Face | `huggingface-cli download Qwen/Qwen-VL-Chat` |

### 环境变量配置

```bash
# 设置文本模型路径（必须，测试前必须配置）
export FD_TEST_TEXT_MODEL_PATH=/path/to/Qwen3-0.6B

# 设置多模态模型路径（多模态测试必须）
export FD_TEST_MM_MODEL_PATH=/path/to/Ernie-4.5-VL-Lite

# 设置 GPU ID
export FD_TEST_GPU_ID=0
```

**注意**：
- 文本模型：基础功能测试必须依赖，未配置时测试报错
- 多模态模型：多模态测试必须依赖，未配置时相关测试报错

### 模型路径查找策略

```python
# conftest.py 中的配置逻辑
def get_text_model_path():
    """获取文本模型路径，按优先级查找"""
    # 1. 环境变量
    path = os.environ.get("FD_TEST_TEXT_MODEL_PATH")
    if path and os.path.exists(path):
        return path

    # 2. 默认路径（本地已有）
    default_paths = [
        "/root/paddlejob/workspace/.../models/Qwen/Qwen2.5-7B",
        "./models/Qwen3-0.6B",
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path

    # 3. 无可用模型，直接报错
    raise RuntimeError(
        f"Text model not found. Please set FD_TEST_TEXT_MODEL_PATH "
        f"or ensure model exists at one of: {default_paths}"
    )

def get_multimodal_model_path():
    """获取多模态模型路径"""
    path = os.environ.get("FD_TEST_MM_MODEL_PATH")
    if path and os.path.exists(path):
        return path

    raise RuntimeError(
        "Multimodal model not found. Please set FD_TEST_MM_MODEL_PATH."
    )
```

---

## 运行测试

```bash
# 运行所有测试（前提：已配置模型路径）
pytest tests/worker/gpu_model_runner/

# 运行特定类别
pytest tests/worker/gpu_model_runner/ -m "multimodal"
pytest tests/worker/gpu_model_runner/ -m "distributed"

# 跳过慢速测试
pytest tests/worker/gpu_model_runner/ -m "not slow"

# 调试单个测试
pytest tests/worker/gpu_model_runner/ -k "test_name" -vv -s
```

---

## 环境变量

| 环境变量 | 说明 | 要求 |
|---------|------|------|
| `FD_TEST_TEXT_MODEL_PATH` | 文本模型路径 | 必须配置 |
| `FD_TEST_MM_MODEL_PATH` | 多模态模型路径 | 多模态测试必须配置 |
| `FD_TEST_GPU_ID` | 测试用的 GPU ID | 0（可选） |
| `FD_ENABLE_V1_KVCACHE_SCHEDULER` | 启用 V1 调度器 | 0（可选） |

---

## 常见问题

### Q: 模型加载报错怎么办？
A: 根据报错类型排查：

| 错误类型 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `FileNotFoundError` | 模型路径不存在 | 检查 `FD_TEST_TEXT_MODEL_PATH` 路径是否正确，确认目录存在 |
| `KeyError: config.json` | 模型目录缺少配置文件 | 确保模型目录包含 `config.json` 或 `config.yaml` |
| `ValueError: model_type` | 模型类型不支持 | 检查 config.json 中的 model_type 是否为 FastDeploy 支持的类型 |
| `OSError: weight` | 权重文件缺失或路径错误 | 确认模型目录完整，检查 `model_state.pdparams` 或 `.bin` 文件存在 |
| `RuntimeError: CUDA OOM` | 显存不足 | 减小 batch size，使用更小的模型，或设置 `FD_TEST_GPU_ID` 指向更空闲的 GPU |
| `ImportError: No module` | 依赖缺失 | 安装模型依赖：`pip install paddlepaddle-paddle paddlenlp transformers` |

### Q: 模型测试可以使用 @pytest.mark.requires_model 吗？
A: 不可以。测试必须要有模型，未配置模型路径时测试直接报错，不会自动 skip。请在运行测试前确保已设置 `FD_TEST_TEXT_MODEL_PATH`。

### Q: 如何测试 CUDA Graph？
A: 使用 `@pytest.mark.cudagraph` 标记，且需要真实 GPU 环境。

### Q: 如何测试多模态功能？
A: 优先使用真实 vision encoder，配合标准尺寸（224x224）的固定测试图片。设置随机种子保证输出确定性，验证形状和流程正确性而非精确数值。

### Q: 多模态测试需要真实的图片吗？
A: 需要，使用标准尺寸的固定图片（可内嵌 base64 或预置文件）。真实尺寸图片能发现边界问题（如序列超长、位置编码溢出），且时间成本可忽略（瓶颈在模型加载）。

### Q: 如何调试单个测试？
A: 使用 `pytest -k "test_name" -vv -s` 查看详细输出。

---

## 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| 基础功能（状态检查） | > 90% |
| 输入处理 | > 80% |
| 多模态 | > 70% |
| 模型执行 | > 60% |
| 优化功能 | > 50% |

---

## 持续改进

1. **定期回顾** - 检查是否有不必要的 mock 被引入
2. **性能监控** - 关注测试运行时间，及时优化
3. **新增功能测试** - 新增功能时同步添加测试
4. **文档更新** - 测试策略变更时更新本文档
