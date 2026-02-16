# GPUModelRunner 重构设计文档

> **文档版本**: v1.0
> **创建日期**: 2026-02-16
> **状态**: 设计中
> **关联分析**: [God Class 分析报告](god_class_analysis.md)

---

## 一、背景与目标

### 1.1 当前问题

GPUModelRunner 是 FastDeploy 中最严重的 God Class，承担了 8+ 种职责：

| 指标 | 值 | 评估 |
|------|-----|------|
| 代码行数 | 3123 | 超大型类 |
| 方法数量 | 62 | 🔴 过多（>40） |
| 实例变量 | 41 | 🔴 复杂（>30） |
| 平均圈复杂度 | B (8.08) | 🟡 中等 |
| 最高圈复杂度 | E (40) `insert_tasks_v1` | 🔴 不稳定 |
| share_inputs 字段数 | 81 | 🔴 数据容器过大 |

### 1.2 重构目标

1. **降低职责耦合**：将独立职责拆分为独立组件
2. **提高可测试性**：拆分后的组件可独立测试，减少 mock 需求
3. **保持接口稳定**：对外接口 `execute_model()` 签名不变
4. **支持子类兼容**：IluvatarModelRunner、DCUModelRunner 继续正常工作

---

## 二、职责分析

### 2.1 现有职责清单

```
GPUModelRunner 承担的职责（8+ 种）：
├── 模型执行
│     └── execute_model(), execute_model_normal(), execute_model_overlap()
├── KV 缓存管理
│     └── initialize_kv_cache(), clear_cache(), update_share_input_block_num()
├── 输入预处理
│     └── insert_tasks_v1(), insert_prefill_inputs(), _prepare_inputs()
├── 采样逻辑
│     └── _dummy_sampler_run(), _pool()
├── CUDA Graph 管理
│     └── capture_model(), capture_model_prefill_and_mixed(), padding_cudagraph_inputs()
├── 视觉特征提取
│     └── extract_vision_features(), extract_vision_features_ernie(), extract_vision_features_qwen()
├── 推测解码
│     └── _init_speculative_proposer(), 相关推测解码逻辑
└── 热身/优化
      └── sot_warmup(), profile_run(), vision_encoder_compile()
```

### 2.2 高复杂度方法（CC ≥ 20）

| 方法 | CC 等级 | 复杂度值 | 建议 |
|------|---------|---------|------|
| `insert_tasks_v1()` | **E** | 40 | 🔴 必须拆分 |
| `_process_mm_features()` | **E** | 38 | 🔴 考虑拆分 |
| `_postprocess()` | **E** | 38 | 🔴 考虑拆分 |
| `insert_prefill_inputs()` | **E** | 34 | 🟡 关注 |
| `_prepare_inputs()` | **D** | 29 | 🟡 关注 |
| `_get_p_done_idxs_gd()` | **D** | 23 | 🟡 关注 |
| `initialize_kv_cache()` | **D** | 22 | 🟡 关注 |

---

## 三、状态耦合分析

### 3.1 共享状态清单

> ⚠️ **关键发现**：GPUModelRunner 的多个职责通过共享实例变量紧密耦合，简单按职责拆分会破坏数据一致性。

| 共享变量 | 涉及职责 | 严重程度 | 说明 |
|---------|---------|---------|------|
| `share_inputs` | 几乎所有职责 | 🔴 严重 | 81 字段的字典，是核心数据容器 |
| `forward_meta` | 模型执行/元数据/KV Cache/CUDA Graph | 🔴 严重 | 前向传播元数据，多处读写 |
| `model` | 模型执行/多模态/采样/权重管理 | 🟡 中等 | 模型实例，多职责依赖 |
| `attn_backends` | 元数据初始化/KV Cache/CUDA Graph | 🟡 中等 | Attention 后端列表 |
| `sampler` | 模型执行/输入管理/采样 | 🟡 中等 | 采样器实例 |
| `proposer` | 推测解码/模型执行/CUDA Graph | 🟡 中等 | 推测解码 Proposer |
| `cache_kvs_map` | KV Cache 管理/清理 | 🟡 中等 | KV Cache 映射 |

### 3.2 状态共享关系图

```
                    ┌───────────────────────────────────────────┐
                    │              share_inputs                 │
                    │      (81 fields, core data container)     │
                    └─────────────────────┬─────────────────────┘
                                          │
     ┌────────────┬────────────┬──────────┼──────────┬────────────┐
     ▼            ▼            ▼          ▼          ▼            ▼
 InputTask    ModelExec    KVCache    CUDAGraph   Multimodal
 Management   Core         Mgmt       Optimize    Processing

                    ┌───────────────────────────────────────────┐
                    │              forward_meta                 │
                    │       (forward propagation metadata)      │
                    └─────────────────────┬─────────────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          ▼                               ▼                               ▼
     ModelExec Core                 Metadata Init                 Attention Backend
```

### 3.3 方法调用依赖

```
execute_model (入口)
└── execute_model_normal / execute_model_overlap
    ├── _preprocess_and_execute_model
    │   ├── _get_p_done_idxs_gd
    │   ├── _process_reorder
    │   ├── _prepare_inputs
    │   │   └── initialize_forward_meta ← 元数据初始化
    │   └── padding_cudagraph_inputs ← CUDA Graph
    └── _postprocess
        ├── _pool (for pooling model)
        ├── model.compute_logits
        └── sampler(...) ← 采样

insert_tasks_v1 / insert_prefill_inputs (任务插入)
├── initialize_kv_cache ← KV Cache (懒初始化!)
├── _init_logits_processor
└── _process_mm_features
    ├── extract_vision_features ← 视觉特征
    │   ├── extract_vision_features_ernie
    │   ├── extract_vision_features_qwen
    │   └── extract_vision_features_paddleocr
    └── prepare_rope3d

capture_model (CUDA Graph 捕获)
└── _dummy_run
    ├── _prepare_inputs
    ├── padding_cudagraph_inputs
    └── _dummy_sampler_run / _dummy_pooler_run
```

---

## 四、拆分方案

### 4.1 拆分可行性评估

| 组件 | 可行性 | 风险点 | 现有模式参考 |
|------|--------|--------|-------------|
| `VisionFeatureExtractor` | ✅ 高 | 逻辑相对独立 | `EncoderCacheManager` |
| `CudaGraphManager` | ✅ 高 | 已有成熟模式 | `CudaGraphPiecewiseBackend` |
| `KVCacheManager` | ⚠️ 中 | 与 `share_inputs["caches"]` 深度耦合 | `PrefixCacheManager` |
| `SpeculativeDecodingExecutor` | ⚠️ 中 | 与 sampler 交互复杂 | `MTPProposer`/`NgramProposer` |
| `InputPreprocessor` | ❌ 低 | 核心执行流程，拆分成本 > 收益 | - |
| `SamplerExecutor` | ❌ 低 | 核心执行流程，拆分成本 > 收益 | - |

### 4.2 建议拆分组件

| 组件 | 职责 | 预估行数 | 优先级 | Phase |
|------|------|----------|--------|-------|
| `VisionFeatureExtractor` | 视觉特征提取 | ~400 | **P0** | Phase 1 |
| `CudaGraphManager` | CUDA Graph 捕获和管理 | ~350 | **P1** | Phase 2 |
| `KVCacheManager` | KV 缓存初始化和管理 | ~300 | P2 | Phase 3 |
| `SpeculativeDecodingExecutor` | 推测解码逻辑 | ~300 | P2 | Phase 3 |
| GPUModelRunner（精简后） | 模型执行协调 | ~1300 | - | - |

### 4.3 重构后架构

```
GPUModelRunner (精简后，~1300行，核心协调)
├── VisionFeatureExtractor (~400行)
│     └── extract_vision_features(), extract_vision_features_*()
├── CudaGraphManager (~350行)
│     └── capture_model(), padding_cudagraph_inputs()
├── KVCacheManager (~300行，待评估)
│     └── initialize_kv_cache(), clear_cache()
└── SpeculativeDecodingExecutor (~300行，待评估)
      └── _init_speculative_proposer(), speculative 相关逻辑
```

---

## 五、接口设计

### 5.1 VisionFeatureExtractor

```python
class VisionFeatureExtractor:
    """
    视觉特征提取器 - 从 GPUModelRunner 拆分

    职责:
    1. 提取图像/视频的视觉特征
    2. 支持多种模型后端 (Qwen-VL, ERNIE-VL, PaddleOCR 等)
    3. 管理视觉编码器缓存
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self._encoder_cache: Optional[EncoderCacheManager] = None

    def extract_features(
        self,
        images: List[torch.Tensor],
        model_type: str = "default"
    ) -> torch.Tensor:
        """
        提取视觉特征

        Args:
            images: 图像张量列表
            model_type: 模型类型 (qwen/ernie/paddleocr)

        Returns:
            视觉特征张量
        """
        pass

    def extract_features_qwen(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Qwen-VL 特定的特征提取"""
        pass

    def extract_features_ernie(self, images: List[torch.Tensor]) -> torch.Tensor:
        """ERNIE-VL 特定的特征提取"""
        pass

    def compile_encoder(self) -> None:
        """编译视觉编码器以优化性能"""
        pass
```

### 5.2 CudaGraphManager

```python
class CudaGraphManager:
    """
    CUDA Graph 管理器 - 从 GPUModelRunner 拆分

    职责:
    1. CUDA Graph 捕获和缓存
    2. 根据 batch 特征选择合适的 CUDA Graph
    3. 管理 CUDA Graph 的生命周期

    参考: vLLM CUDAGraphWrapper / SGLang CudaGraphRunner
    """

    def __init__(self, config: CudaGraphConfig):
        self.mode = config.cudagraph_mode  # NONE/PIECEWISE/FULL
        self.capture_sizes = config.capture_sizes
        self._full_graphs: Dict[BatchDescriptor, torch.cuda.CUDAGraph] = {}
        self._piecewise_graphs: Dict[BatchDescriptor, torch.cuda.CUDAGraph] = {}

    def dispatch(
        self,
        batch_desc: BatchDescriptor
    ) -> Tuple[RuntimeMode, BatchDescriptor]:
        """
        根据 batch 特征决定使用哪个 CUDA Graph

        Args:
            batch_desc: batch 描述符 (num_tokens, num_reqs, uniform 等)

        Returns:
            (runtime_mode, 调整后的 batch_desc)
        """
        pass

    def capture_if_needed(
        self,
        batch_desc: BatchDescriptor,
        forward_fn: Callable[[], torch.Tensor]
    ) -> None:
        """
        按需捕获 CUDA Graph

        Args:
            batch_desc: batch 描述符
            forward_fn: 前向传播函数
        """
        pass

    def replay(
        self,
        batch_desc: BatchDescriptor
    ) -> Optional[torch.Tensor]:
        """
        回放 CUDA Graph

        Args:
            batch_desc: batch 描述符

        Returns:
            如果有缓存的 graph，返回输出；否则返回 None
        """
        pass

    def padding_inputs(
        self,
        share_inputs: Dict[str, Any],
        batch_desc: BatchDescriptor
    ) -> Dict[str, Any]:
        """
        为 CUDA Graph 填充输入

        Args:
            share_inputs: 共享输入字典
            batch_desc: batch 描述符

        Returns:
            填充后的输入字典
        """
        pass
```

### 5.3 ModelRunner Public API

```python
class ModelRunnerBase(ABC):
    """
    Public API - 模型运行器抽象基类

    重构后 GPUModelRunner 继承此接口，保证签名稳定
    """

    @abstractmethod
    def execute_model(self, model_input: ModelInput) -> ModelOutput:
        """
        执行模型推理 - 签名稳定

        Args:
            model_input: 模型输入

        Returns:
            模型输出
        """
        pass

    @abstractmethod
    def initialize_kv_cache(self, num_blocks: int) -> None:
        """
        初始化 KV Cache - 签名稳定

        Args:
            num_blocks: KV Cache 块数量
        """
        pass

    @abstractmethod
    def insert_tasks(self, tasks: List[Task]) -> None:
        """
        插入推理任务 - 签名稳定

        Args:
            tasks: 任务列表
        """
        pass
```

---

## 六、子类兼容性

### 6.1 现有继承关系

```python
GPUModelRunner
├── IluvatarModelRunner  # 覆盖: __init__, _initialize_attn_backend
└── DCUModelRunner       # 覆盖: __init__, initialize_forward_meta
```

### 6.2 兼容策略

| 子类 | 覆盖方法 | 兼容方案 |
|------|---------|---------|
| IluvatarModelRunner | `__init__` | 保持 GPUModelRunner 构造函数签名不变 |
| IluvatarModelRunner | `_initialize_attn_backend` | 方法保留在 GPUModelRunner，不拆分 |
| DCUModelRunner | `__init__` | 保持 GPUModelRunner 构造函数签名不变 |
| DCUModelRunner | `initialize_forward_meta` | 方法保留在 GPUModelRunner，不拆分 |

> **原则**：被子类覆盖的方法不拆分，确保子类无需修改。

---

## 七、业界参考

### 7.1 vLLM / SGLang 对比

> ⚠️ 以下数据为参考值，实际实现时请参考最新版本源码。

| 对比维度 | FastDeploy | vLLM | SGLang |
|---------|-----------|---------|--------|
| 文件行数 | 3123 | ~4500* | ~2677* |
| CUDA Graph | ❌ 内嵌 | ✅ `CUDAGraphWrapper` 独立 | ✅ `CudaGraphRunner` 独立 |
| KV Cache | ❌ 内嵌 | ✅ `KVConnectorMixin` | ✅ `KVCacheMixin` |
| 多模态 | ❌ 内嵌 | ✅ 独立模块 | 部分独立 |
| Sampler | ✅ 独立 | ✅ 独立 | ✅ 独立 |

### 7.2 设计启示

- vLLM 和 SGLang 都使用 **Mixin 模式** 分离 KV Cache 逻辑
- CUDA Graph 管理都是 **独立组件**（CUDAGraphWrapper / CudaGraphRunner）
- 推荐参考 SGLang 的 `CudaGraphRunner` 和 `ModelRunnerKVCacheMixin`

---

## 八、实施计划

### 8.1 Phase 1: VisionFeatureExtractor 抽取

**目标**：将视觉特征提取逻辑独立

**范围**：
- `extract_vision_features()`
- `extract_vision_features_qwen()`
- `extract_vision_features_ernie()`
- `extract_vision_features_paddleocr()`
- `vision_encoder_compile()`

**预期收益**：
- 减少 ~400 行代码
- 多模态功能可独立测试
- 支持后续视觉编码器优化

### 8.2 Phase 2: CudaGraphManager 抽取

**目标**：将 CUDA Graph 管理逻辑独立

**范围**：
- `capture_model()`
- `capture_model_prefill_and_mixed()`
- `padding_cudagraph_inputs()`
- 相关 CUDA Graph 状态

**预期收益**：
- 减少 ~350 行代码
- CUDA Graph 策略可独立配置
- 参考 vLLM CUDAGraphWrapper 支持更多 mode

### 8.3 Phase 3: 评估进一步拆分

**待评估组件**：
- KVCacheManager：与 share_inputs 耦合严重，需评估解耦成本
- SpeculativeDecodingExecutor：与 sampler 交互复杂，需评估影响

---

## 九、测试策略

### 9.1 单元测试

```python
# VisionFeatureExtractor 单元测试
def test_extract_features_qwen():
    extractor = VisionFeatureExtractor(model, config, device)
    images = [torch.randn(3, 224, 224)]
    features = extractor.extract_features_qwen(images)
    assert features.shape == expected_shape

# CudaGraphManager 单元测试
def test_cudagraph_capture():
    manager = CudaGraphManager(config)
    batch_desc = BatchDescriptor(num_tokens=32, num_reqs=1)
    manager.capture_if_needed(batch_desc, forward_fn)
    assert batch_desc in manager._full_graphs
```

### 9.2 集成测试

```python
# 确保拆分后 GPUModelRunner 行为不变
def test_gpu_model_runner_integration():
    runner = GPUModelRunner(config)
    # 使用真实输入测试完整流程
    output = runner.execute_model(model_input)
    assert output == expected_output
```

### 9.3 性能基准

重构前后需对比以下指标：

| 指标 | 容忍阈值 |
|------|----------|
| Prefill 延迟 | ≤ 5% 退化 |
| Decode 吞吐 | ≤ 3% 退化 |
| 多模态推理延迟 | ≤ 5% 退化 |
| GPU 内存占用 | ≤ 2% 增加 |

---

## 十、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 回归风险 | 重构前建立集成测试覆盖；每 PR 独立验证 |
| 性能影响 | 热路径保持内联；拆分后性能基准测试 |
| 子类兼容 | 保持被覆盖方法签名不变；子类同步更新 |
| 回滚需求 | 保留旧代码路径，环境变量 `FD_USE_LEGACY_GPU_MODEL_RUNNER=1` 切换 |

---

## 附录：CUDA Graph 解耦参考

### A.1 vLLM 的 CUDA Graph 架构

```
CudagraphDispatcher (中央调度器)
├── 维护 FULL/PIECEWISE 两套 dispatching keys
├── 根据 BatchDescriptor 决定 runtime_mode
└── 是 CUDA Graph 的唯一真相来源

CUDAGraphWrapper (封装器)
├── 封装 capture/replay 逻辑
├── 每个 wrapper 绑定特定 runtime_mode
└── 信任 forward_context 中的调度决策

BatchDescriptor (调度 Key)
├── num_tokens: 填充后的 token 数
├── num_reqs: 请求数
├── uniform: 是否所有请求 query_len 相同
└── has_lora: 是否使用 LoRA
```

### A.2 CUDA Graph 模式对比

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `NONE` | 关闭 CUDA Graph | 调试 |
| `PIECEWISE` | 分段 CUDA Graph | 最灵活，兼容所有 backend |
| `FULL` | 完整 CUDA Graph | 小模型、短 prompt |
| `FULL_DECODE_ONLY` | 仅 decode 使用完整 CUDA Graph | P/D 分离场景 |
| `FULL_AND_PIECEWISE` | decode 用 FULL，prefill 用 PIECEWISE | **推荐默认** |

### A.3 Attention Backend 兼容性

| Attention Backend | CUDA Graph 支持级别 |
|-------------------|---------------------|
| FlashAttention v2 | `UNIFORM_BATCH` |
| FlashAttention v3 | `ALWAYS` |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` |
| Triton Attention | `ALWAYS` |
