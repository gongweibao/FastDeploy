# 配置系统重构详细设计

> **文档版本**: v1.0
> **创建日期**: 2026-02-14
> **关联问题**: config.py 过大、EngineArgs 参数爆炸、环境变量混乱

---

## 一、背景与目标

### 1.1 当前问题

| 问题 | 现状 | 影响 |
|------|------|------|
| config.py 过大 | 2184 行，20+ 配置类 | 维护困难、合并冲突 |
| EngineArgs 参数爆炸 | 100+ 参数在单个 dataclass | 学习成本高、参数依赖隐式 |
| 环境变量混乱 | 95+ 变量，命名不统一 | 配置来源分散、难以追踪 |

### 1.2 重构目标

- config.py 拆分为独立模块，每个模块 < 500 行
- EngineArgs 按功能分组，支持 YAML 配置文件
- 环境变量统一管理，添加类型验证

---

## 二、config.py 拆分方案

### 2.1 拆分后的目录结构

```
fastdeploy/config/
├── __init__.py              # 统一导出，保持向后兼容
├── base.py                  # 基础类和工具函数 (~80行)
├── model.py                 # 模型相关配置 (~500行)
├── parallel.py              # 并行相关配置 (~170行)
├── cache.py                 # 缓存相关配置 (~180行)
├── speculative.py           # 投机解码配置 (~150行)
├── optimization.py          # 图优化配置 (~280行)
├── device.py                # 设备配置 (~20行)
├── load.py                  # 模型加载配置 (~40行)
├── scheduler.py             # 调度相关配置 (~60行)
└── fd_config.py             # 顶层 FDConfig (~550行)
```

### 2.2 类到文件的映射

| 原类名 | 行号范围 | 目标文件 | 说明 |
|-------|---------|---------|------|
| `MoEPhase` | 111-130 | `base.py` | 基础枚举 |
| `ErnieArchitectures` | 131-193 | `model.py` | 模型架构定义 |
| `ModelConfig` | 194-601 | `model.py` | 模型配置（408行） |
| `ParallelConfig` | 602-712 | `parallel.py` | 并行配置 |
| `SpeculativeConfig` | 713-861 | `speculative.py` | 投机解码 |
| `DeviceConfig` | 862-876 | `device.py` | 设备配置 |
| `GraphOptimizationConfig` | 877-1070 | `optimization.py` | 图优化 |
| `PlasAttentionConfig` | 1071-1134 | `optimization.py` | 注意力优化 |
| `EarlyStopConfig` | 1135-1202 | `optimization.py` | 早停配置 |
| `LoadChoices` | 1203-1210 | `load.py` | 加载选项枚举 |
| `LoadConfig` | 1211-1240 | `load.py` | 加载配置 |
| `PoolerConfig` | 1241-1275 | `model.py` | 池化配置 |
| `EPLBConfig` | 1276-1332 | `parallel.py` | 专家负载均衡 |
| `CacheConfig` | 1333-1506 | `cache.py` | 缓存配置（174行） |
| `RouterConfig` | 1507-1528 | `scheduler.py` | 路由配置 |
| `CommitConfig` | 1529-1585 | `base.py` | 提交配置 |
| `StructuredOutputsConfig` | 1586-1607 | `model.py` | 结构化输出 |
| `RoutingReplayConfig` | 1608-1641 | `scheduler.py` | 路由重放 |
| `FDConfig` | 1642-2184 | `fd_config.py` | 顶层配置（542行） |

### 2.3 向后兼容：`__init__.py`

```python
# fastdeploy/config/__init__.py
"""
配置模块 - 保持向后兼容的导出

使用方式不变：
    from fastdeploy.config import ModelConfig, CacheConfig, FDConfig
"""

from fastdeploy.config.base import MoEPhase, CommitConfig
from fastdeploy.config.model import (
    ErnieArchitectures,
    ModelConfig,
    PoolerConfig,
    StructuredOutputsConfig,
)
from fastdeploy.config.parallel import ParallelConfig, EPLBConfig
from fastdeploy.config.cache import CacheConfig
from fastdeploy.config.speculative import SpeculativeConfig
from fastdeploy.config.optimization import (
    GraphOptimizationConfig,
    PlasAttentionConfig,
    EarlyStopConfig,
)
from fastdeploy.config.device import DeviceConfig
from fastdeploy.config.load import LoadChoices, LoadConfig
from fastdeploy.config.scheduler import RouterConfig, RoutingReplayConfig
from fastdeploy.config.fd_config import FDConfig

__all__ = [
    "MoEPhase",
    "ErnieArchitectures",
    "ModelConfig",
    "ParallelConfig",
    "SpeculativeConfig",
    "DeviceConfig",
    "GraphOptimizationConfig",
    "PlasAttentionConfig",
    "EarlyStopConfig",
    "LoadChoices",
    "LoadConfig",
    "PoolerConfig",
    "EPLBConfig",
    "CacheConfig",
    "RouterConfig",
    "CommitConfig",
    "StructuredOutputsConfig",
    "RoutingReplayConfig",
    "FDConfig",
]
```

### 2.4 兼容层（可选）

保留原 `config.py` 作为兼容层，避免修改所有 import：

```python
# fastdeploy/config.py (兼容层)
"""
此文件保留用于向后兼容，请使用 fastdeploy.config 模块
"""
from fastdeploy.config import *  # noqa: F401, F403
```

---

## 三、EngineArgs 重构方案

### 3.1 当前问题

```python
# 当前：100+ 参数混在一起
@dataclass
class EngineArgs:
    model: str = "baidu/ernie-45-turbo"
    tokenizer: str = None                    # 类型错误
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    # ... 100+ 参数
```

### 3.2 重构后：分组嵌套

```python
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelArgs:
    """模型相关参数"""
    model: str = "baidu/ernie-45-turbo"
    tokenizer: Optional[str] = None
    max_model_len: int = 2048
    trust_remote_code: bool = True
    dtype: str = "auto"


@dataclass
class ParallelArgs:
    """并行相关参数"""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1


@dataclass
class CacheArgs:
    """缓存相关参数"""
    gpu_memory_utilization: float = 0.9
    block_size: int = 64
    enable_prefix_caching: bool = True
    max_num_seqs: int = 8


@dataclass
class SchedulerArgs:
    """调度相关参数"""
    scheduler_name: str = "local"
    max_num_batched_tokens: Optional[int] = None
    splitwise_role: str = "mixed"


@dataclass
class EngineArgs:
    """引擎配置 - 组合各子配置"""
    model: ModelArgs = field(default_factory=ModelArgs)
    parallel: ParallelArgs = field(default_factory=ParallelArgs)
    cache: CacheArgs = field(default_factory=CacheArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)

    @classmethod
    def from_yaml(cls, path: str) -> "EngineArgs":
        """从 YAML 配置文件加载"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelArgs(**data.get('model', {})),
            parallel=ParallelArgs(**data.get('parallel', {})),
            cache=CacheArgs(**data.get('cache', {})),
            scheduler=SchedulerArgs(**data.get('scheduler', {})),
        )

    def validate(self) -> list:
        """验证参数合法性"""
        errors = []
        if self.cache.gpu_memory_utilization > 1.0:
            errors.append("gpu_memory_utilization must be <= 1.0")
        if self.parallel.tensor_parallel_size < 1:
            errors.append("tensor_parallel_size must be >= 1")
        return errors
```

### 3.3 YAML 配置文件示例

```yaml
# config.yaml
model:
  model: "baidu/ernie-45-turbo"
  max_model_len: 8192
  dtype: "bfloat16"

parallel:
  tensor_parallel_size: 4
  pipeline_parallel_size: 1

cache:
  gpu_memory_utilization: 0.9
  block_size: 64
  enable_prefix_caching: true

scheduler:
  scheduler_name: "local"
  max_num_seqs: 16
```

### 3.4 CLI 兼容

```python
# 保持 CLI 参数兼容
@staticmethod
def add_cli_args(parser: ArgumentParser) -> ArgumentParser:
    # 模型参数组
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default="baidu/ernie-45-turbo")
    model_group.add_argument("--max-model-len", type=int, default=2048)

    # 并行参数组
    parallel_group = parser.add_argument_group("Parallel Configuration")
    parallel_group.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)

    # 支持从配置文件加载
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    return parser
```

---

## 四、环境变量统一方案

### 4.1 当前问题

```python
# 命名不统一
FD_DEBUG                        # FD_ 前缀
ENABLE_V1_KVCACHE_SCHEDULER     # ENABLE_ 前缀
USE_FLASH_ATTN                  # 无前缀

# 代码中直接使用 os.getenv（绕过 envs.py）
os.getenv("CUDA_VISIBLE_DEVICES")
os.environ.get("FD_SOME_FLAG", "0")
```

### 4.2 重构后：类型安全的环境变量

```python
# fastdeploy/envs.py

from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Optional
import os

T = TypeVar('T')


@dataclass
class EnvVar(Generic[T]):
    """类型安全的环境变量定义"""
    name: str
    default: T
    doc: str
    parser: Callable[[str], T] = str

    def get(self) -> T:
        """获取环境变量值"""
        raw = os.getenv(self.name)
        if raw is None:
            return self.default
        try:
            return self.parser(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for {self.name}: {raw}") from e

    def set(self, value: T) -> None:
        """设置环境变量值"""
        os.environ[self.name] = str(value)


class Envs:
    """
    统一的环境变量定义

    命名规范: FD_<CATEGORY>_<NAME>

    Categories:
    - DEBUG: 调试相关
    - SCHEDULER: 调度相关
    - ATTENTION: 注意力相关
    - CACHE: 缓存相关
    - COMM: 通信相关
    """

    # ========== 调试相关 ==========
    DEBUG_LEVEL = EnvVar(
        name="FD_DEBUG_LEVEL",
        default=0,
        doc="调试级别 (0=关闭, 1=基础, 2=详细, 3=全部)",
        parser=int
    )

    DEBUG_LOG_DIR = EnvVar(
        name="FD_DEBUG_LOG_DIR",
        default="log",
        doc="日志输出目录"
    )

    # ========== 调度相关 ==========
    SCHEDULER_VERSION = EnvVar(
        name="FD_SCHEDULER_VERSION",
        default="v1",
        doc="调度器版本 (v0/v1)"
    )

    SCHEDULER_POLL_INTERVAL_MS = EnvVar(
        name="FD_SCHEDULER_POLL_INTERVAL_MS",
        default=1,
        doc="调度器轮询间隔（毫秒）",
        parser=int
    )

    # ========== 注意力相关 ==========
    ATTENTION_BACKEND = EnvVar(
        name="FD_ATTENTION_BACKEND",
        default="APPEND_ATTN",
        doc="注意力计算后端 (APPEND_ATTN/FLASH_ATTN/XFORMERS)"
    )

    # ========== 缓存相关 ==========
    CACHE_BLOCK_SIZE = EnvVar(
        name="FD_CACHE_BLOCK_SIZE",
        default=64,
        doc="KV Cache 块大小",
        parser=int
    )

    CACHE_ENABLE_PREFIX = EnvVar(
        name="FD_CACHE_ENABLE_PREFIX",
        default=True,
        doc="是否启用前缀缓存",
        parser=lambda x: x.lower() in ('true', '1', 'yes')
    )

    # ========== 通信相关 ==========
    COMM_TIMEOUT_MS = EnvVar(
        name="FD_COMM_TIMEOUT_MS",
        default=30000,
        doc="通信超时时间（毫秒）",
        parser=int
    )


# 使用方式
# from fastdeploy.envs import Envs
# debug_level = Envs.DEBUG_LEVEL.get()
# Envs.DEBUG_LEVEL.set(2)
```

### 4.3 旧环境变量兼容

```python
# fastdeploy/envs.py 中添加兼容映射

# 旧名称 -> 新名称的映射
_LEGACY_MAPPING = {
    "FD_DEBUG": "FD_DEBUG_LEVEL",
    "ENABLE_V1_KVCACHE_SCHEDULER": "FD_SCHEDULER_VERSION",  # 需要特殊处理
    "USE_FLASH_ATTN": "FD_ATTENTION_BACKEND",
}


def get_with_legacy_support(new_name: str) -> str:
    """支持旧环境变量名称"""
    # 先尝试新名称
    value = os.getenv(new_name)
    if value is not None:
        return value

    # 再尝试旧名称
    for old_name, mapped_new_name in _LEGACY_MAPPING.items():
        if mapped_new_name == new_name:
            value = os.getenv(old_name)
            if value is not None:
                import warnings
                warnings.warn(
                    f"Environment variable {old_name} is deprecated, "
                    f"use {new_name} instead",
                    DeprecationWarning
                )
                return value

    return None
```

---

## 五、实施步骤

### 阶段一：config.py 拆分（3-5天）

1. 创建 `fastdeploy/config/` 目录
2. 按模块拆分类（只移动代码，不修改逻辑）
3. 创建 `__init__.py` 统一导出
4. 保留原 `config.py` 作为兼容层
5. 运行测试验证

### 阶段二：类型修复（1-2天）

1. 修复所有 `xxx: Type = None` 为 `xxx: Optional[Type] = None`
2. 添加 mypy 配置和 CI 检查

### 阶段三：EngineArgs 重构（3-5天）

1. 创建子配置类（ModelArgs, CacheArgs 等）
2. 重构 EngineArgs 为组合结构
3. 添加 YAML 配置文件支持
4. 保持 CLI 兼容

### 阶段四：环境变量统一（2-3天）

1. 重构 `envs.py` 为类型安全版本
2. 添加旧名称兼容映射
3. 清理代码中直接使用 `os.getenv` 的地方

---

## 六、验证方法

### 6.1 单元测试

```python
# tests/config/test_config_import.py
def test_backward_compatibility():
    """验证向后兼容"""
    # 旧的导入方式仍然有效
    from fastdeploy.config import ModelConfig, CacheConfig, FDConfig

    assert ModelConfig is not None
    assert CacheConfig is not None
    assert FDConfig is not None


def test_new_import_style():
    """验证新的导入方式"""
    from fastdeploy.config.model import ModelConfig
    from fastdeploy.config.cache import CacheConfig

    assert ModelConfig is not None
    assert CacheConfig is not None
```

### 6.2 类型检查

```bash
# 运行 mypy 检查
mypy fastdeploy/config/ --strict
```

### 6.3 集成测试

```bash
# 运行现有测试套件
pytest tests/engine/ -v
pytest tests/config/ -v
```

---

## 七、风险与回滚

### 7.1 风险点

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 导入路径变化导致外部代码失败 | 中 | 高 | 保留兼容层 |
| 环境变量重命名导致配置失效 | 中 | 高 | 添加旧名称映射 |
| 类型修复引入运行时错误 | 低 | 中 | 充分测试 |

### 7.2 回滚方案

1. 保留原 `config.py` 文件，随时可以恢复
2. 使用 Feature Flag 控制新旧代码路径
3. 分阶段发布，逐步验证

---

*文档创建时间: 2026-02-14*
