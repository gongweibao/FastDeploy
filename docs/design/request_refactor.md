# Request 模块重构详细设计

> **文档版本**: v1.0
> **创建日期**: 2026-02-14
> **关联问题**: request.py 职责过多（1418行，50+属性）

---

## 一、背景与目标

### 1.1 当前问题

`fastdeploy/engine/request.py` 文件包含 19 个类，1418 行代码。核心的 `Request` 类有 50+ 属性，混合了：

- 请求定义（request_id, prompt, sampling_params）
- 运行时状态（status, task_type, output_token_ids）
- 资源跟踪（block_tables, extend_block_tables）
- 多模态数据（multimodal_inputs, image_start/end）
- 性能指标（metrics）

### 1.2 重构目标

- 按职责拆分为独立模块
- 提高代码可读性和可维护性
- 保持向后兼容

---

## 二、拆分方案

本方案采用**方案一：按文件拆分**，只移动代码，不修改内部结构。

### 2.1 拆分后的目录结构

```
fastdeploy/engine/request/
├── __init__.py              # 统一导出，保持向后兼容
├── base.py                  # 枚举和基础类 (~30行)
├── request.py               # Request 主类 (~480行)
├── control.py               # 控制请求/响应 (~160行)
├── output.py                # 输出相关类 (~320行)
├── metrics.py               # 指标类 (~160行)
└── pooling.py               # Pooling/Embedding/Reward 输出 (~240行)
```

### 2.2 类到文件的映射

| 原类名 | 行号范围 | 目标文件 | 说明 |
|-------|---------|---------|------|
| `RequestStatus` | 51-57 | `base.py` | 请求状态枚举 |
| `RequestType` | 59-65 | `base.py` | 请求类型枚举 |
| `ImagePosition` | 67-74 | `base.py` | 图像位置数据类 |
| `Request` | 76-546 | `request.py` | **核心请求类，470行** |
| `ControlRequest` | 548-633 | `control.py` | 控制请求 |
| `ControlResponse` | 635-698 | `control.py` | 控制响应 |
| `CompletionOutput` | 700-793 | `output.py` | 补全输出 |
| `RequestMetrics` | 795-949 | `metrics.py` | **请求指标，155行** |
| `RequestOutput` | 951-1187 | `output.py` | **请求输出，237行** |
| `PoolingOutput` | 1189-1210 | `pooling.py` | 池化输出基类 |
| `PoolingRequestOutput` | 1212-1263 | `pooling.py` | 池化请求输出 |
| `EmbeddingOutput` | 1265-1292 | `pooling.py` | Embedding 输出 |
| `EmbeddingRequestOutput` | 1294-1304 | `pooling.py` | Embedding 请求输出 |
| `ClassificationOutput` | 1306-1331 | `pooling.py` | 分类输出 |
| `ClassificationRequestOutput` | 1333-1343 | `pooling.py` | 分类请求输出 |
| `ScoringOutput` | 1345-1367 | `pooling.py` | 评分输出 |
| `ScoringRequestOutput` | 1369-1379 | `pooling.py` | 评分请求输出 |
| `RewardOutput` | 1381-1408 | `pooling.py` | 奖励输出 |
| `RewardRequestOutput` | 1410-1418 | `pooling.py` | 奖励请求输出 |

---

## 三、详细设计

### 3.1 base.py - 基础类

```python
# fastdeploy/engine/request/base.py
"""
请求模块基础类和枚举定义
"""

from enum import Enum
from dataclasses import dataclass
from typing import List


class RequestStatus(Enum):
    """请求状态"""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"
    ABORTED = "aborted"


class RequestType(Enum):
    """请求类型"""
    UNKNOWN = "unknown"
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


@dataclass
class ImagePosition:
    """图像在输入中的位置"""
    start: int
    end: int
    image_index: int
```

### 3.2 request.py - 核心请求类

```python
# fastdeploy/engine/request/request.py
"""
核心 Request 类定义
"""

from typing import Any, Dict, List, Optional, TypeVar
from dataclasses import dataclass

from fastdeploy.engine.request.base import RequestStatus, RequestType, ImagePosition
from fastdeploy.engine.request.metrics import RequestMetrics
from fastdeploy.engine.sampling_params import SamplingParams

T = TypeVar("T", bound="Request")


class Request:
    """
    推理请求类

    包含请求的所有信息：输入、参数、状态、资源分配等
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        # ... 其他参数保持不变
    ):
        # 代码保持不变，只是移动到新文件
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "Request":
        """从字典创建请求"""
        ...

    def to_dict(self) -> dict:
        """转换为字典"""
        ...

    # ... 其他方法保持不变
```

### 3.3 control.py - 控制请求/响应

```python
# fastdeploy/engine/request/control.py
"""
控制请求和响应类

用于引擎控制操作，如暂停、恢复、更新权重等
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from starlette.responses import JSONResponse


class ControlRequest:
    """控制请求"""

    def __init__(
        self,
        request_id: str,
        method: str,
        args: Optional[Dict[str, Any]] = None,
    ):
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "ControlRequest":
        ...

    def to_dict(self) -> dict:
        ...

    @staticmethod
    def is_control_request(d: dict) -> bool:
        """判断是否为控制请求"""
        ...


class ControlResponse:
    """控制响应"""

    def __init__(
        self,
        request_id: str,
        success: bool,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        ...

    def to_api_json_response(self) -> JSONResponse:
        ...
```

### 3.4 output.py - 输出类

```python
# fastdeploy/engine/request/output.py
"""
请求输出相关类
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastdeploy.engine.request.metrics import RequestMetrics


@dataclass
class CompletionOutput:
    """单个补全输出"""
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[str] = None

    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, req_dict: dict) -> "CompletionOutput":
        ...


class RequestOutput:
    """
    请求输出

    包含请求的输出结果、状态和指标
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        outputs: Optional[List[CompletionOutput]] = None,
        finished: bool = False,
        metrics: Optional[RequestMetrics] = None,
        error_code: int = 0,
        error_msg: Optional[str] = None,
    ):
        ...

    def add(self, next_output: "RequestOutput") -> None:
        """添加增量输出"""
        ...

    def accumulate(self, next_output: "RequestOutput") -> None:
        """累积输出"""
        ...

    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "RequestOutput":
        ...
```

### 3.5 metrics.py - 指标类

```python
# fastdeploy/engine/request/metrics.py
"""
请求性能指标
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class RequestMetrics:
    """
    请求性能指标

    记录请求处理过程中的各种时间戳和统计信息
    """
    # 时间戳
    arrival_time: float = 0.0
    first_scheduled_time: float = 0.0
    first_token_time: float = 0.0
    finished_time: float = 0.0

    # Token 相关
    prompt_tokens: int = 0
    generated_tokens: int = 0

    # 延迟统计
    time_to_first_token: float = 0.0
    inter_token_latencies: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.arrival_time == 0.0:
            self.arrival_time = time.time()

    def record_recv_first_token(self):
        """记录收到首个 token 的时间"""
        self.first_token_time = time.time()
        self.time_to_first_token = self.first_token_time - self.arrival_time

    def record_recv_token(self, cur_time: float = None):
        """记录收到 token 的时间"""
        ...

    def cal_cost_time(self) -> Dict[str, float]:
        """计算各阶段耗时"""
        ...

    def to_dict(self) -> dict:
        ...

    @classmethod
    def from_dict(cls, req_dict: dict) -> "RequestMetrics":
        ...
```

### 3.6 pooling.py - Pooling 相关输出

```python
# fastdeploy/engine/request/pooling.py
"""
Pooling、Embedding、Reward 等特殊输出类
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

_O = TypeVar("_O")


@dataclass
class PoolingOutput:
    """池化输出基类"""
    data: List[float]


class PoolingRequestOutput(Generic[_O]):
    """池化请求输出"""

    def __init__(
        self,
        request_id: str,
        outputs: Optional[_O] = None,
        prompt_token_ids: Optional[List[int]] = None,
        finished: bool = False,
    ):
        ...


@dataclass
class EmbeddingOutput(PoolingOutput):
    """Embedding 输出"""
    embedding: List[float] = None


class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    """Embedding 请求输出"""
    pass


@dataclass
class ClassificationOutput(PoolingOutput):
    """分类输出"""
    label: str = ""
    score: float = 0.0


class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    """分类请求输出"""
    pass


@dataclass
class ScoringOutput(PoolingOutput):
    """评分输出"""
    score: float = 0.0


class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    """评分请求输出"""
    pass


@dataclass
class RewardOutput(PoolingOutput):
    """奖励输出"""
    reward: float = 0.0


class RewardRequestOutput(PoolingRequestOutput[RewardOutput]):
    """奖励请求输出"""
    pass
```

### 3.7 `__init__.py` - 统一导出

```python
# fastdeploy/engine/request/__init__.py
"""
请求模块 - 保持向后兼容的导出

使用方式不变：
    from fastdeploy.engine.request import Request, RequestOutput
"""

from fastdeploy.engine.request.base import (
    RequestStatus,
    RequestType,
    ImagePosition,
)
from fastdeploy.engine.request.request import Request
from fastdeploy.engine.request.control import ControlRequest, ControlResponse
from fastdeploy.engine.request.output import CompletionOutput, RequestOutput
from fastdeploy.engine.request.metrics import RequestMetrics
from fastdeploy.engine.request.pooling import (
    PoolingOutput,
    PoolingRequestOutput,
    EmbeddingOutput,
    EmbeddingRequestOutput,
    ClassificationOutput,
    ClassificationRequestOutput,
    ScoringOutput,
    ScoringRequestOutput,
    RewardOutput,
    RewardRequestOutput,
)

__all__ = [
    # base
    "RequestStatus",
    "RequestType",
    "ImagePosition",
    # request
    "Request",
    # control
    "ControlRequest",
    "ControlResponse",
    # output
    "CompletionOutput",
    "RequestOutput",
    # metrics
    "RequestMetrics",
    # pooling
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "RewardOutput",
    "RewardRequestOutput",
]
```

---

## 四、后续优化方向（方案二预览）

拆分文件后，可以进一步考虑重构 `Request` 类的内部结构：

```python
# 未来方向：将 Request 的 50+ 属性拆分为组合结构

@dataclass(frozen=True)
class RequestInput:
    """请求输入 - 不可变"""
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    multimodal_inputs: Optional[Dict] = None


@dataclass
class RequestState:
    """运行时状态 - 可变"""
    status: RequestStatus = RequestStatus.WAITING
    task_type: RequestType = RequestType.UNKNOWN
    output_token_ids: List[int] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class RequestResources:
    """资源分配信息"""
    idx: int = -1
    block_tables: List[int] = field(default_factory=list)
    extend_block_tables: List[int] = field(default_factory=list)
    num_cached_tokens: int = 0


class Request:
    """组合类"""
    def __init__(self, input: RequestInput):
        self.input = input
        self.state = RequestState()
        self.resources = RequestResources()
        self.metrics = RequestMetrics()
```

**注意**：方案二改动较大，需要修改所有使用 `Request` 属性的地方，建议在方案一完成后再考虑。

---

## 五、实施步骤

### 步骤 1：创建目录结构（10分钟）

```bash
mkdir -p fastdeploy/engine/request
touch fastdeploy/engine/request/__init__.py
touch fastdeploy/engine/request/base.py
touch fastdeploy/engine/request/request.py
touch fastdeploy/engine/request/control.py
touch fastdeploy/engine/request/output.py
touch fastdeploy/engine/request/metrics.py
touch fastdeploy/engine/request/pooling.py
```

### 步骤 2：拆分代码（2-3小时）

1. 将 `RequestStatus`, `RequestType`, `ImagePosition` 移动到 `base.py`
2. 将 `Request` 类移动到 `request.py`
3. 将 `ControlRequest`, `ControlResponse` 移动到 `control.py`
4. 将 `CompletionOutput`, `RequestOutput` 移动到 `output.py`
5. 将 `RequestMetrics` 移动到 `metrics.py`
6. 将所有 Pooling 相关类移动到 `pooling.py`

### 步骤 3：处理 import（1-2小时）

1. 在每个新文件中添加必要的 import
2. 处理类之间的依赖关系
3. 创建 `__init__.py` 统一导出

### 步骤 4：保留兼容层（10分钟）

```python
# fastdeploy/engine/request.py (兼容层)
"""
此文件保留用于向后兼容
"""
from fastdeploy.engine.request import *  # noqa: F401, F403
```

### 步骤 5：验证测试（1小时）

```bash
# 运行现有测试
pytest tests/engine/test_request.py -v
pytest tests/engine/test_request_output.py -v
pytest tests/engine/test_control_request_response.py -v
```

---

## 六、验证清单

- [ ] 所有原有测试通过
- [ ] `from fastdeploy.engine.request import Request` 仍然有效
- [ ] `from fastdeploy.engine.request import RequestOutput` 仍然有效
- [ ] mypy 类型检查通过
- [ ] 没有循环 import

---

## 七、风险与回滚

### 7.1 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 循环 import | 中 | 高 | 仔细处理依赖顺序 |
| 遗漏类导出 | 低 | 中 | 完整的 `__all__` 列表 |
| 外部代码 import 失败 | 低 | 高 | 保留兼容层 |

### 7.2 回滚方案

保留原 `request.py` 文件作为备份，随时可以恢复：

```bash
# 回滚命令
rm -rf fastdeploy/engine/request/
git checkout fastdeploy/engine/request.py
```

---

*文档创建时间: 2026-02-14*
