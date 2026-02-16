# P/D 分离逻辑聚合详细设计

> **文档版本**: v1.0
> **创建日期**: 2026-02-14
> **关联问题**: P/D 分离逻辑散布在 54 个文件中

---

## 一、背景与目标

### 1.1 什么是 P/D 分离

**P/D 分离**（Prefill-Decode Disaggregation）是一种 LLM 推理优化架构：

- **Prefill（首次推理）**: 处理完整的输入 prompt，计算量大，适合高算力节点
- **Decode（增量推理）**: 逐 token 生成，计算量小，适合高带宽节点

将 Prefill 和 Decode 部署在不同的机器/GPU 上，可以提高整体吞吐量。

### 1.2 当前问题

P/D 分离相关代码散布在 **54 个文件**中：

```bash
# 统计涉及 splitwise/disaggregate 的文件
$ grep -rl "splitwise\|disaggregate" fastdeploy/ | wc -l
54
```

**主要分布**:

| 文件/目录 | 涉及内容 |
|----------|---------|
| `engine/common_engine.py` | SplitwiseConnector 集成、任务发送 |
| `engine/resource_manager.py` | disaggregate_info 处理 |
| `engine/resource_manager_v1.py` | P/D 资源预分配 |
| `scheduler/splitwise_scheduler.py` | P/D 专用调度器 |
| `splitwise/` | 连接器、传输协议 |
| `cache_manager/prefix_cache_manager.py` | KV Cache 跨节点同步 |
| `worker/gpu_model_runner.py` | Worker 中的 P/D 逻辑 |

### 1.3 重构目标

- 将 P/D 相关代码聚合到独立模块
- 定义清晰的抽象接口
- 降低与核心引擎的耦合
- 便于添加新的分离策略

---

## 二、架构设计

### 2.1 聚合后的目录结构

```
fastdeploy/disaggregation/
├── __init__.py                    # 模块导出
├── base.py                        # 抽象接口定义
├── config.py                      # P/D 配置
├── connector/                     # 节点间连接
│   ├── __init__.py
│   ├── base.py                    # 连接器接口
│   ├── tcp_connector.py           # TCP 连接器
│   └── rdma_connector.py          # RDMA 连接器
├── strategies/                    # 分离策略
│   ├── __init__.py
│   ├── base.py                    # 策略接口
│   ├── mixed.py                   # 混合模式（不分离）
│   ├── prefill_decode.py          # P/D 分离模式
│   └── speculative.py             # 投机解码分离
├── scheduler/                     # P/D 调度
│   ├── __init__.py
│   ├── splitwise_scheduler.py     # 分离调度器
│   └── coordinator.py             # P/D 协调器
├── resource/                      # 资源适配
│   ├── __init__.py
│   ├── prefill_adapter.py         # Prefill 节点资源
│   └── decode_adapter.py          # Decode 节点资源
└── cache_sync/                    # KV Cache 同步
    ├── __init__.py
    ├── base.py                    # 同步接口
    ├── kv_transfer.py             # KV 传输
    └── block_mapping.py           # 块映射
```

### 2.2 核心接口设计

```python
# fastdeploy/disaggregation/base.py
"""
P/D 分离核心接口定义
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

from fastdeploy.engine.request import Request


class DisaggregationRole(Enum):
    """节点角色"""
    MIXED = "mixed"      # 混合（不分离）
    PREFILL = "prefill"  # Prefill 节点
    DECODE = "decode"    # Decode 节点


@dataclass
class TransferTask:
    """KV Cache 传输任务"""
    request_id: str
    source_node: str
    target_node: str
    block_ids: List[int]
    token_count: int


class DisaggregationStrategy(ABC):
    """
    分离策略接口

    定义 P/D 分离的核心行为
    """

    @property
    @abstractmethod
    def role(self) -> DisaggregationRole:
        """当前节点角色"""
        pass

    @abstractmethod
    def should_transfer(self, request: Request) -> bool:
        """
        判断请求是否需要转移到其他节点

        Args:
            request: 推理请求

        Returns:
            True 如果需要转移
        """
        pass

    @abstractmethod
    def prepare_transfer(self, request: Request) -> Optional[TransferTask]:
        """
        准备传输任务

        Args:
            request: 推理请求

        Returns:
            传输任务，None 如果不需要传输
        """
        pass

    @abstractmethod
    def handle_incoming(self, request: Request) -> Request:
        """
        处理从其他节点接收的请求

        Args:
            request: 接收到的请求

        Returns:
            处理后的请求
        """
        pass
```

### 2.3 连接器接口

```python
# fastdeploy/disaggregation/connector/base.py
"""
节点间连接器接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator

from fastdeploy.disaggregation.base import TransferTask


class ConnectorBase(ABC):
    """
    节点间连接器接口

    负责 P/D 节点之间的通信
    """

    @abstractmethod
    async def connect(self, target_address: str) -> bool:
        """建立连接"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    async def send_request(self, request: Request) -> bool:
        """发送请求到目标节点"""
        pass

    @abstractmethod
    async def receive_requests(self) -> AsyncIterator[Request]:
        """接收来自其他节点的请求"""
        pass

    @abstractmethod
    async def transfer_kv_cache(self, task: TransferTask) -> bool:
        """传输 KV Cache"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass
```

### 2.4 KV Cache 同步接口

```python
# fastdeploy/disaggregation/cache_sync/base.py
"""
KV Cache 同步接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class CacheSyncBase(ABC):
    """
    KV Cache 同步接口

    负责跨节点的 KV Cache 传输和管理
    """

    @abstractmethod
    def prepare_for_transfer(
        self,
        request_id: str,
        block_ids: List[int],
    ) -> Dict:
        """
        准备 KV Cache 传输

        Returns:
            传输元数据（块映射、偏移等）
        """
        pass

    @abstractmethod
    async def send_kv_cache(
        self,
        request_id: str,
        block_ids: List[int],
        target_connector: "ConnectorBase",
    ) -> bool:
        """发送 KV Cache 到目标节点"""
        pass

    @abstractmethod
    async def receive_kv_cache(
        self,
        request_id: str,
        metadata: Dict,
        source_connector: "ConnectorBase",
    ) -> List[int]:
        """
        接收 KV Cache

        Returns:
            本地分配的块 ID 列表
        """
        pass

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        """清理请求相关的同步状态"""
        pass
```

---

## 三、详细设计

### 3.1 混合模式策略（不分离）

```python
# fastdeploy/disaggregation/strategies/mixed.py
"""
混合模式 - 不进行 P/D 分离
"""

from fastdeploy.disaggregation.base import (
    DisaggregationStrategy,
    DisaggregationRole,
    TransferTask,
)


class MixedStrategy(DisaggregationStrategy):
    """
    混合模式策略

    Prefill 和 Decode 在同一节点执行，不进行分离
    """

    @property
    def role(self) -> DisaggregationRole:
        return DisaggregationRole.MIXED

    def should_transfer(self, request) -> bool:
        # 混合模式不需要传输
        return False

    def prepare_transfer(self, request) -> None:
        # 混合模式不需要传输
        return None

    def handle_incoming(self, request):
        # 混合模式不接收外部请求
        return request
```

### 3.2 P/D 分离策略

```python
# fastdeploy/disaggregation/strategies/prefill_decode.py
"""
Prefill-Decode 分离策略
"""

from typing import Optional

from fastdeploy.disaggregation.base import (
    DisaggregationStrategy,
    DisaggregationRole,
    TransferTask,
)
from fastdeploy.disaggregation.connector import ConnectorBase
from fastdeploy.disaggregation.cache_sync import CacheSyncBase
from fastdeploy.engine.request import Request, RequestType


class PrefillDecodeStrategy(DisaggregationStrategy):
    """
    Prefill-Decode 分离策略

    - Prefill 节点：执行首次推理，完成后将请求转移到 Decode 节点
    - Decode 节点：接收 Prefill 完成的请求，执行增量推理
    """

    def __init__(
        self,
        role: DisaggregationRole,
        connector: ConnectorBase,
        cache_sync: CacheSyncBase,
    ):
        self._role = role
        self.connector = connector
        self.cache_sync = cache_sync

    @property
    def role(self) -> DisaggregationRole:
        return self._role

    def should_transfer(self, request: Request) -> bool:
        """
        判断是否需要转移

        Prefill 节点在完成 Prefill 后需要转移
        """
        if self._role == DisaggregationRole.PREFILL:
            # Prefill 完成后转移到 Decode 节点
            return request.task_type == RequestType.PREFILL and request.prefill_done
        return False

    def prepare_transfer(self, request: Request) -> Optional[TransferTask]:
        """准备传输任务"""
        if not self.should_transfer(request):
            return None

        return TransferTask(
            request_id=request.request_id,
            source_node=self._get_local_address(),
            target_node=self._get_decode_node_address(),
            block_ids=request.block_tables,
            token_count=request.num_computed_tokens,
        )

    def handle_incoming(self, request: Request) -> Request:
        """处理从 Prefill 节点接收的请求"""
        if self._role == DisaggregationRole.DECODE:
            # 标记为已完成 Prefill
            request.task_type = RequestType.DECODE
            request.prefill_done = True
        return request

    async def execute_transfer(self, request: Request) -> bool:
        """执行传输"""
        task = self.prepare_transfer(request)
        if task is None:
            return False

        # 1. 传输 KV Cache
        success = await self.cache_sync.send_kv_cache(
            request_id=task.request_id,
            block_ids=task.block_ids,
            target_connector=self.connector,
        )

        if not success:
            return False

        # 2. 发送请求元数据
        return await self.connector.send_request(request)


class PrefillNodeStrategy(PrefillDecodeStrategy):
    """Prefill 节点策略"""

    def __init__(self, connector, cache_sync):
        super().__init__(
            role=DisaggregationRole.PREFILL,
            connector=connector,
            cache_sync=cache_sync,
        )


class DecodeNodeStrategy(PrefillDecodeStrategy):
    """Decode 节点策略"""

    def __init__(self, connector, cache_sync):
        super().__init__(
            role=DisaggregationRole.DECODE,
            connector=connector,
            cache_sync=cache_sync,
        )
```

### 3.3 P/D 协调器

```python
# fastdeploy/disaggregation/scheduler/coordinator.py
"""
P/D 分离协调器

负责协调 Prefill 和 Decode 节点之间的工作
"""

import asyncio
from typing import List, Optional, Dict
from collections import defaultdict

from fastdeploy.disaggregation.base import (
    DisaggregationStrategy,
    DisaggregationRole,
    TransferTask,
)
from fastdeploy.disaggregation.connector import ConnectorBase
from fastdeploy.engine.request import Request


class DisaggregationCoordinator:
    """
    P/D 分离协调器

    职责：
    - 管理节点角色
    - 协调请求在节点间的流转
    - 监控传输状态
    """

    def __init__(
        self,
        strategy: DisaggregationStrategy,
        connector: ConnectorBase,
    ):
        self.strategy = strategy
        self.connector = connector

        # 传输状态跟踪
        self.pending_transfers: Dict[str, TransferTask] = {}
        self.completed_transfers: Dict[str, bool] = {}

        # 统计信息
        self.stats = defaultdict(int)

    @property
    def role(self) -> DisaggregationRole:
        """当前节点角色"""
        return self.strategy.role

    def is_prefill_node(self) -> bool:
        """是否为 Prefill 节点"""
        return self.role == DisaggregationRole.PREFILL

    def is_decode_node(self) -> bool:
        """是否为 Decode 节点"""
        return self.role == DisaggregationRole.DECODE

    def is_mixed_node(self) -> bool:
        """是否为混合节点"""
        return self.role == DisaggregationRole.MIXED

    async def process_request(self, request: Request) -> Optional[Request]:
        """
        处理请求

        根据节点角色和请求状态，决定：
        - 本地处理
        - 转移到其他节点
        - 等待接收传输
        """
        if self.is_mixed_node():
            # 混合模式：直接本地处理
            return request

        if self.is_prefill_node():
            return await self._process_on_prefill_node(request)
        else:
            return await self._process_on_decode_node(request)

    async def _process_on_prefill_node(self, request: Request) -> Optional[Request]:
        """Prefill 节点处理逻辑"""
        if self.strategy.should_transfer(request):
            # 需要转移到 Decode 节点
            success = await self._transfer_to_decode(request)
            if success:
                self.stats["transfers_sent"] += 1
                return None  # 已转移，本地不再处理
            else:
                self.stats["transfer_failures"] += 1
                # 转移失败，继续本地处理
                return request
        return request

    async def _process_on_decode_node(self, request: Request) -> Optional[Request]:
        """Decode 节点处理逻辑"""
        # 处理从 Prefill 节点接收的请求
        processed = self.strategy.handle_incoming(request)
        self.stats["transfers_received"] += 1
        return processed

    async def _transfer_to_decode(self, request: Request) -> bool:
        """转移请求到 Decode 节点"""
        task = self.strategy.prepare_transfer(request)
        if task is None:
            return False

        self.pending_transfers[request.request_id] = task

        try:
            # 执行传输
            if isinstance(self.strategy, PrefillDecodeStrategy):
                success = await self.strategy.execute_transfer(request)
            else:
                success = await self.connector.send_request(request)

            self.completed_transfers[request.request_id] = success
            return success
        finally:
            del self.pending_transfers[request.request_id]

    async def start_receiver(self):
        """启动接收协程（Decode 节点）"""
        if not self.is_decode_node():
            return

        async for request in self.connector.receive_requests():
            processed = await self.process_request(request)
            if processed:
                # 将接收到的请求放入本地调度队列
                await self._enqueue_local(processed)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return dict(self.stats)
```

### 3.4 TCP 连接器实现

```python
# fastdeploy/disaggregation/connector/tcp_connector.py
"""
TCP 连接器实现
"""

import asyncio
import pickle
from typing import AsyncIterator, Optional

from fastdeploy.disaggregation.connector.base import ConnectorBase
from fastdeploy.engine.request import Request


class TCPConnector(ConnectorBase):
    """
    TCP 连接器

    使用 TCP 进行节点间通信
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._server: Optional[asyncio.Server] = None
        self._connected = False

    async def connect(self, target_address: str) -> bool:
        """建立到目标节点的连接"""
        try:
            host, port = target_address.split(":")
            self._reader, self._writer = await asyncio.open_connection(
                host, int(port)
            )
            self._connected = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False

    async def send_request(self, request: Request) -> bool:
        """发送请求"""
        if not self._connected or not self._writer:
            return False

        try:
            data = pickle.dumps(request.to_dict())
            # 发送长度前缀
            self._writer.write(len(data).to_bytes(4, "big"))
            self._writer.write(data)
            await self._writer.drain()
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            return False

    async def receive_requests(self) -> AsyncIterator[Request]:
        """接收请求（作为服务端）"""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

        async with self._server:
            await self._server.serve_forever()

    async def _handle_connection(self, reader, writer):
        """处理连接"""
        while True:
            try:
                # 读取长度前缀
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, "big")

                # 读取数据
                data = await reader.readexactly(length)
                request_dict = pickle.loads(data)
                request = Request.from_dict(request_dict)

                yield request
            except asyncio.IncompleteReadError:
                break

    def is_connected(self) -> bool:
        return self._connected
```

---

## 四、与现有代码的集成

### 4.1 EngineService 集成

```python
# fastdeploy/engine/common_engine.py (修改后)

from fastdeploy.disaggregation import (
    DisaggregationCoordinator,
    create_strategy,
    create_connector,
)


class EngineService:
    def __init__(self, cfg):
        # ... 其他初始化

        # P/D 分离组件初始化
        self.disagg_coordinator = self._init_disaggregation(cfg)

    def _init_disaggregation(self, cfg) -> DisaggregationCoordinator:
        """初始化 P/D 分离组件"""
        role = cfg.scheduler_config.splitwise_role

        # 创建策略和连接器
        connector = create_connector(cfg.disaggregation_config)
        strategy = create_strategy(role, connector)

        return DisaggregationCoordinator(strategy, connector)

    async def _schedule_request(self, request):
        """调度请求（集成 P/D 分离）"""
        # 通过协调器处理
        processed = await self.disagg_coordinator.process_request(request)

        if processed is None:
            # 请求已转移到其他节点
            return

        # 本地处理
        await self._local_schedule(processed)
```

### 4.2 配置集成

```python
# fastdeploy/disaggregation/config.py
"""
P/D 分离配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DisaggregationConfig:
    """P/D 分离配置"""

    # 节点角色
    role: str = "mixed"  # mixed/prefill/decode

    # 连接配置
    connector_type: str = "tcp"  # tcp/rdma
    peer_address: Optional[str] = None  # 对端地址

    # 传输配置
    transfer_timeout_ms: int = 5000
    max_pending_transfers: int = 100

    # KV Cache 同步配置
    cache_sync_method: str = "direct"  # direct/staged
    cache_block_size: int = 64
```

---

## 五、迁移步骤

### 阶段一：创建模块结构（3天）

1. 创建 `fastdeploy/disaggregation/` 目录
2. 定义核心接口（base.py）
3. 实现混合模式策略

### 阶段二：迁移现有代码（1周）

1. 将 `splitwise/` 下的代码迁移到新模块
2. 将 `scheduler/splitwise_scheduler.py` 迁移
3. 抽取 `common_engine.py` 中的 P/D 逻辑

### 阶段三：实现新接口（1周）

1. 实现 P/D 分离策略
2. 实现连接器（TCP/RDMA）
3. 实现 KV Cache 同步

### 阶段四：集成测试（1周）

1. 单元测试各组件
2. 集成测试完整流程
3. 性能测试

### 阶段五：清理旧代码（3天）

1. 删除散布的 P/D 代码
2. 更新文档
3. 发布

---

## 六、测试策略

### 6.1 单元测试

```python
# tests/disaggregation/test_strategies.py

class TestMixedStrategy:
    def test_should_not_transfer(self):
        """混合模式不应该转移"""
        strategy = MixedStrategy()
        request = create_test_request()

        assert strategy.should_transfer(request) is False

    def test_role_is_mixed(self):
        """角色应该是混合"""
        strategy = MixedStrategy()
        assert strategy.role == DisaggregationRole.MIXED


class TestPrefillDecodeStrategy:
    def test_prefill_node_should_transfer_after_prefill(self):
        """Prefill 节点应该在完成后转移"""
        strategy = PrefillNodeStrategy(mock_connector, mock_cache_sync)
        request = create_test_request()
        request.task_type = RequestType.PREFILL
        request.prefill_done = True

        assert strategy.should_transfer(request) is True

    def test_decode_node_should_not_transfer(self):
        """Decode 节点不应该转移"""
        strategy = DecodeNodeStrategy(mock_connector, mock_cache_sync)
        request = create_test_request()

        assert strategy.should_transfer(request) is False
```

### 6.2 集成测试

```python
# tests/disaggregation/test_e2e.py

class TestE2EDisaggregation:
    async def test_prefill_to_decode_transfer(self):
        """测试从 Prefill 到 Decode 的完整传输"""
        # 启动 Prefill 节点
        prefill_engine = create_prefill_engine()

        # 启动 Decode 节点
        decode_engine = create_decode_engine()

        # 发送请求到 Prefill 节点
        request = create_test_request()
        await prefill_engine.add_request(request)

        # 等待处理完成
        result = await wait_for_result(request.request_id)

        # 验证结果
        assert result.finished
        assert len(result.outputs) > 0
```

---

## 七、风险与回滚

### 7.1 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 接口不兼容 | 中 | 高 | 完善测试覆盖 |
| 性能回退 | 低 | 高 | 性能基准测试 |
| 网络传输异常 | 中 | 中 | 重试机制、超时处理 |

### 7.2 回滚方案

保留原有代码路径，通过配置切换：

```python
if cfg.use_new_disaggregation:
    from fastdeploy.disaggregation import DisaggregationCoordinator
    self.disagg = DisaggregationCoordinator(...)
else:
    # 使用原有逻辑
    self.split_connector = SplitwiseConnector(...)
```

---

## 八、收益预估

| 指标 | 重构前 | 重构后 | 提升 |
|------|-------|-------|------|
| P/D 代码分布 | 54 个文件 | 1 个模块 | **集中管理** |
| 添加新策略 | 需要修改多处 | 只需实现接口 | **显著简化** |
| 代码可读性 | P/D 逻辑与核心混杂 | 清晰分离 | **大幅提升** |
| 测试覆盖 | 难以单独测试 | 可独立测试 | **易于验证** |

---

*文档创建时间: 2026-02-14*
