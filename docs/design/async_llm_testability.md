# AsyncLLM Testability Design

## Overview

This document describes the testability improvement design for `AsyncLLM` class, focusing on interface definitions, dependency injection, and clear responsibility boundaries.

**Current Location**: `fastdeploy/engine/async_llm.py`

---

## 1. Current Problems

### 1.1 Direct External Dependency Creation

| Dependency | Location | Problem |
|------------|----------|---------|
| **ZMQ Client** | Line 314 | `ZmqIpcClient(...)` created directly |
| **ZMQ Connection Manager** | Line 318 | `DealerConnectionManager(...)` created directly |
| **IPC Signal** | Line 194 | `IPCSignal(...)` created directly |
| **multiprocessing.Process** | Line 127-162 | Process created directly |

### 1.2 Mixed Responsibilities

Current `AsyncLLM` handles too many concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        AsyncLLM                             │
│  Current responsibilities (too many):                       │
│  1. Engine subprocess lifecycle (EngineServiceClient)       │
│  2. Input preprocessing (InputPreprocessor)                 │
│  3. Request sending (ZmqIpcClient)                          │
│  4. Response receiving (DealerConnectionManager)            │
│  5. Output post-processing (AsyncOutputProcessor)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Interface Definitions

### 2.1 External Interfaces (for users)

```python
class AsyncLLM:
    """
    Async LLM client for inference.

    External Interfaces:
    - generate(prompt, params) -> AsyncGenerator[RequestOutput]
    - add_request(request_id, prompt, params) -> None
    - abort_request(request_id) -> None
    - shutdown() -> None
    """

    async def generate(
        self,
        prompt: Union[str, List[str], Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Async generation interface.

        Args:
            prompt: Input prompt (string, token ids, or dict)
            sampling_params: Sampling parameters
            request_id: Optional request identifier
            **kwargs: Additional parameters (enable_thinking, etc.)

        Yields:
            RequestOutput: Generated output chunks

        Raises:
            EngineNotReadyError: Engine not initialized
            RequestError: Invalid request parameters
        """
        ...

    async def add_request(
        self,
        request_id: str,
        prompt: Union[str, List[str], Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        arrival_time: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Add a request to the engine.

        Args:
            request_id: Unique request identifier
            prompt: Input prompt
            sampling_params: Sampling parameters
            arrival_time: Request arrival timestamp

        Raises:
            RequestError: Invalid prompt or parameters (400)
            ConnectionError: Failed to send to engine (500)
        """
        ...

    async def abort_request(self, request_id: str) -> None:
        """
        Abort a pending request.

        Args:
            request_id: Request to abort

        Note:
            Best effort - does not raise on failure
        """
        ...

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the engine.

        Note:
            Best effort cleanup - does not raise
        """
        ...
```

### 2.2 Internal Interfaces (injected dependencies)

```python
from typing import Protocol, Any, Tuple
import asyncio

class IRequestSender(Protocol):
    """
    Interface for sending requests to engine.

    Implementations:
    - ZmqIpcClient (production)
    - MockRequestSender (test)
    """

    def send_json(self, data: dict) -> None:
        """Send JSON-serializable request."""
        ...

    def send_pyobj(self, data: Any) -> None:
        """Send Python object (for multimodal data)."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class IResponseReceiver(Protocol):
    """
    Interface for receiving responses from engine.

    Implementations:
    - DealerConnectionManager (production)
    - MockResponseReceiver (test)
    """

    async def initialize(self) -> None:
        """Initialize connections."""
        ...

    async def get_connection(
        self,
        request_id: str,
        num_choices: int
    ) -> Tuple[Any, asyncio.Queue]:
        """
        Get connection for receiving responses.

        Args:
            request_id: Request identifier
            num_choices: Number of parallel completions

        Returns:
            Tuple of (dealer, response_queue)
        """
        ...

    async def cleanup_request(self, request_id: str) -> None:
        """Cleanup resources for a request."""
        ...

    async def close(self) -> None:
        """Close all connections."""
        ...


class IEngineProcessManager(Protocol):
    """
    Interface for managing engine subprocess.

    Implementations:
    - DefaultEngineProcessManager (production)
    - MockEngineProcessManager (test)
    """

    def start(self, cfg: Any, engine_pid: int) -> None:
        """Start the engine process."""
        ...

    def is_ready(self, timeout: float = 500.0) -> bool:
        """
        Check if engine is ready.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if engine is ready
        """
        ...

    def shutdown(self) -> None:
        """Shutdown the engine process."""
        ...
```

---

## 3. Refactored Design

### 3.1 Responsibility Boundaries

```
┌──────────────────────────────────────────────────────────────┐
│                        AsyncLLM                              │
│  Responsibility: Coordinate request/response flow            │
├──────────────────────────────────────────────────────────────┤
│  External Interfaces (user-facing):                          │
│  ├─ generate(prompt, params) -> AsyncGenerator[Output]       │
│  ├─ add_request(request_id, prompt, params) -> None          │
│  ├─ abort_request(request_id) -> None                        │
│  └─ shutdown() -> None                                       │
├──────────────────────────────────────────────────────────────┤
│  Injected Dependencies (mockable):                           │
│  ├─ IRequestSender      ←── ZmqIpcClient / Mock              │
│  ├─ IResponseReceiver   ←── DealerConnectionManager / Mock   │
│  └─ IEngineProcessManager ←── DefaultManager / Mock          │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Constructor with Dependency Injection

```python
class AsyncLLM:
    """
    Async LLM client with dependency injection for testability.
    """

    def __init__(
        self,
        cfg,
        pid: int,
        # Injected dependencies (default to None, created lazily if not provided)
        request_sender: Optional[IRequestSender] = None,
        response_receiver: Optional[IResponseReceiver] = None,
        engine_manager: Optional[IEngineProcessManager] = None,
    ):
        """
        Initialize AsyncLLM client.

        Args:
            cfg: Engine configuration
            pid: Process identifier for IPC
            request_sender: Optional injected request sender
            response_receiver: Optional injected response receiver
            engine_manager: Optional injected engine process manager
        """
        self.cfg = cfg
        self.engine_pid = pid

        # Injected or lazy-created dependencies
        self._request_sender = request_sender
        self._response_receiver = response_receiver
        self._engine_manager = engine_manager or DefaultEngineProcessManager()

        # Internal state (not exposed)
        self._prompt_metadata: Dict[str, Dict[str, Any]] = {}
        self._running = False

        # Input/output processing (internal)
        self._input_processor = InputPreprocessor(...)
        self._output_processor = AsyncOutputProcessor(...)

    async def init_connections(self) -> None:
        """
        Initialize ZMQ connections.

        Creates real implementations if not injected.
        """
        if self._request_sender is None:
            self._request_sender = ZmqIpcClient(name=self.engine_pid, mode=zmq.PUSH)
            self._request_sender.connect()

        if self._response_receiver is None:
            self._response_receiver = DealerConnectionManager(
                pid=self.engine_pid,
                max_connections=int(os.getenv("FD_DEALER_CONNECTIONS", 50))
            )
            await self._response_receiver.initialize()
```

---

## 4. Error Handling

### 4.1 Exception Hierarchy

```python
class AsyncLLMError(Exception):
    """Base exception for AsyncLLM operations."""
    error_code: int = 500

class EngineNotReadyError(AsyncLLMError):
    """Engine not initialized or connection failed."""
    error_code = 503

class RequestError(AsyncLLMError):
    """Invalid request parameters or prompt."""
    error_code = 400

class ConnectionError(AsyncLLMError):
    """Failed to communicate with engine."""
    error_code = 500
```

### 4.2 Error Contract by Method

| Method | Success Output | Possible Errors |
|--------|---------------|-----------------|
| `generate()` | `AsyncGenerator[RequestOutput]` | `EngineNotReadyError`, `RequestError` |
| `add_request()` | `None` | `RequestError` (400), `ConnectionError` (500) |
| `init_connections()` | `None` | `ConnectionError` |
| `abort_request()` | `None` | None (best effort) |
| `shutdown()` | `None` | None (best effort) |

---

## 5. Test Implementation

### 5.1 Mock Implementations

```python
class MockRequestSender:
    """Test double for IRequestSender."""

    def __init__(self):
        self.sent_requests: List[dict] = []
        self.closed = False

    def send_json(self, data: dict) -> None:
        self.sent_requests.append(("json", data))

    def send_pyobj(self, data: Any) -> None:
        self.sent_requests.append(("pyobj", data))

    def close(self) -> None:
        self.closed = True

    # Test helpers
    def get_last_request(self) -> dict:
        """Get the most recently sent request."""
        return self.sent_requests[-1][1] if self.sent_requests else None

    def clear(self) -> None:
        """Clear sent requests history."""
        self.sent_requests.clear()


class MockResponseReceiver:
    """Test double for IResponseReceiver."""

    def __init__(self):
        self._responses: Dict[str, asyncio.Queue] = {}
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def get_connection(
        self,
        request_id: str,
        num_choices: int
    ) -> Tuple[Any, asyncio.Queue]:
        if request_id not in self._responses:
            self._responses[request_id] = asyncio.Queue()
        return (MockDealer(), self._responses[request_id])

    async def cleanup_request(self, request_id: str) -> None:
        self._responses.pop(request_id, None)

    async def close(self) -> None:
        self._responses.clear()

    # Test helpers
    def push_response(self, request_id: str, response: dict) -> None:
        """Inject a response for testing."""
        if request_id not in self._responses:
            self._responses[request_id] = asyncio.Queue()
        self._responses[request_id].put_nowait([response])

    def push_responses(self, request_id: str, responses: List[dict]) -> None:
        """Inject multiple responses (streaming)."""
        for resp in responses:
            self.push_response(request_id, resp)


class MockEngineProcessManager:
    """Test double for IEngineProcessManager."""

    def __init__(self, ready: bool = True):
        self._ready = ready
        self._started = False
        self._shutdown = False

    def start(self, cfg: Any, engine_pid: int) -> None:
        self._started = True

    def is_ready(self, timeout: float = 500.0) -> bool:
        return self._ready

    def shutdown(self) -> None:
        self._shutdown = True

    # Test helpers
    def set_ready(self, ready: bool) -> None:
        """Control engine readiness for testing."""
        self._ready = ready


class MockDealer:
    """Mock ZMQ dealer for testing."""

    def __init__(self):
        self.written: List[List[bytes]] = []

    def write(self, data: List[bytes]) -> None:
        self.written.append(data)
```

### 5.2 Test Cases

```python
import pytest
from unittest.mock import MagicMock

def create_test_config():
    """Create minimal config for testing."""
    cfg = MagicMock()
    cfg.model_config.max_model_len = 4096
    cfg.model_config.enable_mm = False
    return cfg


@pytest.mark.asyncio
async def test_add_request_sends_preprocessed_request():
    """
    Test: add_request should preprocess prompt and send via IRequestSender.

    Verifies:
    - Request is sent to sender
    - Request contains preprocessed token ids
    - Request has correct metadata
    """
    # Arrange
    mock_sender = MockRequestSender()
    mock_receiver = MockResponseReceiver()
    mock_engine = MockEngineProcessManager(ready=True)

    llm = AsyncLLM(
        cfg=create_test_config(),
        pid=12345,
        request_sender=mock_sender,
        response_receiver=mock_receiver,
        engine_manager=mock_engine,
    )
    await llm.init_connections()

    # Act
    await llm.add_request(
        request_id="test-001",
        prompt="Hello, world!",
        sampling_params=SamplingParams(max_tokens=10),
    )

    # Assert
    assert len(mock_sender.sent_requests) == 1
    sent = mock_sender.get_last_request()
    assert sent["request_id"] == "test-001"
    assert "prompt_token_ids" in sent
    assert sent["max_tokens"] == 10


@pytest.mark.asyncio
async def test_generate_yields_processed_outputs():
    """
    Test: generate should yield processed outputs from IResponseReceiver.

    Verifies:
    - Responses are received from receiver
    - Outputs are properly post-processed
    - Generator completes when finished=True
    """
    # Arrange
    mock_sender = MockRequestSender()
    mock_receiver = MockResponseReceiver()

    llm = AsyncLLM(
        cfg=create_test_config(),
        pid=12345,
        request_sender=mock_sender,
        response_receiver=mock_receiver,
    )
    await llm.init_connections()

    # Inject mock response
    mock_receiver.push_response("req-001", {
        "request_id": "req-001",
        "outputs": {"text": "Hello", "token_ids": [1, 2], "send_idx": 0},
        "finished": True,
    })

    # Act
    outputs = []
    async for output in llm.generate(prompt="Hi", request_id="req-001"):
        outputs.append(output)

    # Assert
    assert len(outputs) == 1
    assert outputs[0].outputs.text == "Hello"
    assert outputs[0].finished is True


@pytest.mark.asyncio
async def test_generate_streams_multiple_chunks():
    """
    Test: generate should yield multiple chunks for streaming response.
    """
    # Arrange
    mock_sender = MockRequestSender()
    mock_receiver = MockResponseReceiver()

    llm = AsyncLLM(
        cfg=create_test_config(),
        pid=12345,
        request_sender=mock_sender,
        response_receiver=mock_receiver,
    )
    await llm.init_connections()

    # Inject streaming responses
    mock_receiver.push_response("req-002", {
        "request_id": "req-002",
        "outputs": {"text": "Hello", "send_idx": 0},
        "finished": False,
    })
    mock_receiver.push_response("req-002", {
        "request_id": "req-002",
        "outputs": {"text": " World", "send_idx": 1},
        "finished": True,
    })

    # Act
    outputs = []
    async for output in llm.generate(prompt="Hi", request_id="req-002"):
        outputs.append(output)

    # Assert
    assert len(outputs) == 2
    assert outputs[0].finished is False
    assert outputs[1].finished is True


@pytest.mark.asyncio
async def test_generate_raises_when_not_initialized():
    """
    Test: generate should raise EngineNotReadyError when not initialized.
    """
    # Arrange
    llm = AsyncLLM(cfg=create_test_config(), pid=12345)
    # Note: init_connections() not called

    # Act & Assert
    with pytest.raises(EngineNotReadyError):
        async for _ in llm.generate(prompt="Hi"):
            pass


@pytest.mark.asyncio
async def test_abort_request_cleans_up_resources():
    """
    Test: abort_request should cleanup via IResponseReceiver.
    """
    # Arrange
    mock_receiver = MockResponseReceiver()
    mock_receiver.push_response("req-003", {"request_id": "req-003"})

    llm = AsyncLLM(
        cfg=create_test_config(),
        pid=12345,
        response_receiver=mock_receiver,
    )

    # Act
    await llm.abort_request("req-003")

    # Assert
    assert "req-003" not in mock_receiver._responses


@pytest.mark.asyncio
async def test_shutdown_closes_all_connections():
    """
    Test: shutdown should close sender and receiver.
    """
    # Arrange
    mock_sender = MockRequestSender()
    mock_receiver = MockResponseReceiver()
    mock_engine = MockEngineProcessManager()

    llm = AsyncLLM(
        cfg=create_test_config(),
        pid=12345,
        request_sender=mock_sender,
        response_receiver=mock_receiver,
        engine_manager=mock_engine,
    )
    await llm.init_connections()

    # Act
    await llm.shutdown()

    # Assert
    assert mock_sender.closed is True
    assert mock_engine._shutdown is True
```

---

## 6. Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Responsibility** | 5 mixed concerns | Focused on "coordinate request/response flow" |
| **Dependencies** | Created internally (ZMQ/IPC) | Injected via interfaces |
| **Testability** | Requires real processes | Pure in-memory mocks |
| **Error Handling** | Generic EngineError | Typed exception hierarchy |

### Key Design Decisions

1. **Protocol-based interfaces** - No explicit inheritance required
2. **Optional injection** - Default `None` parameters, production code unchanged
3. **Test helpers on mocks** - `push_response()`, `get_last_request()` for easy test setup
4. **Best-effort cleanup** - `abort_request()` and `shutdown()` don't raise exceptions

### Implementation Priority

1. Define `IRequestSender`, `IResponseReceiver`, `IEngineProcessManager` protocols
2. Modify `AsyncLLM.__init__()` to accept optional dependencies
3. Create mock implementations
4. Write unit tests
5. Validate with integration tests
