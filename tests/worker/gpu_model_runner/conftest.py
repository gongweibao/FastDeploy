# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures and helper functions for GPUModelRunner tests.

This module consolidates common mock creation patterns found across
multiple test files to reduce duplication and ensure consistency.
"""

import numpy as np
import paddle
from dataclasses import dataclass
from typing import Optional, List, Any, Dict
from unittest.mock import Mock

import pytest

from fastdeploy.worker.gpu_model_runner import GPUModelRunner


@dataclass
class TestRequest:
    """Test request dataclass for multimodal inputs."""
    multimodal_inputs: dict = None


def create_mock_fd_config(
    max_model_len: int = 4096,
    max_num_seqs: int = 10,
    enable_mm: bool = False,
    use_ep: bool = False,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    splitwise_role: str = "mixed",
    speculative_method: str = None,
) -> Mock:
    """
    Helper to create a mock FD configuration.

    Args:
        max_model_len: Maximum model sequence length
        max_num_seqs: Maximum number of sequences in batch
        enable_mm: Enable multimodal support
        use_ep: Enable expert parallel
        enable_prefix_caching: Enable prefix caching
        enable_chunked_prefill: Enable chunked prefill
        splitwise_role: Scheduler splitwise role (mixed/encoder/decoder)
        speculative_method: Speculative decoding method (ngram/mtp/None)

    Returns:
        Mock object configured as fd_config
    """
    mock_fd_config = Mock()

    # Model config
    mock_model_config = Mock()
    mock_model_config.max_model_len = max_model_len
    mock_model_config.eos_tokens_lens = 1
    mock_model_config.max_stop_seqs_num = 4
    mock_model_config.enable_mm = enable_mm
    mock_model_config.bad_tokens_len = 1
    mock_model_config.max_prompt_embedding_table_size = 0
    mock_model_config.max_encoder_len = 512 if enable_mm else 0
    mock_model_config.dtype = "float16"
    mock_model_config.num_hidden_layers = 24
    mock_model_config.hidden_size = 4096
    mock_model_config.vocab_size = 32000
    mock_model_config.ori_vocab_size = 32000
    mock_model_config.kv_lora_rank = 0
    mock_model_config.qk_rope_head_dim = 64
    mock_model_config.num_attention_heads = 32
    mock_model_config.num_kv_heads = 4
    mock_model_config.head_dim = 128
    mock_model_config.kv_num_heads = 32
    mock_model_config.logprobs_mode = "raw_logprobs"
    mock_model_config.lora_request_ids_to_shard_id = None
    mock_fd_config.model_config = mock_model_config

    # Cache config
    mock_cache_config = Mock()
    mock_cache_config.block_size = 16
    mock_cache_config.enc_dec_block_num = 0
    mock_cache_config.total_block_num = 10000
    mock_cache_config.num_gpu_blocks = 1000
    mock_cache_config.num_cpu_blocks = 0
    mock_cache_config.kvcache_storage_backend = None
    mock_cache_config.kv_cache_dtype = "float16"
    mock_cache_config.enable_prefix_caching = enable_prefix_caching
    mock_cache_config.enable_chunked_prefill = enable_chunked_prefill
    mock_cache_config.max_chunked_prefill_len = 4096
    mock_cache_config.max_encoder_cache = 100 if enable_mm else 0
    mock_cache_config.use_mla_cache = False
    mock_cache_config.cache_k = None
    mock_cache_config.cache_v = None
    mock_cache_config.cache_cpu_block_num = 0
    mock_cache_config.sliding_window = -1
    mock_fd_config.cache_config = mock_cache_config

    # Scheduler config
    mock_scheduler_config = Mock()
    mock_scheduler_config.splitwise_role = splitwise_role
    mock_scheduler_config.max_num_seqs = max_num_seqs
    mock_scheduler_config.enable_overlap_schedule = False
    mock_fd_config.scheduler_config = mock_scheduler_config

    # Parallel config
    mock_parallel_config = Mock()
    mock_parallel_config.use_ep = use_ep
    mock_parallel_config.tensor_parallel_size = 1
    mock_parallel_config.pipeline_parallel_size = 1
    mock_parallel_config.enable_expert_parallel = False
    mock_parallel_config.enable_chunked_moe = False
    mock_fd_config.parallel_config = mock_parallel_config

    # Speculative config
    mock_speculative_config = Mock()
    mock_speculative_config.method = speculative_method
    mock_speculative_config.num_speculative_tokens = 4
    mock_speculative_config.num_gpu_block_expand_ratio = 0
    mock_fd_config.speculative_config = mock_speculative_config

    # Routing replay config
    mock_routing_replay_config = Mock()
    mock_routing_replay_config.enable_routing_replay = False
    mock_fd_config.routing_replay_config = mock_routing_replay_config

    # Other configs
    mock_fd_config.quant_config = None
    mock_fd_config.graph_opt_config = Mock()
    mock_fd_config.graph_opt_config.use_cudagraph = False
    mock_fd_config.graph_opt_config.cudagraph_capture_sizes = []
    mock_fd_config.graph_opt_config.cudagraph_capture_sizes_prefill = []
    mock_fd_config.graph_opt_config.sot_warmup_sizes = []
    mock_fd_config.graph_opt_config.cudagraph_only_prefill = False
    mock_fd_config.early_stop_config = Mock()
    mock_fd_config.early_stop_config.enable_early_stop = False

    return mock_fd_config


def create_mock_request(
    task_type: Any = 0,
    idx: int = 0,
    token_ids: Optional[List[int]] = None,
    output_ids: Optional[List[int]] = None,
    with_image: bool = False,
    block_tables: Optional[List[int]] = None,
    **kwargs
) -> Mock:
    """
    Helper to create a mock request for GPUModelRunner tests.

    Args:
        task_type: Task type (RequestType enum or int value)
        idx: Request index
        token_ids: Prompt token IDs (defaults to [1,2,3,4,5])
        output_ids: Output token IDs (defaults to [])
        with_image: Whether this request includes image data
        block_tables: KV Cache block tables
        **kwargs: Additional request attributes

    Returns:
        Mock object configured as a request
    """
    request = Mock()
    request.task_type = Mock()
    request.task_type.value = task_type.value if hasattr(task_type, "value") else task_type
    request.idx = idx
    request.request_id = kwargs.get("request_id", f"req_{idx}")
    request.prompt_token_ids = token_ids if token_ids is not None else [1, 2, 3, 4, 5]
    request.output_token_ids = output_ids if output_ids is not None else []
    request.block_tables = block_tables if block_tables is not None else [0, 1, 2]
    request.eos_token_ids = [0]
    request.prefill_start_index = kwargs.get("prefill_start_index", 0)
    request.prefill_end_index = kwargs.get(
        "prefill_end_index",
        len(request.prompt_token_ids) if request.prompt_token_ids else 0,
    )

    # Sampling params
    request.sampling_params = Mock()
    request.sampling_params.prompt_logprobs = kwargs.get("prompt_logprobs", None)
    request.sampling_params.stop_seqs_len = kwargs.get("stop_seqs_len", [])
    request.sampling_params.min_tokens = kwargs.get("min_tokens", 1)
    request.sampling_params.max_tokens = kwargs.get("max_tokens", 100)
    request.sampling_params.temperature = kwargs.get("temperature", 1.0)
    request.sampling_params.top_p = kwargs.get("top_p", 1.0)
    request.sampling_params.top_k = kwargs.get("top_k", 0)
    request.sampling_params.min_p = kwargs.get("min_p", 0.0)
    request.sampling_params.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
    request.sampling_params.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
    request.sampling_params.presence_penalty = kwargs.get("presence_penalty", 0.0)
    request.sampling_params.bad_tokens = []
    request.sampling_params.bad_tokens_len = 0

    # Guided decoding attributes (set to None to avoid Mock truthy)
    request.guided_json = kwargs.get("guided_json", None)
    request.guided_regex = kwargs.get("guided_regex", None)
    request.guided_grammar = kwargs.get("guided_grammar", None)
    request.structural_tag = kwargs.get("structural_tag", None)
    request.disaggregate_info = kwargs.get("disaggregate_info", None)

    # Thinking-related attributes
    request.enable_thinking = kwargs.get("enable_thinking", None)
    request.reasoning_max_tokens = kwargs.get("reasoning_max_tokens", None)
    request.pooling_params = kwargs.get("pooling_params", None)

    # Store attributes for .get() method
    request.__dict__["_top_p"] = kwargs.get("_top_p", None)
    request.__dict__["_top_k"] = kwargs.get("_top_k", None)
    request.__dict__["_temperature"] = kwargs.get("_temperature", None)
    request.__dict__["_min_p"] = kwargs.get("_min_p", None)

    # Multimodal setup if needed
    if with_image:
        request.multimodal_inputs = {
            "position_ids": kwargs.get("position_ids", np.array([[1, 2, 3]])),
            "images": kwargs.get("images", []),
            "grid_thw": kwargs.get("grid_thw", []),
            "mm_positions": kwargs.get("mm_positions", []),
            "mm_hashes": kwargs.get("mm_hashes", []),
            "vit_seqlen": kwargs.get("vit_seqlen", []),
            "vit_position_ids": kwargs.get("vit_position_ids", []),
            "mm_num_token_func": lambda **kw: 123,
            "num_image_start": kwargs.get("num_image_start", 0),
            "num_image_end": kwargs.get("num_image_end", 0),
            "image_start": kwargs.get("image_start", 0),
            "image_end": kwargs.get("image_end", 0),
        }
        request.with_image = True
    else:
        request.multimodal_inputs = kwargs.get("multimodal_inputs", None)
        request.with_image = False

    # get() method support
    def mock_get(key: str, default=None):
        prefixed_key = f"_{key}"
        if prefixed_key in request.__dict__:
            value = request.__dict__[prefixed_key]
            return value if value is not None else default
        return kwargs.get(key, default)

    request.get = mock_get

    return request


def create_share_inputs_mock(
    batch_size: int = 10, extra_fields: Optional[Dict[str, Any]] = None
) -> Mock:
    """
    Helper to create a share_inputs mock with all required fields.

    Args:
        batch_size: Number of sequences in batch
        extra_fields: Additional fields to add to the mock

    Returns:
        Mock object configured as share_inputs
    """
    share_inputs = Mock()
    share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)

    base_data = {
        "req_ids": [""] * batch_size,
        "preempted_idx": np.zeros((batch_size, 1), dtype="int32"),
        "stop_flags": np.zeros((batch_size,), dtype=bool),
        "seq_lens_decoder": np.zeros((batch_size,), dtype="int32"),
        "seq_lens_encoder": np.zeros((batch_size,), dtype="int32"),
        "seq_lens_this_time_buffer": np.zeros((batch_size,), dtype="int32"),
        "seq_lens_this_time": np.zeros((batch_size,), dtype="int32"),
        "prompt_ids": np.zeros((batch_size, 512), dtype="int64"),
        "input_ids": np.zeros((batch_size, 512), dtype="int64"),
        "encoder_block_lens": np.zeros((batch_size,), dtype="int32"),
        "block_tables": np.full((batch_size, 128), -1, dtype="int32"),
        "step_seq_lens_decoder": np.zeros((batch_size,), dtype="int32"),
        "prompt_lens": np.zeros((batch_size,), dtype="int32"),
        "is_block_step": np.zeros((batch_size,), dtype=bool),
        "is_chunk_step": np.zeros((batch_size,), dtype=bool),
        "step_idx": np.zeros((batch_size,), dtype="int32"),
        "pre_ids": np.full((batch_size, 1), -1, dtype="int64"),
        "eos_token_id": np.zeros((1, 1), dtype="int64"),
        "top_p": np.zeros((batch_size,), dtype="float32"),
        "top_k": np.zeros((batch_size,), dtype="int32"),
        "top_k_list": np.zeros((batch_size,), dtype="int32"),
        "min_p": np.zeros((batch_size,), dtype="float32"),
        "min_p_list": np.zeros((batch_size,), dtype="float32"),
        "temperature": np.zeros((batch_size,), dtype="float32"),
        "penalty_score": np.ones((batch_size,), dtype="float32"),
        "frequency_score": np.zeros((batch_size,), dtype="float32"),
        "presence_score": np.zeros((batch_size,), dtype="float32"),
        "temp_scaled_logprobs": np.zeros((batch_size,), dtype=bool),
        "top_p_normalized_logprobs": np.zeros((batch_size,), dtype=bool),
        "min_dec_len": np.zeros((batch_size,), dtype="int32"),
        "max_dec_len": np.zeros((batch_size,), dtype="int32"),
        "first_token_ids": np.zeros((batch_size, 1), dtype="int64"),
        "infer_seed": np.zeros((batch_size,), dtype="int64"),
        "bad_tokens_len": np.ones((batch_size,), dtype="int32"),
        "bad_tokens": np.full((batch_size, 1), -1, dtype="int64"),
        "stop_seqs_len": np.zeros((batch_size, 4), dtype="int32"),
        "stop_seqs": np.zeros((batch_size, 4, 10), dtype="int64"),
        "not_need_stop": paddle.zeros((1,), dtype="int32"),
        "logits_processors_args": [{}] * batch_size,
        "enable_thinking": np.zeros((batch_size, 1), dtype="int32"),
        "max_think_lens": np.full((batch_size, 1), -1, dtype="int32"),
        "limit_think_status": np.zeros((batch_size, 1), dtype="int32"),
        "num_running_requests": 0,
        "running_requests_ids": [],
    }

    if extra_fields:
        base_data.update(extra_fields)

    share_inputs.__getitem__ = Mock(side_effect=lambda key: base_data[key])
    share_inputs.__setitem__ = Mock(side_effect=lambda k, v: base_data.__setitem__(k, v))
    share_inputs.__contains__ = Mock(side_effect=lambda k: k in base_data)
    share_inputs.update = Mock(side_effect=lambda d: base_data.update(d))

    return share_inputs


def create_gpu_runner_mock(
    mock_fd_config: Optional[Mock] = None,
    share_inputs: Optional[Mock] = None,
) -> GPUModelRunner:
    """
    Helper to create a GPUModelRunner mock with common configurations.

    Args:
        mock_fd_config: Pre-configured mock fd_config (will create default if None)
        share_inputs: Pre-configured share_inputs (will create default if None)

    Returns:
        GPUModelRunner instance (not fully initialized, just __new__)
    """
    if mock_fd_config is None:
        mock_fd_config = create_mock_fd_config()

    if share_inputs is None:
        share_inputs = create_share_inputs_mock(mock_fd_config.scheduler_config.max_num_seqs)

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.fd_config = mock_fd_config
    runner.model_config = mock_fd_config.model_config
    runner.cache_config = mock_fd_config.cache_config
    runner.scheduler_config = mock_fd_config.scheduler_config
    runner.parallel_config = mock_fd_config.parallel_config
    runner.speculative_config = mock_fd_config.speculative_config
    runner.speculative_method = mock_fd_config.speculative_config.method
    runner.routing_replay_config = mock_fd_config.routing_replay_config
    runner.routing_replay_manager = None
    runner.routing_replay_config = mock_fd_config.routing_replay_config
    runner.quant_config = mock_fd_config.quant_config
    runner.share_inputs = share_inputs
    runner.forward_batch_reqs_list = [None] * mock_fd_config.scheduler_config.max_num_seqs
    runner.pooling_params = []
    runner.prompt_logprobs_reqs = {}
    runner.in_progress_prompt_logprobs = {}
    runner.exist_prefill_flag = False
    runner.enable_mm = mock_fd_config.model_config.enable_mm
    runner.use_cudagraph = mock_fd_config.graph_opt_config.use_cudagraph
    runner.cudagraph_only_prefill = mock_fd_config.graph_opt_config.cudagraph_only_prefill
    runner.speculative_decoding = mock_fd_config.speculative_config.method is not None
    runner.enable_overlap_schedule = mock_fd_config.scheduler_config.enable_overlap_schedule
    runner.is_pooling_model = False
    runner.device_id = 0
    runner.local_rank = 0
    runner.num_gpu_blocks = mock_fd_config.cache_config.num_gpu_blocks
    runner.cache_kvs_map = {}
    runner.restore_chunked_prefill_request = {}
    runner.attn_backends = []
    runner.sampler = Mock()
    runner.sampler.apply_logits_processor = Mock()
    runner.forward_meta = None
    runner.ori_vocab_size = mock_fd_config.model_config.ori_vocab_size
    runner.guided_backend = None
    runner.proposer = None

    return runner


# ============== Pytest Fixtures ==============


@pytest.fixture
def mock_fd_config():
    """Standard mock FD configuration."""
    return create_mock_fd_config()


@pytest.fixture
def base_runner(mock_fd_config):
    """Base GPUModelRunner mock with common setup."""
    return create_gpu_runner_mock(mock_fd_config=mock_fd_config)


@pytest.fixture
def sample_request():
    """A sample request for testing."""
    return create_mock_request()


@pytest.fixture
def sample_share_inputs():
    """A sample share_inputs mock for testing."""
    return create_share_inputs_mock()


@pytest.fixture
def sample_share_inputs_with_images():
    """A sample share_inputs mock with image-related fields."""
    return create_share_inputs_mock(extra_fields={
        "image_features_list": [],
        "image_grid_thw_list": [],
    })
