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

"""
Behavior-oriented unit tests for GPUModelRunner.

Testing principles:
  1. Test observable behavior (return values, state changes), NOT implementation details.
  2. NEVER mock any function defined inside gpu_model_runner.py.
  3. Only mock genuine external systems: GPU ops (set_stop, get_stop), distributed
     communication (paddle.distributed), IPC (ZmqIpcClient, IPCSignal).
  4. NEVER use assert_called / patch on internal functions.
  5. Tests must survive internal refactoring as long as the contract is preserved.

Mock justification is documented inline for every mock used.
"""

from unittest.mock import Mock, patch

import numpy as np
import paddle
import pytest

from fastdeploy.engine.request import ImagePosition, RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner

# ============================================================================
# Helper functions
# ============================================================================


def create_runner_minimal():
    """Create a GPUModelRunner instance bypassing __init__.

    WHY bypass __init__: The constructor connects to real external systems
    (GPU tensors via InputBatch, IPC sockets via ZmqIpcClient/IPCSignal,
    daemon threads). Bypassing is equivalent to dependency injection.
    """
    return GPUModelRunner.__new__(GPUModelRunner)


def create_runner_for_config(
    max_model_len=4096,
    block_size=16,
    enc_dec_block_num=0,
    enable_expert_parallel=False,
    head_dim=128,
    kv_num_heads=32,
    num_hidden_layers=24,
    use_mla_cache=False,
    kv_lora_rank=0,
    qk_rope_head_dim=64,
    kv_cache_quant_type=None,
    speculative_method=None,
    num_gpu_block_expand_ratio=0,
):
    """Create a runner with config attributes set for pure-computation tests."""
    runner = create_runner_minimal()

    runner.model_config = Mock()
    runner.model_config.max_model_len = max_model_len
    runner.model_config.head_dim = head_dim
    runner.model_config.kv_num_heads = kv_num_heads
    runner.model_config.num_hidden_layers = num_hidden_layers
    runner.model_config.kv_lora_rank = kv_lora_rank
    runner.model_config.qk_rope_head_dim = qk_rope_head_dim

    runner.cache_config = Mock()
    runner.cache_config.block_size = block_size
    runner.cache_config.enc_dec_block_num = enc_dec_block_num
    runner.cache_config.use_mla_cache = use_mla_cache

    runner.fd_config = Mock()
    runner.fd_config.parallel_config = Mock()
    runner.fd_config.parallel_config.enable_expert_parallel = enable_expert_parallel
    runner.fd_config.cache_config = runner.cache_config
    runner.fd_config.model_config = runner.model_config

    if kv_cache_quant_type is not None:
        runner.quant_config = Mock()
        runner.quant_config.kv_cache_quant_type = kv_cache_quant_type
    else:
        runner.quant_config = None

    runner.speculative_method = speculative_method
    runner.speculative_config = Mock()
    runner.speculative_config.num_gpu_block_expand_ratio = num_gpu_block_expand_ratio

    return runner


def create_share_inputs_dict(batch_size=10, max_seq_len=512):
    """Create a real dict backed by numpy arrays for share_inputs.

    This allows tests to inspect actual array values after method calls,
    rather than asserting on mock interactions.
    """
    data = {
        "req_ids": [""] * batch_size,
        "preempted_idx": np.zeros((batch_size, 1), dtype="int32"),
        "stop_flags": np.zeros((batch_size,), dtype=bool),
        "seq_lens_decoder": np.zeros((batch_size,), dtype="int32"),
        "seq_lens_encoder": np.zeros((batch_size,), dtype="int32"),
        "seq_lens_this_time_buffer": np.zeros((batch_size,), dtype="int32"),
        "seq_lens_this_time": np.zeros((batch_size,), dtype="int32"),
        "prompt_ids": np.zeros((batch_size, max_seq_len), dtype="int64"),
        "input_ids": np.zeros((batch_size, max_seq_len), dtype="int64"),
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
        "caches": True,  # prevent initialize_kv_cache call in insert_tasks_v1
        "image_features_list": [-1] * batch_size,
    }

    class DictLikeShareInputs:
        """A simple dict wrapper that supports __getitem__, __setitem__, __contains__."""

        def __init__(self, d):
            self._data = d

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def __contains__(self, key):
            return key in self._data

        def update(self, d):
            self._data.update(d)

        def pop(self, key, *args):
            return self._data.pop(key, *args)

        def get_index_by_batch_id(self, idx):
            return idx

    return DictLikeShareInputs(data)


def create_mock_request(
    task_type=RequestType.PREFILL,
    idx=0,
    token_ids=None,
    output_ids=None,
    block_tables=None,
    request_id=None,
    **kwargs,
):
    """Create a request-like object for testing insert_tasks_v1."""
    request = Mock()
    request.task_type = Mock()
    request.task_type.value = task_type.value
    request.idx = idx
    request.request_id = request_id or f"req_{idx}"

    request.prompt_token_ids = token_ids if token_ids is not None else [1, 2, 3, 4, 5]
    request.output_token_ids = output_ids if output_ids is not None else []
    request.block_tables = block_tables if block_tables is not None else [0, 1, 2]
    request.eos_token_ids = kwargs.get("eos_token_ids", [2])

    request.prefill_start_index = kwargs.get("prefill_start_index", 0)
    request.prefill_end_index = kwargs.get(
        "prefill_end_index",
        len(request.prompt_token_ids) + len(request.output_token_ids),
    )

    # Sampling params
    request.sampling_params = Mock()
    request.sampling_params.prompt_logprobs = kwargs.get("prompt_logprobs", None)
    request.sampling_params.stop_seqs_len = kwargs.get("stop_seqs_len", [])
    request.sampling_params.min_tokens = kwargs.get("min_tokens", 1)

    # Guided decoding (set to None to not trigger guided decoding path)
    request.guided_json = kwargs.get("guided_json", None)
    request.guided_regex = kwargs.get("guided_regex", None)
    request.guided_grammar = kwargs.get("guided_grammar", None)
    request.structural_tag = kwargs.get("structural_tag", None)

    # Thinking attributes
    request.enable_thinking = kwargs.get("enable_thinking", None)
    request.reasoning_max_tokens = kwargs.get("reasoning_max_tokens", None)

    # Disaggregate info
    request.disaggregate_info = kwargs.get("disaggregate_info", None)

    # Multimodal
    request.multimodal_inputs = kwargs.get("multimodal_inputs", None)
    request.with_image = False

    # Pooling
    request.pooling_params = kwargs.get("pooling_params", None)

    # get() method for attribute access pattern used by insert_tasks_v1
    _extra = dict(kwargs)

    def mock_get(key, default=None):
        if key == "top_p":
            return kwargs.get("top_p", default)
        elif key == "top_k":
            return kwargs.get("top_k", default)
        elif key == "temperature":
            return kwargs.get("temperature", default)
        elif key == "min_p":
            return kwargs.get("min_p", default)
        elif key == "repetition_penalty":
            return kwargs.get("repetition_penalty", default)
        elif key == "frequency_penalty":
            return kwargs.get("frequency_penalty", default)
        elif key == "presence_penalty":
            return kwargs.get("presence_penalty", default)
        elif key == "temp_scaled_logprobs":
            return kwargs.get("temp_scaled_logprobs", default)
        elif key == "top_p_normalized_logprobs":
            return kwargs.get("top_p_normalized_logprobs", default)
        elif key == "min_tokens":
            return kwargs.get("min_tokens", default)
        elif key == "max_tokens":
            return kwargs.get("max_tokens", default)
        elif key == "seed":
            return kwargs.get("seed", default)
        elif key == "bad_words_token_ids":
            return kwargs.get("bad_words_token_ids", default)
        elif key == "stop_token_ids":
            return kwargs.get("stop_token_ids", default)
        elif key == "stop_seqs_len":
            return kwargs.get("stop_seqs_len", default)
        elif key == "logits_processors_args":
            return kwargs.get("logits_processors_args", default)
        elif key == "enable_thinking":
            return kwargs.get("enable_thinking", default)
        elif key == "reasoning_max_tokens":
            return kwargs.get("reasoning_max_tokens", default)
        return _extra.get(key, default)

    request.get = mock_get

    return request


# ============================================================================
# 1. _get_feature_positions — Pure logic, zero mocks
# ============================================================================


class TestGetFeaturePositions:
    """Tests for _get_feature_positions: filtering and adjusting ImagePosition
    objects within a prefill range.

    No mocks needed — this is a pure function that only uses copy.deepcopy
    and arithmetic. It reads no instance state.
    """

    @pytest.fixture
    def runner(self):
        return create_runner_minimal()

    def test_does_not_mutate_original(self, runner):
        """Original ImagePosition objects must not be modified."""
        original = ImagePosition(offset=10, length=5)
        positions = [original]
        runner._get_feature_positions(positions, prefill_start_index=0, prefill_end_index=30)
        assert original.offset == 10, "Original offset was mutated"
        assert original.length == 5, "Original length was mutated"

    def test_single_token_position_inside(self, runner):
        """A position with length=1 fully inside the range."""
        result = runner._get_feature_positions(
            [ImagePosition(offset=5, length=1)],
            prefill_start_index=0,
            prefill_end_index=10,
        )
        assert len(result) == 1
        assert result[0].offset == 0
        assert result[0].length == 1

    def test_single_token_position_at_boundary_start(self, runner):
        """A single-token position at the exact start of range."""
        result = runner._get_feature_positions(
            [ImagePosition(offset=10, length=1)],
            prefill_start_index=10,
            prefill_end_index=20,
        )
        assert len(result) == 1
        assert result[0].offset == 0
        assert result[0].length == 1

    def test_single_token_position_at_boundary_end_excluded(self, runner):
        """A single-token position at the exact end of range (exclusive)."""
        result = runner._get_feature_positions(
            [ImagePosition(offset=20, length=1)],
            prefill_start_index=10,
            prefill_end_index=20,
        )
        assert len(result) == 0

    def test_many_positions(self, runner):
        """Many positions, all inside range. Should all be returned."""
        positions = [ImagePosition(offset=i, length=1) for i in range(100)]
        result = runner._get_feature_positions(positions, prefill_start_index=0, prefill_end_index=100)
        assert len(result) == 100

    def test_position_exactly_at_range_start_excluded(self, runner):
        """Position ends exactly at range start — should be excluded."""
        result = runner._get_feature_positions(
            [ImagePosition(offset=5, length=5)],  # [5, 10)
            prefill_start_index=10,
            prefill_end_index=20,
        )
        assert len(result) == 0

    def test_position_starts_exactly_at_range_end(self, runner):
        """Position starts exactly at range end — should be excluded."""
        result = runner._get_feature_positions(
            [ImagePosition(offset=20, length=5)],  # [20, 25)
            prefill_start_index=10,
            prefill_end_index=20,
        )
        assert len(result) == 0

    def test_position_spans_entire_range(self, runner):
        """A position that starts before and ends after the range."""
        result = runner._get_feature_positions(
            [ImagePosition(offset=5, length=20)],  # [5, 25)
            prefill_start_index=10,
            prefill_end_index=20,
        )
        assert len(result) == 1
        # offset = prefill_start - position_start = 10 - 5 = 5
        # length = min(25, 20) - 10 = 10
        assert result[0].offset == 5
        assert result[0].length == 10

    def test_empty_range(self, runner):
        """Zero-width range should return no positions."""
        result = runner._get_feature_positions(
            [ImagePosition(offset=10, length=5)],
            prefill_start_index=15,
            prefill_end_index=15,
        )
        assert len(result) == 0

    def test_empty_positions_list(self, runner):
        """Empty input list returns empty output."""
        result = runner._get_feature_positions([], prefill_start_index=0, prefill_end_index=100)
        assert len(result) == 0


# ============================================================================
# 2. get_input_length_list — Near-pure logic, zero mocks
# ============================================================================


class TestGetInputLengthList:
    """Tests for get_input_length_list: generates dummy input distributions
    for CUDA graph capture.

    No external system mocks needed — only config attributes are set.
    """

    def test_batch_size_zero_returns_empty(self):
        runner = create_runner_for_config()
        result = runner.get_input_length_list(num_tokens=100, batch_size=0, expected_decode_len=1)
        assert result == ([], [], 0)

    def test_normal_decode_evenly_distributed(self):
        runner = create_runner_for_config(max_model_len=4096, block_size=16, enc_dec_block_num=0)
        lengths, dec_lens, block_num = runner.get_input_length_list(
            num_tokens=160, batch_size=2, expected_decode_len=100
        )
        # input_length = min(160//2, 4096-101) = min(80, 3995) = 80
        assert lengths == [80, 80]
        assert dec_lens == [101, 101]
        # block_num = ceil(80/16) + 0 = 5
        assert block_num == 5

    def test_capture_prefill_concentrates_tokens(self):
        runner = create_runner_for_config(max_model_len=4096, block_size=16)
        lengths, dec_lens, block_num = runner.get_input_length_list(
            num_tokens=160, batch_size=4, expected_decode_len=1, capture_prefill=True
        )
        # capture_prefill: [1, 1, 1, 160-3=157]
        assert lengths == [1, 1, 1, 157]
        assert len(dec_lens) == 4
        assert all(d == 2 for d in dec_lens)  # max(1+1, 1) = 2

    def test_capture_prefill_few_tokens(self):
        """When num_tokens < batch_size, output list is shorter than batch_size."""
        runner = create_runner_for_config(max_model_len=4096, block_size=16)
        lengths, dec_lens, block_num = runner.get_input_length_list(
            num_tokens=2, batch_size=4, expected_decode_len=1, capture_prefill=True
        )
        assert lengths == [1, 1]
        assert len(dec_lens) == 2

    def test_max_model_len_constraint(self):
        """input_length is capped by max_model_len - max_dec_len."""
        runner = create_runner_for_config(max_model_len=100, block_size=16)
        lengths, dec_lens, block_num = runner.get_input_length_list(
            num_tokens=1000, batch_size=2, expected_decode_len=50
        )
        # input_length = min(1000//2, 100-51) = min(500, 49) = 49
        assert lengths == [49, 49]
        assert dec_lens == [51, 51]

    def test_expert_parallel_caps_at_32(self):
        runner = create_runner_for_config(max_model_len=4096, block_size=16, enable_expert_parallel=True)
        lengths, _, _ = runner.get_input_length_list(num_tokens=1000, batch_size=2, expected_decode_len=1)
        assert all(l <= 32 for l in lengths)

    def test_enc_dec_block_num_added_to_block_num(self):
        runner = create_runner_for_config(max_model_len=4096, block_size=16, enc_dec_block_num=3)
        _, _, block_num = runner.get_input_length_list(num_tokens=17, batch_size=1, expected_decode_len=1)
        # input_length = 17, block_num = ceil(17/16) + 3 = 2 + 3 = 5
        assert block_num == 5

    def test_negative_decode_len_floor_at_1(self):
        runner = create_runner_for_config()
        _, dec_lens, _ = runner.get_input_length_list(num_tokens=100, batch_size=2, expected_decode_len=-5)
        # max_dec_len = max(-5+1, 1) = 1
        assert all(d == 1 for d in dec_lens)

    def test_single_batch(self):
        runner = create_runner_for_config(max_model_len=4096, block_size=16)
        lengths, dec_lens, _ = runner.get_input_length_list(num_tokens=64, batch_size=1, expected_decode_len=1)
        assert lengths == [64]
        assert len(dec_lens) == 1

    def test_large_num_tokens_capped(self):
        """num_tokens much larger than max_model_len is properly constrained."""
        runner = create_runner_for_config(max_model_len=200, block_size=16)
        lengths, _, _ = runner.get_input_length_list(num_tokens=100000, batch_size=1, expected_decode_len=1)
        # input_length = min(100000, 200-2) = 198
        assert lengths == [198]


# ============================================================================
# 3. cal_theortical_kvcache — Pure arithmetic, zero mocks
# ============================================================================


class TestCalTheorticalKvcache:
    """Tests for cal_theortical_kvcache: calculates bytes per KV cache block.

    No external system mocks — only config attributes.
    """

    def test_default_bf16(self):
        runner = create_runner_for_config(head_dim=128, kv_num_heads=32, num_hidden_layers=24, block_size=16)
        result = runner.cal_theortical_kvcache()
        hidden_dim = 128 * 32  # 4096
        expected = 2 * 2 * 16 * hidden_dim * 24  # byte_of_dtype=2, k+v=2
        assert result == expected

    def test_int8_quant(self):
        runner = create_runner_for_config(
            head_dim=128,
            kv_num_heads=32,
            num_hidden_layers=24,
            block_size=16,
            kv_cache_quant_type="int8",
        )
        result = runner.cal_theortical_kvcache()
        hidden_dim = 128 * 32
        expected = 1 * 2 * 16 * hidden_dim * 24  # byte_of_dtype=1
        assert result == expected

    def test_fp8_quant(self):
        runner = create_runner_for_config(
            head_dim=128,
            kv_num_heads=32,
            num_hidden_layers=24,
            block_size=16,
            kv_cache_quant_type="fp8",
        )
        result = runner.cal_theortical_kvcache()
        hidden_dim = 128 * 32
        expected = 1 * 2 * 16 * hidden_dim * 24
        assert result == expected

    def test_mla_cache(self):
        runner = create_runner_for_config(
            head_dim=128,
            kv_num_heads=32,
            num_hidden_layers=24,
            block_size=16,
            use_mla_cache=True,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        result = runner.cal_theortical_kvcache()
        expected = 2 * (512 + 64) * 16 * 24  # compress_kv + k_pe
        assert result == expected

    def test_mla_with_int8(self):
        runner = create_runner_for_config(
            head_dim=128,
            kv_num_heads=32,
            num_hidden_layers=24,
            block_size=16,
            use_mla_cache=True,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            kv_cache_quant_type="int8",
        )
        result = runner.cal_theortical_kvcache()
        expected = 1 * (512 + 64) * 16 * 24
        assert result == expected

    def test_mtp_speculative_adds_layers(self):
        runner = create_runner_for_config(
            head_dim=128,
            kv_num_heads=32,
            num_hidden_layers=24,
            block_size=16,
            speculative_method="mtp",
            num_gpu_block_expand_ratio=1,
        )
        result = runner.cal_theortical_kvcache()
        hidden_dim = 128 * 32
        num_layers = 24 + 1  # mtp adds expand_ratio
        expected = 2 * 2 * 16 * hidden_dim * num_layers
        assert result == expected

    def test_ngram_speculative_no_extra_layers(self):
        runner = create_runner_for_config(
            head_dim=128,
            kv_num_heads=32,
            num_hidden_layers=24,
            block_size=16,
            speculative_method="ngram",
            num_gpu_block_expand_ratio=1,
        )
        result = runner.cal_theortical_kvcache()
        hidden_dim = 128 * 32
        expected = 2 * 2 * 16 * hidden_dim * 24  # ngram does NOT add layers
        assert result == expected

    def test_single_layer(self):
        runner = create_runner_for_config(head_dim=64, kv_num_heads=8, num_hidden_layers=1, block_size=8)
        result = runner.cal_theortical_kvcache()
        hidden_dim = 64 * 8
        expected = 2 * 2 * 8 * hidden_dim * 1
        assert result == expected


# ============================================================================
# 4. exist_prefill / exist_decode — Need paddle tensors + envs
# ============================================================================


class TestExistPrefill:
    """Tests for exist_prefill().

    Mock justification:
      - envs.ENABLE_V1_KVCACHE_SCHEDULER: global environment variable system,
        not an internal function of GPUModelRunner.
    """

    def _make_runner_with_tensors(self, seq_lens_encoder, seq_lens_decoder=None):
        """Create runner with real paddle tensors in share_inputs."""
        runner = create_runner_minimal()
        data = {}
        data["seq_lens_encoder"] = paddle.to_tensor(seq_lens_encoder, dtype="int32")
        if seq_lens_decoder is not None:
            data["seq_lens_decoder"] = paddle.to_tensor(seq_lens_decoder, dtype="int32")

        class TensorShareInputs:
            def __getitem__(self, key):
                return data[key]

        runner.share_inputs = TensorShareInputs()
        runner.exist_prefill_flag = False
        return runner

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_v1_scheduler_flag_true(self, mock_envs):
        """V1 scheduler path: returns the flag directly."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = True
        runner = self._make_runner_with_tensors([0, 0, 0])
        runner.exist_prefill_flag = True
        assert runner.exist_prefill() is True

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_v1_scheduler_flag_false(self, mock_envs):
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = True
        runner = self._make_runner_with_tensors([0, 0, 0])
        runner.exist_prefill_flag = False
        assert runner.exist_prefill() is False

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_non_v1_has_encoder(self, mock_envs):
        """Non-V1 path: checks seq_lens_encoder tensor."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner_with_tensors([0, 5, 0])
        assert runner.exist_prefill() is True

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_non_v1_no_encoder(self, mock_envs):
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner_with_tensors([0, 0, 0])
        assert runner.exist_prefill() is False


class TestExistDecode:
    """Tests for exist_decode(). Uses real paddle tensors."""

    def _make_runner(self, seq_lens_decoder):
        runner = create_runner_minimal()
        data = {"seq_lens_decoder": paddle.to_tensor(seq_lens_decoder, dtype="int32")}

        class TensorShareInputs:
            def __getitem__(self, key):
                return data[key]

        runner.share_inputs = TensorShareInputs()
        return runner

    def test_has_decoder(self):
        runner = self._make_runner([0, 5, 0])
        assert runner.exist_decode() is True

    def test_no_decoder(self):
        runner = self._make_runner([0, 0, 0])
        assert runner.exist_decode() is False

    def test_single_item_batch(self):
        runner = self._make_runner([1])
        assert runner.exist_decode() is True

    def test_single_item_zero(self):
        runner = self._make_runner([0])
        assert runner.exist_decode() is False


# ============================================================================
# 5. only_prefill / only_decode — Non-EP and EP paths
# ============================================================================


class TestOnlyPrefill:
    """Tests for only_prefill().

    Mock justification:
      - paddle.distributed.all_gather_object: NCCL distributed GPU communication,
        requires multi-GPU cluster. This is a real external system.
      - envs: global environment variable system.
    """

    def _make_runner(self, use_ep=False, seq_lens_encoder=None, seq_lens_decoder=None):
        runner = create_runner_minimal()
        enc = seq_lens_encoder or [0, 0, 0]
        dec = seq_lens_decoder or [0, 0, 0]
        data = {
            "seq_lens_encoder": paddle.to_tensor(enc, dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor(dec, dtype="int32"),
        }

        class TensorShareInputs:
            def __getitem__(self, key):
                return data[key]

        runner.share_inputs = TensorShareInputs()
        runner.exist_prefill_flag = any(e > 0 for e in enc)
        runner.fd_config = Mock()
        runner.fd_config.parallel_config = Mock()
        runner.fd_config.parallel_config.use_ep = use_ep
        runner.fd_config.scheduler_config = Mock()
        runner.fd_config.scheduler_config.splitwise_role = "mixed"
        return runner

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_non_ep_no_decode(self, mock_envs):
        """Non-EP: only_prefill is True when no decode exists."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=False, seq_lens_encoder=[5, 0], seq_lens_decoder=[0, 0])
        assert runner.only_prefill() is True

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_non_ep_has_decode(self, mock_envs):
        """Non-EP: only_prefill is False when decode exists."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=False, seq_lens_encoder=[5, 0], seq_lens_decoder=[0, 3])
        assert runner.only_prefill() is False

    @patch("paddle.distributed.all_gather_object")
    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_ep_all_no_decode(self, mock_envs, mock_all_gather):
        """EP: all ranks report no decode."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=True, seq_lens_encoder=[5], seq_lens_decoder=[0])

        # Mock: all_gather_object appends results to the list
        def fake_all_gather(result_list, local_value):
            result_list.extend([True, True])  # all ranks: not decode_exists

        mock_all_gather.side_effect = fake_all_gather
        assert runner.only_prefill() is True

    @patch("paddle.distributed.all_gather_object")
    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_ep_some_decode(self, mock_envs, mock_all_gather):
        """EP: one rank has decode, so only_prefill is False."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=True, seq_lens_encoder=[5], seq_lens_decoder=[0])

        def fake_all_gather(result_list, local_value):
            result_list.extend([True, False])  # one rank has decode

        mock_all_gather.side_effect = fake_all_gather
        assert runner.only_prefill() is False


class TestOnlyDecode:
    """Tests for only_decode(). Symmetric to only_prefill.

    Mock justification: same as TestOnlyPrefill.
    """

    def _make_runner(self, use_ep=False, seq_lens_encoder=None, seq_lens_decoder=None):
        runner = create_runner_minimal()
        enc = seq_lens_encoder or [0, 0, 0]
        dec = seq_lens_decoder or [0, 0, 0]
        data = {
            "seq_lens_encoder": paddle.to_tensor(enc, dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor(dec, dtype="int32"),
        }

        class TensorShareInputs:
            def __getitem__(self, key):
                return data[key]

        runner.share_inputs = TensorShareInputs()
        runner.exist_prefill_flag = any(e > 0 for e in enc)
        runner.fd_config = Mock()
        runner.fd_config.parallel_config = Mock()
        runner.fd_config.parallel_config.use_ep = use_ep
        runner.fd_config.scheduler_config = Mock()
        runner.fd_config.scheduler_config.splitwise_role = "mixed"
        return runner

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_non_ep_no_prefill(self, mock_envs):
        """Non-EP: only_decode is True when no prefill exists."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=False, seq_lens_encoder=[0, 0], seq_lens_decoder=[5, 3])
        assert runner.only_decode() is True

    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_non_ep_has_prefill(self, mock_envs):
        """Non-EP: only_decode is False when prefill exists."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=False, seq_lens_encoder=[5, 0], seq_lens_decoder=[0, 3])
        assert runner.only_decode() is False

    @patch("paddle.distributed.all_gather_object")
    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_ep_all_no_prefill(self, mock_envs, mock_all_gather):
        """EP: all ranks report no prefill."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=True, seq_lens_encoder=[0], seq_lens_decoder=[5])

        def fake_all_gather(result_list, local_value):
            result_list.extend([True, True])  # all ranks: not prefill_exists

        mock_all_gather.side_effect = fake_all_gather
        assert runner.only_decode() is True

    @patch("paddle.distributed.all_gather_object")
    @patch("fastdeploy.worker.gpu_model_runner.envs")
    def test_ep_some_prefill(self, mock_envs, mock_all_gather):
        """EP: one rank has prefill, so only_decode is False."""
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        runner = self._make_runner(use_ep=True, seq_lens_encoder=[0], seq_lens_decoder=[5])

        def fake_all_gather(result_list, local_value):
            result_list.extend([True, False])  # one rank has prefill

        mock_all_gather.side_effect = fake_all_gather
        assert runner.only_decode() is False


# ============================================================================
# 6. clear_requests — State reset verification
# ============================================================================


class TestClearRequests:
    """Tests for clear_requests(): verifies observable state changes.

    No mocking of internal functions. The routing_replay_manager mock is
    justified because it connects to an external IPC/routing store.
    """

    def _make_runner(self, batch_size=10, enable_routing_replay=False):
        runner = create_runner_minimal()
        runner.share_inputs = {"stop_flags": np.zeros((batch_size,), dtype=bool)}
        runner.prompt_logprobs_reqs = {"req_1": Mock(), "req_2": Mock()}
        runner.in_progress_prompt_logprobs = {"req_1": Mock()}
        runner.forward_batch_reqs_list = [Mock() for _ in range(batch_size)]
        runner.exist_prefill_flag = True
        runner.scheduler_config = Mock()
        runner.scheduler_config.max_num_seqs = batch_size
        runner.fd_config = Mock()
        runner.fd_config.routing_replay_config = Mock()
        runner.fd_config.routing_replay_config.enable_routing_replay = enable_routing_replay
        if enable_routing_replay:
            # Mock because routing_replay_manager connects to external IPC store
            runner.routing_replay_manager = Mock()
        return runner

    def test_resets_stop_flags_all_true(self):
        runner = self._make_runner()
        runner.clear_requests()
        assert np.all(runner.share_inputs["stop_flags"] == True)

    def test_clears_prompt_logprobs(self):
        runner = self._make_runner()
        runner.clear_requests()
        assert len(runner.prompt_logprobs_reqs) == 0

    def test_clears_in_progress_logprobs(self):
        runner = self._make_runner()
        runner.clear_requests()
        assert len(runner.in_progress_prompt_logprobs) == 0

    def test_resets_forward_batch_list(self):
        runner = self._make_runner(batch_size=5)
        runner.clear_requests()
        assert all(r is None for r in runner.forward_batch_reqs_list)
        assert len(runner.forward_batch_reqs_list) == 5

    def test_resets_prefill_flag(self):
        runner = self._make_runner()
        runner.clear_requests()
        assert runner.exist_prefill_flag is False

    def test_idempotent(self):
        """Calling clear_requests on already-cleared state doesn't error."""
        runner = self._make_runner()
        runner.clear_requests()
        runner.clear_requests()  # second call
        assert np.all(runner.share_inputs["stop_flags"] == True)
        assert len(runner.prompt_logprobs_reqs) == 0

    def test_routing_replay_puts_table_to_store(self):
        """When routing replay is enabled, the external store is notified."""
        runner = self._make_runner(enable_routing_replay=True)
        runner.clear_requests()
        # This IS an external IPC system call, so we verify it was called
        runner.routing_replay_manager.put_table_to_store.assert_called_once()


# ============================================================================
# 7. insert_tasks_v1 — Core batch setup
# ============================================================================


class TestInsertTasksV1:
    """Tests for insert_tasks_v1: the core batch setup method.

    Mock justification:
      - set_stop: CUDA custom kernel (C++ GPU op, set_stop_value). Requires
        physical GPU. We patch at module level where it was imported.
      - self.sampler: GPU sampling component that runs CUDA kernels.
      - self._process_mm_features: We do NOT mock it. Instead, enable_mm=False
        causes it to return immediately.
    """

    def _make_runner(self, batch_size=10):
        runner = create_runner_minimal()
        runner.share_inputs = create_share_inputs_dict(batch_size=batch_size)
        runner.enable_mm = False  # skip _process_mm_features without mocking
        runner.is_pooling_model = False
        runner.model_config = Mock()
        runner.model_config.eos_tokens_lens = 1
        runner.model_config.max_model_len = 4096
        runner.model_config.max_stop_seqs_num = 4
        runner.cache_config = Mock()
        runner.cache_config.enable_chunked_prefill = False
        runner.scheduler_config = Mock()
        runner.scheduler_config.max_num_seqs = batch_size
        runner.scheduler_config.splitwise_role = "mixed"
        runner.fd_config = Mock()
        runner.fd_config.scheduler_config = runner.scheduler_config
        runner.fd_config.routing_replay_config = Mock()
        runner.fd_config.routing_replay_config.enable_routing_replay = False
        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_method = None
        runner.speculative_decoding = False
        runner.forward_batch_reqs_list = [None] * batch_size
        runner.prompt_logprobs_reqs = {}
        runner.in_progress_prompt_logprobs = {}
        runner.exist_prefill_flag = False
        runner.pooling_params = []
        runner.guided_backend = None

        # Mock sampler because it's a GPU component
        runner.sampler = Mock()
        runner.sampler.apply_logits_processor = Mock()

        return runner

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_single_prefill_sets_state(self, mock_set_stop):
        """A PREFILL request correctly populates share_inputs arrays."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[10, 20, 30, 40],
            eos_token_ids=[2],
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        si = runner.share_inputs
        assert si["req_ids"][0] == "req_0"
        assert np.array_equal(si["input_ids"][0, :4], [10, 20, 30, 40])
        assert np.array_equal(si["prompt_ids"][0, :4], [10, 20, 30, 40])
        assert si["seq_lens_encoder"][0] == 4
        assert si["stop_flags"][0] == False
        assert runner.exist_prefill_flag is True
        assert runner.forward_batch_reqs_list[0] is req

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_single_decode_sets_block_tables(self, mock_set_stop):
        """A DECODE request correctly updates block tables."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.DECODE,
            idx=0,
            block_tables=[5, 6, 7, 8],
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        si = runner.share_inputs
        assert si["encoder_block_lens"][0] == 4
        assert np.array_equal(si["block_tables"][0, :4], [5, 6, 7, 8])

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_preempted_clears_state(self, mock_set_stop):
        """A PREEMPTED request clears its slot."""
        runner = self._make_runner()
        runner.forward_batch_reqs_list[0] = Mock()

        req = create_mock_request(
            task_type=RequestType.PREEMPTED,
            idx=0,
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        si = runner.share_inputs
        assert si["stop_flags"][0] == True
        assert si["seq_lens_decoder"][0] == 0
        assert si["seq_lens_encoder"][0] == 0
        assert np.all(si["block_tables"][0, :] == -1)
        assert runner.forward_batch_reqs_list[0] is None

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_batch_mixed_types(self, mock_set_stop):
        """A batch with PREFILL, DECODE, and PREEMPTED requests."""
        runner = self._make_runner()
        prefill_req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1, 2, 3],
            eos_token_ids=[2],
        )
        decode_req = create_mock_request(
            task_type=RequestType.DECODE,
            idx=1,
            block_tables=[10, 11],
        )
        preempted_req = create_mock_request(
            task_type=RequestType.PREEMPTED,
            idx=2,
        )

        runner.insert_tasks_v1(
            [prefill_req, decode_req, preempted_req],
            num_running_requests=3,
        )

        si = runner.share_inputs
        # Prefill slot
        assert si["seq_lens_encoder"][0] == 3
        # Note: exist_prefill_flag is False because the PREEMPTED handler
        # (processed after PREFILL) resets it. This is the actual behavior.
        assert runner.exist_prefill_flag is False
        # Decode slot
        assert si["encoder_block_lens"][1] == 2
        # Preempted slot
        assert si["stop_flags"][2] == True

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_sampling_params_propagation(self, mock_set_stop):
        """Sampling params are written to share_inputs."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1, 2],
            eos_token_ids=[2],
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        si = runner.share_inputs
        assert si["temperature"][0] == pytest.approx(0.5)
        assert si["top_p"][0] == pytest.approx(0.9)
        assert si["top_k"][0] == 50
        assert si["min_p"][0] == pytest.approx(0.1)
        assert si["penalty_score"][0] == pytest.approx(1.2)
        assert si["frequency_score"][0] == pytest.approx(0.5)
        assert si["presence_score"][0] == pytest.approx(0.3)

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_prompt_logprobs_tracking(self, mock_set_stop):
        """Requests with prompt_logprobs are tracked."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1, 2, 3],
            eos_token_ids=[2],
            request_id="logprob_req",
            prompt_logprobs=5,
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        assert "logprob_req" in runner.prompt_logprobs_reqs

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_chunked_prefill_state(self, mock_set_stop):
        """Chunked prefill: prefill_start_index > 0."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            eos_token_ids=[2],
            prefill_start_index=5,
            prefill_end_index=8,
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        si = runner.share_inputs
        assert si["seq_lens_decoder"][0] == 5  # prefill_start_index
        assert si["seq_lens_encoder"][0] == 3  # length = 8 - 5
        assert si["is_chunk_step"][0] == True  # 8 < 10

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_sets_num_running_requests(self, mock_set_stop):
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1],
            eos_token_ids=[2],
        )

        runner.insert_tasks_v1([req], num_running_requests=3)

        assert runner.share_inputs["num_running_requests"] == 3

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_thinking_enabled(self, mock_set_stop):
        """enable_thinking and reasoning_max_tokens are set."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1, 2],
            eos_token_ids=[2],
            enable_thinking=1,
            reasoning_max_tokens=100,
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        si = runner.share_inputs
        assert si["enable_thinking"][0, 0] == 1
        assert si["max_think_lens"][0, 0] == 100

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_thinking_disabled(self, mock_set_stop):
        """When enable_thinking is None, max_think_lens is -1."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1, 2],
            eos_token_ids=[2],
            enable_thinking=None,
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        si = runner.share_inputs
        assert si["max_think_lens"][0, 0] == -1

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_empty_request_list(self, mock_set_stop):
        """Empty request list should not error and set_stop should not be called."""
        runner = self._make_runner()
        runner.insert_tasks_v1([], num_running_requests=0)
        # No state changes for individual slots
        assert runner.exist_prefill_flag is False

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_eos_token_propagation(self, mock_set_stop):
        """EOS token IDs are written to share_inputs."""
        runner = self._make_runner()
        req = create_mock_request(
            task_type=RequestType.PREFILL,
            idx=0,
            token_ids=[1, 2],
            eos_token_ids=[99],
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        assert runner.share_inputs["eos_token_id"][0, 0] == 99

    @patch("fastdeploy.worker.gpu_model_runner.set_stop")
    def test_preempted_clears_prompt_logprobs(self, mock_set_stop):
        """PREEMPTED request removes prompt logprobs tracking."""
        runner = self._make_runner()
        runner.prompt_logprobs_reqs["req_0"] = Mock()
        runner.in_progress_prompt_logprobs["req_0"] = Mock()

        req = create_mock_request(
            task_type=RequestType.PREEMPTED,
            idx=0,
            request_id="req_0",
        )

        runner.insert_tasks_v1([req], num_running_requests=1)

        assert "req_0" not in runner.prompt_logprobs_reqs
        assert "req_0" not in runner.in_progress_prompt_logprobs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
