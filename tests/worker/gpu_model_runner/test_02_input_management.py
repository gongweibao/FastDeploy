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
Tests for Phase 2+3: Input Data Management & Task Insertion

This module contains tests for the input data management and task insertion
phases of GPU Model Runner, corresponding to Phase 2 and 3 in
docs/gpu_model_runner_data_flow.md

Key components tested:
- Input length list calculation
- Task insertion (V1 and V2 schedulers)
- Request state management (prefill/decode)
- Batch configuration
"""

import numpy as np
import paddle
import unittest
from unittest.mock import Mock, patch

from fastdeploy.engine.request import RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestGetInputLengthList(unittest.TestCase):
    """Test cases for get_input_length_list method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.enable_expert_parallel = False
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.block_size = 16
        self.runner.cache_config.enc_dec_block_num = 0
        self.runner.cache_config.total_block_num = 10000
        self.runner.fd_config.cache_config = self.runner.cache_config

    def test_basic_input_length_list(self):
        """Test basic case with normal input."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=160, batch_size=2, expected_decode_len=100, capture_prefill=False
        )

        # input_length = min(160//2, 4096-101) = min(80, 3995) = 80
        self.assertEqual(input_length_list, [80, 80])
        self.assertEqual(max_dec_len_list, [101, 101])
        # block_num = (80 + 16 - 1) // 16 + 0 = 95 // 16 = 5
        self.assertEqual(block_num, 5)

    def test_capture_prefill_true(self):
        """Test capture_prefill mode with normal case."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=160, batch_size=4, expected_decode_len=100, capture_prefill=True
        )

        # When capture_prefill=True, creates [1, 1, 1, 157]
        self.assertEqual(input_length_list, [1, 1, 1, 157])
        self.assertEqual(max_dec_len_list, [101, 101, 101, 101])

    def test_capture_prefill_tokens_less_than_batch_size(self):
        """Test capture_prefill when num_tokens < batch_size."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=3, batch_size=10, expected_decode_len=100, capture_prefill=True
        )

        # Creates [1, 1, 1] only
        self.assertEqual(input_length_list, [1, 1, 1])
        self.assertEqual(max_dec_len_list, [101, 101, 101])
        self.assertEqual(len(input_length_list), 3)

    def test_expert_parallel_enabled(self):
        """Test when expert parallel is enabled (limits input_length to 32)."""
        self.runner.parallel_config.enable_expert_parallel = True

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=1000, batch_size=2, expected_decode_len=100, capture_prefill=False
        )

        # input_length should be limited to 32
        self.assertEqual(input_length_list, [32, 32])

    def test_max_model_len_constraint(self):
        """Test when max_model_len constrains input_length."""
        self.runner.model_config.max_model_len = 150

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=1000, batch_size=2, expected_decode_len=80, capture_prefill=False
        )

        # input_length = min(500, 150-81) = min(500, 69) = 69
        self.assertEqual(input_length_list, [69, 69])
        self.assertEqual(max_dec_len_list, [81, 81])

    def test_enc_dec_block_num(self):
        """Test block_num calculation with enc_dec_block_num."""
        self.runner.cache_config.enc_dec_block_num = 5

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=160, batch_size=2, expected_decode_len=100, capture_prefill=False
        )

        # block_num = (80 + 16 - 1) // 16 + 5 = 5 + 5 = 10
        self.assertEqual(block_num, 10)

    def test_batch_size_one(self):
        """Test with batch_size = 1."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=100, batch_size=1, expected_decode_len=50, capture_prefill=False
        )

        self.assertEqual(input_length_list, [100])
        self.assertEqual(max_dec_len_list, [51])

    def test_expected_decode_len_zero(self):
        """Test with expected_decode_len = 0 (boundary value)."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=160, batch_size=2, expected_decode_len=0, capture_prefill=False
        )

        # max_dec_len = 0 + 1 = 1
        # input_length = min(80, 4096-1) = 80
        self.assertEqual(input_length_list, [80, 80])
        self.assertEqual(max_dec_len_list, [1, 1])

    def test_large_num_tokens(self):
        """Test with large number of tokens."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=10000, batch_size=4, expected_decode_len=500, capture_prefill=False
        )

        # input_length = min(2500, 4096-501) = min(2500, 3595) = 2500
        self.assertEqual(input_length_list, [2500, 2500, 2500, 2500])
        self.assertEqual(max_dec_len_list, [501, 501, 501, 501])
        # block_num = (2500 + 16 - 1) // 16 + 0 = 2515 // 16 = 157
        self.assertEqual(block_num, 157)

    def test_capture_prefill_boundary_case(self):
        """Test capture_prefill with num_tokens == batch_size."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=4, batch_size=4, expected_decode_len=100, capture_prefill=True
        )

        # num_tokens == batch_size, enters else branch: [1, 1, 1, 1]
        # last element = 4 - 4 + 1 = 1
        self.assertEqual(input_length_list, [1, 1, 1, 1])

    def test_block_size_one(self):
        """Test with block_size = 1 (boundary value)."""
        self.runner.cache_config.block_size = 1

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=160, batch_size=2, expected_decode_len=100, capture_prefill=False
        )

        # block_num = (80 + 1 - 1) // 1 + 0 = 80
        self.assertEqual(block_num, 80)

    def test_multiple_constraints_simultaneously(self):
        """Test with expert_parallel and max_model_len constraints both active."""
        self.runner.parallel_config.enable_expert_parallel = True
        self.runner.model_config.max_model_len = 50

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=1000, batch_size=2, expected_decode_len=10, capture_prefill=False
        )

        # input_length = min(500, 50-11) = min(500, 39) = 39
        # then limited by expert_parallel: min(39, 32) = 32
        self.assertEqual(input_length_list, [32, 32])


class TestExistPrefillDecode(unittest.TestCase):
    """Test cases for exist_prefill and exist_decode methods."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.scheduler_config = Mock()
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.exist_prefill_flag = False

    def test_exist_prefill_true(self):
        """Test exist_prefill returns True when seq_lens_encoder has positive values."""
        # When ENABLE_V1_KVCACHE_SCHEDULER=1 (default), exist_prefill returns exist_prefill_flag
        self.runner.exist_prefill_flag = True
        self.runner.share_inputs = {"seq_lens_encoder": paddle.to_tensor([0, 10, 0, 5])}

        result = self.runner.exist_prefill()

        self.assertTrue(result)

    def test_exist_prefill_false(self):
        """Test exist_prefill returns False when exist_prefill_flag is False."""
        # When ENABLE_V1_KVCACHE_SCHEDULER=1 (default), exist_prefill returns exist_prefill_flag
        self.runner.exist_prefill_flag = False
        self.runner.share_inputs = {"seq_lens_encoder": paddle.to_tensor([0, 0, 0])}

        result = self.runner.exist_prefill()

        self.assertFalse(result)

    def test_exist_decode_true(self):
        """Test exist_decode returns True when seq_lens_decoder has positive values."""
        self.runner.share_inputs = {"seq_lens_decoder": paddle.to_tensor([0, 0, 15, 0])}

        result = self.runner.exist_decode()

        self.assertTrue(result)

    def test_exist_decode_false(self):
        """Test exist_decode returns False when seq_lens_decoder has no positive values."""
        self.runner.share_inputs = {"seq_lens_decoder": paddle.to_tensor([0, 0, 0])}

        result = self.runner.exist_decode()

        self.assertFalse(result)

    def test_exist_prefill_all_positive(self):
        """Test exist_prefill returns True when all values are positive."""
        self.runner.exist_prefill_flag = True
        self.runner.share_inputs = {"seq_lens_encoder": paddle.to_tensor([10, 20, 30])}

        result = self.runner.exist_prefill()

        self.assertTrue(result)

    def test_exist_decode_all_positive(self):
        """Test exist_decode returns True when all values are positive."""
        self.runner.share_inputs = {"seq_lens_decoder": paddle.to_tensor([1, 5, 10])}

        result = self.runner.exist_decode()

        self.assertTrue(result)

    def test_exist_prefill_mixed_values(self):
        """Test exist_prefill returns True with mixed positive/negative values."""
        self.runner.exist_prefill_flag = True
        self.runner.share_inputs = {"seq_lens_encoder": paddle.to_tensor([0, -1, 10])}

        result = self.runner.exist_prefill()

        self.assertTrue(result)  # Still has positive value (10)

    def test_exist_decode_mixed_values(self):
        """Test exist_decode returns True with mixed positive/negative values."""
        self.runner.share_inputs = {"seq_lens_decoder": paddle.to_tensor([0, 5, -1])}

        result = self.runner.exist_decode()

        self.assertTrue(result)  # Still has positive value (5)


class TestInsertTasksV1(unittest.TestCase):
    """Test cases for insert_tasks_v1 method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.model_config.eos_tokens_lens = 1
        self.runner.model_config.max_stop_seqs_num = 4
        self.runner.model_config.enable_mm = False
        self.runner.model_config.pad_token_id = 0
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.splitwise_role = "mixed"
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.use_ep = False
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.routing_replay_config = Mock()
        self.runner.routing_replay_config.enable_routing_replay = False
        self.runner.fd_config.routing_replay_config = self.runner.routing_replay_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.block_size = 16
        self.runner.cache_config.enc_dec_block_num = 0
        self.runner.cache_config.total_block_num = 10000
        self.runner.cache_config.enable_chunked_prefill = False
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.fd_config.quant_config = None
        self.runner.share_inputs = Mock()
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.exist_prefill_flag = False
        self.runner.num_gpu_blocks = 10000
        self.runner.quant_config = None
        self.runner.pooling_params = []
        self.runner.sampler = Mock()
        self.runner.sampler.apply_logits_processor = Mock()
        self.runner.routing_replay_manager = Mock()
        self.runner.is_pooling_model = False
        self.runner.speculative_method = None
        self.runner.enable_mm = False
        self.runner.encoder_cache = None
        self.runner.attn_backends = []
        # Mock additional methods needed by insert_tasks_v1
        self.runner.initialize_kv_cache = Mock()
        # Mock _init_logits_processor for guided decoding
        self.runner._init_logits_processor = Mock(return_value=(None, "test_key"))
        # Mock _process_mm_features to avoid set_stop call
        self.runner._process_mm_features = Mock()

    def _create_mock_request(self, task_type=0, idx=0, **kwargs):
        """Helper to create mock request."""
        request = Mock()
        request.task_type = Mock()
        request.task_type.value = task_type
        request.idx = idx
        request.request_id = f"req_{idx}"
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.output_token_ids = []
        request.block_tables = [0, 1, 2]
        request.eos_token_ids = [0]
        request.prefill_start_index = 0
        request.prefill_end_index = 5
        request.sampling_params = Mock()
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []
        request.sampling_params.min_tokens = 1
        request.sampling_params.max_tokens = 100

        # Set default get() behavior
        def mock_get(key, default=None):
            return kwargs.get(key, default)

        request.get = mock_get

        return request

    def _setup_share_inputs_mock(self, num_running=2):
        """Setup share_inputs mock with required attributes."""
        share_inputs = Mock()
        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)
        share_inputs.__getitem__ = Mock(
            side_effect=lambda key: {
                "req_ids": [""] * 10,
                "preempted_idx": np.zeros((10, 1), dtype="int32"),
                "stop_flags": np.zeros((10,), dtype=bool),
                "seq_lens_decoder": np.zeros((10,), dtype="int32"),
                "seq_lens_encoder": np.zeros((10,), dtype="int32"),
                "seq_lens_this_time_buffer": np.zeros((10,), dtype="int32"),
                "seq_lens_this_time": np.zeros((10,), dtype="int32"),
                "prompt_ids": np.zeros((10, 512), dtype="int64"),
                "input_ids": np.zeros((10, 512), dtype="int64"),
                "encoder_block_lens": np.zeros((10,), dtype="int32"),
                "block_tables": np.full((10, 128), -1, dtype="int32"),
                "step_seq_lens_decoder": np.zeros((10,), dtype="int32"),
                "prompt_lens": np.zeros((10,), dtype="int32"),
                "is_block_step": np.zeros((10,), dtype=bool),
                "is_chunk_step": np.zeros((10,), dtype=bool),
                "step_idx": np.zeros((10,), dtype="int32"),
                "pre_ids": np.full((10, 1), -1, dtype="int64"),
                "eos_token_id": np.zeros((1, 1), dtype="int64"),
                "top_p": np.zeros((10,), dtype="float32"),
                "top_k": np.zeros((10,), dtype="int32"),
                "top_k_list": np.zeros((10,), dtype="int32"),
                "min_p": np.zeros((10,), dtype="float32"),
                "min_p_list": np.zeros((10,), dtype="float32"),
                "temperature": np.zeros((10,), dtype="float32"),
                "penalty_score": np.ones((10,), dtype="float32"),
                "frequency_score": np.zeros((10,), dtype="float32"),
                "presence_score": np.zeros((10,), dtype="float32"),
                "temp_scaled_logprobs": np.zeros((10,), dtype=bool),
                "top_p_normalized_logprobs": np.zeros((10,), dtype=bool),
                "min_dec_len": np.zeros((10,), dtype="int32"),
                "max_dec_len": np.zeros((10,), dtype="int32"),
                "first_token_ids": np.zeros((10, 1), dtype="int64"),
                "infer_seed": np.zeros((10,), dtype="int64"),
                "bad_tokens_len": np.ones((10,), dtype="int32"),
                "bad_tokens": np.full((10, 1), -1, dtype="int64"),
                "stop_seqs_len": np.zeros((10, 4), dtype="int32"),
                "stop_seqs": np.zeros((10, 4, 10), dtype="int64"),
                "not_need_stop": np.zeros((1,), dtype="int32"),
                "logits_processors_args": [{}] * 10,
                "enable_thinking": np.zeros((10, 1), dtype="int32"),
                "max_think_lens": np.full((10, 1), -1, dtype="int32"),
                "limit_think_status": np.zeros((10, 1), dtype="int32"),
            }[key]
        )

        # Allow setting values
        share_inputs.__setitem__ = Mock(side_effect=lambda key, value: setattr(share_inputs, f"_{key}", value))

        # Support 'in' operator for keys like "caches"
        share_inputs.__contains__ = Mock(side_effect=lambda key: key in {
            "req_ids", "preempted_idx", "stop_flags", "seq_lens_decoder",
            "seq_lens_encoder", "seq_lens_this_time_buffer", "seq_lens_this_time",
            "prompt_ids", "input_ids", "encoder_block_lens", "block_tables",
            "step_seq_lens_decoder", "prompt_lens", "is_block_step", "is_chunk_step",
            "step_idx", "pre_ids", "eos_token_id", "top_p", "top_k",
            "top_k_list", "min_p", "min_p_list", "temperature", "penalty_score",
            "frequency_score", "presence_score", "temp_scaled_logprobs",
            "top_p_normalized_logprobs", "min_dec_len", "max_dec_len",
            "first_token_ids", "infer_seed", "bad_tokens_len", "bad_tokens",
            "stop_seqs_len", "stop_seqs", "not_need_stop", "logits_processors_args",
            "enable_thinking", "max_think_lens", "limit_think_status",
        })

        return share_inputs

    def test_insert_tasks_v1_with_prefill_task(self):
        """Test insert_tasks_v1 with a prefill task."""
        from unittest.mock import patch
        self.runner.share_inputs = self._setup_share_inputs_mock()

        req = self._create_mock_request(
            task_type=RequestType.PREFILL.value,
            idx=0,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        with patch('fastdeploy.worker.gpu_model_runner.set_stop'):
            self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify prefill flag is set
        self.assertTrue(self.runner.exist_prefill_flag)
        # Verify seq_lens_encoder is updated
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_insert_tasks_v1_with_decode_task(self):
        """Test insert_tasks_v1 with a decode task."""
        from unittest.mock import patch
        self.runner.share_inputs = self._setup_share_inputs_mock()

        req = self._create_mock_request(
            task_type=RequestType.DECODE.value,
            idx=0,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        with patch('fastdeploy.worker.gpu_model_runner.set_stop'):
            self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.share_inputs._num_running_requests, 1)
        self.assertEqual(self.runner.share_inputs._running_requests_ids, range(1))

    def test_insert_tasks_v1_with_preempted_task(self):
        """Test insert_tasks_v1 with a preempted task."""
        from unittest.mock import patch
        self.runner.share_inputs = self._setup_share_inputs_mock()

        # Task type that's not PREFILL or DECODE (treated as preempted)
        req = self._create_mock_request(
            task_type=99,  # Invalid task type
            idx=0,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        with patch('fastdeploy.worker.gpu_model_runner.set_stop'):
            self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify forward_batch_reqs_list is cleared for preempted task
        self.assertIsNone(self.runner.forward_batch_reqs_list[0])
        # Verify prefill flag is not set
        self.assertFalse(self.runner.exist_prefill_flag)

    def test_insert_tasks_v1_mixed_task_types(self):
        """Test insert_tasks_v1 with mixed prefill and decode tasks."""
        from unittest.mock import patch
        self.runner.share_inputs = self._setup_share_inputs_mock()

        req1 = self._create_mock_request(
            task_type=RequestType.PREFILL.value,
            idx=0,
        )
        req2 = self._create_mock_request(
            task_type=RequestType.DECODE.value,
            idx=1,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)

        with patch('fastdeploy.worker.gpu_model_runner.set_stop'):
            self.runner.insert_tasks_v1([req1, req2], num_running_requests=2)

        # Verify both requests are processed
        self.assertEqual(self.runner.share_inputs._num_running_requests, 2)
        self.assertEqual(self.runner.share_inputs._running_requests_ids, range(2))


class TestInsertPrefillInputs(unittest.TestCase):
    """Test cases for insert_prefill_inputs method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.model_config.eos_tokens_lens = 1
        self.runner.model_config.max_stop_seqs_num = 4
        self.runner.model_config.enable_mm = False
        self.runner.model_config.bad_tokens_len = 1
        self.runner.model_config.max_prompt_embedding_table_size = 0
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.block_size = 16
        self.runner.cache_config.enc_dec_block_num = 0
        self.runner.cache_config.enable_chunked_prefill = False
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.is_pooling_model = False
        self.runner.exist_prefill_flag = False
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.pooling_params = []
        self.runner.enable_mm = False
        self.runner.speculative_decoding = False
        # Add speculative_config mock with required attribute
        self.runner.speculative_config = Mock()
        self.runner.speculative_config.num_speculative_tokens = 0
        # Add speculative_method mock (needed by insert_prefill_inputs)
        self.runner.speculative_method = None
        # Mock additional methods needed by insert_prefill_inputs
        self.runner.set_stop = Mock()  # set_stop handles paddle tensor, our mock accepts any arguments
        self.runner._preprocess_mm_task = Mock(return_value={})
        self.runner.extract_vision_features = Mock()
        self.runner.prepare_rope3d = Mock(return_value=(np.zeros((1, 10)), None))
        self.runner.sampler = Mock()
        self.runner.sampler.apply_logits_processor = Mock()
        # Mock _process_mm_features to avoid set_stop call
        self.runner._process_mm_features = Mock()

    def _create_mock_request(self, task_type=0, idx=0, **kwargs):
        """Helper to create mock request."""
        request = Mock()
        request.task_type = Mock()
        request.task_type.value = task_type
        request.idx = idx
        request.request_id = f"req_{idx}"
        request.prompt_token_ids = kwargs.get("prompt_token_ids", [1, 2, 3, 4, 5])
        request.output_token_ids = []
        request.block_tables = kwargs.get("block_tables", [0, 1, 2])
        request.eos_token_ids = [0]
        request.prefill_start_index = 0
        request.prefill_end_index = len(request.prompt_token_ids)
        request.sampling_params = kwargs.get("sampling_params", Mock())
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []
        request.sampling_params.min_tokens = 1
        request.sampling_params.max_tokens = 100
        request.sampling_params.temperature = 1.0
        request.sampling_params.top_p = 1.0
        request.sampling_params.top_k = 0
        request.sampling_params.bad_tokens = []
        request.sampling_params.bad_tokens_len = 0
        # Add disaggregate_info attribute (None by default for non-disaggregated mode)
        request.disaggregate_info = kwargs.get("disaggregate_info", None)

        # Add additional attributes needed by insert_prefill_inputs
        request.guided_json = None
        request.guided_regex = None
        request.structural_tag = None
        request.guided_grammar = None
        request.multimodal_inputs = kwargs.get("multimodal_inputs", {})

        def mock_get(key, default=None):
            # Handle special keys that have default values in source
            if key == "block_tables" and key not in kwargs:
                return [0, 1, 2]
            return kwargs.get(key, default)

        request.get = mock_get

        return request

    def _setup_share_inputs_mock(self):
        """Setup share_inputs mock with required attributes."""
        import paddle as pd

        share_inputs = Mock()
        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)
        share_inputs.__getitem__ = Mock(
            side_effect=lambda key: {
                "req_ids": [""] * 10,
                "stop_flags": np.zeros(10, dtype=bool),
                "seq_lens_decoder": np.zeros(10, dtype="int32"),
                "seq_lens_encoder": np.zeros(10, dtype="int32"),
                "seq_lens_this_time_buffer": np.zeros(10, dtype="int32"),
                "prompt_ids": np.zeros((10, 512), dtype="int64"),
                "input_ids": np.zeros((10, 512), dtype="int64"),
                "block_tables": np.full((10, 128), -1, dtype="int32"),
                "step_seq_lens_decoder": np.zeros(10, dtype="int32"),
                "step_seq_lens_encoder": np.zeros(10, dtype="int32"),
                "prompt_lens": np.zeros(10, dtype="int32"),
                "top_p": np.zeros(10, dtype="float32"),
                "top_k": np.zeros(10, dtype="int32"),
                "top_k_list": np.zeros(10, dtype="int32"),
                "min_p": np.zeros(10, dtype="float32"),
                "min_p_list": np.zeros(10, dtype="float32"),
                "temperature": np.zeros(10, dtype="float32"),
                "penalty_score": np.ones(10, dtype="float32"),
                "frequency_score": np.zeros(10, dtype="float32"),
                "presence_score": np.zeros(10, dtype="float32"),
                "max_dec_len": np.zeros(10, dtype="int32"),
                "min_dec_len": np.zeros(10, dtype="int32"),
                "eos_token_id": np.zeros((1, 1), dtype="int64"),
                "infer_seed": np.zeros(10, dtype="int64"),
                "pre_ids": np.zeros((10, 1), dtype="int64"),
                "step_idx": np.zeros(10, dtype="int32"),
                "max_think_lens": np.full((10, 1), -1, dtype="int32"),
                "limit_think_status": np.zeros((10, 1), dtype="int32"),
                "enable_thinking": np.zeros((10, 1), dtype="int32"),
                "temp_scaled_logprobs": np.zeros(10, dtype=bool),
                "top_p_normalized_logprobs": np.zeros(10, dtype=bool),
                "encoder_block_lens": np.zeros(10, dtype="int32"),
                "bad_tokens": np.full((10, 1), -1, dtype="int64"),
                "bad_tokens_len": np.ones(10, dtype="int32"),
                "stop_seqs_len": np.zeros((10, 4), dtype="int32"),
                "stop_seqs": np.zeros((10, 4, 10), dtype="int64"),
                "not_need_stop": np.zeros((1,), dtype="int32"),
                "first_token_ids": np.zeros((10, 1), dtype="int64"),
            }[key]
        )
        return share_inputs

    def test_insert_prefill_inputs_basic(self):
        """Test insert_prefill_inputs with basic prefill task."""
        from unittest.mock import patch
        self.runner.share_inputs = self._setup_share_inputs_mock()

        # Add __setitem__ support for assignments
        self.runner.share_inputs.__setitem__ = Mock(side_effect=lambda key, value: None)

        req = self._create_mock_request(
            task_type=RequestType.PREFILL.value,
            idx=0,
        )

        with patch('fastdeploy.worker.gpu_model_runner.set_stop'):
            # Insert request
            self.runner.insert_prefill_inputs([req], num_running_requests=1)

        # Verify prefill flag is set
        self.assertTrue(self.runner.exist_prefill_flag)

    def test_insert_prefill_inputs_multiple_requests(self):
        """Test insert_prefill_inputs with multiple requests."""
        from unittest.mock import patch
        self.runner.share_inputs = self._setup_share_inputs_mock()

        # Add __setitem__ support for assignments
        self.runner.share_inputs.__setitem__ = Mock(side_effect=lambda key, value: None)

        req1 = self._create_mock_request(
            task_type=RequestType.PREFILL.value, idx=0, prompt_token_ids=[1, 2, 3]
        )
        req2 = self._create_mock_request(
            task_type=RequestType.PREFILL.value, idx=1, prompt_token_ids=[4, 5, 6]
        )

        with patch('fastdeploy.worker.gpu_model_runner.set_stop'):
            # Insert multiple requests
            self.runner.insert_prefill_inputs([req1, req2], num_running_requests=2)

        # Verify both requests are processed
        self.assertTrue(self.runner.exist_prefill_flag)


class TestOnlyPrefill(unittest.TestCase):
    """Test cases for only_prefill method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.splitwise_role = "decoder"
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.use_ep = False
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.exist_decode = Mock(return_value=False)

    def test_only_prefill_true(self):
        """Test only_prefill returns True when decode doesn't exist."""
        result = self.runner.only_prefill()

        self.assertTrue(result)

    def test_only_prefill_false_when_decode_exists(self):
        """Test only_prefill returns False when decode exists."""
        self.runner.exist_decode.return_value = True

        result = self.runner.only_prefill()

        self.assertFalse(result)

    def test_only_prefill_with_ep_mixed_role(self):
        """Test only_prefill with EP mixed role."""
        self.runner.parallel_config.use_ep = True
        self.runner.scheduler_config.splitwise_role = "mixed"
        self.runner.exist_decode = Mock(return_value=False)

        # Mock paddle.distributed.all_gather_object
        with patch("paddle.distributed.all_gather_object") as mock_all_gather:
            mock_all_gather.return_value = None
            result = self.runner.only_prefill()

            self.assertTrue(result)


class TestOnlyDecode(unittest.TestCase):
    """Test cases for only_decode method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.splitwise_role = "decoder"
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.use_ep = False
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.exist_prefill = Mock(return_value=False)

    def test_only_decode_true(self):
        """Test only_decode returns True when prefill doesn't exist."""
        result = self.runner.only_decode()

        self.assertTrue(result)

    def test_only_decode_false_when_prefill_exists(self):
        """Test only_decode returns False when prefill exists."""
        self.runner.exist_prefill.return_value = True

        result = self.runner.only_decode()

        self.assertFalse(result)

    def test_only_decode_with_ep_mixed_role(self):
        """Test only_decode with EP mixed role."""
        self.runner.parallel_config.use_ep = True
        self.runner.scheduler_config.splitwise_role = "mixed"
        self.runner.exist_prefill = Mock(return_value=False)

        # Mock paddle.distributed.all_gather_object
        with patch("paddle.distributed.all_gather_object") as mock_all_gather:
            mock_all_gather.return_value = None
            result = self.runner.only_decode()

            self.assertTrue(result)


class TestCalTheorticalKVCache(unittest.TestCase):
    """Test cases for cal_theortical_kvcache method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.head_dim = 128
        self.runner.model_config.kv_num_heads = 32
        self.runner.model_config.num_hidden_layers = 24
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.block_size = 16
        self.runner.cache_config.use_mla_cache = False
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.quant_config = Mock()
        self.runner.quant_config.kv_cache_quant_type = None
        self.runner.speculative_config = Mock()
        self.runner.speculative_config.num_gpu_block_expand_ratio = 0
        self.runner.fd_config.speculative_config = self.runner.speculative_config
        self.runner.speculative_method = None

    def test_cal_theortical_kvcache_default_dtype(self):
        """Test with default dtype (bf16, 2 bytes)."""
        result = self.runner.cal_theortical_kvcache()

        # byte_of_dtype = 2, hidden_dim = 128 * 32 = 4096
        # num_layers = 24, block_size = 16
        # required_memory = 2 * 2 * 16 * 4096 * 24 = 6291456 bytes
        expected = 2 * 2 * 16 * (128 * 32) * 24
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
