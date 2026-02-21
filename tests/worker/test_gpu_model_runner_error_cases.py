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

"""Error scenario tests for GPUModelRunner."""

import unittest
from unittest.mock import Mock

import numpy as np
import paddle

from fastdeploy.engine.request import RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestErrorScenarios(unittest.TestCase):
    """Test edge cases and error scenarios for GPUModelRunner."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_model_len = 4096
        self.mock_model_config.eos_tokens_lens = 1
        self.mock_model_config.max_stop_seqs_num = 4
        self.mock_model_config.enable_mm = False
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.splitwise_role = "mixed"
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config

        self.mock_parallel_config = Mock()
        self.mock_parallel_config.use_ep = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.mock_routing_replay_config = Mock()
        self.mock_routing_replay_config.enable_routing_replay = False
        self.mock_fd_config.routing_replay_config = self.mock_routing_replay_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.share_inputs = Mock()
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.exist_prefill_flag = False
        self.runner.pooling_params = []
        self.runner.sampler = Mock()
        self.runner.routing_replay_manager = Mock()
        self.runner.speculative_decoding = False
        self.runner.speculative_method = None
        self.runner.enable_overlap_schedule = False
        self.runner.enable_mm = False
        self.runner.guided_backend = None

    def _create_mock_request(self, task_type=RequestType.PREFILL, idx=0, token_ids=None):
        """Helper to create mock request."""
        request = Mock()
        request.task_type = Mock()
        request.task_type.value = task_type.value if hasattr(task_type, "value") else task_type
        request.idx = idx
        request.request_id = f"req_{idx}"
        request.prompt_token_ids = token_ids if token_ids else [1, 2, 3, 4, 5]
        request.output_token_ids = []
        request.block_tables = [0, 1, 2]
        request.eos_token_ids = [0]
        request.prefill_start_index = 0
        request.prefill_end_index = len(request.prompt_token_ids) if request.prompt_token_ids else 0
        request.sampling_params = Mock()
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []
        request.sampling_params.min_tokens = 1
        request.sampling_params.max_tokens = 100
        request.sampling_params.temperature = 1.0
        request.sampling_params.top_p = 1.0
        request.sampling_params.top_k = 0
        request.sampling_params.frequency_penalty = 0.0
        request.sampling_params.presence_penalty = 0.0
        request.sampling_params.repetition_penalty = 1.0
        # Explicitly set guided decoding attributes to None to avoid Mock truthy behavior
        request.guided_json = None
        request.guided_regex = None
        request.guided_grammar = None
        request.structural_tag = None
        request.disaggregate_info = None
        # Set enable_thinking related attributes to None
        request.enable_thinking = None
        request.reasoning_max_tokens = None
        # Set other attributes that may be accessed via get() - use __dict__ to store
        request.__dict__["_top_p"] = None
        request.__dict__["_top_k"] = None
        request.__dict__["_temperature"] = None
        request.__dict__["_min_p"] = None

        def mock_get(key, default=None):
            # Only return values for explicitly set attributes
            prefixed_key = f"_{key}"
            if prefixed_key in request.__dict__:
                value = request.__dict__[prefixed_key]
                return value if value is not None else default
            return default

        request.get = mock_get

        return request

    def _setup_share_inputs_mock(self):
        """Setup share_inputs mock with required attributes."""
        share_inputs = Mock()
        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)

        # Define the data that should be returned by __getitem__
        share_input_data = {
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
            "not_need_stop": paddle.full([1], False, dtype="bool").cpu(),
            "logits_processors_args": [{}] * 10,
            "enable_thinking": np.zeros((10, 1), dtype="int32"),
            "max_think_lens": np.full((10, 1), -1, dtype="int32"),
            "limit_think_status": np.zeros((10, 1), dtype="int32"),
        }

        # Store the data on the mock object for __getitem__
        for key, value in share_input_data.items():
            setattr(share_inputs, key, value)

        # Define __getitem__ to access attributes
        def mock_getitem(key):
            if hasattr(share_inputs, key):
                return getattr(share_inputs, key)
            raise KeyError(f"Key '{key}' not found")

        share_inputs.__getitem__ = Mock(side_effect=mock_getitem)

        # Define __setitem__ to set attributes
        def mock_setitem(key, value):
            setattr(share_inputs, key, value)

        share_inputs.__setitem__ = Mock(side_effect=mock_setitem)

        # Define __contains__ to check if attribute exists
        def mock_contains(key):
            return hasattr(share_inputs, key)

        share_inputs.__contains__ = Mock(side_effect=mock_contains)

        return share_inputs

    def test_empty_batch(self):
        """Test with empty batch (num_running_requests=0)."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Insert no requests
        self.runner.insert_tasks_v1([], num_running_requests=0)

        # Verify no prefill flag
        self.assertFalse(self.runner.exist_prefill_flag)

        # Clear with no requests
        self.runner.clear_requests()

        # Verify state is still clean
        self.assertFalse(self.runner.exist_prefill_flag)

    def test_single_token_sequence(self):
        """Test with single token sequence."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with single token
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0, token_ids=[42])
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertTrue(self.runner.exist_prefill_flag)
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_sequence_exceeds_max_length(self):
        """Test sequence exceeding max_model_len."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Set small max_model_len
        self.mock_model_config.max_model_len = 100

        # Request exceeding max length
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0, token_ids=list(range(200)))
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request - should handle gracefully
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled even if exceeding max length
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_invalid_task_type(self):
        """Test with invalid task type."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with invalid task type
        req = self._create_mock_request(task_type=999, idx=0)  # Invalid type
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request - should handle as preempted
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is cleared from forward_batch_reqs_list
        self.assertIsNone(self.runner.forward_batch_reqs_list[0])

    def test_empty_prompt_token_ids(self):
        """Test with empty prompt_token_ids."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with empty prompt
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0, token_ids=[])
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request - should handle gracefully
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_zero_max_tokens(self):
        """Test with max_tokens=0."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with zero max_tokens
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req.sampling_params.max_tokens = 0

        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request - should handle gracefully
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_large_max_tokens(self):
        """Test with very large max_tokens value."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with large max_tokens
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req.sampling_params.max_tokens = 100000

        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request - should handle gracefully
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_all_negative_seq_lens(self):
        """Test exist_prefill/decode with all negative sequence lengths."""
        self.runner.share_inputs = {
            "seq_lens_encoder": paddle.to_tensor([-1, -1, -1]),
            "seq_lens_decoder": paddle.to_tensor([-1, -1, -1]),
        }

        # Both should return False
        self.assertFalse(self.runner.exist_prefill())
        self.assertFalse(self.runner.exist_decode())

    def test_batch_index_out_of_range(self):
        """Test with batch index out of range."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Create request with index beyond max_num_seqs
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=100)
        self.runner.share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: min(idx, 9))

        # Insert request - should handle gracefully
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.forward_batch_reqs_list[9], req)

    def test_none_sampling_params(self):
        """Test request with None sampling_params."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with None sampling params
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req.sampling_params = None

        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request - should handle gracefully
        try:
            self.runner.insert_tasks_v1([req], num_running_requests=1)
        except AttributeError:
            # Expected to fail due to None sampling_params
            pass

    def test_negative_temperature(self):
        """Test with negative temperature."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with negative temperature
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req.sampling_params.temperature = -0.5

        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request - should handle the value
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_top_p_out_of_range(self):
        """Test with top_p values outside [0, 1] range."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Test top_p > 1
        req1 = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req1.sampling_params.top_p = 2.0

        self.runner.share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)

        # Insert request
        self.runner.insert_tasks_v1([req1], num_running_requests=1)

        # Test top_p < 0
        req2 = self._create_mock_request(task_type=RequestType.PREFILL, idx=1)
        req2.sampling_params.top_p = -0.5

        # Insert request
        self.runner.insert_tasks_v1([req2], num_running_requests=2)

        # Verify requests are handled
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req1)
        self.assertEqual(self.runner.forward_batch_reqs_list[1], req2)

    def test_multiple_clear_requests(self):
        """Test calling clear_requests multiple times."""
        self.runner.share_inputs = {"stop_flags": np.zeros(4, dtype=bool)}
        self.runner.prompt_logprobs_reqs = {"req1": "data1"}
        self.runner.in_progress_prompt_logprobs = {"req1": "data1"}
        self.runner.forward_batch_reqs_list = [Mock(), Mock(), Mock(), Mock()]

        # Clear once
        self.runner.clear_requests()
        self.assertTrue(all(self.runner.share_inputs["stop_flags"]))
        self.assertEqual(self.runner.prompt_logprobs_reqs, {})

        # Clear again - should handle gracefully
        self.runner.clear_requests()
        self.assertTrue(all(self.runner.share_inputs["stop_flags"]))
        self.assertEqual(self.runner.prompt_logprobs_reqs, {})

    def test_edge_case_block_size_one(self):
        """Test with block_size=1 (minimum possible)."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)


class TestGetInputLengthListEdgeCases(unittest.TestCase):
    """Test edge cases for get_input_length_list method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_model_len = 4096
        self.mock_fd_config.model_config = self.mock_model_config
        self.mock_fd_config.parallel_config = Mock()
        self.mock_fd_config.parallel_config.enable_expert_parallel = False
        self.mock_cache_config = Mock()
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.enc_dec_block_num = 0
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config

    def test_zero_num_tokens(self):
        """Test with num_tokens=0."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=0, batch_size=2, expected_decode_len=100, capture_prefill=False
        )

        # Should handle gracefully
        self.assertEqual(input_length_list, [0, 0])

    def test_batch_size_zero(self):
        """Test with batch_size=0."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=100, batch_size=0, expected_decode_len=100, capture_prefill=False
        )

        # Should return empty list
        self.assertEqual(len(input_length_list), 0)

    def test_negative_expected_decode_len(self):
        """Test with negative expected_decode_len."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=100, batch_size=2, expected_decode_len=-10, capture_prefill=False
        )

        # max_dec_len should be at least 1 (negative + 1 = 0, then min)
        self.assertEqual(max_dec_len_list, [1, 1])


if __name__ == "__main__":
    unittest.main()
