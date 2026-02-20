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

"""Unit tests for public methods of GPUModelRunner."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import paddle

from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestGetInputLengthList(unittest.TestCase):
    """Test cases for get_input_length_list method."""

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
        self.mock_fd_config.parallel_config.enable_expert_parallel = True

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=1000, batch_size=2, expected_decode_len=100, capture_prefill=False
        )

        # input_length should be limited to 32
        self.assertEqual(input_length_list, [32, 32])

    def test_max_model_len_constraint(self):
        """Test when max_model_len constrains the input_length."""
        self.mock_model_config.max_model_len = 150

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=1000, batch_size=2, expected_decode_len=80, capture_prefill=False
        )

        # input_length = min(500, 150-81) = min(500, 69) = 69
        self.assertEqual(input_length_list, [69, 69])
        self.assertEqual(max_dec_len_list, [81, 81])

    def test_enc_dec_block_num(self):
        """Test block_num calculation with enc_dec_block_num."""
        self.mock_cache_config.enc_dec_block_num = 5

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
        self.mock_cache_config.block_size = 1

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=160, batch_size=2, expected_decode_len=100, capture_prefill=False
        )

        # block_num = (80 + 1 - 1) // 1 + 0 = 80
        self.assertEqual(block_num, 80)

    def test_multiple_constraints_simultaneously(self):
        """Test with expert_parallel and max_model_len constraints both active."""
        self.mock_fd_config.parallel_config.enable_expert_parallel = True
        self.mock_model_config.max_model_len = 50

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=1000, batch_size=2, expected_decode_len=10, capture_prefill=False
        )

        # input_length = min(500, 50-11) = min(500, 39) = 39
        # then limited by expert_parallel: min(39, 32) = 32
        self.assertEqual(input_length_list, [32, 32])


class TestGetSupportedPoolingTasks(unittest.TestCase):
    """Test cases for get_supported_pooling_tasks method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_fd_config.cache_config = Mock()
        self.mock_fd_config.cache_config.enable_chunked_prefill = False

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.cache_config = self.mock_fd_config

    def test_non_pooling_model(self):
        """Test returns empty list for non-pooling model."""
        self.runner.is_pooling_model = False

        result = self.runner.get_supported_pooling_tasks()

        self.assertEqual(result, [])

    def test_pooling_model_with_tasks(self):
        """Test returns tasks from pooler for pooling model."""
        self.runner.is_pooling_model = True
        self.runner.get_model = Mock()
        mock_model = Mock()
        mock_pooler = Mock()
        mock_pooler.get_supported_tasks = Mock(return_value=["encode", "cls"])
        mock_model.pooler = mock_pooler
        self.runner.get_model.return_value = mock_model

        result = self.runner.get_supported_pooling_tasks()

        self.assertEqual(result, ["encode", "cls"])
        mock_pooler.get_supported_tasks.assert_called_once()

    def test_pooling_model_removes_encode_when_chunked_prefill(self):
        """Test removes 'encode' task when chunked_prefill is enabled."""
        self.mock_fd_config.cache_config.enable_chunked_prefill = True
        self.runner.is_pooling_model = True
        self.runner.get_model = Mock()
        mock_model = Mock()
        mock_pooler = Mock()
        mock_pooler.get_supported_tasks = Mock(return_value=["encode", "cls"])
        mock_model.pooler = mock_pooler
        self.runner.get_model.return_value = mock_model

        result = self.runner.get_supported_pooling_tasks()

        self.assertEqual(result, ["cls"])
        self.assertNotIn("encode", result)

    def test_pooling_model_chunked_prefill_no_encode_task(self):
        """Test when chunked_prefill is enabled but no encode task exists."""
        self.mock_fd_config.cache_config.enable_chunked_prefill = True
        self.runner.is_pooling_model = True
        self.runner.get_model = Mock()
        mock_model = Mock()
        mock_pooler = Mock()
        mock_pooler.get_supported_tasks = Mock(return_value=["cls", "mean"])
        mock_model.pooler = mock_pooler
        self.runner.get_model.return_value = mock_model

        result = self.runner.get_supported_pooling_tasks()

        self.assertEqual(result, ["cls", "mean"])

    def test_pooling_model_empty_tasks(self):
        """Test when pooler returns empty tasks."""
        self.runner.is_pooling_model = True
        self.runner.get_model = Mock()
        mock_model = Mock()
        mock_pooler = Mock()
        mock_pooler.get_supported_tasks = Mock(return_value=[])
        mock_model.pooler = mock_pooler
        self.runner.get_model.return_value = mock_model

        result = self.runner.get_supported_pooling_tasks()

        self.assertEqual(result, [])


class TestExistPrefillDecode(unittest.TestCase):
    """Test cases for exist_prefill and exist_decode methods."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_scheduler_config = Mock()
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_fd_config.model_config = Mock()
        self.mock_fd_config.model_config.enable_mm = False

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.model_config = self.mock_fd_config.model_config

    def test_exist_prefill_true(self):
        """Test exist_prefill returns True when seq_lens_encoder has positive values."""
        self.runner.share_inputs = {"seq_lens_encoder": paddle.to_tensor([0, 10, 0, 5])}

        result = self.runner.exist_prefill()

        self.assertTrue(result)

    def test_exist_prefill_false(self):
        """Test exist_prefill returns False when seq_lens_encoder has no positive values."""
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
        self.runner.share_inputs = {"seq_lens_encoder": paddle.to_tensor([10, 20, 30])}

        result = self.runner.exist_prefill()

        self.assertTrue(result)

    def test_exist_decode_all_positive(self):
        """Test exist_decode returns True when all values are positive."""
        self.runner.share_inputs = {"seq_lens_decoder": paddle.to_tensor([1, 5, 10])}

        result = self.runner.exist_decode()

        self.assertTrue(result)

    def test_exist_prefill_mixed_values(self):
        """Test exist_prefill returns True with mixed positive/negative values (should handle as if negative)."""
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

        from fastdeploy.engine.request import RequestType

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.cache_config = Mock()
        self.runner.share_inputs = Mock()
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.exist_prefill_flag = False
        self.runner.pooling_params = []
        self.runner.sampler = Mock()
        self.runner.routing_replay_manager = Mock()
        self.RequestType = RequestType

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
                "pre_ids": np.full((10, 1), -1, dtype="int32"),
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

        return share_inputs

    def test_insert_tasks_v1_with_prefill_task(self):
        """Test insert_tasks_v1 with a prefill task."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        req = self._create_mock_request(
            task_type=self.RequestType.PREFILL.value,
            idx=0,
            temperature=0.9,
            top_p=0.8,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify prefill flag is set
        self.assertTrue(self.runner.exist_prefill_flag)
        # Verify seq_lens_encoder is updated
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_insert_tasks_v1_with_decode_task(self):
        """Test insert_tasks_v1 with a decode task."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        req = self._create_mock_request(
            task_type=self.RequestType.DECODE.value,
            idx=0,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is handled
        self.assertEqual(self.runner.share_inputs._num_running_requests, 1)
        self.assertEqual(self.runner.share_inputs._running_requests_ids, range(1))

    def test_insert_tasks_v1_with_preempted_task(self):
        """Test insert_tasks_v1 with a preempted task."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Task type that's not PREFILL or DECODE (treated as preempted)
        req = self._create_mock_request(
            task_type=99,  # Invalid task type
            idx=0,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify forward_batch_reqs_list is cleared for preempted task
        self.assertIsNone(self.runner.forward_batch_reqs_list[0])
        # Verify prefill flag is not set
        self.assertFalse(self.runner.exist_prefill_flag)

    def test_insert_tasks_v1_mixed_task_types(self):
        """Test insert_tasks_v1 with mixed prefill and decode tasks."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        req1 = self._create_mock_request(
            task_type=self.RequestType.PREFILL.value,
            idx=0,
        )
        req2 = self._create_mock_request(
            task_type=self.RequestType.DECODE.value,
            idx=1,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)

        self.runner.insert_tasks_v1([req1, req2], num_running_requests=2)

        # Verify both requests are processed
        self.assertEqual(self.runner.share_inputs._num_running_requests, 2)
        self.assertEqual(self.runner.share_inputs._running_requests_ids, range(2))

    def test_insert_tasks_v1_with_sampling_params(self):
        """Test insert_tasks_v1 with custom sampling parameters."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        req = self._create_mock_request(
            task_type=self.RequestType.PREFILL.value,
            idx=0,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            min_tokens=5,
            max_tokens=200,
            seed=42,
        )
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify sampling params are set (via mock)
        self.assertEqual(self.runner.share_inputs._num_running_requests, 1)


class TestOnlyPrefill(unittest.TestCase):
    """Test cases for only_prefill method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.splitwise_role = "decoder"
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.use_ep = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.scheduler_config = self.mock_scheduler_config
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
        self.mock_parallel_config.use_ep = True
        self.mock_scheduler_config.splitwise_role = "mixed"
        self.runner.exist_decode = Mock(return_value=False)

        # Mock paddle.distributed.all_gather_object
        with patch("paddle.distributed.all_gather_object") as mock_all_gather:
            mock_all_gather.return_value = None
            result = self.runner.only_prefill()

            self.assertTrue(result)


class TestOnlyDecode(unittest.TestCase):
    """Test cases for only_decode method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.splitwise_role = "decoder"
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.use_ep = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.scheduler_config = self.mock_scheduler_config
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
        self.mock_parallel_config.use_ep = True
        self.mock_scheduler_config.splitwise_role = "mixed"
        self.runner.exist_prefill = Mock(return_value=False)

        # Mock paddle.distributed.all_gather_object
        with patch("paddle.distributed.all_gather_object") as mock_all_gather:
            mock_all_gather.return_value = None
            result = self.runner.only_decode()

            self.assertTrue(result)


class TestCalTheorticalKVCache(unittest.TestCase):
    """Test cases for cal_theortical_kvcache method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.head_dim = 128
        self.mock_model_config.kv_num_heads = 32
        self.mock_model_config.num_hidden_layers = 24
        self.mock_cache_config = Mock()
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.use_mla_cache = False
        self.mock_quant_config = Mock()
        self.mock_quant_config.kv_cache_quant_type = None
        self.mock_speculative_config = Mock()
        self.mock_speculative_config.num_gpu_block_expand_ratio = 0

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.quant_config = self.mock_quant_config
        self.runner.speculative_config = self.mock_speculative_config
        self.runner.speculative_method = None
        self.mock_fd_config.cache_config = self.mock_cache_config
        self.mock_fd_config.model_config = self.mock_model_config

    def test_cal_theortical_kvcache_default_dtype(self):
        """Test with default dtype (bf16, 2 bytes)."""
        result = self.runner.cal_theortical_kvcache()

        # byte_of_dtype = 2, hidden_dim = 128 * 32 = 4096
        # num_layers = 24, block_size = 16
        # required_memory = 2 * 2 * 16 * 4096 * 24 = 6291456 bytes
        expected = 2 * 2 * 16 * (128 * 32) * 24
        self.assertEqual(result, expected)

    def test_cal_theortical_kvcache_int8_dtype(self):
        """Test with int8 dtype (1 byte)."""
        self.mock_quant_config.kv_cache_quant_type = "int8"

        result = self.runner.cal_theortical_kvcache()

        # byte_of_dtype = 1
        expected = 1 * 2 * 16 * (128 * 32) * 24
        self.assertEqual(result, expected)

    def test_cal_theortical_kvcache_with_mtp(self):
        """Test with MTP speculative method."""
        self.runner.speculative_method = "mtp"
        self.mock_speculative_config.num_gpu_block_expand_ratio = 2

        result = self.runner.cal_theortical_kvcache()

        # num_layers = 24 + 2 = 26
        expected = 2 * 2 * 16 * (128 * 32) * 26
        self.assertEqual(result, expected)

    def test_cal_theortical_kvcache_with_mla(self):
        """Test with MLA cache."""
        self.mock_cache_config.use_mla_cache = True
        self.mock_model_config.kv_lora_rank = 64
        self.mock_model_config.qk_rope_head_dim = 64

        result = self.runner.cal_theortical_kvcache()

        # MLA: compress_kv + k_pe = (64 + 64) * 16 * num_layers
        expected = 2 * (64 + 64) * 16 * 26
        self.assertEqual(result, expected)


class TestClearRequests(unittest.TestCase):
    """Test cases for clear_requests method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 4
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_routing_replay_config = Mock()
        self.mock_routing_replay_config.enable_routing_replay = False
        self.mock_fd_config.routing_replay_config = self.mock_routing_replay_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.share_inputs = {"stop_flags": np.array([False, False, False, False])}
        self.runner.prompt_logprobs_reqs = {"req1": "data1", "req2": "data2"}
        self.runner.in_progress_prompt_logprobs = {"req3": "data3"}
        self.runner.forward_batch_reqs_list = [Mock(), Mock(), Mock(), Mock()]
        self.runner.routing_replay_manager = Mock()

    def test_clear_requests_resets_state(self):
        """Test clear_requests resets all request-related state."""
        self.runner.clear_requests()

        # Verify stop_flags are all True
        self.assertTrue(all(self.runner.share_inputs["stop_flags"]))

        # Verify prompt logprobs are cleared
        self.assertEqual(self.runner.prompt_logprobs_reqs, {})
        self.assertEqual(self.runner.in_progress_prompt_logprobs, {})

        # Verify forward_batch_reqs_list is reset
        expected = [None] * 4
        self.assertEqual(self.runner.forward_batch_reqs_list, expected)

    def test_clear_requests_with_routing_replay(self):
        """Test clear_requests with routing replay enabled."""
        self.mock_routing_replay_config.enable_routing_replay = True
        self.runner.clear_requests()

        # Verify routing_replay_manager.put_table_to_store is called
        self.runner.routing_replay_manager.put_table_to_store.assert_called_once()


class TestUpdateWeights(unittest.TestCase):
    """Test cases for update_weights method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.shutdown_comm_group_if_worker_idle = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.dynamic_weight_manager = Mock()

    def test_update_weights_default_params(self):
        """Test update_weights with default parameters."""
        self.runner.update_weights()

        self.runner.dynamic_weight_manager.update_weights_by_rdma.assert_called_once_with(None, None)

    def test_update_weights_with_version(self):
        """Test update_weights with version parameter."""
        self.runner.update_weights(version="v1.0.0")

        self.runner.dynamic_weight_manager.update_weights_by_rdma.assert_called_once_with("v1.0.0", None)

    def test_update_weights_with_rsync_config(self):
        """Test update_weights with rsync_config."""
        rsync_config = {"source": "path1", "destination": "path2"}
        self.runner.update_weights(rsync_config=rsync_config)

        self.runner.dynamic_weight_manager.update_weights_by_rdma.assert_called_once_with(None, rsync_config)


class TestGetPromptLogprobsList(unittest.TestCase):
    """Test cases for _get_prompt_logprobs_list method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.logprobs_mode = "raw_logprobs"
        self.mock_cache_config = Mock()
        self.mock_cache_config.enable_prefix_caching = False
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 10

        self.mock_fd_config.model_config = self.mock_model_config
        self.mock_fd_config.cache_config = self.mock_cache_config
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.ori_vocab_size = 50000
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.share_inputs = Mock()

    def _create_mock_request(
        self, request_id="req_1", prompt_tokens=10, num_logprobs=5, prefill_start=0, prefill_end=10
    ):
        """Helper to create mock request."""
        request = Mock()
        request.request_id = request_id
        request.idx = 0
        request.prompt_token_ids = list(range(prompt_tokens))
        request.prefill_start_index = prefill_start
        request.prefill_end_index = prefill_end
        request.sampling_params = Mock()
        request.sampling_params.prompt_logprobs = num_logprobs
        return request

    def test_prompt_logprobs_empty_requests(self):
        """Test with no pending prompt logprobs requests."""
        hidden_states = paddle.zeros((10, 768))

        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # Should return list of None values
        self.assertEqual(len(result), 10)
        self.assertTrue(all(x is None for x in result))

    def test_prompt_logprobs_all_tokens(self):
        """Test with num_prompt_logprobs=-1 (all tokens)."""
        request = self._create_mock_request(num_logprobs=-1, prompt_tokens=10)
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_model = Mock()
        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((1, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((1, 1)), paddle.zeros((1, 1)), paddle.zeros((1,)))
        )
        self.runner.model = mock_model
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((1, 768)))

        hidden_states = paddle.zeros((20, 768))

        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # num_prompt_logprobs should be set to ori_vocab_size (50000)
        self.assertEqual(request.sampling_params.prompt_logprobs, 50000)
        # Result should have the request data
        self.assertIsNotNone(result[0])

    def test_prompt_logprobs_partial(self):
        """Test with num_prompt_logprobs < num_tokens."""
        request = self._create_mock_request(num_logprobs=5, prompt_tokens=10)
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_model = Mock()
        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((1, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((1, 6)), paddle.zeros((1, 6)), paddle.zeros((1,)))
        )
        self.runner.model = mock_model
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((1, 768)))

        hidden_states = paddle.zeros((20, 768))

        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # Result should have the request data
        self.assertIsNotNone(result[0])

    def test_prompt_logprobs_chunked_prefill(self):
        """Test prompt logprobs with chunked prefill."""
        # First chunk
        request1 = self._create_mock_request(
            request_id="req_1", prompt_tokens=20, num_logprobs=5, prefill_start=0, prefill_end=10  # First chunk
        )
        self.runner.prompt_logprobs_reqs = {"req_1": request1}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_model = Mock()
        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((10, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((10, 6)), paddle.zeros((10, 6)), paddle.zeros((10,)))
        )
        self.runner.model = mock_model
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((10, 768)))

        hidden_states = paddle.zeros((20, 768))

        # First chunk - should not return logprobs yet
        result = self.runner._get_prompt_logprobs_list(hidden_states)
        self.assertIsNone(result[0])

        # Verify in_progress_logprobs is created
        self.assertIn("req_1", self.runner.in_progress_prompt_logprobs)

    def test_prompt_logprobs_completion(self):
        """Test logprobs completion on last chunk."""
        # Last chunk
        request = self._create_mock_request(
            request_id="req_1", prompt_tokens=10, num_logprobs=5, prefill_start=0, prefill_end=10  # Complete
        )
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_model = Mock()
        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((9, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((9, 6)), paddle.zeros((9, 6)), paddle.zeros((9,)))
        )
        self.runner.model = mock_model
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((9, 768)))

        hidden_states = paddle.zeros((20, 768))

        # Complete request - should return logprobs
        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # Verify result is not None
        self.assertIsNotNone(result[0])

        # Verify request is removed from pending
        self.assertNotIn("req_1", self.runner.prompt_logprobs_reqs)
        self.assertNotIn("req_1", self.runner.in_progress_prompt_logprobs)

    def test_prompt_logprobs_raw_logits_mode(self):
        """Test with logprobs_mode='raw_logits'."""
        self.mock_model_config.logprobs_mode = "raw_logits"

        request = self._create_mock_request(num_logprobs=5, prompt_tokens=10)
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_model = Mock()
        mock_sampler = Mock()
        # For raw_logits mode, gather_logprobs is called directly with logits
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((9, 6)), paddle.zeros((9, 6)), paddle.zeros((9,)))
        )
        self.runner.model = mock_model
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((9, 768)))

        hidden_states = paddle.zeros((20, 768))

        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # Verify compute_logprobs is NOT called in raw_logits mode
        mock_sampler.compute_logprobs.assert_not_called()
        self.assertIsNotNone(result[0])

    def test_prompt_logprobs_with_prefix_caching_error(self):
        """Test that prefix caching is disabled with prompt_logprobs."""
        self.mock_cache_config.enable_prefix_caching = True

        request = self._create_mock_request()
        self.runner.prompt_logprobs_reqs = {"req_1": request}

        hidden_states = paddle.zeros((10, 768))

        # Should raise AssertionError
        with self.assertRaises(AssertionError):
            self.runner._get_prompt_logprobs_list(hidden_states)

    def test_prompt_logprobs_none_prompt_tokens(self):
        """Test with None prompt_token_ids."""
        request = self._create_mock_request(prompt_tokens=None)
        request.prompt_token_ids = None
        self.runner.prompt_logprobs_reqs = {"req_1": request}

        hidden_states = paddle.zeros((10, 768))

        # Should handle gracefully without error
        result = self.runner._get_prompt_logprobs_list(hidden_states)
        self.assertIsNone(result[0])

    def test_prompt_logprobs_none_num_logprobs(self):
        """Test with None num_prompt_logprobs."""
        request = self._create_mock_request(num_logprobs=None)
        request.sampling_params.prompt_logprobs = None
        self.runner.prompt_logprobs_reqs = {"req_1": request}

        hidden_states = paddle.zeros((10, 768))

        # Should handle gracefully without error
        result = self.runner._get_prompt_logprobs_list(hidden_states)
        self.assertIsNone(result[0])

    def test_prompt_logprobs_in_progress_accumulation(self):
        """Test logprobs accumulation across multiple chunks."""
        # Simulate accumulated logprobs from previous chunk
        from fastdeploy.worker.output import LogprobsTensors

        existing_logprobs = LogprobsTensors.empty_cpu(19, 6)  # 20 tokens - 1, 6 logprobs
        self.runner.in_progress_prompt_logprobs = {"req_1": existing_logprobs}

        # Final chunk
        request = self._create_mock_request(
            request_id="req_1", prompt_tokens=20, num_logprobs=5, prefill_start=10, prefill_end=20
        )
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_model = Mock()
        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((10, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((10, 6)), paddle.zeros((10, 6)), paddle.zeros((10,)))
        )
        self.runner.model = mock_model
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((10, 768)))

        hidden_states = paddle.zeros((20, 768))

        # Should use existing in_progress_logprobs and complete
        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # Verify result is not None
        self.assertIsNotNone(result[0])


if __name__ == "__main__":
    unittest.main()
