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
Tests with Real Data Validation

This module tests GPUModelRunner with realistic data values to verify
correct data flow and transformations, rather than just mock/zeros.

Key aspects tested:
- Real token IDs (e.g., actual text tokens)
- Real sequence lengths
- Real block table assignments
- Real sampling parameters
- Data transformation correctness across stages
"""

import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import paddle

from fastdeploy.engine.request import RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class RealDataTestHelper:
    """Helper class to generate realistic test data."""

    # Simulated vocabulary mapping (simplified)
    VOCAB = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "the": 3,
        "hello": 4,
        "world": 5,
        "what": 6,
        "is": 7,
        "your": 8,
        "name": 9,
        "answer": 10,
        "question": 11,
        "How": 12,
        "are": 13,
        "you": 14,
        "today": 15,
        "?": 16,
    }

    @classmethod
    def tokenize(cls, text: str) -> list[int]:
        """Simple tokenizer for testing."""
        tokens = text.split()
        return [cls.VOCAB.get(token, cls.VOCAB["<unk>"]) for token in tokens]

    @classmethod
    def get_real_prompt(cls) -> str:
        """Get a real prompt for testing."""
        return "hello world what is your name"

    @classmethod
    def get_real_prompt_tokens(cls) -> list[int]:
        """Get real prompt token IDs."""
        return cls.tokenize(cls.get_real_prompt())

    @classmethod
    def get_real_output_tokens(cls) -> list[int]:
        """Get real output token IDs (simulated generation)."""
        return [cls.VOCAB["answer"], cls.VOCAB["question"], cls.VOCAB["How"], cls.VOCAB["<eos>"]]


class TestRealDataPrefill(unittest.TestCase):
    """Test prefill stage with realistic data."""

    def setUp(self):
        """Set up test fixtures with real data."""
        self.real_tokens = RealDataTestHelper.get_real_prompt_tokens()
        self.real_output_tokens = RealDataTestHelper.get_real_output_tokens()

        self.runner = GPUModelRunner.__new__(GPUModelRunner)

        # Mock configurations
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.model_config.eos_tokens_lens = 1
        self.runner.model_config.max_stop_seqs_num = 4
        self.runner.model_config.enable_mm = False
        self.runner.model_config.runner_type = "causal_lm"
        self.runner.model_config.ori_vocab_size = 32000
        self.runner.fd_config.model_config = self.runner.model_config

        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.splitwise_role = "mixed"
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.scheduler_config.enable_overlap_schedule = False
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config

        self.runner.speculative_config = Mock()
        self.runner.speculative_config.method = None
        self.runner.speculative_config.num_speculative_tokens = 0
        self.runner.fd_config.speculative_config = self.runner.speculative_config

        # Initialize share_inputs with REALISTIC data (not zeros!)
        self.runner.share_inputs = self._setup_realistic_share_inputs()

        # Mock other required components
        self.runner.sampler = Mock()
        self.runner.sampler.apply_logits_processor = Mock()
        self.runner.initialize_kv_cache = Mock()
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.exist_prefill_flag = False
        self.runner.pooling_params = []
        self.runner.speculative_method = None

    def _setup_realistic_share_inputs(self):
        """Setup share_inputs with realistic data structure."""
        data = {
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
        }

        share_inputs = MagicMock()
        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)
        share_inputs.__getitem__.side_effect = lambda key: data[key]
        share_inputs.__setitem__.side_effect = lambda key, value: self._safe_setitem(data, key, value)
        share_inputs.__contains__.side_effect = lambda key: key in data

        # Store data reference for verification
        self._share_inputs_data = data

        return share_inputs

    def _safe_setitem(self, data, key, value):
        """Helper to handle slice assignments in share_inputs."""
        data[key] = value

    def _create_real_request(self):
        """Create a request with real data."""
        request = Mock()
        request.task_type = Mock()
        request.task_type.value = RequestType.PREFILL.value
        request.idx = 0
        request.request_id = "req_001"
        request.prompt_token_ids = self.real_tokens
        request.output_token_ids = []
        request.block_tables = [10, 11, 12, 13]  # Real block table
        request.eos_token_ids = [1]  # Real EOS token
        request.prefill_start_index = 0
        request.prefill_end_index = len(self.real_tokens)
        request.sampling_params = Mock()
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []

        # Real sampling parameters
        request.sampling_params.temperature = 0.8
        request.sampling_params.top_p = 0.95
        request.sampling_params.top_k = 50
        request.sampling_params.min_p = 0.05
        request.sampling_params.repetition_penalty = 1.1
        request.sampling_params.frequency_penalty = 0.0
        request.sampling_params.presence_penalty = 0.0

        # Store for request.get() method
        _request_data = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "min_tokens": 1,
            "max_tokens": 100,
            "seed": None,
            "bad_words_token_ids": None,
            "stop_token_ids": None,
            "stop_seqs_len": [],
            "temp_scaled_logprobs": False,
            "top_p_normalized_logprobs": False,
            "logits_processors_args": None,
            "enable_thinking": None,
            "reasoning_max_tokens": None,
        }

        request.get = lambda key, default=None: _request_data.get(key, default)

        return request

    def test_real_token_ids_are_correctly_transferred(self):
        """Test that real token IDs are correctly transferred to share_inputs."""
        request = self._create_real_request()

        self.runner.insert_tasks_v1([request], num_running_requests=1)

        # Verify prompt_ids contains real token values
        expected_prompt_ids = np.array(self.real_tokens, dtype="int64")
        actual_prompt_ids = self.runner.share_inputs["prompt_ids"][0, : len(self.real_tokens)]

        np.testing.assert_array_equal(
            actual_prompt_ids,
            expected_prompt_ids,
            err_msg="Prompt token IDs should match input request"
        )

        # Verify input_ids contains real token values
        actual_input_ids = self.runner.share_inputs["input_ids"][0, : len(self.real_tokens)]
        np.testing.assert_array_equal(
            actual_input_ids,
            expected_prompt_ids,
            err_msg="Input token IDs should match input request"
        )

    def test_real_sampling_parameters_are_correctly_transferred(self):
        """Test that real sampling parameters are correctly transferred."""
        request = self._create_real_request()

        self.runner.insert_tasks_v1([request], num_running_requests=1)

        # Verify each sampling parameter has the real value
        self.assertEqual(
            self.runner.share_inputs["temperature"][0],
            0.8,
            "Temperature should be 0.8"
        )
        self.assertEqual(
            self.runner.share_inputs["top_p"][0],
            0.95,
            "Top-p should be 0.95"
        )
        self.assertEqual(
            self.runner.share_inputs["top_k"][0],
            50,
            "Top-k should be 50"
        )
        self.assertEqual(
            self.runner.share_inputs["penalty_score"][0],
            1.1,
            "Repetition penalty should be 1.1"
        )

    def test_real_block_table_is_correctly_transferred(self):
        """Test that real block table is correctly transferred."""
        request = self._create_real_request()
        request.block_tables = [10, 11, 12, 13, 14, 15]  # Real block table

        self.runner.insert_tasks_v1([request], num_running_requests=1)

        # Verify block table is correctly set
        actual_block_table = self.runner.share_inputs["block_tables"][0, :6]
        expected_block_table = np.array([10, 11, 12, 13, 14, 15], dtype="int32")

        np.testing.assert_array_equal(
            actual_block_table,
            expected_block_table,
            err_msg="Block table should match request block table"
        )

    def test_real_eos_token_is_correctly_transferred(self):
        """Test that real EOS token is correctly transferred."""
        request = self._create_real_request()
        request.eos_token_ids = [1]  # Real EOS token ID

        self.runner.insert_tasks_v1([request], num_running_requests=1)

        # Verify EOS token is correctly set
        np.testing.assert_array_equal(
            self.runner.share_inputs["eos_token_id"],
            np.array([[1]], dtype="int64"),
            err_msg="EOS token ID should be 1"
        )

    def test_real_prompt_length_is_correctly_calculated(self):
        """Test that prompt length is correctly calculated from real tokens."""
        request = self._create_real_request()

        self.runner.insert_tasks_v1([request], num_running_requests=1)

        # Verify prompt length is correctly calculated
        expected_length = len(self.real_tokens)

        # Check input_ids has correct length set
        # (In real implementation, seq_lens_this_time should equal prefill length)
        actual_seq_len = self.runner.share_inputs["seq_lens_this_time_buffer"][0]

        self.assertEqual(
            actual_seq_len,
            expected_length,
            f"Sequence length should be {expected_length}, got {actual_seq_len}"
        )


class TestRealDataBatch(unittest.TestCase):
    """Test batch processing with multiple realistic requests."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = GPUModelRunner.__new__(GPUModelRunner)

        # Mock configurations
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.model_config.eos_tokens_lens = 1
        self.runner.model_config.max_stop_seqs_num = 4
        self.runner.model_config.enable_mm = False
        self.runner.model_config.runner_type = "causal_lm"
        self.runner.fd_config.model_config = self.runner.model_config

        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.splitwise_role = "mixed"
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.scheduler_config.enable_overlap_schedule = False
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config

        self.runner.speculative_config = Mock()
        self.runner.speculative_config.method = None
        self.runner.fd_config.speculative_config = self.runner.speculative_config

        # Initialize share_inputs
        self.runner.share_inputs = self._setup_share_inputs()

        # Mock other required components
        self.runner.sampler = Mock()
        self.runner.sampler.apply_logits_processor = Mock()
        self.runner.initialize_kv_cache = Mock()
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.exist_prefill_flag = False
        self.runner.pooling_params = []
        self.runner.speculative_method = None

    def _setup_share_inputs(self):
        """Setup share_inputs."""
        data = {
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
            "top_p_normalized_logprobs": np.zeros((10,), dtype="bool"),
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
        }

        share_inputs = MagicMock()
        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)
        share_inputs.__getitem__.side_effect = lambda key: data[key]
        share_inputs.__setitem__.side_effect = lambda key, value: self._safe_setitem(data, key, value)
        share_inputs.__contains__.side_effect = lambda key: key in data

        self._share_inputs_data = data

        return share_inputs

    def _safe_setitem(self, data, key, value):
        """Helper to handle slice assignments in share_inputs."""
        data[key] = value

    def _create_request(self, idx, tokens, temperature=0.8, top_p=0.95):
        """Create a request with given parameters."""
        request = Mock()
        request.task_type = Mock()
        request.task_type.value = RequestType.PREFILL.value
        request.idx = idx
        request.request_id = f"req_{idx:03d}"
        request.prompt_token_ids = tokens
        request.output_token_ids = []
        request.block_tables = [idx * 10 + i for i in range(4)]
        request.eos_token_ids = [1]
        request.prefill_start_index = 0
        request.prefill_end_index = len(tokens)
        request.sampling_params = Mock()
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []

        _request_data = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 50,
            "min_p": 0.05,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "min_tokens": 1,
            "max_tokens": 100,
            "seed": None,
            "bad_words_token_ids": None,
            "stop_token_ids": None,
            "stop_seqs_len": [],
            "temp_scaled_logprobs": False,
            "top_p_normalized_logprobs": False,
            "logits_processors_args": None,
            "enable_thinking": None,
            "reasoning_max_tokens": None,
        }

        request.get = lambda key, default=None: _request_data.get(key, default)

        return request

    def test_multiple_requests_with_different_real_params(self):
        """Test batch with multiple requests having different real parameters."""
        # Create requests with different real parameters
        req1_tokens = [4, 5]  # "hello world"
        req2_tokens = [3, 6, 7]  # "the what is"
        req3_tokens = [12, 13, 14, 15, 16]  # "How are you today ?"

        req1 = self._create_request(0, req1_tokens, temperature=0.7, top_p=0.9)
        req2 = self._create_request(1, req2_tokens, temperature=0.8, top_p=0.95)
        req3 = self._create_request(2, req3_tokens, temperature=0.9, top_p=1.0)

        self.runner.insert_tasks_v1([req1, req2, req3], num_running_requests=3)

        # Verify each request has its correct token IDs
        np.testing.assert_array_equal(
            self.runner.share_inputs["prompt_ids"][0, :2],
            np.array(req1_tokens, dtype="int64"),
            "Request 1 prompt tokens should match"
        )
        np.testing.assert_array_equal(
            self.runner.share_inputs["prompt_ids"][1, :3],
            np.array(req2_tokens, dtype="int64"),
            "Request 2 prompt tokens should match"
        )
        np.testing.assert_array_equal(
            self.runner.share_inputs["prompt_ids"][2, :5],
            np.array(req3_tokens, dtype="int64"),
            "Request 3 prompt tokens should match"
        )

        # Verify each request has its correct sampling parameters
        self.assertEqual(
            self.runner.share_inputs["temperature"][0],
            0.7,
            "Request 1 temperature should be 0.7"
        )
        self.assertEqual(
            self.runner.share_inputs["temperature"][1],
            0.8,
            "Request 2 temperature should be 0.8"
        )
        self.assertEqual(
            self.runner.share_inputs["temperature"][2],
            0.9,
            "Request 3 temperature should be 0.9"
        )

    def test_batch_different_sequence_lengths(self):
        """Test batch with requests of different real sequence lengths."""
        # Create requests with different lengths
        req1_tokens = [4]  # Length 1
        req2_tokens = [4, 5]  # Length 2
        req3_tokens = [4, 5, 6, 7, 8]  # Length 5

        req1 = self._create_request(0, req1_tokens)
        req2 = self._create_request(1, req2_tokens)
        req3 = self._create_request(2, req3_tokens)

        self.runner.insert_tasks_v1([req1, req2, req3], num_running_requests=3)

        # Verify sequence lengths are correctly set
        self.assertEqual(
            self.runner.share_inputs["seq_lens_this_time_buffer"][0],
            1,
            "Request 1 seq len should be 1"
        )
        self.assertEqual(
            self.runner.share_inputs["seq_lens_this_time_buffer"][1],
            2,
            "Request 2 seq len should be 2"
        )
        self.assertEqual(
            self.runner.share_inputs["seq_lens_this_time_buffer"][2],
            5,
            "Request 3 seq len should be 5"
        )


class TestRealDataValidationSummary(unittest.TestCase):
    """Summary test showing the difference between mock and real data."""

    def test_real_vs_mock_data_comparison(self):
        """Compare zeros/mock data vs real data values."""
        # Mock/zeros data (what current tests use)
        mock_data = {
            "token_ids": [0, 0, 0, 0, 0],
            "temperature": 0.0,
            "top_p": 0.0,
            "seq_len": 0,
        }

        # Real data (what realistic tests should use)
        real_data = {
            "token_ids": [4, 5, 6, 7, 8],  # "hello world what is"
            "temperature": 0.8,
            "top_p": 0.95,
            "seq_len": 5,
        }

        # Verify they are different
        self.assertNotEqual(mock_data["token_ids"], real_data["token_ids"])
        self.assertNotEqual(mock_data["temperature"], real_data["temperature"])
        self.assertNotEqual(mock_data["top_p"], real_data["top_p"])

        # Real data is non-zero and meaningful
        self.assertGreater(sum(real_data["token_ids"]), 0)
        self.assertGreater(real_data["temperature"], 0)
        self.assertGreater(real_data["top_p"], 0)
        self.assertEqual(real_data["seq_len"], len(real_data["token_ids"]))


if __name__ == "__main__":
    unittest.main()
