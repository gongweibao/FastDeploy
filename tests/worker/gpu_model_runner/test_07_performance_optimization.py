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
Tests for Performance Optimization

This module contains tests for performance optimization features in GPU Model Runner,
including CUDA Graph, Speculative Decoding, Chunked Prefill, and Warmup.

Key components tested:
- Speculative Decoding (Ngram/MTP)
- Chunked Prefill
- CUDA Graph padding
- SOT warmup
- Profile run
"""

import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestSpeculativeDecoding(unittest.TestCase):
    """Test cases for speculative decoding functionality."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.speculative_config = Mock()
        self.runner.speculative_config.num_speculative_tokens = 4
        self.runner.speculative_config.num_gpu_block_expand_ratio = 1
        self.runner.fd_config.speculative_config = self.runner.speculative_config
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.block_size = 16
        self.runner.cache_config.enc_dec_block_num = 0
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.share_inputs = Mock()
        self.runner.share_inputs["seq_lens_this_time_buffer"] = Mock()
        self.runner.speculative_decoding = False

    def test_speculative_decoding_disabled(self):
        """Test with speculative_decoding disabled."""
        self.runner.speculative_method = None

        self.runner._init_speculative_proposer()

        # Verify proposer is None
        self.assertIsNone(self.runner.proposer)

    def test_ngram_proposer_initialization(self):
        """Test NgramProposer initialization."""
        self.runner.speculative_method = "ngram"
        self.runner.local_rank = 0
        self.runner.device_id = 0

        with patch("fastdeploy.worker.gpu_model_runner.NgramProposer") as mock_ngram_proposer:
            mock_ngram_proposer.return_value = Mock()

            self.runner._init_speculative_proposer()

            # Verify NgramProposer is created
            mock_ngram_proposer.assert_called_once_with(self.runner.fd_config)
            self.assertIsNotNone(self.runner.proposer)

    def test_mtp_proposer_initialization(self):
        """Test MTPProposer initialization."""
        self.runner.speculative_method = "mtp"
        self.runner.local_rank = 0
        self.runner.device_id = 0

        mock_model = Mock()
        self.runner.get_model = Mock(return_value=mock_model)

        with patch("fastdeploy.worker.gpu_model_runner.MTPProposer") as mock_mtp_proposer:
            mock_mtp_proposer.return_value = Mock()

            self.runner._init_speculative_proposer()

            # Verify MTPProposer is created with correct parameters
            mock_mtp_proposer.assert_called_once_with(
                self.runner.fd_config, mock_model, 0, 0, self.runner.share_inputs
            )
            self.assertIsNotNone(self.runner.proposer)

    def test_mtp_kvcache_calculation(self):
        """Test KV cache calculation with MTP."""
        self.runner.speculative_config.num_gpu_block_expand_ratio = 2
        self.runner.speculative_method = "mtp"
        self.runner.model_config.head_dim = 128
        self.runner.model_config.kv_num_heads = 32
        self.runner.model_config.num_hidden_layers = 24

        result = self.runner.cal_theortical_kvcache()

        # With MTP, num_layers should be expanded
        # num_layers = 24 + 2 = 26
        expected = 2 * 2 * 16 * (128 * 32) * 26
        self.assertEqual(result, expected)

    def test_speculative_decode_in_execute_model_normal(self):
        """Test execute_model_normal with speculative_decoding."""
        self.runner.speculative_decoding = True
        self.runner.use_cudagraph = False

        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(model_forward_batch=[Mock()], num_running_requests=1)

                    # With speculative decoding, _save_model_output should NOT be called
                    mock_save.assert_not_called()

    def test_speculative_token_count_configuration(self):
        """Test with different num_speculative_tokens values."""
        for num_tokens in [1, 4, 8, 16]:
            self.runner.speculative_config.num_speculative_tokens = num_tokens

            # Verify configuration is applied
            self.assertEqual(self.runner.speculative_config.num_speculative_tokens, num_tokens)


class TestChunkedPrefill(unittest.TestCase):
    """Test cases for chunked prefill functionality."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.enable_chunked_prefill = False
        self.runner.cache_config.max_chunked_prefill_len = 4096
        self.runner.cache_config.block_size = 16
        self.runner.cache_config.enc_dec_block_num = 0
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.share_inputs = Mock()
        self.runner.restore_chunked_prefill_request = {}

    def test_get_input_length_list_capture_prefill(self):
        """Test get_input_length_list with capture_prefill=True."""
        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=160, batch_size=4, expected_decode_len=100, capture_prefill=True
        )

        # With capture_prefill=True, creates [1, 1, 1, 157] pattern
        self.assertEqual(input_length_list, [1, 1, 1, 157])
        self.assertEqual(len(input_length_list), 4)

    def test_large_prompt_chunked(self):
        """Test large prompt processed in chunks."""
        # Simulate processing a large prompt in chunks
        num_tokens = 10000
        batch_size = 4

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=num_tokens, batch_size=batch_size, expected_decode_len=100, capture_prefill=True
        )

        # Verify total tokens sum to num_tokens
        self.assertEqual(sum(input_length_list), num_tokens)
        self.assertEqual(len(input_length_list), batch_size)

    def test_chunked_prefill_state_continuity(self):
        """Test state continuity across chunks."""
        self.runner.restore_chunked_prefill_request = {}

        # Simulate chunked prefill with state restoration
        req_id = "req_1"
        chunk_state = {"prefill_start_index": 0, "prefill_end_index": 100, "prompt_token_ids": list(range(200))}

        self.runner.restore_chunked_prefill_request[req_id] = chunk_state

        # Verify state is stored
        self.assertIn(req_id, self.runner.restore_chunked_prefill_request)
        self.assertEqual(self.runner.restore_chunked_prefill_request[req_id]["prefill_start_index"], 0)
        self.assertEqual(self.runner.restore_chunked_prefill_request[req_id]["prefill_end_index"], 100)

    def test_chunked_prefill_with_pooling(self):
        """Test chunked prefill with pooling model."""
        self.runner.is_pooling_model = True
        self.runner.cache_config.enable_chunked_prefill = True

        # Get supported pooling tasks - encode should be removed
        self.runner.get_model = Mock()
        mock_model = Mock()
        mock_pooler = Mock()
        mock_pooler.get_supported_tasks = Mock(return_value=["encode", "cls"])
        mock_model.pooler = mock_pooler
        self.runner.get_model.return_value = mock_model

        result = self.runner.get_supported_pooling_tasks()

        # "encode" should be removed when chunked_prefill is enabled
        self.assertEqual(result, ["cls"])
        self.assertNotIn("encode", result)

    def test_max_chunked_prefill_len(self):
        """Test max_chunked_prefill_len constraint."""
        self.runner.cache_config.max_chunked_prefill_len = 512

        # Request tokens exceeding max chunked prefill length
        num_tokens = 1000
        batch_size = 1

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=num_tokens, batch_size=batch_size, expected_decode_len=100, capture_prefill=True
        )

        # Total should still equal num_tokens
        self.assertEqual(sum(input_length_list), num_tokens)


class TestPaddingCudagraphInputs(unittest.TestCase):
    """Test cases for padding_cudagraph_inputs method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.share_inputs = Mock()
        self.runner.share_inputs.pad_to_max_seq_len = Mock()
        self.runner.use_cudagraph = True

    def test_padding_cudagraph_inputs_basic(self):
        """Test padding_cudagraph_inputs calls padding function."""
        self.runner.padding_cudagraph_inputs()

        # Verify padding function is called
        self.runner.share_inputs.pad_to_max_seq_len.assert_called_once()

    def test_padding_cudagraph_inputs_without_cudagraph(self):
        """Test padding_cudagraph_inputs when use_cudagraph is False."""
        self.runner.use_cudagraph = False

        # Should not call padding when cudagraph is disabled
        self.runner.padding_cudagraph_inputs()

        # Verify padding function is NOT called
        self.runner.share_inputs.pad_to_max_seq_len.assert_not_called()


class TestSotWarmup(unittest.TestCase):
    """Test cases for sot_warmup method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.share_inputs = Mock()
        self.runner.share_inputs.update = Mock()
        self.runner.model = Mock()

    def test_sot_warmup_basic(self):
        """Test sot_warmup calls model warmup."""
        self.runner.sot_warmup()

        # Verify share_inputs is updated
        self.runner.share_inputs.update.assert_called()
        # Verify model warmup is called
        self.runner.model.run_warmup.assert_called_once()

    def test_sot_warmup_with_inputs(self):
        """Test sot_warmup with specific warmup inputs."""
        self.runner.sot_warmup(num_tokens=100)

        # Verify share_inputs is updated
        self.runner.share_inputs.update.assert_called()


class TestProfileRun(unittest.TestCase):
    """Test cases for profile_run method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.share_inputs = Mock()
        self.runner.clear_cache = Mock()
        self.runner.execute_model = Mock()
        self.runner.clear_parameters = Mock()
        self.runner.clear_requests = Mock()
        self.runner.forward_batch_reqs_list = []

    def test_profile_run_basic(self):
        """Test profile_run executes profile workflow."""
        self.runner.profile_run()

        # Verify profile workflow is executed
        self.runner.clear_cache.assert_called_once()
        self.runner.execute_model.assert_called_once()
        self.runner.clear_parameters.assert_called_once()
        self.runner.clear_requests.assert_called_once()

    def test_profile_run_with_prefill(self):
        """Test profile_run with prefill requests."""
        self.runner.forward_batch_reqs_list = [Mock()]

        self.runner.profile_run()

        # Verify all profile steps are executed
        self.runner.clear_cache.assert_called_once()
        self.runner.execute_model.assert_called_once()
        self.runner.clear_parameters.assert_called_once()


if __name__ == "__main__":
    unittest.main()
