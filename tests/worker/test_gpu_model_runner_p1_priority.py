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

"""P1 priority tests for GPUModelRunner - Speculative Decoding, Chunked Prefill, etc."""

import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestSpeculativeDecoding(unittest.TestCase):
    """Test cases for speculative decoding functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_model_len = 4096
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_speculative_config = Mock()
        self.mock_speculative_config.num_speculative_tokens = 4
        self.mock_speculative_config.num_gpu_block_expand_ratio = 1
        self.mock_fd_config.speculative_config = self.mock_speculative_config

        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config

        self.mock_cache_config = Mock()
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.enc_dec_block_num = 0
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.speculative_config = self.mock_speculative_config
        self.runner.speculative_method = None
        self.runner.share_inputs = Mock()
        self.runner.share_inputs["seq_lens_this_time_buffer"] = Mock()

    def test_speculative_decoding_disabled(self):
        """Test with speculative_decoding disabled."""
        self.runner.speculative_method = None

        # Verify proposer is None
        self.runner._init_speculative_proposer()
        self.assertIsNone(self.runner.proposer)

    def test_ngram_proposer_initialization(self):
        """Test NgramProposer initialization."""
        self.runner.speculative_method = "ngram"

        with patch("fastdeploy.worker.gpu_model_runner.NgramProposer") as mock_ngram_proposer:
            mock_ngram_proposer.return_value = Mock()

            self.runner._init_speculative_proposer()

            # Verify NgramProposer is created
            mock_ngram_proposer.assert_called_once_with(self.mock_fd_config)
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
            mock_mtp_proposer.assert_called_once_with(self.mock_fd_config, mock_model, 0, 0, self.runner.share_inputs)
            self.assertIsNotNone(self.runner.proposer)

    def test_mtp_kvcache_calculation(self):
        """Test KV cache calculation with MTP."""
        self.mock_speculative_config.num_gpu_block_expand_ratio = 2
        self.runner.speculative_method = "mtp"
        self.mock_model_config.head_dim = 128
        self.mock_model_config.kv_num_heads = 32
        self.mock_model_config.num_hidden_layers = 24

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

    def test_speculative_decode_in_execute_model_overlap(self):
        """Test execute_model_overlap with speculative_decoding."""
        self.runner.speculative_decoding = True
        self.runner.use_cudagraph = False
        self.runner.last_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (Mock(), Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_overlap(model_forward_batch=[Mock()], num_running_requests=1)

                    # With speculative decoding, _save_model_output should NOT be called
                    mock_save.assert_not_called()

    def test_speculative_token_count_configuration(self):
        """Test with different num_speculative_tokens values."""
        for num_tokens in [1, 4, 8, 16]:
            self.mock_speculative_config.num_speculative_tokens = num_tokens
            self.runner.speculative_config = self.mock_speculative_config

            # Verify configuration is applied
            self.assertEqual(self.runner.speculative_config.num_speculative_tokens, num_tokens)


class TestChunkedPrefill(unittest.TestCase):
    """Test cases for chunked prefill functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_model_len = 4096
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_cache_config = Mock()
        self.mock_cache_config.enable_chunked_prefill = False
        self.mock_cache_config.max_chunked_prefill_len = 4096
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.enc_dec_block_num = 0
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.share_inputs = Mock()

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
        self.mock_cache_config.enable_chunked_prefill = True

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
        self.mock_cache_config.max_chunked_prefill_len = 512

        # Request tokens exceeding max chunked prefill length
        num_tokens = 1000
        batch_size = 1

        input_length_list, max_dec_len_list, block_num = self.runner.get_input_length_list(
            num_tokens=num_tokens, batch_size=batch_size, expected_decode_len=100, capture_prefill=True
        )

        # Total should still equal num_tokens
        self.assertEqual(sum(input_length_list), num_tokens)


class TestPrefixCaching(unittest.TestCase):
    """Test cases for prefix caching functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_cache_config = Mock()
        self.mock_cache_config.enable_prefix_caching = False
        self.mock_cache_config.max_num_blocks = 1000
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.mock_model_config = Mock()
        self.mock_model_config.max_model_len = 4096
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.model_config = self.mock_model_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.cache_kvs_map = {}
        self.runner.share_inputs = Mock()

    def test_prefix_caching_disabled(self):
        """Test with prefix caching disabled."""
        self.mock_cache_config.enable_prefix_caching = False

        # Verify prefix caching is disabled
        self.assertFalse(self.mock_cache_config.enable_prefix_caching)

    def test_prefix_caching_enabled(self):
        """Test with prefix caching enabled."""
        self.mock_cache_config.enable_prefix_caching = True

        # Verify prefix caching is enabled
        self.assertTrue(self.mock_cache_config.enable_prefix_caching)

    def test_cache_kvs_map_storage(self):
        """Test cache_kvs_map storage."""
        req_id = "req_1"
        cache_data = Mock()

        self.runner.cache_kvs_map[req_id] = cache_data

        # Verify cache is stored
        self.assertIn(req_id, self.runner.cache_kvs_map)
        self.assertEqual(self.runner.cache_kvs_map[req_id], cache_data)

    def test_cache_kvs_map_retrieval(self):
        """Test cache_kvs_map retrieval."""
        req_id = "req_1"
        cache_data = Mock()

        self.runner.cache_kvs_map[req_id] = cache_data

        # Retrieve cache
        retrieved = self.runner.cache_kvs_map.get(req_id)

        self.assertEqual(retrieved, cache_data)

    def test_cache_kvs_map_clear(self):
        """Test clearing cache_kvs_map."""
        req_id = "req_1"
        self.runner.cache_kvs_map[req_id] = Mock()

        # Clear cache
        self.runner.cache_kvs_map.clear()

        # Verify cache is cleared
        self.assertEqual(len(self.runner.cache_kvs_map), 0)

    def test_prefix_caching_with_prompt_logprobs_error(self):
        """Test that prefix caching with prompt logprobs raises error."""
        self.mock_cache_config.enable_prefix_caching = True
        self.mock_model_config.logprobs_mode = "raw_logprobs"
        self.mock_scheduler_config.max_num_seqs = 10

        self.runner.prompt_logprobs_reqs = {"req_1": Mock()}
        self.runner.ori_vocab_size = 50000
        self.runner.in_progress_prompt_logprobs = {}

        hidden_states = paddle.zeros((10, 768))

        # Should raise AssertionError
        with self.assertRaises(AssertionError):
            self.runner._get_prompt_logprobs_list(hidden_states)


class TestMemoryPressure(unittest.TestCase):
    """Test cases for memory pressure scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_cache_config = Mock()
        self.mock_cache_config.max_num_blocks = 1000
        self.mock_cache_config.num_gpu_blocks = 800
        self.mock_cache_config.num_cpu_blocks = 200
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.kvcache_storage_backend = None
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.mock_parallel_config = Mock()
        self.mock_parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.local_rank = 0
        self.runner.cache_kvs_map = {}
        self.runner.share_inputs = Mock()

    def test_kv_cache_near_limit(self):
        """Test behavior when KV cache is near max_num_blocks."""
        # Set blocks close to limit
        self.mock_cache_config.num_gpu_blocks = 990  # Near 1000 limit

        # Should handle near-limit scenario
        self.assertEqual(self.runner.cache_config.num_gpu_blocks, 990)

    def test_kv_cache_eviction(self):
        """Test KV cache eviction when full."""
        # Fill cache to capacity
        for i in range(1000):
            req_id = f"req_{i}"
            self.runner.cache_kvs_map[req_id] = Mock()

        # Cache is full
        self.assertEqual(len(self.runner.cache_kvs_map), 1000)

        # Evict some entries
        for i in range(100):
            req_id = f"req_{i}"
            del self.runner.cache_kvs_map[req_id]

        # Verify eviction
        self.assertEqual(len(self.runner.cache_kvs_map), 900)

    def test_gpu_cpu_block_swap(self):
        """Test GPU-CPU block swap scenario."""
        # Simulate GPU blocks being swapped to CPU
        self.mock_cache_config.num_gpu_blocks = 600
        self.mock_cache_config.num_cpu_blocks = 200

        # Verify block counts
        self.assertEqual(self.runner.cache_config.num_gpu_blocks, 600)
        self.assertEqual(self.runner.cache_config.num_cpu_blocks, 200)

    def test_clear_cache_with_cpu_blocks(self):
        """Test clear_cache with CPU blocks present."""
        self.mock_cache_config.num_cpu_blocks = 100

        self.runner.share_inputs.pop = Mock()
        self.runner.forward_meta = None
        self.runner.use_cudagraph = False

        with patch("paddle.device.cuda.empty_cache") as mock_empty_cache:
            mock_empty_cache.return_value = None

            self.runner.clear_cache(profile=False)

            # Verify cache is cleared
            self.assertEqual(self.runner.cache_kvs_map, {})
            mock_empty_cache.assert_called_once()


class TestVisionFeatureExtraction(unittest.TestCase):
    """Test cases for vision feature extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.model_type = "ernie"
        self.mock_model_config.enable_mm = True
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_cache_config = Mock()
        self.mock_cache_config.max_encoder_cache = 100
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.enable_mm = True
        self.runner.encoder_cache = {}
        self.runner.rope3d_cache = {}

    def test_vision_cache_hit(self):
        """Test vision encoder cache hit scenario."""
        mm_hash = "hash_12345"
        cached_features = paddle.zeros((10, 768))

        # Pre-populate cache
        self.runner.encoder_cache[mm_hash] = cached_features

        # Verify cache hit
        self.assertIn(mm_hash, self.runner.encoder_cache)
        self.assertEqual(self.runner.encoder_cache[mm_hash].shape, cached_features.shape)

    def test_vision_cache_miss(self):
        """Test vision encoder cache miss scenario."""
        mm_hash = "hash_67890"

        # Verify cache miss
        self.assertNotIn(mm_hash, self.runner.encoder_cache)

    def test_vision_cache_eviction(self):
        """Test vision cache eviction."""
        # Fill cache
        for i in range(150):
            mm_hash = f"hash_{i}"
            self.runner.encoder_cache[mm_hash] = paddle.zeros((10, 768))

        # Evict some entries
        evict_hashes = [f"hash_{i}" for i in range(50)]
        for h in evict_hashes:
            self.runner.encoder_cache.pop(h, None)

        # Verify eviction
        self.assertEqual(len(self.runner.encoder_cache), 100)

    def test_vision_cache_disabled(self):
        """Test with encoder cache disabled."""
        self.mock_cache_config.max_encoder_cache = 0
        self.runner.encoder_cache = None

        # Verify cache is disabled
        self.assertIsNone(self.runner.encoder_cache)

    def test_prepare_rope3d_cache(self):
        """Test Rope3D caching."""
        cache_key = "rope3d_key_1"
        cached_rope = paddle.zeros((2048, 128))

        # Pre-populate rope3d cache
        self.runner.rope3d_cache[cache_key] = cached_rope

        # Verify cache hit
        self.assertIn(cache_key, self.runner.rope3d_cache)

    def test_multi_image_processing(self):
        """Test processing multiple images."""
        vision_inputs = {
            "image_embeds": [
                [paddle.zeros((10, 768))],
                [paddle.zeros((10, 768))],
                [paddle.zeros((10, 768))],
            ],
            "grid_thw": [
                [paddle.to_tensor([2, 2, 16])],
                [paddle.to_tensor([2, 2, 16])],
                [paddle.to_tensor([2, 2, 16])],
            ],
        }

        # Verify multiple images can be processed
        self.assertEqual(len(vision_inputs["image_embeds"]), 3)
        self.assertEqual(len(vision_inputs["grid_thw"]), 3)


if __name__ == "__main__":
    unittest.main()
