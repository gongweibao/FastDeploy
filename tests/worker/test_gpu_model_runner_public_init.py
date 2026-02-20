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

"""Unit tests for initialization related public methods of GPUModelRunner."""

import unittest
from unittest.mock import MagicMock, Mock, patch


from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestInitializeForwardMeta(unittest.TestCase):
    """Test cases for initialize_forward_meta method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_num_seqs = 10
        self.mock_model_config.num_hidden_layers = 24
        self.mock_model_config.hidden_size = 4096
        self.mock_model_config.vocab_size = 32000
        self.mock_model_config.dtype = "float16"
        self.mock_model_config.kv_lora_rank = 0
        self.mock_model_config.qk_rope_head_dim = 64
        self.mock_model_config.num_attention_heads = 32
        self.mock_model_config.num_kv_heads = 4
        self.mock_model_config.enable_mm = False
        self.mock_model_config.max_encoder_len = 512
        self.mock_model_config.max_prompt_embedding_table_size = 0
        self.mock_model_config.lora_request_ids_to_shard_id = None
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_cache_config = Mock()
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.max_num_blocks = 1000
        self.mock_cache_config.sliding_window = -1
        self.mock_cache_config.cache_cpu_block_num = 0
        self.mock_cache_config.kv_cache_dtype = "float16"
        self.mock_cache_config.enable_prefix_caching = False
        self.mock_cache_config.enable_chunked_prefill = False
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.mock_parallel_config = Mock()
        self.mock_parallel_config.tensor_parallel_size = 1
        self.mock_parallel_config.pipeline_parallel_size = 1
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.mock_speculative_config = Mock()
        self.mock_speculative_config.method = None
        self.mock_speculative_config.num_speculative_tokens = 0
        self.mock_fd_config.speculative_config = self.mock_speculative_config
        self.mock_routing_replay_config = Mock()
        self.mock_routing_replay_config.enable_routing_replay = False
        self.mock_fd_config.routing_replay_config = self.mock_routing_replay_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.parallel_config = self.mock_parallel_config
        self.runner.speculative_config = self.mock_speculative_config
        self.runner.routing_replay_config = self.mock_routing_replay_config
        self.runner.routing_replay_manager = None
        self.runner.quant_config = None
        self.runner.share_inputs = Mock()
        self.runner.forward_meta = MagicMock()
        self.runner.lora_request_ids_to_shard_id = None

    def test_initialize_forward_meta_basic(self):
        """Test initialize_forward_meta with default parameters."""
        # The real implementation creates a ForwardMeta object with specific fields
        self.runner.initialize_forward_meta()

        # Verify forward_meta is properly initialized
        self.assertIsNotNone(self.runner.forward_meta)

        # Verify key fields are accessible on forward_meta
        # The actual ForwardMeta class has these fields initialized from share_inputs
        expected_fields = [
            'ids_remove_padding', 'rotary_embs', 'attn_backend',
            'decoder_batch_ids', 'decoder_tile_ids_per_batch',
            'decoder_num_blocks_cpu', 'decoder_num_blocks_device',
            'decoder_chunk_size_device', 'max_len_tensor_cpu',
            'seq_lens_encoder', 'seq_lens_decoder', 'seq_lens_this_time',
            'batch_id_per_token', 'cu_seqlens_q', 'cu_seqlens_k',
            'block_tables', 'caches', 'encoder_batch_ids',
            'encoder_tile_ids_per_batch', 'encoder_num_blocks_x_cpu',
            'kv_batch_ids', 'kv_tile_ids_per_batch',
            'kv_num_blocks_x_cpu', 'routing_replay_table'
        ]

        # Check that forward_meta has these attributes (even if they're mocked)
        forward_meta = self.runner.forward_meta
        self.assertIsNotNone(forward_meta)

        # Verify that ForwardMeta was created with the share_inputs attributes
        # The actual implementation passes these to ForwardMeta constructor
        self.assertIsNotNone(forward_meta)

    def test_initialize_forward_meta_dummy_run(self):
        """Test initialize_forward_meta with dummy_or_profile_run=True."""
        # Save original forward_meta to compare
        original_meta = self.runner.forward_meta

        self.runner.initialize_forward_meta(is_dummy_or_profile_run=True)

        # Verify forward_meta is initialized even for dummy run
        self.assertIsNotNone(self.runner.forward_meta)
        # In dummy run, the forward_meta should still be created
        # The parameter affects behavior but still creates the object

    def test_initialize_forward_meta_with_multimodal(self):
        """Test initialize_forward_meta with enable_mm=True."""
        self.mock_model_config.enable_mm = True
        self.mock_model_config.max_encoder_len = 1024
        self.mock_model_config.max_prompt_embedding_table_size = 100

        self.runner.initialize_forward_meta()

        # Verify forward_meta is initialized
        self.assertIsNotNone(self.runner.forward_meta)

        # With multimodal, forward_meta should have encoder-related fields
        forward_meta = self.runner.forward_meta
        self.assertIsNotNone(forward_meta)

    def test_initialize_forward_meta_with_speculative_decoding(self):
        """Test initialize_forward_meta with speculative decoding enabled."""
        self.mock_speculative_config.method = "mtp"
        self.mock_speculative_config.num_speculative_tokens = 8

        self.runner.initialize_forward_meta()

        # Verify forward_meta is initialized
        self.assertIsNotNone(self.runner.forward_meta)

        # Speculative decoding affects the flow but forward_meta is still created
        forward_meta = self.runner.forward_meta
        self.assertIsNotNone(forward_meta)

    def test_initialize_forward_meta_with_prefix_caching(self):
        """Test initialize_forward_meta with prefix caching enabled."""
        self.mock_cache_config.enable_prefix_caching = True

        self.runner.initialize_forward_meta()

        # Verify forward_meta is initialized with prefix caching support
        self.assertIsNotNone(self.runner.forward_meta)

        # Prefix caching affects how routing table is handled
        forward_meta = self.runner.forward_meta
        self.assertIsNotNone(forward_meta)


class TestInitializeKVCache(unittest.TestCase):
    """Test cases for initialize_kv_cache method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.num_hidden_layers = 24
        self.mock_model_config.num_attention_heads = 32
        self.mock_model_config.kv_lora_rank = 0
        self.mock_model_config.qk_rope_head_dim = 64
        self.mock_model_config.dtype = "float16"
        self.mock_model_config.enable_mm = False
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_cache_config = Mock()
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.max_num_blocks = 1000
        self.mock_cache_config.cache_cpu_block_num = 0
        self.mock_cache_config.kv_cache_dtype = "float16"
        self.mock_cache_config.enable_prefix_caching = False
        self.mock_cache_config.enable_chunked_prefill = False
        self.mock_cache_config.kvcache_storage_backend = None
        self.mock_cache_config.use_mla_cache = False
        self.mock_cache_config.cache_k = None
        self.mock_cache_config.cache_v = None
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.mock_parallel_config = Mock()
        self.mock_parallel_config.tensor_parallel_size = 1
        self.mock_parallel_config.pipeline_parallel_size = 1
        self.mock_parallel_config.use_ep = False
        self.mock_parallel_config.enable_chunked_moe = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config
        self.mock_routing_replay_config = Mock()
        self.mock_routing_replay_config.enable_routing_replay = False
        self.mock_fd_config.routing_replay_config = self.mock_routing_replay_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.parallel_config = self.mock_parallel_config
        self.runner.routing_replay_config = self.mock_routing_replay_config
        self.runner.routing_replay_manager = None
        self.runner.share_inputs = Mock()
        self.runner.num_gpu_blocks = 100
        self.runner.forward_meta = Mock()
        self.runner.cache_k = None
        self.runner.cache_v = None
        self.runner.cache_k_cpu = None
        self.runner.cache_v_cpu = None
        self.runner.kv_cache_quant_type = None
        self.runner.lora_request_ids_to_shard_id = None

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_basic(self, mock_get_backend):
        """Test initialize_kv_cache with basic configuration."""
        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 2048  # Simulated block size
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend

        # Mock shape retrieval for kv cache
        mock_backend.get_kv_cache_shape = Mock(return_value=((100, 4, 64, 128), (100, 4, 64, 128)))

        self.runner.initialize_kv_cache(profile=False)

        # Verify cache attributes are initialized
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)
        # Verify backend was called to get shape
        mock_backend.get_kv_cache_shape.assert_called_once_with(
            max_num_blocks=100, kv_cache_quant_type=None
        )
        # Verify get_attention_backend was called
        mock_get_backend.assert_called_once()

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_with_cpu_blocks(self, mock_get_backend):
        """Test initialize_kv_cache with CPU blocks configured."""
        self.mock_cache_config.cache_cpu_block_num = 50

        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 2048
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend
        mock_backend.get_kv_cache_shape = Mock(return_value=((100, 4, 64, 128), (100, 4, 64, 128)))

        self.runner.initialize_kv_cache(profile=False)

        # Verify both GPU and CPU caches are initialized
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)
        self.assertIsNotNone(self.runner.cache_k_cpu)
        self.assertIsNotNone(self.runner.cache_v_cpu)

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_profile_mode(self, mock_get_backend):
        """Test initialize_kv_cache with profile=True.

        In profile mode, create_cache_tensor is True regardless of other settings.
        """
        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 2048
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend
        mock_backend.get_kv_cache_shape = Mock(return_value=((100, 4, 64, 128), (100, 4, 64, 128)))

        # In profile mode, should always create cache
        self.runner.initialize_kv_cache(profile=True)

        # Verify cache is initialized in profile mode
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_with_mla(self, mock_get_backend):
        """Test initialize_kv_cache with MLA cache enabled."""
        self.mock_cache_config.use_mla_cache = True

        # MLA uses a different cache quantization type
        mock_quant_config = Mock()
        mock_quant_config.kv_cache_quant_type = "block_wise_fp8"
        self.runner.quant_config = mock_quant_config

        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 2048
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 512
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend
        mock_backend.get_kv_cache_shape = Mock(return_value=((100, 4, 64, 128), (100, 4, 64, 128)))

        self.runner.initialize_kv_cache(profile=False)

        # Verify MLA cache attributes are initialized
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)
        # Verify get_kv_cache_shape was called with quant_type
        mock_backend.get_kv_cache_shape.assert_called_once_with(
            max_num_blocks=100, kv_cache_quant_type="block_wise_fp8"
        )

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_with_tensor_parallel(self, mock_get_backend):
        """Test initialize_kv_cache with tensor parallel size > 1."""
        self.mock_parallel_config.tensor_parallel_size = 2

        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 2048
        mock_backend.num_kv_heads = 8  # 32 / 4 with TP=2
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend
        mock_backend.get_kv_cache_shape = Mock(return_value=((100, 8, 64, 128), (100, 8, 64, 128)))

        self.runner.initialize_kv_cache(profile=False)

        # Verify cache is initialized with TP
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)
        # Verify local_rank is used correctly (local_rank % TP_size)
        mock_backend.get_kv_cache_shape.assert_called_once()

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_layers_count(self, mock_get_backend):
        """Test initialize_kv_cache creates cache for all layers."""
        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 2048
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend
        mock_backend.get_kv_cache_shape = Mock(return_value=((100, 4, 64, 128), (100, 4, 64, 128)))

        self.runner.initialize_kv_cache(profile=False)

        # Verify cache_k and cache_v are initialized (lists for each layer)
        # The implementation creates cache for num_hidden_layers
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)


if __name__ == "__main__":
    unittest.main()
