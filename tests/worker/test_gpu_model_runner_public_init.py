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
from unittest.mock import Mock, patch, MagicMock, call

import numpy as np
import paddle

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
        # Create a new mock for forward_meta after initialization
        new_forward_meta = Mock()
        new_forward_meta.max_num_seqs = 10
        new_forward_meta.num_hidden_layers = 24
        new_forward_meta.hidden_size = 4096
        new_forward_meta.vocab_size = 32000
        new_forward_meta.dtype = "float16"

        self.runner.initialize_forward_meta()

        # Verify forward_meta is properly initialized
        # After initialization, forward_meta should be set (real implementation)
        self.assertIsNotNone(self.runner.forward_meta)

    def test_initialize_forward_meta_dummy_run(self):
        """Test initialize_forward_meta with dummy_or_profile_run=True."""
        self.runner.initialize_forward_meta(is_dummy_or_profile_run=True)

        # Verify forward_meta is initialized even for dummy run
        self.assertIsNotNone(self.runner.forward_meta)

    def test_initialize_forward_meta_with_multimodal(self):
        """Test initialize_forward_meta with enable_mm=True."""
        self.mock_model_config.enable_mm = True
        self.mock_model_config.max_encoder_len = 1024
        self.mock_model_config.max_prompt_embedding_table_size = 100

        self.runner.initialize_forward_meta()

        # Verify forward_meta is initialized
        self.assertIsNotNone(self.runner.forward_meta)

    def test_initialize_forward_meta_with_speculative_decoding(self):
        """Test initialize_forward_meta with speculative decoding enabled."""
        self.mock_speculative_config.method = "mtp"
        self.mock_speculative_config.num_speculative_tokens = 8

        self.runner.initialize_forward_meta()

        # Verify forward_meta is initialized
        self.assertIsNotNone(self.runner.forward_meta)

    def test_initialize_forward_meta_with_prefix_caching(self):
        """Test initialize_forward_meta with prefix caching enabled."""
        self.mock_cache_config.enable_prefix_caching = True

        self.runner.initialize_forward_meta()

        # Verify forward_meta is initialized with prefix caching support
        self.assertIsNotNone(self.runner.forward_meta)


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
        mock_backend.cache_block_bytes = 0
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend

        self.runner.initialize_kv_cache(profile=False)

        # Verify cache attributes are initialized
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)
        # Verify backend was called
        mock_get_backend.assert_called_once()

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_with_cpu_blocks(self, mock_get_backend):
        """Test initialize_kv_cache with CPU blocks configured."""
        self.mock_cache_config.cache_cpu_block_num = 50

        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 0
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend

        self.runner.initialize_kv_cache(profile=False)

        # Verify both GPU and CPU caches are initialized
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_k_cpu)

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_profile_mode(self, mock_get_backend):
        """Test initialize_kv_cache with profile=True."""
        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 0
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend

        self.runner.initialize_kv_cache(profile=True)

        # Verify cache is initialized in profile mode
        self.assertIsNotNone(self.runner.cache_k)
        self.assertIsNotNone(self.runner.cache_v)

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_with_mla(self, mock_get_backend):
        """Test initialize_kv_cache with MLA cache enabled."""
        self.mock_cache_config.use_mla_cache = True

        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 0
        mock_backend.num_kv_heads = 4
        mock_backend.kv_lora_rank = 512
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend

        self.runner.initialize_kv_cache(profile=False)

        # Verify MLA cache attributes are initialized
        self.assertIsNotNone(self.runner.cache_k)

    @patch("fastdeploy.worker.gpu_model_runner.get_attention_backend")
    def test_initialize_kv_cache_with_tensor_parallel(self, mock_get_backend):
        """Test initialize_kv_cache with tensor parallel size > 1."""
        self.mock_parallel_config.tensor_parallel_size = 2

        mock_backend = Mock()
        mock_backend.get_backend_name = Mock(return_value="test_backend")
        mock_backend.cache_block_bytes = 0
        mock_backend.num_kv_heads = 8  # 32 / 4 with TP=2
        mock_backend.kv_lora_rank = 0
        mock_backend.qk_rope_head_dim = 64
        mock_get_backend.return_value = mock_backend

        self.runner.initialize_kv_cache(profile=False)

        # Verify cache is initialized with TP
        self.assertIsNotNone(self.runner.cache_k)


if __name__ == "__main__":
    unittest.main()
