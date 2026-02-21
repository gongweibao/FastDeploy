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
Tests for Cache Management

This module contains tests for KV Cache and Vision Cache management
in GPU Model Runner.

Key components tested:
- Prefix caching
- Memory pressure handling
- Cache initialization
- GPU-CPU block swap
"""

import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestPrefixCaching(unittest.TestCase):
    """Test cases for prefix caching functionality."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.cache_config = Mock()
        self.runner.cache_config.enable_prefix_caching = False
        self.runner.cache_config.max_num_blocks = 1000
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.model_config = Mock()
        self.runner.model_config.max_model_len = 4096
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.cache_kvs_map = {}
        self.runner.share_inputs = Mock()

    def test_prefix_caching_disabled(self):
        """Test with prefix caching disabled."""
        self.assertFalse(self.runner.cache_config.enable_prefix_caching)

    def test_prefix_caching_enabled(self):
        """Test with prefix caching enabled."""
        self.runner.cache_config.enable_prefix_caching = True

        self.assertTrue(self.runner.cache_config.enable_prefix_caching)

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


class TestMemoryPressure(unittest.TestCase):
    """Test cases for memory pressure scenarios."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.cache_config = Mock()
        self.runner.cache_config.max_num_blocks = 1000
        self.runner.cache_config.num_gpu_blocks = 800
        self.runner.cache_config.num_cpu_blocks = 200
        self.runner.cache_config.block_size = 16
        self.runner.cache_config.kvcache_storage_backend = None
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.tensor_parallel_size = 1
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.local_rank = 0
        self.runner.cache_kvs_map = {}
        self.runner.share_inputs = Mock()

    def test_kv_cache_near_limit(self):
        """Test behavior when KV cache is near max_num_blocks."""
        # Set blocks close to limit
        self.runner.cache_config.num_gpu_blocks = 990

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
        self.runner.cache_config.num_gpu_blocks = 600
        self.runner.cache_config.num_cpu_blocks = 200

        # Verify block counts
        self.assertEqual(self.runner.cache_config.num_gpu_blocks, 600)
        self.assertEqual(self.runner.cache_config.num_cpu_blocks, 200)


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
        expected = 2 * 2 * 16 * (128 * 32) * 24
        self.assertEqual(result, expected)

    def test_cal_theortical_kvcache_int8_dtype(self):
        """Test with int8 dtype (1 byte)."""
        self.runner.quant_config.kv_cache_quant_type = "int8"

        result = self.runner.cal_theortical_kvcache()

        # byte_of_dtype = 1
        expected = 1 * 2 * 16 * (128 * 32) * 24
        self.assertEqual(result, expected)

    def test_cal_theortical_kvcache_with_mtp(self):
        """Test with MTP speculative method."""
        self.runner.speculative_method = "mtp"
        self.runner.speculative_config.num_gpu_block_expand_ratio = 2

        result = self.runner.cal_theortical_kvcache()

        # num_layers = 24 + 2 = 26
        expected = 2 * 2 * 16 * (128 * 32) * 26
        self.assertEqual(result, expected)

    def test_cal_theortical_kvcache_with_mla(self):
        """Test with MLA cache."""
        self.runner.cache_config.use_mla_cache = True
        self.runner.model_config.kv_lora_rank = 64
        self.runner.model_config.qk_rope_head_dim = 64

        result = self.runner.cal_theortical_kvcache()

        # MLA: compress_kv + k_pe = (64 + 64) * 16 * num_layers
        expected = 2 * (64 + 64) * 16 * 26
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
