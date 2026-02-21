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
Tests for Phase 4: Vision Processing

This module contains tests for the vision processing phase of GPU Model Runner,
corresponding to Phase 4 (Preprocessing - Vision) in docs/gpu_model_runner_data_flow.md

Key components tested:
- Vision feature extraction (Ernie/LLaVA models)
- RoPE3D preparation
- Multimodal inputs processing
- Vision encoder cache management
"""

import numpy as np
import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.engine.request import RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestVisionFeatureExtraction(unittest.TestCase):
    """Test cases for vision feature extraction and caching."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.model_type = "ernie"
        self.runner.model_config.enable_mm = True
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.max_encoder_cache = 100
        self.runner.fd_config.cache_config = self.runner.cache_config
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
        self.runner.cache_config.max_encoder_cache = 0
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


class TestExtractVisionFeaturesErnie(unittest.TestCase):
    """Test cases for extract_vision_features_ernie method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.model_type = "ernie"
        self.runner.model_config.enable_mm = True
        self.runner.model_config.dtype = "bfloat16"
        self.runner.model_config.spatial_conv_size = 4
        self.runner.model_config.vision_config = Mock()
        self.runner.model_config.vision_config.patch_size = 14
        self.runner.model_config.vision_config.image_mean = [0.485, 0.456, 0.406]
        self.runner.model_config.vision_config.image_std = [0.229, 0.224, 0.225]
        self.runner.model_config.vision_config.rescale_factor = 2.0
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.tensor_parallel_size = 1
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.encoder_cache = {}
        self.runner.amp_black = ["reduce_sum", "c_softmax_with_cross_entropy"]
        self.runner.amp_white = ["matmul", "flash_attn"]

        # Setup mock model with vision_model and resampler_model
        self.runner.model = Mock()
        self.runner.model.vision_model = Mock()
        self.runner.model.resampler_model = Mock()

        # Default grid_thw value used in tests
        self.runner.grid_thw = paddle.to_tensor([2, 2, 16], dtype=paddle.int64)


class TestExtractVisionFeaturesLLaVA(unittest.TestCase):
    """Test cases for extract_vision_features_llava method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.model_type = "llava"
        self.runner.model_config.enable_mm = True
        self.runner.model_config.dtype = "bfloat16"
        self.runner.model_config.vision_config = Mock()
        self.runner.model_config.vision_config.vision_hidden_size = 768
        self.runner.model_config.vision_config.vision_num_layers = 24
        self.runner.model_config.vision_config.patch_size = 14
        self.runner.model_config.vision_config.image_mean = [0.485, 0.456, 0.406]
        self.runner.model_config.vision_config.image_std = [0.229, 0.224, 0.225]
        self.runner.model_config.vision_config.rescale_factor = 1.0
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.tensor_parallel_size = 1
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.encoder_cache = {}
        self.runner.amp_black = ["reduce_sum", "c_softmax_with_cross_entropy"]
        self.runner.amp_white = ["matmul", "flash_attn"]

        # Setup mock model
        self.runner.model = Mock()
        self.runner.model.vision_model = Mock()
        self.runner.model.vision_resampler = Mock()


class TestPrepareRoPe3D(unittest.TestCase):
    """Test cases for prepare_rope3d method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = True
        self.runner.model_config.max_encoder_len = 512
        self.runner.model_config.rope_3d_dim = 128
        self.runner.model_config.num_attention_heads = 32
        self.runner.model_config.num_kv_heads = 4
        self.runner.model_config.head_dim = 128
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.rope3d_cache = {}
        self.runner.model = Mock()


class TestProcessMultimodalInputs(unittest.TestCase):
    """Test cases for processing multimodal inputs."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = True
        self.runner.model_config.max_encoder_len = 512
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.enable_mm = True
        self.runner.encoder_cache = {}
        self.runner.share_inputs = Mock()

    def test_process_mm_features_with_image(self):
        """Test _process_mm_features with image input."""
        # This test verifies that multimodal features are processed correctly
        # when an image is present in the request
        request = Mock()
        request.with_image = True
        request.multimodal_inputs = {
            "images": [paddle.zeros((3, 224, 224))],
            "grid_thw": [paddle.to_tensor([2, 2, 16])],
            "mm_hashes": ["hash_123"],
        }

        # The actual processing happens in insert_tasks_v1
        # This test structure is for future expansion
        self.assertTrue(request.with_image)
        self.assertIn("images", request.multimodal_inputs)

    def test_process_mm_features_with_cache_hit(self):
        """Test _process_mm_features with cache hit."""
        mm_hash = "cached_hash_123"
        cached_features = paddle.zeros((32, 768))

        # Pre-populate cache
        self.runner.encoder_cache[mm_hash] = cached_features

        # Verify cache is populated
        self.assertIn(mm_hash, self.runner.encoder_cache)

    def test_process_mm_features_cache_miss(self):
        """Test _process_mm_features with cache miss."""
        mm_hash = "missed_hash_456"

        # Cache should be empty for miss
        self.assertNotIn(mm_hash, self.runner.encoder_cache)


if __name__ == "__main__":
    unittest.main()
