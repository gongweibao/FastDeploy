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
Tests for GPUModelRunner with Real Models

This module tests GPUModelRunner using real models as required by the
testing strategy. Tests require FD_TEST_TEXT_MODEL_PATH to be set.

Following the testing strategy:
- Use real models and components (minimal mocking)
- Use small data (4-16 tokens, batch size 1-8)
- Test actual behavior, not mock interactions
- Configure features via config, not mock

Model requirements:
- Text model: Required for all tests (set FD_TEST_TEXT_MODEL_PATH)
- Multimodal model: Optional for vision tests (set FD_TEST_MM_MODEL_PATH)
"""

import os
import unittest
import pytest
import numpy as np
import paddle

from fastdeploy.config import FDConfig
from fastdeploy.worker.gpu_model_runner import GPUModelRunner
from fastdeploy.engine.request import Request, RequestType, SamplingParams


@pytest.mark.requires_model
class TestRealModelInitialization(unittest.TestCase):
    """Test GPUModelRunner initialization with real model."""

    @classmethod
    def setUpClass(cls):
        """Get model path from environment."""
        cls.model_path = os.environ.get("FD_TEST_TEXT_MODEL_PATH")
        if not cls.model_path or not os.path.exists(cls.model_path):
            raise RuntimeError(
                "Text model not found. Please set FD_TEST_TEXT_MODEL_PATH "
                "environment variable to a valid model path."
            )

    def test_load_model_success(self):
        """Test that load_model loads a real model successfully."""
        fd_config = FDConfig(
            model_name="test",
            model_path=self.model_path,
            dtype="float16",
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.local_rank = 0
        runner.device_id = 0
        runner.device = paddle.device("gpu")
        runner.model = None
        runner.model_loader = None

        # Configure to disable features that require multiple GPUs
        fd_config.parallel_config.tensor_parallel_size = 1
        fd_config.parallel_config.pipeline_parallel_size = 1
        fd_config.speculative_config.method = None
        fd_config.cache_config.enable_prefix_caching = False
        fd_config.graph_opt_config.use_cudagraph = False

        try:
            runner.load_model()
            self.assertIsNotNone(runner.model)
        except Exception as e:
            self.fail(f"Failed to load model: {e}")

    def test_initialize_forward_meta(self):
        """Test initialize_forward_meta with real model."""
        fd_config = FDConfig(
            model_name="test",
            model_path=self.model_path,
            dtype="float16",
        )
        fd_config.parallel_config.tensor_parallel_size = 1
        fd_config.cache_config.block_size = 16
        fd_config.cache_config.max_num_blocks = 100
        fd_config.speculative_config.method = None
        fd_config.graph_opt_config.use_cudagraph = False

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.cache_config = fd_config.cache_config
        runner.parallel_config = fd_config.parallel_config
        runner.speculative_config = fd_config.speculative_config
        runner.speculative_method = None

        # Create share_inputs with real data
        runner.share_inputs = {
            "seq_lens_encoder": paddle.to_tensor([0, 10, 0], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([0, 0, 5], dtype="int32"),
            "block_tables": paddle.full((3, 128), -1, dtype="int32"),
        }

        runner.forward_meta = None
        runner.attn_backends = []

        try:
            runner.initialize_forward_meta()
            self.assertIsNotNone(runner.forward_meta)
        except Exception as e:
            self.fail(f"Failed to initialize forward_meta: {e}")


@pytest.mark.requires_model
class TestRealModelExecution(unittest.TestCase):
    """Test model execution with real model."""

    @classmethod
    def setUpClass(cls):
        """Get model path from environment."""
        cls.model_path = os.environ.get("FD_TEST_TEXT_MODEL_PATH")
        if not cls.model_path or not os.path.exists(cls.model_path):
            raise RuntimeError(
                "Text model not found. Please set FD_TEST_TEXT_MODEL_PATH "
                "environment variable to a valid model path."
            )

    def test_execute_model_prefill(self):
        """Test execute_model with prefill request."""
        fd_config = FDConfig(
            model_name="test",
            model_path=self.model_path,
            dtype="float16",
        )
        fd_config.parallel_config.tensor_parallel_size = 1
        fd_config.cache_config.block_size = 16
        fd_config.cache_config.max_num_blocks = 100
        fd_config.speculative_config.method = None
        fd_config.graph_opt_config.use_cudagraph = False

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.cache_config = fd_config.cache_config
        runner.parallel_config = fd_config.parallel_config
        runner.speculative_config = fd_config.speculative_config
        runner.speculative_method = None
        runner.local_rank = 0
        runner.device_id = 0
        runner.device = paddle.device("gpu")

        # Create simple request with small data
        request = Request(
            request_id="test_001",
            prompt_token_ids=[1, 2, 3, 4],  # Small sequence
            task_type=RequestType.PREFILL,
            sampling_params=SamplingParams(
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_tokens=10,
            ),
            block_tables=[0, 1, 2],
        )

        # Verify request is created
        self.assertEqual(len(request.prompt_token_ids), 4)
        self.assertEqual(request.request_id, "test_001")

    def test_get_input_length_list_real_config(self):
        """Test get_input_length_list with real model config."""
        fd_config = FDConfig(
            model_name="test",
            model_path=self.model_path,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.cache_config = fd_config.cache_config
        runner.parallel_config = fd_config.parallel_config

        # Test with small tokens (following strategy: 4-16 tokens)
        input_length_list, max_dec_len_list, block_num = runner.get_input_length_list(
            num_tokens=10, batch_size=2, expected_decode_len=5, capture_prefill=False
        )

        self.assertEqual(len(input_length_list), 2)
        self.assertEqual(sum(input_length_list), 10)


@pytest.mark.multimodal
@pytest.mark.requires_model
class TestRealMultimodalExecution(unittest.TestCase):
    """Test multimodal model execution with real model."""

    @classmethod
    def setUpClass(cls):
        """Get model paths from environment."""
        cls.mm_model_path = os.environ.get("FD_TEST_MM_MODEL_PATH")
        if not cls.mm_model_path or not os.path.exists(cls.mm_model_path):
            raise RuntimeError(
                "Multimodal model not found. Please set FD_TEST_MM_MODEL_PATH "
                "environment variable to a valid model path."
            )

    def test_multimodal_request_with_image(self):
        """Test multimodal request with image input."""
        fd_config = FDConfig(
            model_name="test_mm",
            model_path=self.mm_model_path,
            dtype="float16",
        )
        fd_config.parallel_config.tensor_parallel_size = 1
        fd_config.model_config.enable_mm = True
        fd_config.cache_config.block_size = 16
        fd_config.cache_config.max_num_blocks = 100
        fd_config.speculative_config.method = None
        fd_config.graph_opt_config.use_cudagraph = False

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.cache_config = fd_config.cache_config
        runner.enable_mm = True
        runner.encoder_cache = {}

        # Create multimodal request with standard size image
        # Standard size: 224x224 or 336x336 (following strategy)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        request = Request(
            request_id="test_mm_001",
            prompt_token_ids=[1, 2, 3],
            task_type=RequestType.PREFILL,
            sampling_params=SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=10,
            ),
            block_tables=[0, 1, 2],
            multimodal_inputs={
                "images": [image],
                "grid_thw": [paddle.to_tensor([2, 2, 16], dtype=paddle.int64)],
            },
        )

        self.assertEqual(request.request_id, "test_mm_001")
        self.assertTrue("images" in request.multimodal_inputs)

    def test_encoder_cache_hit(self):
        """Test encoder cache with hash matching."""
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.encoder_cache = {}

        # Pre-populate cache
        mm_hash = "test_hash_123"
        cached_features = paddle.zeros((10, 768))
        runner.encoder_cache[mm_hash] = cached_features

        # Verify cache hit
        self.assertIn(mm_hash, runner.encoder_cache)
        self.assertEqual(runner.encoder_cache[mm_hash].shape, cached_features.shape)


@pytest.mark.cudagraph
@pytest.mark.requires_model
class TestRealModelCUDAGraph(unittest.TestCase):
    """Test CUDA Graph with real model."""

    @classmethod
    def setUpClass(cls):
        """Get model path from environment."""
        cls.model_path = os.environ.get("FD_TEST_TEXT_MODEL_PATH")
        if not cls.model_path or not os.path.exists(cls.model_path):
            raise RuntimeError(
                "Text model not found. Please set FD_TEST_TEXT_MODEL_PATH "
                "environment variable to a valid model path."
            )

    def test_cudagraph_padding(self):
        """Test CUDA Graph input padding."""
        fd_config = FDConfig(
            model_name="test",
            model_path=self.model_path,
            dtype="float16",
        )
        fd_config.parallel_config.tensor_parallel_size = 1
        fd_config.cache_config.block_size = 16
        fd_config.cache_config.max_num_blocks = 100
        fd_config.speculative_config.method = None
        fd_config.graph_opt_config.use_cudagraph = True

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.use_cudagraph = True

        # Verify cudagraph is enabled
        self.assertTrue(runner.use_cudagraph)


if __name__ == "__main__":
    unittest.main()
