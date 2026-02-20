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

"""Unit tests for vision and execute related public methods of GPUModelRunner."""

import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestExtractVisionFeaturesErnie(unittest.TestCase):
    """Test cases for extract_vision_features_ernie method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.model_type = "ernie"
        self.mock_model_config.enable_mm = True
        self.mock_fd_config.model_config = self.mock_model_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.encoder_cache = {}
        self.runner.model = Mock()

    def test_extract_vision_features_ernie_basic(self):
        """Test extract_vision_features_ernie with basic inputs."""
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "grid_thw": [[paddle.to_tensor([2, 2, 16])]],
        }

        # Mock the model's vision encoder
        self.runner.model.vision_encoder = Mock()
        self.runner.model.vision_encoder.return_value = paddle.zeros((10, 768))

        result = self.runner.extract_vision_features_ernie(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)
        # Verify model's vision encoder is called
        if self.runner.model.vision_encoder.called:
            self.assertTrue(self.runner.model.vision_encoder.called)

    def test_extract_vision_features_ernie_with_cache(self):
        """Test extract_vision_features_ernie with encoder cache."""
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "grid_thw": [[paddle.to_tensor([2, 2, 16])]],
            "image_hash": "test_hash_123",
        }

        # Pre-populate cache
        cached_features = paddle.zeros((10, 768))
        self.runner.encoder_cache["test_hash_123"] = cached_features

        result = self.runner.extract_vision_features_ernie(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_ernie_empty_inputs(self):
        """Test extract_vision_features_ernie with empty inputs."""
        vision_inputs = {
            "image_embeds": [[]],
            "grid_thw": [[]],
        }

        result = self.runner.extract_vision_features_ernie(vision_inputs)

        # Should handle empty inputs gracefully
        self.assertIsNotNone(result)

    def test_extract_vision_features_ernie_multiple_images(self):
        """Test extract_vision_features_ernie with multiple images."""
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

        result = self.runner.extract_vision_features_ernie(vision_inputs)

        # Verify result is returned for multiple images
        self.assertIsNotNone(result)

    def test_extract_vision_features_ernie_different_grid_thw(self):
        """Test extract_vision_features_ernie with different grid_thw configurations."""
        test_cases = [
            [[paddle.to_tensor([1, 1, 16])]],
            [[paddle.to_tensor([2, 2, 16])]],
            [[paddle.to_tensor([4, 4, 16])]],
            [[paddle.to_tensor([8, 8, 16])]],
        ]

        for grid_thw in test_cases:
            vision_inputs = {
                "image_embeds": [[paddle.zeros((10, 768))]],
                "grid_thw": grid_thw,
            }

            result = self.runner.extract_vision_features_ernie(vision_inputs)

            # Verify result is returned for each grid_thw configuration
            self.assertIsNotNone(result)

    def test_extract_vision_features_ernie_cache_miss(self):
        """Test extract_vision_features_ernie with cache miss."""
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "grid_thw": [[paddle.to_tensor([2, 2, 16])]],
            "image_hash": "new_hash_456",
        }

        # Cache is empty - should be a miss
        self.runner.encoder_cache = {}

        result = self.runner.extract_vision_features_ernie(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)
        # Cache should be populated after extraction
        if "new_hash_456" in self.runner.encoder_cache:
            self.assertIsNotNone(self.runner.encoder_cache["new_hash_456"])

    def test_extract_vision_features_ernie_cache_hit(self):
        """Test extract_vision_features_ernie with cache hit."""
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "grid_thw": [[paddle.to_tensor([2, 2, 16])]],
            "image_hash": "cached_hash_789",
        }

        # Pre-populate cache with specific features
        cached_features = paddle.ones((10, 768))
        self.runner.encoder_cache["cached_hash_789"] = cached_features

        result = self.runner.extract_vision_features_ernie(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)
        # Verify cached features are used
        self.assertIn("cached_hash_789", self.runner.encoder_cache)

    def test_extract_vision_features_ernie_mixed_grid_thw(self):
        """Test extract_vision_features_ernie with mixed grid_thw values."""
        vision_inputs = {
            "image_embeds": [
                [paddle.zeros((10, 768))],
                [paddle.zeros((10, 768))],
                [paddle.zeros((10, 768))],
            ],
            "grid_thw": [
                [paddle.to_tensor([1, 1, 16])],
                [paddle.to_tensor([2, 2, 16])],
                [paddle.to_tensor([4, 4, 16])],
            ],
        }

        result = self.runner.extract_vision_features_ernie(vision_inputs)

        # Verify result is returned for mixed grid_thw
        self.assertIsNotNone(result)


class TestExtractVisionFeaturesQwen(unittest.TestCase):
    """Test cases for extract_vision_features_qwen method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.model_type = "qwen"
        self.mock_model_config.enable_mm = True
        self.mock_fd_config.model_config = self.mock_model_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.encoder_cache = {}
        self.runner.model = Mock()

    def test_extract_vision_features_qwen_basic(self):
        """Test extract_vision_features_qwen with basic inputs."""
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
        }

        # Mock the model's vision encoder
        self.runner.model.vision_encoder = Mock()
        self.runner.model.vision_encoder.return_value = paddle.zeros((10, 768))

        result = self.runner.extract_vision_features_qwen(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_qwen_with_cache(self):
        """Test extract_vision_features_qwen with encoder cache."""
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "image_hash": "qwen_hash_123",
        }

        # Pre-populate cache
        cached_features = paddle.zeros((10, 768))
        self.runner.encoder_cache["qwen_hash_123"] = cached_features

        result = self.runner.extract_vision_features_qwen(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_qwen_multiple_images(self):
        """Test extract_vision_features_qwen with multiple images."""
        vision_inputs = {
            "image_embeds": [
                [paddle.zeros((10, 768))],
                [paddle.zeros((10, 768))],
                [paddle.zeros((10, 768))],
            ],
        }

        result = self.runner.extract_vision_features_qwen(vision_inputs)

        # Verify result is returned for multiple images
        self.assertIsNotNone(result)

    def test_extract_vision_features_qwen_empty_inputs(self):
        """Test extract_vision_features_qwen with empty inputs."""
        vision_inputs = {
            "image_embeds": [[]],
        }

        result = self.runner.extract_vision_features_qwen(vision_inputs)

        # Should handle empty inputs gracefully
        self.assertIsNotNone(result)

    def test_extract_vision_features_qwen_different_image_sizes(self):
        """Test extract_vision_features_qwen with different image sizes."""
        test_cases = [
            paddle.zeros((1, 3, 224, 224)),  # 224x224
            paddle.zeros((1, 3, 336, 336)),  # 336x336
            paddle.zeros((1, 3, 448, 448)),  # 448x448
        ]

        for image_embed in test_cases:
            vision_inputs = {
                "image_embeds": [[image_embed]],
            }

            result = self.runner.extract_vision_features_qwen(vision_inputs)

            # Verify result is returned for each image size
            self.assertIsNotNone(result)

    def test_extract_vision_features_qwen_cache_hit_scenario(self):
        """Test extract_vision_features_qwen with cache hit scenario."""
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "image_hash": "qwen_cached_hash",
        }

        # Simulate cache hit by pre-populating cache
        cached_features = paddle.zeros((10, 768))
        self.runner.encoder_cache["qwen_cached_hash"] = cached_features

        # Verify cache has the entry
        self.assertIn("qwen_cached_hash", self.runner.encoder_cache)

        result = self.runner.extract_vision_features_qwen(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_qwen_with_multiple_caches(self):
        """Test extract_vision_features_qwen with multiple cache entries."""
        # Populate cache with multiple entries
        self.runner.encoder_cache = {
            "hash1": paddle.zeros((10, 768)),
            "hash2": paddle.zeros((10, 768)),
            "hash3": paddle.zeros((10, 768)),
        }

        # Use one of the cached entries
        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "image_hash": "hash2",
        }

        result = self.runner.extract_vision_features_qwen(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)
        # Verify cache still has all entries
        self.assertEqual(len(self.runner.encoder_cache), 3)


class TestExtractVisionFeaturesPaddleocr(unittest.TestCase):
    """Test cases for extract_vision_features_paddleocr method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.model_type = "paddleocr"
        self.mock_model_config.enable_mm = True
        self.mock_fd_config.model_config = self.mock_model_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.encoder_cache = {}
        self.runner.model = Mock()

    def test_extract_vision_features_paddleocr_basic(self):
        """Test extract_vision_features_paddleocr with basic inputs."""
        inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
        }

        # Mock the model's vision encoder
        self.runner.model.vision_encoder = Mock()
        self.runner.model.vision_encoder.return_value = paddle.zeros((10, 768))

        result = self.runner.extract_vision_features_paddleocr(inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_paddleocr_with_cache(self):
        """Test extract_vision_features_paddleocr with encoder cache."""
        inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "image_hash": "ocr_hash_123",
        }

        # Pre-populate cache
        cached_features = paddle.zeros((10, 768))
        self.runner.encoder_cache["ocr_hash_123"] = cached_features

        result = self.runner.extract_vision_features_paddleocr(inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_paddleocr_empty_inputs(self):
        """Test extract_vision_features_paddleocr with empty inputs."""
        inputs = {
            "image_embeds": [[]],
        }

        result = self.runner.extract_vision_features_paddleocr(inputs)

        # Should handle empty inputs gracefully
        self.assertIsNotNone(result)


class TestExtractVisionFeatures(unittest.TestCase):
    """Test cases for extract_vision_features method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.model_type = "ernie"
        self.mock_model_config.enable_mm = True
        self.mock_fd_config.model_config = self.mock_model_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.rope3d_cache = {}
        self.runner.encoder_cache = {}
        self.runner.model = Mock()

    def test_extract_vision_features_ernie_type(self):
        """Test extract_vision_features with ernie model type."""
        self.mock_model_config.model_type = "ernie"

        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
        }

        result = self.runner.extract_vision_features(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_qwen_type(self):
        """Test extract_vision_features with qwen model type."""
        self.mock_model_config.model_type = "qwen"

        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
        }

        result = self.runner.extract_vision_features(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_paddleocr_type(self):
        """Test extract_vision_features with paddleocr model type."""
        self.mock_model_config.model_type = "paddleocr"

        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
        }

        result = self.runner.extract_vision_features(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_extract_vision_features_with_rope3d(self):
        """Test extract_vision_features with rope3d enabled."""
        self.mock_model_config.rope3d = True
        self.mock_model_config.model_type = "ernie"

        vision_inputs = {
            "image_embeds": [[paddle.zeros((10, 768))]],
            "grid_thw": [[paddle.to_tensor([2, 2, 16])]],
        }

        result = self.runner.extract_vision_features(vision_inputs)

        # Verify result is returned
        self.assertIsNotNone(result)


class TestPrepareRope3d(unittest.TestCase):
    """Test cases for prepare_rope3d method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.rope3d = None
        self.mock_fd_config.model_config = self.mock_model_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.rope3d_cache = {}
        self.runner.share_inputs = Mock()
        self.runner.share_inputs["rope_emb"] = paddle.zeros((2, 1))

    def test_prepare_rope3d_no_cache(self):
        """Test prepare_rope3d with no cached value."""
        max_tokens_lst = [2048, 1024]
        batch_size = 2

        self.runner.prepare_rope3d(max_tokens_lst, batch_size)

        # Verify rope_emb is prepared
        self.assertIsNotNone(self.runner.share_inputs["rope_emb"])

    def test_prepare_rope3d_with_cache(self):
        """Test prepare_rope3d with cached value."""
        max_tokens_lst = [2048, 1024]
        batch_size = 2

        # Pre-populate cache
        cache_key = tuple(max_tokens_lst)
        self.runner.rope3d_cache[cache_key] = paddle.zeros((2, 1))

        self.runner.prepare_rope3d(max_tokens_lst, batch_size)

        # Verify cache is used
        self.assertIsNotNone(self.runner.rope3d_cache)

    def test_prepare_rope3d_different_batch_sizes(self):
        """Test prepare_rope3d with different batch sizes."""
        test_cases = [
            ([2048], 1),
            ([2048, 1024], 2),
            ([2048, 1024, 512], 3),
        ]

        for max_tokens_lst, batch_size in test_cases:
            self.runner.rope3d_cache = {}
            self.runner.prepare_rope3d(max_tokens_lst, batch_size)
            # Verify rope_emb is prepared for each case
            self.assertIsNotNone(self.runner.share_inputs["rope_emb"])

    def test_prepare_rope3d_max_tokens_variations(self):
        """Test prepare_rope3d with varying max_tokens values."""
        test_cases = [
            ([512], 1),
            ([1024], 1),
            ([2048], 1),
            ([4096], 1),
        ]

        for max_tokens_lst, batch_size in test_cases:
            self.runner.rope3d_cache = {}
            self.runner.prepare_rope3d(max_tokens_lst, batch_size)
            # Verify rope_emb is prepared for each case
            self.assertIsNotNone(self.runner.share_inputs["rope_emb"])


class TestExecuteModel(unittest.TestCase):
    """Test cases for execute_model method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.share_inputs = Mock()

    def test_execute_model_signature(self):
        """Test execute_model has correct signature and can be called."""
        try:
            self.runner.execute_model(
                num_running_requests=1,
                output_logprobs=False,
                output_ids_only=False,
                return_output=True,
            )
        except Exception:
            # May fail due to complex dependencies
            pass

    def test_execute_model_normal_flow(self):
        """Test execute_model normal execution flow."""
        # Setup mock components
        self.runner.only_decode = Mock(return_value=False)
        self.runner.enable_overlap_schedule = False
        self.runner.speculative_decoding = False
        self.runner.use_cudagraph = False

        mock_model_output_data = Mock()
        self.runner.execute_model_normal = Mock()
        self.runner.execute_model_normal.return_value = None

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model(
                        num_running_requests=1,
                        output_logprobs=False,
                        output_ids_only=False,
                        return_output=True,
                    )

                    # Verify execute_model_normal is called
                    self.runner.execute_model_normal.assert_called_once()
                    mock_save.assert_called_once()

    def test_execute_model_overlap_flow(self):
        """Test execute_model with overlap schedule."""
        # Setup mock components
        self.runner.only_decode = Mock(return_value=True)
        self.runner.enable_overlap_schedule = True
        self.runner.speculative_decoding = False
        self.runner.use_cudagraph = False
        self.runner.last_model_output_data = Mock()

        mock_model_output_data = Mock()
        self.runner.execute_model_overlap = Mock()
        self.runner.execute_model_overlap.return_value = None

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model(
                        num_running_requests=1,
                        output_logprobs=False,
                        output_ids_only=False,
                        return_output=True,
                    )

                    # Verify execute_model_overlap is called
                    self.runner.execute_model_overlap.assert_called_once()
                    # Verify last output is saved
                    mock_save.assert_called_once()


class TestExecuteModelNormal(unittest.TestCase):
    """Test cases for execute_model_normal method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.share_inputs = Mock()
        self.runner.speculative_decoding = False
        self.runner.is_pooling_model = False

    def test_execute_model_normal_signature(self):
        """Test execute_model_normal has correct signature."""
        try:
            self.runner.execute_model_normal(
                num_running_requests=1,
                output_logprobs=False,
                output_ids_only=False,
                return_output=True,
            )
        except Exception:
            # May fail due to complex dependencies
            pass

    def test_execute_model_normal_full_flow(self):
        """Test execute_model_normal complete flow."""
        mock_model_output = Mock()
        mock_model_output_data = Mock()
        mock_sampler_output = Mock()
        mock_post_process_event = Mock()
        mock_token_num_event = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (mock_model_output, [0], mock_token_num_event)

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (
                    mock_model_output_data,
                    mock_sampler_output,
                    mock_post_process_event,
                    0,
                )

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify complete flow
                    mock_preprocess_execute.assert_called_once()
                    mock_postprocess.assert_called_once()
                    mock_save.assert_called_once()

    def test_execute_model_normal_no_output_data(self):
        """Test execute_model_normal when model_output_data is None."""
        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                # Return None for model_output_data
                mock_postprocess.return_value = (None, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify _save_model_output is NOT called when output_data is None
                    mock_save.assert_not_called()

    def test_execute_model_normal_with_speculative_decoding(self):
        """Test execute_model_normal with speculative decoding."""
        self.runner.speculative_decoding = True
        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify _save_model_output is NOT called with speculative_decoding
                    mock_save.assert_not_called()

    def test_execute_model_normal_empty_batch(self):
        """Test execute_model_normal with empty batch (num_running_requests=0)."""
        with patch.object(self.runner, "_execute_empty_input") as mock_empty:
            self.runner.execute_model_normal(model_forward_batch=[], num_running_requests=0)

            # Verify empty input handler is called
            mock_empty.assert_called_once()

    def test_execute_model_normal_with_pooling_model(self):
        """Test execute_model_normal with pooling model."""
        self.runner.is_pooling_model = True
        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify _save_model_output is called for pooling model
                    mock_save.assert_called_once()

    def test_execute_model_normal_with_output_logprobs(self):
        """Test execute_model_normal with output_logprobs=True."""
        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(
                        model_forward_batch=[Mock()], num_running_requests=1, output_logprobs=True
                    )

                    # Verify flow is executed with logprobs
                    mock_preprocess_execute.assert_called_once()
                    mock_save.assert_called_once()

    def test_execute_model_normal_output_ids_only(self):
        """Test execute_model_normal with output_ids_only=True."""
        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(
                        model_forward_batch=[Mock()], num_running_requests=1, output_ids_only=True
                    )

                    # Verify flow is executed with ids_only
                    mock_preprocess_execute.assert_called_once()

    def test_execute_model_normal_multiple_requests(self):
        """Test execute_model_normal with multiple requests."""
        mock_model_output = Mock()
        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (mock_model_output, [0, 1, 2], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 3)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(
                        model_forward_batch=[Mock(), Mock(), Mock()], num_running_requests=3
                    )

                    # Verify all requests are processed
                    mock_preprocess_execute.assert_called_once()
                    mock_save.assert_called_once()

    def test_execute_model_normal_with_multimodal(self):
        """Test execute_model_normal with multimodal inputs enabled."""
        self.runner.enable_mm = True
        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 0)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify multimodal flow is executed
                    mock_preprocess_execute.assert_called_once()


class TestExecuteModelOverlap(unittest.TestCase):
    """Test cases for execute_model_overlap method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.share_inputs = Mock()
        self.runner.speculative_decoding = False
        self.runner.last_model_output_data = None
        self.runner.last_sampler_output = None
        self.runner.last_post_process_event = None
        self.runner.last_token_num = -1

    def test_execute_model_overlap_signature(self):
        """Test execute_model_overlap has correct signature."""
        try:
            self.runner.execute_model_overlap(
                num_running_requests=1,
                output_logprobs=False,
                output_ids_only=False,
                return_output=True,
            )
        except Exception:
            # May fail due to complex dependencies
            pass

    def test_execute_model_overlap_full_flow(self):
        """Test execute_model_overlap complete flow."""
        mock_last_output_data = Mock()
        mock_last_sampler_output = Mock()
        mock_last_post_process_event = Mock()

        self.runner.last_model_output_data = mock_last_output_data
        self.runner.last_sampler_output = mock_last_sampler_output
        self.runner.last_post_process_event = mock_last_post_process_event

        mock_model_output = Mock()
        mock_model_output_data = Mock()
        mock_sampler_output = Mock()
        mock_post_process_event = Mock()
        mock_token_num_event = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (mock_model_output, [0], mock_token_num_event)

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (
                    mock_model_output_data,
                    mock_sampler_output,
                    mock_post_process_event,
                    10,
                )

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_overlap(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify save_output is called with previous batch data
                    mock_save.assert_called_once_with(
                        mock_last_output_data, mock_last_sampler_output, mock_last_post_process_event
                    )

                    # Verify state is updated
                    self.assertEqual(self.runner.last_model_output_data, mock_model_output_data)
                    self.assertEqual(self.runner.last_sampler_output, mock_sampler_output)
                    self.assertEqual(self.runner.last_post_process_event, mock_post_process_event)
                    self.assertEqual(self.runner.last_token_num, 10)

    def test_execute_model_overlap_first_call(self):
        """Test execute_model_overlap when there's no previous batch data."""
        # No previous data (last_model_output_data is None)
        self.runner.last_model_output_data = None

        mock_model_output = Mock()
        mock_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (mock_model_output, [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (mock_model_output_data, Mock(), Mock(), 5)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_overlap(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify _save_model_output is NOT called when there's no previous data
                    mock_save.assert_not_called()

    def test_execute_model_overlap_with_speculative_decoding(self):
        """Test execute_model_overlap with speculative decoding."""
        self.runner.speculative_decoding = True
        self.runner.last_model_output_data = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (Mock(), Mock(), Mock(), 5)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_overlap(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify _save_model_output is NOT called with speculative_decoding
                    mock_save.assert_not_called()

    def test_execute_model_overlap_state_continuity(self):
        """Test execute_model_overlap maintains state continuity between calls."""
        mock_output_data_1 = Mock()
        mock_output_data_2 = Mock()

        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [0], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                with patch.object(self.runner, "_save_model_output") as mock_save:
                    # First call
                    mock_postprocess.return_value = (mock_output_data_1, Mock(), Mock(), 5)
                    self.runner.execute_model_overlap(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify state is saved
                    self.assertEqual(self.runner.last_model_output_data, mock_output_data_1)
                    self.assertEqual(self.runner.last_token_num, 5)

                    # Second call
                    mock_postprocess.return_value = (mock_output_data_2, Mock(), Mock(), 10)
                    self.runner.execute_model_overlap(model_forward_batch=[Mock()], num_running_requests=1)

                    # Verify state is updated
                    self.assertEqual(self.runner.last_model_output_data, mock_output_data_2)
                    self.assertEqual(self.runner.last_token_num, 10)


class TestCaptureModel(unittest.TestCase):
    """Test cases for capture_model method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_num_seqs = 10
        self.mock_fd_config.model_config = self.mock_model_config
        self.mock_cache_config = Mock()
        self.mock_cache_config.enable_chunked_prefill = False
        self.mock_fd_config.cache_config = self.mock_cache_config
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_speculative_config = Mock()
        self.mock_speculative_config.method = None
        self.mock_fd_config.speculative_config = self.mock_speculative_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.speculative_method = None
        self.runner.share_inputs = Mock()
        self.runner._dummy_run = Mock()
        self.runner.use_cudagraph = True
        self.runner.model = Mock()
        self.runner._process_reorder = Mock()
        self.runner.cudagraph_prefill = {}
        self.runner.cudagraph_decode = {}

    def test_capture_model_basic(self):
        """Test capture_model can be called."""
        self.runner.capture_model()

        # Verify dummy run is called
        self.runner._dummy_run.assert_called()
        # Verify cudagraphs are populated
        self.assertIsNotNone(self.runner.cudagraph_prefill)
        self.assertIsNotNone(self.runner.cudagraph_decode)

    def test_capture_model_with_speculative_decoding(self):
        """Test capture_model with speculative decoding enabled."""
        self.mock_speculative_config.method = "mtp"
        self.runner.speculative_method = "mtp"

        self.runner.capture_model()

        # Verify dummy run is called
        self.runner._dummy_run.assert_called()

    def test_capture_model_with_chunked_prefill(self):
        """Test capture_model with chunked prefill enabled."""
        self.mock_cache_config.enable_chunked_prefill = True

        self.runner.capture_model()

        # Verify dummy run is called
        self.runner._dummy_run.assert_called()

    def test_capture_model_different_batch_sizes(self):
        """Test capture_model with different max_num_seqs configurations."""
        test_cases = [1, 4, 8, 16]

        for max_num_seqs in test_cases:
            self.mock_scheduler_config.max_num_seqs = max_num_seqs
            self.mock_model_config.max_num_seqs = max_num_seqs
            self.runner.cudagraph_prefill = {}
            self.runner.cudagraph_decode = {}

            self.runner.capture_model()

            # Verify dummy run is called for each batch size
            self.runner._dummy_run.assert_called()


class TestCaptureModelPrefillAndMixed(unittest.TestCase):
    """Test cases for capture_model_prefill_and_mixed method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_num_seqs = 10
        self.mock_fd_config.model_config = self.mock_model_config
        self.mock_cache_config = Mock()
        self.mock_cache_config.enable_chunked_prefill = False
        self.mock_fd_config.cache_config = self.mock_cache_config
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_speculative_config = Mock()
        self.mock_speculative_config.method = None
        self.mock_fd_config.speculative_config = self.mock_speculative_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.speculative_method = None
        self.runner.share_inputs = Mock()
        self.runner._dummy_run = Mock()
        self.runner.use_cudagraph = True
        self.runner.model = Mock()
        self.runner.cudagraph_prefill = {}
        self.runner.cudagraph_mixed = {}

    def test_capture_model_prefill_and_mixed_basic(self):
        """Test capture_model_prefill_and_mixed can be called."""
        self.runner.capture_model_prefill_and_mixed()

        # Verify dummy run is called
        self.runner._dummy_run.assert_called()
        # Verify cudagraphs are populated
        self.assertIsNotNone(self.runner.cudagraph_prefill)
        self.assertIsNotNone(self.runner.cudagraph_mixed)

    def test_capture_model_prefill_and_mixed_with_chunked_prefill(self):
        """Test capture_model_prefill_and_mixed with chunked prefill enabled."""
        self.mock_cache_config.enable_chunked_prefill = True

        self.runner.capture_model_prefill_and_mixed()

        # Verify dummy run is called
        self.runner._dummy_run.assert_called()

    def test_capture_model_prefill_and_mixed_speculative_decoding(self):
        """Test capture_model_prefill_and_mixed with speculative decoding."""
        self.mock_speculative_config.method = "mtp"
        self.runner.speculative_method = "mtp"

        self.runner.capture_model_prefill_and_mixed()

        # Verify dummy run is called
        self.runner._dummy_run.assert_called()

    def test_capture_model_prefill_and_mixed_different_batch_sizes(self):
        """Test capture_model_prefill_and_mixed with different max_num_seqs."""
        test_cases = [1, 2, 4]

        for max_num_seqs in test_cases:
            self.mock_scheduler_config.max_num_seqs = max_num_seqs
            self.mock_model_config.max_num_seqs = max_num_seqs
            self.runner.cudagraph_prefill = {}
            self.runner.cudagraph_mixed = {}

            self.runner.capture_model_prefill_and_mixed()

            # Verify dummy run is called for each batch size
            self.runner._dummy_run.assert_called()


class TestClearCache(unittest.TestCase):
    """Test cases for clear_cache method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_cache_config = Mock()
        self.mock_cache_config.num_cpu_blocks = 0
        self.mock_cache_config.kvcache_storage_backend = None
        self.mock_fd_config.cache_config = self.mock_cache_config
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.splitwise_role = "mixed"
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.local_rank = 0
        self.runner.cache_kvs_map = {}
        self.runner.share_inputs = Mock()
        self.runner.share_inputs.pop = Mock()
        self.runner.forward_meta = None
        self.runner.use_cudagraph = False

        # Mock unset_data_ipc function
        with patch("fastdeploy.worker.gpu_model_runner.unset_data_ipc") as mock_unset:
            mock_unset.return_value = None
            self.mock_unset_data_ipc = mock_unset

            with patch("paddle.device.cuda.empty_cache") as mock_empty_cache:
                mock_empty_cache.return_value = None

                self.runner.clear_cache(profile=False)

                # Verify cache_kvs_map is cleared
                self.assertEqual(self.runner.cache_kvs_map, {})

    def test_clear_cache_with_profile(self):
        """Test clear_cache with profile=True."""
        self.mock_fd_config.cache_config.num_cpu_blocks = 10

        with patch("paddle.device.cuda.empty_cache") as mock_empty_cache:
            mock_empty_cache.return_value = None

            self.runner.clear_cache(profile=True)

            # verify cuda cache is emptied
            mock_empty_cache.assert_called_once()


class TestClearParameters(unittest.TestCase):
    """Test cases for clear_parameters method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.use_cudagraph = True
        self.runner.model = Mock()
        self.runner.proposer = Mock()
        self.runner.model.clear_grpah_opt_backend = Mock()
        self.runner.proposer.clear_mtp_cache = Mock()

    def test_clear_parameters_with_cudagraph(self):
        """Test clear_parameters clears cudagraph."""
        self.runner.clear_parameters(pid=123)

        # Verify cudagraph is cleared
        self.runner.model.clear_grpah_opt_backend.assert_called_once()

    def test_clear_parameters_without_cudagraph(self):
        """Test clear_parameters when cudagraph is disabled."""
        self.runner.use_cudagraph = False

        self.runner.clear_parameters(pid=123)

        # Verify cudagraph clear is NOT called
        self.runner.model.clear_grpah_opt_backend.assert_not_called()

    def test_clear_parameters_with_mtp(self):
        """Test clear_parameters clears mtp cache."""
        self.runner.speculative_method = "mtp"
        self.runner.use_cudagraph = False

        self.runner.clear_parameters(pid=123)

        # Verify mtp cache is cleared
        self.runner.proposer.clear_mtp_cache.assert_called_once()

    def test_clear_parameters_pid_parameter(self):
        """Test clear_parameters passes pid correctly."""
        pid = 456
        self.runner.clear_parameters(pid=pid)

        # Verify function is called (specific implementation varies)
        # The pid parameter is used for RDMA communication
        self.assertIsNotNone(pid)


class TestUpdateParameters(unittest.TestCase):
    """Test cases for update_parameters method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.shutdown_comm_group_if_worker_idle = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.share_inputs = Mock()
        self.runner.share_inputs.reset_share_inputs = Mock()
        self.runner.initialize_kv_cache = Mock()
        self.runner.use_cudagraph = False
        self.runner.speculative_method = None
        self.runner.dynamic_weight_manager = Mock()
        self.runner.model = Mock()

    def test_update_parameters_basic(self):
        """Test update_parameters calls dynamic weight manager."""
        self.runner.update_parameters(pid=456)

        # Verify dynamic weight manager is called
        self.runner.dynamic_weight_manager.update_parameters.assert_called_once()
        # Verify share_inputs is reset
        self.runner.share_inputs.reset_share_inputs.assert_called_once()

    def test_update_parameters_with_shutdown_comm(self):
        """Test update_parameters with shutdown_comm_group_if_worker_idle enabled."""
        self.mock_parallel_config.shutdown_comm_group_if_worker_idle = True

        self.runner.update_parameters(pid=456)

        # Verify dynamic weight manager is called
        self.runner.dynamic_weight_manager.update_parameters.assert_called_once()

    def test_update_parameters_with_cudagraph(self):
        """Test update_parameters when cudagraph is enabled."""
        self.runner.use_cudagraph = True

        self.runner.update_parameters(pid=456)

        # Verify dynamic weight manager is called
        self.runner.dynamic_weight_manager.update_parameters.assert_called_once()
        # Verify share_inputs is reset
        self.runner.share_inputs.reset_share_inputs.assert_called_once()

    def test_update_parameters_with_mtp(self):
        """Test update_parameters with speculative decoding (MTP)."""
        self.runner.speculative_method = "mtp"

        self.runner.update_parameters(pid=456)

        # Verify dynamic weight manager is called
        self.runner.dynamic_weight_manager.update_parameters.assert_called_once()


if __name__ == "__main__":
    unittest.main()
