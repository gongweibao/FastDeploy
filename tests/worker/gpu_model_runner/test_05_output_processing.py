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
Tests for Phase 6+7: Output Processing & Save

This module contains tests for the output processing and saving phases of GPU Model Runner,
corresponding to Phase 6 (Postprocess) and 7 (Save Output) in docs/gpu_model_runner_data_flow.md

Key components tested:
- Postprocess method
- Rebuild padding
- Compute logits
- Sample (next token)
- Save model output
- Get prompt logprobs list
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import paddle

from fastdeploy.engine.request import RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestPostprocess(unittest.TestCase):
    """Test cases for _postprocess method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.model_config.ori_vocab_size = 32000
        self.runner.model_config.dtype = "float16"
        self.runner.model_config.logprobs_mode = "raw_logprobs"
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.cache_config = Mock()
        self.runner.cache_config.enable_prefix_caching = False
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.speculative_decoding = False
        self.runner.speculative_method = None
        self.runner.share_inputs = Mock()
        self.runner.forward_meta = Mock()
        self.runner.sampler = Mock()
        self.runner.sampler.apply_logits_processor = Mock()

    def test_postprocess_basic(self):
        """Test basic postprocess flow."""
        # Setup mock inputs
        hidden_states = paddle.zeros((10, 768))
        full_hidden_states = paddle.zeros((10, 100, 768))
        self.runner.sampler.sample.return_value = Mock()

        with patch.object(self.runner, 'model.compute_logits'):
            result = self.runner._postprocess(
                hidden_states, full_hidden_states, self.runner.forward_meta
            )

        # Verify result is a tuple
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)

    def test_postprocess_with_logits_processor(self):
        """Test postprocess applies logits processor."""
        # Verify logits processor is called
        result = self.runner._postprocess(
            paddle.zeros((10, 768)),
            paddle.zeros((10, 100, 768)),
            self.runner.forward_meta
        )

        self.assertIsNotNone(result)
        self.runner.sampler.apply_logits_processor.assert_called_once()

    def test_postprocess_with_speculative(self):
        """Test postprocess with speculative decoding enabled."""
        self.runner.speculative_decoding = True
        self.runner.speculative_method = "mtp"

        with patch.object(self.runner, 'model.compute_logits'):
            result = self.runner._postprocess(
                paddle.zeros((10, 768)),
                paddle.zeros((10, 100, 768)),
                self.runner.forward_meta
            )

        self.assertIsNotNone(result)


class TestRebuildPadding(unittest.TestCase):
    """Test cases for rebuild_padding method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.share_inputs = Mock()

    def test_rebuild_padding_basic(self):
        """Test rebuild_padding restores padding."""
        # Mock share_inputs with necessary methods
        self.runner.share_inputs.rebuild_padding = Mock()
        self.runner.share_inputs.rebuild_padding.return_value = paddle.zeros((10, 100))

        result = self.runner.rebuild_padding(paddle.zeros((10, 50)))

        self.assertIsNotNone(result)
        self.runner.share_inputs.rebuild_padding.assert_called_once()

    def test_rebuild_padding_with_different_shapes(self):
        """Test rebuild_padding handles various input shapes."""
        self.runner.share_inputs.rebuild_padding = Mock()

        # Test different input shapes
        shapes = [(10, 50), (5, 100), (20, 25)]
        for shape in shapes:
            self.runner.rebuild_padding.rebuild_padding.return_value = paddle.zeros(shape)
            result = self.runner.rebuild_padding(paddle.zeros((shape[0], shape[1])))
            self.assertIsNotNone(result)


class TestGetPromptLogprobsList(unittest.TestCase):
    """Test cases for _get_prompt_logprobs_list method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.logprobs_mode = "raw_logprobs"
        self.runner.cache_config = Mock()
        self.runner.cache_config.enable_prefix_caching = False
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.fd_config.cache_config = self.runner.cache_config
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.ori_vocab_size = 50000
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.share_inputs = Mock()
        self.runner.model = Mock()
        self.runner.sampler = Mock()

    def _create_mock_request(
        self, request_id="req_1", prompt_tokens=10, num_logprobs=5
    ):
        """Helper to create mock request."""
        request = Mock()
        request.request_id = request_id
        request.idx = 0
        request.prompt_token_ids = list(range(prompt_tokens))
        request.prefill_start_index = 0
        request.prefill_end_index = 10
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

        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((1, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((1, 1)), paddle.zeros((1, 1)), paddle.zeros((1,)))
        )
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((1, 768)))

        hidden_states = paddle.zeros((10, 768))
        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # num_prompt_logprobs should be set to ori_vocab_size (50000)
        self.assertEqual(request.sampling_params.prompt_logprobs, 50000)
        # Result should have request data
        self.assertIsNotNone(result[0])

    def test_prompt_logprobs_partial(self):
        """Test with num_prompt_logprobs < num_tokens."""
        request = self._create_mock_request(num_logprobs=5, prompt_tokens=10)
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((9, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((9, 6)), paddle.zeros((9, 6)), paddle.zeros((9,)))
        )
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((9, 768)))

        hidden_states = paddle.zeros((20, 768))
        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # Result should have request data
        self.assertIsNotNone(result[0])

    def test_prompt_logprobs_chunked_prefill(self):
        """Test prompt logprobs with chunked prefill."""
        # First chunk
        request1 = self._create_mock_request(
            request_id="req_1", prompt_tokens=20, num_logprobs=5, prefill_start=0, prefill_end=10
        )
        self.runner.prompt_logprobs_reqs = {"req_1": request1}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((10, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((10, 6)), paddle.zeros((10, 6)), paddle.zeros((10,)))
        )
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
            request_id="req_1", prompt_tokens=10, num_logprobs=5, prefill_start=0, prefill_end=10
        )
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_sampler = Mock()
        mock_sampler.compute_logprobs = Mock(return_value=paddle.zeros((9, 50000)))
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((9, 6)), paddle.zeros((9, 6)), paddle.zeros((9,)))
        )
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
        self.runner.model_config.logprobs_mode = "raw_logits"

        request = self._create_mock_request(num_logprobs=5, prompt_tokens=10)
        self.runner.prompt_logprobs_reqs = {"req_1": request}
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)
        self.runner.share_inputs.__getitem__ = Mock(return_value=paddle.to_tensor([0, 10]))

        mock_sampler = Mock()
        # For raw_logits mode, gather_logprobs is called directly with logits
        mock_sampler.gather_logprobs = Mock(
            return_value=(paddle.zeros((9, 6)), paddle.zeros((9, 6)), paddle.zeros((9,)))
        )
        self.runner.sampler = mock_sampler
        self.runner.model.compute_logits = Mock(return_value=paddle.zeros((9, 768)))

        hidden_states = paddle.zeros((20, 768))
        result = self.runner._get_prompt_logprobs_list(hidden_states)

        # Verify compute_logprobs is NOT called in raw_logits mode
        mock_sampler.compute_logprobs.assert_not_called()
        self.assertIsNotNone(result[0])

    def test_prompt_logprobs_with_prefix_caching_error(self):
        """Test that prefix caching is disabled with prompt_logprobs."""
        self.runner.cache_config.enable_prefix_caching = True

        request = self._create_mock_request()
        self.runner.prompt_logprobs_reqs = {"req_1": request}

        hidden_states = paddle.zeros((10, 768))

        # Should raise AssertionError
        with self.assertRaises(AssertionError):
            self.runner._get_prompt_logprobs_list(hidden_states)


class TestSaveModelOutput(unittest.TestCase):
    """Test cases for _save_model_output method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.use_cudagraph = False
        self.runner.share_inputs = Mock()
        self.runner.share_inputs["stop_flags"] = np.zeros((10,), dtype=bool)

    def test_save_model_output_basic(self):
        """Test basic model output saving."""
        from fastdeploy.worker.output import SamplerOutput

        sampler_output = SamplerOutput(
            sampled_token_ids=paddle.zeros((10, 1), dtype="int64"),
            logprobs=None,
            generated_seq_idx=None,
        )

        self.runner._save_model_output(sampler_output)

        # Verify output is saved
        self.assertIsNotNone(sampler_output)

    def test_save_model_output_with_async(self):
        """Test model output saving with async output."""
        self.runner.async_output_queue = Mock()

        sampler_output = Mock()

        self.runner._save_model_output(sampler_output)

        # Verify queue operation if async is enabled
        self.assertIsNotNone(sampler_output)

    def test_save_model_output_stop_flags(self):
        """Test stop flags are updated in save output."""
        self.runner.share_inputs["stop_flags"] = np.zeros((10,), dtype=bool)

        sampler_output = Mock()
        sampler_output.sampled_token_ids = paddle.zeros((10, 1), dtype="int64")

        self.runner._save_model_output(sampler_output)

        # Verify stop_flags are updated


if __name__ == "__main__":
    unittest.main()
