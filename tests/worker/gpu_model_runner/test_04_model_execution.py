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
Tests for Phase 5: Model Execution

This module contains tests for the model execution phase of GPU Model Runner,
corresponding to Phase 5 (Preprocessing + Model Execution) in docs/gpu_model_runner_data_flow.md

Key components tested:
- Preprocess and execute model
- Collect distributed status
- Execute model (normal/overlap)
- Input preparation
- Reorder processing
"""

import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.engine.request import RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestPreprocessAndExecuteModel(unittest.TestCase):
    """Test cases for _preprocess_and_execute_model method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.use_cudagraph = False
        self.runner.share_inputs = Mock()
        self.runner.forward_meta = Mock()
        self.runner.forward_meta.is_dummy_or_profile_run = False
        self.runner.speculative_decoding = False

    def test_preprocess_and_execute_model_basic(self):
        """Test basic preprocess and execute flow."""
        # Mock the internal components
        self.runner._process_reorder = Mock()
        self.runner._prepare_inputs = Mock()
        self.runner.model = Mock()

        # Mock model forward return value
        hidden_states = paddle.zeros((10, 768))
        self.runner.model.forward.return_value = hidden_states

        with patch.object(self.runner, 'model.forward') as mock_forward:
            mock_forward.return_value = hidden_states
            result = self.runner._preprocess_and_execute_model()

        # Verify result is returned
        self.assertIsNotNone(result)

    def test_preprocess_and_execute_model_with_cudagraph(self):
        """Test preprocess and execute with CUDA Graph enabled."""
        self.runner.use_cudagraph = True
        self.runner._process_reorder = Mock()
        self.runner._prepare_inputs = Mock()
        self.runner.model = Mock()

        hidden_states = paddle.zeros((10, 768))
        self.runner.model.forward.return_value = hidden_states

        with patch.object(self.runner, 'model.forward') as mock_forward:
            mock_forward.return_value = hidden_states
            result = self.runner._preprocess_and_execute_model()

        self.assertIsNotNone(result)

    def test_preprocess_and_execute_model_dummy_run(self):
        """Test preprocess and execute in dummy/profile run mode."""
        self.runner.forward_meta.is_dummy_or_profile_run = True
        self.runner._process_reorder = Mock()
        self.runner._prepare_inputs = Mock()
        self.runner.model = Mock()

        with patch.object(self.runner, 'model.forward'):
            result = self.runner._preprocess_and_execute_model()

        self.assertIsNotNone(result)


class TestCollectDistributedStatus(unittest.TestCase):
    """Test cases for collect_distributed_status method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.splitwise_role = "decoder"
        self.runner.fd_config.scheduler_config = self.runner.scheduler_config
        self.runner.parallel_config = Mock()
        self.runner.parallel_config.use_ep = False
        self.runner.parallel_config.enable_chunked_moe = False
        self.runner.fd_config.parallel_config = self.runner.parallel_config
        self.runner.share_inputs = Mock()
        self.runner.share_inputs["ids_remove_padding"] = paddle.zeros((10,), dtype="int32")
        self.runner.forward_meta = Mock()
        self.runner.forward_meta.moe_num_chunk = 1

    def test_collect_distributed_status_basic(self):
        """Test collect_distributed_status returns a DistributedOut object."""
        result = self.runner.collect_distributed_status()

        # Verify result is a DistributedOut object (Mock proxy)
        self.assertIsNotNone(result)

    def test_collect_distributed_status_with_ep(self):
        """Test collect_distributed_status with expert parallel."""
        self.runner.parallel_config.use_ep = True
        self.runner.scheduler_config.splitwise_role = "mixed"
        self.runner.exist_prefill = Mock(return_value=True)

        result = self.runner.collect_distributed_status()

        self.assertIsNotNone(result)


class TestExecuteModelNormal(unittest.TestCase):
    """Test cases for execute_model_normal method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.use_cudagraph = False
        self.runner.speculative_decoding = False
        self.runner.share_inputs = Mock()
        self.runner.forward_meta = Mock()
        self.runner.exist_prefill = Mock(return_value=False)
        self.runner.only_prefill = Mock(return_value=False)

    def test_execute_model_normal_basic(self):
        """Test execute_model_normal with basic input."""
        # Mock internal methods
        self.runner._preprocess_and_execute_model = Mock()
        self.runner._postprocess = Mock()
        self.runner._save_model_output = Mock()

        # Setup mock returns
        hidden_states = paddle.zeros((10, 768))
        full_hidden_states = paddle.zeros((10, 100, 768))
        sampler_output = Mock()
        sampler_output.sampled_token_ids = paddle.zeros((10, 1), dtype="int64")

        self.runner._preprocess_and_execute_model.return_value = (
            hidden_states, full_hidden_states, self.runner.forward_meta
        )
        self.runner._postprocess.return_value = (
            sampler_output, sampler_output, sampler_output, 5
        )

        model_forward_batch = [Mock()]
        num_running_requests = 1

        self.runner.execute_model_normal(model_forward_batch, num_running_requests)

        # Verify preprocessing was called
        self.runner._preprocess_and_execute_model.assert_called_once()
        # Verify postprocessing was called
        self.runner._postprocess.assert_called_once()
        # Verify save output was called
        self.runner._save_model_output.assert_called_once()


class TestExecuteModelOverlap(unittest.TestCase):
    """Test cases for execute_model_overlap method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.use_cudagraph = False
        self.runner.speculative_decoding = False
        self.runner.enable_overlap_schedule = False
        self.runner.share_inputs = Mock()
        self.runner.last_model_output_data = None

    def test_execute_model_overlap_basic(self):
        """Test execute_model_overlap with basic input."""
        # Mock internal methods
        self.runner._preprocess_and_execute_model = Mock()
        self.runner._postprocess = Mock()
        self.runner._save_model_output = Mock()

        # Setup mock returns
        hidden_states = paddle.zeros((10, 768))
        full_hidden_states = paddle.zeros((10, 100, 768))
        sampler_output = Mock()

        self.runner._preprocess_and_execute_model.return_value = (
            hidden_states, full_hidden_states, self.runner.forward_meta
        )
        self.runner._postprocess.return_value = (
            sampler_output, sampler_output, sampler_output, 5
        )

        model_forward_batch = [Mock()]
        num_running_requests = 1

        self.runner.execute_model_overlap(model_forward_batch, num_running_requests)

        # Verify preprocessing was called
        self.runner._preprocess_and_execute_model.assert_called_once()
        # Verify postprocessing was called
        self.runner._postprocess.assert_called_once()


class TestPrepareInputs(unittest.TestCase):
    """Test cases for _prepare_inputs method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.use_cudagraph = False
        self.runner.share_inputs = Mock()
        self.runner.forward_meta = Mock()
        self.runner.sampler = Mock()

    def test_prepare_inputs_basic(self):
        """Test basic input preparation."""
        # Mock the pre_process call
        self.runner.share_inputs.pre_process = Mock()

        with patch.object(self.runner, 'sampler.pre_process'):
            result = self.runner._prepare_inputs()

        self.assertIsNotNone(result)

    def test_prepare_inputs_with_vision(self):
        """Test input preparation with multimodal enabled."""
        self.runner.model_config.enable_mm = True

        # Vision features should be handled
        result = self.runner._prepare_inputs()

        self.assertIsNotNone(result)


class TestProcessReorder(unittest.TestCase):
    """Test cases for _process_reorder method."""

    def setUp(self):
        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = Mock()
        self.runner.model_config = Mock()
        self.runner.model_config.enable_mm = False
        self.runner.fd_config.model_config = self.runner.model_config
        self.runner.use_cudagraph = False
        self.runner.exist_prefill = Mock(return_value=False)
        self.runner.share_inputs = Mock()

    def test_process_reorder_decode_only(self):
        """Test reorder when only decode exists."""
        self.runner.exist_prefill.return_value = False

        with patch.object(self.runner, 'share_inputs.condense'):
            result = self.runner._process_reorder()

        self.assertIsNotNone(result)

    def test_process_reorder_mixed(self):
        """Test reorder with mixed prefill and decode."""
        self.runner.exist_prefill.return_value = True

        with patch.object(self.runner, 'share_inputs.condense'):
            result = self.runner._process_reorder()

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
