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

"""Unit tests for simple public methods of GPUModelRunner."""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import paddle

from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestNotNeedStop(unittest.TestCase):
    """Test cases for not_need_stop method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_fd_config.scheduler_config = Mock()
        self.mock_fd_config.scheduler_config.splitwise_role = "mixed"
        self.mock_fd_config.cache_config = Mock()
        self.mock_fd_config.cache_config.num_cpu_blocks = 0
        self.mock_fd_config.cache_config.kvcache_storage_backend = None
        self.mock_fd_config.scheduler_config.splitwise_role = "mixed"
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.share_inputs = {"not_need_stop": paddle.zeros((1,), dtype="int32")}

    def test_not_need_stop_false(self):
        """Test not_need_stop returns False."""
        self.runner.share_inputs["not_need_stop"] = paddle.full((1,), 0, dtype="int32")

        result = self.runner.not_need_stop()

        self.assertFalse(result)

    def test_not_need_stop_true(self):
        """Test not_need_stop returns True."""
        self.runner.share_inputs["not_need_stop"] = paddle.full((1,), 1, dtype="int32")

        result = self.runner.not_need_stop()

        self.assertTrue(result)


class TestGetModel(unittest.TestCase):
    """Test cases for get_model method."""

    def setUp(self):
        self.mock_model = Mock()

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.model = self.mock_model

    def test_get_model_returns_model(self):
        """Test get_model returns the model."""
        result = self.runner.get_model()

        self.assertEqual(result, self.mock_model)


class TestInsertPrefillInputs(unittest.TestCase):
    """Test cases for insert_prefill_inputs method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_model_len = 4096
        self.mock_model_config.eos_tokens_lens = 1
        self.mock_model_config.max_stop_seqs_num = 4
        self.mock_model_config.enable_mm = False
        self.mock_model_config.bad_tokens_len = 1
        self.mock_model_config.max_prompt_embedding_table_size = 0
        self.mock_fd_config.model_config = self.mock_model_config
        self.mock_cache_config = Mock()
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.enc_dec_block_num = 0
        self.mock_fd_config.cache_config = self.mock_cache_config

        from fastdeploy.engine.request import RequestType

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.is_pooling_model = False
        self.runner.RequestType = RequestType
        self.runner.exist_prefill_flag = False
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.pooling_params = []
        self.runner.enable_mm = False

    def _create_mock_request(self, task_type=0, idx=0, **kwargs):
        """Helper to create mock request."""
        request = Mock()
        request.task_type = Mock()
        request.task_type.value = task_type
        request.idx = idx
        request.request_id = f"req_{idx}"
        request.prompt_token_ids = kwargs.get('prompt_token_ids', [1, 2, 3, 4, 5])
        request.output_token_ids = []
        request.block_tables = kwargs.get('block_tables', [0, 1, 2])
        request.eos_token_ids = [0]
        request.prefill_start_index = 0
        request.prefill_end_index = len(request.prompt_token_ids)
        request.sampling_params = kwargs.get('sampling_params', Mock())
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []
        request.sampling_params.min_tokens = 1
        request.sampling_params.max_tokens = 100
        request.sampling_params.temperature = 1.0
        request.sampling_params.top_p = 1.0
        request.sampling_params.top_k = 0
        request.sampling_params.bad_tokens = []
        request.sampling_params.bad_tokens_len = 0

        def mock_get(key, default=None):
            return kwargs.get(key, default)
        request.get = mock_get

        return request

    def _setup_share_inputs_mock(self):
        """Setup share_inputs mock with required attributes."""
        share_inputs = Mock()
        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)
        share_inputs.__getitem__ = Mock(side_effect=lambda key: {
            "req_ids": [""] * 10,
            "stop_flags": np.zeros(10, dtype=bool),
            "seq_lens_decoder": np.zeros(10, dtype="int32"),
            "seq_lens_encoder": np.zeros(10, dtype="int32"),
            "seq_lens_this_time_buffer": np.zeros(10, dtype="int32"),
            "prompt_ids": np.zeros((10, 512), dtype="int64"),
            "input_ids": np.zeros((10, 512), dtype="int64"),
            "block_tables": np.full((10, 128), -1, dtype="int32"),
            "step_seq_lens_decoder": np.zeros(10, dtype="int32"),
            "prompt_lens": np.zeros(10, dtype="int32"),
            "top_p": np.zeros(10, dtype="float32"),
            "top_k": np.zeros(10, dtype="int32"),
            "temperature": np.zeros(10, dtype="float32"),
            "penalty_score": np.ones(10, dtype="float32"),
            "frequency_score": np.zeros(10, dtype="float32"),
            "presence_score": np.zeros(10, dtype="float32"),
            "max_dec_len": np.zeros(10, dtype="int32"),
            "min_dec_len": np.zeros(10, dtype="int32"),
            "eos_token_id": np.zeros((1, 1), dtype="int64"),
            "infer_seed": np.zeros(10, dtype="int64"),
        }[key])
        return share_inputs

    def test_insert_prefill_inputs_basic(self):
        """Test insert_prefill_inputs with basic prefill task."""
        self.runner.share_inputs = self._setup_share_inputs_mock()

        req = self._create_mock_request(
            task_type=self.runner.RequestType.PREFILL.value,
            idx=0,
        )

        # Insert the request
        self.runner.insert_prefill_inputs([req], num_running_requests=1)

        # Verify prefill flag is set
        self.assertTrue(self.runner.exist_prefill_flag)

    def test_insert_prefill_inputs_empty_list(self):
        """Test insert_prefill_inputs with empty request list."""
        self.runner.share_inputs = self._setup_share_inputs_mock()

        # Insert with empty list - should handle gracefully
        self.runner.insert_prefill_inputs([], num_running_requests=0)

        # Verify prefill flag is not set
        self.assertFalse(self.runner.exist_prefill_flag)

    def test_insert_prefill_inputs_multiple_requests(self):
        """Test insert_prefill_inputs with multiple requests."""
        self.runner.share_inputs = self._setup_share_inputs_mock()

        req1 = self._create_mock_request(
            task_type=self.runner.RequestType.PREFILL.value,
            idx=0,
            prompt_token_ids=[1, 2, 3]
        )
        req2 = self._create_mock_request(
            task_type=self.runner.RequestType.PREFILL.value,
            idx=1,
            prompt_token_ids=[4, 5, 6]
        )

        # Insert multiple requests
        self.runner.insert_prefill_inputs([req1, req2], num_running_requests=2)

        # Verify both requests are processed
        self.assertTrue(self.runner.exist_prefill_flag)

    def test_insert_prefill_inputs_with_stop_seqs(self):
        """Test insert_prefill_inputs with stop sequences."""
        self.runner.share_inputs = self._setup_share_inputs_mock()

        sampling_params = Mock()
        sampling_params.stop_seqs = [[1, 2], [3, 4]]
        sampling_params.stop_seqs_len = [2, 2]
        sampling_params.prompt_logprobs = None
        sampling_params.min_tokens = 1
        sampling_params.max_tokens = 100
        sampling_params.temperature = 1.0
        sampling_params.top_p = 1.0
        sampling_params.top_k = 0
        sampling_params.bad_tokens = []
        sampling_params.bad_tokens_len = 0

        req = self._create_mock_request(
            task_type=self.runner.RequestType.PREFILL.value,
            idx=0,
            sampling_params=sampling_params
        )

        # Insert request with stop sequences
        self.runner.insert_prefill_inputs([req], num_running_requests=1)

        # Verify request is processed
        self.assertTrue(self.runner.exist_prefill_flag)


class TestCollectDistributedStatus(unittest.TestCase):
    """Test cases for collect_distributed_status method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.enable_mm = False
        self.mock_fd_config.model_config = self.mock_model_config
        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.splitwise_role = "decoder"
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.use_ep = False
        self.mock_parallel_config.enable_chunked_moe = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.scheduler_config = self.mock_scheduler_config
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
        self.mock_parallel_config.use_ep = True
        self.mock_scheduler_config.splitwise_role = "mixed"

        self.runner.exist_prefill = Mock(return_value=True)

        result = self.runner.collect_distributed_status()

        self.assertIsNotNone(result)


class TestLoadModel(unittest.TestCase):
    """Test cases for load_model method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_fd_config.model_config = Mock()
        self.mock_fd_config.model_config.model_name = "test_model"
        self.mock_fd_config.model_config.enable_mm = False

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model = None
        self.runner.model_loader = None

    def test_load_model_sets_model(self):
        """Test load_model sets the model attribute."""
        # Mock the model loader
        self.runner.model_loader = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        self.runner.model_loader.return_value = mock_model

        self.runner.load_model()

        # Verify model is loaded
        self.assertIsNotNone(self.runner.model)
        self.assertEqual(self.runner.model, mock_model)
        self.runner.model_loader.assert_called_once()

    def test_load_model_with_existing_model(self):
        """Test load_model when model already exists."""
        self.runner.model_loader = Mock()
        mock_model = Mock()
        mock_model.eval = Mock()
        self.runner.model_loader.return_value = mock_model

        # Set an existing model
        existing_model = Mock()
        self.runner.model = existing_model

        # Load model again - should replace existing
        self.runner.load_model()

        # Verify model is replaced
        self.assertEqual(self.runner.model, mock_model)
        self.assertNotEqual(self.runner.model, existing_model)


class TestUpdateShareInputBlockNum(unittest.TestCase):
    """Test cases for update_share_input_block_num method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_cache_config = Mock()
        self.mock_cache_config.block_size = 16
        self.mock_cache_config.kv_cache_ratio = 1.0
        self.mock_fd_config.cache_config = self.mock_cache_config
        self.mock_parallel_config = Mock()
        self.mock_parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.num_gpu_blocks = 100
        self.runner.share_inputs = Mock()
        self.runner.share_inputs.update = Mock()
        self.runner.initialize_kv_cache = Mock()
        self.runner.speculative_method = None

    def test_update_share_input_block_num_sets_blocks(self):
        """Test update_share_input_block_num sets num_gpu_blocks."""
        self.runner.update_share_input_block_num(200)

        self.assertEqual(self.runner.num_gpu_blocks, 200)
        self.runner.initialize_kv_cache.assert_called_once()
        self.runner.share_inputs.update.assert_called()

    def test_update_share_input_block_num_with_mtp(self):
        """Test update_share_input_block_num with MTP method."""
        self.runner.speculative_method = "mtp"
        self.runner.proposer = Mock()
        self.runner.proposer.update_mtp_block_num = Mock()

        self.runner.update_share_input_block_num(150)

        self.assertEqual(self.runner.num_gpu_blocks, 150)
        self.runner.proposer.update_mtp_block_num.assert_called_once_with(150)


class TestPaddingCudagraphInputs(unittest.TestCase):
    """Test cases for padding_cudagraph_inputs method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_fd_config.model_config = Mock()
        self.mock_fd_config.model_config.enable_mm = False

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
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


class TestVisionEncoderCompile(unittest.TestCase):
    """Test cases for vision_encoder_compile method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.enable_mm = True
        self.mock_fd_config.model_config = self.mock_model_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.model = Mock()
        self.runner.model.layers = []

    def test_vision_encoder_compile_with_layers(self):
        """Test vision_encoder_compile with model layers."""
        mock_layer = Mock()
        mock_layer.apply_compile = Mock(return_value=mock_layer)
        self.runner.model.layers = [mock_layer]

        self.runner.vision_encoder_compile()

        # Verify compile is applied to layer
        mock_layer.apply_compile.assert_called_once()

    def test_vision_encoder_compile_multiple_layers(self):
        """Test vision_encoder_compile with multiple layers."""
        mock_layer1 = Mock()
        mock_layer1.apply_compile = Mock(return_value=mock_layer1)
        mock_layer2 = Mock()
        mock_layer2.apply_compile = Mock(return_value=mock_layer2)
        self.runner.model.layers = [mock_layer1, mock_layer2]

        self.runner.vision_encoder_compile()

        # Verify compile is applied to all layers
        mock_layer1.apply_compile.assert_called_once()
        mock_layer2.apply_compile.assert_called_once()

    def test_vision_encoder_compile_no_layers(self):
        """Test vision_encoder_compile with no layers."""
        self.runner.model.layers = []

        # Should not raise error
        self.runner.vision_encoder_compile()

        # No layers to compile, no calls made
        self.assertEqual(self.runner.model.layers, [])


class TestSotWarmup(unittest.TestCase):
    """Test cases for sot_warmup method."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_fd_config.model_config = Mock()
        self.mock_fd_config.model_config.enable_mm = False

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model = Mock()
        self.runner.share_inputs = Mock()
        self.runner.share_inputs.update = Mock()

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
        self.mock_fd_config = Mock()
        self.mock_fd_config.model_config = Mock()
        self.mock_fd_config.model_config.enable_mm = False

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
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
