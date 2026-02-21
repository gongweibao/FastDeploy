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
Tests for End-to-End Integration (Phase 1-7)

This module contains end-to-end integration tests for GPU Model Runner,
covering the complete data flow from initialization through output saving,
as documented in docs/gpu_model_runner_data_flow.md

Key components tested:
- Complete prefill -> decode -> stop flow
- Multiple requests concurrent processing
- Request state transitions
- Mixed task types batch processing
- Sampling parameters flow
- Prompt logprobs integration
- Routing replay integration
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from fastdeploy.engine.request import RequestType
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class TestGPURunnerE2E(unittest.TestCase):
    """End-to-end integration tests for GPUModelRunner."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.max_model_len = 4096
        self.mock_model_config.eos_tokens_lens = 1
        self.mock_model_config.max_stop_seqs_num = 4
        self.mock_model_config.enable_mm = False
        self.mock_model_config.runner_type = "causal_lm"
        self.mock_model_config.ori_vocab_size = 32000
        self.mock_fd_config.model_config = self.mock_model_config

        self.mock_scheduler_config = Mock()
        self.mock_scheduler_config.splitwise_role = "mixed"
        self.mock_scheduler_config.max_num_seqs = 10
        self.mock_scheduler_config.enable_overlap_schedule = False
        self.mock_fd_config.scheduler_config = self.mock_scheduler_config

        self.mock_parallel_config = Mock()
        self.mock_parallel_config.use_ep = False
        self.mock_fd_config.parallel_config = self.mock_parallel_config

        self.mock_routing_replay_config = Mock()
        self.mock_routing_replay_config.enable_routing_replay = False
        self.mock_fd_config.routing_replay_config = self.mock_routing_replay_config

        self.mock_cache_config = Mock()
        self.mock_cache_config.enable_prefix_caching = False
        self.mock_cache_config.enable_chunked_prefill = False
        self.mock_fd_config.cache_config = self.mock_cache_config

        self.mock_speculative_config = Mock()
        self.mock_speculative_config.method = None
        self.mock_speculative_config.num_speculative_tokens = 0
        self.mock_fd_config.speculative_config = self.mock_speculative_config

        self.mock_early_stop_config = Mock()
        self.mock_early_stop_config.enable_early_stop = False
        self.mock_fd_config.early_stop_config = self.mock_early_stop_config

        self.mock_graph_opt_config = Mock()
        self.mock_graph_opt_config.use_cudagraph = False
        self.mock_graph_opt_config.cudagraph_capture_sizes = []
        self.mock_graph_opt_config.cudagraph_capture_sizes_prefill = []
        self.mock_graph_opt_config.sot_warmup_sizes = []
        self.mock_graph_opt_config.cudagraph_only_prefill = False
        self.mock_fd_config.graph_opt_config = self.mock_graph_opt_config

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.scheduler_config = self.mock_scheduler_config
        self.runner.cache_config = self.mock_cache_config
        self.runner.speculative_config = self.mock_speculative_config
        self.runner.speculative_decoding = False
        self.runner.speculative_method = None
        self.runner.routing_replay_config = self.mock_routing_replay_config
        self.runner.routing_replay_manager = Mock()
        self.runner.quant_config = None
        self.runner.enable_overlap_schedule = False
        self.runner.enable_mm = False
        self.runner.device_id = 0
        self.runner.is_pooling_model = False
        self.runner.use_cudagraph = False
        self.runner.cudagraph_only_prefill = False

        # Initialize KV cache mock
        self.runner.initialize_kv_cache = Mock()
        self.runner.initialize_kv_cache.return_value = None

        # Add caches to share_inputs mock to skip initialization
        self.runner.share_inputs = Mock()
        self.runner.share_inputs.__contains__ = Mock(side_effect=lambda key: key == "caches")
        self.runner.share_inputs.__getitem__ = Mock(side_effect=lambda key: {
            "stop_flags": np.zeros((10,), dtype=bool),
        }[key])
        self.runner.share_inputs.update = Mock()

        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.exist_prefill_flag = False
        self.runner.pooling_params = []
        self.runner.sampler = Mock()
        self.runner.sampler.apply_logits_processor = Mock()
        self.runner.guided_backend = None
        self.runner.proposer = None

    def tearDown(self):
        """Clean up after tests."""
        pass

    def _create_mock_request(self, task_type=RequestType.PREFILL, idx=0, token_ids=None, output_ids=None):
        """Helper to create mock request."""
        request = Mock()
        request.task_type = Mock()
        request.task_type.value = task_type.value if hasattr(task_type, "value") else task_type
        request.idx = idx
        request.request_id = f"req_{idx}"
        request.prompt_token_ids = token_ids if token_ids else [1, 2, 3, 4, 5]
        request.output_token_ids = output_ids if output_ids else []
        request.block_tables = [0, 1, 2]
        request.eos_token_ids = [0]
        request.prefill_start_index = 0
        request.prefill_end_index = len(request.prompt_token_ids)
        request.sampling_params = Mock()
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []
        request.sampling_params.min_tokens = 1
        request.sampling_params.max_tokens = 100
        request.sampling_params.temperature = 1.0
        request.sampling_params.top_p = 1.0
        request.sampling_params.top_k = 0

        # Explicitly set guided decoding attributes to None to avoid Mock truthy behavior
        request.guided_json = None
        request.guided_regex = None
        request.guided_grammar = None
        request.structural_tag = None
        request.enable_thinking = None
        request.reasoning_max_tokens = None
        request.pooling_params = None

        # Store attributes for .get() method to retrieve
        _request_data = {
            "enable_thinking": request.enable_thinking,
            "reasoning_max_tokens": request.reasoning_max_tokens,
        }

        def mock_get(key, default=None):
            return _request_data.get(key, default)

        request.get = mock_get

        return request

    def _setup_share_inputs_mock(self, num_running=2):
        """Setup share_inputs mock with required attributes."""
        share_inputs = Mock()

        # Dictionary of string keys to numpy arrays
        data = {
            "req_ids": [""] * 10,
            "preempted_idx": np.zeros((10, 1), dtype="int32"),
            "stop_flags": np.zeros((10,), dtype=bool),
            "seq_lens_decoder": np.zeros((10,), dtype="int32"),
            "seq_lens_encoder": np.zeros((10,), dtype="int32"),
            "seq_lens_this_time_buffer": np.zeros((10,), dtype="int32"),
            "seq_lens_this_time": np.zeros((10,), dtype="int32"),
            "prompt_ids": np.zeros((10, 512), dtype="int64"),
            "input_ids": np.zeros((10, 512), dtype="int64"),
            "encoder_block_lens": np.zeros((10,), dtype="int32"),
            "block_tables": np.full((10, 128), -1, dtype="int32"),
            "step_seq_lens_decoder": np.zeros((10,), dtype="int32"),
            "prompt_lens": np.zeros((10,), dtype="int32"),
            "is_block_step": np.zeros((10,), dtype=bool),
            "is_chunk_step": np.zeros((10,), dtype=bool),
            "step_idx": np.zeros((10,), dtype="int32"),
            "pre_ids": np.full((10, 1), -1, dtype="int64"),
            "eos_token_id": np.zeros((1, 1), dtype="int64"),
            "top_p": np.zeros((10,), dtype="float32"),
            "top_k": np.zeros((10,), dtype="int32"),
            "top_k_list": np.zeros((10,), dtype="int32"),
            "min_p": np.zeros((10,), dtype="float32"),
            "min_p_list": np.zeros((10,), dtype="float32"),
            "temperature": np.zeros((10,), dtype="float32"),
            "penalty_score": np.ones((10,), dtype="float32"),
            "frequency_score": np.zeros((10,), dtype="float32"),
            "presence_score": np.zeros((10,), dtype="float32"),
            "temp_scaled_logprobs": np.zeros((10,), dtype=bool),
            "top_p_normalized_logprobs": np.zeros((10,), dtype=bool),
            "min_dec_len": np.zeros((10,), dtype="int32"),
            "max_dec_len": np.zeros((10,), dtype="int32"),
            "first_token_ids": np.zeros((10, 1), dtype="int64"),
            "infer_seed": np.zeros((10,), dtype="int64"),
            "bad_tokens_len": np.ones((10,), dtype="int32"),
            "bad_tokens": np.full((10, 1), -1, dtype="int64"),
            "stop_seqs_len": np.zeros((10, 4), dtype="int32"),
            "stop_seqs": np.zeros((10, 4, 10), dtype="int64"),
            "not_need_stop": np.zeros((1,), dtype="int32"),
            "logits_processors_args": [{}] * 10,
            "enable_thinking": np.zeros((10, 1), dtype="int32"),
            "max_think_lens": np.full((10, 1), -1, dtype="int32"),
            "limit_think_status": np.zeros((10, 1), dtype="int32"),
            "num_running_requests": 0,
            "running_requests_ids": [],
        }

        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)
        share_inputs.__getitem__.side_effect = lambda key: data[key]
        share_inputs.__setitem__.side_effect = lambda key, value: data.__setitem__(key, value))
        share_inputs.__contains__.side_effect = lambda key: key in data

        return share_inputs

    def test_complete_prefill_decode_flow(self):
        """Test complete prefill -> decode -> stop flow."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # 1. Create prefill request
        prefill_req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # 2. Insert prefill task
        self.runner.insert_tasks_v1([prefill_req], num_running_requests=1)

        # Verify prefill flag is set
        self.assertTrue(self.runner.exist_prefill_flag)

        # 3. Mock execute model
        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (Mock(), Mock(), Mock(), 5)

                with patch.object(self.runner, "_save_model_output") as mock_save:
                    self.runner.execute_model_normal(model_forward_batch=[prefill_req], num_running_requests=1)

                    # Verify flow is executed
                    mock_preprocess_execute.assert_called_once()
                    mock_postprocess.assert_called_once()
                    mock_save.assert_called_once()

        # 4. Clear requests
        self.runner.clear_requests()

        # Verify request state is reset
        self.assertFalse(self.runner.exist_prefill_flag)
        self.assertEqual(self.runner.prompt_logprobs_reqs, {})

    def test_multiple_requests_concurrent(self):
        """Test multiple requests processed concurrently in batch."""
        self.runner.share_inputs = self._setup_share_inputs_mock(num_running=3)
        self.runner.is_pooling_model = False

        # Create multiple requests
        req1 = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req2 = self._create_mock_request(task_type=RequestType.PREFILL, idx=1)
        req3 = self._create_mock_request(task_type=RequestType.DECODE, idx=2)

        self.runner.share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)

        # Insert all requests
        self.runner.insert_tasks_v1([req1, req2, req3], num_running_requests=3)

        # Verify prefill requests are in forward_batch_reqs_list
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req1)
        self.assertEqual(self.runner.forward_batch_reqs_list[1], req2)
        # Note: DECODE requests are NOT added to forward_batch_reqs_list in insert_tasks_v1
        # They should already be there from the prefill phase

        # Verify prefill flag is set for prefill requests
        self.assertTrue(self.runner.exist_prefill_flag)

    def test_request_state_transitions(self):
        """Test request state transitions: prefill -> decode -> stop."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Initial state: prefill
        prefill_req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert prefill
        self.runner.insert_tasks_v1([prefill_req], num_running_requests=1)
        self.assertTrue(self.runner.exist_prefill_flag)

        # Simulate decode step
        decode_req = self._create_mock_request(
            task_type=RequestType.DECODE, idx=0, token_ids=[], output_ids=[10, 20]
        )
        decode_req.output_token_ids = [10, 20]

        # Mock execution
        with patch.object(self.runner, "_preprocess_and_execute_model") as mock_preprocess_execute:
            mock_preprocess_execute.return_value = (Mock(), [], Mock())

            with patch.object(self.runner, "_postprocess") as mock_postprocess:
                mock_postprocess.return_value = (Mock(), Mock(), Mock(), 5)

                with patch.object(self.runner, "_save_model_output"):
                    self.runner.execute_model_normal(model_forward_batch=[decode_req], num_running_requests=1)

        # Clear requests
        self.runner.clear_requests()
        self.assertFalse(self.runner.exist_prefill_flag)

    def test_batch_with_mixed_task_types(self):
        """Test batch with mixed prefill and decode tasks."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Mixed requests
        prefill_req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        decode_req = self._create_mock_request(task_type=RequestType.DECODE, idx=1)

        self.runner.share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)

        # Insert mixed tasks
        self.runner.insert_tasks_v1([prefill_req, decode_req], num_running_requests=2)

        # Verify prefill request is in forward_batch_reqs_list
        self.assertEqual(self.runner.forward_batch_reqs_list[0], prefill_req)

        # Verify prefill flag is set
        self.assertTrue(self.runner.exist_prefill_flag)

    def test_sampling_params_flow(self):
        """Test sampling parameters flow through the system."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with custom sampling params
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req.sampling_params.temperature = 0.8
        req.sampling_params.top_p = 0.95
        req.sampling_params.top_k = 50
        req.sampling_params.min_p = 0.05
        req.sampling_params.repetition_penalty = 1.1
        req.sampling_params.frequency_penalty = 0.1
        req.sampling_params.presence_penalty = 0.2

        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is in the batch
        self.assertEqual(self.runner.forward_batch_reqs_list[0], req)

    def test_prompt_logprobs_integration(self):
        """Test prompt logprobs in the full flow."""
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False

        # Request with prompt logprobs
        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        req.sampling_params.prompt_logprobs = 5

        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        # Insert request
        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Verify request is tracked for prompt logprobs
        self.assertIn(req.request_id, self.runner.prompt_logprobs_reqs)

        # Clear requests
        self.runner.clear_requests()

        # Verify prompt logprobs tracking is cleared
        self.assertEqual(self.runner.prompt_logprobs_reqs, {})

    def test_routing_replay_integration(self):
        """Test routing replay in the flow."""
        self.mock_routing_replay_config.enable_routing_replay = True

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.scheduler_config = Mock()
        self.runner.scheduler_config.max_num_seqs = 10
        self.runner.fd_config.scheduler_config = self.mock_scheduler_config
        self.runner.routing_replay_config = self.mock_routing_replay_config
        self.runner.routing_replay_manager = Mock()
        self.runner.share_inputs = self._setup_share_inputs_mock()
        self.runner.is_pooling_model = False
        self.runner.forward_batch_reqs_list = [None] * 10
        self.runner.prompt_logprobs_reqs = {}
        self.runner.in_progress_prompt_logprobs = {}
        self.runner.exist_prefill_flag = False
        self.runner.pooling_params = []
        self.runner.sampler = Mock()
        self.runner.sampler.apply_logits_processor = Mock()

        req = self._create_mock_request(task_type=RequestType.PREFILL, idx=0)
        self.runner.share_inputs.get_index_by_batch_id = Mock(return_value=0)

        self.runner.insert_tasks_v1([req], num_running_requests=1)

        # Clear requests with routing replay enabled
        self.runner.clear_requests()

        # Verify routing replay manager is called
        self.runner.routing_replay_manager.put_table_to_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
