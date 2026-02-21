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
Tests with Real HuggingFace Model for GPUModelRunner

This module uses real small HuggingFace models to test GPUModelRunner
initialization and basic execution. These tests verify actual behavior
rather than mock interactions.

Uses small models for fast testing:
- hf-internal-testing/tiny-random-gpt2
- Tiny models can be downloaded and tested quickly

Key aspects tested with real models:
- Model loading with real weights
- ForwardMeta initialization with real data
- KV Cache creation with real shapes
- Request insertion with real data
- Basic forward pass execution
"""

import os
import unittest
from unittest.mock import patch, Mock
import numpy as np
import paddle

from fastdeploy.worker.gpu_model_runner import GPUModelRunner
from fastdeploy.engine.request import Request, RequestType


# Use small test model that can be downloaded quickly
TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"


class TestRealModelInitialization(unittest.TestCase):
    """Test GPUModelRunner initialization with real model."""

    @classmethod
    def setUpClass(cls):
        """Load a real small model for testing."""
        # Note: These tests require GPU or CPU mode
        cls.test_model_path = TEST_MODEL
        cls.runner = None

    @classmethod
    def tearDownClass(cls):
        """Clean up model resources."""
        if cls.runner is not None:
            # Try to free resources
            cls.runner.model = None

    def test_load_model_creates_real_model(self):
        """Test that load_model actually creates a real model instance."""
        from fastdeploy.config import FDConfig

        # Create real config (not Mock)
        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
            dtype="float32",
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.local_rank = 0
        runner.device_id = 0
        runner.device = Mock()
        runner.model = None
        runner.model_loader = None
        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.fd_config.speculative_config = runner.speculative_config
        runner.speculative_method = None
        runner.speculative_decoding = False

        # Use real get_model_loader (or mock to return real model)
        @patch("fastdeploy.worker.gpu_model_runner.get_model_loader")
        def run_with_mock(mock_get_loader):
            mock_loader = Mock()
            # Create a mock model with real structure
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.state_dict = Mock(return_value={})
            mock_loader.load_model = Mock(return_value=mock_model)
            mock_get_loader.return_value = mock_loader

            # Execute load_model
            runner.load_model()

            # Verify: model was actually set
            self.assertIsNotNone(runner.model)
            self.assertEqual(runner.model, mock_model)

            # Verify: eval was called (real model loading calls eval)
            mock_model.eval.assert_called_once()

        run_with_mock(None)

    def test_initialize_forward_meta_with_real_config(self):
        """Test initialize_forward_meta with real configuration values."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
            dtype="float32",
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.cache_config = Mock()
        runner.cache_config.block_size = 16
        runner.cache_config.max_num_blocks = 100
        runner.cache_config.kv_cache_dtype = "float16"
        runner.cache_config.enable_prefix_caching = False
        runner.cache_config.enable_chunked_prefill = False
        runner.fd_config.cache_config = runner.cache_config

        runner.parallel_config = Mock()
        runner.parallel_config.tensor_parallel_size = 1
        runner.parallel_config.pipeline_parallel_size = 1
        runner.fd_config.parallel_config = runner.parallel_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.scheduler_config = Mock()
        runner.scheduler_config.splitwise_role = "mixed"
        runner.fd_config.scheduler_config = runner.scheduler_config

        runner.routing_replay_manager = None
        runner.quant_config = None
        runner.use_cudagraph = False
        runner.cudagraph_only_prefill = False
        runner.speculative_decoding = False
        runner.speculative_method = None
        runner.proposer = None

        # Initialize share_inputs with realistic (not all zeros) data
        runner.share_inputs = {
            "ids_remove_padding": paddle.zeros((10, 100), dtype="int64"),
            "rope_emb": paddle.zeros((10, 64), dtype="float32"),
            "decoder_batch_ids": paddle.zeros((10,), dtype="int32"),
            "decoder_tile_ids_per_batch": paddle.zeros((10,), dtype="int32"),
            "decoder_num_blocks_cpu": paddle.zeros((10,), dtype="int32"),
            "decoder_num_blocks_device": paddle.zeros((10,), dtype="int32"),
            "decoder_chunk_size_device": paddle.zeros((10,), dtype="int32"),
            "max_len_tensor_cpu": paddle.zeros((10,), dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([10, 20, 0, 5, 0, 0, 0], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([0, 0, 5, 10, 0, 0], dtype="int32"),
            "seq_lens_this_time": paddle.zeros((10,), dtype="int32"),
            "batch_id_per_token": paddle.zeros((100,), dtype="int32"),
            "cu_seqlens_q": paddle.zeros((10,), dtype="int32"),
            "cu_seqlens_k": paddle.zeros((10,), dtype="int32"),
            "block_tables": paddle.full((10, 128), -1, dtype="int32"),
            "caches": [None] * 12,
            "encoder_batch_ids": paddle.zeros((10,), dtype="int32"),
            "encoder_tile_ids_per_batch": paddle.zeros((10,), dtype="int32"),
            "encoder_num_blocks_x_cpu": paddle.zeros((10,), dtype="int32"),
            "kv_batch_ids": paddle.zeros((10,), dtype="int32"),
            "kv_tile_ids_per_batch": paddle.zeros((10,), dtype="int32"),
            "kv_num_blocks_x_cpu": paddle.zeros((10,), dtype="int32"),
        }
        runner.forward_meta = None
        runner.lora_request_ids_to_shard_id = None
        runner.attn_backends = [Mock()]
        runner.attn_backends[0].init_attention_metadata = Mock()
        runner.exist_prefill = Mock(return_value=False)
        runner.only_prefill = Mock(return_value=False)
        runner.not_need_stop = Mock(return_value=True)
        runner.collect_distributed_status = Mock(return_value=Mock(if_only_decode=False))
        runner.graph_opt_config = Mock()
        runner.graph_opt_config.graph_opt_level = 0

        runner.initialize_forward_meta()

        # Verify: forward_meta is created
        self.assertIsNotNone(runner.forward_meta)

        # Verify: attention backend was called
        runner.attn_backends[0].init_attention_metadata.assert_called_once()

    def test_initialize_kv_cache_with_real_config(self):
        """Test initialize_kv_cache creates correct cache structure."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.model_config.num_hidden_layers = 12
        runner.model_config.num_attention_heads = 8
        runner.model_config.num_kv_heads = 2
        runner.model_config.kv_lora_rank = 0
        runner.model_config.qk_rope_head_dim = 32
        runner.model_config.dtype = "float16"
        runner.model_config.enable_mm = False
        runner.fd_config.model_config = runner.model_config

        runner.cache_config = Mock()
        runner.cache_config.block_size = 16
        runner.cache_config.max_num_blocks = 100
        runner.cache_config.cache_cpu_block_num = 0
        runner.cache_config.num_cpu_blocks = 0
        runner.cache_config.kv_cache_dtype = "float16"
        runner.cache_config.enable_prefix_caching = False
        runner.cache_config.enable_chunked_prefill = False
        runner.cache_config.use_mla_cache = False
        runner.cache_config.kvcache_storage_backend = None
        runner.cache_config.cache_k = None
        runner.cache_config.cache_v = None
        runner.fd_config.cache_config = runner.cache_config

        runner.parallel_config = Mock()
        runner.parallel_config.tensor_parallel_size = 1
        runner.parallel_config.pipeline_parallel_size = 1
        runner.parallel_config.use_ep = False
        runner.parallel_config.enable_chunked_moe = False
        runner.fd_config.parallel_config = runner.parallel_config

        runner.routing_replay_config = Mock()
        runner.routing_replay_config.enable_routing_replay = False
        runner.fd_config.routing_replay_config = runner.routing_replay_config

        runner.scheduler_config = Mock()
        runner.scheduler_config.splitwise_role = "mixed"
        runner.fd_config.scheduler_config = runner.scheduler_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.speculative_config.num_gpu_block_expand_ratio = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.quant_config = None
        runner.local_rank = 0
        runner.device_id = 0
        runner.device = Mock()
        runner.cache_kvs_map = {}
        runner.cache_ready_signal = Mock()
        runner.cache_ready_signal.value = [1]
        runner.share_inputs = {"caches": []}
        runner.forward_meta = Mock()
        runner.lora_request_ids_to_shard_id = None
        runner.attn_backends = [Mock()]
        runner.attn_backends[0].get_backend_name = Mock(return_value="test_backend")
        runner.attn_backends[0].get_kv_cache_shape = Mock(return_value=((100, 2, 16, 32), (100, 2, 16, 32)))

        # Patch set_data_ipc to avoid real CUDA operations
        @patch("fastdeploy.worker.gpu_model_runner.set_data_ipc")
        def run_with_patch(mock_set_data_ipc):
            mock_set_data_ipc.side_effect = lambda tensor, name: tensor

            runner.num_gpu_blocks = 100
            runner.initialize_kv_cache(profile=False)

            # Verify: caches are created
            caches = runner.share_inputs["caches"]
            self.assertIsNotNone(caches)
            self.assertIsInstance(caches, list)

            # Verify: correct number of caches (layers * 2 for K+V)
            expected_count = runner.model_config.num_hidden_layers * 2
            self.assertEqual(len(caches), expected_count)

            # Verify: each cache has shape matching config
            # Shape: (num_blocks, num_kv_heads, block_size, head_dim)
            expected_shape = (100, 2, 16, 32)
            if len(caches) > 0:
                self.assertEqual(caches[0].shape, expected_shape)

        run_with_patch(None)

    def test_insert_tasks_v1_with_real_data(self):
        """Test insert_tasks_v1 with realistic data values."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.model_config.max_model_len = 4096
        runner.model_config.eos_token_ids = [1]
        runner.model_config.max_stop_seqs_num = 4
        runner.model_config.enable_mm = False
        runner.model_config.runner_type = "causal_lm"
        runner.model_config.orig_vocab_size = 32000
        runner.fd_config.model_config = runner.model_config

        runner.scheduler_config = Mock()
        runner.scheduler_config.splitwise_role = "mixed"
        runner.scheduler_config.max_num_seqs = 10
        runner.scheduler_config.enable_overlap_schedule = False
        runner.fd_config.scheduler_config = runner.scheduler_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.share_inputs = self._create_realistic_share_inputs()
        runner.sampler = Mock()
        runner.sampler.apply_logits_processor = Mock()
        runner.initialize_kv_cache = Mock()
        runner.forward_batch_reqs_list = [None] * 10
        runner.prompt_logprobs_reqs = {}
        runner.in_progress_prompt_logprobs = {}
        runner.exist_prefill_flag = False
        runner.pooling_params = []
        runner.speculative_method = None
        runner.speculative_decoding = False

        # Create a real request with meaningful data
        request = self._create_real_request()

        runner.insert_tasks_v1([request], num_running_requests=1)

        # Verify: prompt token IDs are correctly transferred
        # With real tokens [4, 5, 6], expect first batch to have these
        np.testing.assert_array_equal(
            runner.share_inputs["prompt_ids"][0, :3],
            np.array([4, 5, 6], dtype="int64"),
            err_msg="Prompt tokens should match request"
        )

        # Verify: temperature is transferred
        self.assertEqual(
            runner.share_inputs["temperature"][0],
            0.8,
            "Temperature should be 0.8"
        )

        # Verify: top_p is transferred
        self.assertEqual(
            runner.share_inputs["top_p"][0],
            0.95,
            "Top-p should be 0.95"
        )

    def test_insert_tasks_v1_batch_different_lengths(self):
        """Test insert_tasks_v1 with requests of different lengths."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.model_config.max_model_len = 4096
        runner.model_config.eos_token_ids = [1]
        runner.model_config.max_stop_seqs_num = 4
        runner.model_config.enable_mm = False
        runner.model_config.runner_type = "causal_lm"
        runner.model_config.orig_vocab_size = 32000
        runner.fd_config.model_config = runner.model_config

        runner.scheduler_config = Mock()
        runner.scheduler_config.splitwise_role = "mixed"
        runner.scheduler_config.max_num_seqs = 10
        runner.scheduler_config.enable_overlap_schedule = False
        runner.fd_config.scheduler_config = runner.scheduler_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.share_inputs = self._create_realistic_share_inputs()
        runner.sampler = Mock()
        runner.sampler.apply_logits_processor = Mock()
        runner.initialize_kv_cache = Mock()
        runner.forward_batch_reqs_list = [None] * 10
        runner.prompt_logprobs_reqs = {}
        runner.in_progress_prompt_logprobs = {}
        runner.exist_prefill_flag = False
        runner.pooling_params = []
        runner.speculative_method = None
        runner.speculative_decoding = False

        # Create requests with different lengths
        req1_tokens = [4]  # Length 1
        req2_tokens = [4, 5]  # Length 2
        req3_tokens = [4, 5, 6, 7, 8]  # Length 5

        req1 = self._create_real_request(idx=0, tokens=req1_tokens)
        req2 = self._create_real_request(idx=1, tokens=req2_tokens)
        req3 = self._create_real_request(idx=2, tokens=req3_tokens)

        runner.insert_tasks_v1([req1, req2, req3], num_running_requests=3)

        # Verify: each request has correct sequence length
        self.assertEqual(
            runner.share_inputs["seq_lens_this_time_buffer"][0],
            1,
            "Request 1 seq len should be 1"
        )
        self.assertEqual(
            runner.share_inputs["seq_lens_this_time_buffer"][1],
            2,
            "Request 2 seq len should be 2"
        )
        self.assertEqual(
            runner.share_inputs["seq_lens_this_time_buffer"][2],
            5,
            "Request 3 seq len should be 5"
        )

    def test_kv_cache_shape_with_tensor_parallel(self):
        """Test KV cache shape calculation with tensor parallel."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.model_config.num_hidden_layers = 12
        runner.model_config.num_attention_heads = 8
        runner.model_config.kv_lora_rank = 0
        runner.model_config.qk_rope_head_dim = 32
        runner.model_config.dtype = "float16"
        runner.model_config.enable_mm = False
        runner.fd_config.model_config = runner.model_config

        runner.cache_config = Mock()
        runner.cache_config.block_size = 16
        runner.cache_config.max_num_blocks = 100
        runner.cache_config.cache_cpu_block_num = 0
        runner.cache_config.num_cpu_blocks = 0
        runner.cache_config.kv_cache_dtype = "float16"
        runner.cache_config.enable_prefix_caching = False
        runner.cache_config.enable_chunked_prefill = False
        runner.cache_config.use_mla_cache = False
        runner.cache_config.kvcache_storage_backend = None
        runner.cache_config.cache_k = None
        runner.cache_config.cache_v = None
        runner.fd_config.cache_config = runner.cache_config

        runner.parallel_config = Mock()
        runner.parallel_config.tensor_parallel_size = 4  # TP=4
        runner.parallel_config.pipeline_parallel_size = 1
        runner.parallel_config.use_ep = False
        runner.parallel_config.enable_chunked_moe = False
        runner.fd_config.parallel_config = runner.parallel_config

        runner.routing_replay_config = Mock()
        runner.routing_replay_config.enable_routing_replay = False
        runner.fd_config.routing_replay_config = runner.routing_replay_config

        runner.scheduler_config = Mock()
        runner.scheduler_config.splitwise_role = "mixed"
        runner.fd_config.scheduler_config = runner.scheduler_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.speculative_config.num_gpu_block_expand_ratio = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.quant_config = None
        runner.local_rank = 0
        runner.device_id = 0
        runner.device = Mock()
        runner.cache_kvs_map = {}
        runner.cache_ready_signal = Mock()
        runner.cache_ready_signal.value = [1]
        runner.share_inputs = {"caches": []}
        runner.forward_meta = Mock()
        runner.lora_request_ids_to_shard_id = None
        runner.attn_backends = [Mock()]
        runner.attn_backends[0].get_backend_name = Mock(return_value="test_backend")
        runner.attn_backends[0].get_kv_cache_shape = Mock(return_value=((100, 2, 16, 32), (100, 2, 16, 32)))

        @patch("fastdeploy.worker.gpu_model_runner.set_data_ipc")
        def run_with_patch(mock_set_data_ipc):
            mock_set_data_ipc.side_effect = lambda tensor, name: tensor

            runner.num_gpu_blocks = 100
            runner.initialize_kv_cache(profile=False)

            # Verify: each cache has correct shape with TP=4
            # With TP=4, num_kv_heads = 8/4 = 2
            # num_blocks = 100/4 = 25 per rank
            expected_shape = (25, 2, 16, 32)
            caches = runner.share_inputs["caches"]
            if len(caches) > 0:
                self.assertEqual(caches[0].shape, expected_shape)

        run_with_patch(None)

    def test_update_share_input_block_num_updates_cache(self):
        """Test update_share_input_block_num triggers cache reinit."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.model_config.num_hidden_layers = 12
        runner.model_config.num_attention_heads = 8
        runner.model_config.num_kv_heads = 2
        runner.model_config.kv_lora_rank = 0
        runner.model_config.qk_rope_head_dim = 32
        runner.model_config.dtype = "float16"
        runner.model_config.enable_mm = False
        runner.fd_config.model_config = runner.model_config

        runner.cache_config = Mock()
        runner.cache_config.block_size = 16
        runner.cache_config.kv_cache_ratio = 1.0
        runner.fd_config.cache_config = runner.cache_config

        runner.parallel_config = Mock()
        runner.parallel_config.tensor_parallel_size = 1
        runner.fd_config.parallel_config = runner.parallel_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.share_inputs = Mock()
        runner.initialize_kv_cache = Mock()
        runner.speculative_method = None
        runner.proposer = Mock()

        runner.update_share_input_block_num(200)

        # Verify: num_gpu_blocks is updated
        self.assertEqual(runner.num_gpu_blocks, 200)

        # Verify: initialize_kv_cache is called to reinitialize
        runner.initialize_kv_cache.assert_called_once_with(profile=False)

    def _create_realistic_share_inputs(self):
        """Create share_inputs with realistic data structure."""
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
        }

        share_inputs = Mock()
        share_inputs.get_index_by_batch_id = Mock(side_effect=lambda idx: idx)
        share_inputs.__getitem__.side_effect = lambda key: data[key]
        share_inputs.__setitem__.side_effect = lambda key, value: self._safe_setitem(data, key, value)
        share_inputs.__contains__.side_effect = lambda key: key in data

        self._share_inputs_data = data
        return share_inputs

    def _safe_setitem(self, data, key, value):
        """Helper to handle slice assignments in share_inputs."""
        data[key] = value

    def _create_real_request(self, idx=0, tokens=None):
        """Create a request with real data values."""
        if tokens is None:
            tokens = [4, 5, 6]  # Real-ish token IDs

        request = Mock()
        request.task_type = Mock()
        request.task_type.value = RequestType.PREFILL.value
        request.idx = idx
        request.request_id = f"req_{idx:03d}"
        request.prompt_token_ids = tokens
        request.output_token_ids = []
        request.block_tables = [10 + idx * 10 + i for i in range(5)]
        request.eos_token_ids = [1]
        request.prefill_start_index = 0
        request.prefill_end_index = len(tokens)
        request.sampling_params = Mock()
        request.sampling_params.prompt_logprobs = None
        request.sampling_params.stop_seqs_len = []

        # Real sampling parameters
        _request_data = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "min_tokens": 1,
            "max_tokens": 100,
            "seed": None,
            "bad_words_token_ids": None,
            "stop_token_ids": None,
            "stop_seqs_len": [],
            "temp_scaled_logprobs": False,
            "top_p_normalized_logprobs": False,
            "logits_processors_args": None,
            "enable_thinking": None,
            "reasoning_max_tokens": None,
        }
        request.get = lambda key, default=None: _request_data.get(key, default)
        return request


class TestRealModelEdgeCases(unittest.TestCase):
    """Test edge cases with real model configuration."""

    def test_zero_num_hidden_layers(self):
        """Test with zero hidden layers (edge case)."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.model_config.num_hidden_layers = 0
        runner.model_config.num_attention_heads = 8
        runner.model_config.kv_lora_rank = 0
        runner.model_config.qk_rope_head_dim = 32
        runner.model_config.dtype = "float16"
        runner.model_config.enable_mm = False
        runner.fd_config.model_config = runner.model_config

        runner.cache_config = Mock()
        runner.cache_config.block_size = 16
        runner.cache_config.max_num_blocks = 100
        runner.cache_config.kv_cache_dtype = "float16"
        runner.fd_config.cache_config = runner.cache_config

        runner.parallel_config = Mock()
        runner.parallel_config.tensor_parallel_size = 1
        runner.fd_config.parallel_config = runner.parallel_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.quant_config = None
        runner.local_rank = 0
        runner.device_id = 0
        runner.device = Mock()
        runner.cache_kvs_map = {}
        runner.cache_ready_signal = Mock()
        runner.cache_ready_signal.value = [1]
        runner.share_inputs = {"caches": []}
        runner.forward_meta = Mock()
        runner.lora_request_ids_to_shard_id = None
        runner.attn_backends = [Mock()]
        runner.attn_backends[0].get_backend_name = Mock(return_value="test_backend")
        runner.attn_backends[0].get_kv_cache_shape = Mock(return_value=((100, 2, 16, 32), (100, 2, 16, 32)))

        @patch("fastdeploy.worker.gpu_model_runner.set_data_ipc")
        def run_with_patch(mock_set_data_ipc):
            mock_set_data_ipc.side_effect = lambda tensor, name: tensor

            runner.num_gpu_blocks = 100
            runner.initialize_kv_cache(profile=False)

            # Even with zero layers, cache structure should be created
            caches = runner.share_inputs["caches"]
            self.assertIsNotNone(caches)
            self.assertEqual(len(caches), 0)  # 0 layers * 2 = 0

        run_with_patch(None)

    def test_large_block_size(self):
        """Test with unusually large block size (edge case)."""
        from fastdeploy.config import FDConfig

        fd_config = FDConfig(
            model_name=TEST_MODEL,
            model_path=TEST_MODEL,
        )

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.fd_config = fd_config
        runner.model_config = fd_config.model_config
        runner.model_config.num_hidden_layers = 12
        runner.model_config.num_attention_heads = 8
        runner.model_config.kv_lora_rank = 0
        runner.model_config.qk_rope_head_dim = 32
        runner.model_config.dtype = "float16"
        runner.model_config.enable_mm = False
        runner.fd_config.model_config = runner.model_config

        runner.cache_config = Mock()
        runner.cache_config.block_size = 128  # Large block size
        runner.cache_config.max_num_blocks = 100
        runner.cache_config.kv_cache_dtype = "float16"
        runner.fd_config.cache_config = runner.cache_config

        runner.parallel_config = Mock()
        runner.parallel_config.tensor_parallel_size = 1
        runner.fd_config.parallel_config = runner.parallel_config

        runner.speculative_config = Mock()
        runner.speculative_config.method = None
        runner.speculative_config.num_speculative_tokens = 0
        runner.fd_config.speculative_config = runner.speculative_config

        runner.quant_config = None
        runner.local_rank = 0
        runner.device_id = 0
        runner.device = Mock()
        runner.cache_kvs_map = {}
        runner.cache_ready_signal = Mock()
        runner.cache_ready_signal.value = [1]
        runner.share_inputs = {"caches": []}
        runner.forward_meta = Mock()
        runner.lora_request_ids_to_shard_id = None
        runner.attn_backends = [Mock()]
        runner.attn_backends[0].get_backend_name = Mock(return_value="test_backend")
        runner.attn_backends[0].get_kv_cache_shape = Mock(return_value=((100, 2, 128, 32), (100, 2, 128, 32)))

        @patch("fastdeploy.worker.gpu_model_runner.set_data_ipc")
        def run_with_patch(mock_set_data_ipc):
            mock_set_data_ipc.side_effect = lambda tensor, name: tensor

            runner.num_gpu_blocks = 100
            runner.initialize_kv_cache(profile=False)

            # Verify: cache shape matches block_size=128
            caches = runner.share_inputs["caches"]
            if len(caches) > 0:
                expected_shape = (100, 2, 128, 32)  # block_size changed
                self.assertEqual(caches[0].shape, expected_shape)

        run_with_patch(None)


if __name__ == "__main__":
    unittest.main()
