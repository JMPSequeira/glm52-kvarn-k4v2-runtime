FROM localhost/glm52-shared-h-mixed-mcg:r634-k34-v3-native-k5-v1

# GLM-5.2 EXL3 TR3 3.40bpw + KVarN K4V2 runtime — overlays baked.
# Sources: ~/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src (see MANIFEST.md) and
# the local-inference-lab/vllm fork tree. Champion-active set only.

# --- active-r634-b12x-compact (b12x + KVarN serving path) ---
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-b12x-compact/b12x_mla_sparse.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-b12x-compact/mla_attention.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-b12x-compact/kvarn_mla.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/kvarn_mla.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-b12x-compact/config.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/kvarn/config.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-b12x-compact/kv_cache_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-b12x-compact/kvarn_api_k4.py /opt/venv/lib/python3.12/site-packages/sparkinfer/attention/kvarn_mla/api.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-b12x-compact/triton_kvarn_sinkhorn.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_kvarn_sinkhorn.py

# --- active-r634-kvarn-k4-native (K4V2 native reader) ---
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-kvarn-k4-native/kvarn_mla_state.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/kvarn_mla_state.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-kvarn-k4-native/kvarn_mla/io.py /opt/venv/lib/python3.12/site-packages/sparkinfer/attention/kvarn_mla/io.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-kvarn-k4-native/_shared/mla/kernel.py /opt/venv/lib/python3.12/site-packages/sparkinfer/attention/_shared/mla/kernel.py

# --- active-r634-runner + h16-v1 runner/spec/config ---
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/active-r634-runner/model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/worker/gpu_model_runner.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/worker/workspace.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/workspace.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/attention/backends/flash_attn.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/attention/backends/flex_attention.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/flex_attention.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/attention/ops/dcp_alltoall.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/dcp_alltoall.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/compilation/passes/fusion/allreduce_rms_fusion.py /opt/venv/lib/python3.12/site-packages/vllm/compilation/passes/fusion/allreduce_rms_fusion.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dspark/utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/models/deepseek_v4/nvidia/dspark.py /opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/dspark.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/model_executor/models/qwen3_dflash.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/transformers_utils/configs/speculators/algos.py /opt/venv/lib/python3.12/site-packages/vllm/transformers_utils/configs/speculators/algos.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/config/cache.py /opt/venv/lib/python3.12/site-packages/vllm/config/cache.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/config/vllm.py /opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/config/speculative.py /opt/venv/lib/python3.12/site-packages/vllm/config/speculative.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/utils/torch_utils.py /opt/venv/lib/python3.12/site-packages/vllm/utils/torch_utils.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v1/vllm/envs.py /opt/venv/lib/python3.12/site-packages/vllm/envs.py

# --- h16-v7 (EXL3 loader + MoE kernels) ---
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v7/vllm/model_executor/layers/quantization/exl3.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v7/sparkinfer/moe/_shared/kernels/w4a16/route_pack.py /opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/route_pack.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/build-exact-h16-v7/sparkinfer/moe/_shared/kernels/w4a16/mixed_trellis.py /opt/venv/lib/python3.12/site-packages/sparkinfer/moe/_shared/kernels/w4a16/mixed_trellis.py

# --- mtp/prefix overlays ---
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/r634-mtp-overlays/vllm/v1/attention/backends/mla/indexer.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/r634-prefix-overlays/vllm/platforms/cuda.py /opt/venv/lib/python3.12/site-packages/vllm/platforms/cuda.py

# --- sparkinfer (shared-h-stack) ---
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/shared-h-stack/sparkinfer-c40eb08f-full/sparkinfer/gemm/mxfp8_linear/_kernel.py /opt/venv/lib/python3.12/site-packages/sparkinfer/gemm/mxfp8_linear/_kernel.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/shared-h-stack/sparkinfer-c40eb08f-full/sparkinfer/gemm/_bmm/api.py /opt/venv/lib/python3.12/site-packages/sparkinfer/gemm/_bmm/api.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/shared-h-stack/sparkinfer-c40eb08f-full/sparkinfer/gemm/_shared/bf16_bmm.py /opt/venv/lib/python3.12/site-packages/sparkinfer/gemm/_shared/bf16_bmm.py

# --- runtime-vllm-extras ---
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/runtime-vllm-extras/vllm/v1/attention/backends/mla/sparse_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/sparse_utils.py
COPY GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src/runtime-vllm-extras/vllm/model_executor/layers/quantization/nvfp4_nf3_hybrid.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/nvfp4_nf3_hybrid.py

# --- local-inference-lab/vllm fork tree overlays ---
COPY projects/optimize-vllm/vllm/v1/worker/gpu/warmup.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/warmup.py
COPY projects/optimize-vllm/vllm/model_executor/layers/logits_processor.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py
COPY projects/optimize-vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
COPY projects/optimize-vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
COPY projects/optimize-vllm/vllm/v1/worker/gpu/spec_decode/eagle/utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/eagle/utils.py
COPY projects/optimize-vllm/vllm/distributed/device_communicators/all_reduce_utils.py /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/all_reduce_utils.py
COPY projects/optimize-vllm/vllm/distributed/device_communicators/pynccl_allocator.py /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/pynccl_allocator.py
COPY projects/optimize-vllm/vllm/v1/worker/gpu/attn_utils.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/attn_utils.py
COPY projects/optimize-vllm/vllm/v1/kv_cache_interface.py /opt/venv/lib/python3.12/site-packages/vllm/v1/kv_cache_interface.py
