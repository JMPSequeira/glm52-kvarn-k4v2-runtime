#!/bin/bash
set -euo pipefail

ROOT=/home/js/projects/glm52-shared-h-current
MODEL_ID=${MODEL_ID:-jpsequeira/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2}
HF_HOME_HOST=${HF_HOME_HOST:-/home/js/.cache/huggingface}
CONTAINER_NAME=${CONTAINER_NAME:-glm52-full-expert-production}
PORT=${PORT:-8001}
SRC=/home/js/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src
HYBRID_MOE_OVERLAY=${HYBRID_MOE_OVERLAY:-}
VLLM_ENVS_OVERLAY=${VLLM_ENVS_OVERLAY:-}
CACHE=/home/js/.cache/vllm-glm52
IMAGE=${IMAGE:-localhost/glm52-kvarn-k4v2:tr3}
ROUTE_PACK_OVERLAY=${ROUTE_PACK_OVERLAY:-}
EXL3_PREFILL_CAPACITY=${EXL3_PREFILL_CAPACITY:-2048}
EXL3_PREFILL_BLOCK_M=${EXL3_PREFILL_BLOCK_M:-32}
EXL3_PREFILL_TILE_CONFIG=${EXL3_PREFILL_TILE_CONFIG:-}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.979}
DCP_SIZE=${DCP_SIZE:-4}
NUM_GPU_BLOCKS=${NUM_GPU_BLOCKS:-2000}
TARGET_ATTENTION_BACKEND=${TARGET_ATTENTION_BACKEND:-B12X_MLA_SPARSE}
TARGET_KV_CACHE_DTYPE=${TARGET_KV_CACHE_DTYPE:-kvarn_mla_k4_g64}
MTP=${MTP:-3}
KVARN_NATIVE_CKV_GATHER=${KVARN_NATIVE_CKV_GATHER:-0}
KVARN_FUSED_CURRENT_STAGE=${KVARN_FUSED_CURRENT_STAGE:-0}
MTP_FUSED_TRAILING_AR=${MTP_FUSED_TRAILING_AR:-0}
MTP_GREEDY_DRAFT=${MTP_GREEDY_DRAFT:-0}
B12X_INDEXER_FUSED_QS_GATHER=${B12X_INDEXER_FUSED_QS_GATHER:-0}
B12X_BF16_BMM_QUERY=${B12X_BF16_BMM_QUERY:-0}
B12X_BF16_BMM_DCP_PROJECT=${B12X_BF16_BMM_DCP_PROJECT:-0}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-}
B12X_MOE_DECODE_M=${B12X_MOE_DECODE_M:-0}
PCIE_ONESHOT_ALLREDUCE_MAX=${PCIE_ONESHOT_ALLREDUCE_MAX:-84KB}
PCIE_ONESHOT_FUSED_MAX=${PCIE_ONESHOT_FUSED_MAX:-84KB}
ESTIMATE_CUDAGRAPHS=${ESTIMATE_CUDAGRAPHS:-1}
SPEC_METHOD=${SPEC_METHOD:-mtp}
SPEC_MODEL=${SPEC_MODEL:-}
SPEC_TOKENS=${SPEC_TOKENS:-}
SPEC_QUANTIZATION=${SPEC_QUANTIZATION:-}
SPEC_ATTENTION_BACKEND=${SPEC_ATTENTION_BACKEND:-}
SPEC_KV_CACHE_DTYPE=${SPEC_KV_CACHE_DTYPE:-bfloat16}
SPEC_SAMPLE_METHOD=${SPEC_SAMPLE_METHOD:-greedy}
DSPARK_SPS_CURVE=${DSPARK_SPS_CURVE:-}
DSPARK_DYNAMIC_DRAFT_DEPTH=${DSPARK_DYNAMIC_DRAFT_DEPTH:-0}
DSPARK_CAPACITY_LOG_INTERVAL=${DSPARK_CAPACITY_LOG_INTERVAL:-0}
DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=${DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE:-0}
DSPARK_DRAFT_TOKEN_BUDGET=${DSPARK_DRAFT_TOKEN_BUDGET:-0}
DSPARK_NGRAM_ASSIST=${DSPARK_NGRAM_ASSIST:-0}
DSPARK_NGRAM_K=${DSPARK_NGRAM_K:-5}
DSPARK_NGRAM_MAX_BATCH=${DSPARK_NGRAM_MAX_BATCH:-64}
DSPARK_NGRAM_MAX_INDEX_TOKENS=${DSPARK_NGRAM_MAX_INDEX_TOKENS:-65536}
DSPARK_NGRAM_ASSIST_LOG=${DSPARK_NGRAM_ASSIST_LOG:-}
DSPARK_NGRAM_ASSIST_OVERLAY=${DSPARK_NGRAM_ASSIST_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dspark/ngram_assist.py}
DSPARK_NGRAM_DEBUG=${DSPARK_NGRAM_DEBUG:-0}
MTP_TABLE_EXTENSION=${MTP_TABLE_EXTENSION:-0}
MTP_TABLE_K=${MTP_TABLE_K:-1}
MTP_TABLE_LOG=${MTP_TABLE_LOG:-}
MTP_TABLE_SPECULATOR_OVERLAY=${MTP_TABLE_SPECULATOR_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py}
MTP_TABLE_EXTENSION_OVERLAY=${MTP_TABLE_EXTENSION_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/mtp/table_extension.py}
EXL3_TRELLIS_ROUTE_BLOCK_SIZE=${EXL3_TRELLIS_ROUTE_BLOCK_SIZE:-8}
ENABLE_SP=${ENABLE_SP:-auto}
case $ENABLE_SP in
  auto) COMPILATION_PASS_CONFIG= ;;
  true|false)
    COMPILATION_PASS_CONFIG=",\"pass_config\":{\"enable_sp\":$ENABLE_SP}"
    ;;
  *)
    echo "ENABLE_SP must be auto, true, or false; got $ENABLE_SP" >&2
    exit 2
    ;;
esac
EXTRACT_HIDDEN_STATES_DIR=${EXTRACT_HIDDEN_STATES_DIR:-}
DFLASH_SPECULATOR_OVERLAY=${DFLASH_SPECULATOR_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py}
DSPARK_SPECULATOR_OVERLAY=${DSPARK_SPECULATOR_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py}
V1_MODEL_RUNNER_OVERLAY=${V1_MODEL_RUNNER_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu_model_runner.py}
EXTRACT_HIDDEN_STATE_LAYERS=${EXTRACT_HIDDEN_STATE_LAYERS:-2,20,39,58,75,78}
DISABLE_PREFIX_CACHING=${DISABLE_PREFIX_CACHING:-0}
USE_V2_MODEL_RUNNER=${USE_V2_MODEL_RUNNER:-1}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}
DCP_A2A_LARGE_BACKEND=${DCP_A2A_LARGE_BACKEND:-ag_rs}
DCP_A2A_MAX_TOKENS=${DCP_A2A_MAX_TOKENS:-4}
DCP_PREFILL_WORKSPACE=${DCP_PREFILL_WORKSPACE:-1}
CKV_GATHER=${CKV_GATHER:-1}
CKV_GATHER_MIN_TOKENS=${CKV_GATHER_MIN_TOKENS:-1024}
CKV_GATHER_MAX_TOKENS=${CKV_GATHER_MAX_TOKENS:-65540}
CKV_PREFETCH_DEPTH=${CKV_PREFETCH_DEPTH:-1}
CKV_EXECUTION_LANES=${CKV_EXECUTION_LANES:-2}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-0}
ENABLE_DBO=${ENABLE_DBO:-0}
NCCL_SYMM_MEM=${NCCL_SYMM_MEM:-0}
SYMM_MEM_PCIE_SAFE_BARRIER=${SYMM_MEM_PCIE_SAFE_BARRIER:-0}
NCCL_PROTO=${NCCL_PROTO:-LL,LL128,Simple}
VLLM_GREEDY_LOCAL_ARGMAX=${VLLM_GREEDY_LOCAL_ARGMAX:-0}
VLLM_KVARN_DETERMINISTIC_DCP_COMPACT=${VLLM_KVARN_DETERMINISTIC_DCP_COMPACT:-0}
VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-0}
VLLM_USE_B12X_SPARSE_INDEXER=${VLLM_USE_B12X_SPARSE_INDEXER:-1}
VLLM_INDEXER_HASH_DIAG=${VLLM_INDEXER_HASH_DIAG:-0}
VLLM_DETERMINISTIC_LOCAL_PREFILL_TOPK=${VLLM_DETERMINISTIC_LOCAL_PREFILL_TOPK:-0}
VLLM_DETERMINISTIC_DECODE_TOPK=${VLLM_DETERMINISTIC_DECODE_TOPK:-hierarchy}
VLLM_GLM_STAGE_HASH=${VLLM_GLM_STAGE_HASH:-0}
VLLM_EXL3_FUSED_ROUTE_PACK=${VLLM_EXL3_FUSED_ROUTE_PACK:-0}
SPARKINFER_KVARN_MLA_M9_HPP4_MERGE=${SPARKINFER_KVARN_MLA_M9_HPP4_MERGE:-0}
VLLM_KVARN_DETERMINISTIC_SINKHORN=${VLLM_KVARN_DETERMINISTIC_SINKHORN:-0}
VLLM_FULL_GRAPH_REQUEST_STATE_GUARD=${VLLM_FULL_GRAPH_REQUEST_STATE_GUARD:-0}
VLLM_B12X_CKV_POOL_STATE=${VLLM_B12X_CKV_POOL_STATE:-0}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
PCIE_DMA_PIECES=${PCIE_DMA_PIECES:-0}
MODEL_RUNNER_OVERLAY=${MODEL_RUNNER_OVERLAY:-$SRC/active-r634-runner/model_runner.py}
WARMUP_OVERLAY=${WARMUP_OVERLAY:-/home/js/projects/optimize-vllm/vllm/v1/worker/gpu/warmup.py}
LOGITS_PROCESSOR_OVERLAY=${LOGITS_PROCESSOR_OVERLAY:-/home/js/projects/optimize-vllm/vllm/model_executor/layers/logits_processor.py}
REJECTION_SAMPLER_OVERLAY=${REJECTION_SAMPLER_OVERLAY:-/home/js/projects/optimize-vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py}
REJECTION_SAMPLER_UTILS_OVERLAY=${REJECTION_SAMPLER_UTILS_OVERLAY:-/home/js/projects/optimize-vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py}
ALLREDUCE_RMS_FUSION_OVERLAY=${ALLREDUCE_RMS_FUSION_OVERLAY-$SRC/build-exact-h16-v1/vllm/compilation/passes/fusion/allreduce_rms_fusion.py}
MXFP8_KERNEL_OVERLAY=${MXFP8_KERNEL_OVERLAY:-$SRC/shared-h-stack/sparkinfer-c40eb08f-full/sparkinfer/gemm/mxfp8_linear/_kernel.py}
MIXED_TRELLIS_OVERLAY=${MIXED_TRELLIS_OVERLAY:-$SRC/build-exact-h16-v7/sparkinfer/moe/_shared/kernels/w4a16/mixed_trellis.py}
WORKSPACE_OVERLAY=${WORKSPACE_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/workspace.py}
B12X_MLA_OVERLAY=${B12X_MLA_OVERLAY:-$SRC/active-r634-b12x-compact/b12x_mla_sparse.py}
MLA_ATTENTION_OVERLAY=${MLA_ATTENTION_OVERLAY:-$SRC/active-r634-b12x-compact/mla_attention.py}
SPARKINFER_BMM_API_OVERLAY=${SPARKINFER_BMM_API_OVERLAY:-$SRC/shared-h-stack/sparkinfer-c40eb08f-full/sparkinfer/gemm/_bmm/api.py}
SPARKINFER_BF16_BMM_OVERLAY=${SPARKINFER_BF16_BMM_OVERLAY:-$SRC/shared-h-stack/sparkinfer-c40eb08f-full/sparkinfer/gemm/_shared/bf16_bmm.py}
SPARSE_UTILS_OVERLAY=${SPARSE_UTILS_OVERLAY:-$SRC/runtime-vllm-extras/vllm/v1/attention/backends/mla/sparse_utils.py}
SPARSE_ATTN_INDEXER_OVERLAY=${SPARSE_ATTN_INDEXER_OVERLAY:-}
DCP_INDEXER_CUTEDSL_OVERLAY=${DCP_INDEXER_CUTEDSL_OVERLAY:-}
FUSED_INDEXER_OVERLAY=${FUSED_INDEXER_OVERLAY:-}
TILED_TOPK_OVERLAY=${TILED_TOPK_OVERLAY:-}
FLASHINFER_TOPK_HEADER_OVERLAY=${FLASHINFER_TOPK_HEADER_OVERLAY:-}
GLM_TARGET_MODEL_OVERLAY=${GLM_TARGET_MODEL_OVERLAY:-}
KVARN_MLA_OPS_OVERLAY=${KVARN_MLA_OPS_OVERLAY:-$SRC/active-r634-b12x-compact/kvarn_mla.py}
KVARN_CONFIG_OVERLAY=${KVARN_CONFIG_OVERLAY:-$SRC/active-r634-b12x-compact/config.py}
KVARN_NATIVE_API_OVERLAY=${KVARN_NATIVE_API_OVERLAY:-$SRC/active-r634-b12x-compact/kvarn_api_k4.py}
KVARN_NATIVE_IO_OVERLAY=${KVARN_NATIVE_IO_OVERLAY:-$SRC/active-r634-kvarn-k4-native/kvarn_mla/io.py}
KVARN_NATIVE_KERNEL_OVERLAY=${KVARN_NATIVE_KERNEL_OVERLAY:-$SRC/active-r634-kvarn-k4-native/_shared/mla/kernel.py}
ALL_REDUCE_UTILS_OVERLAY=${ALL_REDUCE_UTILS_OVERLAY:-/home/js/projects/optimize-vllm/vllm/distributed/device_communicators/all_reduce_utils.py}
PYNCCL_ALLOCATOR_OVERLAY=${PYNCCL_ALLOCATOR_OVERLAY:-/home/js/projects/optimize-vllm/vllm/distributed/device_communicators/pynccl_allocator.py}
KVARN_MLA_STATE_OVERLAY=${KVARN_MLA_STATE_OVERLAY:-$SRC/active-r634-kvarn-k4-native/kvarn_mla_state.py}
KVARN_SINKHORN_OVERLAY=${KVARN_SINKHORN_OVERLAY:-$SRC/active-r634-b12x-compact/triton_kvarn_sinkhorn.py}
EAGLE_UTILS_OVERLAY=${EAGLE_UTILS_OVERLAY:-/home/js/projects/optimize-vllm/vllm/v1/worker/gpu/spec_decode/eagle/utils.py}
FLASH_ATTN_OVERLAY=${FLASH_ATTN_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/attention/backends/flash_attn.py}
FLEX_ATTN_OVERLAY=${FLEX_ATTN_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/attention/backends/flex_attention.py}
DFLASH_CUDAGRAPH_OVERLAY=${DFLASH_CUDAGRAPH_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py}
ATTN_UTILS_OVERLAY=${ATTN_UTILS_OVERLAY:-/home/js/projects/optimize-vllm/vllm/v1/worker/gpu/attn_utils.py}
KV_CACHE_INTERFACE_OVERLAY=${KV_CACHE_INTERFACE_OVERLAY:-/home/js/projects/optimize-vllm/vllm/v1/kv_cache_interface.py}
DSPARK_UTILS_OVERLAY=${DSPARK_UTILS_OVERLAY:-$SRC/build-exact-h16-v1/vllm/v1/worker/gpu/spec_decode/dspark/utils.py}
GLM_DSPARK_MODEL_OVERLAY=${GLM_DSPARK_MODEL_OVERLAY:-$SRC/build-exact-h16-v1/vllm/models/deepseek_v4/nvidia/dspark.py}
QWEN3_DFLASH_OVERLAY=${QWEN3_DFLASH_OVERLAY:-$SRC/build-exact-h16-v1/vllm/model_executor/models/qwen3_dflash.py}
SPECULATORS_ALGOS_OVERLAY=${SPECULATORS_ALGOS_OVERLAY:-$SRC/build-exact-h16-v1/vllm/transformers_utils/configs/speculators/algos.py}
KV_CACHE_UTILS_OVERLAY=${KV_CACHE_UTILS_OVERLAY:-$SRC/active-r634-b12x-compact/kv_cache_utils.py}
CACHE_CONFIG_OVERLAY=${CACHE_CONFIG_OVERLAY:-$SRC/build-exact-h16-v1/vllm/config/cache.py}
VLLM_CONFIG_OVERLAY=${VLLM_CONFIG_OVERLAY:-$SRC/build-exact-h16-v1/vllm/config/vllm.py}
TORCH_UTILS_OVERLAY=${TORCH_UTILS_OVERLAY:-$SRC/build-exact-h16-v1/vllm/utils/torch_utils.py}
SPEC_CONFIG_OVERLAY=${SPEC_CONFIG_OVERLAY:-$SRC/build-exact-h16-v1/vllm/config/speculative.py}
KVARN_MLA_PRECISION_TAIL_TOKENS=${KVARN_MLA_PRECISION_TAIL_TOKENS:-3072}
KVARN_MLA_DIAG_EXACT_ROWS=${KVARN_MLA_DIAG_EXACT_ROWS:-0}
KVARN_DIRECT_PACKED=${KVARN_DIRECT_PACKED:-1}
KVARN_NATIVE_CUTE=${KVARN_NATIVE_CUTE:-1}
PROMPT_LOGITS_PLAN=${PROMPT_LOGITS_PLAN:-}
PROMPT_LOGITS_PLAN_SHA256=${PROMPT_LOGITS_PLAN_SHA256:-}
PROMPT_LOGITS_OUTPUT=${PROMPT_LOGITS_OUTPUT:-}
PROMPT_LOGITS_CAPTURE_OVERLAY=${PROMPT_LOGITS_CAPTURE_OVERLAY:-}
PROMPT_LOGPROB_OVERLAY=${PROMPT_LOGPROB_OVERLAY:-}
PROMPT_LOGPROBS_CHUNK_SIZE=${PROMPT_LOGPROBS_CHUNK_SIZE:-}
PROFILER_OUTPUT=${PROFILER_OUTPUT:-}
PROFILER_MAX_ITERATIONS=${PROFILER_MAX_ITERATIONS:-8}
PROFILER_ARGS=()
SPECULATIVE_ARGS=()
KV_TRANSFER_ARGS=()
HIDDEN_STATES_MOUNT_ARGS=()
SCHEDULER_ARGS=(
  --enable-chunked-prefill
  --enable-prefix-caching
  --async-scheduling
)
if (( DISABLE_PREFIX_CACHING )); then
  SCHEDULER_ARGS=(
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --async-scheduling
  )
fi
if [[ $SPEC_METHOD == extract_hidden_states ]]; then
  [[ -n $EXTRACT_HIDDEN_STATES_DIR ]] || {
    echo "EXTRACT_HIDDEN_STATES_DIR is required for extract_hidden_states" >&2
    exit 2
  }
  mkdir -p "$EXTRACT_HIDDEN_STATES_DIR"
  USE_V2_MODEL_RUNNER=0
  PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
  DCP_SIZE=1
  TARGET_ATTENTION_BACKEND=B12X_MLA_SPARSE
  TARGET_KV_CACHE_DTYPE=kvarn_mla_k5_g64
  CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-1}
  EXL3_TRELLIS_MAX_M=${EXL3_TRELLIS_MAX_M:-32}
  SPECULATIVE_ARGS=(
    --speculative-config
    "{\"method\":\"extract_hidden_states\",\"num_speculative_tokens\":1,\"draft_model_config\":{\"hf_config\":{\"eagle_aux_hidden_state_layer_ids\":[$EXTRACT_HIDDEN_STATE_LAYERS]}}}"
  )
  KV_TRANSFER_ARGS=(
    --kv-transfer-config
    '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"allow_custom_save_path":true,"num_writer_threads":8,"shared_storage_path":"/hidden-states"}}'
  )
  HIDDEN_STATES_MOUNT_ARGS=(
    --volume "$EXTRACT_HIDDEN_STATES_DIR:/hidden-states"
  )
  SCHEDULER_ARGS=(
    --no-enable-chunked-prefill
    --no-enable-prefix-caching
    --no-async-scheduling
  )
elif [[ $SPEC_METHOD == dflash || $SPEC_METHOD == dspark ]]; then
  [[ -n $SPEC_MODEL ]] || {
    echo "SPEC_MODEL is required for $SPEC_METHOD" >&2
    exit 2
  }
  if [[ $SPEC_METHOD == dspark ]]; then
    SPEC_TOKENS=${SPEC_TOKENS:-8}
    SPEC_ATTENTION_BACKEND=${SPEC_ATTENTION_BACKEND:-FLEX_ATTENTION}
    CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-9,1}
    EXL3_TRELLIS_MAX_M=${EXL3_TRELLIS_MAX_M:-$((SPEC_TOKENS + 1))}
    if [[ -n $DSPARK_SPS_CURVE ]]; then
      DSPARK_CONFIG_JSON=",\"dspark_sps_curve\":\"$DSPARK_SPS_CURVE\""
    else
      DSPARK_CONFIG_JSON=
    fi
  else
    SPEC_TOKENS=${SPEC_TOKENS:-15}
    SPEC_ATTENTION_BACKEND=${SPEC_ATTENTION_BACKEND:-FLASH_ATTN}
    CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-17,16,4,1}
    EXL3_TRELLIS_MAX_M=${EXL3_TRELLIS_MAX_M:-32}
    DSPARK_CONFIG_JSON=
  fi
  if [[ -e $SPEC_MODEL ]]; then
    SPEC_MODEL_CONTAINER=/spec-model
  else
    SPEC_MODEL_CONTAINER=$SPEC_MODEL
  fi
   SPEC_QUANT_JSON=
  if [[ -n $SPEC_QUANTIZATION ]]; then
    SPEC_QUANT_JSON=",\"quantization\":\"$SPEC_QUANTIZATION\""
  fi
  SPECULATIVE_ARGS=(
    --speculative-config
    "{\"attention_backend\":\"$SPEC_ATTENTION_BACKEND\",\"draft_sample_method\":\"$SPEC_SAMPLE_METHOD\",\"kv_cache_dtype\":\"$SPEC_KV_CACHE_DTYPE\",\"method\":\"$SPEC_METHOD\",\"model\":\"$SPEC_MODEL_CONTAINER\",\"num_speculative_tokens\":$SPEC_TOKENS$SPEC_QUANT_JSON$DSPARK_CONFIG_JSON}"
  )
elif [[ $MTP == 0 ]]; then
  CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-1}
  EXL3_TRELLIS_MAX_M=${EXL3_TRELLIS_MAX_M:-32}
else
  MTP_LOCAL_ARGMAX_JSON=
  if [[ ${MTP_LOCAL_ARGMAX:-0} == 1 ]]; then
    MTP_LOCAL_ARGMAX_JSON=",\"use_local_argmax_reduction\":true"
  fi
  MTP_SPEC_PER_BATCH_JSON=
  if [[ -n ${MTP_SPEC_TOKENS_PER_BATCH:-} ]]; then
    MTP_SPEC_PER_BATCH_JSON=",\"num_speculative_tokens_per_batch_size\":${MTP_SPEC_TOKENS_PER_BATCH}"
  fi
  MTP_DRAFT_SAMPLE_METHOD=probabilistic
  if [[ ${MTP_GREEDY_DRAFT:-0} == 1 ]]; then
    MTP_DRAFT_SAMPLE_METHOD=greedy
  fi
  if ((MTP_TABLE_EXTENSION > 0)); then
    if [[ ${MTP_GREEDY_DRAFT:-0} != 1 ]]; then
      echo "MTP_TABLE_EXTENSION requires MTP_GREEDY_DRAFT=1 (greedy draft)" >&2
      exit 2
    fi
    # Width split: the engine/draft block widens to MTP + extension
    # (verify rows +1); the MTP head itself still runs only $MTP steps.
    MTP_SPEC_TOKENS=$((MTP + MTP_TABLE_EXTENSION))
    CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-$((MTP_SPEC_TOKENS + 1)),1}
  else
    MTP_SPEC_TOKENS=$MTP
    CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-4,1}
  fi
  # Natural MTP: the draft head builds from the target checkpoint itself
  # (speculative.py resolves method=mtp + no model to the target model), so
  # layer-78 weights ship inside the main checkpoint index. No draft dir.
  SPECULATIVE_ARGS=(
    --speculative-config
    "{\"attention_backend\":\"B12X_MLA_SPARSE\",\"draft_sample_method\":\"$MTP_DRAFT_SAMPLE_METHOD\",\"method\":\"mtp\",\"moe_backend\":\"b12x\",\"num_speculative_tokens\":$MTP_SPEC_TOKENS$MTP_LOCAL_ARGMAX_JSON$MTP_SPEC_PER_BATCH_JSON}"
  )
  EXL3_TRELLIS_MAX_M=${EXL3_TRELLIS_MAX_M:-$((MTP_SPEC_TOKENS + 1))}
fi
if [[ $SPEC_METHOD == dflash || $SPEC_METHOD == dspark ]]; then
  B12X_SPEC_DECODE_MAX_Q=${B12X_SPEC_DECODE_MAX_Q:-$((SPEC_TOKENS + 1))}
elif ((MTP > 0)); then
  B12X_SPEC_DECODE_MAX_Q=${B12X_SPEC_DECODE_MAX_Q:-$((MTP_SPEC_TOKENS + 1))}
else
  B12X_SPEC_DECODE_MAX_Q=${B12X_SPEC_DECODE_MAX_Q:-1}
fi
# Single source of truth: DISABLE_PREFIX_CACHING. (Two gates previously
# fought: SCHEDULER_ARGS honored DISABLE while this later array defaulted
# to no-prefix-caching and overrode it — last argparse flag wins.)
if (( DISABLE_PREFIX_CACHING )); then
  PREFIX_CACHING_ARGS=(--no-enable-prefix-caching)
else
  PREFIX_CACHING_ARGS=(--enable-prefix-caching)
fi
ALLREDUCE_ARGS=()
if [[ $DISABLE_CUSTOM_ALL_REDUCE == 1 ]]; then
  ALLREDUCE_ARGS=(--disable-custom-all-reduce)
fi
NCCL_ARGS=(
  --env NCCL_PROTO="$NCCL_PROTO"
  --env VLLM_GREEDY_LOCAL_ARGMAX="$VLLM_GREEDY_LOCAL_ARGMAX"
)
if [[ -n ${NCCL_ALGO:-} ]]; then
  NCCL_ARGS+=(--env NCCL_ALGO="$NCCL_ALGO")
fi
DBO_ARGS=()
if [[ $ENABLE_DBO == 1 ]]; then
  DBO_ARGS=(--enable-dbo)
fi
ROUTE_PACK_ARGS=()
RUNTIME_OVERLAY_ARGS=()
if [[ -n $PROFILER_OUTPUT ]]; then
  RUNTIME_OVERLAY_ARGS+=(
    --volume "$PROFILER_OUTPUT:/profiles"
  )
  PROFILER_ARGS=(
    --profiler-config
    "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/profiles\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"torch_profiler_record_shapes\":true,\"max_iterations\":$PROFILER_MAX_ITERATIONS}"
  )
fi
if [[ -n $MODEL_RUNNER_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $WARMUP_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $LOGITS_PROCESSOR_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $REJECTION_SAMPLER_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $REJECTION_SAMPLER_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $ALLREDUCE_RMS_FUSION_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $MXFP8_KERNEL_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $PYNCCL_ALLOCATOR_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $ALL_REDUCE_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $MIXED_TRELLIS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $WORKSPACE_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $B12X_MLA_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $MLA_ATTENTION_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $SPARKINFER_BMM_API_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $SPARKINFER_BF16_BMM_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $SPARSE_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $SPARSE_ATTN_INDEXER_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $DCP_INDEXER_CUTEDSL_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $FUSED_INDEXER_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $TILED_TOPK_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $FLASHINFER_TOPK_HEADER_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
    --env FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-sm120
  )
fi
if [[ -n $GLM_TARGET_MODEL_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KVARN_MLA_OPS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KVARN_CONFIG_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KVARN_NATIVE_API_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KVARN_NATIVE_IO_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KVARN_NATIVE_KERNEL_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $HYBRID_MOE_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $VLLM_ENVS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $V1_MODEL_RUNNER_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KVARN_MLA_STATE_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KVARN_SINKHORN_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KV_CACHE_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $CACHE_CONFIG_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $VLLM_CONFIG_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $TORCH_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $SPEC_CONFIG_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $EAGLE_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $FLASH_ATTN_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $FLEX_ATTN_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $DFLASH_CUDAGRAPH_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $ATTN_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $KV_CACHE_INTERFACE_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $DSPARK_UTILS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $GLM_DSPARK_MODEL_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $QWEN3_DFLASH_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $SPECULATORS_ALGOS_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $DFLASH_SPECULATOR_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
if [[ -n $DSPARK_SPECULATOR_OVERLAY ]]; then
  RUNTIME_OVERLAY_ARGS+=(
  )
fi
NGRAM_ENV_ARGS=()
if [[ $DSPARK_NGRAM_ASSIST == 1 ]]; then
  # Host-side n-gram draft override (default-off; control runs expand
  # this and the module mount to nothing).
  NGRAM_ENV_ARGS=(
    --env VLLM_DSPARK_NGRAM_ASSIST=1
    --env VLLM_DSPARK_NGRAM_K="$DSPARK_NGRAM_K"
    --env VLLM_DSPARK_NGRAM_MAX_BATCH="$DSPARK_NGRAM_MAX_BATCH"
    --env VLLM_DSPARK_NGRAM_MAX_INDEX_TOKENS="$DSPARK_NGRAM_MAX_INDEX_TOKENS"
  )
  if [[ -n $DSPARK_NGRAM_ASSIST_LOG ]]; then
    NGRAM_ENV_ARGS+=(--env VLLM_DSPARK_NGRAM_ASSIST_LOG="$DSPARK_NGRAM_ASSIST_LOG")
  fi
  if [[ $DSPARK_NGRAM_DEBUG != 0 ]]; then
    NGRAM_ENV_ARGS+=(--env VLLM_DSPARK_NGRAM_DEBUG="$DSPARK_NGRAM_DEBUG")
  fi
  if [[ -n $DSPARK_NGRAM_ASSIST_OVERLAY ]]; then
    RUNTIME_OVERLAY_ARGS+=(
    )
  fi
fi

MTP_TABLE_ENV_ARGS=()
if ((MTP_TABLE_EXTENSION > 0)); then
  # Host-side n-gram table extension beyond the MTP head (default-off;
  # control runs with MTP_TABLE_EXTENSION=0 expand this and both module
  # mounts to nothing, leaving the launch byte-identical).
  if [[ $SPEC_METHOD != mtp ]]; then
    echo "MTP_TABLE_EXTENSION requires SPEC_METHOD=mtp; got $SPEC_METHOD" >&2
    exit 2
  fi
  if ((MTP_TABLE_EXTENSION >= MTP)); then
    echo "MTP_TABLE_EXTENSION must leave at least one MTP head step" >&2
    exit 2
  fi
  MTP_TABLE_ENV_ARGS=(
    --env VLLM_MTP_TABLE_EXTENSION="$MTP_TABLE_EXTENSION"
    --env VLLM_MTP_TABLE_K="$MTP_TABLE_K"
  )
  if [[ -n $MTP_TABLE_LOG ]]; then
    MTP_TABLE_ENV_ARGS+=(--env VLLM_MTP_TABLE_LOG="$MTP_TABLE_LOG")
  fi
  if [[ -n $MTP_TABLE_SPECULATOR_OVERLAY ]]; then
    RUNTIME_OVERLAY_ARGS+=(
    )
  fi
  if [[ -n $MTP_TABLE_EXTENSION_OVERLAY ]]; then
    RUNTIME_OVERLAY_ARGS+=(
    )
  fi
fi
if [[ -n $PROMPT_LOGITS_PLAN ]]; then
  if [[ -z $PROMPT_LOGITS_PLAN_SHA256 || -z $PROMPT_LOGITS_OUTPUT \
    || -z $PROMPT_LOGITS_CAPTURE_OVERLAY || -z $PROMPT_LOGPROB_OVERLAY ]]; then
    echo "Prompt-logits capture requires plan hash, output, and both overlays" >&2
    exit 2
  fi
  RUNTIME_OVERLAY_ARGS+=(
    --volume "$PROMPT_LOGITS_PLAN:/capture-plan.json:ro"
    --volume "$PROMPT_LOGITS_OUTPUT:/capture-output"
    --env GLM_ITEM13_PROMPT_LOGITS_PLAN=/capture-plan.json
    --env GLM_ITEM13_PROMPT_LOGITS_PLAN_SHA256="$PROMPT_LOGITS_PLAN_SHA256"
    --env GLM_ITEM13_PROMPT_LOGITS_DIR=/capture-output
  )
  if [[ -n $PROMPT_LOGPROBS_CHUNK_SIZE ]]; then
    RUNTIME_OVERLAY_ARGS+=(
      --env VLLM_PROMPT_LOGPROBS_CHUNK_SIZE="$PROMPT_LOGPROBS_CHUNK_SIZE"
    )
  fi
fi

exec podman run --rm --pull=never --replace \
  --name "$CONTAINER_NAME" \
  --volume "$SRC/active-r634-runner/model_runner.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py" \
  --volume "$SRC/active-r634-b12x-compact/b12x_mla_sparse.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py" \
  --network host --ipc host --security-opt label=disable \
  --pids-limit 4096 \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 \
  --volume "$HF_HOME_HOST:/hf-cache:ro" \
  $([[ -e ${SPEC_MODEL:-} ]] && echo --volume "${SPEC_MODEL}:/spec-model:ro") \
  "${ROUTE_PACK_ARGS[@]}" \
  "${RUNTIME_OVERLAY_ARGS[@]}" \
  --volume "$CACHE:/cache" \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_CACHE=/hf-cache/hub \
  --env MAX_JOBS=4 --env MAX_WORKERS=4 \
  --env OMP_NUM_THREADS=8 --env MKL_NUM_THREADS=8 --env NUMEXPR_MAX_THREADS=8 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  --env VLLM_USE_V2_MODEL_RUNNER="$USE_V2_MODEL_RUNNER" \
  --env VLLM_USE_B12X_MHC=1 \
  --env VLLM_USE_B12X_SPARSE_INDEXER="$VLLM_USE_B12X_SPARSE_INDEXER" \
  --env VLLM_KVARN_MLA_FUSED_CURRENT_STAGE="$KVARN_FUSED_CURRENT_STAGE" \
  --env VLLM_KVARN_MLA_NATIVE_CKV_GATHER="$KVARN_NATIVE_CKV_GATHER" \
  --env VLLM_MTP_FUSED_TRAILING_AR="$MTP_FUSED_TRAILING_AR" \
  --env VLLM_FIXED_MTP3_PROPOSER_SUPERGRAPH="${MTP3_PROPOSER_SUPERGRAPH:-0}" \
  --env VLLM_B12X_INDEXER_FUSED_QS_GATHER="$B12X_INDEXER_FUSED_QS_GATHER" \
  --env VLLM_B12X_BF16_BMM_QUERY="$B12X_BF16_BMM_QUERY" \
  --env VLLM_B12X_BF16_BMM_DCP_PROJECT="$B12X_BF16_BMM_DCP_PROJECT" \
  --env VLLM_B12X_MOE_DECODE_M="$B12X_MOE_DECODE_M" \
  --env VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="$PCIE_ONESHOT_ALLREDUCE_MAX" \
  --env VLLM_EXL3_TRELLIS_MIN_M="${VLLM_EXL3_TRELLIS_MIN_M:-1}" \
  --env VLLM_EXL3_TRELLIS_MAX_M="${EXL3_TRELLIS_MAX_M:-}" \
  --env VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="$PCIE_ONESHOT_FUSED_MAX" \
  --env VLLM_INDEXER_HASH_DIAG="$VLLM_INDEXER_HASH_DIAG" \
  --env VLLM_DETERMINISTIC_LOCAL_PREFILL_TOPK="$VLLM_DETERMINISTIC_LOCAL_PREFILL_TOPK" \
  --env VLLM_DETERMINISTIC_DECODE_TOPK="$VLLM_DETERMINISTIC_DECODE_TOPK" \
  --env VLLM_GLM_STAGE_HASH="$VLLM_GLM_STAGE_HASH" \
  --env VLLM_EXL3_FUSED_ROUTE_PACK="$VLLM_EXL3_FUSED_ROUTE_PACK" \
  --env SPARKINFER_KVARN_MLA_M9_HPP4_MERGE="$SPARKINFER_KVARN_MLA_M9_HPP4_MERGE" \
  --env VLLM_KVARN_DETERMINISTIC_SINKHORN="$VLLM_KVARN_DETERMINISTIC_SINKHORN" \
  --env VLLM_FULL_GRAPH_REQUEST_STATE_GUARD="$VLLM_FULL_GRAPH_REQUEST_STATE_GUARD" \
  --env VLLM_B12X_CKV_POOL_STATE="$VLLM_B12X_CKV_POOL_STATE" \
  --env VLLM_USE_B12X_PCIE_DMA=1 \
  --env VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  --env VLLM_USE_B12X_DCP_A2A=1 \
  --env VLLM_B12X_DCP_LSE_REDUCE=0 \
  --env VLLM_KVARN_DETERMINISTIC_DCP_COMPACT="$VLLM_KVARN_DETERMINISTIC_DCP_COMPACT" \
  --env VLLM_DEBUG_HYBRID_GATE="${VLLM_DEBUG_HYBRID_GATE:-}" \
  --env VLLM_BATCH_INVARIANT="$VLLM_BATCH_INVARIANT" \
  --env VLLM_DCP_A2A_MAX_TOKENS="$DCP_A2A_MAX_TOKENS" \
  --env VLLM_DCP_A2A_LARGE_BACKEND="$DCP_A2A_LARGE_BACKEND" \
  --env VLLM_EXL3_PREFILL_CAPACITY="$EXL3_PREFILL_CAPACITY" \
  --env VLLM_EXL3_PREFILL_BLOCK_M="$EXL3_PREFILL_BLOCK_M" \
  --env VLLM_EXL3_PREFILL_TILE_CONFIG="$EXL3_PREFILL_TILE_CONFIG" \
  --env VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="$ESTIMATE_CUDAGRAPHS" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  --env VLLM_USE_NCCL_SYMM_MEM="$NCCL_SYMM_MEM" \
  --env VLLM_SYMM_MEM_PCIE_SAFE_BARRIER="$SYMM_MEM_PCIE_SAFE_BARRIER" \
  "${HIDDEN_STATES_MOUNT_ARGS[@]}" \
  --env VLLM_CUDAGRAPH_ESTIMATED_MEMORY_GIB=0.36 \
  --env VLLM_SHARE_TARGET_DRAFT_WORKSPACE=1 \
  --env VLLM_DCP_GLOBAL_TOPK=1 \
  --env VLLM_DCP_SHARD_DRAFT=1 \
  "${NCCL_ARGS[@]}" \
  --env LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib \
  --env VLLM_DCP_PROJECT_BEFORE_MERGE=1 \
  --env VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS=1024 \
  --env VLLM_DCP_QUERY_SPLIT=1 \
  --env VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE="$DCP_PREFILL_WORKSPACE" \
  --env VLLM_B12X_MLA_CKV_GATHER="$CKV_GATHER" \
  --env VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS="$CKV_GATHER_MIN_TOKENS" \
  --env VLLM_NCCL_SO_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
  --env VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS="$CKV_GATHER_MAX_TOKENS" \
  "${MTP_TABLE_ENV_ARGS[@]}" \
  "${NGRAM_ENV_ARGS[@]}" \
  --env VLLM_B12X_MLA_CKV_PREFETCH_DEPTH="$CKV_PREFETCH_DEPTH" \
  --env VLLM_B12X_MLA_CKV_EXECUTION_LANES="$CKV_EXECUTION_LANES" \
  --env VLLM_EXL3_TRELLIS_ROUTE_BLOCK_SIZE="$EXL3_TRELLIS_ROUTE_BLOCK_SIZE" \
  --env VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=8192 \
  --env VLLM_DCP_TOPK_OWNER_MERGE=1 \
  --env SPARKINFER_KVARN_MLA_M4_CHUNKS_PER_SPLIT="${KVARN_M4_CHUNKS_PER_SPLIT:-1}" \
  --env SPARKINFER_PCIE_DMA_PIECES="$PCIE_DMA_PIECES" \
  --env VLLM_KVARN_MLA_DIRECT_PACKED_DECODE="$KVARN_DIRECT_PACKED" \
  --env VLLM_KVARN_MLA_NATIVE_CUTE="$KVARN_NATIVE_CUTE" \
  --env KVARN_MLA_PRECISION_TAIL_TOKENS="$KVARN_MLA_PRECISION_TAIL_TOKENS" \
  --env KVARN_MLA_DIAG_EXACT_ROWS="$KVARN_MLA_DIAG_EXACT_ROWS" \
  --env VLLM_B12X_MLA_SPEC_DECODE_MAX_Q="$B12X_SPEC_DECODE_MAX_Q" \
  --env VLLM_EXL3_TRELLIS_MAX_M="$EXL3_TRELLIS_MAX_M" \
  --env VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="$DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE" \
  --env VLLM_DSPARK_DRAFT_TOKEN_BUDGET="$DSPARK_DRAFT_TOKEN_BUDGET" \
  --env VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH="$DSPARK_DYNAMIC_DRAFT_DEPTH" \
  --env VLLM_DSPARK_CAPACITY_LOG_INTERVAL="$DSPARK_CAPACITY_LOG_INTERVAL" \
  --env NCCL_DEBUG=WARN --env PYTHONUNBUFFERED=1 \
  --env SAFETENSORS_FAST_GPU=1 \
  --env SAFETENSORS_FAST_GPU_SPILL_CHECK=1 \
  --env SAFETENSORS_FAST_GPU_CONVERT=1 \
  --env HF_HOME=/cache/huggingface \
  $([[ -n ${HF_HOST_CACHE:-} ]] && echo --env HF_HOME=/hf-host-cache --volume "$HF_HOST_CACHE:/hf-host-cache") \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/full-expert-335-b12x-query/torch_compile_cache \
  --env FLASHINFER_WORKSPACE_BASE=/cache/full-expert-335-b12x-query/flashinfer_workspace \
  --entrypoint /opt/venv/bin/python \
  "$IMAGE" -m vllm.entrypoints.openai.api_server \
  --model $MODEL_ID \
  --served-model-name GLM-5.2 \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size "$DCP_SIZE" \
  --dcp-comm-backend a2a \
  --attention-backend "$TARGET_ATTENTION_BACKEND" \
  "${ALLREDUCE_ARGS[@]}" \
  "${DBO_ARGS[@]}" \
  --kv-cache-dtype "$TARGET_KV_CACHE_DTYPE" \
  --block-size 64 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  $( if [[ ${NUM_GPU_BLOCKS:-} != auto ]]; then echo --num-gpu-blocks-override "$NUM_GPU_BLOCKS"; fi ) \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${SCHEDULER_ARGS[@]}" \
  "${PREFIX_CACHING_ARGS[@]}" \
  --disable-uvicorn-access-log \
  --seed 0 \
  --quantization exl3 \
  --load-format safetensors \
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]$COMPILATION_PASS_CONFIG}" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  "${PROFILER_ARGS[@]}" \
  "${KV_TRANSFER_ARGS[@]}" \
  "${SPECULATIVE_ARGS[@]}"
