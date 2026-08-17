# Overlay → Image → Upstream Manifest

Source of truth for every code change in the TR3/KVarN-K4V2 runtime.
Each row: overlay file in this folder, its destination inside the image
(site-packages paths), and the upstream repo the change belongs to for PRs.

Upstreams: local-inference-lab/vllm (fork branch
autoresearch/glm52-prefill-pp-20260727 in /home/js/projects/optimize-vllm,
ahead 41 of origin/dev/gilded-gnosis), b12x (B12X MLA sparse backend
upstream), sparkinfer (sparkinfer attention kernels).

## active-r634-b12x-compact/ → vllm site-packages (b12x + vLLM PR candidates)

- b12x_mla_sparse.py → vllm/v1/attention/backends/mla/b12x_mla_sparse.py (b12x; includes DCP1 fail-closed gate, DCP2 (4,2) prefill whitelist, direct-packed DCP4 contract)
- mla_attention.py → vllm/v1/attention/backends/mla/mla_attention.py (vLLM)
- kvarn_mla.py → vllm/v1/attention/ops/kvarn_mla.py (vLLM)
- config.py → vllm/model_executor/layers/quantization/kvarn/config.py (vLLM; KVarN dtype registry)
- kvarn_api_k4.py → sparkinfer/attention/kvarn_mla k4 API (sparkinfer; stage_k5_as_fp8_records)
- kv_cache_utils.py → vllm/v1/kv_cache_utils.py (vLLM; DCP scheduler block sizing)
- triton_kvarn_sinkhorn.py → vllm/v1/attention/ops/triton_kvarn_sinkhorn.py (vLLM; not on B12X hot path, retirement pool only)

## active-r634-kvarn-k4-native/ → vllm + sparkinfer (K4V2 native reader)

- kvarn_mla_state.py → vllm/v1/attention/backends/mla/kvarn_mla_state.py (vLLM)
- kvarn_mla/io.py, _shared/mla/kernel.py → sparkinfer KVarN native reader (sparkinfer)

## active-r634-runner/ → vllm

- model_runner.py → vllm/v1/worker/gpu/model_runner.py (vLLM)

## build-exact-h16-v1/ → vllm (spec decode + attention + config overlays)

- v1/worker/gpu/warmup.py, workspace.py, attn_utils.py
- v1/worker/gpu/spec_decode/{mtp/speculator.py, mtp/table_extension.py, dflash/*, dspark/*, eagle/utils.py}
- v1/attention/backends/{flash_attn.py, flex_attention.py}
- v1/attention/ops/dcp_alltoall.py
- compilation/passes/fusion/allreduce_rms_fusion.py
- config/{cache.py, vllm.py, speculative.py}, utils/torch_utils.py
- models/deepseek_v4/nvidia/dspark.py, model_executor/models/qwen3_dflash.py
- transformers_utils/configs/speculators/algos.py
- vllm/envs.py (env registry overlay, mounted via VLLM_ENVS_OVERLAY)

## build-exact-h16-v7/ → vllm + sparkinfer

- vllm/model_executor/layers/quantization/exl3.py (EXL3 loader fixes; vLLM PR)
- sparkinfer/moe/_shared/kernels/w4a16/{route_pack.py, mixed_trellis.py} (sparkinfer)

## r634-mtp-overlays/, r634-prefix-overlays/

- mla/indexer.py → vllm/v1/attention/backends/mla/indexer.py (vLLM)
- platforms/cuda.py → vllm/platforms/cuda.py (vLLM)

## shared-h-stack/sparkinfer-c40eb08f-full/ → sparkinfer

- sparkinfer/gemm/mxfp8_linear/_kernel.py, gemm/_bmm/api.py, gemm/_shared/bf16_bmm.py

## Fork-tree overlays (NOT in this folder — live in /home/js/projects/optimize-vllm, git-tracked)

- vllm/v1/worker/gpu/warmup.py (WARMUP_OVERLAY), model layers logits_processor.py,
  spec_decode/rejection_sampler{,_utils}.py, eagle/utils.py,
  distributed/device_communicators/{all_reduce_utils.py, pynccl_allocator.py},
  v1/worker/gpu/attn_utils.py, v1/kv_cache_interface.py
  → PR vehicle: the fork itself; commit dirty state before any branch work.

## Launcher

- start-full-expert-335-512k-c2-b2048-g4.sh — full env/mount contract;
  the baked image (glm52-kvarn-k4v2:tr3) removes the need for overlay
  mounts; only /model, /mtp, /cache remain runtime mounts.

Note: individual overlay files may differ from their fork-tree equivalents
(the launcher picks per file which tree wins). Before any PR, diff the
overlay file against the fork and upstream; this manifest records where
each running byte came from.
