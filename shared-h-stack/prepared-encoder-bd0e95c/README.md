# Calibration + EXL3-Trellis encoder (reproduction bundle)

These are the exact scripts and calibration corpus used to produce
**GLM-5.2-EXL3-TR3-3.0bpw** from the BF16 base
[`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2).

The routed MoE experts (layers 3–77, all 256 experts) are quantized to a 3.0-bpw
EXL3 **Trellis** representation with a **calibrated LDLQ** pass (proxy Hessians
from the corpus below); attention, dense MLPs (0–2), shared experts, router/gates,
MTP layer 78, embeddings, and LM head stay BF16. Weights are pre-sliced for **TP4**
and packed with the **MCG** codebook. It is **not** NVIDIA ModelOpt — the
`quant_method: modelopt` field in the model `config.json` is only a loader-dispatch
shim; the actual encoder is the calibrated LDLQ/Trellis pipeline here.

## Files

| File | Role |
| --- | --- |
| `encode_tr3_v31.py` | Production LDLQ/Trellis encoder (v3 cross-slice lockstep). SHA-256 `e9a85a47…75032`, pinned by `encode_b300.py`. |
| `encode_b300.py` | B300 adapter around `encode_tr3_v31.py`: source IO, RAM capture IO, all-256 tiering, resume, assembly, and the model `config.json` dispatch shim. |
| `capture_b300.py` | Builds the deterministic capture plan and captures per-layer proxy Hessians from the corpus. |
| `bootstrap_ext_b300.py` | Builds/loads exllamav3 0.0.43's six quantizer ops (`had_r_128`, `pack_trellis`, `quantize_tiles`, `reconstruct`, `reconstruct_slice`, `unpack_trellis`) as an `sm_100` extension. |
| `preflight_b300.py` | Environment/HBM/disk/corpus smoke checks. |
| `convert_b300.sh` | Orchestrator (stage runner). |
| `calibration/reap_recall_calib.jsonl` | The calibration corpus (12,228 samples). SHA-256 `cf247acc…44df4`, pinned by `capture_b300.py`. |

## Requirements

- **exllamav3 == 0.0.43**, installed with its source package (the extension is built
  from its sources; the package is never imported at runtime). `bootstrap_ext_b300.py`
  locates the installed sources and refuses any other version.
- **NVIDIA Blackwell** GPUs. The reference run used a **B300 (SM 10.0 / `sm_100`)** node
  with **8 GPUs** (`TORCH_CUDA_ARCH_LIST=10.0`, capture at TP8, packed for TP4).
- **CUDA 12.9** (`CUDA_HOME=/usr/local/cuda-12.9`), PyTorch with matching CUDA.
- The BF16 base model `zai-org/GLM-5.2`.
- Large tmpfs (`/dev/shm`) for the Hessian capture windows and ~0.5 TB scratch for
  assembly. The orchestrator enforces disk/RAM guards.

## Corpus schema

JSONL, one object per line:

```json
{"axis": "axis2_legal", "source": "neo4j_headnote:text", "text": "{\"messages\":[...]}", "meta": {...}}
```

`text` is a serialized chat/messages blob. 12,228 rows are balanced across four axes
(≈3,057 each): `axis1_general`, `axis2_legal`, `axis3_code_agentic`,
`axis4_reasoning_termination`. The legal axis is public-record material (case law,
headnotes, statutes) plus synthetic legal-reasoning items. To calibrate on your own
data, keep the same schema and re-point `--corpus`; the pinned SHA guard in
`capture_b300.py` is only there to reproduce *this* build byte-for-byte, so relax it
if you substitute a corpus.

## Reproduce

Set the paths the orchestrator expects (or export overrides), then run the stages.
`encode_tr3_v31.py` and the corpus must sit where the scripts look, e.g.:

```bash
export WORK_ROOT=/workspace/tr3
export BF16_SRC=/workspace/bf16                         # zai-org/GLM-5.2 (BF16)
export OWNER_CORPUS=$WORK_ROOT/calib/reap_recall_calib.jsonl
export BASE_ENCODER_PY=$WORK_ROOT/encode_tr3_v31.py
export CUDA_HOME=/usr/local/cuda-12.9

./convert_b300.sh preflight        # env + corpus + HBM smoke checks
./convert_b300.sh ext              # build the sm_100 exllamav3 0.0.43 ops
./convert_b300.sh plan             # deterministic capture manifest

# Process the MoE tail (layers 3..77) in windows of <=8 layers:
for W in 3-10 11-18 19-26 27-34 35-42 43-50 51-58 59-66 67-74 75-77; do
  LAYERS=$W ./convert_b300.sh capture-window   # proxy Hessians -> /dev/shm
  LAYERS=$W ./convert_b300.sh encode-window    # calibrated LDLQ/Trellis encode
done

./convert_b300.sh assemble         # assemble the final TP4 checkpoint + config
```

The output is a rank-sliced EXL3-Trellis checkpoint with the `hybrid_tr3_tail`
metadata block that vLLM + Sparkinfer use to take the pre-planned (CUDA-graph-safe)
kernel path. Serve it per the model card's runtime section with `-tp 4`.

## Method summary

Per routed-expert projection, per TP4 slice: Hadamard incoherence rotation (su/sv)
→ block LDL of the calibrated proxy Hessian → blocked LDLQ quantization with error
feedback → MCG trellis codebook at 3 bits. The 64 experts with the highest trellis
round-trip error can optionally be kept as an NVFP4 sidecar (disabled in this build,
`nvfp4_keep_per_layer: 0`).
