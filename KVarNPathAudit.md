# KVarN MLA Path — Full Read-Only Audit

**Date:** 2026-08-21
**Repo:** `GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2-src` (+ live fork tree `/home/js/projects/optimize-vllm`, dirty working tree)
**Mandate:** full sweep of the KVarN MLA KV-cache path; find the cause of sporadic garbled generation. No code changed, nothing executed.
**Method:** parent manual audit of all core store/read kernels + 12 read-only scout audits (first batch of 6 killed mid-run by a session interrupt; 6 successors resumed from the intact transcripts). Every claim below is cited to a file:line verified by at least one auditor; cross-checks are noted where two independent auditors covered the same link.

---

## 1. Executive summary

1. **The live decode path is fully mapped and every statically checkable link is verified consistent.** Quantization/dequantization identity (all writers and all readers), the Hadamard rotation contract, the DCP(4,1) global→local index chain, the exact-pool ownership state machine, CUDA-graph replay freshness, stream ordering, LSE math, and scheduler byte accounting all pass line-level verification in the *currently deployed (dirty) tree*.
2. **No statically-provable defect was found.** Seven initial hypotheses were raised; six were refuted with citations and one survives only as a *runtime* window (see §7). The failure class of the observed symptom (silent, row-selective, wrong-but-valid KV reads) requires a mapping/ownership/publication error, and all such *static* windows were closed.
3. **What remains:** (a) four runtime-only windows that static analysis cannot close, (b) one piece of dead defensive code that would make any future divergence of this class *silent*, and (c) one decisive data question — whether garbling onset tracks **seq_len > 2048** (top-k sparsification) or **~3264 tokens** (first packed row under the 3072-token precision tail) — which separates the two remaining code regions.
4. **The observed corruption is real but narrow in the evidence:** the hard case is `captured_cli` in `tests/goldens/kvarn_tr32_corrupt_run1.json` (tight 2-token repetition loop from generated token 1; control run on the same prompt is clean). The other battery cases in the same "corrupt" golden pass all rescore gates and likely include ordinary cross-format divergence, not corruption (§4).

**Top recommendation:** instrument one garbled run with the repo's built-in probe `KVARN_MLA_DIAG_EXACT_ROWS` plus a `seq_len` log at first-garble and a one-line scatter assert (§8). That run will localize the remaining runtime windows within minutes of wall-clock.

---

## 2. Scope and the live bytes

### 2.1 Live byte map

The container serves exactly these bytes (verified against both launchers' `--mount` lists and the Containerfile):

| Live path (container) | Source (host) |
|---|---|
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | `active-r634-b12x-compact/b12x_mla_sparse.py` (4702 lines) |
| `vllm/model_executor/layers/attention/mla_attention.py` | `active-r634-b12x-compact/mla_attention.py` (3835 lines, MLA-common layer) |
| `vllm/v1/attention/ops/kvarn_mla.py` | `active-r634-b12x-compact/kvarn_mla.py` (1579 lines) |
| `vllm/model_executor/layers/quantization/kvarn/config.py` | `active-r634-b12x-compact/config.py` |
| `vllm/v1/attention/ops/triton_kvarn_sinkhorn.py` | `active-r634-b12x-compact/triton_kvarn_sinkhorn.py` |
| `vllm/v1/core/kv_cache_utils.py` | `active-r634-b12x-compact/kv_cache_utils.py` |
| `sparkinfer/attention/kvarn_mla/api.py` | `active-r634-b12x-compact/kvarn_api_k4.py` (1504 lines) |
| `vllm/v1/attention/backends/mla/kvarn_mla_state.py` | `active-r634-kvarn-k4-native/kvarn_mla_state.py` (805 lines) |
| `sparkinfer/attention/kvarn_mla/io.py` | `active-r634-kvarn-k4-native/kvarn_mla/io.py` (445 lines) |
| `sparkinfer/attention/_shared/mla/kernel.py` | `active-r634-kvarn-k4-native/_shared/mla/kernel.py` (3737 lines) |
| `vllm/v1/worker/gpu/model_runner.py` | `active-r634-runner/model_runner.py` |
| `vllm/v1/attention/backends/mla/sparse_utils.py` | `runtime-vllm-extras/vllm/v1/attention/backends/mla/sparse_utils.py` |
| everything else (`kv_cache_interface.py`, `attn_utils.py`, `kvarn_decode.py`, `kvarn_store.py`, `quantization/kvarn/sinkhorn.py`, `block_table.py`, `cp_utils.py`, `single_type_kv_cache_manager.py`, speculator, decode_math/smem/traits of sparkinfer) | **fork tree** `/home/js/projects/optimize-vllm` (dirty working tree, HEAD `6ceef07ccb`), or base image |

Notes established during the sweep:
- `bake-context/kvarn_mla_fix2.py`, `kvarn_mla_state_fix2.py`, `b12x_mla_sparse_fixed.py` are **byte-identical** to the active files (md5-verified) — the baked "fix" variants are the running bytes; the `*dbg*` files are older debug builds.
- The fork's base `vllm/v1/attention/backends/mla/kvarn_mla_state.py` is a **superseded 464-line variant** — the live one is the 805-line active overlay (OwnResume file-identity fix; the predecessor scout's "24 kvarn hooks in mla_attention.py" were actually `b12x_mla_sparse.py` hits).
- The fork working tree is dirty (~30 modified vllm files). ForkResume verified the dirty hunks: none touch the KVarN/DCP index code in `vllm/v1/core/*`, `block_table.py`, or `cp_utils.py`; the dirty DCP-adjacent b12x hunks are prefetch gating + execution-lanes env only. Commit `6ceef07ccb` ("reconcile MLA physical cache ownership", 23 files) and `5518756cf0` ("Optimize KVarN MLA prefill and decode") are all ownership-safety fixes / materialize pool-bound OOB fix / K5 Triton RTN serialize.
- `shared-h-stack/kquant-bd0e95c` is an **EXL3 encoder recipe** (zero KVarN symbols — grep-verified); it is *not* a KVarN reference implementation. `shared-h-stack/sparkinfer-c40eb08f-full` is the pre-KVarN baseline: the entire KVarN decode is a pure additive overlay delta (delta list in NativeResume report).

### 2.2 Live deployment configuration (from launchers)

| Setting | Value | Source |
|---|---|---|
| KV dtype | `kvarn_mla_k4_g64` (4-bit latent, group 64) | `boot-from-image.sh:188-193` (default; K5 only when `SPEC_METHOD==extract_hidden_states`) |
| TP / DCP | `--tensor-parallel-size 4`, `--decode-context-parallel-size 4`, `--dcp-comm-backend a2a` | `:698-700` |
| Spec decode | MTP, `MTP=3` (3 draft tokens) | `:26` |
| Cache | `NUM_GPU_BLOCKS=2000`, `MAX_NUM_SEQS=1` | `start-full-expert-...sh:12,16` |
| Native CuTe K4 reader | `VLLM_KVARN_MLA_NATIVE_CUTE=1` | `:157` (overrides the in-code default 0 — **the live decode reader is the native CuTe grid op**) |
| Direct packed | `KVARN_DIRECT_PACKED=1` | `:156` |
| Precision tail | `KVARN_MLA_PRECISION_TAIL_TOKENS=3072` (FP8 pool keeps sink 128 + last 3072 tokens; packed rows only beyond ~3.2k) | `:155` |
| Native CKV gather | `VLLM_KVARN_MLA_NATIVE_CKV_GATHER=0` (Triton compact wire staging is live instead) | `:27` |
| CKV gather / prefetch | `VLLM_B12X_MLA_CKV_GATHER=1`, `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1`, min/max tokens 1024/65540 | `:86-89, 663-669` |
| Fused current stage | `VLLM_KVARN_MLA_FUSED_CURRENT_STAGE=0` | `:28` |
| DCP topk | `VLLM_DCP_TOPK_OWNER_MERGE=1`, `VLLM_DCP_GLOBAL_TOPK=1`, `VLLM_DCP_SHARD_DRAFT=1`, `VLLM_DCP_PROJECT_BEFORE_MERGE=1` (min prefill 1024), `VLLM_DCP_QUERY_SPLIT=1` (min context 8192), `VLLM_DCP_A2A_MAX_TOKENS=4`, `VLLM_B12X_DCP_LSE_REDUCE=0`, `VLLM_USE_B12X_DCP_A2A=1` | `:83-84, 637-673` |
| Determinism knobs | `VLLM_KVARN_DETERMINISTIC_DCP_COMPACT=0`, `VLLM_KVARN_DETERMINISTIC_SINKHORN=0`, `SPARKINFER_KVARN_MLA_M9_HPP4_MERGE=0` | `:97, 105-106` |
| Top-k | 2048 rows | `kvarn_api_k4.py` / kernel constants |
| Sinkhorn iters | 8 (env `KVARN_SINKHORN_ITERS` default; reference default is 16) | `config.py:~745` |

**Consequences of this configuration** (used throughout the report):
- Live K4 decode reader = `kvarn_mla_sm120_decode_grid` (native CuTe, `packed_bits=4`, `chunks=32`, `num_splits=32`, Triton split-merge). The FP8-record staging reader and the K5-only Triton split kernel are **not live** (staging is fallback-only; the Triton path fail-closes on K4 geometry).
- Contexts **< ~3.3k tokens are served entirely from the FP8 exact pool** (sink 128 + tail 3072 + current 64). Packed K4 rows are read only for longer contexts — and, in that regime, from **generated token 1**.
- Sparse top-k selection only engages when `seq_len > 2048`; below that, all rows are selected (identity map), so any selection-domain bug is invisible.

---

## 3. Architecture as verified (the data flow)

```
per-step (CPU):  resolve_pending (async MTP acceptance, CPU-blocked on copy_event)
                 -> finish/remove/add/update requests (block tables, CoW, zero-new-blocks)
                 -> _update_kvarn_mla_ownership (tracker.update; publishes block->fill snapshot)
per-step (GPU, one stream):
                 metadata builder -> KVarNMLAStateManager.prepare_step
                     = fill snapshot -> retire (retired = mapped \ needed)
                       -> flush (retired & fill>=64): pool tile -> sinkhorn -> K4 RTN -> 26752B record -> paged cache
                       -> free slots (fill<64: discard, no packed copy)
                       -> allocate missing slots (LIFO; fail-fast on pool exhaustion)
                       -> rehydrate (missing ∩ flushed: paged record -> pool, prefix hits)
                       -> ONE batched mirror update (block_to_slot; -1 = packed)
per-layer forward:
                 store:  latent --Hadacore 512-D rotation--> scatter_kvarn_mla_exact -> FP8 latent pool + BF16 rope pool
                         (tokens whose block maps to a valid pool slot; invalid mapping => row SILENTLY DROPPED)
                 indexer: per-rank top-2048 (global logical token ids under DCP) -> A2A exchange -> owner-merge
                 convert: global -> local physical slots (exact inverse of write-side slot mapping) -> compact-to-front
                 decode:  kvarn_mla_sm120_decode_grid
                     per selected row: block_to_slot >= 0 -> FP8 pool row | == -1 -> packed K4 tile, in-kernel dequant
                     Q: unrotated in, in-kernel fused Hadamard (fuse_kvarn_hadamard=True), output inverse-rotated
                 DCP combine: LSE-weighted cross-rank a2a merge
```

### 3.1 Layout contract (verified identical across all tables)

K4/G64 paged tile = **26752 B**: `[0,16384)` packed 4-bit latent (dim-major, code for token *t*, dim *d* at bit `(d·64+t)·4`, low nibble = even token) · `[16384,17408)` **s_col-field** = 512×fp16, PER-DIM · `[17408,18432)` **zp-field** = 512×fp16, PER-DIM · `[18432,18560)` **s_row-field** = 64×fp16, PER-TOKEN · `[18560,26752)` rope 64×64 BF16 (exact copy).

**Naming trap (verified consistent everywhere, but documented for future edits):** the record *field* names are swapped relative to the sinkhorn *variable* names — `s_col-field` = sinkhorn `s_row` × RTN scale (per-dim), `s_row-field` = sinkhorn `s_col` (per-token). Every reader implements the same identity:

```
value[d,t] = (code[d,t] · s_col_field[d] + zp_field[d]) · s_row_field[t]
```

Pool slot = **40960 B** = 64 × (512 × fp8_e4m3fn latent + 64 × bf16 rope). The "exact" pool is FP8 for the latent by design (config docstring); exact only applies to RoPE.

Verified at: `config.py:604-640/658-666`, `kvarn_api_k4.py:9-28, 957-961`, `_shared/mla/kernel.py:148-151`, fork `kvarn_store.py:204-232`, fork `kvarn_decode.py:118-148` (three independent cross-checks — parent, StoreResume, NativeResume, B12XResume).

### 3.2 Write path

`b12x_mla_sparse.py:2930-2984` (`do_kv_cache_update`): latent rotated with `kvarn_hadamard` (fork `kvarn_decode.py:27-49` — CUDA `hadacore_transform`, tensor-core MMA composition, full 512-wide orthonormal FHT, involution test-proven) → `scatter_kvarn_mla_exact` (`kvarn_mla.py:32-90`) into per-rank pools. Slot mapping write side: fork `block_table.py:301-346` (`_compute_slot_mappings_kernel`), DCP branch: `is_local=(off//I)%W==rank`, `local_offset=(off//(I·W))·I+off%I`, `slot=phys·64+local_offset`.

Flush: `pack_kvarn_mla_blocks` (`kvarn_mla.py:322-381`): pool-gather → `kvarn_sinkhorn_triton` → **K4: fork `kvarn_store_tile_k_batch_from_sinkhorn`** (`kvarn_store.py:224-260`; per-dim min-max RTN over the 64 tokens, banker's rounding, per-dim `scale=(hi-lo)/15 clamp_min 1e-10`, `zp=lo`, scale/zp absorbed with sinkhorn scales: `s_col_K = s_row_chan·scale` [per-dim], `zp_K = s_row_chan·zp` [per-dim], `s_row_K = s_col` [per-token]; affine refit **default OFF for K4**, ON for K5-only Triton path) → record assembly → paged `index_copy_` (same stream).

---

## 4. The symptom and what the evidence actually shows

From `tests/goldens/` (kvarn corrupt run vs nvfp4 controls) and the user report (garbled output, sporadic):

| Case | Corrupt run `kvarn_tr32_corrupt_run1.json` | Control `nvfp4_control_run1.json` |
|---|---|---|
| `captured_cli` (real tool-bearing request) | **`output_text = " - 0, - - - 0. - - - 0..."` — tight 2-token loop, tokens `[220,481,481,481,220,...]`, from generated token 1.** Rescore mean −0.909 (worst), frac_below_m3 0.084, jaccard vs control 0.004 | `"Hey! What can I do for you in the superqwen project?"` — clean |
| `sky_blue` (224-token prompt) | 201 tokens; LCP 37 vs control golden; rescore mean −0.223, frac_m3 0.005 — **passes all gates** | clean |
| `multiturn`, `tool_math`, `haiku` | rescores pass all gates (means −0.23…−0.39); divergences from control golden of LCP 27–65 | clean |

**Interpretation (important for scoping the fix):**
- The *hard* corruption is `captured_cli`: a tight repetition loop is the signature of attention collapsing onto a small set of rows — consistent with a **constant foreign-KV block** in the attended set (plausible content, wrong owner), or a constant mis-mapped row. It is *not* the signature of random garbage or NaNs.
- `captured_cli`'s prompt (system + tools + history) is long-context; the loop starts at **token 1** → the first decode step already consumed mis-sourced rows.
- The sub-2048-token cases in the same golden **pass every rescore gate**; their divergence from the NVFP4 golden is at or below the level expected from KV-format differences (this run's KV is FP8-pool/4-bit, the control's NVFP4). The "30–60 plausible tokens then diverge" reading of `sky_blue` is therefore *not established as corruption* — treat it as a secondary signal only.
- Rescore is a fresh dense prefill of prompt+continuation; it is clean by construction of the test and cannot detect decode-path row errors — the test gates are necessary but not sufficient, and the "corrupt" run only fails on `captured_cli`.

**Failure class implied:** silent (no crash, no log — the kernel is fail-soft by design), row-selective (only rows of affected blocks), decode/sparse-read-specific, history/batch-dependent (sporadic), long-context-accented (packed rows + sparse selection only exist past the 2048/3264 boundaries).

---

## 5. Audit matrix — subsystem verdicts

| # | Subsystem | Verdict | Primary evidence |
|---|---|---|---|
| 1 | **Quant/dequant identity, all hops** | ✅ verified symmetric — no formula/offset/bit-order/domain/rounding asymmetry on the live K4 path | StoreResume (full report §A/§B); parent manual trace; B12XResume third cross-check. Covers: fork K4 store op, Triton K5 serialize (K5-only), materialize, rehydrate, FP8 staging kernel, DCP compact wire, native CuTe producer, fork reference dequant |
| 2 | **Sinkhorn** (Triton variants vs PyTorch reference) | ✅ consistent; deltas = FP reduction order (~1e-7, cannot break symmetry — readers never re-run sinkhorn) + explicit 8-vs-16 iteration default | StoreResume §D; parent trace of `_sinkhorn_log_kernel` vs `_sinkhorn_tiled` (identical update order, clamps, best-so-far tracking) |
| 3 | **Hadamard rotation contract** | ✅ one 512-D orthonormal transform on both sides. K side: Hadacore (MMA-FHT, intermediate bf16 sub-rounding). Q side (live native path, `fuse_kvarn_hadamard=True`): in-kernel fp32 butterfly FHT, single bf16 round at store, `inv_sqrt512=0.04419417382415922`. Delta ≈ 1–2 bf16 ulps/element (~1e-3 rel), **uniform across rows → quality-class, cannot produce row-selective garbling**. RoPE 64-D never rotated anywhere | LayerResume (both citations); StoreResume addendum (quantified the delta); ForkResume (transform contract, `test_hadacore` involution) |
| 4 | **DCP(4,1) physical-index chain** (scheduler allocation → block table → write mapping → read de-interleave → tracker) | ✅ verified consistent **in the dirty tree**; write side and read side are exact inverses; 1:1 manager-block↔tile (256-token manager blocks, `blocks_per_kv_block=1`, `blocks_per_manager_block=1`); 64-token tile never split across ranks; owner-merge uses the same owner formula; no dirty hunks in core files | ForkResume (P0-1 verification); LayerResume (indexer pack vs backend de-interleave algebra) |
| 5 | **Ownership state machine** (`kvarn_mla_state.py` + runner wiring) | ✅ **correct in every audited window** — invariants I1–I6 (below) | OwnResume (final report, line-level) |
| 6 | **b12x backend decode staging / graph replay** | ✅ index domain equality (global→local conversion = exact inverse of store mapping; `block_to_pool_slot.shape[0]=2000` = kernel `num_blocks`); selected-index buffers fully rewritten every graph replay (whole `[rows,2048]` re-filled −1 + counts re-zeroed + prefix re-scattered; padded rows `fill_(-1)` inside the graph); mirror updates run EAGERLY in the metadata builder, never captured; workspace sizing fail-closed; LSE base e end-to-end (b12x / kernel s7 / Triton merge / mla_attention) | B12XResume (final report) |
| 7 | **Native CuTe K4 reader** (`io.py` + `kernel.py` + traits/smem/decode_math) | ✅ self-consistent; fail-soft on out-of-domain indices (zero-fill + finite −1e30 mask + neutral epilogue); stride question **refuted** (656B GLM-h8 override unreachable from the KVarN launch; live uses `kv_smem_stride=1040` writer=consumer); single-buffer lockstep requires `kv_bufs==1` — verified in KVarN-lineage smem (live image smem not in repo; high confidence + no-deadlock evidence) | NativeResume (final report) |
| 8 | **Scheduler byte accounting** | ✅ `real_page_size_bytes = config.tile_bytes = 26752` with `block_size==group(64)` assert; solve `N·26752·L + envelope(N)` == worker tensors; `block_size=64` enforced in 3 sites. Gap (P1, capacity-only, cannot garble): per-layer exact pools excluded from the solve | LayerResume §(4) |
| 9 | **CKV side-stream prefetch** | ✅ prefill-only eligibility (`num_decode_tokens==0` required, b12x:3550-3581) + event-synced in code (`wait_stream` before side-stream gather, consumer `event.wait`, fallback `wait_for_pending_writes`) | StoreResume addendum; B12XResume (independent) |
| 10 | **Sylvester current-row staging** (`stage_bf16_sylvester_as_exact_pool_fp8_records`) | ✅ dead in live config (`FUSED_CURRENT_STAGE=0`; and inside the CKV-gather call chain, itself prefill-only). Where live, input is RAW unrotated latent — single rotation, same domain | StoreResume addendum |
| 11 | **Fork dirty-tree delta + history** | ✅ dirty hunks don't touch KVarN/DCP index code; `6ceef07ccb`/`5518756cf0` are ownership-safety fixes; mla.py + deepseek_v2.py verified KVarN-free (write hook = b12x `do_kv_cache_update` via MLAAttention, MTP draft shares identical hook) | ForkResume |

### 5.1 OwnResume invariants (I1–I6) — the ownership proof

- **I1 single-stream ordering.** No side-stream KV ops anywhere in the KVarN path (the only side stream is the acceptance-count D2H, CPU-pinned, `copy_event.synchronize()` before the next ownership update). Order: `[prev attention] → [zero/CoW] → [prepare_step: pack → rehydrate → mirror] → per-layer [scatter → attention]`. A retired-then-flushed block is fully packed before its mirror goes −1 and before any same-step reader; a freed slot is re-read pooled only after the new owner's scatter/rehydrate.
- **I2 conservative-fill bound.** Every published fill ≤ rows valid after that step. Touched-block fill = `min(64, max(local_min_end − 64k, 0))`; resolved fill from `actual_end = start + scheduled − num_rejected` (exact accepted count). **Fill=64 proof:** published fill = max(conservative, resolved) ≤ true accepted count; a rejected (never-final) draft token can never raise a fill to 64, its stale pool row is overwritten by the next step's scatter before any read, and the later retire-flush therefore always packs 64 valid rows.
- **I3 same-step physical block reuse.** A block in `needed` keeps its slot; the new owner's scatter overwrites its read range in the same step before any reader. Full-block prefix hit ⇒ original owner retired it full ⇒ flushed ⇒ in `state.flushed` ⇒ rehydrated on re-entry; partial blocks are never prefix-cached. CoW: new-request partial-hit = in-place table swap visible to the tracker (full copy at add); running-request CoW = append-only.
- **I4 target/draft (MTP) isolation.** `group_key = tuple(layer names)`; target and draft groups have disjoint trackers, `_GroupState`s, pools, mirrors, physical block namespaces. The draft's `prepare_step` (after the target forward) cannot touch a block the target reads; a duplicate `prepare_step` on a shared key is idempotent; draft reads of target KV happen only after all target scatters were enqueued (same stream) → fresh.
- **I5 mirror window.** `retired→−1` and `new→slot` share one `mirror_updates` dict applied by **one** `_update_mirror` per device (single `index_put`); no reader exists between them or before scatter; `ensure_mirror` syncs from `state.mapping` on first creation, before the first scatter.
- **I6 resolution lifecycle.** Pending epochs created on every MTP step; the runner guard **raises if any scheduled tracker_key still has an unresolved epoch** (fail-closed, runner 1481-1486). Resolved fills are consumed at the next `update()` even for same-step-removed owners (max one-step slot retention for a dead block — safe per I1/I3; no flush can fire on stale rows). Rollback free-without-flush (fill<64) is safe: no reader reads past fill; a partial block can never be a prefix-hit target.

---

## 6. Refuted hypotheses (the original seven + late arrivals)

| # | Hypothesis | Fate | Decisive evidence |
|---|---|---|---|
| 1 | GLM-h8 staged-KV stride mismatch (656B consumer vs 1024B writer) in the native grid kernel | **Refuted** | The 656B `staged_kv_stride` override only fires when `native_glm_h8=True`; `_kvarn_mla_decode_grid_flat_launch` (kernel.py:2953-3008) never sets it. Live K4 uses the `KVARN_K5` traits entry with `kv_smem_stride=1040` writer=consumer (traits.py:163-171). The 656B layout belongs to the non-KVarN ARBITRARY_FP32 path. (NativeResume) |
| 2 | K4 quantizer / zp-domain asymmetry (store vs any reader) | **Refuted** | All readers implement the exact stored identity; field offsets byte-identical across 5 tables; packing dim-major LSB-first on both sides; banker's rounding matches torch store op and Triton; affine-refit state irrelevant to readers (they use stored fields only). (StoreResume + parent) |
| 3 | Hadamard rotation domain mismatch (store rotated / read unrotated or double-rotated) | **Refuted** | Same unique 512-D orthonormal transform both sides; residual rounding delta ~1e-3, uniform (quality). (LayerResume + StoreResume + ForkResume) |
| 4 | DCP global→local interleave / per-rank block-table chain misrouting rows | **Refuted (statically)** | Exact-inverse chain verified end-to-end in the dirty tree, 1:1 allocation, no cross-rank tile split. (ForkResume) — *residual*: runtime A2A metadata sync, see §7 |
| 5 | `block_to_slot == −1` for a live block ⇒ packed read of uninitialized/foreign record (mirror publication staleness, graph-replay republish, A2A sync, draft metadata) | **Refuted (statically); runtime windows remain** | All −1-at-scatter paths verified prevented (I1–I6, §5.1); graph replay rewrites all index buffers and runs the mirror update eagerly; fail-soft kernel verified. (OwnResume + B12XResume + NativeResume) — *residual*: §7 items 2–4 |
| 6 | MTP(3) rollback freeing below-full-fill blocks without packed copy, read by draft/next step | **Refuted** | I2 fill=64 proof + I4 isolation + "no reader reads past fill" + partial blocks never prefix-cached. (OwnResume) |
| 7 | FP8 pool overflow (|rotated latent| > 448 → inf → NaN) | **Effectively refuted** | Staging path scales by `amax/448` (overflow impossible); pool scatter saturates (consistent loss, not asymmetry); post-rotation latents O(1)–O(10) by design. (StoreResume P2-2) |
| 8 | CKV side-stream prefetch racing flush/rehydrate | **Refuted** | Prefill-only eligibility + event sync in code. (StoreResume + B12XResume) |
| 9 | Sylvester current-row staging double-rotation / wrong domain | **Refuted (dead + same-domain)** | Gated off live; input is raw unrotated; single 512-Sylvester, verified axes. (StoreResume addendum) |
| 10 | Sinkhorn variant divergence (in-register vs tiled vs g64_512) | **Refuted** | All variants implement the reference exactly; `_sinkhorn_tiled` (the live one for 512×64) scores the same fully-updated candidates. (StoreResume §D + parent) |

---

## 7. Surviving findings — ranked

Nothing below is a *proven* live bug; all are the residue after full static verification. Ranked by (fit to the sporadic garbling signature) × (reachability in the live config).

### F1 — The onset-boundary data question (decisive, not a code location)
Everything static passes, so the single fact that separates the remaining theories: does garbling onset track **seq_len > 2048** (top-k selection first becomes sparse — points at the selection/compaction domain: `sparse_attn_indexer.py` pack/owner-merge, `sparse_utils.py` atomic compact-to-front under `VLLM_KVARN_DETERMINISTIC_DCP_COMPACT=0`) or **~3264 tokens** (first packed row exists — points at the packed-content/ownership domain: flush/record/mirror)? Both domains are statically clean, so the observation itself is the next real signal. Note B12XResume's P1-1: b12x validates only block-table bounds on the converted indices, **not** topk context-length validity (fail-open).

### F2 — `mark_request_unknown` is DEAD in the live runner (defensive gap; would make any future divergence silent)
`kvarn_mla_state.py:316-326` defines the stale-fill guard; the superseded build's runner called it (`build-exact-h16-v1/vllm/v1/worker/gpu/model_runner.py:1556`); the live `_remove_request` (`active-r634-runner/model_runner.py:1557-1563`) does not. Every *current* path is verified to prevent a `block_to_slot == −1` at scatter — but the scatter kernel drops such rows **silently** (`kvarn_mla.py:62-68`), and the decode kernel fail-opens the other way. Any future scheduler/runner/speculator divergence in this class would manifest as exactly the observed symptom: row-selective garbling, no error, no log. (OwnResume P1-2)

### F3 — Runtime-only windows static analysis cannot close
1. **DCP4 A2A metadata sync** — `dcp_rank_lengths`/`padded_rank_tokens` (b12x:1456-1467, 4203-4206). Formulas verified; the wire synchronization itself is not statically checkable. (ForkResume/B12XResume)
2. **Mirror-pointer staleness under graph replay on cache resize** — `ensure_mirror` (`kvarn_mla_state.py:573-585`) recreates the mirror tensor if the block count grows mid-run; graph-captured scatter/decode kernels keep the **old** mirror storage (pointers captured at capture time; `impl._kvarn_block_to_slot` reassigned at b12x:2695-2699). Inert with fixed `NUM_GPU_BLOCKS=2000`; requires a runtime cache resize without recapture. (OwnResume P2-2)
3. **Non-deterministic DCP compact** (`DETERMINISTIC_DCP_COMPACT=0`) — multi-tile 128-wide atomic compaction with "prefix order unspecified (only the set matters)". Verified safe for set/slot-semantics consumers; the verification rests on the consumer audit, so treat as confirmed-with-dependency. (ForkResume, P2)
4. **Live-image smem.py not in repo** — single-buffer lockstep (`kv_bufs==1`) verified in the KVarN-lineage smem source with high confidence, but the exact baked image file couldn't be opened. No-deadlock operation is corroborating evidence. (B12XResume)

### F4 — Fail-open design surface (why this bug class is invisible)
Invalid selected index → zero-fill + mask (`io.py` meta pass; S3 mask −1e30, decode_math); `block_to_slot == −1` → packed read (no liveness check); scatter row-drop on invalid mapping (`kvarn_mla.py:62-68`). Each individually safe, collectively they convert *any* mapping/ownership error into **plausible** dropped/foreign rows with zero diagnostics. This is the single most important design property for the fix effort: the system cannot tell you the bug is happening. (B12XResume P1-2; OwnResume P1-2)

### F5 — Latent traps (unreachable in the live config; document or gate)
1. **`exact_pool_only` rope-zero trap** — `io_issue_kvarn_k5_gather` folded-rope fast path zero-fills rope for non-exact rows when `exact_fp8 ∧ exact_fast_io` (io.py ~350-365). Unreachable today: live decode hardcodes `exact_pool_only=False` (b12x:4372) and `SPARKINFER_MLA_SM120_KVARN_EXACT_FAST_IO` is off. **Trap:** the eager purity validation runs only when *not* graph-capturing (kvarn_api_k4.py:1407-1411); a graph replay of a row that became packed would silently lose its rope term (row-selective). (StoreResume bonus)
2. **`kvarn_sinkhorn_g64_512` scratch hazard** — writes 1729 scratch words into caller-aliased RoPE region with no guard; safe as deployed (zero live callers; the sole historical caller serializes the rope record first on the same stream and passes only retired slots). Caller-invariant hazard. (StoreResume P2-5)
3. **K5-only Triton split kernel** hardcodes 5-bit unpack (`kvarn_mla.py:1091`); `direct_packed_kvarn_mla_decode` fail-closes on K4 geometry (1393-1399) and delegates to `native_decode` **before** the gate (1352-1369) — would raise, not corrupt. (OwnResume P2-3; NativeResume)
4. **K5 `AFFINE_REFIT` default asymmetry** — K4 store default OFF (fork `kvarn_store.py:108`), K5 Triton default ON (`kvarn_mla.py:317`). No correctness impact (readers use stored fields only); config surprise if a reference run used refit=1. (StoreResume P2-3)
5. **`num_chunks_ptr` device-checked but never passed to the CuTe op** (`kvarn_api_k4.py:1398-1457`). (NativeResume P2)
6. **Per-layer exact pools excluded from the `num_blocks` solve** (`L·pool_size·40960` B, b12x:2152-2167) — capacity under-reservation only, cannot garble. (LayerResume P1)
7. **MTP draft zero-pool packing** — draft layers can pack their own never-written (zero) pools into their own pages; draft quality only, target output unaffected (I4). (LayerResume P2-1)
8. **Rehydrate double quantization** — prefix-hit rows restored as `fp8(dequant(K4(fp8(x))))` vs original `fp8(x)` (`kvarn_mla.py:650`); quantization-noise level, prefix-hit rows only. (StoreResume P2-1)

---

## 8. Recommendations

### 8.1 To localize the bug (runtime; run when permitted)
1. **`KVARN_MLA_DIAG_EXACT_ROWS`** (built-in probe, ForkResume-identified): dump per-row sourcing (pool slot / packed / dropped) for a garbled run. Foreign-slot reads name the block, step, and rank immediately.
2. **Log `seq_len` at first-garbled token** across ≥5 garbled runs (and 5 clean runs). Onset at **2048** ⇒ instrument the selection/compaction domain (`sparse_utils.py` compact-to-front, owner-merge); onset at **~3264** ⇒ instrument the packed/ownership domain (flush records, mirror contents).
3. **One-line scatter assert** (converts F4's fail-open into a crash at the exact corrupting step): in `_scatter_kvarn_mla_exact_kernel`, raise/log when a token with a valid `slot_mapping` has `block_to_slot == −1`.
4. **Mirror-identity check under graph replay**: log `data_ptr(impl._kvarn_block_to_slot)` at capture and at N replays (closes F3.2).
5. **Record-level spot check** (cheap, no model run): for one flushed block, recompute the record from the pool with the reference identity (fork `kvarn_decode.py:91-118`) and byte-compare — closes F3.4 and any image-level divergence in one shot.

### 8.2 Hardening (small, low-risk, in scope of the live overlay)
1. **Revive `mark_request_unknown`** in the live `_remove_request` (or assert its precondition) — closes F2.
2. **Make the scatter drop loud**: even a rate-limited GPU→CPU counter + warning per step turns the F4 surface into a detectable event.
3. **Gate or document the F5 traps**: `exact_pool_only` graph-replay validation gap (run the purity check on first replay too), `kvarn_sinkhorn_g64_512` slot-range guard, K4/K5 `AFFINE_REFIT` default alignment.
4. **Include exact pools in the `num_blocks` solve** (F5.6) or assert pool bytes fit in the profiling headroom.

### 8.3 Process
- The "corrupt" golden's non-`captured_cli` cases pass all gates; add a **token-loop detector** to `test_generation_coherence` (n-gram repetition over the first 64 generated tokens) — the current gates (mean logprob / frac-below-m3) provably do not catch this failure mode on a per-case basis.
- Keep the dirty-tree discipline: three of the audited "bugs" (the 464-line superseded state file, the dead guard, the old build's speculator) are artifacts of **multiple coexisting generations of the same code** in this tree. Pin the live byte-set per file (the launchers' mount lists are the source of truth) before any further KVarN work.

---

## Appendix A — Live decode path, per rank (DCP4), fully cited

Per-step CPU: `_resolve_pending_kvarn_mla_output` (runner 2091-2092 → 1402-1408; async `resolve_completion` CPU-blocks on `copy_event`, fork async_utils 103-113; callback runs `tracker.resolve_async`, runner 1539-1552) → `update_batch` (finish/remove 1557-1563, add 1630-1645, update 1672-1695, `apply_staged_writes` 2100) → `_update_kvarn_mla_ownership` (2101 → 1449-1509: `scheduled_T = num_scheduled + lookahead` 1471, `rollback_T = prev_drafts + cur_drafts + lookahead` 1472-1476, **pending guard raise** 1481-1486, per-group `tracker.update` 1487-1495, publish + `speculator.set_kvarn_mla_block_fills` 1505-1509) → metadata build (per-group fills slice, fork attn_utils 723-756).

Per-step GPU (one stream): `prepare_step` (b12x 1373-1379; state 613-735: fill snapshot 644-652, retired 654, flush_ids 655-659 → `pack_kvarn_mla_blocks` 669 → b12x 2726-2733 → kvarn_mla 322-381; flushed 672; free 674-681 (sub-full discard 677-681); alloc 683-694 (exhaustion raise 684-689); rehydrate 704-732 → kvarn_mla 585-702; **one batched mirror update** 734-735 → 601-610) → per layer: `do_kv_cache_update` (2930-2984: `ensure_mirror` 2954 → 2684-2716, Hadacore 2975, `scatter_kvarn_mla_exact` 2976-2983 → kvarn_mla 32-90, silent drop 62-68) → indexer top-2048 (global ids under DCP) → A2A owner-merge → `triton_filter_and_convert_dcp_index` (sparse_utils 185-280, 424-537) → pad rows (b12x 4269-4324) → `direct_packed_kvarn_mla_decode` (4316-4378; K5-Triton gate fail-closed 1393-1399; native delegation 1352-1369) → `native_packed_k5_decode` (kvarn_api_k4 1285-1360: `packed_bits_by_tile {18560:2, 26752:4, 30848:5}`, `num_splits=32`) → `kvarn_mla_sm120_decode_grid` (kernel 2953-3050; producer `io_issue_kvarn_k5_gather`, io.py 87-445; S0–S7, fused Hadamard in/out) → Triton split-merge (kvarn_api_k4 1180-1284, base-2→natural LSE) → DCP a2a LSE-weighted combine → output (unrotated).

Post-forward: async output copy + resolution callback stash (runner 2595-2615, overwrite guard 2611-2614). Draft model: own metadata build → own `prepare_step` (disjoint `_GroupState`; fork speculator 339).

## Appendix B — Audit coverage map

| Auditor | Scope | Verdicts delivered |
|---|---|---|
| Parent (manual) | `config.py` layout; `kvarn_mla.py` scatter/serialize/materialize/rehydrate/staging/K5-split; `triton_kvarn_sinkhorn.py` (all 3 variants); `kvarn_mla_state.py` state machine; `model_runner.py` ownership driver; launchers | Layout contract; field-swap trap; sinkhorn equivalence; conservative-fill math; live env gates |
| NativeResume | `kvarn_api_k4.py`, `kvarn_mla/io.py`, `_shared/mla/kernel.py` + base-image traits/smem/decode_math + reference stack | Stride refutation; in-kernel self-consistency; fail-soft semantics; reference delta (KVarN = pure additive overlay) |
| StoreResume | `kvarn_mla.py` (all), `triton_kvarn_sinkhorn.py` (all), fork `kvarn_store.py`/`kvarn_decode.py`/`sinkhorn.py`, `kvarn_api_k4.py` | Store/read identity (all hops); sinkhorn-vs-reference; staging gate; Q-vs-K rounding delta; prefetch closure; F5 traps |
| OwnResume | `kvarn_mla_state.py` (all), runner KVarN regions, b12x hooks, fork `attn_utils`/`single_type_kv_cache_manager`/CoW utils | Invariants I1–I6; timelines (prefill/decode/spec-accept); F2 dead guard; F3.2 mirror resize |
| LayerResume | `mla_attention.py`, `sparse_attn_indexer.py`, `kv_cache_utils.py`, fork `kv_cache_interface.py` | Rotation consistency; flush sync; scheduler bytes; DCP interleave (in-file); byte-accounting gap |
| ForkResume | Fork dirty tree (full diff), un-overridden fork files, reference stacks, launchers | DCP chain exonerated; per-hunk verdicts; live launcher pinning; non-deterministic-compact verification |
| B12XResume | `b12x_mla_sparse.py` (all), live `sparse_utils.py`, `kvarn_mla.py` ops, `kvarn_api_k4.py` | Reader identification; index-domain equality; replay freshness; workspace fail-closed; LSE end-to-end; F1 boundary question |

First batch (B12XDecode, StorePath, Ownership, NativeK4, LayerRunner, ForkRef) was killed mid-run by a session interrupt; all six transcripts were intact on disk and the successors resumed from them (no rework; the killed batch had completed ~80–95% of reading).

## Appendix C — Reproduction/verification pointers

- Garbling evidence: `tests/goldens/kvarn_tr32_corrupt_run1.json` (case `captured_cli`) vs `nvfp4_control_run1.json`.
- Coherence test + gates: `tests/test_generation_coherence.py` (mean_cont_logprob ≥ −1.0, frac_below_m3 ≤ 0.10, golden-similarity soft check) — see §8.3 for the detector gap.
- Round-trip test: `tests/test_kvarn_record_roundtrip.py` (torch reference vs GPU pack; K4 tile constants 26752/16384/17408/18432/18560).
- Live env: §2.2 table; the two authoritative launchers are `boot-from-image.sh` and `start-full-expert-335-512k-c2-b2048-g4.sh` (`boot-tr3-k4v2.sh` is an earlier generation with identical KVarN gates).