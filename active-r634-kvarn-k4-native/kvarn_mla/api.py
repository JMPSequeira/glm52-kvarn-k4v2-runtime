from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_K5_GROUP = 64
_K5_LATENT_DIM = 512
_K5_ROPE_DIM = 64
_K5_BITS = 5
_K5_TILE_BYTES = 30_848
_K5_S_COL_OFFSET = 20_480
_K5_ZP_OFFSET = 21_504
_K5_S_ROW_OFFSET = 22_528
_K5_ROPE_OFFSET = 22_656
_GLM_FP8_RECORD_BYTES = 656
_GLM_FP8_SCALE_OFFSET = 512
_GLM_FP8_ROPE_OFFSET = 528
_FP8_E4M3_MAX = 448.0
_TL_K5_GROUP = tl.constexpr(_K5_GROUP)
_TL_K5_ROPE_DIM = tl.constexpr(_K5_ROPE_DIM)
_TL_K5_BITS = tl.constexpr(_K5_BITS)
_TL_K5_S_COL_OFFSET = tl.constexpr(_K5_S_COL_OFFSET)
_TL_K5_ZP_OFFSET = tl.constexpr(_K5_ZP_OFFSET)
_TL_K5_S_ROW_OFFSET = tl.constexpr(_K5_S_ROW_OFFSET)
_TL_K5_ROPE_OFFSET = tl.constexpr(_K5_ROPE_OFFSET)
_TL_GLM_FP8_SCALE_OFFSET = tl.constexpr(_GLM_FP8_SCALE_OFFSET)
_TL_GLM_FP8_ROPE_OFFSET = tl.constexpr(_GLM_FP8_ROPE_OFFSET)
_TL_FP8_E4M3_MAX = tl.constexpr(_FP8_E4M3_MAX)
_M4_CHUNKS_PER_SPLIT_ENV = "SPARKINFER_KVARN_MLA_M4_CHUNKS_PER_SPLIT"
_EXACT_H16_ENV = "SPARKINFER_KVARN_MLA_EXACT_H16"

_M5_NATIVE_ENV = "SPARKINFER_KVARN_MLA_NATIVE_M5"


def _native_m5_split_family_enabled() -> bool:
    raw = os.environ.get(_M5_NATIVE_ENV)
    return raw is not None and raw.strip().lower() in {"1", "true", "on", "yes"}



def _mixed_m4_chunks_per_split() -> int:
    raw = os.environ.get(_M4_CHUNKS_PER_SPLIT_ENV)
    if raw is None:
        return 3
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_M4_CHUNKS_PER_SPLIT_ENV} must be an integer in [1,4]"
        ) from exc
    if not 1 <= value <= 4:
        raise ValueError(
            f"{_M4_CHUNKS_PER_SPLIT_ENV} must be in [1,4], got {value}"
        )
    return value


def _exact_h16_enabled() -> bool:
    raw = os.environ.get(_EXACT_H16_ENV)
    return raw is not None and raw.strip().lower() in {"1", "true", "on", "yes"}




@triton.jit
def _unpack_k5(payload, value_indices, mask):
    bit_positions = value_indices * _TL_K5_BITS
    byte_offsets = bit_positions // 8
    shifts = bit_positions % 8
    low = tl.load(payload + byte_offsets, mask=mask, other=0).to(tl.uint32)
    high = tl.load(payload + byte_offsets + 1, mask=mask, other=0).to(tl.uint32)
    return ((low | (high << 8)) >> shifts) & 31


@triton.jit
def _stage_k5_as_fp8_records_kernel(
    physical_slots_ptr,
    k5_cache_ptr,
    block_to_pool_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    output_ptr,
    cache_stride_block: tl.constexpr,
    latent_pool_stride_slot: tl.constexpr,
    latent_pool_stride_token: tl.constexpr,
    rope_pool_stride_slot: tl.constexpr,
    rope_pool_stride_token: tl.constexpr,
    output_stride_row: tl.constexpr,
    num_blocks,
    num_pool_slots,
):
    row = tl.program_id(0)
    physical_slot = tl.load(physical_slots_ptr + row)
    block = physical_slot // _TL_K5_GROUP
    token = physical_slot % _TL_K5_GROUP
    valid = (physical_slot >= 0) & (block >= 0) & (block < num_blocks)
    safe_block = tl.where(valid, block, 0)
    pool_slot = tl.load(
        block_to_pool_slot_ptr + safe_block,
        mask=valid,
        other=-1,
    )
    valid &= pool_slot < num_pool_slots

    if valid:
        exact = pool_slot >= 0
        body = ~exact
        safe_pool_slot = tl.maximum(pool_slot, 0)

        scale_groups = tl.arange(0, 4)
        group_dims = tl.arange(0, 128)
        dims = scale_groups[:, None] * 128 + group_dims[None, :]
        record = k5_cache_ptr + block * cache_stride_block
        indices = dims * _TL_K5_GROUP + token
        codes = _unpack_k5(record, indices, body).to(tl.float32)
        fp16_record = record.to(tl.pointer_type(tl.float16))
        s_col = tl.load(
            fp16_record + _TL_K5_S_COL_OFFSET // 2 + dims,
            mask=body,
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            fp16_record + _TL_K5_ZP_OFFSET // 2 + dims,
            mask=body,
            other=0.0,
        ).to(tl.float32)
        s_row = tl.load(
            fp16_record + _TL_K5_S_ROW_OFFSET // 2 + token,
            mask=body,
            other=0.0,
        ).to(tl.float32)
        body_latent = (codes * s_col + zero) * s_row
        exact_latent = tl.load(
            latent_pool_ptr
            + safe_pool_slot * latent_pool_stride_slot
            + token * latent_pool_stride_token
            + dims,
            mask=exact,
            other=0.0,
        ).to(tl.float32)
        latent = (
            tl.where(exact, exact_latent, body_latent).to(tl.bfloat16).to(tl.float32)
        )

        amax = tl.max(tl.abs(latent), axis=1)
        scales = tl.where(amax > 0.0, amax / _TL_FP8_E4M3_MAX, 1.0)
        quantized = tl.maximum(
            tl.minimum(latent / scales[:, None], _TL_FP8_E4M3_MAX),
            -_TL_FP8_E4M3_MAX,
        ).to(tl.float8e4nv)
        output_record = output_ptr + row * output_stride_row
        fp8_output = output_record.to(tl.pointer_type(tl.float8e4nv))
        tl.store(fp8_output + dims, quantized)
        fp32_output = output_record.to(tl.pointer_type(tl.float32))
        tl.store(
            fp32_output + _TL_GLM_FP8_SCALE_OFFSET // 4 + scale_groups,
            scales,
        )

        rope_dims = tl.arange(0, _TL_K5_ROPE_DIM)
        body_rope = tl.load(
            (record + _TL_K5_ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
            + token * _TL_K5_ROPE_DIM
            + rope_dims,
            mask=body,
            other=0.0,
        )
        exact_rope = tl.load(
            rope_pool_ptr
            + safe_pool_slot * rope_pool_stride_slot
            + token * rope_pool_stride_token
            + rope_dims,
            mask=exact,
            other=0.0,
        )
        rope = tl.where(exact, exact_rope, body_rope)
        tl.store(
            (output_record + _TL_GLM_FP8_ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
            + rope_dims,
            rope,
        )


def stage_k5_as_fp8_records(
    physical_slots: torch.Tensor,
    k5_cache: torch.Tensor,
    block_to_pool_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Stage KVarN5 MLA tiles as SparkInfer GLM FP8 cache records.

    Invalid physical slots leave the corresponding output rows unchanged.
    """
    if physical_slots.ndim != 1 or physical_slots.dtype != torch.int32:
        raise ValueError("physical_slots must be a flat int32 tensor")
    if not physical_slots.is_contiguous():
        raise ValueError("physical_slots must be contiguous")
    if k5_cache.dtype != torch.uint8 or k5_cache.ndim != 3:
        raise ValueError("k5_cache must be a rank-3 uint8 tensor")
    if not k5_cache.is_contiguous():
        raise ValueError("k5_cache must be contiguous")
    if k5_cache.stride(0) != _K5_TILE_BYTES:
        raise ValueError(
            f"KVarN5 cache block stride must be {_K5_TILE_BYTES} bytes, "
            f"got {k5_cache.stride(0)}"
        )
    if block_to_pool_slot.ndim != 1 or block_to_pool_slot.dtype != torch.int32:
        raise ValueError("block_to_pool_slot must be a flat int32 tensor")
    if not block_to_pool_slot.is_contiguous():
        raise ValueError("block_to_pool_slot must be contiguous")
    if block_to_pool_slot.numel() != k5_cache.shape[0]:
        raise ValueError("block_to_pool_slot must contain one entry per cache block")
    if latent_pool.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("latent_pool must use BF16 or float8_e4m3fn")
    if latent_pool.ndim != 3 or tuple(latent_pool.shape[1:]) != (
        _K5_GROUP,
        _K5_LATENT_DIM,
    ):
        raise ValueError("latent_pool must have shape (slots,64,512)")
    if not latent_pool.is_contiguous():
        raise ValueError("latent_pool must be contiguous")
    if rope_pool.dtype != torch.bfloat16 or rope_pool.ndim != 3:
        raise ValueError("rope_pool must be a rank-3 BF16 tensor")
    if tuple(rope_pool.shape[1:]) != (_K5_GROUP, _K5_ROPE_DIM):
        raise ValueError("rope_pool must have shape (slots,64,64)")
    if not rope_pool.is_contiguous():
        raise ValueError("rope_pool must be contiguous")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("latent_pool and rope_pool must have equal slot capacities")
    if output.dtype != torch.uint8 or output.ndim != 2:
        raise ValueError("output must be a rank-2 uint8 tensor")
    if output.shape != (physical_slots.numel(), _GLM_FP8_RECORD_BYTES):
        raise ValueError(
            f"output must have shape ({physical_slots.numel()},{_GLM_FP8_RECORD_BYTES})"
        )
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    tensors = (
        physical_slots,
        k5_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        output,
    )
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("all KVarN5 staging tensors must share one device")
    if output.device.type != "cuda":
        raise ValueError("KVarN5 staging tensors must be CUDA tensors")
    aligned_tensors = (
        (physical_slots, 4, "physical_slots"),
        (k5_cache, 2, "k5_cache"),
        (block_to_pool_slot, 4, "block_to_pool_slot"),
        (latent_pool, latent_pool.element_size(), "latent_pool"),
        (rope_pool, 2, "rope_pool"),
        (output, 16, "output"),
    )
    for tensor, alignment, name in aligned_tensors:
        if tensor.data_ptr() % alignment:
            raise ValueError(f"{name} must be {alignment}-byte aligned")
    if physical_slots.numel() == 0:
        return

    _stage_k5_as_fp8_records_kernel[(physical_slots.numel(),)](
        physical_slots,
        k5_cache,
        block_to_pool_slot,
        latent_pool,
        rope_pool,
        output,
        k5_cache.stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        output.stride(0),
        num_blocks=k5_cache.shape[0],
        num_pool_slots=latent_pool.shape[0],
        num_warps=4,
    )

@triton.jit
def _native_k5_fused_merge_kernel(
    split_o, split_lse, output, output_lse,
    so_m: tl.constexpr, so_h: tl.constexpr, so_s: tl.constexpr,
    sl_m: tl.constexpr, sl_h: tl.constexpr, sl_s: tl.constexpr,
    out_m: tl.constexpr, out_h: tl.constexpr,
    ol_m: tl.constexpr, ol_h: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_SPLITS: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, BLOCK_SPLITS)
    lse = tl.load(
        split_lse + row * sl_m + head * sl_h + splits * sl_s,
        mask=splits < SPLITS,
        other=-float("inf"),
    )
    gmax = tl.max(lse, axis=0)
    valid = gmax != -float("inf")
    safe = tl.where(valid, gmax, 0.0)
    weights = tl.where(lse != -float("inf"), tl.exp2(lse - safe), 0.0)
    denom = tl.sum(weights, axis=0)
    norm = tl.where(denom > 0.0, weights / denom, 0.0)
    natural_lse = tl.where(
        valid, (safe + tl.log2(denom)) * 0.6931471805599453, -float("inf")
    )
    tl.store(output_lse + row * ol_m + head * ol_h, natural_lse)
    for d0 in range(0, D, BLOCK_D):
        dims = d0 + tl.arange(0, BLOCK_D)
        partial = tl.load(
            split_o + row * so_m + head * so_h
            + splits[:, None] * so_s + dims[None, :],
            mask=(lse != -float("inf"))[:, None] & (dims < D)[None, :],
            other=0.0,
        ).to(tl.float32)
        merged = tl.sum(partial * norm[:, None], axis=0)
        tl.store(
            output + row * out_m + head * out_h + dims,
            merged,
            mask=dims < D,
        )


def native_packed_k5_decode(
    q: torch.Tensor,
    selected_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    k5_cache: torch.Tensor,
    block_to_pool_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    num_chunks_ptr: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor,
    *,
    sm_scale: float,
    candidate_envelope: int,
    exact_pool_only: bool = False,
    fuse_kvarn_hadamard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native SM120 sparse MLA over K5/exact rows without a global row stage.

    ``exact_pool_only`` and ``fuse_kvarn_hadamard`` are Python-static launch
    modes. Fused Hadamard consumes unrotated Q-NoPE and returns unrotated output.
    Exact eager calls validate that every active selected row maps into the exact
    side pool; CUDA graph replay relies on the cache tracker's invariant.
    ``SPARKINFER_KVARN_MLA_EXACT_H16=1`` routes only exact-pool launches to
    the H16/block512 BF16-math specialization; mixed K5 and the unset default
    retain their existing launch specializations.

    """
    if q.ndim != 3:
        raise ValueError("native K5 query must be rank 3")
    rows, heads, width = map(int, q.shape)
    if rows not in (1, 4, 16) or heads != 64 or width != 576:
        raise ValueError("native K5 decode requires M1/M4/M16, H64, D576")
    if (
        q.dtype != torch.bfloat16
        or q.stride(2) != 1
        or q.stride(1) < width
        or q.stride(0) < heads * q.stride(1)
    ):
        raise ValueError("native K5 query must have a non-overlapping unit-inner BF16 layout")
    if selected_indices.dtype != torch.int32 or selected_indices.shape != (rows, 2048):
        raise ValueError("native K5 selected indices must be int32[M,2048]")
    if not selected_indices.is_contiguous():
        raise ValueError("native K5 selected indices must be contiguous")
    if valid_counts.dtype != torch.int32 or valid_counts.shape != (rows,):
        raise ValueError("native K5 valid counts must be int32[M]")
    if k5_cache.dtype != torch.uint8 or k5_cache.ndim != 3:
        raise ValueError("native K5 cache must be rank-3 uint8")
    if tuple(k5_cache.shape[1:]) != (1, _K5_TILE_BYTES) or not k5_cache.is_contiguous():
        raise ValueError("native K5 cache must be contiguous [blocks,1,30848]")
    if (
        block_to_pool_slot.dtype != torch.int32
        or block_to_pool_slot.shape != (k5_cache.shape[0],)
        or not block_to_pool_slot.is_contiguous()
    ):
        raise ValueError("native K5 block map must contain one contiguous int32 per block")
    if (
        latent_pool.dtype != torch.float8_e4m3fn
        or tuple(latent_pool.shape[1:]) != (64, 512)
        or not latent_pool.is_contiguous()
    ):
        raise ValueError("native K5 exact latent pool must be contiguous E4M3[slots,64,512]")
    if (
        rope_pool.dtype != torch.bfloat16
        or tuple(rope_pool.shape[1:]) != (64, 64)
        or not rope_pool.is_contiguous()
    ):
        raise ValueError("native K5 exact RoPE pool must be contiguous BF16[slots,64,64]")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("native K5 exact pools must have equal capacities")
    if type(exact_pool_only) is not bool:
        raise TypeError("native K5 exact_pool_only must be a Python bool")
    if type(fuse_kvarn_hadamard) is not bool:
        raise TypeError("native K5 fuse_kvarn_hadamard must be a Python bool")
    if not 0 <= int(candidate_envelope) <= 2048:
        raise ValueError("native K5 candidate envelope must be in [0,2048]")
    chunks = (int(candidate_envelope) + 63) // 64
    exact_h16 = bool(exact_pool_only and _exact_h16_enabled())
    chunks_per_split = 1 if rows == 1 else (3 if rows == 4 else 4)
    if rows == 4 and (not exact_pool_only or exact_h16):
        chunks_per_split = _mixed_m4_chunks_per_split()
    elif (
        _native_m5_split_family_enabled()
        and 2 <= rows <= 7
        and (not exact_pool_only or exact_h16)
    ):
        # M5/M6/M7 share the M4 mixed split family (default-off). The native
        # grid is one CTA per row, so per-row chunk-range walk and FP32
        # accumulation order are then identical to the M=4 verify path; only
        # the runtime grid dimension changes. With the knob unset, rows 2..7
        # keep chunks_per_split=4 exactly as before (fail-closed default).
        chunks_per_split = _mixed_m4_chunks_per_split()
    num_splits = (chunks + chunks_per_split - 1) // chunks_per_split
    if num_splits > int(split_output.shape[2]) or num_splits > int(split_lse.shape[2]):
        raise ValueError("native K5 split scratch is smaller than the DCP-local plan")
    tensors = (
        selected_indices, valid_counts, k5_cache, block_to_pool_slot,
        latent_pool, rope_pool, split_output, split_lse, num_chunks_ptr,
        output, output_lse,
    )
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("native K5 tensors must share one CUDA device")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise ValueError("native K5 decode requires SM120")
    if exact_pool_only and not torch.cuda.is_current_stream_capturing():
        prefix = selected_indices[:, : int(candidate_envelope)]
        active = (
            torch.arange(int(candidate_envelope), device=q.device)[None, :]
            < valid_counts[:, None]
        )
        max_physical = int(k5_cache.shape[0]) * 64
        if max_physical == 0 and bool(active.any().item()):
            raise ValueError("native exact-pool decode has active rows but an empty cache")
        if max_physical:
            physical_ok = (prefix >= 0) & (prefix < max_physical)
            if bool((active & ~physical_ok).any().item()):
                raise ValueError("native exact-pool decode selected an invalid physical slot")
            safe_blocks = prefix.clamp(0, max_physical - 1) // 64
            # Profiling/dummy MTP runs before ownership maps exist and presents
            # an entirely unmapped (-1) block map. The kernel masks those rows.
            # Once any exact mapping exists, retain fail-closed validation for
            # every active selected candidate so real requests cannot mix K5.
            all_unmapped = bool((block_to_pool_slot == -1).all().item())
            if not all_unmapped:
                pool_slots = block_to_pool_slot[safe_blocks]
                exact = (pool_slots >= 0) & (pool_slots < int(latent_pool.shape[0]))
                if bool((active & ~exact).any().item()):
                    raise ValueError("native exact-pool decode selected a non-exact cache row")
    if chunks == 0:
        output[:rows].zero_()
        output_lse[:rows].fill_(-float("inf"))
        return output[:rows], output_lse[:rows]

    if num_splits == 1:
        mid_out = output[:rows, :heads, :512].unsqueeze(2)
        mid_lse = output_lse[:rows, :heads].unsqueeze(2)
    else:
        mid_out = split_output[:rows, :heads, :num_splits, :512]
        mid_lse = split_lse[:rows, :heads, :num_splits]
    # Import registers the CuTe custom op; kept lazy to avoid package cycles.
    from sparkinfer.attention._shared.mla import kernel as _native_kernel  # noqa: F401
    op = (
        torch.ops.sparkinfer.kvarn_mla_sm120_decode_grid_exact_h16
        if exact_h16
        else torch.ops.sparkinfer.kvarn_mla_sm120_decode_grid
    )
    op(
        q, k5_cache.view(-1), selected_indices, mid_out, mid_lse,
        valid_counts, block_to_pool_slot, latent_pool, rope_pool,
        float(sm_scale), num_splits, chunks_per_split, exact_pool_only,
        fuse_kvarn_hadamard,
    )
    if num_splits == 1:
        return output[:rows, :heads, :512], output_lse[:rows, :heads]
    # Keep each live FP32 partial tile at no more than eight elements per thread.
    # Grouping heads here spills the M1/S32 tile and turns the merge into local
    # scratch traffic; independent heads provide enough parallelism instead.
    merge_warps = 8 if num_splits > 16 else 4
    _native_k5_fused_merge_kernel[(rows, heads)](
        mid_out, mid_lse, output, output_lse,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        output.stride(0), output.stride(1),
        output_lse.stride(0), output_lse.stride(1),
        num_splits, triton.next_power_of_2(num_splits), 512, 64,
        num_warps=merge_warps,
    )
    return output[:rows, :heads, :512], output_lse[:rows, :heads]


__all__ = ["stage_k5_as_fp8_records", "native_packed_k5_decode"]
