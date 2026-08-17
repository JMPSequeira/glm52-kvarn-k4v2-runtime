# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed KVarN cache operations for MLA latent caches."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNMLAConfig
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kvarn_store import (
    kvarn_store_tile_k_batch_from_sinkhorn,
)
from vllm.v1.attention.ops.triton_kvarn_sinkhorn import (
    kvarn_sinkhorn_g64_512,
    kvarn_sinkhorn_triton,
)

# One G64 hybrid tile is 64 token-major compact NVFP4/FP8-RoPE records
# followed by 512 BF16 KVarN channel factors.  Keep these offsets explicit:
# they are the cross-kernel ABI shared with the regular compact writer.
KVARN_NVFP4_GROUP = 64
KVARN_NVFP4_LATENT_DIM = 512
KVARN_NVFP4_ROPE_DIM = 64
KVARN_NVFP4_RECORD_BYTES = 368
KVARN_NVFP4_RECORDS_BYTES = KVARN_NVFP4_GROUP * KVARN_NVFP4_RECORD_BYTES
KVARN_NVFP4_CHANNEL_OFFSET = KVARN_NVFP4_RECORDS_BYTES
KVARN_NVFP4_TILE_BYTES = 24_576
KVARN_NVFP4_E2M1_OFFSET = 0
KVARN_NVFP4_E4M3_SCALE_OFFSET = 256
KVARN_NVFP4_ROPE_SCALE_OFFSET = 288
KVARN_NVFP4_LATENT_OUTER_OFFSET = 292
KVARN_NVFP4_ROPE_OFFSET = 304
_KVARN_NVFP4_GROUP_TL = tl.constexpr(KVARN_NVFP4_GROUP)
_KVARN_NVFP4_LATENT_DIM_TL = tl.constexpr(KVARN_NVFP4_LATENT_DIM)
_KVARN_NVFP4_ROPE_DIM_TL = tl.constexpr(KVARN_NVFP4_ROPE_DIM)
_KVARN_NVFP4_RECORD_BYTES_TL = tl.constexpr(KVARN_NVFP4_RECORD_BYTES)
_KVARN_NVFP4_CHANNEL_OFFSET_TL = tl.constexpr(KVARN_NVFP4_CHANNEL_OFFSET)
_KVARN_NVFP4_E4M3_SCALE_OFFSET_TL = tl.constexpr(KVARN_NVFP4_E4M3_SCALE_OFFSET)
_KVARN_NVFP4_ROPE_SCALE_OFFSET_TL = tl.constexpr(KVARN_NVFP4_ROPE_SCALE_OFFSET)
_KVARN_NVFP4_LATENT_OUTER_OFFSET_TL = tl.constexpr(KVARN_NVFP4_LATENT_OUTER_OFFSET)
_KVARN_NVFP4_ROPE_OFFSET_TL = tl.constexpr(KVARN_NVFP4_ROPE_OFFSET)


def _is_kvarn_nvfp4(config: KVarNMLAConfig) -> bool:
    return bool(getattr(config, "is_nvfp4_ds", False))


def _validate_kvarn_nvfp4_geometry(
    kv_cache: torch.Tensor, config: KVarNMLAConfig
) -> None:
    if (
        config.group != KVARN_NVFP4_GROUP
        or config.latent_dim != KVARN_NVFP4_LATENT_DIM
        or config.rope_dim != KVARN_NVFP4_ROPE_DIM
    ):
        raise ValueError("KVarN-NVFP4 requires fixed G64 D512+R64 geometry")
    if kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous():
        raise ValueError("KVarN-NVFP4 cache must be contiguous uint8")
    if (
        kv_cache.ndim == 0
        or kv_cache.numel() != kv_cache.shape[0] * KVARN_NVFP4_TILE_BYTES
    ):
        raise ValueError("KVarN-NVFP4 cache must contain exactly 24,576 bytes per tile")


@triton.jit
def _e2m1_from_nibble(nibble):
    magnitude = nibble & 0x07
    value = tl.where(magnitude == 1, 0.5, 0.0)
    value = tl.where(magnitude == 2, 1.0, value)
    value = tl.where(magnitude == 3, 1.5, value)
    value = tl.where(magnitude == 4, 2.0, value)
    value = tl.where(magnitude == 5, 3.0, value)
    value = tl.where(magnitude == 6, 4.0, value)
    value = tl.where(magnitude == 7, 6.0, value)
    return tl.where((nibble & 0x08) != 0, -value, value)


@triton.jit
def _quantize_e2m1_nibble(value):
    """RTNE E2M1 encoding used by NVIDIA's satfinite FP4 conversion."""
    magnitude = tl.abs(value)
    code = tl.where(magnitude > 5.0, 7, 0)
    code = tl.where((magnitude >= 3.5) & (magnitude <= 5.0), 6, code)
    code = tl.where((magnitude > 2.5) & (magnitude < 3.5), 5, code)
    code = tl.where((magnitude >= 1.75) & (magnitude <= 2.5), 4, code)
    code = tl.where((magnitude > 1.25) & (magnitude < 1.75), 3, code)
    code = tl.where((magnitude >= 0.75) & (magnitude <= 1.25), 2, code)
    code = tl.where((magnitude > 0.25) & (magnitude < 0.75), 1, code)
    return code | tl.where(value < 0.0, 0x08, 0)


@triton.jit
def _serialize_kvarn_nvfp4_rope_kernel(
    rope_pool_ptr,
    block_ids_ptr,
    pool_slots_ptr,
    cache_ptr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    cache_stride_b: tl.constexpr,
):
    program = tl.program_id(0)
    tile = program // _KVARN_NVFP4_GROUP_TL
    token = program % _KVARN_NVFP4_GROUP_TL
    physical_block = tl.load(block_ids_ptr + tile)
    pool_slot = tl.load(pool_slots_ptr + tile)
    cols = tl.arange(0, _KVARN_NVFP4_ROPE_DIM_TL)
    rope = tl.load(
        rope_pool_ptr
        + pool_slot * rope_pool_stride_s
        + token * rope_pool_stride_t
        + cols
    ).to(tl.float32)
    rope_scale = tl.max(tl.abs(rope), axis=0) * (1.0 / 448.0)
    quantized = tl.where(rope_scale == 0.0, 0.0, rope / rope_scale).to(tl.float8e4nv)
    record = (
        cache_ptr
        + physical_block * cache_stride_b
        + token * _KVARN_NVFP4_RECORD_BYTES_TL
    )
    f32_record = record.to(tl.pointer_type(tl.float32))
    fp8_record = record.to(tl.pointer_type(tl.float8e4nv))
    tl.store(f32_record + _KVARN_NVFP4_ROPE_SCALE_OFFSET_TL // 4, rope_scale)
    tl.store(fp8_record + _KVARN_NVFP4_ROPE_OFFSET_TL + cols, quantized)
    pad = tl.arange(0, 16)
    tl.store(record + _KVARN_NVFP4_LATENT_OUTER_OFFSET_TL + pad, 0, mask=pad < 12)


@triton.jit
def _serialize_kvarn_nvfp4_latent_kernel(
    latent_pool_ptr,
    s_col_ptr,
    s_row_ptr,
    block_ids_ptr,
    pool_slots_ptr,
    cache_ptr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    latent_pool_stride_d: tl.constexpr,
    s_col_stride_n: tl.constexpr,
    s_row_stride_n: tl.constexpr,
    cache_stride_b: tl.constexpr,
):
    program = tl.program_id(0)
    tile = program // _KVARN_NVFP4_GROUP_TL
    token = program % _KVARN_NVFP4_GROUP_TL
    physical_block = tl.load(block_ids_ptr + tile)
    pool_slot = tl.load(pool_slots_ptr + tile)
    token_factor = tl.load(s_col_ptr + pool_slot * s_col_stride_n + token).to(
        tl.float32
    )
    cols = tl.arange(0, _KVARN_NVFP4_LATENT_DIM_TL)
    channel_factors = tl.load(s_row_ptr + pool_slot * s_row_stride_n + cols).to(
        tl.float32
    )
    latent = tl.load(
        latent_pool_ptr
        + pool_slot * latent_pool_stride_s
        + token * latent_pool_stride_t
        + cols * latent_pool_stride_d
    ).to(tl.float32)
    balanced = latent / token_factor / channel_factors
    groups = tl.reshape(balanced, (32, 16))
    group_amax = tl.max(tl.abs(groups), axis=1)
    token_amax = tl.max(group_amax, axis=0)
    latent_outer = token_amax * (1.0 / (6.0 * 448.0))
    inv_outer = tl.where(latent_outer == 0.0, 0.0, 1.0 / latent_outer)
    group_scales = group_amax * inv_outer * (1.0 / 6.0)
    group_scales_fp8 = group_scales.to(tl.float8e4nv)
    decoded_scales = group_scales_fp8.to(tl.float32)
    inv_group = tl.where(decoded_scales == 0.0, 0.0, 1.0 / decoded_scales)
    scaled = groups * inv_group[:, None] * inv_outer
    nibbles = tl.reshape(
        _quantize_e2m1_nibble(tl.reshape(scaled, (_KVARN_NVFP4_LATENT_DIM_TL,))),
        (_KVARN_NVFP4_LATENT_DIM_TL // 2, 2),
    )
    even, odd = tl.split(nibbles)
    packed = tl.reshape(even | (odd << 4), (256,))
    packed_cols = tl.arange(0, _KVARN_NVFP4_LATENT_DIM_TL // 2)
    record = (
        cache_ptr
        + physical_block * cache_stride_b
        + token * _KVARN_NVFP4_RECORD_BYTES_TL
    )
    fp8_record = record.to(tl.pointer_type(tl.float8e4nv))
    f32_record = record.to(tl.pointer_type(tl.float32))
    tl.store(record + packed_cols, packed.to(tl.uint8))
    scale_cols = tl.arange(0, 32)
    tl.store(
        fp8_record + _KVARN_NVFP4_E4M3_SCALE_OFFSET_TL + scale_cols,
        group_scales_fp8,
    )
    tl.store(
        f32_record + _KVARN_NVFP4_LATENT_OUTER_OFFSET_TL // 4,
        latent_outer * token_factor,
    )
    pad = tl.arange(0, 8)
    tl.store(record + _KVARN_NVFP4_LATENT_OUTER_OFFSET_TL + 4 + pad, 0)

    footer = (
        cache_ptr + physical_block * cache_stride_b + _KVARN_NVFP4_CHANNEL_OFFSET_TL
    ).to(tl.pointer_type(tl.bfloat16))
    tl.store(footer + cols, channel_factors, mask=token == 0)


def pack_kvarn_nvfp4_mla_blocks(
    kv_cache: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Serialize retired G64 tiles as dynamic NVFP4/FP8-RoPE plus BF16 KVarN."""
    _validate_kvarn_nvfp4_geometry(kv_cache, config)
    if block_ids.numel() == 0:
        return
    if block_ids.shape != pool_slots.shape:
        raise ValueError("KVarN retirement block and pool-slot shapes must match")
    if not block_ids.is_contiguous() or not pool_slots.is_contiguous():
        raise ValueError("KVarN retirement indices must be contiguous")

    cache_bytes = kv_cache.view(torch.uint8)
    rows = block_ids.numel() * KVARN_NVFP4_GROUP
    # Sinkhorn uses retired RoPE slots as scratch, so the RoPE record must be
    # complete before that scratch is touched on the same stream.
    _serialize_kvarn_nvfp4_rope_kernel[(rows,)](
        rope_pool,
        block_ids,
        pool_slots,
        cache_bytes,
        rope_pool.stride(0),
        rope_pool.stride(1),
        cache_bytes.stride(0),
        num_warps=4,
    )
    s_col, s_row = kvarn_sinkhorn_g64_512(
        latent_pool,
        pool_slots,
        rope_pool,
        iterations=config.sinkhorn_iters,
    )
    _serialize_kvarn_nvfp4_latent_kernel[(rows,)](
        latent_pool,
        s_col,
        s_row,
        block_ids,
        pool_slots,
        cache_bytes,
        latent_pool.stride(0),
        latent_pool.stride(1),
        latent_pool.stride(2),
        s_col.stride(0),
        s_row.stride(0),
        cache_bytes.stride(0),
        num_warps=8,
    )


@triton.jit
def _unpack_dense_bits(payload_ptr, value_indices, mask, bits: tl.constexpr):
    bit_offsets = value_indices * bits
    byte_offsets = bit_offsets // 8
    shifts = bit_offsets % 8
    lo = tl.load(payload_ptr + byte_offsets, mask=mask, other=0).to(tl.uint32)
    hi = tl.load(payload_ptr + byte_offsets + 1, mask=mask, other=0).to(tl.uint32)
    return ((lo | (hi << 8)) >> shifts) & ((1 << bits) - 1)


@triton.jit
def _scatter_kvarn_mla_exact_kernel(
    latent_ptr,
    rope_ptr,
    slot_mapping_ptr,
    block_to_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    latent_stride_t: tl.constexpr,
    rope_stride_t: tl.constexpr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_POOL_SLOTS: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0)
    physical_slot = tl.load(slot_mapping_ptr + token)
    valid_slot = physical_slot >= 0
    block = physical_slot // GROUP
    offset = physical_slot % GROUP
    pool_slot = tl.load(
        block_to_slot_ptr + block,
        mask=valid_slot & (block < NUM_BLOCKS),
        other=-1,
    )
    valid = (
        valid_slot
        & (block < NUM_BLOCKS)
        & (pool_slot >= 0)
        & (pool_slot < NUM_POOL_SLOTS)
    )
    safe_pool_slot = tl.where(valid, pool_slot, 0)

    latent_cols = tl.arange(0, LATENT_DIM)
    latent = tl.load(latent_ptr + token * latent_stride_t + latent_cols)
    tl.store(
        latent_pool_ptr
        + safe_pool_slot * latent_pool_stride_s
        + offset * latent_pool_stride_t
        + latent_cols,
        latent,
        mask=valid,
    )

    rope_cols = tl.arange(0, ROPE_DIM)
    rope = tl.load(rope_ptr + token * rope_stride_t + rope_cols)
    tl.store(
        rope_pool_ptr
        + safe_pool_slot * rope_pool_stride_s
        + offset * rope_pool_stride_t
        + rope_cols,
        rope,
        mask=valid,
    )


def scatter_kvarn_mla_exact(
    latent_rotated: torch.Tensor,
    rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
) -> None:
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    if latent_rotated.shape[0] < num_tokens or rope.shape[0] < num_tokens:
        raise ValueError("KVarN MLA exact scatter inputs have too few rows")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")
    if not slot_mapping.is_contiguous() or not block_to_slot.is_contiguous():
        raise ValueError("KVarN MLA exact scatter index buffers must be contiguous")
    _scatter_kvarn_mla_exact_kernel[(num_tokens,)](
        latent_rotated,
        rope,
        slot_mapping,
        block_to_slot,
        latent_pool,
        rope_pool,
        latent_rotated.stride(0),
        rope.stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        NUM_BLOCKS=block_to_slot.shape[0],
        NUM_POOL_SLOTS=latent_pool.shape[0],
        GROUP=latent_pool.shape[1],
        LATENT_DIM=latent_pool.shape[2],
        ROPE_DIM=rope_pool.shape[2],
        num_warps=4,
    )


@triton.jit
def _round_to_even(values):
    lower = tl.floor(values)
    fraction = values - lower
    lower_int = lower.to(tl.int32)
    ties_up = (lower_int & 1) != 0
    return tl.where(
        fraction > 0.5,
        lower + 1.0,
        tl.where(fraction < 0.5, lower, lower + ties_up),
    )


@triton.jit
def _quantize_rtn(values, lower, scale, qmax: tl.constexpr):
    return tl.maximum(
        tl.minimum(_round_to_even((values - lower) / scale), qmax),
        0.0,
    )


@triton.jit
def _copy_retired_kvarn_mla_rope_kernel(
    rope_pool_ptr,
    block_ids_ptr,
    pool_slots_ptr,
    cache_ptr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    cache_stride_b: tl.constexpr,
    GROUP: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
):
    program = tl.program_id(0)
    tile = program // GROUP
    token = program % GROUP
    pool_slot = tl.load(pool_slots_ptr + tile)
    physical_block = tl.load(block_ids_ptr + tile)
    cols = tl.arange(0, ROPE_DIM)
    rope = tl.load(
        rope_pool_ptr
        + pool_slot * rope_pool_stride_s
        + token * rope_pool_stride_t
        + cols
    )
    record = cache_ptr + physical_block * cache_stride_b + ROPE_OFFSET
    rope_record = record.to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_record + token * ROPE_DIM + cols, rope)


def _copy_retired_kvarn_mla_rope(
    kv_cache: torch.Tensor,
    rope_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    cache_bytes = kv_cache.view(torch.uint8)
    _copy_retired_kvarn_mla_rope_kernel[(block_ids.numel() * config.group,)](
        rope_pool,
        block_ids,
        pool_slots,
        cache_bytes,
        rope_pool.stride(0),
        rope_pool.stride(1),
        cache_bytes.stride(0),
        GROUP=config.group,
        ROPE_DIM=config.rope_dim,
        ROPE_OFFSET=config.rope_offset,
        num_warps=4,
    )


@triton.jit
def _serialize_kvarn_mla_blocks_kernel(
    latent_pool_ptr,
    s_col_ptr,
    s_row_ptr,
    block_ids_ptr,
    pool_slots_ptr,
    cache_ptr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    latent_pool_stride_d: tl.constexpr,
    s_col_stride_n: tl.constexpr,
    s_row_stride_n: tl.constexpr,
    cache_stride_b: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    BITS: tl.constexpr,
    PACKED_ROW_BYTES: tl.constexpr,
    S_COL_OFFSET: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    S_ROW_OFFSET: tl.constexpr,
    AFFINE_REFIT: tl.constexpr,
):
    program = tl.program_id(0)
    tile = program // LATENT_DIM
    row = program % LATENT_DIM
    tokens = tl.arange(0, GROUP)
    pool_slot = tl.load(pool_slots_ptr + tile)
    row_scale = tl.load(s_row_ptr + pool_slot * s_row_stride_n + row).to(tl.float32)
    token_scales = tl.load(s_col_ptr + pool_slot * s_col_stride_n + tokens).to(
        tl.float32
    )
    values = tl.load(
        latent_pool_ptr
        + pool_slot * latent_pool_stride_s
        + tokens * latent_pool_stride_t
        + row * latent_pool_stride_d
    ).to(tl.float32)
    values = values / token_scales / row_scale

    qmax: tl.constexpr = (1 << BITS) - 1
    lower = tl.min(values, axis=0)
    upper = tl.max(values, axis=0)
    quant_scale = tl.maximum((upper - lower) / qmax, 1e-10)
    scale = quant_scale
    zero = lower
    codes = _quantize_rtn(values, lower, quant_scale, qmax)

    if AFFINE_REFIT:
        weights = token_scales * token_scales
        weight_sum = tl.maximum(tl.sum(weights, axis=0), 1e-20)
        code_mean = tl.sum(weights * codes, axis=0) / weight_sum
        value_mean = tl.sum(weights * values, axis=0) / weight_sum
        centered_codes = codes - code_mean
        denominator = tl.sum(weights * centered_codes * centered_codes, axis=0)
        numerator = tl.sum(
            weights * centered_codes * (values - value_mean),
            axis=0,
        )
        fitted_scale = tl.maximum(numerator / tl.maximum(denominator, 1e-20), 1e-10)
        fitted_zero = value_mean - fitted_scale * code_mean
        usable = denominator > 1e-20
        scale = tl.where(usable, fitted_scale, scale)
        zero = tl.where(usable, fitted_zero, zero)

    physical_block = tl.load(block_ids_ptr + tile)
    record = cache_ptr + physical_block * cache_stride_b
    fp16_record = record.to(tl.pointer_type(tl.float16))
    tl.store(fp16_record + S_COL_OFFSET // 2 + row, row_scale * scale)
    tl.store(fp16_record + ZP_OFFSET // 2 + row, row_scale * zero)

    packed_offsets = tl.arange(0, 64)
    packed_mask = packed_offsets < PACKED_ROW_BYTES
    bit_offsets = packed_offsets * 8
    source = bit_offsets // BITS
    shifts = bit_offsets % BITS

    source0_scale = tl.load(
        s_col_ptr + pool_slot * s_col_stride_n + source,
        mask=packed_mask & (source < GROUP),
        other=1.0,
    ).to(tl.float32)
    source1_scale = tl.load(
        s_col_ptr + pool_slot * s_col_stride_n + source + 1,
        mask=packed_mask & (source + 1 < GROUP),
        other=1.0,
    ).to(tl.float32)
    source2_scale = tl.load(
        s_col_ptr + pool_slot * s_col_stride_n + source + 2,
        mask=packed_mask & (source + 2 < GROUP),
        other=1.0,
    ).to(tl.float32)
    source0 = tl.load(
        latent_pool_ptr
        + pool_slot * latent_pool_stride_s
        + source * latent_pool_stride_t
        + row * latent_pool_stride_d,
        mask=packed_mask & (source < GROUP),
        other=0.0,
    ).to(tl.float32)
    source1 = tl.load(
        latent_pool_ptr
        + pool_slot * latent_pool_stride_s
        + (source + 1) * latent_pool_stride_t
        + row * latent_pool_stride_d,
        mask=packed_mask & (source + 1 < GROUP),
        other=0.0,
    ).to(tl.float32)
    source2 = tl.load(
        latent_pool_ptr
        + pool_slot * latent_pool_stride_s
        + (source + 2) * latent_pool_stride_t
        + row * latent_pool_stride_d,
        mask=packed_mask & (source + 2 < GROUP),
        other=0.0,
    ).to(tl.float32)
    source0 = source0 / source0_scale / row_scale
    source1 = source1 / source1_scale / row_scale
    source2 = source2 / source2_scale / row_scale
    code0 = _quantize_rtn(source0, lower, quant_scale, qmax).to(tl.uint32)
    code1 = _quantize_rtn(source1, lower, quant_scale, qmax).to(tl.uint32)
    code2 = _quantize_rtn(source2, lower, quant_scale, qmax).to(tl.uint32)
    words = code0 | (code1 << BITS) | (code2 << (2 * BITS))
    packed = (words >> shifts) & 0xFF
    tl.store(
        record + row * PACKED_ROW_BYTES + packed_offsets,
        packed.to(tl.uint8),
        mask=packed_mask,
    )

    shared_offsets = tl.arange(0, 64)
    shared_mask = shared_offsets < GROUP
    token_scales = tl.load(
        s_col_ptr + pool_slot * s_col_stride_n + shared_offsets,
        mask=shared_mask,
    )
    tl.store(
        fp16_record + S_ROW_OFFSET // 2 + shared_offsets,
        token_scales,
        mask=(row == 0) & shared_mask,
    )


def _serialize_kvarn_mla_blocks(
    kv_cache: torch.Tensor,
    latent_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    packed_row_bytes = config.latent_packed_bytes // config.latent_dim
    cache_bytes = kv_cache.view(torch.uint8)
    _serialize_kvarn_mla_blocks_kernel[(block_ids.numel() * config.latent_dim,)](
        latent_pool,
        s_col,
        s_row,
        block_ids,
        pool_slots,
        cache_bytes,
        latent_pool.stride(0),
        latent_pool.stride(1),
        latent_pool.stride(2),
        s_col.stride(0),
        s_row.stride(0),
        cache_bytes.stride(0),
        GROUP=config.group,
        LATENT_DIM=config.latent_dim,
        BITS=config.bits,
        PACKED_ROW_BYTES=packed_row_bytes,
        S_COL_OFFSET=config.latent_s_col_offset,
        ZP_OFFSET=config.latent_zp_offset,
        S_ROW_OFFSET=config.latent_s_row_offset,
        AFFINE_REFIT=os.environ.get("KVARN_AFFINE_REFIT", "1") == "1",
        num_warps=4,
    )


def pack_kvarn_mla_blocks(
    kv_cache: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Quantize complete latent tiles into the configured packed format."""
    if _is_kvarn_nvfp4(config):
        pack_kvarn_nvfp4_mla_blocks(
            kv_cache,
            latent_pool,
            rope_pool,
            block_ids,
            pool_slots,
            config,
        )
        return
    if block_ids.numel() == 0:
        return
    if block_ids.shape != pool_slots.shape:
        raise ValueError("KVarN retirement block and pool-slot shapes must match")
    if not block_ids.is_contiguous() or not pool_slots.is_contiguous():
        raise ValueError("KVarN retirement indices must be contiguous")
    quantile = float(os.environ.get("KVARN_RTN_QUANTILE", "") or 0.0)
    fixed_g64_512 = (
        config.bits == 5
        and config.group == 64
        and config.latent_dim == 512
        and config.rope_dim == 64
        and quantile <= 0.0
    )
    if fixed_g64_512:
        # This launch must precede every scratch write: the fixed Sinkhorn path
        # reuses only retired RoPE slots, and the final serializer completes
        # before the state manager can release those exact-pool slots.
        _copy_retired_kvarn_mla_rope(
            kv_cache,
            rope_pool,
            block_ids,
            pool_slots,
            config,
        )
        s_col, s_row = kvarn_sinkhorn_g64_512(
            latent_pool,
            pool_slots,
            rope_pool,
            iterations=config.sinkhorn_iters,
        )
        _serialize_kvarn_mla_blocks(
            kv_cache,
            latent_pool,
            block_ids,
            pool_slots,
            s_col,
            s_row,
            config,
        )
        return

    latent = latent_pool.index_select(0, pool_slots).float()
    latent_tiles = latent.transpose(1, 2).contiguous()
    balanced, s_col, s_row = kvarn_sinkhorn_triton(
        latent_tiles, iterations=config.sinkhorn_iters
    )
    packed = kvarn_store_tile_k_batch_from_sinkhorn(
        balanced, s_col, s_row, bits=config.bits
    )

    num_blocks = block_ids.numel()
    records = torch.zeros(
        (num_blocks, config.tile_bytes), dtype=torch.uint8, device=kv_cache.device
    )
    records[:, : config.latent_packed_bytes].copy_(
        packed["q_packed_uint8"].reshape(num_blocks, -1)
    )
    records[
        :,
        config.latent_s_col_offset : config.latent_zp_offset,
    ].copy_(packed["s_col_K"].contiguous().view(torch.uint8))
    records[
        :,
        config.latent_zp_offset : config.latent_s_row_offset,
    ].copy_(packed["zp_K"].contiguous().view(torch.uint8))
    records[:, config.latent_s_row_offset : config.rope_offset].copy_(
        packed["s_row_K"].contiguous().view(torch.uint8)
    )
    rope = rope_pool.index_select(0, pool_slots).to(torch.bfloat16).contiguous()
    records[:, config.rope_offset :].copy_(
        rope.view(torch.uint8).reshape(num_blocks, -1)
    )
    kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1).index_copy_(
        0, block_ids, records
    )


@triton.jit
def _materialize_selected_kvarn_mla_kernel(
    selected_ptr,
    cache_ptr,
    block_to_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    output_ptr,
    selected_stride: tl.constexpr,
    cache_stride_b: tl.constexpr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    output_stride: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_POOL_SLOTS: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BITS: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    S_COL_OFFSET: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    S_ROW_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    width = LATENT_DIM + ROPE_DIM
    col_mask = cols < width
    physical_slot = tl.load(selected_ptr + row * selected_stride)
    valid_slot = physical_slot >= 0
    block = physical_slot // GROUP
    token = physical_slot % GROUP
    valid_block = valid_slot & (block < NUM_BLOCKS)
    pool_slot = tl.load(block_to_slot_ptr + block, mask=valid_block, other=-1)
    exact = valid_block & (pool_slot >= 0) & (pool_slot < NUM_POOL_SLOTS)
    safe_pool_slot = tl.where(exact, pool_slot, 0)
    safe_block = tl.where(valid_block, block, 0)
    record = cache_ptr + safe_block * cache_stride_b

    latent_mask = col_mask & (cols < LATENT_DIM) & valid_block
    body_latent_mask = latent_mask & ~exact
    latent_indices = cols * GROUP + token
    q = _unpack_dense_bits(record, latent_indices, body_latent_mask, BITS).to(
        tl.float32
    )
    fp16_record = record.to(tl.pointer_type(tl.float16))
    s_col = tl.load(
        fp16_record + S_COL_OFFSET // 2 + cols,
        mask=body_latent_mask,
        other=0.0,
    )
    zp = tl.load(
        fp16_record + ZP_OFFSET // 2 + cols,
        mask=body_latent_mask,
        other=0.0,
    )
    s_row = tl.load(
        fp16_record + S_ROW_OFFSET // 2 + token,
        mask=valid_block & ~exact,
        other=0.0,
    )
    body_latent = (q * s_col + zp) * s_row
    exact_latent = tl.load(
        latent_pool_ptr
        + safe_pool_slot * latent_pool_stride_s
        + token * latent_pool_stride_t
        + cols,
        mask=latent_mask & exact,
        other=0.0,
    ).to(tl.float32)
    latent = tl.where(exact, exact_latent, body_latent)

    rope_cols = cols - LATENT_DIM
    rope_mask = col_mask & (cols >= LATENT_DIM)
    body_rope_ptr = (record + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
    body_rope = tl.load(
        body_rope_ptr + token * ROPE_DIM + rope_cols,
        mask=rope_mask & valid_block & ~exact,
        other=0.0,
    ).to(tl.float32)
    exact_rope = tl.load(
        rope_pool_ptr
        + safe_pool_slot * rope_pool_stride_s
        + token * rope_pool_stride_t
        + rope_cols,
        mask=rope_mask & exact,
        other=0.0,
    ).to(tl.float32)
    rope_value = tl.where(exact, exact_rope, body_rope)
    value = tl.where(cols < LATENT_DIM, latent, rope_value)
    tl.store(
        output_ptr + row * output_stride + cols,
        value,
        mask=col_mask & valid_block,
    )


@triton.jit
def _materialize_selected_kvarn_nvfp4_mla_kernel(
    selected_ptr,
    cache_ptr,
    block_to_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    output_ptr,
    cache_stride_b: tl.constexpr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    output_stride: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_POOL_SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    physical_slot = tl.load(selected_ptr + row)
    valid_slot = physical_slot >= 0
    block = physical_slot // _KVARN_NVFP4_GROUP_TL
    token = physical_slot % _KVARN_NVFP4_GROUP_TL
    valid = valid_slot & (block < NUM_BLOCKS)
    pool_slot = tl.load(block_to_slot_ptr + block, mask=valid, other=-1)
    exact = valid & (pool_slot >= 0) & (pool_slot < NUM_POOL_SLOTS)
    safe_pool_slot = tl.where(exact, pool_slot, 0)
    safe_block = tl.where(valid, block, 0)
    safe_token = tl.where(valid, token, 0)
    tile = cache_ptr + safe_block * cache_stride_b
    record = tile + safe_token * _KVARN_NVFP4_RECORD_BYTES_TL

    cols = tl.arange(0, _KVARN_NVFP4_LATENT_DIM_TL)
    packed = tl.load(record + cols // 2, mask=valid & ~exact, other=0)
    nibble = tl.where((cols & 1) == 0, packed & 0x0F, packed >> 4)
    fp4 = _e2m1_from_nibble(nibble)
    raw_inner = tl.load(
        record + _KVARN_NVFP4_E4M3_SCALE_OFFSET_TL + cols // 16,
        mask=valid & ~exact,
        other=0,
    )
    inner = tl.cast(raw_inner, tl.float8e4nv, bitcast=True).to(tl.float32)
    f32_record = record.to(tl.pointer_type(tl.float32))
    outer = tl.load(
        f32_record + _KVARN_NVFP4_LATENT_OUTER_OFFSET_TL // 4,
        mask=valid & ~exact,
        other=0.0,
    )
    footer = (tile + _KVARN_NVFP4_CHANNEL_OFFSET_TL).to(tl.pointer_type(tl.bfloat16))
    channel = tl.load(footer + cols, mask=valid & ~exact, other=0.0).to(tl.float32)
    packed_latent = fp4 * inner * outer * channel
    exact_latent = tl.load(
        latent_pool_ptr
        + safe_pool_slot * latent_pool_stride_s
        + safe_token * latent_pool_stride_t
        + cols,
        mask=valid & exact,
        other=0.0,
    ).to(tl.float32)
    latent = tl.where(exact, exact_latent, packed_latent)
    tl.store(output_ptr + row * output_stride + cols, latent, mask=valid)

    rope_cols = tl.arange(0, _KVARN_NVFP4_ROPE_DIM_TL)
    raw_rope = tl.load(
        record + _KVARN_NVFP4_ROPE_OFFSET_TL + rope_cols,
        mask=valid & ~exact,
        other=0,
    )
    rope_q = tl.cast(raw_rope, tl.float8e4nv, bitcast=True).to(tl.float32)
    rope_scale = tl.load(
        f32_record + _KVARN_NVFP4_ROPE_SCALE_OFFSET_TL // 4,
        mask=valid & ~exact,
        other=0.0,
    )
    packed_rope = rope_q * rope_scale
    exact_rope = tl.load(
        rope_pool_ptr
        + safe_pool_slot * rope_pool_stride_s
        + safe_token * rope_pool_stride_t
        + rope_cols,
        mask=valid & exact,
        other=0.0,
    ).to(tl.float32)
    rope = tl.where(exact, exact_rope, packed_rope)
    tl.store(
        output_ptr + row * output_stride + _KVARN_NVFP4_LATENT_DIM_TL + rope_cols,
        rope,
        mask=valid,
    )


def materialize_selected_kvarn_nvfp4_mla(
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
    remapped: torch.Tensor | None,
    config: KVarNMLAConfig,
) -> None:
    """Materialize packed and exact hybrid rows without changing row identity."""
    _validate_kvarn_nvfp4_geometry(kv_cache, config)
    if selected_indices.dtype != torch.int32 or not selected_indices.is_contiguous():
        raise ValueError("KVarN MLA selected indices must be contiguous int32")
    if not output.is_contiguous():
        raise ValueError("KVarN MLA dense workspace must be contiguous")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")
    if (
        block_to_slot.dtype != torch.int32
        or not block_to_slot.is_contiguous()
        or block_to_slot.numel() < kv_cache.shape[0]
    ):
        raise ValueError("KVarN MLA block-to-slot map must cover the cache as int32")

    rows = selected_indices.numel()
    flat_output = output.view(-1, KVARN_NVFP4_LATENT_DIM + KVARN_NVFP4_ROPE_DIM)
    if rows > flat_output.shape[0]:
        raise ValueError(
            f"KVarN MLA dense workspace has {flat_output.shape[0]} rows, "
            f"requires {rows}"
        )
    flat_remapped = None
    if remapped is not None:
        if remapped.dtype != torch.int32 or not remapped.is_contiguous():
            raise ValueError("KVarN MLA remap workspace must be contiguous int32")
        flat_remapped = remapped.view(-1)
        if rows > flat_remapped.numel():
            raise ValueError(
                f"KVarN MLA remap workspace has {flat_remapped.numel()} entries, "
                f"requires {rows}"
            )
    if rows == 0:
        return

    _materialize_selected_kvarn_nvfp4_mla_kernel[(rows,)](
        selected_indices.view(-1),
        kv_cache.view(torch.uint8),
        block_to_slot,
        latent_pool,
        rope_pool,
        flat_output,
        kv_cache.view(torch.uint8).stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        flat_output.stride(0),
        NUM_BLOCKS=kv_cache.shape[0],
        NUM_POOL_SLOTS=latent_pool.shape[0],
        num_warps=8,
    )
    if flat_remapped is not None:
        _linearize_selected_kernel[(triton.cdiv(rows, 256),)](
            selected_indices.view(-1),
            flat_remapped,
            n_elements=rows,
            max_physical_slots=kv_cache.shape[0] * KVARN_NVFP4_GROUP,
        )


@triton.jit
def _linearize_selected_kernel(
    selected_ptr,
    remapped_ptr,
    n_elements: tl.constexpr,
    max_physical_slots: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    selected = tl.load(selected_ptr + offsets, mask=mask, other=-1)
    valid = (selected >= 0) & (selected < max_physical_slots)
    tl.store(remapped_ptr + offsets, tl.where(valid, offsets, -1), mask=mask)


def materialize_selected_kvarn_mla(
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
    remapped: torch.Tensor | None,
    config: KVarNMLAConfig,
) -> None:
    if _is_kvarn_nvfp4(config):
        materialize_selected_kvarn_nvfp4_mla(
            selected_indices,
            kv_cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            output,
            remapped,
            config,
        )
        return
    if selected_indices.dtype != torch.int32 or not selected_indices.is_contiguous():
        raise ValueError("KVarN MLA selected indices must be contiguous int32")
    if not output.is_contiguous():
        raise ValueError("KVarN MLA dense workspace must be contiguous")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")
    if block_to_slot.numel() < kv_cache.shape[0]:
        raise ValueError("KVarN MLA block-to-slot map is smaller than the cache")
    if block_to_slot.dtype != torch.int32:
        raise ValueError("KVarN MLA block-to-slot map must be int32")

    flat_selected = selected_indices.view(-1)
    flat_output = output.view(-1, config.latent_dim + config.rope_dim)
    rows = flat_selected.numel()
    if rows > flat_output.shape[0]:
        raise ValueError(
            f"KVarN MLA dense workspace has {flat_output.shape[0]} rows, "
            f"requires {rows}"
        )
    flat_remapped = None
    if remapped is not None:
        if remapped.dtype != torch.int32 or not remapped.is_contiguous():
            raise ValueError("KVarN MLA remap workspace must be contiguous int32")
        flat_remapped = remapped.view(-1)
        if rows > flat_remapped.numel():
            raise ValueError(
                f"KVarN MLA remap workspace has {flat_remapped.numel()} entries, "
                f"requires {rows}"
            )
    if rows == 0:
        return

    grid = (rows, triton.cdiv(config.latent_dim + config.rope_dim, 64))
    _materialize_selected_kvarn_mla_kernel[grid](
        flat_selected,
        kv_cache.view(torch.uint8),
        block_to_slot,
        latent_pool,
        rope_pool,
        flat_output,
        1,
        kv_cache.view(torch.uint8).stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        flat_output.stride(0),
        NUM_BLOCKS=kv_cache.shape[0],
        NUM_POOL_SLOTS=latent_pool.shape[0],
        GROUP=config.group,
        LATENT_DIM=config.latent_dim,
        ROPE_DIM=config.rope_dim,
        BITS=config.bits,
        PACKED_BYTES=config.latent_packed_bytes,
        S_COL_OFFSET=config.latent_s_col_offset,
        ZP_OFFSET=config.latent_zp_offset,
        S_ROW_OFFSET=config.latent_s_row_offset,
        ROPE_OFFSET=config.rope_offset,
        BLOCK_D=64,
        num_warps=4,
    )
    if flat_remapped is not None:
        _linearize_selected_kernel[(triton.cdiv(rows, 256),)](
            flat_selected,
            flat_remapped,
            n_elements=rows,
            max_physical_slots=kv_cache.shape[0] * config.group,
        )


@triton.jit
def _build_live_physical_slots_kernel(
    block_table_ptr,
    seq_lens_ptr,
    physical_slots_ptr,
    block_table_stride: tl.constexpr,
    max_blocks: tl.constexpr,
    num_blocks: tl.constexpr,
    group: tl.constexpr,
):
    dense_block = tl.program_id(0)
    request = dense_block // max_blocks
    logical_block = dense_block % max_blocks
    seq_len = tl.load(seq_lens_ptr + request)
    block_start = logical_block * group
    live_block = block_start < seq_len
    physical_block = tl.load(
        block_table_ptr + request * block_table_stride + logical_block,
        mask=live_block,
        other=-1,
    )
    live_block = live_block & (physical_block >= 0) & (physical_block < num_blocks)

    token = tl.arange(0, group)
    live_token = live_block & (block_start + token < seq_len)
    physical = physical_block * group + token
    tl.store(
        physical_slots_ptr + dense_block * group + token,
        tl.where(live_token, physical, -1),
    )


@triton.jit
def _build_live_block_lookup_kernel(
    block_table_ptr,
    seq_lens_ptr,
    block_to_logical_ptr,
    block_table_stride: tl.constexpr,
    max_blocks: tl.constexpr,
    num_blocks: tl.constexpr,
    group: tl.constexpr,
):
    dense_block = tl.program_id(0)
    request = dense_block // max_blocks
    logical_block = dense_block % max_blocks
    seq_len = tl.load(seq_lens_ptr + request)
    live_block = logical_block * group < seq_len
    physical_block = tl.load(
        block_table_ptr + request * block_table_stride + logical_block,
        mask=live_block,
        other=-1,
    )
    live_block = live_block & (physical_block >= 0) & (physical_block < num_blocks)
    tl.store(
        block_to_logical_ptr + physical_block,
        dense_block,
        mask=live_block,
    )


@triton.jit
def _remap_live_selected_kernel(
    selected_ptr,
    block_table_ptr,
    block_to_logical_ptr,
    seq_lens_ptr,
    remapped_ptr,
    block_table_stride: tl.constexpr,
    n_elements: tl.constexpr,
    num_reqs: tl.constexpr,
    num_blocks: tl.constexpr,
    max_blocks: tl.constexpr,
    group: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    physical = tl.load(selected_ptr + offsets, mask=mask, other=-1)
    physical_block = physical // group
    token = physical % group
    physical_valid = (physical >= 0) & (physical_block < num_blocks)
    dense_block = tl.load(
        block_to_logical_ptr + physical_block,
        mask=mask & physical_valid,
        other=-1,
    )
    request = dense_block // max_blocks
    logical_block = dense_block % max_blocks
    mapped = (dense_block >= 0) & (request < num_reqs)
    current_block = tl.load(
        block_table_ptr + request * block_table_stride + logical_block,
        mask=mask & mapped,
        other=-1,
    )
    seq_len = tl.load(
        seq_lens_ptr + request,
        mask=mask & mapped,
        other=0,
    )
    compact = dense_block * group + token
    live = (
        physical_valid
        & mapped
        & (current_block == physical_block)
        & (logical_block * group + token < seq_len)
    )
    tl.store(remapped_ptr + offsets, tl.where(live, compact, -1), mask=mask)


def stage_live_kvarn_mla_fp8_batch(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    physical_slots: torch.Tensor,
    output_records: torch.Tensor,
    block_to_logical: torch.Tensor,
    remapped: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Stage live request-major KVarN rows as native SparkInfer FP8 records."""
    stage_k5_as_fp8_records: Callable[..., None] | None = None
    if _is_kvarn_nvfp4(config):
        _validate_kvarn_nvfp4_geometry(kv_cache, config)
    else:
        try:
            from sparkinfer.attention.kvarn_mla import stage_k5_as_fp8_records
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "KVarN MLA requires SparkInfer with "
                "sparkinfer.attention.kvarn_mla.stage_k5_as_fp8_records"
            ) from exc

    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("KVarN MLA block table must be a two-dimensional int32 tensor")
    if block_table.stride(1) != 1:
        raise ValueError("KVarN MLA block table must have contiguous rows")
    num_reqs, max_blocks = block_table.shape
    if (
        seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
        or seq_lens.numel() != num_reqs
    ):
        raise ValueError(
            f"KVarN MLA live sequence lengths must be contiguous int32[{num_reqs}]"
        )
    if (
        selected_indices.dtype != torch.int32
        or not selected_indices.is_contiguous()
        or remapped.dtype != torch.int32
        or not remapped.is_contiguous()
    ):
        raise ValueError("KVarN MLA selected/remap buffers must be contiguous int32")
    if selected_indices.numel() > remapped.numel():
        raise ValueError(
            f"KVarN MLA remap workspace has {remapped.numel()} entries, "
            f"requires {selected_indices.numel()}"
        )
    if physical_slots.dtype != torch.int32 or not physical_slots.is_contiguous():
        raise ValueError("KVarN MLA live-slot workspace must be contiguous int32")
    if (
        block_to_logical.dtype != torch.int32
        or not block_to_logical.is_contiguous()
        or block_to_logical.numel() < kv_cache.shape[0]
    ):
        raise ValueError(
            "KVarN MLA physical-to-compact lookup must be contiguous int32 "
            "and cover the local cache"
        )
    if block_to_slot.numel() < kv_cache.shape[0]:
        raise ValueError("KVarN MLA block-to-slot map is smaller than the cache")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")

    dense_blocks = num_reqs * max_blocks
    dense_rows = dense_blocks * config.group
    if physical_slots.numel() < dense_rows:
        raise ValueError(
            f"KVarN MLA live-slot workspace has {physical_slots.numel()} rows, "
            f"requires {dense_rows}"
        )
    if (
        output_records.dtype != torch.uint8
        or not output_records.is_contiguous()
        or output_records.shape != (dense_rows, 656)
    ):
        raise ValueError(f"KVarN MLA FP8 workspace must have shape ({dense_rows}, 656)")
    if output_records.data_ptr() % 16:
        raise ValueError("KVarN MLA FP8 workspace must be 16-byte aligned")

    if dense_blocks == 0:
        remapped.view(-1)[: selected_indices.numel()].fill_(-1)
        return

    live_slots = physical_slots[:dense_rows]
    _build_live_physical_slots_kernel[(dense_blocks,)](
        block_table,
        seq_lens,
        live_slots,
        block_table.stride(0),
        max_blocks=max_blocks,
        num_blocks=kv_cache.shape[0],
        group=config.group,
        num_warps=4,
    )
    if _is_kvarn_nvfp4(config):
        _stage_selected_kvarn_nvfp4_mla_fp8_kernel[(dense_rows,)](
            live_slots,
            kv_cache.view(torch.uint8),
            block_to_slot,
            latent_pool,
            rope_pool,
            output_records,
            kv_cache.view(torch.uint8).stride(0),
            latent_pool.stride(0),
            latent_pool.stride(1),
            rope_pool.stride(0),
            rope_pool.stride(1),
            NUM_BLOCKS=kv_cache.shape[0],
            NUM_POOL_SLOTS=latent_pool.shape[0],
            num_warps=8,
        )
    else:
        assert stage_k5_as_fp8_records is not None
        stage_k5_as_fp8_records(
            live_slots,
            kv_cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            output_records,
        )

    # Do not clear the page-sized lookup. The remap kernel validates every
    # lookup hit against this step's block table and sequence length, so stale
    # entries are harmless and lookup work stays bounded by live requests.
    _build_live_block_lookup_kernel[(dense_blocks,)](
        block_table,
        seq_lens,
        block_to_logical,
        block_table.stride(0),
        max_blocks=max_blocks,
        num_blocks=kv_cache.shape[0],
        group=config.group,
        num_warps=4,
    )
    n_selected = selected_indices.numel()
    if n_selected:
        _remap_live_selected_kernel[(triton.cdiv(n_selected, 256),)](
            selected_indices.view(-1),
            block_table,
            block_to_logical,
            seq_lens,
            remapped.view(-1),
            block_table_stride=block_table.stride(0),
            n_elements=n_selected,
            num_reqs=num_reqs,
            num_blocks=kv_cache.shape[0],
            max_blocks=max_blocks,
            group=config.group,
        )


@triton.jit
def _copy_bounded_physical_indices_kernel(
    selected_ptr,
    remapped_ptr,
    n_elements: tl.constexpr,
    max_physical_slots: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    physical = tl.load(selected_ptr + offsets, mask=mask, other=-1)
    valid = (physical >= 0) & (physical < max_physical_slots)
    tl.store(remapped_ptr + offsets, tl.where(valid, physical, -1), mask=mask)


def remap_kvarn_mla_physical_indices(
    selected_indices: torch.Tensor,
    remapped: torch.Tensor,
    *,
    max_physical_slots: int,
) -> None:
    if (
        selected_indices.dtype != torch.int32
        or remapped.dtype != torch.int32
        or not selected_indices.is_contiguous()
        or not remapped.is_contiguous()
    ):
        raise ValueError("KVarN MLA physical index buffers must be contiguous int32")
    rows = selected_indices.numel()
    if rows > remapped.numel():
        raise ValueError(
            f"KVarN MLA remap workspace has {remapped.numel()} entries, requires {rows}"
        )
    if rows == 0:
        return
    _copy_bounded_physical_indices_kernel[(triton.cdiv(rows, 256),)](
        selected_indices.view(-1),
        remapped.view(-1),
        n_elements=rows,
        max_physical_slots=max_physical_slots,
    )


def materialize_physical_kvarn_mla(
    physical_slots: torch.Tensor,
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output: torch.Tensor,
    remapped: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Materialize the local page arena at its stable physical row indices."""
    page_rows = kv_cache.shape[0] * config.group
    if (
        physical_slots.dtype != torch.int32
        or not physical_slots.is_contiguous()
        or physical_slots.numel() != page_rows
    ):
        raise ValueError(
            "KVarN MLA physical-slot workspace must be contiguous int32 "
            f"with {page_rows} entries"
        )
    materialize_selected_kvarn_mla(
        physical_slots,
        kv_cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        output,
        None,
        config,
    )
    remap_kvarn_mla_physical_indices(
        selected_indices,
        remapped,
        max_physical_slots=page_rows,
    )


def stage_physical_kvarn_mla_fp8(
    physical_slots: torch.Tensor,
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    output_records: torch.Tensor,
    remapped: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Stage the local page arena in SparkInfer's physical token order."""
    try:
        from sparkinfer.attention.kvarn_mla import stage_k5_as_fp8_records
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "KVarN MLA requires SparkInfer with "
            "sparkinfer.attention.kvarn_mla.stage_k5_as_fp8_records"
        ) from exc

    page_rows = kv_cache.shape[0] * config.group
    if (
        physical_slots.dtype != torch.int32
        or not physical_slots.is_contiguous()
        or physical_slots.numel() != page_rows
    ):
        raise ValueError(
            "KVarN MLA physical-slot workspace must be contiguous int32 "
            f"with {page_rows} entries"
        )
    if (
        output_records.dtype != torch.uint8
        or not output_records.is_contiguous()
        or output_records.shape != (page_rows, 656)
    ):
        raise ValueError(f"KVarN MLA FP8 workspace must have shape ({page_rows}, 656)")
    if output_records.data_ptr() % 16:
        raise ValueError("KVarN MLA FP8 workspace must be 16-byte aligned")
    if block_to_slot.numel() < kv_cache.shape[0]:
        raise ValueError("KVarN MLA block-to-slot map is smaller than the cache")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")

    stage_k5_as_fp8_records(
        physical_slots,
        kv_cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        output_records,
    )
    remap_kvarn_mla_physical_indices(
        selected_indices,
        remapped,
        max_physical_slots=page_rows,
    )


@triton.jit
def _prepare_selected_kvarn_mla_fp8_kernel(
    selected_ptr,
    physical_slots_ptr,
    remapped_ptr,
    n_elements: tl.constexpr,
    max_physical_slots: tl.constexpr,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    physical = tl.load(selected_ptr + offsets, mask=mask, other=-1)
    valid = (physical >= 0) & (physical < max_physical_slots)
    # SparkInfer dereferences every input row. Invalid selections therefore
    # point at a known-valid cache row, while the native sparse kernel sees -1
    # and never consumes the staged record.
    tl.store(
        physical_slots_ptr + offsets,
        tl.where(valid, physical, 0),
        mask=mask,
    )
    tl.store(remapped_ptr + offsets, tl.where(valid, offsets, -1), mask=mask)


@triton.jit
def _stage_selected_kvarn_nvfp4_mla_fp8_kernel(
    physical_slots_ptr,
    cache_ptr,
    block_to_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    output_ptr,
    cache_stride_b: tl.constexpr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_POOL_SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    physical_slot = tl.load(physical_slots_ptr + row)
    valid = (physical_slot >= 0) & (physical_slot < NUM_BLOCKS * _KVARN_NVFP4_GROUP_TL)
    safe_physical = tl.where(valid, physical_slot, 0)
    block = safe_physical // _KVARN_NVFP4_GROUP_TL
    token = safe_physical % _KVARN_NVFP4_GROUP_TL
    pool_slot = tl.load(block_to_slot_ptr + block)
    exact = valid & (pool_slot >= 0) & (pool_slot < NUM_POOL_SLOTS)
    safe_pool_slot = tl.where(exact, pool_slot, 0)
    tile = cache_ptr + block * cache_stride_b
    record = tile + token * _KVARN_NVFP4_RECORD_BYTES_TL

    cols = tl.arange(0, _KVARN_NVFP4_LATENT_DIM_TL)
    packed = tl.load(record + cols // 2, mask=~exact, other=0)
    nibble = tl.where((cols & 1) == 0, packed & 0x0F, packed >> 4)
    fp4 = _e2m1_from_nibble(nibble)
    raw_inner = tl.load(
        record + _KVARN_NVFP4_E4M3_SCALE_OFFSET_TL + cols // 16,
        mask=~exact,
        other=0,
    )
    inner = tl.cast(raw_inner, tl.float8e4nv, bitcast=True).to(tl.float32)
    f32_record = record.to(tl.pointer_type(tl.float32))
    outer = tl.load(
        f32_record + _KVARN_NVFP4_LATENT_OUTER_OFFSET_TL // 4,
        mask=~exact,
        other=0.0,
    )
    footer = (tile + _KVARN_NVFP4_CHANNEL_OFFSET_TL).to(tl.pointer_type(tl.bfloat16))
    channel = tl.load(footer + cols, mask=~exact, other=0.0).to(tl.float32)
    packed_latent = fp4 * inner * outer * channel
    exact_latent = tl.load(
        latent_pool_ptr
        + safe_pool_slot * latent_pool_stride_s
        + token * latent_pool_stride_t
        + cols,
        mask=exact,
        other=0.0,
    ).to(tl.float32)
    latent = tl.where(exact, exact_latent, packed_latent)
    latent_2d = tl.reshape(latent, (4, 128))
    fp8_scales = tl.maximum(
        tl.max(tl.abs(latent_2d), axis=1, keep_dims=True) * (1.0 / 448.0),
        1.1754944e-38,
    )
    quantized = tl.reshape(
        (latent_2d / fp8_scales).to(tl.float8e4nv),
        (_KVARN_NVFP4_LATENT_DIM_TL,),
    )
    output_record = output_ptr + row * 656
    output_fp8 = output_record.to(tl.pointer_type(tl.float8e4nv))
    output_f32 = output_record.to(tl.pointer_type(tl.float32))
    tl.store(output_fp8 + cols, quantized)
    scale_cols = tl.arange(0, 4)
    tl.store(
        output_f32 + _KVARN_NVFP4_LATENT_DIM_TL // 4 + scale_cols,
        tl.reshape(fp8_scales, (4,)),
    )

    rope_cols = tl.arange(0, _KVARN_NVFP4_ROPE_DIM_TL)
    raw_rope = tl.load(
        record + _KVARN_NVFP4_ROPE_OFFSET_TL + rope_cols,
        mask=~exact,
        other=0,
    )
    rope_q = tl.cast(raw_rope, tl.float8e4nv, bitcast=True).to(tl.float32)
    rope_scale = tl.load(
        f32_record + _KVARN_NVFP4_ROPE_SCALE_OFFSET_TL // 4,
        mask=~exact,
        other=0.0,
    )
    packed_rope = rope_q * rope_scale
    exact_rope = tl.load(
        rope_pool_ptr
        + safe_pool_slot * rope_pool_stride_s
        + token * rope_pool_stride_t
        + rope_cols,
        mask=exact,
        other=0.0,
    ).to(tl.float32)
    rope = tl.where(exact, exact_rope, packed_rope)
    output_rope = (output_record + 528).to(tl.pointer_type(tl.bfloat16))
    tl.store(output_rope + rope_cols, rope)


def stage_selected_kvarn_nvfp4_mla_fp8(
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    physical_slots: torch.Tensor,
    output_records: torch.Tensor,
    remapped: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Stage hybrid packed/exact selections as standard 656-byte FP8 records."""
    _validate_kvarn_nvfp4_geometry(kv_cache, config)
    if selected_indices.dtype != torch.int32 or not selected_indices.is_contiguous():
        raise ValueError("KVarN MLA selected indices must be contiguous int32")
    if (
        block_to_slot.dtype != torch.int32
        or not block_to_slot.is_contiguous()
        or block_to_slot.numel() < kv_cache.shape[0]
    ):
        raise ValueError("KVarN MLA block-to-slot map must cover the cache as int32")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")

    rows = selected_indices.numel()
    if (
        physical_slots.dtype != torch.int32
        or not physical_slots.is_contiguous()
        or physical_slots.numel() < rows
    ):
        raise ValueError(
            "KVarN MLA selected physical-slot workspace must be contiguous "
            f"int32 with at least {rows} entries"
        )
    if (
        remapped.dtype != torch.int32
        or not remapped.is_contiguous()
        or remapped.numel() < rows
    ):
        raise ValueError(
            "KVarN MLA selected remap workspace must be contiguous int32 "
            f"with at least {rows} entries"
        )
    if (
        output_records.dtype != torch.uint8
        or not output_records.is_contiguous()
        or output_records.shape != (rows, 656)
    ):
        raise ValueError(
            f"KVarN MLA selected FP8 workspace must have shape ({rows}, 656)"
        )
    if output_records.data_ptr() % 16:
        raise ValueError("KVarN MLA selected FP8 workspace must be 16-byte aligned")
    if rows == 0:
        return

    safe_physical = physical_slots.view(-1)[:rows]
    flat_remapped = remapped.view(-1)[:rows]
    _prepare_selected_kvarn_mla_fp8_kernel[(triton.cdiv(rows, 256),)](
        selected_indices.view(-1),
        safe_physical,
        flat_remapped,
        n_elements=rows,
        max_physical_slots=kv_cache.shape[0] * KVARN_NVFP4_GROUP,
    )
    _stage_selected_kvarn_nvfp4_mla_fp8_kernel[(rows,)](
        safe_physical,
        kv_cache.view(torch.uint8),
        block_to_slot,
        latent_pool,
        rope_pool,
        output_records,
        kv_cache.view(torch.uint8).stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        NUM_BLOCKS=kv_cache.shape[0],
        NUM_POOL_SLOTS=latent_pool.shape[0],
        num_warps=8,
    )


def stage_selected_kvarn_mla_fp8(
    selected_indices: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    physical_slots: torch.Tensor,
    output_records: torch.Tensor,
    remapped: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Stage selected packed/exact rows as standard SparkInfer FP8 records."""
    if _is_kvarn_nvfp4(config):
        stage_selected_kvarn_nvfp4_mla_fp8(
            selected_indices,
            kv_cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            physical_slots,
            output_records,
            remapped,
            config,
        )
        return
    try:
        from sparkinfer.attention.kvarn_mla import stage_k5_as_fp8_records
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "KVarN MLA requires SparkInfer with "
            "sparkinfer.attention.kvarn_mla.stage_k5_as_fp8_records"
        ) from exc

    if selected_indices.dtype != torch.int32 or not selected_indices.is_contiguous():
        raise ValueError("KVarN MLA selected indices must be contiguous int32")
    if block_to_slot.dtype != torch.int32 or not block_to_slot.is_contiguous():
        raise ValueError("KVarN MLA block-to-slot map must be contiguous int32")
    if block_to_slot.numel() < kv_cache.shape[0]:
        raise ValueError("KVarN MLA block-to-slot map is smaller than the cache")
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")

    rows = selected_indices.numel()
    if (
        physical_slots.dtype != torch.int32
        or not physical_slots.is_contiguous()
        or physical_slots.numel() < rows
    ):
        raise ValueError(
            "KVarN MLA selected physical-slot workspace must be contiguous "
            f"int32 with at least {rows} entries"
        )
    if (
        remapped.dtype != torch.int32
        or not remapped.is_contiguous()
        or remapped.numel() < rows
    ):
        raise ValueError(
            "KVarN MLA selected remap workspace must be contiguous int32 "
            f"with at least {rows} entries"
        )
    if (
        output_records.dtype != torch.uint8
        or not output_records.is_contiguous()
        or output_records.shape != (rows, 656)
    ):
        raise ValueError(
            f"KVarN MLA selected FP8 workspace must have shape ({rows}, 656)"
        )
    if output_records.data_ptr() % 16:
        raise ValueError("KVarN MLA selected FP8 workspace must be 16-byte aligned")
    if rows == 0:
        return

    safe_physical = physical_slots.view(-1)[:rows]
    flat_remapped = remapped.view(-1)[:rows]
    _prepare_selected_kvarn_mla_fp8_kernel[(triton.cdiv(rows, 256),)](
        selected_indices.view(-1),
        safe_physical,
        flat_remapped,
        n_elements=rows,
        max_physical_slots=kv_cache.shape[0] * config.group,
    )
    stage_k5_as_fp8_records(
        safe_physical,
        kv_cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        output_records,
    )


_DIRECT_PACKED_GROUP = 64
_DIRECT_PACKED_LATENT_DIM = 512
_DIRECT_PACKED_ROPE_DIM = 64
_DIRECT_PACKED_TOPK = 2048
_DIRECT_PACKED_SPLIT = 64
_DIRECT_PACKED_HEADS_PER_CTA = 16


@triton.jit
def _direct_packed_kvarn_mla_split_kernel(
    q_ptr,
    selected_ptr,
    valid_counts_ptr,
    cache_ptr,
    block_to_slot_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    split_output_ptr,
    split_lse_ptr,
    q_stride_m: tl.constexpr,
    q_stride_h: tl.constexpr,
    selected_stride_m: tl.constexpr,
    cache_stride_b: tl.constexpr,
    latent_pool_stride_s: tl.constexpr,
    latent_pool_stride_t: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    split_output_stride_m: tl.constexpr,
    split_output_stride_h: tl.constexpr,
    split_output_stride_s: tl.constexpr,
    split_lse_stride_m: tl.constexpr,
    split_lse_stride_h: tl.constexpr,
    split_lse_stride_s: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_POOL_SLOTS: tl.constexpr,
    S_COL_OFFSET: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    S_ROW_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
):
    row = tl.program_id(0)
    head_group = tl.program_id(1)
    split = tl.program_id(2)
    heads = head_group * HEADS_PER_CTA + tl.arange(0, HEADS_PER_CTA)
    valid_count = tl.load(valid_counts_ptr + row)
    count_ok = (valid_count >= 0) & (valid_count <= TOPK)
    valid_count = tl.where(count_ok, valid_count, 0)
    split_start = split * SPLIT_SIZE
    if split_start >= valid_count:
        tl.store(
            split_lse_ptr
            + row * split_lse_stride_m
            + heads * split_lse_stride_h
            + split * split_lse_stride_s,
            -float("inf"),
        )
        return

    candidates = split_start + tl.arange(0, SPLIT_SIZE)
    active = candidates < valid_count
    physical_slot = tl.load(
        selected_ptr + row * selected_stride_m + candidates,
        mask=active,
        other=-1,
    )
    active = active & (physical_slot >= 0) & (physical_slot < NUM_BLOCKS * GROUP)
    block = physical_slot // GROUP
    token = physical_slot % GROUP
    safe_block = tl.where(active, block, 0)
    pool_slot = tl.load(
        block_to_slot_ptr + safe_block,
        mask=active,
        other=-2,
    )
    packed = active & (pool_slot == -1)
    live = active & (pool_slot >= 0) & (pool_slot < NUM_POOL_SLOTS)
    active = packed | live
    safe_pool_slot = tl.where(live, pool_slot, 0)
    record = cache_ptr + safe_block[None, :] * cache_stride_b
    fp16_record = record.to(tl.pointer_type(tl.float16))
    # This scale depends only on the selected token. Keep it live across both
    # reconstruction passes rather than reloading it for every 32-D tile.
    s_row = tl.load(
        fp16_record + S_ROW_OFFSET // 2 + token[None, :],
        mask=packed[None, :],
        other=0.0,
    )

    logits = tl.zeros((HEADS_PER_CTA, SPLIT_SIZE), tl.float32)
    for d_start in range(0, LATENT_DIM, 32):
        dims = d_start + tl.arange(0, 32)
        q = tl.load(
            q_ptr + row * q_stride_m + heads[:, None] * q_stride_h + dims[None, :]
        )
        value_indices = dims[:, None] * GROUP + token[None, :]
        codes = _unpack_dense_bits(
            record,
            value_indices,
            packed[None, :],
            5,
        ).to(tl.float32)
        s_col = tl.load(
            fp16_record + S_COL_OFFSET // 2 + dims[:, None],
            mask=packed[None, :],
            other=0.0,
        )
        zp = tl.load(
            fp16_record + ZP_OFFSET // 2 + dims[:, None],
            mask=packed[None, :],
            other=0.0,
        )
        body = (codes * s_col + zp) * s_row
        exact = tl.load(
            latent_pool_ptr
            + safe_pool_slot[None, :] * latent_pool_stride_s
            + token[None, :] * latent_pool_stride_t
            + dims[:, None],
            mask=live[None, :],
            other=0.0,
        ).to(tl.float32)
        latent = tl.where(live[None, :], exact, body).to(tl.bfloat16)
        logits += tl.dot(q, latent)

    rope_dims = tl.arange(0, ROPE_DIM)
    q_rope = tl.load(
        q_ptr
        + row * q_stride_m
        + heads[:, None] * q_stride_h
        + LATENT_DIM
        + rope_dims[None, :]
    )
    body_rope_ptr = (record + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
    body_rope = tl.load(
        body_rope_ptr + token[None, :] * ROPE_DIM + rope_dims[:, None],
        mask=packed[None, :],
        other=0.0,
    )
    exact_rope = tl.load(
        rope_pool_ptr
        + safe_pool_slot[None, :] * rope_pool_stride_s
        + token[None, :] * rope_pool_stride_t
        + rope_dims[:, None],
        mask=live[None, :],
        other=0.0,
    )
    rope = tl.where(live[None, :], exact_rope, body_rope)
    logits += tl.dot(q_rope, rope)
    logits *= sm_scale
    logits = tl.where(active[None, :], logits, -float("inf"))

    has_values = tl.sum(active.to(tl.int32), axis=0) > 0
    local_max = tl.max(logits, axis=1)
    safe_max = tl.where(has_values, local_max, 0.0)
    numerators = tl.where(active[None, :], tl.exp(logits - safe_max[:, None]), 0.0)
    denominator = tl.sum(numerators, axis=1)
    probabilities = tl.where(
        denominator[:, None] > 0,
        numerators / denominator[:, None],
        0.0,
    ).to(tl.bfloat16)
    local_lse = tl.where(has_values, safe_max + tl.log(denominator), -float("inf"))
    tl.store(
        split_lse_ptr
        + row * split_lse_stride_m
        + heads * split_lse_stride_h
        + split * split_lse_stride_s,
        local_lse,
    )

    for d_start in range(0, LATENT_DIM, 32):
        dims = d_start + tl.arange(0, 32)
        value_indices = dims[:, None] * GROUP + token[None, :]
        codes = _unpack_dense_bits(
            record,
            value_indices,
            packed[None, :],
            5,
        ).to(tl.float32)
        s_col = tl.load(
            fp16_record + S_COL_OFFSET // 2 + dims[:, None],
            mask=packed[None, :],
            other=0.0,
        )
        zp = tl.load(
            fp16_record + ZP_OFFSET // 2 + dims[:, None],
            mask=packed[None, :],
            other=0.0,
        )
        body = (codes * s_col + zp) * s_row
        exact = tl.load(
            latent_pool_ptr
            + safe_pool_slot[None, :] * latent_pool_stride_s
            + token[None, :] * latent_pool_stride_t
            + dims[:, None],
            mask=live[None, :],
            other=0.0,
        ).to(tl.float32)
        values = tl.where(live[None, :], exact, body).to(tl.bfloat16)
        partial = tl.dot(probabilities, tl.trans(values))
        tl.store(
            split_output_ptr
            + row * split_output_stride_m
            + heads[:, None] * split_output_stride_h
            + split * split_output_stride_s
            + dims[None, :],
            partial,
        )


@triton.jit
def _direct_packed_kvarn_mla_merge_kernel(
    split_output_ptr,
    split_lse_ptr,
    output_ptr,
    output_lse_ptr,
    split_output_stride_m: tl.constexpr,
    split_output_stride_h: tl.constexpr,
    split_output_stride_s: tl.constexpr,
    split_lse_stride_m: tl.constexpr,
    split_lse_stride_h: tl.constexpr,
    split_lse_stride_s: tl.constexpr,
    output_stride_m: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_lse_stride_m: tl.constexpr,
    output_lse_stride_h: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    LATENT_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * HEADS_PER_CTA + tl.arange(0, HEADS_PER_CTA)
    splits = tl.arange(0, NUM_SPLITS)
    split_lse = tl.load(
        split_lse_ptr
        + row * split_lse_stride_m
        + heads[:, None] * split_lse_stride_h
        + splits[None, :] * split_lse_stride_s
    )
    global_max = tl.max(split_lse, axis=1)
    has_values = global_max != -float("inf")
    safe_max = tl.where(has_values, global_max, 0.0)
    weights = tl.where(
        split_lse != -float("inf"),
        tl.exp(split_lse - safe_max[:, None]),
        0.0,
    )
    denominator = tl.sum(weights, axis=1)
    normalized = tl.where(
        denominator[:, None] > 0,
        weights / denominator[:, None],
        0.0,
    )
    final_lse = tl.where(has_values, safe_max + tl.log(denominator), -float("inf"))
    tl.store(
        output_lse_ptr + row * output_lse_stride_m + heads * output_lse_stride_h,
        final_lse,
    )
    for d_start in range(0, LATENT_DIM, 32):
        dims = d_start + tl.arange(0, 32)
        partials = tl.load(
            split_output_ptr
            + row * split_output_stride_m
            + heads[:, None, None] * split_output_stride_h
            + splits[None, :, None] * split_output_stride_s
            + dims[None, None, :],
            mask=(split_lse != -float("inf"))[:, :, None],
            other=0.0,
        ).to(tl.float32)
        merged = tl.sum(partials * normalized[:, :, None], axis=1)
        tl.store(
            output_ptr
            + row * output_stride_m
            + heads[:, None] * output_stride_h
            + dims[None, :],
            merged,
        )


def _direct_packed_nonoverlapping(tensor: torch.Tensor) -> bool:
    required_span = 1
    dimensions = sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        if size > 1
    )
    for stride, size in dimensions:
        if stride < required_span:
            return False
        required_span = stride * size
    return True


def _direct_packed_unit_inner_nonoverlapping(tensor: torch.Tensor) -> bool:
    return tensor.stride(-1) == 1 and _direct_packed_nonoverlapping(tensor)


def _require_direct_packed_kvarn_mla_sm120(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("Direct packed KVarN MLA requires one CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise ValueError(
            f"Direct packed KVarN MLA requires SM120; got CUDA capability {capability}"
        )


def direct_packed_kvarn_mla_decode(
    q: torch.Tensor,
    selected_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    kv_cache: torch.Tensor,
    block_to_slot: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor,
    *,
    sm_scale: float,
    candidate_envelope: int,
    config: KVarNMLAConfig,
    native_decode: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
    static_inputs_validated: bool = False,
    exact_pool_only: bool = False,
    fuse_kvarn_hadamard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode directly from exact K5/G64 pages and live side pools."""
    # The backend owns graph-static query/index/scratch views. Once one row
    # geometry has passed the checks below, subsequent graph replays can enter
    # the required native kernel without repeating dozens of Python tensor
    # predicates or querying the CUDA capability at every MLA layer.
    if static_inputs_validated:
        if native_decode is None:
            raise RuntimeError("Validated direct KVarN decode requires a native kernel")
        rows, heads, _ = q.shape
        if candidate_envelope == 0:
            output[:rows].zero_()
            output_lse[:rows].fill_(-float("inf"))
            return output[:rows], output_lse[:rows]
        if heads != 64:
            raise RuntimeError("Native packed K5 decode requires 64 gathered heads")
        return native_decode(
            q,
            selected_indices,
            valid_counts,
            kv_cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            split_output,
            split_lse,
            valid_counts[:1],
            output,
            output_lse,
            sm_scale=sm_scale,
            candidate_envelope=candidate_envelope,
            exact_pool_only=exact_pool_only,
            fuse_kvarn_hadamard=fuse_kvarn_hadamard,
        )
    geometry = (
        config.group,
        config.latent_dim,
        config.rope_dim,
        config.bits,
        config.tile_bytes,
    )
    if geometry != (64, 512, 64, 5, 30_848):
        raise ValueError(
            "Direct packed KVarN MLA requires K5/G64 geometry (64,512,64,5,30848)"
        )
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("Direct packed KVarN MLA scale must be finite and positive")
    if (
        q.dtype != torch.bfloat16
        or q.ndim != 3
        or not _direct_packed_unit_inner_nonoverlapping(q)
    ):
        raise ValueError(
            "Direct packed KVarN MLA query must be rank-3 BF16 with a "
            "non-overlapping unit-inner layout"
        )
    rows, heads, width = q.shape
    if rows not in (1, 4, 16) or heads not in (16, 64) or width != 576:
        raise ValueError(
            "Direct packed KVarN MLA supports only M1/M4/M16, H16/H64, D576"
        )
    if (
        selected_indices.dtype != torch.int32
        or selected_indices.shape != (rows, _DIRECT_PACKED_TOPK)
        or not selected_indices.is_contiguous()
    ):
        raise ValueError(
            "Direct packed KVarN MLA selected indices must be contiguous "
            f"int32[{rows},{_DIRECT_PACKED_TOPK}]"
        )
    if (
        valid_counts.dtype != torch.int32
        or valid_counts.shape != (rows,)
        or not valid_counts.is_contiguous()
    ):
        raise ValueError(
            f"Direct packed KVarN MLA valid counts must be contiguous int32[{rows}]"
        )
    if not 0 <= candidate_envelope <= _DIRECT_PACKED_TOPK:
        raise ValueError(
            "Direct packed KVarN MLA candidate envelope must be between 0 "
            f"and {_DIRECT_PACKED_TOPK}, got {candidate_envelope}"
        )
    num_splits = (candidate_envelope + _DIRECT_PACKED_SPLIT - 1) // _DIRECT_PACKED_SPLIT
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or tuple(kv_cache.shape[1:]) != (1, config.tile_bytes)
        or kv_cache.stride(0) != config.tile_bytes
        or not kv_cache.is_contiguous()
    ):
        raise ValueError(
            "Direct packed KVarN MLA cache must be contiguous uint8[blocks,1,30848]"
        )
    blocks = kv_cache.shape[0]
    if (
        block_to_slot.dtype != torch.int32
        or block_to_slot.shape != (blocks,)
        or not block_to_slot.is_contiguous()
    ):
        raise ValueError(
            "Direct packed KVarN MLA block-to-slot map must contain one "
            "contiguous int32 entry per cache block"
        )
    if (
        latent_pool.dtype != torch.float8_e4m3fn
        or latent_pool.ndim != 3
        or tuple(latent_pool.shape[1:]) != (64, 512)
        or not latent_pool.is_contiguous()
    ):
        raise ValueError(
            "Direct packed KVarN MLA live latent pool must be contiguous "
            "float8_e4m3fn[slots,64,512]"
        )
    if (
        rope_pool.dtype != torch.bfloat16
        or rope_pool.ndim != 3
        or tuple(rope_pool.shape[1:]) != (64, 64)
        or not rope_pool.is_contiguous()
        or rope_pool.shape[0] != latent_pool.shape[0]
    ):
        raise ValueError(
            "Direct packed KVarN MLA live RoPE pool must be contiguous "
            "BF16[slots,64,64] with matching slot capacity"
        )
    if (
        split_output.dtype != torch.bfloat16
        or split_output.ndim != 4
        or split_output.shape[0] < rows
        or split_output.shape[1] != heads
        or split_output.shape[2] < num_splits
        or split_output.shape[3] != 512
        or not _direct_packed_unit_inner_nonoverlapping(split_output)
    ):
        raise ValueError(
            "Direct packed KVarN MLA split output scratch has incompatible geometry"
        )
    if (
        split_lse.dtype != torch.float32
        or split_lse.ndim != 3
        or split_lse.shape[0] < rows
        or split_lse.shape[1] != heads
        or split_lse.shape[2] < num_splits
        or not _direct_packed_nonoverlapping(split_lse)
    ):
        raise ValueError(
            "Direct packed KVarN MLA split LSE scratch has incompatible geometry"
        )
    if (
        output.dtype != torch.bfloat16
        or output.ndim != 3
        or output.shape[0] < rows
        or tuple(output.shape[1:]) != (heads, 512)
        or not _direct_packed_unit_inner_nonoverlapping(output)
    ):
        raise ValueError("Direct packed KVarN MLA output has incompatible geometry")
    if (
        output_lse.dtype != torch.float32
        or output_lse.ndim != 2
        or output_lse.shape[0] < rows
        or output_lse.shape[1] != heads
        or not _direct_packed_nonoverlapping(output_lse)
    ):
        raise ValueError("Direct packed KVarN MLA LSE has incompatible geometry")
    tensors = (
        q,
        selected_indices,
        valid_counts,
        kv_cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        split_output,
        split_lse,
        output,
        output_lse,
    )
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("Direct packed KVarN MLA tensors must share one CUDA device")
    _require_direct_packed_kvarn_mla_sm120(q.device)

    if num_splits == 0:
        output[:rows].zero_()
        output_lse[:rows].fill_(-float("inf"))
        return output[:rows], output_lse[:rows]

    if native_decode is not None:
        if heads != 64:
            raise RuntimeError("Native packed K5 decode requires 64 gathered heads")
        return native_decode(
            q,
            selected_indices,
            valid_counts,
            kv_cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            split_output,
            split_lse,
            valid_counts[:1],
            output,
            output_lse,
            sm_scale=sm_scale,
            candidate_envelope=candidate_envelope,
            exact_pool_only=exact_pool_only,
            fuse_kvarn_hadamard=fuse_kvarn_hadamard,
        )

    split_grid = (
        rows,
        heads // _DIRECT_PACKED_HEADS_PER_CTA,
        num_splits,
    )
    _direct_packed_kvarn_mla_split_kernel[split_grid](
        q,
        selected_indices,
        valid_counts,
        kv_cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        split_output,
        split_lse,
        q.stride(0),
        q.stride(1),
        selected_indices.stride(0),
        kv_cache.stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        split_output.stride(0),
        split_output.stride(1),
        split_output.stride(2),
        split_lse.stride(0),
        split_lse.stride(1),
        split_lse.stride(2),
        sm_scale=sm_scale,
        NUM_BLOCKS=blocks,
        NUM_POOL_SLOTS=latent_pool.shape[0],
        S_COL_OFFSET=config.latent_s_col_offset,
        ZP_OFFSET=config.latent_zp_offset,
        S_ROW_OFFSET=config.latent_s_row_offset,
        ROPE_OFFSET=config.rope_offset,
        GROUP=_DIRECT_PACKED_GROUP,
        LATENT_DIM=_DIRECT_PACKED_LATENT_DIM,
        ROPE_DIM=_DIRECT_PACKED_ROPE_DIM,
        TOPK=_DIRECT_PACKED_TOPK,
        SPLIT_SIZE=_DIRECT_PACKED_SPLIT,
        HEADS_PER_CTA=_DIRECT_PACKED_HEADS_PER_CTA,
        num_warps=8,
    )
    merge_grid = (rows, heads // _DIRECT_PACKED_HEADS_PER_CTA)
    _direct_packed_kvarn_mla_merge_kernel[merge_grid](
        split_output,
        split_lse,
        output,
        output_lse,
        split_output.stride(0),
        split_output.stride(1),
        split_output.stride(2),
        split_lse.stride(0),
        split_lse.stride(1),
        split_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output_lse.stride(0),
        output_lse.stride(1),
        HEADS_PER_CTA=_DIRECT_PACKED_HEADS_PER_CTA,
        NUM_SPLITS=num_splits,
        LATENT_DIM=_DIRECT_PACKED_LATENT_DIM,
        num_warps=8,
    )
    return output[:rows], output_lse[:rows]
