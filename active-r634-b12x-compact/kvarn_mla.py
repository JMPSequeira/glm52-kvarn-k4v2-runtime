# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVarN K5 cache operations for MLA latent caches."""

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
from vllm.v1.attention.ops.triton_kvarn_sinkhorn import kvarn_sinkhorn_triton
from vllm.logger import init_logger

_logger = init_logger(__name__)
_pack_err_diag_count = [0]


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
def _serialize_kvarn_mla_blocks_kernel(
    balanced_ptr,
    s_col_ptr,
    s_row_ptr,
    rope_pool_ptr,
    block_ids_ptr,
    pool_slots_ptr,
    cache_ptr,
    balanced_stride_n: tl.constexpr,
    balanced_stride_r: tl.constexpr,
    s_col_stride_n: tl.constexpr,
    s_row_stride_n: tl.constexpr,
    rope_pool_stride_s: tl.constexpr,
    rope_pool_stride_t: tl.constexpr,
    cache_stride_b: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BITS: tl.constexpr,
    PACKED_ROW_BYTES: tl.constexpr,
    S_COL_OFFSET: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    S_ROW_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
    AFFINE_REFIT: tl.constexpr,
):
    program = tl.program_id(0)
    tile = program // LATENT_DIM
    row = program % LATENT_DIM
    tokens = tl.arange(0, GROUP)
    values = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + tokens
    ).to(tl.float32)

    qmax: tl.constexpr = (1 << BITS) - 1
    lower = tl.min(values, axis=0)
    upper = tl.max(values, axis=0)
    quant_scale = tl.maximum((upper - lower) / qmax, 1e-10)
    scale = quant_scale
    zero = lower
    codes = _quantize_rtn(values, lower, quant_scale, qmax)

    if AFFINE_REFIT:
        token_scales = tl.load(s_col_ptr + tile * s_col_stride_n + tokens).to(
            tl.float32
        )
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
    row_scale = tl.load(s_row_ptr + tile * s_row_stride_n + row).to(tl.float32)
    fp16_record = record.to(tl.pointer_type(tl.float16))
    tl.store(fp16_record + S_COL_OFFSET // 2 + row, row_scale * scale)
    tl.store(fp16_record + ZP_OFFSET // 2 + row, row_scale * zero)

    packed_offsets = tl.arange(0, 64)
    packed_mask = packed_offsets < PACKED_ROW_BYTES
    bit_offsets = packed_offsets * 8
    source = bit_offsets // BITS
    shifts = bit_offsets % BITS

    source0 = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + source,
        mask=packed_mask & (source < GROUP),
        other=lower,
    ).to(tl.float32)
    source1 = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + source + 1,
        mask=packed_mask & (source + 1 < GROUP),
        other=lower,
    ).to(tl.float32)
    source2 = tl.load(
        balanced_ptr + tile * balanced_stride_n + row * balanced_stride_r + source + 2,
        mask=packed_mask & (source + 2 < GROUP),
        other=lower,
    ).to(tl.float32)
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
        s_col_ptr + tile * s_col_stride_n + shared_offsets,
        mask=shared_mask,
    )
    tl.store(
        fp16_record + S_ROW_OFFSET // 2 + shared_offsets,
        token_scales,
        mask=(row == 0) & shared_mask,
    )

    pool_slot = tl.load(pool_slots_ptr + tile)
    rope = tl.load(
        rope_pool_ptr
        + pool_slot * rope_pool_stride_s
        + row * rope_pool_stride_t
        + shared_offsets,
        mask=(row < GROUP) & (shared_offsets < ROPE_DIM),
    )
    rope_record = (record + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
    tl.store(
        rope_record + row * ROPE_DIM + shared_offsets,
        rope,
        mask=(row < GROUP) & (shared_offsets < ROPE_DIM),
    )


def _serialize_kvarn_mla_blocks(
    kv_cache: torch.Tensor,
    rope_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    balanced: torch.Tensor,
    s_col: torch.Tensor,
    s_row: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    packed_row_bytes = config.latent_packed_bytes // config.latent_dim
    _serialize_kvarn_mla_blocks_kernel[(block_ids.numel() * config.latent_dim,)](
        balanced,
        s_col,
        s_row,
        rope_pool,
        block_ids,
        pool_slots,
        kv_cache.view(torch.uint8),
        balanced.stride(0),
        balanced.stride(1),
        s_col.stride(0),
        s_row.stride(0),
        rope_pool.stride(0),
        rope_pool.stride(1),
        kv_cache.view(torch.uint8).stride(0),
        GROUP=config.group,
        LATENT_DIM=config.latent_dim,
        ROPE_DIM=config.rope_dim,
        BITS=config.bits,
        PACKED_ROW_BYTES=packed_row_bytes,
        S_COL_OFFSET=config.latent_s_col_offset,
        ZP_OFFSET=config.latent_zp_offset,
        S_ROW_OFFSET=config.latent_s_row_offset,
        ROPE_OFFSET=config.rope_offset,
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
    """Quantize complete latent tiles and serialize exact BF16 RoPE tiles."""
    if block_ids.numel() == 0:
        return
    latent = latent_pool.index_select(0, pool_slots).float()
    latent_tiles = latent.transpose(1, 2).contiguous()
    balanced, s_col, s_row = kvarn_sinkhorn_triton(
        latent_tiles, iterations=config.sinkhorn_iters
    )
    quantile = float(os.environ.get("KVARN_RTN_QUANTILE", "") or 0.0)
    if config.bits == 5 and quantile <= 0.0:
        _serialize_kvarn_mla_blocks(
            kv_cache,
            rope_pool,
            block_ids,
            pool_slots,
            balanced,
            s_col,
            s_row,
            config,
        )
        return
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

    _diag_limit = int(os.environ.get("KVARN_MLA_DIAG_PACK_ERR", "0") or 0)
    if _diag_limit > 0 and _pack_err_diag_count[0] < _diag_limit:
        _pack_err_diag_count[0] += 1
        _ps = int(pool_slots.max().item()) + 1
        _re_lat = torch.zeros(
            _ps, latent_pool.shape[1], latent_pool.shape[2],
            dtype=latent_pool.dtype, device=latent_pool.device,
        )
        _re_rope = torch.zeros(
            _ps, rope_pool.shape[1], rope_pool.shape[2],
            dtype=rope_pool.dtype, device=rope_pool.device,
        )
        rehydrate_kvarn_mla_blocks(
            kv_cache, _re_lat, _re_rope, block_ids, pool_slots, config
        )
        _src = latent_pool.index_select(0, pool_slots).float()
        _sn = _src.norm().item()
        _rn = _re_lat.index_select(0, pool_slots).float().norm().item()
        _err = (_re_lat.float() - _src).norm() / max(_sn, 1e-9)
        _rec_sum = int(
            kv_cache.view(torch.uint8)
            .reshape(kv_cache.shape[0], -1)
            .index_select(0, block_ids)
            .sum()
            .item()
        )
        _dim = _src.reshape(-1, _src.shape[-1])
        _rng = (_dim.max(0).values - _dim.min(0).values)
        _std = _dim.std(0).clamp_min(1e-6)
        _ratio = float((_rng / _std).mean().item())
        _kurt = float(
            (((_dim - _dim.mean(0)) / _std) ** 4).mean().item()
        )
        _pack_dump = os.environ.get("KVARN_MLA_DIAG_PACK_DUMP", "")
        if (
            _pack_dump
            and block_ids.numel() >= 4
            and sum(1 for _ in [0]) == 1
            and globals().setdefault("_PACK_DUMP_COUNT", [0])[0]
            < int(os.environ.get("KVARN_MLA_DIAG_PACK_DUMP_MAX", "400"))
        ):
            globals()["_PACK_DUMP_COUNT"][0] += 1
            import os as _os
            _dump_path = _os.path.join(
                _pack_dump,
                f"kvarn_pack_{config.bits}b_{block_ids[0].item()}-{block_ids[-1].item()}_"
                f"{id(kv_cache) % 100000}.pt",
            )
            torch.save(
                {
                    "block_ids": block_ids.detach().cpu(),
                    "pool_slots": pool_slots.detach().cpu(),
                    "records": (
                        kv_cache.view(torch.uint8)
                        .reshape(kv_cache.shape[0], -1)
                        .index_select(0, block_ids)
                        .detach()
                        .cpu()
                    ),
                    "bits": config.bits,
                },
                _dump_path,
            )
            _logger.warning("KVarN pack dump -> %s", _dump_path)
        if False:  # legacy conditional dump below disabled under pack-dump mode        if _rn < _sn / 4 and _pack_err_diag_count[0] > 20:
            import os as _os
            _dump_dir = _os.environ.get("KVARN_MLA_DIAG_DUMP_DIR", "/tmp")
            _dump_path = _os.path.join(
                _dump_dir,
                f"kvarn_badpack_{_pack_err_diag_count[0]}_{id(kv_cache)%100000}.pt",
            )
            if not _os.path.exists(_dump_path):
                torch.save(
                    {
                        "block_ids": block_ids.detach().cpu(),
                        "pool_slots": pool_slots.detach().cpu(),
                        "src_latent": _src.detach().cpu(),
                        "src_rope": rope_pool.index_select(0, pool_slots)
                        .detach()
                        .cpu(),
                        "record": kv_cache.view(torch.uint8)
                        .reshape(kv_cache.shape[0], -1)
                        .index_select(0, block_ids)
                        .detach()
                        .cpu(),
                        "bits": config.bits,
                        "rel_l2": _err.item(),
                    },
                    _dump_path,
                )
                _logger.warning("KVarN bad-pack dumped to %s", _dump_path)
        _logger.warning(
            "KVarN pack-err diag #%d bits=%d blocks=%d rel_l2=%.4f src_norm=%.4f re_norm=%.4f rec_sum=%d range/std=%.1f kurt=%.1f pool_dtype=%s",
            _pack_err_diag_count[0],
            config.bits,
            num_blocks,
            _err.item(),
            _sn,
            _rn,
            _rec_sum,
            _ratio,
            _kurt,
            latent_pool.dtype,
        )


@triton.jit
def _pack_dump_placeholder():
    pass

    _kvarn_pack_dump_at_exit(kv_cache, block_ids, pool_slots, config)

def _kvarn_pack_dump_at_exit(
    kv_cache, block_ids, pool_slots, config
):
    _pack_dump = os.environ.get("KVARN_MLA_DIAG_PACK_DUMP", "")
    if (
        _pack_dump
        and block_ids.numel() >= 4
        and globals().setdefault("_PACK_DUMP_COUNT", [0])[0]
        < int(os.environ.get("KVARN_MLA_DIAG_PACK_DUMP_MAX", "400"))
    ):
        globals()["_PACK_DUMP_COUNT"][0] += 1
        _dump_path = os.path.join(
            _pack_dump,
            f"kvarn_pack_{config.bits}b_{block_ids[0].item()}-{block_ids[-1].item()}_"
            f"{id(kv_cache) % 100000}.pt",
        )
        torch.save(
            {
                "block_ids": block_ids.detach().cpu(),
                "pool_slots": pool_slots.detach().cpu(),
                "records": (
                    kv_cache.view(torch.uint8)
                    .reshape(kv_cache.shape[0], -1)
                    .index_select(0, block_ids)
                    .detach()
                    .cpu()
                ),
                "bits": config.bits,
            },
            _dump_path,
        )
        _logger.warning("KVarN pack dump #%d -> %s", globals()["_PACK_DUMP_COUNT"][0], _dump_path)


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
    exact = (
        valid_block & (pool_slot >= 0) & (pool_slot < NUM_POOL_SLOTS)
    )
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
    if (
        selected_indices.dtype != torch.int32
        or not selected_indices.is_contiguous()
    ):
        raise ValueError(
            "KVarN MLA selected indices must be contiguous int32"
        )
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
def _rehydrate_kvarn_mla_blocks_kernel(
    cache_ptr,
    block_ids_ptr,
    pool_slots_ptr,
    latent_pool_ptr,
    rope_pool_ptr,
    cache_stride_b,
    latent_pool_stride_s,
    latent_pool_stride_t,
    rope_pool_stride_s,
    rope_pool_stride_t,
    BLOCK_L: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUP: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BITS: tl.constexpr,
    S_COL_OFFSET: tl.constexpr,
    ZP_OFFSET: tl.constexpr,
    S_ROW_OFFSET: tl.constexpr,
    ROPE_OFFSET: tl.constexpr,
):
    """Rebuild exact pool rows for one packed block from its paged record.

    Inverse of ``pack_kvarn_mla_blocks``: dequantizes the packed latent tile
    and copies the serialized RoPE rows back into the exact side pool so a
    cache-hit block that re-enters ownership reads its own KV instead of a
    recycled slot's previous occupant.
    """
    entry = tl.program_id(0)
    token = tl.program_id(1)
    block = tl.load(block_ids_ptr + entry)
    slot = tl.load(pool_slots_ptr + entry)
    record = cache_ptr + block * cache_stride_b

    cols = tl.arange(0, BLOCK_L)
    latent_mask = cols < LATENT_DIM
    q = _unpack_dense_bits(
        record, cols * GROUP + token, latent_mask, BITS
    ).to(tl.float32)
    fp16_record = record.to(tl.pointer_type(tl.float16))
    s_col = tl.load(fp16_record + S_COL_OFFSET // 2 + cols, mask=latent_mask, other=0.0)
    zp = tl.load(fp16_record + ZP_OFFSET // 2 + cols, mask=latent_mask, other=0.0)
    s_row = tl.load(fp16_record + S_ROW_OFFSET // 2 + token)
    latent = (q * s_col + zp) * s_row
    tl.store(
        latent_pool_ptr
        + slot * latent_pool_stride_s
        + token * latent_pool_stride_t
        + cols,
        latent.to(latent_pool_ptr.dtype.element_ty),
        mask=latent_mask,
    )

    rope_cols = tl.arange(0, BLOCK_R)
    rope_mask = rope_cols < ROPE_DIM
    body_rope = tl.load(
        (record + ROPE_OFFSET).to(tl.pointer_type(tl.bfloat16))
        + token * ROPE_DIM
        + rope_cols,
        mask=rope_mask,
        other=0.0,
    )
    tl.store(
        rope_pool_ptr
        + slot * rope_pool_stride_s
        + token * rope_pool_stride_t
        + rope_cols,
        body_rope,
        mask=rope_mask,
    )


def rehydrate_kvarn_mla_blocks(
    kv_cache: torch.Tensor,
    latent_pool: torch.Tensor,
    rope_pool: torch.Tensor,
    block_ids: torch.Tensor,
    pool_slots: torch.Tensor,
    config: KVarNMLAConfig,
) -> None:
    """Restore exact-pool rows for packed blocks (paged record -> pool slot)."""
    if block_ids.numel() == 0:
        return
    if latent_pool.shape[0] != rope_pool.shape[0]:
        raise ValueError("KVarN MLA exact latent/RoPE pools must have equal slots")
    if block_ids.dtype != torch.long or pool_slots.dtype != torch.long:
        raise ValueError("KVarN MLA rehydrate index buffers must be int64")
    if not block_ids.is_contiguous() or not pool_slots.is_contiguous():
        raise ValueError("KVarN MLA rehydrate index buffers must be contiguous")
    _rh_dump_dir = os.environ.get("KVARN_MLA_DIAG_REHYDR_DUMP", "")
    if _rh_dump_dir and block_ids.numel() >= 8:
        _pre_pool = latent_pool.index_select(0, pool_slots).detach().cpu().clone()
        _pre_rope = rope_pool.index_select(0, pool_slots).detach().cpu().clone()
        _rec = (
            kv_cache.view(torch.uint8)
            .reshape(kv_cache.shape[0], -1)
            .index_select(0, block_ids)
            .detach()
            .cpu()
            .clone()
        )
        _path = os.path.join(
            os.environ["KVARN_MLA_DIAG_REHYDR_DUMP"],
            f"kvarn_rehydrate_{block_ids.numel()}blk_{id(latent_pool) % 100000}.pt",
        )
        torch.save(
            {
                "block_ids": block_ids.detach().cpu(),
                "pool_slots": pool_slots.detach().cpu(),
                "pre_pool": _pre_pool,
                "pre_rope": _pre_rope,
                "records": _rec,
                "bits": config.bits,
                "tile_bytes": config.tile_bytes,
                "pool_shape": tuple(latent_pool.shape),
                "pool_strides": tuple(latent_pool.stride()),
                "kv_row_stride": kv_cache.view(torch.uint8).stride(0),
                "pool_id": id(latent_pool),
            },
            _path,
        )
        _logger.warning("KVarN rehydrate dump -> %s", _path)
    cache_bytes = kv_cache.view(torch.uint8)
    _rehydrate_kvarn_mla_blocks_kernel[
        (block_ids.numel(), config.group)
    ](
        cache_bytes,
        block_ids,
        pool_slots,
        latent_pool,
        rope_pool,
        cache_bytes.stride(0),
        latent_pool.stride(0),
        latent_pool.stride(1),
        rope_pool.stride(0),
        rope_pool.stride(1),
        BLOCK_L=triton.next_power_of_2(config.latent_dim),
        BLOCK_R=triton.next_power_of_2(config.rope_dim),
        GROUP=config.group,
        LATENT_DIM=config.latent_dim,
        ROPE_DIM=config.rope_dim,
        BITS=config.bits,
        S_COL_OFFSET=config.latent_s_col_offset,
        ZP_OFFSET=config.latent_zp_offset,
        S_ROW_OFFSET=config.latent_s_row_offset,
        ROPE_OFFSET=config.rope_offset,
        num_warps=4,
    )
    if (
        _rh_dump_dir
        and block_ids.numel() >= 8
    ):
        torch.cuda.synchronize()
        torch.save(
            {
                "post_pool": latent_pool.detach().cpu().clone(),
                "post_rope": rope_pool.detach().cpu().clone(),
                "block_ids": block_ids.detach().cpu(),
                "pool_slots": pool_slots.detach().cpu(),
                "records": _rec,
                "bits": config.bits,
                "pool_id": id(latent_pool),
            },
            _path.replace(".pt", "_post.pt"),
        )
        _logger.warning("KVarN rehydrate POST dump captured")


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
        raise ValueError(
            "KVarN MLA physical index buffers must be contiguous int32"
        )
    rows = selected_indices.numel()
    if rows > remapped.numel():
        raise ValueError(
            f"KVarN MLA remap workspace has {remapped.numel()} entries, "
            f"requires {rows}"
        )
    if rows == 0:
        return
    _copy_bounded_physical_indices_kernel[(triton.cdiv(rows, 256),)](
        selected_indices.view(-1),
        remapped.view(-1),
        n_elements=rows,
        max_physical_slots=max_physical_slots,
    )

@triton.jit
def _compact_physical_slots_kernel(
    block_table_ptr,
    cu_seq_lens_ptr,
    output_ptr,
    block_table_stride,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(cu_seq_lens_ptr + request)
    end = tl.load(cu_seq_lens_ptr + request + 1)
    mask = offsets < end - start
    blocks = tl.load(
        block_table_ptr
        + request * block_table_stride
        + offsets // PAGE_SIZE,
        mask=mask,
        other=0,
    )
    physical_slots = blocks * PAGE_SIZE + offsets % PAGE_SIZE
    tl.store(output_ptr + start + offsets, physical_slots, mask=mask)


def build_compact_kvarn_mla_physical_slots(
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    batch_size: int,
    total_tokens: int,
    page_size: int,
) -> None:
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or cu_seq_lens.dtype != torch.int32
        or cu_seq_lens.ndim != 1
        or output.dtype != torch.int32
        or output.ndim != 1
    ):
        raise ValueError("KVarN MLA compact slot buffers must be int32")
    if not block_table.is_contiguous() or not cu_seq_lens.is_contiguous():
        raise ValueError("KVarN MLA compact slot inputs must be contiguous")
    if not output.is_contiguous() or output.numel() < total_tokens:
        raise ValueError("KVarN MLA compact slot output is too small")
    if batch_size <= 0 or cu_seq_lens.numel() < batch_size + 1:
        raise ValueError("KVarN MLA compact slot batch metadata is invalid")
    if total_tokens <= 0:
        return
    _compact_physical_slots_kernel[
        (batch_size, triton.cdiv(total_tokens, 256))
    ](
        block_table,
        cu_seq_lens,
        output,
        block_table.stride(0),
        PAGE_SIZE=page_size,
        BLOCK=256,
        num_warps=4,
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
        raise ValueError(
            f"KVarN MLA FP8 workspace must have shape ({page_rows}, 656)"
        )
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
    tl.store(
        physical_slots_ptr + offsets,
        tl.where(valid, physical, 0),
        mask=mask,
    )
    tl.store(remapped_ptr + offsets, tl.where(valid, offsets, -1), mask=mask)


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
    """Stage selected packed/exact K5 rows as standard FP8 records."""
    try:
        from sparkinfer.attention.kvarn_mla import stage_k5_as_fp8_records
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "KVarN MLA requires SparkInfer selected K5 staging support"
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
        raise ValueError("KVarN MLA selected physical-slot workspace is invalid")
    if (
        remapped.dtype != torch.int32
        or not remapped.is_contiguous()
        or remapped.numel() < rows
    ):
        raise ValueError("KVarN MLA selected remap workspace is invalid")
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
_DIRECT_PACKED_NUM_SPLITS = _DIRECT_PACKED_TOPK // _DIRECT_PACKED_SPLIT
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
    candidates = split * SPLIT_SIZE + tl.arange(0, SPLIT_SIZE)
    valid_count = tl.load(valid_counts_ptr + row)
    count_ok = (valid_count >= 0) & (valid_count <= TOPK)
    valid_count = tl.where(count_ok, valid_count, 0)
    active = candidates < valid_count
    physical_slot = tl.load(
        selected_ptr + row * selected_stride_m + candidates,
        mask=active,
        other=-1,
    )
    active = active & (physical_slot >= 0) & (
        physical_slot < NUM_BLOCKS * GROUP
    )
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

    logits = tl.zeros((HEADS_PER_CTA, SPLIT_SIZE), tl.float32)
    for d_start in range(0, LATENT_DIM, 32):
        dims = d_start + tl.arange(0, 32)
        q = tl.load(
            q_ptr
            + row * q_stride_m
            + heads[:, None] * q_stride_h
            + dims[None, :]
        )
        record = cache_ptr + safe_block[None, :] * cache_stride_b
        value_indices = dims[:, None] * GROUP + token[None, :]
        codes = _unpack_dense_bits(
            record,
            value_indices,
            packed[None, :],
            5,
        ).to(tl.float32)
        fp16_record = record.to(tl.pointer_type(tl.float16))
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
        s_row = tl.load(
            fp16_record + S_ROW_OFFSET // 2 + token[None, :],
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
    record = cache_ptr + safe_block[None, :] * cache_stride_b
    body_rope_ptr = (record + ROPE_OFFSET).to(
        tl.pointer_type(tl.bfloat16)
    )
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
    numerators = tl.where(
        active[None, :], tl.exp(logits - safe_max[:, None]), 0.0
    )
    denominator = tl.sum(numerators, axis=1)
    probabilities = tl.where(
        denominator[:, None] > 0,
        numerators / denominator[:, None],
        0.0,
    ).to(tl.bfloat16)
    local_lse = tl.where(
        has_values, safe_max + tl.log(denominator), -float("inf")
    )
    tl.store(
        split_lse_ptr
        + row * split_lse_stride_m
        + heads * split_lse_stride_h
        + split * split_lse_stride_s,
        local_lse,
    )

    for d_start in range(0, LATENT_DIM, 32):
        dims = d_start + tl.arange(0, 32)
        record = cache_ptr + safe_block[None, :] * cache_stride_b
        value_indices = dims[:, None] * GROUP + token[None, :]
        codes = _unpack_dense_bits(
            record,
            value_indices,
            packed[None, :],
            5,
        ).to(tl.float32)
        fp16_record = record.to(tl.pointer_type(tl.float16))
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
        s_row = tl.load(
            fp16_record + S_ROW_OFFSET // 2 + token[None, :],
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
    final_lse = tl.where(
        has_values, safe_max + tl.log(denominator), -float("inf")
    )
    tl.store(
        output_lse_ptr
        + row * output_lse_stride_m
        + heads * output_lse_stride_h,
        final_lse,
    )
    for d_start in range(0, LATENT_DIM, 32):
        dims = d_start + tl.arange(0, 32)
        partials = tl.load(
            split_output_ptr
            + row * split_output_stride_m
            + heads[:, None, None] * split_output_stride_h
            + splits[None, :, None] * split_output_stride_s
            + dims[None, None, :]
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
            "Direct packed KVarN MLA requires SM120; "
            f"got CUDA capability {capability}"
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
    exact_pool_only: bool = False,
    fuse_kvarn_hadamard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode directly from exact K5/G64 pages and live side pools."""
    if native_decode is not None:
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
            "Direct packed KVarN MLA requires K5/G64 geometry "
            "(64,512,64,5,30848)"
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
    if not 1 <= rows <= 16 or heads not in (16, 64) or width != 576:
        raise ValueError(
            "Direct packed KVarN MLA supports M in [1,16], H16/H64, D576"
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
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or tuple(kv_cache.shape[1:]) != (1, config.tile_bytes)
        or kv_cache.stride(0) != config.tile_bytes
        or not kv_cache.is_contiguous()
    ):
        raise ValueError(
            "Direct packed KVarN MLA cache must be contiguous "
            "uint8[blocks,1,30848]"
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
        or split_output.shape[2] != _DIRECT_PACKED_NUM_SPLITS
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
        or tuple(split_lse.shape[1:]) != (heads, _DIRECT_PACKED_NUM_SPLITS)
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
        raise ValueError(
            "Direct packed KVarN MLA tensors must share one CUDA device"
        )
    _require_direct_packed_kvarn_mla_sm120(q.device)

    split_grid = (
        rows,
        heads // _DIRECT_PACKED_HEADS_PER_CTA,
        _DIRECT_PACKED_NUM_SPLITS,
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
        NUM_SPLITS=_DIRECT_PACKED_NUM_SPLITS,
        LATENT_DIM=_DIRECT_PACKED_LATENT_DIM,
        num_warps=8,
    )
    return output[:rows], output_lse[:rows]
