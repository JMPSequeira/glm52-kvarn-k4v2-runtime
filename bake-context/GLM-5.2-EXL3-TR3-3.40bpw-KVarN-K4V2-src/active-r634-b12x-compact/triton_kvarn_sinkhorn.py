# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused log-domain iterative variance-normalization for KVarN.

Matches the PyTorch reference in
``vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`` semantically —
same 16 alternating col/row std-normalization passes, same best-so-far
tracking via the imbalance metric, same clamps. One Triton program per
``[R, C]`` tile; the grid dim is the number of tiles in the batch.

For ``R = C = 128`` the full tile is 64 KB fp32 — fits in a single Triton
block's register/SMEM budget on current GPUs.
"""

from __future__ import annotations
import os

import torch

from vllm.triton_utils import tl, triton

_CLIP_STD_MIN = 1e-3
_CLIP_STD_MAX = 1e3
_LOG_S_MIN = -0.3
_LOG_S_MAX = 10.0
_DETERMINISTIC_SINKHORN_ENV = "VLLM_KVARN_DETERMINISTIC_SINKHORN"
_DETERMINISTIC_SINKHORN_RAW = os.environ.get(
    _DETERMINISTIC_SINKHORN_ENV, "0"
)
_DETERMINISTIC_SINKHORN = _DETERMINISTIC_SINKHORN_RAW == "1"
_SINKHORN_REDUCTION_WARPS = 1 if _DETERMINISTIC_SINKHORN else 4
_SINKHORN_MARKER_EMITTED = False


_G64_LOG_S_COL_OFFSET = 0
_G64_BEST_S_COL_OFFSET = 64
_G64_COL_STD_OFFSET = 128
_G64_LOG_S_ROW_OFFSET = 192
_G64_BEST_S_ROW_OFFSET = 704
_G64_ROW_STD_OFFSET = 1216
_G64_BEST_IMBALANCE_OFFSET = 1728
_G64_SCRATCH_WORDS = 1729


@triton.jit
def _sinkhorn_log_kernel(
    Tile_ptr,  # [N, R, C] fp32 input — rotated tile
    Balanced_ptr,  # [N, R, C] fp32 output
    SCol_ptr,  # [N, C] fp32 output (s_col, per-column)
    SRow_ptr,  # [N, R] fp32 output (s_row, per-row)
    # Strides
    stride_tn,
    stride_tr,
    stride_bn,
    stride_br,
    stride_sc_n,
    stride_sr_n,
    # Dims
    R: tl.constexpr,
    C: tl.constexpr,
    ITERATIONS: tl.constexpr,
    # Algorithm params (kept as tl.constexpr for the compiler)
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
):
    """One program per tile. Loads a R x C tile into registers, does
    ``ITERATIONS`` alternating col/row log-domain normalizations, tracks
    the best-so-far (lowest-imbalance) scales, and writes (balanced, s_col,
    s_row).
    """
    pid = tl.program_id(0)

    r_offs = tl.arange(0, R)
    c_offs = tl.arange(0, C)

    # Load tile [R, C] into registers
    tile_base = pid * stride_tn
    tile_ptrs = Tile_ptr + tile_base + r_offs[:, None] * stride_tr + c_offs[None, :]
    tile = tl.load(tile_ptrs).to(tl.float32)

    # log_s_col [C], log_s_row [R]; initialised at zero (exp = 1)
    log_s_col = tl.zeros([C], dtype=tl.float32)
    log_s_row = tl.zeros([R], dtype=tl.float32)

    # cur = tile / s_col / s_row = tile (with mu = 1 initially)
    cur = tile

    # ── initial imbalance score + best snapshot ───────────────────────────
    col_mean0 = tl.sum(cur, axis=0) / R
    col_var0 = tl.sum(cur * cur, axis=0) / R - col_mean0 * col_mean0
    col_std0 = tl.sqrt(tl.maximum(col_var0 * R / (R - 1), 0.0))
    row_mean0 = tl.sum(cur, axis=1) / C
    row_var0 = tl.sum(cur * cur, axis=1) / C - row_mean0 * row_mean0
    row_std0 = tl.sqrt(tl.maximum(row_var0 * C / (C - 1), 0.0))
    col_max0 = tl.max(col_std0)
    col_min0 = tl.maximum(tl.min(col_std0), 1e-8)
    row_max0 = tl.max(row_std0)
    row_min0 = tl.maximum(tl.min(row_std0), 1e-8)
    score_best = col_max0 / col_min0 + row_max0 / row_min0

    sc_best = tl.exp(log_s_col)
    sr_best = tl.exp(log_s_row)

    # ── iterations ────────────────────────────────────────────────────────
    for _ in tl.static_range(ITERATIONS):
        # Update column scales from cur's per-column std
        col_mean = tl.sum(cur, axis=0) / R
        col_var = tl.sum(cur * cur, axis=0) / R - col_mean * col_mean
        col_std = tl.sqrt(tl.maximum(col_var * R / (R - 1), 0.0))
        col_std_clipped = tl.maximum(tl.minimum(col_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_col = log_s_col + tl.log(col_std_clipped)
        log_s_col = tl.maximum(tl.minimum(log_s_col, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        # Update row scales from new cur's per-row std
        row_mean = tl.sum(cur, axis=1) / C
        row_var = tl.sum(cur * cur, axis=1) / C - row_mean * row_mean
        row_std = tl.sqrt(tl.maximum(row_var * C / (C - 1), 0.0))
        row_std_clipped = tl.maximum(tl.minimum(row_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_row = log_s_row + tl.log(row_std_clipped)
        log_s_row = tl.maximum(tl.minimum(log_s_row, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        # Imbalance score at this candidate
        col_mean_n = tl.sum(cur, axis=0) / R
        col_var_n = tl.sum(cur * cur, axis=0) / R - col_mean_n * col_mean_n
        col_std_n = tl.sqrt(tl.maximum(col_var_n * R / (R - 1), 0.0))
        row_mean_n = tl.sum(cur, axis=1) / C
        row_var_n = tl.sum(cur * cur, axis=1) / C - row_mean_n * row_mean_n
        row_std_n = tl.sqrt(tl.maximum(row_var_n * C / (C - 1), 0.0))
        col_max_n = tl.max(col_std_n)
        col_min_n = tl.maximum(tl.min(col_std_n), 1e-8)
        row_max_n = tl.max(row_std_n)
        row_min_n = tl.maximum(tl.min(row_std_n), 1e-8)
        score = col_max_n / col_min_n + row_max_n / row_min_n

        better = score <= score_best
        sc_best = tl.where(better, s_col_lin, sc_best)
        sr_best = tl.where(better, s_row_lin, sr_best)
        score_best = tl.where(better, score, score_best)

    # ── final: balanced = tile / sc_best / sr_best, write outputs ─────────
    balanced = tile / sc_best[None, :] / sr_best[:, None]
    bal_ptrs = (
        Balanced_ptr + pid * stride_bn + r_offs[:, None] * stride_br + c_offs[None, :]
    )
    tl.store(bal_ptrs, balanced)
    tl.store(SCol_ptr + pid * stride_sc_n + c_offs, sc_best)
    tl.store(SRow_ptr + pid * stride_sr_n + r_offs, sr_best)


@triton.jit
def _sinkhorn_g64_row_phase_kernel(
    Pool,
    PoolSlots,
    LogSCol,
    LogSRow,
    RowStd,
    pool_stride_s: tl.constexpr,
    pool_stride_t: tl.constexpr,
    pool_stride_d: tl.constexpr,
    scratch_stride_s: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    INITIAL: tl.constexpr,
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
):
    """Initialize row scores or perform one row-normalization phase."""
    pid = tl.program_id(0)
    tile = pid // R
    row = pid % R
    pool_slot = tl.load(PoolSlots + tile)
    cols = tl.arange(0, C)
    values = tl.load(
        Pool
        + pool_slot * pool_stride_s
        + cols * pool_stride_t
        + row * pool_stride_d
    ).to(tl.float32)

    if INITIAL:
        current = values
        tl.store(LogSRow + pool_slot * scratch_stride_s + row, 0.0)
    else:
        log_s_col = tl.load(LogSCol + pool_slot * scratch_stride_s + cols)
        log_s_row = tl.load(LogSRow + pool_slot * scratch_stride_s + row)
        current = values / tl.exp(log_s_col) / tl.exp(log_s_row)
        mean = tl.sum(current, axis=0) / C
        centered = current - mean
        std = tl.sqrt(tl.sum(centered * centered, axis=0) / (C - 1))
        std = tl.maximum(tl.minimum(std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_row += tl.log(std)
        log_s_row = tl.maximum(tl.minimum(log_s_row, LOG_S_MAX), LOG_S_MIN)
        tl.store(LogSRow + pool_slot * scratch_stride_s + row, log_s_row)
        current = values / tl.exp(log_s_col) / tl.exp(log_s_row)

    current_mean = tl.sum(current, axis=0) / C
    current_centered = current - current_mean
    current_std = tl.sqrt(
        tl.sum(current_centered * current_centered, axis=0) / (C - 1)
    )
    tl.store(RowStd + pool_slot * scratch_stride_s + row, current_std)


@triton.jit
def _sinkhorn_g64_column_phase_kernel(
    Pool,
    PoolSlots,
    LogSCol,
    LogSRow,
    BestSCol,
    BestSRow,
    ColStd,
    RowStd,
    BestImbalance,
    pool_stride_s: tl.constexpr,
    pool_stride_t: tl.constexpr,
    pool_stride_d: tl.constexpr,
    scratch_stride_s: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    COLS_PER_CHUNK: tl.constexpr,
    FIRST: tl.constexpr,
    NORMALIZE: tl.constexpr,
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
):
    """Score one candidate and optionally perform its column-normalization."""
    tile = tl.program_id(0)
    pool_slot = tl.load(PoolSlots + tile)
    rows = tl.arange(0, R)
    log_s_row = tl.load(LogSRow + pool_slot * scratch_stride_s + rows)
    s_row = tl.exp(log_s_row)
    col_min = 1.0e30
    col_max = 0.0

    for col_base in tl.static_range(0, C, COLS_PER_CHUNK):
        cols = col_base + tl.arange(0, COLS_PER_CHUNK)
        values = tl.load(
            Pool
            + pool_slot * pool_stride_s
            + cols[None, :] * pool_stride_t
            + rows[:, None] * pool_stride_d
        ).to(tl.float32)
        log_s_col = tl.load(
            LogSCol + pool_slot * scratch_stride_s + cols,
            mask=not FIRST,
            other=0.0,
        )
        current = values / tl.exp(log_s_col)[None, :] / s_row[:, None]
        mean = tl.sum(current, axis=0) / R
        centered = current - mean[None, :]
        std = tl.sqrt(tl.sum(centered * centered, axis=0) / (R - 1))
        tl.store(ColStd + pool_slot * scratch_stride_s + cols, std)
        col_min = tl.minimum(col_min, tl.min(std, axis=0))
        col_max = tl.maximum(col_max, tl.max(std, axis=0))

    row_std = tl.load(RowStd + pool_slot * scratch_stride_s + rows)
    score = col_max / tl.maximum(col_min, 1.0e-8)
    score += tl.max(row_std, axis=0) / tl.maximum(tl.min(row_std, axis=0), 1.0e-8)
    previous_best = tl.load(
        BestImbalance + pool_slot * scratch_stride_s,
        mask=not FIRST,
        other=float("inf"),
    )
    better = score <= previous_best
    tl.store(
        BestImbalance + pool_slot * scratch_stride_s,
        tl.where(better, score, previous_best),
    )
    tl.store(
        BestSRow + pool_slot * scratch_stride_s + rows,
        s_row,
        mask=better,
    )

    for col_base in tl.static_range(0, C, COLS_PER_CHUNK):
        cols = col_base + tl.arange(0, COLS_PER_CHUNK)
        log_s_col = tl.load(
            LogSCol + pool_slot * scratch_stride_s + cols,
            mask=not FIRST,
            other=0.0,
        )
        tl.store(
            BestSCol + pool_slot * scratch_stride_s + cols,
            tl.exp(log_s_col),
            mask=better,
        )
        if NORMALIZE:
            std = tl.load(ColStd + pool_slot * scratch_stride_s + cols)
            std = tl.maximum(tl.minimum(std, CLIP_STD_MAX), CLIP_STD_MIN)
            log_s_col += tl.log(std)
            log_s_col = tl.maximum(tl.minimum(log_s_col, LOG_S_MAX), LOG_S_MIN)
            tl.store(LogSCol + pool_slot * scratch_stride_s + cols, log_s_col)


@triton.jit
def _axis_std_kernel(
    Values,
    Output,
    stride_n,
    stride_outer,
    N_OUTER: tl.constexpr,
    WIDTH: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // N_OUTER
    outer = pid % N_OUTER
    offsets = tl.arange(0, WIDTH)
    if TRANSPOSED:
        pointers = Values + n * stride_n + offsets * stride_outer + outer
    else:
        pointers = Values + n * stride_n + outer * stride_outer + offsets
    values = tl.load(pointers).to(tl.float32)
    mean = tl.sum(values, axis=0) / WIDTH
    centered = values - mean
    std = tl.sqrt(tl.sum(centered * centered, axis=0) / (WIDTH - 1))
    tl.store(Output + pid, std)


@triton.jit
def _normalize_columns_kernel(
    Tiles,
    Current,
    LogSCol,
    LogSRow,
    stride_n,
    stride_r,
    R: tl.constexpr,
    C: tl.constexpr,
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    col = pid % C
    rows = tl.arange(0, R)
    pointers = Current + n * stride_n + rows * stride_r + col
    values = tl.load(pointers).to(tl.float32)
    mean = tl.sum(values, axis=0) / R
    centered = values - mean
    std = tl.sqrt(tl.sum(centered * centered, axis=0) / (R - 1))
    std = tl.maximum(tl.minimum(std, CLIP_STD_MAX), CLIP_STD_MIN)
    log_s_col = tl.load(LogSCol + n * C + col) + tl.log(std)
    log_s_col = tl.maximum(tl.minimum(log_s_col, LOG_S_MAX), LOG_S_MIN)
    tl.store(LogSCol + n * C + col, log_s_col)
    log_s_row = tl.load(LogSRow + n * R + rows)
    tile = tl.load(Tiles + n * stride_n + rows * stride_r + col).to(tl.float32)
    tl.store(pointers, tile / tl.exp(log_s_col) / tl.exp(log_s_row))


@triton.jit
def _normalize_rows_kernel(
    Tiles,
    Current,
    LogSCol,
    LogSRow,
    RowStd,
    stride_n,
    stride_r,
    R: tl.constexpr,
    C: tl.constexpr,
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // R
    row = pid % R
    cols = tl.arange(0, C)
    pointers = Current + n * stride_n + row * stride_r + cols
    values = tl.load(pointers).to(tl.float32)
    mean = tl.sum(values, axis=0) / C
    centered = values - mean
    std = tl.sqrt(tl.sum(centered * centered, axis=0) / (C - 1))
    std = tl.maximum(tl.minimum(std, CLIP_STD_MAX), CLIP_STD_MIN)
    log_s_row = tl.load(LogSRow + n * R + row) + tl.log(std)
    log_s_row = tl.maximum(tl.minimum(log_s_row, LOG_S_MAX), LOG_S_MIN)
    tl.store(LogSRow + n * R + row, log_s_row)
    log_s_col = tl.load(LogSCol + n * C + cols)
    tile = tl.load(Tiles + n * stride_n + row * stride_r + cols).to(tl.float32)
    current = tile / tl.exp(log_s_row) / tl.exp(log_s_col)
    tl.store(pointers, current)
    current_mean = tl.sum(current, axis=0) / C
    current_centered = current - current_mean
    current_std = tl.sqrt(tl.sum(current_centered * current_centered, axis=0) / (C - 1))
    tl.store(RowStd + n * R + row, current_std)


@triton.jit
def _update_best_kernel(
    ColStd,
    RowStd,
    LogSCol,
    LogSRow,
    BestImbalance,
    BestSCol,
    BestSRow,
    R: tl.constexpr,
    C: tl.constexpr,
):
    n = tl.program_id(0)
    rows = tl.arange(0, R)
    cols = tl.arange(0, C)
    row_std = tl.load(RowStd + n * R + rows)
    col_std = tl.load(ColStd + n * C + cols)
    imbalance = tl.max(row_std) / tl.maximum(tl.min(row_std), 1e-8)
    imbalance += tl.max(col_std) / tl.maximum(tl.min(col_std), 1e-8)
    best = tl.load(BestImbalance + n)
    better = imbalance <= best
    tl.store(BestImbalance + n, tl.where(better, imbalance, best))
    tl.store(
        BestSCol + n * C + cols,
        tl.exp(tl.load(LogSCol + n * C + cols)),
        mask=better,
    )
    tl.store(
        BestSRow + n * R + rows,
        tl.exp(tl.load(LogSRow + n * R + rows)),
        mask=better,
    )


@triton.jit
def _finalize_tiled_kernel(
    Tiles,
    Balanced,
    BestSCol,
    BestSRow,
    stride_n,
    stride_r,
    R: tl.constexpr,
    C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // R
    row = pid % R
    cols = tl.arange(0, C)
    tile = tl.load(Tiles + n * stride_n + row * stride_r + cols).to(tl.float32)
    s_col = tl.load(BestSCol + n * C + cols)
    s_row = tl.load(BestSRow + n * R + row)
    tl.store(
        Balanced + n * stride_n + row * stride_r + cols,
        tile / s_col / s_row,
    )


def kvarn_sinkhorn_g64_512(
    latent_pool: torch.Tensor,
    pool_slots: torch.Tensor,
    scratch_pool: torch.Tensor,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-shape Sinkhorn directly from exact-pool storage.

    ``scratch_pool`` aliases the retired slots' RoPE storage. Its contents must
    already have been copied to the final cache records on the same stream.
    Phases then read latent ``[slot, token=64, latent=512]`` storage directly,
    without gather/cast/transpose or a balanced fp32 materialization.
    """
    if latent_pool.ndim != 3 or latent_pool.shape[1:] != (64, 512):
        raise ValueError("fixed KVarN Sinkhorn requires a [slots, 64, 512] pool")
    if pool_slots.ndim != 1 or not pool_slots.is_contiguous():
        raise ValueError("KVarN retirement pool slots must be contiguous")
    if scratch_pool.ndim != 3 or scratch_pool.shape[0] != latent_pool.shape[0]:
        raise ValueError("KVarN retirement scratch must share exact-pool slots")
    if not scratch_pool.is_contiguous():
        raise ValueError("KVarN retirement scratch pool must be contiguous")
    if (
        scratch_pool.device != latent_pool.device
        or pool_slots.device != latent_pool.device
    ):
        raise ValueError("KVarN retirement inputs must reside on one device")
    if iterations <= 0:
        raise ValueError("KVarN Sinkhorn iterations must be positive")

    scratch_slot_bytes = scratch_pool.stride(0) * scratch_pool.element_size()
    required_scratch_bytes = _G64_SCRATCH_WORDS * torch.float32.itemsize
    if scratch_slot_bytes < required_scratch_bytes:
        raise ValueError(
            "KVarN retirement scratch slot has "
            f"{scratch_slot_bytes} bytes, requires {required_scratch_bytes}"
        )
    if scratch_pool.data_ptr() % torch.float32.itemsize or (
        scratch_slot_bytes % torch.float32.itemsize
    ):
        raise ValueError("KVarN retirement scratch must be fp32-aligned per slot")

    scratch_words = scratch_pool.view(torch.float32).reshape(
        scratch_pool.shape[0], -1
    )
    log_s_col = scratch_words[:, _G64_LOG_S_COL_OFFSET:_G64_BEST_S_COL_OFFSET]
    best_s_col = scratch_words[:, _G64_BEST_S_COL_OFFSET:_G64_COL_STD_OFFSET]
    col_std = scratch_words[:, _G64_COL_STD_OFFSET:_G64_LOG_S_ROW_OFFSET]
    log_s_row = scratch_words[:, _G64_LOG_S_ROW_OFFSET:_G64_BEST_S_ROW_OFFSET]
    best_s_row = scratch_words[:, _G64_BEST_S_ROW_OFFSET:_G64_ROW_STD_OFFSET]
    row_std = scratch_words[:, _G64_ROW_STD_OFFSET:_G64_BEST_IMBALANCE_OFFSET]
    best_imbalance = scratch_words[:, _G64_BEST_IMBALANCE_OFFSET]

    num_tiles = pool_slots.numel()
    if num_tiles == 0:
        return best_s_col, best_s_row
    common = {
        "pool_stride_s": latent_pool.stride(0),
        "pool_stride_t": latent_pool.stride(1),
        "pool_stride_d": latent_pool.stride(2),
        "scratch_stride_s": scratch_words.stride(0),
        "R": 512,
        "C": 64,
        "CLIP_STD_MIN": _CLIP_STD_MIN,
        "CLIP_STD_MAX": _CLIP_STD_MAX,
        "LOG_S_MIN": _LOG_S_MIN,
        "LOG_S_MAX": _LOG_S_MAX,
    }
    _sinkhorn_g64_row_phase_kernel[(num_tiles * 512,)](
        latent_pool,
        pool_slots,
        log_s_col,
        log_s_row,
        row_std,
        INITIAL=True,
        num_warps=4,
        **common,
    )
    for iteration in range(iterations):
        _sinkhorn_g64_column_phase_kernel[(num_tiles,)](
            latent_pool,
            pool_slots,
            log_s_col,
            log_s_row,
            best_s_col,
            best_s_row,
            col_std,
            row_std,
            best_imbalance,
            FIRST=iteration == 0,
            NORMALIZE=True,
            COLS_PER_CHUNK=8,
            num_warps=8,
            **common,
        )
        _sinkhorn_g64_row_phase_kernel[(num_tiles * 512,)](
            latent_pool,
            pool_slots,
            log_s_col,
            log_s_row,
            row_std,
            INITIAL=False,
            num_warps=4,
            **common,
        )
    _sinkhorn_g64_column_phase_kernel[(num_tiles,)](
        latent_pool,
        pool_slots,
        log_s_col,
        log_s_row,
        best_s_col,
        best_s_row,
        col_std,
        row_std,
        best_imbalance,
        FIRST=False,
        NORMALIZE=False,
        COLS_PER_CHUNK=8,
        num_warps=8,
        **common,
    )
    return best_s_col, best_s_row


def _sinkhorn_tiled(
    tiles: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, R, C = tiles.shape
    current = tiles.clone()
    balanced = torch.empty_like(tiles)
    log_s_col = torch.zeros(N, C, dtype=torch.float32, device=tiles.device)
    log_s_row = torch.zeros(N, R, dtype=torch.float32, device=tiles.device)
    s_col = torch.ones_like(log_s_col)
    s_row = torch.ones_like(log_s_row)
    col_std = torch.empty_like(log_s_col)
    row_std = torch.empty_like(log_s_row)
    best_imbalance = torch.full(
        (N,), float("inf"), dtype=torch.float32, device=tiles.device
    )

    _axis_std_kernel[(N * C,)](
        current,
        col_std,
        current.stride(0),
        current.stride(1),
        N_OUTER=C,
        WIDTH=R,
        TRANSPOSED=True,
        num_warps=_SINKHORN_REDUCTION_WARPS,
    )
    _axis_std_kernel[(N * R,)](
        current,
        row_std,
        current.stride(0),
        current.stride(1),
        N_OUTER=R,
        WIDTH=C,
        TRANSPOSED=False,
        num_warps=_SINKHORN_REDUCTION_WARPS,
    )
    _update_best_kernel[(N,)](
        col_std,
        row_std,
        log_s_col,
        log_s_row,
        best_imbalance,
        s_col,
        s_row,
        R=R,
        C=C,
        num_warps=_SINKHORN_REDUCTION_WARPS,
    )
    for _ in range(iterations):
        _normalize_columns_kernel[(N * C,)](
            tiles,
            current,
            log_s_col,
            log_s_row,
            tiles.stride(0),
            tiles.stride(1),
            R=R,
            C=C,
            CLIP_STD_MIN=_CLIP_STD_MIN,
            CLIP_STD_MAX=_CLIP_STD_MAX,
            LOG_S_MIN=_LOG_S_MIN,
            LOG_S_MAX=_LOG_S_MAX,
            num_warps=_SINKHORN_REDUCTION_WARPS,
        )
        _normalize_rows_kernel[(N * R,)](
            tiles,
            current,
            log_s_col,
            log_s_row,
            row_std,
            tiles.stride(0),
            tiles.stride(1),
            R=R,
            C=C,
            CLIP_STD_MIN=_CLIP_STD_MIN,
            CLIP_STD_MAX=_CLIP_STD_MAX,
            LOG_S_MIN=_LOG_S_MIN,
            LOG_S_MAX=_LOG_S_MAX,
            num_warps=_SINKHORN_REDUCTION_WARPS,
        )
        _axis_std_kernel[(N * C,)](
            current,
            col_std,
            current.stride(0),
            current.stride(1),
            N_OUTER=C,
            WIDTH=R,
            TRANSPOSED=True,
            num_warps=_SINKHORN_REDUCTION_WARPS,
        )
        _update_best_kernel[(N,)](
            col_std,
            row_std,
            log_s_col,
            log_s_row,
            best_imbalance,
            s_col,
            s_row,
            R=R,
            C=C,
            num_warps=_SINKHORN_REDUCTION_WARPS,
        )
    _finalize_tiled_kernel[(N * R,)](
        tiles,
        balanced,
        s_col,
        s_row,
        tiles.stride(0),
        tiles.stride(1),
        R=R,
        C=C,
    )
    return balanced, s_col, s_row


def kvarn_sinkhorn_triton(
    tiles: torch.Tensor, iterations: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton driver for ``_sinkhorn_log_kernel``.

    Args:
        tiles: ``[N, R, C]`` fp32 (or any real dtype, cast inside). Both R
            and C must be compile-time-constant power-of-2 values; we hard-
            code R = C = 128 for the first PR.
        iterations: number of alternating col/row passes (default 16).

    Returns:
        balanced: ``[N, R, C]`` fp32.
        s_col:    ``[N, C]`` fp32.
        s_row:    ``[N, R]`` fp32.
    """
    assert tiles.ndim == 3
    global _SINKHORN_MARKER_EMITTED
    if not _SINKHORN_MARKER_EMITTED:
        print(
            "KVARN_SINKHORN_ACTIVE "
            f"module={__name__} file={__file__} "
            f"raw_env={_DETERMINISTIC_SINKHORN_RAW} "
            f"reduction_warps={_SINKHORN_REDUCTION_WARPS}"
        )
        _SINKHORN_MARKER_EMITTED = True
    N, R, C = tiles.shape
    tiles = tiles.contiguous().to(torch.float32)
    device = tiles.device

    # The Triton kernel loads the WHOLE [R, C] tile into one program's registers
    # and unrolls the iteration loop. At large head_dim that tile is huge (e.g.
    # head_dim 512 -> [512, 128] = 256 KB) and the Triton compiler hangs/explodes
    # (128/256 compile fine). Route large tiles to the batched PyTorch Sinkhorn
    # (identical algorithm). Flush is infrequent + off the decode hot path, so the
    # cost is fine; head_dim<=256 keeps the fast kernel.

    if 256 < max(R, C) <= 512 and min(R, C) <= 128:
        return _sinkhorn_tiled(tiles, iterations)

    if max(R, C) > 256:
        from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
            variance_normalize_batched,
        )

        bal, s_col_b, s_row_b = variance_normalize_batched(tiles, iterations=iterations)
        return (
            bal.contiguous(),
            s_col_b.reshape(N, C).contiguous(),
            s_row_b.reshape(N, R).contiguous(),
        )

    balanced = torch.empty(N, R, C, dtype=torch.float32, device=device)
    s_col = torch.empty(N, C, dtype=torch.float32, device=device)
    s_row = torch.empty(N, R, dtype=torch.float32, device=device)

    _sinkhorn_log_kernel[(N,)](
        tiles,
        balanced,
        s_col,
        s_row,
        tiles.stride(0),
        tiles.stride(1),
        balanced.stride(0),
        balanced.stride(1),
        s_col.stride(0),
        s_row.stride(0),
        R=R,
        C=C,
        ITERATIONS=iterations,
        CLIP_STD_MIN=_CLIP_STD_MIN,
        CLIP_STD_MAX=_CLIP_STD_MAX,
        LOG_S_MIN=_LOG_S_MIN,
        LOG_S_MAX=_LOG_S_MAX,
        # num_warps=8, not 4: the program keeps the whole [R, C] fp32 tile (plus
        # a working copy) live, so at 4 warps the per-thread footprint is several
        # KB of registers -> the compiler spills to CUDA local memory, and the
        # driver permanently reserves local_bytes x max_threads x num_SMs of
        # device memory for the context (~2 GiB on a 188-SM part for the
        # [256, 128] tile; a missing-KV-capacity component). 8 warps
        # halves the per-thread footprint: ~70% less reserved local memory AND
        # ~4x faster flush (the spills were also the kernel's bottleneck).
        # Balanced-tile output is unchanged within fp32 reduction noise (~5e-7
        # rel); 16 warps saves a bit more memory but is 2x slower than 8.
        num_warps=8,
        num_stages=2,
    )
    return balanced, s_col, s_row
