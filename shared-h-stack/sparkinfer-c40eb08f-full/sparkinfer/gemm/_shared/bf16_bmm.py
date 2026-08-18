"""BF16 x BF16 -> BF16 batched GEMM for :func:`sparkinfer.gemm.bmm`.

Numerics mirror ``mla_query_projection/_bf16.py`` (the in-tree BF16 spec for
the fused query projection): every ``tl.dot`` accumulates into one FP32
accumulator traversing K in increasing order, and the single rounding point is
the final ``acc.to(tl.bfloat16)`` (round-to-nearest-even) at the store. There
is no split-K, no dual accumulator, and no fused scaling, so the value stream
per output element is exactly

    acc = 0.0
    for k_block in range(0, K, BLOCK_K):   # increasing k
        acc += HMMA.16816.F32 partials for k_block   # tl.dot, out_dtype=f32
    out = round_bf16(acc)

The routing knob is default-off; see the vLLM-side callers for the
bit-identity discussion against the incumbent cuBLASLt kernel.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
import triton
import triton.language as tl

from .mxfp8_bmm import _overlaps, _torch_stream

_BLOCK_N = 64
_BLOCK_K = 64
_NUM_WARPS = 4
_NUM_STAGES = 3
_MAX_M = 8192
_MAX_BATCH = 1024
_COMPILED_SIGNATURES: set[tuple[int, int]] = set()


def _tile_m_for_m(m: int) -> int:
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    return 128


@triton.jit(do_not_specialize=["m", "n", "k", "stride_ab", "stride_am", "stride_bb", "stride_bk", "stride_cb", "stride_cm"])
def _bmm_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_ab,
    stride_am,
    stride_bb,
    stride_bk,
    stride_cb,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    batch = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    row_mask = rows < m
    col_mask = cols < n

    a_base = a_ptr + batch * stride_ab
    b_base = b_ptr + batch * stride_bb
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, tl.cdiv(k, BLOCK_K)):
        k_off = k_start * BLOCK_K + ks
        k_mask = k_off < k
        a_tile = tl.load(
            a_base + rows[:, None] * stride_am + k_off[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b_tile = tl.load(
            b_base + k_off[:, None] * stride_bk + cols[None, :],
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    # Single rounding point: FP32 accumulator -> BF16 (round-to-nearest-even),
    # identical in contract to torch.bmm(out=bf16).
    out_tile = acc.to(tl.bfloat16)
    c_base = c_ptr + batch * stride_cb
    tl.store(
        c_base + rows[:, None] * stride_cm + cols[None, :],
        out_tile,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _validate(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    *,
    b_major: str,
) -> tuple[int, int, int, int]:
    if b_major != "n":
        raise NotImplementedError(
            "the BF16 BMM specialization requires b_major='n' "
            f"(row-major [B,K,N] rhs); got {b_major!r}"
        )
    for name, tensor, ndim in (("lhs", lhs, 3), ("rhs", rhs, 3), ("out", out, 3)):
        if tensor.ndim != ndim:
            raise ValueError(
                f"{name} must have shape [B,M,K]/[B,K,N]/[B,M,N]; "
                f"got ndim={tensor.ndim}"
            )
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if int(tensor.stride(-1)) != 1:
            raise ValueError(f"{name} innermost dimension must be contiguous")
    if lhs.device != rhs.device or lhs.device != out.device:
        raise ValueError("BF16 BMM operands must be on the same CUDA device")
    batch, m, k = (int(v) for v in lhs.shape)
    b_rhs, k_rhs, n = (int(v) for v in rhs.shape)
    b_out, m_out, n_out = (int(v) for v in out.shape)
    if not (1 <= batch <= _MAX_BATCH) or b_rhs != batch or b_out != batch:
        raise ValueError(f"inconsistent batch: {batch}/{b_rhs}/{b_out}")
    if not 1 <= m <= _MAX_M or m_out != m:
        raise ValueError(f"inconsistent M: {m}/{m_out} (max supported {_MAX_M})")
    if k != k_rhs or k % 16 != 0:
        raise ValueError(f"inconsistent or unsupported K: {k}/{k_rhs} (need K%16==0)")
    if n != n_out or n % 16 != 0:
        raise ValueError(f"inconsistent or unsupported N: {n}/{n_out} (need N%16==0)")
    if _overlaps(out, lhs) or _overlaps(out, rhs):
        raise ValueError("out must not overlap lhs or rhs")
    return batch, m, n, k


def _launch(lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor) -> None:
    batch, m, n, k = _validate(lhs, rhs, out, b_major="n")
    block_m = _tile_m_for_m(m)
    device_index = int(
        lhs.device.index
        if lhs.device.index is not None
        else torch.cuda.current_device()
    )
    signature = (device_index, block_m)
    if (
        torch.cuda.is_current_stream_capturing()
        and signature not in _COMPILED_SIGNATURES
    ):
        raise RuntimeError(
            "BF16 BMM compile miss during CUDA-graph capture for "
            f"M={m}; call gemm.prewarm_bmm first"
        )
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, _BLOCK_N), batch)
    _bmm_bf16_kernel[grid](
        lhs,
        rhs,
        out,
        m,
        n,
        k,
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _COMPILED_SIGNATURES.add(signature)


@torch.library.custom_op("sparkinfer::bmm_bf16", mutates_args=("out",))
def _op(lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor) -> None:
    _launch(lhs, rhs, out)


@_op.register_fake
def _fake(lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor) -> None:
    del lhs, rhs, out


def mm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
    *,
    b_major: str = "n",
    stream: Optional[object] = None,
) -> torch.Tensor:
    """Launch the BF16 BMM backend (``b_major='n'`` only)."""
    if b_major != "n":
        raise NotImplementedError(
            f"the BF16 BMM specialization requires b_major='n', got {b_major!r}"
        )
    if stream is None:
        torch.ops.sparkinfer.bmm_bf16(lhs, rhs, out)
        return out
    target = _torch_stream(stream, lhs.device)
    with torch.cuda.stream(target):
        torch.ops.sparkinfer.bmm_bf16(lhs, rhs, out)
        for tensor in (lhs, rhs, out):
            tensor.record_stream(target)
    return out


def prewarm(
    rhs: torch.Tensor,
    m_values: Iterable[int],
    *,
    b_major: str = "n",
    stream: Optional[object] = None,
    synchronize: bool = True,
) -> int:
    """Compile and first-launch each caller-declared graph-visible M regime."""
    batch, k, n = (int(v) for v in rhs.shape)
    _validate(
        torch.zeros((batch, 1, k), dtype=torch.bfloat16, device=rhs.device),
        rhs,
        torch.empty((batch, 1, n), dtype=torch.bfloat16, device=rhs.device),
        b_major=b_major,
    )
    unique_m: list[int] = []
    seen: set[int] = set()
    for raw_m in m_values:
        m = int(raw_m)
        if m not in seen:
            unique_m.append(m)
            seen.add(m)
    device = rhs.device
    torch_stream = _torch_stream(stream, device)
    with torch.cuda.stream(torch_stream):
        for m in unique_m:
            lhs = torch.zeros((batch, m, k), dtype=torch.bfloat16, device=device)
            out = torch.empty((batch, m, n), dtype=torch.bfloat16, device=device)
            mm(lhs, rhs, out, b_major=b_major)
            lhs.record_stream(torch_stream)
            out.record_stream(torch_stream)
    if synchronize:
        torch_stream.synchronize()
    return len(unique_m)


def can_implement(
    *,
    batch: int,
    max_m: int,
    n: int,
    k: int,
    b_major: str,
) -> bool:
    """Return whether the qualified BF16 backend covers a geometry."""
    return (
        b_major == "n"
        and 1 <= int(batch) <= _MAX_BATCH
        and 1 <= int(max_m) <= _MAX_M
        and int(k) % 16 == 0
        and int(n) % 16 == 0
        and int(k) >= 16
        and int(n) >= 16
    )


def clear_caches() -> None:
    _COMPILED_SIGNATURES.clear()


__all__ = ["mm", "prewarm", "can_implement", "clear_caches"]
