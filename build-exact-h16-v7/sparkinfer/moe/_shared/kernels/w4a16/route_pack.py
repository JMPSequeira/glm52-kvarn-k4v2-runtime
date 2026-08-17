"""Triton route-packing kernels for W4A16 MoE."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice
import torch

from sparkinfer.moe._shared.kernels.w4a16.host import route_pack_capacity


_COUNT_BLOCK_T = 256
_SORT_BLOCK_T = 256
_POST_PREFIX_BLOCK_T = 256
_SMALL_PREFIX_MAX_PACKED_ROUTES = 4096
_SMALL_PREFIX_MAX_ROUTE_BLOCKS = 128
_SMALL_PREFIX_MAX_EXPERT_BLOCK_PRODUCT = 65536


_FAST_COUNT_BLOCK_T = 1024


@triton.jit
def _w4a16_route_count_kernel(
    topk_ids,
    expert_map,
    counts,
    live_numel,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Parallel atomic histogram of routes per (mapped) expert.

    Multi-program replacement for the single-CTA count loop in
    ``_pack_topk_routes_prefix_kernel`` -- same expert-id resolution as the
    sort kernel (expert_map aware). Writes ``counts[NUM_EXPERTS]``."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
        tl.int32
    )
    valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)
    tl.atomic_add(counts + ids, 1, sem="relaxed", mask=valid)


@triton.jit
def _w4a16_route_prefix_from_counts_kernel(
    counts,
    packed_route_count,
    expert_offsets,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Tiny over-experts block-padded prefix from precomputed counts.

    Emits exactly the ``expert_offsets`` / ``packed_route_count`` contract of
    ``_pack_topk_routes_prefix_kernel`` (clean block-padded prefix; the sort
    kernel later advances expert_offsets in place)."""
    experts = tl.arange(0, BLOCK_E)
    mask = experts < NUM_EXPERTS
    counts_v = tl.load(counts + experts, mask=mask, other=0)
    padded = ((counts_v + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)
    tl.store(expert_offsets + experts, prefix, mask=mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _workspace_slice(
    tensor: torch.Tensor | None,
    *,
    name: str,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    elements = int(elements)
    if tensor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{name} is not initialized for CUDA graph capture; "
                "provide a preallocated W4A16 route-packing workspace"
            )
        return torch.empty((elements,), dtype=dtype, device=device)
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if int(tensor.numel()) < elements:
        raise ValueError(
            f"{name} has {tensor.numel()} elements; need at least {elements}"
        )
    return tensor[:elements]


@triton.jit
def _pack_topk_routes_post_prefix_kernel(
    packed_route_indices,
    block_expert_ids,
    expert_offsets,
    live_numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tl.store(
        packed_route_indices + offsets,
        live_numel,
        mask=offsets < MAX_PACKED_ROUTES,
    )

    block_rows = offsets * BLOCK_SIZE
    block_experts = tl.full((BLOCK_T,), -1, dtype=tl.int32)
    valid_blocks = offsets < MAX_ROUTE_BLOCKS
    for expert in tl.range(0, NUM_EXPERTS):
        start = tl.load(expert_offsets + expert)
        end = tl.load(expert_offsets + expert + 1)
        active = valid_blocks & (block_rows >= start) & (block_rows < end)
        block_experts = tl.where(active, expert, block_experts)
    tl.store(block_expert_ids + offsets, block_experts, mask=valid_blocks)


@triton.jit
def _pack_topk_routes_small_prefix_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    block_expert_ids,
    packed_route_count,
    expert_offsets,
    live_numel,
    NUMEL_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_ROUTE_INIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    counts = tl.zeros((BLOCK_E,), dtype=tl.int32)

    route_offsets = tl.arange(0, BLOCK_T)
    for start in tl.range(0, NUMEL_CAPACITY, BLOCK_T):
        offsets = start + route_offsets
        raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
            tl.int32
        )
        valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
        ids = raw_ids
        if HAS_EXPERT_MAP:
            safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
            ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
            valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

        matches = (
            (experts[:, None] == ids[None, :]) & expert_mask[:, None] & valid[None, :]
        )
        counts += tl.sum(matches.to(tl.int32), axis=1)

    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)

    tl.store(expert_offsets + experts, prefix, mask=expert_mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)

    route_init_offsets = tl.arange(0, BLOCK_ROUTE_INIT)
    tl.store(
        packed_route_indices + route_init_offsets,
        live_numel,
        mask=route_init_offsets < MAX_PACKED_ROUTES,
    )

    block_offsets = tl.arange(0, BLOCK_M)
    block_rows = block_offsets * BLOCK_SIZE
    active = (
        (block_offsets[None, :] < MAX_ROUTE_BLOCKS)
        & expert_mask[:, None]
        & (block_rows[None, :] >= prefix[:, None])
        & (block_rows[None, :] < inclusive[:, None])
    )
    block_experts = tl.max(tl.where(active, experts[:, None], -1), axis=0)
    tl.store(
        block_expert_ids + block_offsets,
        block_experts,
        mask=block_offsets < MAX_ROUTE_BLOCKS,
    )


@triton.jit
def _pack_topk_routes_prefix_kernel(
    topk_ids,
    expert_map,
    packed_route_count,
    expert_offsets,
    live_numel,
    NUMEL_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    counts = tl.zeros((BLOCK_E,), dtype=tl.int32)

    route_offsets = tl.arange(0, BLOCK_T)
    for start in tl.range(0, NUMEL_CAPACITY, BLOCK_T):
        offsets = start + route_offsets
        raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
            tl.int32
        )
        valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
        ids = raw_ids
        if HAS_EXPERT_MAP:
            safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
            ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
            valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

        matches = (
            (experts[:, None] == ids[None, :]) & expert_mask[:, None] & valid[None, :]
        )
        counts += tl.sum(matches.to(tl.int32), axis=1)

    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)

    tl.store(expert_offsets + experts, prefix, mask=expert_mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)


@triton.jit
def _pack_topk_routes_sort_kernel(
    topk_ids,
    expert_map,
    packed_route_indices,
    expert_offsets,
    live_numel,
    NUM_EXPERTS: tl.constexpr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    raw_ids = tl.load(topk_ids + offsets, mask=offsets < live_numel, other=-1).to(
        tl.int32
    )
    valid = (offsets < live_numel) & (raw_ids >= 0) & (raw_ids < NUM_EXPERTS)
    ids = raw_ids
    if HAS_EXPERT_MAP:
        safe_ids = tl.minimum(tl.maximum(raw_ids, 0), NUM_EXPERTS - 1)
        ids = tl.load(expert_map + safe_ids, mask=valid, other=-1).to(tl.int32)
        valid = valid & (ids >= 0) & (ids < NUM_EXPERTS)

    ranks = tl.atomic_add(expert_offsets + ids, 1, sem="relaxed", mask=valid)
    tl.store(packed_route_indices + ranks, offsets, mask=valid)


@triton.jit
def _fused_sigmoid_topk_route_pack_kernel(
    router_logits,
    correction_bias,
    expert_map,
    topk_weights,
    topk_ids,
    packed_route_indices,
    block_expert_ids,
    packed_route_count,
    expert_offsets,
    expert_counts,
    active_m,
    logits_stride_m,
    weight_stride_m,
    ids_stride_m,
    ROUTED_SCALE: tl.constexpr,
    M_CAPACITY: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_PACKED_ROUTES: tl.constexpr,
    MAX_ROUTE_BLOCKS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_E)
    expert_mask = experts < NUM_EXPERTS
    correction = tl.load(
        correction_bias + experts, mask=expert_mask, other=-float("inf")
    ).to(tl.float32)

    for token in tl.static_range(0, M_CAPACITY):
        token_live = token < active_m
        logits = tl.load(
            router_logits + token * logits_stride_m + experts,
            mask=token_live & expert_mask,
            other=-float("inf"),
        ).to(tl.float32)
        unbiased = 0.5 * libdevice.tanh(0.5 * logits) + 0.5
        selection = unbiased + correction
        weight_sum = tl.full((), 0.0, dtype=tl.float32)
        for route in tl.static_range(0, TOPK):
            winner_score = tl.max(selection, axis=0)
            winner = tl.min(
                tl.where(selection == winner_score, experts, NUM_EXPERTS),
                axis=0,
            ).to(tl.int32)
            weight = tl.sum(
                tl.where(experts == winner, unbiased, 0.0), axis=0
            )
            weight_sum = weight_sum + weight
            tl.store(
                topk_ids + token * ids_stride_m + route,
                winner.to(tl.int64),
                mask=token_live,
            )
            tl.store(
                topk_weights + token * weight_stride_m + route,
                weight,
                mask=token_live,
            )
            selection = tl.where(experts == winner, -float("inf"), selection)
        routes = tl.arange(0, TOPK)
        weights = tl.load(
            topk_weights + token * weight_stride_m + routes,
            mask=token_live,
            other=0.0,
        )
        normalized = weights * (ROUTED_SCALE / (weight_sum + 1.0e-20))
        tl.store(
            topk_weights + token * weight_stride_m + routes,
            normalized,
            mask=token_live,
        )

    tl.store(expert_counts + experts, 0, mask=expert_mask)
    packed_offsets = tl.arange(0, BLOCK_PACKED)
    tl.store(
        packed_route_indices + packed_offsets,
        active_m * TOPK,
        mask=packed_offsets < MAX_PACKED_ROUTES,
    )
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    tl.store(
        block_expert_ids + block_offsets,
        -1,
        mask=block_offsets < MAX_ROUTE_BLOCKS,
    )
    tl.debug_barrier()

    routes = tl.arange(0, BLOCK_R)
    live_numel = active_m * TOPK
    valid_routes = routes < live_numel
    raw_ids = tl.load(topk_ids + routes, mask=valid_routes, other=-1).to(tl.int32)
    mapped = tl.load(
        expert_map + tl.maximum(raw_ids, 0), mask=valid_routes, other=-1
    ).to(tl.int32)
    valid_routes = valid_routes & (mapped >= 0) & (mapped < NUM_EXPERTS)
    tl.atomic_add(expert_counts + mapped, 1, mask=valid_routes)
    tl.debug_barrier()

    counts = tl.load(expert_counts + experts, mask=expert_mask, other=0)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = tl.where(expert_mask, padded, 0)
    inclusive = tl.cumsum(padded, axis=0)
    prefix = inclusive - padded
    total = tl.sum(padded, axis=0)
    tl.store(expert_offsets + experts, prefix, mask=expert_mask)
    tl.store(expert_offsets + NUM_EXPERTS, total)
    tl.store(packed_route_count, total)
    start_blocks = prefix // BLOCK_SIZE
    active_experts = expert_mask & (counts > 0)
    tl.store(
        block_expert_ids + start_blocks,
        experts,
        mask=active_experts & (start_blocks < MAX_ROUTE_BLOCKS),
    )
    second_blocks = start_blocks + 1
    tl.store(
        block_expert_ids + second_blocks,
        experts,
        mask=active_experts
        & (counts > BLOCK_SIZE)
        & (second_blocks < MAX_ROUTE_BLOCKS),
    )
    tl.debug_barrier()

    ranks = tl.atomic_add(expert_offsets + mapped, 1, mask=valid_routes)
    tl.store(packed_route_indices + ranks, routes, mask=valid_routes)


def fused_sigmoid_topk_pack(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    routed_scale: float,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_counts: torch.Tensor,
    block_size: int,
) -> None:
    """Fused target-only E256/K8 sigmoid selection and route packing."""
    rows, experts = map(int, router_logits.shape)
    if not 1 <= rows <= 9 or experts != 256:
        raise ValueError("fused route pack requires M in [1,9] and E256")
    if router_logits.stride(1) != 1:
        raise ValueError("fused route-pack logits must be inner-contiguous")
    if correction_bias.dtype != torch.float32 or correction_bias.shape != (256,):
        raise ValueError("fused route-pack correction bias must be FP32[256]")
    if expert_map.dtype != torch.int32 or expert_map.shape != (256,):
        raise ValueError("fused route-pack expert map must be int32[256]")
    if topk_weights.dtype != torch.float32 or topk_weights.shape[1] != 8:
        raise ValueError("fused route-pack weights must be FP32[capacity,8]")
    if topk_ids.dtype != torch.int64 or topk_ids.shape[1] != 8:
        raise ValueError("fused route-pack IDs must be int64[capacity,8]")
    if int(block_size) != 8:
        raise ValueError("fused target route pack requires block size 8")
    tensors = (
        correction_bias,
        expert_map,
        topk_weights,
        topk_ids,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        expert_offsets,
        expert_counts,
    )
    if router_logits.device.type != "cuda" or any(
        tensor.device != router_logits.device for tensor in tensors
    ):
        raise ValueError("fused route-pack tensors must share one CUDA device")
    _fused_sigmoid_topk_route_pack_kernel[(1,)](
        router_logits,
        correction_bias,
        expert_map,
        topk_weights,
        topk_ids,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        expert_offsets,
        expert_counts,
        rows,
        router_logits.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        ROUTED_SCALE=float(routed_scale),
        M_CAPACITY=9,
        TOPK=8,
        NUM_EXPERTS=256,
        BLOCK_SIZE=8,
        MAX_PACKED_ROUTES=packed_route_indices.numel(),
        MAX_ROUTE_BLOCKS=block_expert_ids.numel(),
        BLOCK_E=256,
        BLOCK_R=128,
        BLOCK_PACKED=triton.next_power_of_2(packed_route_indices.numel()),
        BLOCK_BLOCKS=triton.next_power_of_2(block_expert_ids.numel()),
        num_warps=8,
    )


def pack_topk_routes_by_expert(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    expert_map: torch.Tensor | None = None,
    packed_route_indices: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    packed_route_count: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = int(topk_ids.numel())
    topk = int(topk_ids.shape[-1]) if topk_ids.ndim >= 2 else 1
    numel_capacity, capacity_packed_routes, capacity_route_blocks = route_pack_capacity(
        numel,
        int(block_size),
        int(num_experts),
        topk=topk,
    )
    if packed_route_indices is not None and block_expert_ids is not None:
        if (
            int(packed_route_indices.numel()) < capacity_packed_routes
            or int(block_expert_ids.numel()) < capacity_route_blocks
        ):
            (
                numel_capacity,
                capacity_packed_routes,
                capacity_route_blocks,
            ) = route_pack_capacity(
                numel,
                int(block_size),
                int(num_experts),
                topk=topk,
                bucket_tokens=False,
            )
    max_packed_routes = capacity_packed_routes
    max_route_blocks = capacity_route_blocks
    max_packed_routes = max(max_packed_routes, 1)
    max_route_blocks = max(max_route_blocks, 1)

    provided_routes = (
        None if packed_route_indices is None else int(packed_route_indices.numel())
    )
    provided_blocks = (
        None if block_expert_ids is None else int(block_expert_ids.numel())
    )
    if (
        provided_routes is not None
        and provided_blocks is not None
        and (provided_routes < max_packed_routes or provided_blocks < max_route_blocks)
    ):
        raise ValueError(
            "W4A16 route-packing workspace is too small: "
            f"topk_shape={tuple(topk_ids.shape)}, live_routes={numel}, "
            f"capacity_routes={numel_capacity}, block_size={int(block_size)}, "
            f"num_experts={int(num_experts)}, "
            f"packed_route_indices={provided_routes}/{max_packed_routes}, "
            f"block_expert_ids={provided_blocks}/{max_route_blocks}"
        )

    packed_route_indices = _workspace_slice(
        packed_route_indices,
        name="packed_route_indices",
        elements=max_packed_routes,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    block_expert_ids = _workspace_slice(
        block_expert_ids,
        name="block_expert_ids",
        elements=max_route_blocks,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    packed_route_count = _workspace_slice(
        packed_route_count,
        name="packed_route_count",
        elements=1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_offsets = _workspace_slice(
        expert_offsets,
        name="expert_offsets",
        elements=int(num_experts) + 1,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    if numel == 0:
        packed_route_indices.fill_(0)
        block_expert_ids.fill_(-1)
        packed_route_count.zero_()
        return packed_route_indices, block_expert_ids, packed_route_count

    block_e = _next_power_of_2(num_experts)
    sort_grid = (triton.cdiv(numel, _SORT_BLOCK_T),)
    expert_map_tensor = expert_map if expert_map is not None else topk_ids

    block_route_init = _next_power_of_2(max(max_packed_routes, 1))
    block_m = _next_power_of_2(max(max_route_blocks, 1))
    small_count_block_t = min(
        _COUNT_BLOCK_T, _next_power_of_2(numel_capacity)
    )
    use_small_prefix = (
        block_route_init <= _SMALL_PREFIX_MAX_PACKED_ROUTES
        and block_m <= _SMALL_PREFIX_MAX_ROUTE_BLOCKS
        and block_e * block_m <= _SMALL_PREFIX_MAX_EXPERT_BLOCK_PRODUCT
    )
    if use_small_prefix:
        # Decode-sized W4A16 MoE calls are launch-overhead sensitive. Keep the
        # large-shape split kernel below, but fold prefix/post-prefix work into
        # one launch when the vector sizes are safely bounded.
        _pack_topk_routes_small_prefix_kernel[(1,)](
            topk_ids,
            expert_map_tensor,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            expert_offsets,
            numel,
            NUMEL_CAPACITY=numel_capacity,
            BLOCK_SIZE=int(block_size),
            NUM_EXPERTS=int(num_experts),
            MAX_PACKED_ROUTES=max_packed_routes,
            MAX_ROUTE_BLOCKS=max_route_blocks,
            HAS_EXPERT_MAP=expert_map is not None,
            BLOCK_E=block_e,
            BLOCK_T=small_count_block_t,
            BLOCK_ROUTE_INIT=block_route_init,
            BLOCK_M=block_m,
            num_warps=4,
        )
    else:
        post_prefix_grid = (
            max(
                triton.cdiv(max_packed_routes, _POST_PREFIX_BLOCK_T),
                triton.cdiv(max_route_blocks, _POST_PREFIX_BLOCK_T),
            ),
        )
        if expert_counts is not None:
            expert_counts = _workspace_slice(
                expert_counts,
                name="expert_counts",
                elements=int(num_experts),
                dtype=torch.int32,
                device=topk_ids.device,
            )
            expert_counts.zero_()
            _w4a16_route_count_kernel[(triton.cdiv(numel, _FAST_COUNT_BLOCK_T),)](
                topk_ids,
                expert_map_tensor,
                expert_counts,
                numel,
                NUM_EXPERTS=int(num_experts),
                HAS_EXPERT_MAP=expert_map is not None,
                BLOCK_T=_FAST_COUNT_BLOCK_T,
                num_warps=4,
            )
            _w4a16_route_prefix_from_counts_kernel[(1,)](
                expert_counts,
                packed_route_count,
                expert_offsets,
                BLOCK_SIZE=int(block_size),
                NUM_EXPERTS=int(num_experts),
                BLOCK_E=block_e,
                num_warps=4,
            )
        elif not torch.cuda.is_current_stream_capturing():
            # FAST (eager prefill): parallel atomic count + tiny over-experts
            # block-padded prefix, replacing the single-CTA count+prefix
            # (~7-31x measured at prefill). Only the large path (routes > 4096,
            # i.e. > 512 tokens) reaches here, and cudagraph capture is bounded
            # to <=128 tokens (the small-prefix path), so this scratch alloc is
            # never captured. The slow single-CTA kernel stays as the defensive
            # captured-path fallback below.
            expert_counts = torch.zeros(
                int(num_experts), dtype=torch.int32, device=topk_ids.device
            )
            _w4a16_route_count_kernel[(triton.cdiv(numel, _FAST_COUNT_BLOCK_T),)](
                topk_ids,
                expert_map_tensor,
                expert_counts,
                numel,
                NUM_EXPERTS=int(num_experts),
                HAS_EXPERT_MAP=expert_map is not None,
                BLOCK_T=_FAST_COUNT_BLOCK_T,
                num_warps=4,
            )
            _w4a16_route_prefix_from_counts_kernel[(1,)](
                expert_counts,
                packed_route_count,
                expert_offsets,
                BLOCK_SIZE=int(block_size),
                NUM_EXPERTS=int(num_experts),
                BLOCK_E=block_e,
                num_warps=4,
            )
        else:
            _pack_topk_routes_prefix_kernel[(1,)](
                topk_ids,
                expert_map_tensor,
                packed_route_count,
                expert_offsets,
                numel,
                NUMEL_CAPACITY=numel_capacity,
                BLOCK_SIZE=int(block_size),
                NUM_EXPERTS=int(num_experts),
                HAS_EXPERT_MAP=expert_map is not None,
                BLOCK_E=block_e,
                BLOCK_T=_COUNT_BLOCK_T,
                num_warps=8,
            )
        _pack_topk_routes_post_prefix_kernel[post_prefix_grid](
            packed_route_indices,
            block_expert_ids,
            expert_offsets,
            numel,
            BLOCK_SIZE=int(block_size),
            NUM_EXPERTS=int(num_experts),
            MAX_PACKED_ROUTES=max_packed_routes,
            MAX_ROUTE_BLOCKS=max_route_blocks,
            BLOCK_T=_POST_PREFIX_BLOCK_T,
            num_warps=4,
        )
    _pack_topk_routes_sort_kernel[sort_grid](
        topk_ids,
        expert_map_tensor,
        packed_route_indices,
        expert_offsets,
        numel,
        NUM_EXPERTS=int(num_experts),
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_T=_SORT_BLOCK_T,
        num_warps=4,
    )
    return packed_route_indices, block_expert_ids, packed_route_count


__all__ = ["pack_topk_routes_by_expert"]
