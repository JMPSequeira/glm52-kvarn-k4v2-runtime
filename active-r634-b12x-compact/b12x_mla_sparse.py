# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x sparse-MLA backend for SM120 / SM121 (consumer Blackwell).

Counterpart to ``SparseMLASm120Backend`` (FlashInfer V32 v2). Same envelope --
``fp8_ds_mla`` KV cache (656 B/token), head_size = 576, paged block_size = 64,
V32-family models with an ``index_topk`` config (DeepSeek V3.2, GLM-5.1, Kimi
K2.5) -- but the decode/extend kernels come from SparkInfer's unified SM120
backend via the ``sparkinfer.attention.sparse_mla`` front door (``run_decode`` /
``run_extend``). On SM120+ CUDA those front-door functions route to SparkInfer's
unified MLA implementation automatically (GLM_NSA q_head_dim==576 contract).
Selecting this backend also selects SparkInfer's sparse indexer/top-k path.

Scratch philosophy (eager PLAN -> BIND -> KERNEL; no workspace/arena, ever):
b12x workspaces/arenas are sglang-only and forbidden here. We build a caller-
owned-scratch ``plan_sparse_mla_scratch`` PLAN once per mode (decode / extend),
and each forward maps a vLLM ``current_workspace_manager()`` scratch tensor into
a plain ``B12XSparseMLAScratch`` views CONTAINER via ``plan.bind(...)`` -- a pure
narrow()+view() mapping that allocates nothing and constructs no workspace. The
binding holds views (never a ``B12XAttentionWorkspace``); the unified SM120
sparse-MLA decode/extend kernels duck-type the container's
``tmp_output`` / ``tmp_lse`` / ``output_buffer`` / ``final_lse`` /
``num_chunks_ptr`` / ``set_split_chunk_config`` fields, so the binding is a
drop-in with no kernel-signature change. q-concat and the scratch are borrowed in
ONE ``get_simultaneous`` call so they never alias.
"""

import inspect
import math
import os
import json
import re
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import get_mla_dims
from vllm.model_executor.layers.quantization.kvarn.config import (
    is_kvarn_mla_cache_dtype,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.kvarn_decode import kvarn_hadamard
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)
_CKV_POOL_STATE_ENABLED = os.getenv("VLLM_B12X_CKV_POOL_STATE", "0") == "1"
_KVARN_FUSED_CURRENT_STAGE_ENABLED = (
    os.getenv("VLLM_KVARN_MLA_FUSED_CURRENT_STAGE", "0") == "1"
)
_DCP_PROJECT_BF16_BMM_ENABLED = (
    os.getenv("VLLM_B12X_BF16_BMM_DCP_PROJECT", "0") == "1"
)
_SPARSE_META_FUSED_ENABLED = (
    os.getenv("VLLM_B12X_SPARSE_META_FUSED", "0") == "1"
)
_BF16_BMM_SPEC = {
    "a_dtype": "bfloat16",
    "b_dtype": "bfloat16",
    "c_dtype": "bfloat16",
}
_DCP_PROJECT_BF16_BMM_PREWARMED: set[tuple[int, int, int, int]] = set()
_DCP_PROJECT_SPARKINFER_BMM = None
_DCP_PROJECT_SPARKINFER_BMM_MISSING = False


def _import_dcp_project_sparkinfer_bmm():
    """Resolve sparkinfer.gemm.bmm once; None when unavailable (fail closed)."""
    global _DCP_PROJECT_SPARKINFER_BMM, _DCP_PROJECT_SPARKINFER_BMM_MISSING
    if _DCP_PROJECT_SPARKINFER_BMM is not None:
        return _DCP_PROJECT_SPARKINFER_BMM
    if _DCP_PROJECT_SPARKINFER_BMM_MISSING:
        return None
    try:
        from sparkinfer.gemm import bmm as _bmm_fn

        _DCP_PROJECT_SPARKINFER_BMM = _bmm_fn
    except Exception:
        _DCP_PROJECT_SPARKINFER_BMM_MISSING = True
    return _DCP_PROJECT_SPARKINFER_BMM


def _run_dcp_project_bmm(
    projection_input: torch.Tensor,
    w_uv: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """DCP workspace projection BMM with an optional SM120-native route.

    Default-off (VLLM_B12X_BF16_BMM_DCP_PROJECT). The incumbent
    ``torch.bmm`` stays the exact behavior when the knob is off, the operands
    fall outside the sparkinfer BF16 gemm.bmm envelope, or a CUDA-graph
    capture-time compile miss raises (fail closed).
    """
    if _DCP_PROJECT_BF16_BMM_ENABLED:
        _sparkinfer_bmm = _import_dcp_project_sparkinfer_bmm()
        if _sparkinfer_bmm is not None:
            try:
                _sparkinfer_bmm(
                    projection_input,
                    w_uv,
                    out,
                    b_major="n",
                    **_BF16_BMM_SPEC,
                )
                return
            except Exception:
                logger.warning_once(
                    "VLLM_B12X_BF16_BMM_DCP_PROJECT=1 but the sparkinfer BF16 "
                    "gemm.bmm route rejected the DCP projection; keeping "
                    "torch.bmm."
                )
    torch.bmm(projection_input, w_uv, out=out)


def _prewarm_dcp_project_bf16_bmm(
    num_heads: int, latent_dim: int, v_head_dim: int, device_index: int
) -> int:
    """Compile every BLOCK_M regime of the BF16 DCP projection BMM."""
    if not _DCP_PROJECT_BF16_BMM_ENABLED:
        return 0
    key = (int(num_heads), int(latent_dim), int(v_head_dim), int(device_index))
    if key in _DCP_PROJECT_BF16_BMM_PREWARMED:
        return 0
    _DCP_PROJECT_BF16_BMM_PREWARMED.add(key)
    if _import_dcp_project_sparkinfer_bmm() is None:
        return 0
    try:
        rhs = torch.zeros(
            (int(num_heads), int(latent_dim), int(v_head_dim)),
            dtype=torch.bfloat16,
            device=torch.device("cuda", int(device_index)),
        )
        # One launch per BLOCK_M tile regime (16/32/64/128); scalar args are
        # do_not_specialize so these four cover every runtime M.
        from sparkinfer.gemm import prewarm_bmm as _sparkinfer_prewarm

        return int(
            _sparkinfer_prewarm(
                rhs,
                (16, 32, 64, 128),
                b_major="n",
                **_BF16_BMM_SPEC,
            )
        )
    except Exception:
        logger.warning_once(
            "sparkinfer BF16 gemm.bmm prewarm failed for the DCP projection; "
            "the projection will fall back to torch.bmm at runtime."
        )
        return 0


# Split-K tile width. Mirrors SparseMLASm120's _DECODE_SPLIT_TILE: the number of
# split-K chunks is ceil(topk / tile). This bounds the chunk dim of the borrowed
# mid_out/mid_lse scratch and the workspace ``max_chunks_per_row`` cap; b12x's
# wave-balanced planner picks num_splits <= this cap.
_DECODE_SPLIT_TILE = 64
_HEAD_ALIGNMENT = 8
_BF16_BYTES = 2
_INT32_BYTES = 4
_EXTEND_PREWARM_DONE: set[
    tuple[int | None, int, int, int, int, int, bool, str, bool]
] = set()
_KV_FP8_ROPE_REQUESTED = os.getenv("KV_FP8_ROPE", "0") == "1"


@dataclass(frozen=True)
class _KVarNMLAWorkspaceEnvelope:
    dense_rows: int
    remap_elements: int
    rotation_rows: int
    physical_slot_rows: int
    dense_bytes: int
    total_bytes: int


def _kvarn_mla_workspace_envelope(
    *,
    num_kv_pages: int,
    group_size: int,
    latent_dim: int,
    rope_dim: int,
    max_batched_tokens: int,
    max_active_rows: int,
    topk_tokens: int,
    boundary_blocks: int,
    rollback_blocks: int,
) -> _KVarNMLAWorkspaceEnvelope:
    """Return the fixed KVarN staging envelope for one local worker."""
    positive = {
        "num_kv_pages": num_kv_pages,
        "group_size": group_size,
        "latent_dim": latent_dim,
        "rope_dim": rope_dim,
        "max_batched_tokens": max_batched_tokens,
        "max_active_rows": max_active_rows,
        "topk_tokens": topk_tokens,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if boundary_blocks < 0 or rollback_blocks < 0:
        invalid.update(
            boundary_blocks=boundary_blocks,
            rollback_blocks=rollback_blocks,
        )
    if invalid:
        raise ValueError(f"Invalid KVarN MLA workspace dimensions: {invalid}")

    # Full prefill is physically indexed and cannot touch more than the local
    # page arena. Decode/verify instead linearizes top-k rows. The shared BF16
    # buffer is the G64-aligned maximum of those paths and transient staging,
    # never max_model_len multiplied by max_num_seqs.
    page_rows = num_kv_pages * group_size
    selected_rows = max_active_rows * topk_tokens
    # A scheduler step can stage every batched token plus one page per request
    # crossing a G64 boundary and the pages retained for async-spec rollback.
    transient_rows = (
        _cdiv(max_batched_tokens, group_size)
        + boundary_blocks
        + rollback_blocks
    ) * group_size
    dense_rows = _cdiv(
        max(page_rows, selected_rows, transient_rows), group_size
    ) * group_size
    remap_elements = max_batched_tokens * topk_tokens
    dense_bytes = dense_rows * (latent_dim + rope_dim) * _BF16_BYTES
    total_bytes = (
        dense_bytes
        + remap_elements * _INT32_BYTES
        + max_batched_tokens * latent_dim * _BF16_BYTES
        + page_rows * _INT32_BYTES
    )
    return _KVarNMLAWorkspaceEnvelope(
        dense_rows=dense_rows,
        remap_elements=remap_elements,
        rotation_rows=max_batched_tokens,
        physical_slot_rows=page_rows,
        dense_bytes=dense_bytes,
        total_bytes=total_bytes,
    )


_IS_GLM_MOE_DSA_CACHE: bool | None = None


def _is_glm_moe_dsa_model() -> bool:
    """Return true only for GLM or its in-process MTP draft model.

    Robust to being called before the vLLM config context is established (e.g.
    during KV-cache shape resolution / cudagraph compilation in a worker, where
    get_current_vllm_config() raises): fall back to the explicit KV_FP8_ROPE
    request and re-resolve once the config becomes available. Correctness is
    preserved because the fallback is only reached when the user set
    KV_FP8_ROPE=1 for their GLM model; KV_FP8_ROPE=0 short-circuits earlier.
    """
    global _IS_GLM_MOE_DSA_CACHE
    if _IS_GLM_MOE_DSA_CACHE is not None:
        return _IS_GLM_MOE_DSA_CACHE
    from vllm.config import get_current_vllm_config

    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return _KV_FP8_ROPE_REQUESTED
    model_config = vllm_config.model_config
    if model_config is None:
        return False
    model_type = getattr(model_config.hf_config, "model_type", None)
    if model_type == "glm_moe_dsa":
        _IS_GLM_MOE_DSA_CACHE = True
        return True
    speculative_config = getattr(vllm_config, "speculative_config", None)
    target_model_config = getattr(speculative_config, "target_model_config", None)
    target_model_type = (
        getattr(target_model_config.hf_config, "model_type", None)
        if target_model_config is not None
        else None
    )
    result = model_type == "deepseek_mtp" and target_model_type == "glm_moe_dsa"
    _IS_GLM_MOE_DSA_CACHE = result
    return result


def _kv_fp8_rope_enabled() -> bool:
    """Strict public gate plus literal GLM architecture selection."""
    return _KV_FP8_ROPE_REQUESTED and _is_glm_moe_dsa_model()


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed

def _direct_packed_kvarn_effective_a2a() -> bool:
    token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
    return (
        token_cap <= 0
        or token_cap >= 4
        or envs.VLLM_DCP_A2A_LARGE_BACKEND == "a2a"
    )

def _direct_packed_kvarn_mla_rows(
    *, use_decode_kernel: bool, direct_packed_enabled: bool, num_actual_toks: int
) -> int:
    if (
        not use_decode_kernel
        or not direct_packed_enabled
        or not 1 <= num_actual_toks <= 16
    ):
        return 0
    return num_actual_toks


def _direct_packed_selected_buffer_valid(
    selected_indices: torch.Tensor | None, device: torch.device
) -> bool:
    return bool(
        selected_indices is not None
        and selected_indices.dtype == torch.int32
        and selected_indices.device == device
        and selected_indices.ndim == 2
        and selected_indices.shape[0] > 0
        and selected_indices.shape[1] == 2048
        and selected_indices.is_contiguous()
    )


def _enable_direct_packed_kvarn_mla_instance(
    *, requested: bool, has_selected_buffer: bool, dcp_comm_backend: str | None
) -> bool:
    effective_a2a = (
        dcp_comm_backend == "a2a"
        or envs.VLLM_DCP_A2A_LARGE_BACKEND == "a2a"
    )
    return requested and has_selected_buffer and effective_a2a


def _select_kvarn_mla_stage_workspace(
    *,
    direct_packed: bool,
    dense_cache: torch.Tensor | None,
    direct_stage: torch.Tensor | None,
) -> torch.Tensor:
    workspace = direct_stage if direct_packed else dense_cache
    if workspace is None:
        raise RuntimeError("KVarN MLA graph workspace is not initialized")
    return workspace




def _ckv_prefetch_supports_format(kv_cache_dtype: str) -> bool:
    if kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"):
        return True
    return is_kvarn_mla_cache_dtype(kv_cache_dtype)


def _ckv_prefetch_ring_slots(depth: int) -> int:
    return max(0, int(depth)) + 1


def _ckv_prefetch_workspace_nbytes(
    depth: int,
    dcp_world_size: int,
    local_capacity: int,
    record_bytes: int,
) -> int:
    """Return one lane's local staging plus gathered-cache ring size."""
    return (
        (1 + _ckv_prefetch_ring_slots(depth) * int(dcp_world_size))
        * int(local_capacity)
        * int(record_bytes)
    )


def _ckv_native_workspace_nbytes(
    depth: int,
    dcp_world_size: int,
    rank_wire_bytes: int,
) -> int:
    return (
        _ckv_prefetch_ring_slots(depth)
        * int(dcp_world_size)
        * int(rank_wire_bytes)
    )


def _ckv_prefetch_execution_lanes(num_ubatches: int, speculative: bool) -> int:
    return max(1, int(num_ubatches)) * (2 if speculative else 1)


class _CKVPrefetchWorkspacePool:
    """Preallocated CKV rings shared by all attention layers on one device."""

    def __init__(
        self,
        device: torch.device,
        slot_nbytes: int,
        max_slots: int,
    ) -> None:
        if slot_nbytes <= 0 or max_slots <= 0:
            raise ValueError(
                "CKV workspace pool requires positive slot size and count, got "
                f"slot_nbytes={slot_nbytes} max_slots={max_slots}"
            )
        self.device = device
        self.slot_nbytes = int(slot_nbytes)
        self.max_slots = int(max_slots)
        self.storage = torch.empty(
            (self.slot_nbytes * self.max_slots,),
            dtype=torch.uint8,
            device=device,
        )
        self._free_slots = list(reversed(range(self.max_slots)))
        self._leased_slots: set[int] = set()
        self._state_map: dict[Any, Any] = {}

    def acquire(self) -> tuple[int, torch.Tensor]:
        if not self._free_slots:
            raise RuntimeError(
                "CKV prefetch workspace pool exhausted. The runtime created more "
                f"than {self.max_slots} execution lanes; disable CKV prefetch or "
                "increase the configured lane reservation."
            )
        slot = self._free_slots.pop()
        self._leased_slots.add(slot)
        start = slot * self.slot_nbytes
        return slot, self.storage.narrow(0, start, self.slot_nbytes)

    def release(self, slot: int) -> None:
        if slot not in self._leased_slots:
            raise RuntimeError(f"CKV workspace slot {slot} is not leased")
        self._leased_slots.remove(slot)
        self._free_slots.append(slot)


_CKV_PREFETCH_WORKSPACE_POOLS: dict[
    tuple[str, int | None, int, int], _CKVPrefetchWorkspacePool
] = {}


def _get_ckv_prefetch_workspace_pool(
    device: torch.device,
    slot_nbytes: int,
    max_slots: int,
) -> _CKVPrefetchWorkspacePool:
    key = (device.type, device.index, int(slot_nbytes), int(max_slots))
    pool = _CKV_PREFETCH_WORKSPACE_POOLS.get(key)
    if pool is None:
        pool = _CKVPrefetchWorkspacePool(device, slot_nbytes, max_slots)
        _CKV_PREFETCH_WORKSPACE_POOLS[key] = pool
    return pool


def _ckv_prefetch_depth_within_budget(
    requested_depth: int,
    workspace_budget_bytes: int,
    dcp_world_size: int,
    local_capacity: int,
    record_bytes: int,
) -> int:
    """Cap lookahead depth without removing the synchronous gather slot."""
    requested_depth = max(0, int(requested_depth))
    workspace_budget_bytes = int(workspace_budget_bytes)
    if workspace_budget_bytes <= 0:
        return requested_depth
    for depth in range(requested_depth, -1, -1):
        if (
            _ckv_prefetch_workspace_nbytes(
                depth,
                dcp_world_size,
                local_capacity,
                record_bytes,
            )
            <= workspace_budget_bytes
        ):
            return depth
    return 0


def _ckv_prefetch_target_indices(
    layer_idx: int,
    depth: int,
    layer_caches: list[torch.Tensor | None],
    pending_layers: dict[int, tuple[Any, int]],
) -> list[int]:
    targets: list[int] = []
    for distance in range(1, max(0, int(depth)) + 1):
        target_idx = layer_idx + distance
        if target_idx in pending_layers:
            continue
        if target_idx >= len(layer_caches) or layer_caches[target_idx] is None:
            break
        targets.append(target_idx)
    return targets


@dataclass(frozen=True)
class _CKVWorkspaceIdentity:
    device: torch.device
    storage_data_ptr: int
    storage_nbytes: int
    data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


def _ckv_workspace_identity(workspace: torch.Tensor) -> _CKVWorkspaceIdentity:
    storage = workspace.untyped_storage()
    return _CKVWorkspaceIdentity(
        device=workspace.device,
        # WorkspaceManager exposes transient tensor views, so tensor identity is
        # not stable across borrows. The storage base and size are public,
        # stable allocation metadata; liveness is tracked separately below.
        storage_data_ptr=storage.data_ptr(),
        storage_nbytes=storage.nbytes(),
        data_ptr=workspace.data_ptr(),
        storage_offset=workspace.storage_offset(),
        shape=tuple(workspace.shape),
        stride=tuple(workspace.stride()),
        dtype=workspace.dtype,
    )


class _CKVPrefetchState:
    """Cross-layer state for one workspace allocation and execution lane."""

    def __init__(
        self,
        workspace_identity: _CKVWorkspaceIdentity,
        workspace: torch.Tensor,
        workspace_pool: _CKVPrefetchWorkspacePool,
    ) -> None:
        self.workspace_identity = workspace_identity
        self.workspace_storage_ref = weakref.ref(workspace.untyped_storage())
        self.workspace_pool = workspace_pool
        self.layer_caches: list[torch.Tensor | None] = []
        self.layer_owners: list[weakref.ReferenceType[Any] | None] = []
        self.pending_layers: dict[int, tuple[Any, int]] = {}
        self.gather_stream: torch.cuda.Stream | None = None
        self.ckv_workspace: torch.Tensor | None = None
        self.ckv_workspace_slot: int | None = None
        self.ckv_workspace_generation = 0
        self.last_layer_idx: int | None = None

    def begin_step(self) -> None:
        self.wait_for_pending_writes()
        self.pending_layers.clear()
        self.last_layer_idx = None

    def wait_for_pending_writes(self) -> None:
        """Order current-stream fallback work after side-stream gathers."""
        for event, _ in self.pending_layers.values():
            # Preserve ring ordering without blocking the host indefinitely.
            # The next main-stream gather is enqueued after these dependencies.
            event.wait()

    def enter_layer(self, layer_idx: int) -> None:
        if self.last_layer_idx is not None and layer_idx <= self.last_layer_idx:
            self.begin_step()
        self.last_layer_idx = layer_idx

    def register_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        owner: Any,
    ) -> None:
        while len(self.layer_caches) <= layer_idx:
            self.layer_caches.append(None)
            self.layer_owners.append(None)
        existing_owner_ref = self.layer_owners[layer_idx]
        existing_owner = (
            existing_owner_ref() if existing_owner_ref is not None else None
        )
        if existing_owner is not None and existing_owner is not owner:
            raise RuntimeError(
                f"CKV layer {layer_idx} changed staging owner within one registry"
            )
        self.layer_caches[layer_idx] = kv_cache
        self.layer_owners[layer_idx] = weakref.ref(owner)

    def get_gather_stream(self) -> torch.cuda.Stream:
        if self.gather_stream is None:
            self.gather_stream = torch.cuda.Stream(
                device=self.workspace_identity.device
            )
        return self.gather_stream

    def get_ckv_workspace(self, nbytes: int) -> torch.Tensor:
        if nbytes != self.workspace_pool.slot_nbytes:
            raise ValueError(
                "CKV workspace size changed after the persistent pool was "
                f"allocated: pool={self.workspace_pool.slot_nbytes} requested={nbytes}"
            )
        if self.ckv_workspace is None:
            slot, workspace = self.workspace_pool.acquire()
            self.ckv_workspace_slot = slot
            self.ckv_workspace = workspace
            self.ckv_workspace_generation += 1
        return self.ckv_workspace

    def close(self) -> None:
        # ``Event.wait`` only orders work on the current stream. A released pool
        # slot can be acquired by another execution lane and written from a
        # different stream, so retirement must complete outstanding writers.
        for event, _ in self.pending_layers.values():
            event.synchronize()
        self.pending_layers.clear()
        self.last_layer_idx = None
        if self.ckv_workspace_slot is not None:
            self.workspace_pool.release(self.ckv_workspace_slot)
            self.ckv_workspace_slot = None
            self.ckv_workspace = None


_CKV_PREFETCH_STATE_REGISTRIES: weakref.WeakSet = weakref.WeakSet()


class _CKVPrefetchStateRegistry:
    """Builder-owned states partitioned by lane-scoped CKV workspace."""

    def __init__(self, workspace_pool: _CKVPrefetchWorkspacePool | None = None) -> None:
        self.workspace_pool = workspace_pool
        self.states: dict[_CKVWorkspaceIdentity, _CKVPrefetchState] = (
            workspace_pool._state_map
            if _CKV_POOL_STATE_ENABLED and workspace_pool is not None
            else {}
        )
        _CKV_PREFETCH_STATE_REGISTRIES.add(self)

    def _bind_workspace_pool(self, pool: _CKVPrefetchWorkspacePool) -> None:
        if self.workspace_pool is None:
            if _CKV_POOL_STATE_ENABLED:
                if self.states:
                    raise RuntimeError(
                        "CKV pool-state registry must be empty before first bind"
                    )
                self.states = pool._state_map
            self.workspace_pool = pool
        elif self.workspace_pool is not pool:
            raise RuntimeError("CKV prefetch registry cannot switch workspace pools")
        elif _CKV_POOL_STATE_ENABLED and self.states is not pool._state_map:
            raise RuntimeError("CKV prefetch registry lost its pool-owned state map")

    def _retire(self, identities: list[_CKVWorkspaceIdentity]) -> None:
        for identity in identities:
            self.states.pop(identity).close()

    def _prune_released_workspaces(self) -> None:
        self._retire(
            [
                identity
                for identity, state in self.states.items()
                if state.workspace_storage_ref() is None
            ]
        )

    def begin_step(self) -> None:
        self._prune_released_workspaces()
        for state in self.states.values():
            state.begin_step()

    def clear(self) -> None:
        self._retire(list(self.states))

    def for_workspace(
        self,
        workspace: torch.Tensor,
        layer_idx: int | None = None,
        kv_cache: torch.Tensor | None = None,
        workspace_pool: _CKVPrefetchWorkspacePool | None = None,
    ) -> _CKVPrefetchState:
        self._prune_released_workspaces()
        if workspace_pool is not None:
            self._bind_workspace_pool(workspace_pool)
        if self.workspace_pool is None:
            raise RuntimeError("CKV prefetch registry has no persistent workspace pool")
        identity = _ckv_workspace_identity(workspace)
        state = self.states.get(identity)
        if state is None:
            # A resized view may retain its address, while an allocation
            # replacement changes it. A known layer cache identifies the same
            # execution lane across the latter without merging target/draft.
            stale_identities = [
                existing
                for existing, existing_state in self.states.items()
                if (
                    existing.device == identity.device
                    and existing.data_ptr == identity.data_ptr
                )
                or (
                    layer_idx is not None
                    and kv_cache is not None
                    and layer_idx < len(existing_state.layer_caches)
                    and existing_state.layer_caches[layer_idx] is kv_cache
                )
            ]
            self._retire(stale_identities)
            state = _CKVPrefetchState(identity, workspace, self.workspace_pool)
            self.states[identity] = state
        return state


def _dcp_all_gather_current_stream(
    group,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if not input_tensor.is_contiguous() or not output_tensor.is_contiguous():
        raise ValueError("CKV all-gather tensors must be contiguous")
    if (
        output_tensor.shape[0] != input_tensor.shape[0] * group.world_size
        or output_tensor.shape[1:] != input_tensor.shape[1:]
    ):
        raise ValueError("CKV all-gather tensors have incompatible shapes")

    communicator = getattr(group, "device_communicator", None)
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is not None and not getattr(pynccl_comm, "disabled", False):
        pynccl_comm.all_gather(output_tensor, input_tensor)
        return

    device_group = getattr(group, "device_group", None)
    if device_group is None:
        device_group = getattr(communicator, "device_group", None)
    if device_group is not None:
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=device_group,
            async_op=False,
        )
        return

    gathered = group.all_gather(input_tensor, dim=0)
    output_tensor.copy_(gathered)


@triton.jit
def _mask_page_table_after_nsa_len_kernel(
    page_table_ptr,
    nsa_len_ptr,
    clamp_ptr,
    page_stride0,
    page_stride1,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FUSED_MIN: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < width
    nsa_len = tl.load(nsa_len_ptr + row)
    if FUSED_MIN:
        # Fold the separate torch.minimum(nsa_len, clamp) pass: clamp in
        # place (tile 0 stores; other tiles may load either the original or
        # the already-clamped value -- min(min(a, b), b) == min(a, b), so
        # every tile computes the identical mask bound either way).
        limit = tl.load(clamp_ptr + row)
        nsa_len = tl.minimum(nsa_len, limit)
        if tile == 0:
            tl.store(nsa_len_ptr + row, nsa_len)
    tl.store(
        page_table_ptr + row * page_stride0 + offs * page_stride1,
        -1,
        mask=valid & (offs >= nsa_len),
    )


def _mask_page_table_after_nsa_len(
    page_table: torch.Tensor,
    nsa_cache_seqlens: torch.Tensor,
    *,
    clamp: torch.Tensor | None = None,
) -> None:
    width = page_table.shape[1]
    if width == 0 or page_table.shape[0] == 0:
        return
    if clamp is not None and (
        clamp.dtype != nsa_cache_seqlens.dtype
        or clamp.shape != nsa_cache_seqlens.shape
        or clamp.device != nsa_cache_seqlens.device
    ):
        raise ValueError("fused mask clamp must match the nsa length vector")
    block_n = 128
    _mask_page_table_after_nsa_len_kernel[
        (page_table.shape[0], triton.cdiv(width, block_n))
    ](
        page_table,
        nsa_cache_seqlens,
        nsa_cache_seqlens if clamp is None else clamp,
        page_table.stride(0),
        page_table.stride(1),
        width,
        BLOCK_N=block_n,
        FUSED_MIN=clamp is not None,
    )


@triton.jit
def _reset_page_table_and_counts_kernel(
    out_ptr,
    counts_ptr,
    topk_width,
    out_stride0,
    tiles_per_row,
    BLOCK_N: tl.constexpr,
):
    # One fused reset replacing out.fill_(-1) + valid_counts.zero_(): each
    # program resets one row-tile of the shared page-table buffer to the
    # defensive -1 padding and (tile 0) zeroes that row's count slot. The
    # count buffer is either atomically accumulated (pre-zero required) or
    # fully overwritten by the conversion kernel (pre-zero harmless).
    pid = tl.program_id(0)
    row = pid // tiles_per_row
    tile = pid % tiles_per_row
    offs = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + row * out_stride0 + offs, -1, mask=offs < topk_width
    )
    if tile == 0:
        tl.store(counts_ptr + row, 0)


def _reset_page_table_and_counts(
    out: torch.Tensor, valid_counts: torch.Tensor
) -> None:
    """Fused -1/0 reset for the shared DCP page-table and count buffers."""
    rows, width = out.shape
    if rows == 0:
        return
    block_n = 128
    tiles = triton.cdiv(width, block_n)
    _reset_page_table_and_counts_kernel[(rows * tiles,)](
        out,
        valid_counts,
        width,
        out.stride(0),
        tiles,
        BLOCK_N=block_n,
    )


def _global_causal_lens_for_ckv_gather(
    global_seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id_per_token: torch.Tensor,
    num_actual_toks: int,
) -> torch.Tensor:
    """Return each query token's causal length in the gathered global cache."""
    num_reqs = global_seq_lens.shape[0]
    qsl = query_start_loc[: num_reqs + 1].to(torch.int32)
    req_ids = req_id_per_token[:num_actual_toks].to(torch.int64)
    chunk_start = qsl[:-1][req_ids]
    chunk_len = (qsl[1:] - qsl[:-1])[req_ids]
    full_seq = global_seq_lens[req_ids].to(torch.int32)
    token_idx = torch.arange(
        num_actual_toks,
        device=global_seq_lens.device,
        dtype=torch.int32,
    )
    return full_seq - chunk_len + (token_idx - chunk_start) + 1


@triton.jit
def _map_global_topk_to_gathered_ckv_kernel(
    req_id_ptr,
    token_indices_ptr,
    rank_req_starts_ptr,
    rank_req_lens_ptr,
    out_ptr,
    valid_count_ptr,
    starts_stride0,
    starts_stride1,
    lens_stride0,
    lens_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
    padded_rank_tokens,
    DCP_SIZE: tl.constexpr,
    DCP_INTERLEAVE: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < NUM_TOPK_TOKENS

    req = tl.load(req_id_ptr + row)
    tok = tl.load(
        token_indices_ptr + row * ti_stride0 + cols * ti_stride1,
        mask=col_mask,
        other=-1,
    )
    owner = (tok // DCP_INTERLEAVE) % DCP_SIZE
    local_idx = (
        tok // (DCP_SIZE * DCP_INTERLEAVE)
    ) * DCP_INTERLEAVE + tok % DCP_INTERLEAVE
    req_start = tl.load(
        rank_req_starts_ptr + owner * starts_stride0 + req * starts_stride1,
        mask=col_mask & (tok >= 0),
        other=0,
    )
    req_len = tl.load(
        rank_req_lens_ptr + owner * lens_stride0 + req * lens_stride1,
        mask=col_mask & (tok >= 0),
        other=0,
    )
    valid = col_mask & (tok >= 0) & (local_idx >= 0) & (local_idx < req_len)
    gathered_slot = owner * padded_rank_tokens + req_start + local_idx

    valid_i32 = valid.to(tl.int32)
    local_offset = tl.cumsum(valid_i32) - valid_i32
    tile_valid_count = tl.sum(valid_i32)
    if SINGLE_TILE:
        # Preserve the selected-token order for deterministic gathered-prefill
        # reductions. Each row is owned by one program in this mode.
        output_base = 0
        tl.store(valid_count_ptr + row, tile_valid_count)
    else:
        # The throughput path lets tiles reserve compact output ranges
        # independently; their relative output order is unspecified.
        output_base = tl.atomic_add(valid_count_ptr + row, tile_valid_count)
    output_col = output_base + local_offset
    tl.store(
        out_ptr + row * out_stride0 + output_col * out_stride1,
        gathered_slot,
        mask=valid,
    )


def _map_global_topk_to_gathered_ckv(
    req_ids: torch.Tensor,
    token_indices: torch.Tensor,
    rank_req_starts: torch.Tensor,
    rank_req_lens: torch.Tensor,
    out: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    padded_rank_tokens: int,
) -> None:
    if token_indices.shape != out.shape:
        raise ValueError("CKV gather index output shape does not match top-k input")
    if rank_req_starts.shape != rank_req_lens.shape:
        raise ValueError("CKV gather request starts/lens shapes do not match")
    if rank_req_starts.shape[0] != dcp_size:
        raise ValueError("CKV gather request metadata does not match DCP size")
    if any(
        tensor.dtype != torch.int32
        for tensor in (
            req_ids,
            token_indices,
            rank_req_starts,
            rank_req_lens,
            out,
            valid_counts,
        )
    ):
        raise TypeError("CKV gather index metadata must be int32")
    deterministic_order = (
        os.getenv("VLLM_KVARN_DETERMINISTIC_DCP_COMPACT", "0") == "1"
    )
    block_n = token_indices.shape[1] if deterministic_order else 128
    if not deterministic_order and token_indices.shape[1] % block_n != 0:
        raise ValueError("CKV gather top-k width must be divisible by 128")

    out.fill_(-1)
    if not deterministic_order:
        valid_counts.zero_()
    _map_global_topk_to_gathered_ckv_kernel[
        (
            token_indices.shape[0],
            1 if deterministic_order else token_indices.shape[1] // block_n,
        )
    ](
        req_ids,
        token_indices,
        rank_req_starts,
        rank_req_lens,
        out,
        valid_counts,
        rank_req_starts.stride(0),
        rank_req_starts.stride(1),
        rank_req_lens.stride(0),
        rank_req_lens.stride(1),
        token_indices.stride(0),
        token_indices.stride(1),
        out.stride(0),
        out.stride(1),
        padded_rank_tokens,
        DCP_SIZE=dcp_size,
        DCP_INTERLEAVE=cp_kv_cache_interleave_size,
        NUM_TOPK_TOKENS=token_indices.shape[1],
        BLOCK_N=block_n,
        SINGLE_TILE=deterministic_order,
        num_warps=8 if deterministic_order else 4,
    )


class B12xMLASparseBackend(AttentionBackend):
    """b12x unified sparse-MLA backend (SM120 / SM121).

    Same envelope as ``SparseMLASm120Backend`` (head 576, fp8_ds_mla, block 64,
    index_topk) but driven by b12x's unified decode/extend kernels.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
        "fp8",  # aliases for fp8_ds_mla on this backend
        "fp8_e4m3",
        "kvarn_mla_k2_g64",
        "kvarn_mla_k4_g64",
        "kvarn_mla_k5_g64",
        "kvarn_mla_k2_g128",
        "kvarn_mla_k4_g128",
        "kvarn_mla_k5_g128",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Must equal DeepseekV32IndexerBackend.get_supported_kernel_block_sizes
        # on CUDA (= [64]); the unified b12x decode/extend kernels dispatch
        # page_block_size == 64 natively (matches the fp8_ds_mla layout).
        return [64]

    @staticmethod
    def get_name() -> str:
        return "B12X_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["B12xMLASparseImpl"]:
        return B12xMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["B12xMLASparseMetadataBuilder"]:
        return B12xMLASparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope_head_dim
        # (64) = 576. The unified decode raises on any other q_head_dim.
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The unified b12x kernels gate on
        # get_sm_version(device) >= 120 internally.
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Require an indexer-equipped (index_topk) model, same as SPARSE_MLA_SM120.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            if not hasattr(hf_text_config, "index_topk"):
                return "B12X_MLA_SPARSE requires a model with index_topk config"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # = 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if is_kvarn_mla_cache_dtype(cache_dtype_str):
            from vllm.model_executor.layers.quantization.kvarn.config import (
                KVarNMLAConfig,
            )

            config = KVarNMLAConfig.from_cache_dtype(cache_dtype_str)
            if block_size != config.group:
                raise ValueError(
                    f"MLA KVarN requires block_size={config.group}, got {block_size}"
                )
            return (num_blocks, 1, config.tile_bytes)
        if cache_dtype_str == "fp8_ds_mla":
            # V32 fp8_ds_mla packed: 656 B/token (512 NoPE + 16 inline FP32
            # scales + 128 BF16 RoPE). Mirrors the FlashMLA / SPARSE_MLA_SM120
            # layout; b12x's GLM_NSA decode reads the same record.
            return (num_blocks, block_size, 656)
        if cache_dtype_str == "nvfp4_ds_mla":
            # NVFP4 MLA latent: 256 B E2M1 NoPE data + 32 B E4M3 group-16
            # scales. The stock record has 16 B pad + 128 B BF16 RoPE (432 B).
            # KV_FP8_ROPE=1 reuses the pad for one FP32 amax scale and stores
            # 64 E4M3 bytes at the original RoPE offset (368 B total).
            return (
                num_blocks,
                block_size,
                368 if _kv_fp8_rope_enabled() else 432,
            )
        return (num_blocks, block_size, head_size)


@dataclass
class B12xMLASparseMetadata(AttentionMetadata):
    """Attention metadata for the B12X_MLA_SPARSE backend."""

    num_reqs: int
    max_query_len: int
    max_seq_len: int
    min_seq_len: int
    num_actual_tokens: int
    num_decode_tokens: int
    num_prefill_tokens: int
    # Decode/prefill request counts and the prefill max seq len, part of the
    # MLAAttention.forward_impl metadata contract. B12X routes every token
    # through the top-k MQA path (supports_mha_prefill = False), so
    # prefill_max_seq_len only feeds the (dead) dense-MHA routing check.
    num_decodes: int
    num_prefills: int
    prefill_max_seq_len: int
    # True only for a multi-token speculative-verification batch. Unlike a
    # short chunked prefill, every request has completed its prompt.
    is_spec_decode: bool

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    # DCP keeps global logical top-k ids until forward_mqa maps the entries
    # owned by this rank to local physical slots. These buffers are unnecessary
    # for the direct native-slot path when DCP is disabled.
    req_id_per_token: torch.Tensor | None
    page_table_1: torch.Tensor | None
    nsa_cache_seqlens: torch.Tensor | None
    # Per-request computed KV length (decode cache_seqlens_int32).
    seq_lens: torch.Tensor
    cache_seq_lens_per_req: torch.Tensor
    # Per-token causal KV length consumed directly by the sparse MLA kernel.
    # For pure decode this equals ``seq_lens`` (one token per request).
    cache_seq_lens_per_token: torch.Tensor

    # Transient full-CKV prefill gather metadata.
    ckv_page_table_1: torch.Tensor | None = None
    ckv_nsa_cache_seqlens: torch.Tensor | None = None
    dcp_rank_req_starts: torch.Tensor | None = None
    dcp_rank_req_lens: torch.Tensor | None = None
    dcp_local_cu_seq_lens: torch.Tensor | None = None
    global_cache_seq_lens_per_req: torch.Tensor | None = None
    dcp_local_total_tokens: int = 0
    dcp_padded_total_tokens: int = 0
    dcp_rank_req_page_starts: torch.Tensor | None = None
    dcp_rank_req_page_lens: torch.Tensor | None = None
    dcp_padded_total_pages: int = 0
    dcp_padded_exact_pages: int = 0
    dcp_ckv_gather_eligible: bool = False
    ckv_prefetch_registry: _CKVPrefetchStateRegistry | None = None

    block_size: int = 64
    topk_tokens: int = 2048


class B12xMLASparseMetadataBuilder(AttentionMetadataBuilder[B12xMLASparseMetadata]):
    """Builder for B12X_MLA_SPARSE attention metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    supports_exact_metadata_reuse: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device
        self.kvarn_config = None
        cache_dtype_str = getattr(kv_cache_spec, "cache_dtype_str", None)
        if is_kvarn_mla_cache_dtype(cache_dtype_str):
            from vllm.model_executor.layers.quantization.kvarn.config import (
                KVarNMLAConfig,
            )

            self.kvarn_config = KVarNMLAConfig.from_cache_dtype(cache_dtype_str)
            self.supports_exact_metadata_reuse = False

        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        spec_config = getattr(vllm_config, "speculative_config", None)
        self.num_speculative_tokens = int(
            getattr(spec_config, "num_speculative_tokens", 0) or 0
        )

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        if self.kvarn_config is not None:
            num_kv_pages = vllm_config.cache_config.num_gpu_blocks
            if num_kv_pages is None or num_kv_pages <= 0:
                raise RuntimeError(
                    "KVarN MLA workspace initialization requires the allocated "
                    "local KV page count before metadata builders are created"
                )
            B12xMLASparseImpl.initialize_kvarn_workspaces(
                int(num_kv_pages), device
            )
        from vllm import envs as envs_mod

        ckv_gather_requested = envs_mod.VLLM_B12X_MLA_CKV_GATHER
        self.ckv_prefetch_registry = (
            _CKVPrefetchStateRegistry() if ckv_gather_requested else None
        )
        # Max-batched-token scratch buffers so cudagraph capture sees stable
        # allocations (sliced per build()).
        self.cache_seq_lens_per_token_buffer = torch.empty(
            (max_tokens,), dtype=torch.int32, device=device
        )
        self.cache_seq_lens_per_req_buffer = torch.empty(
            (max_seqs,), dtype=torch.int32, device=device
        )
        if self.dcp_world_size > 1:
            self.req_id_per_token_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.page_table_1_buffer = torch.empty(
                (max_tokens, self.topk_tokens), dtype=torch.int32, device=device
            )
            self.nsa_cache_seqlens_buffer = torch.empty(
                (max_tokens,), dtype=torch.int32, device=device
            )
            self.req_ids_arange = torch.arange(
                max_tokens, dtype=torch.int32, device=device
            )
            if ckv_gather_requested:
                self.ckv_page_table_1_buffer = torch.empty(
                    (max_tokens, self.topk_tokens), dtype=torch.int32, device=device
                )
                self.ckv_nsa_cache_seqlens_buffer = torch.empty(
                    (max_tokens,), dtype=torch.int32, device=device
                )
                self.dcp_rank_req_lens_buffer = torch.empty(
                    (self.dcp_world_size, max_seqs), dtype=torch.int32, device=device
                )
                self.dcp_rank_req_starts_buffer = torch.empty(
                    (self.dcp_world_size, max_seqs), dtype=torch.int32, device=device
                )
                self.dcp_local_cu_seq_lens_buffer = torch.empty(
                    (max_seqs + 1,), dtype=torch.int32, device=device
                )
                self.dcp_rank_req_page_lens_buffer = torch.empty(
                    (self.dcp_world_size, max_seqs),
                    dtype=torch.int32,
                    device=device,
                )
                self.dcp_rank_req_page_starts_buffer = torch.empty(
                    (self.dcp_world_size, max_seqs),
                    dtype=torch.int32,
                    device=device,
                )
            else:
                self.ckv_page_table_1_buffer = None
                self.ckv_nsa_cache_seqlens_buffer = None
                self.dcp_rank_req_lens_buffer = None
                self.dcp_rank_req_starts_buffer = None
                self.dcp_local_cu_seq_lens_buffer = None
                self.dcp_rank_req_page_lens_buffer = None
                self.dcp_rank_req_page_starts_buffer = None
        else:
            self.req_id_per_token_buffer = None
            self.page_table_1_buffer = None
            self.nsa_cache_seqlens_buffer = None
            self.req_ids_arange = None
            self.ckv_page_table_1_buffer = None
            self.ckv_nsa_cache_seqlens_buffer = None
            self.dcp_rank_req_lens_buffer = None
            self.dcp_rank_req_starts_buffer = None
            self.dcp_local_cu_seq_lens_buffer = None
            self.dcp_rank_req_page_lens_buffer = None
            self.dcp_rank_req_page_starts_buffer = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12xMLASparseMetadata:
        cm = common_attn_metadata
        if self.kvarn_config is not None:
            from vllm.v1.attention.backends.mla.kvarn_mla_state import (
                KVarNMLAStateManager,
            )

            KVarNMLAStateManager.prepare_step(
                tuple(self.layer_names),
                self.layer_names,
                cm,
                self.kvarn_config,
                self.dcp_world_size,
            )
        num_tokens = cm.num_actual_tokens
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            num_decodes = cm.num_reqs
            num_prefills = 0
            num_decode_tokens = num_tokens
            num_prefill_tokens = 0
        elif cm.batch_topology is not None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                cm.batch_topology.split_decodes_and_prefills(
                    cm,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=True,
                )
            )
        else:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    cm,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=True,
                )
            )
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        is_spec_decode = False
        if (
            self.num_speculative_tokens > 0
            and 1 < cm.max_query_len <= self.num_speculative_tokens + 1
            and cm.is_prefilling is not None
        ):
            is_spec_decode = not bool(torch.any(cm.is_prefilling[: cm.num_reqs]))

        use_dcp = self.dcp_world_size > 1
        seq_lens_for_req = (
            cm.dcp_local_seq_lens
            if use_dcp and cm.dcp_local_seq_lens is not None
            else cm.seq_lens
        )
        req_id_per_token_tensor = None
        dcp_rank_req_lens = None
        dcp_rank_req_starts = None
        dcp_local_cu_seq_lens = None
        dcp_local_total_tokens = 0
        dcp_padded_total_tokens = 0
        dcp_rank_req_page_lens = None
        dcp_rank_req_page_starts = None
        dcp_padded_total_pages = 0
        dcp_padded_exact_pages = 0
        dcp_ckv_gather_eligible = False

        from vllm import envs as envs_mod

        if self.ckv_prefetch_registry is not None:
            self.ckv_prefetch_registry.begin_step()

        if (
            use_dcp
            and envs_mod.VLLM_B12X_MLA_CKV_GATHER
            and num_decode_tokens == 0
            and num_prefill_tokens == num_tokens
            and cm.max_query_len > envs_mod.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS
        ):
            assert self.dcp_rank_req_lens_buffer is not None
            assert self.dcp_rank_req_starts_buffer is not None
            assert self.dcp_local_cu_seq_lens_buffer is not None
            assert self.dcp_rank_req_page_lens_buffer is not None
            assert self.dcp_rank_req_page_starts_buffer is not None
            seq_lens_cpu_src = (
                cm._seq_lens_cpu
                if cm._seq_lens_cpu is not None
                else cm.seq_lens_cpu
            )
            global_seq_lens = cm.seq_lens[: cm.num_reqs]
            all_rank_lens = get_dcp_local_seq_lens(
                global_seq_lens,
                self.dcp_world_size,
                dcp_rank=None,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            ).transpose(0, 1)
            global_seq_lens_cpu = seq_lens_cpu_src[: cm.num_reqs]
            all_rank_lens_cpu = get_dcp_local_seq_lens(
                global_seq_lens_cpu,
                self.dcp_world_size,
                dcp_rank=None,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            ).transpose(0, 1)
            dcp_rank_req_lens = self.dcp_rank_req_lens_buffer[
                : self.dcp_world_size, : cm.num_reqs
            ]
            dcp_rank_req_lens.copy_(all_rank_lens)
            dcp_rank_req_starts = self.dcp_rank_req_starts_buffer[
                : self.dcp_world_size, : cm.num_reqs
            ]
            dcp_rank_req_starts[:, 0].zero_()
            if cm.num_reqs > 1:
                torch.cumsum(
                    dcp_rank_req_lens[:, :-1],
                    dim=1,
                    out=dcp_rank_req_starts[:, 1:],
                )
            dcp_rank_req_page_lens = self.dcp_rank_req_page_lens_buffer[
                : self.dcp_world_size, : cm.num_reqs
            ]
            torch.add(
                dcp_rank_req_lens,
                self.kv_cache_spec.block_size - 1,
                out=dcp_rank_req_page_lens,
            )
            torch.div(
                dcp_rank_req_page_lens,
                self.kv_cache_spec.block_size,
                rounding_mode="floor",
                out=dcp_rank_req_page_lens,
            )
            dcp_rank_req_page_starts = self.dcp_rank_req_page_starts_buffer[
                : self.dcp_world_size, : cm.num_reqs
            ]
            dcp_rank_req_page_starts[:, 0].zero_()
            if cm.num_reqs > 1:
                torch.cumsum(
                    dcp_rank_req_page_lens[:, :-1],
                    dim=1,
                    out=dcp_rank_req_page_starts[:, 1:],
                )

            dcp_local_cu_seq_lens = self.dcp_local_cu_seq_lens_buffer[: cm.num_reqs + 1]
            dcp_local_cu_seq_lens[0].zero_()
            torch.cumsum(
                dcp_rank_req_lens[self.dcp_rank],
                dim=0,
                out=dcp_local_cu_seq_lens[1:],
            )
            rank_totals = all_rank_lens_cpu.sum(dim=1).tolist()
            dcp_local_total_tokens = int(rank_totals[self.dcp_rank])
            dcp_padded_total_tokens = (
                _cdiv(
                    max(int(total) for total in rank_totals),
                    self.kv_cache_spec.block_size,
                )
                * self.kv_cache_spec.block_size
            )
            rank_page_totals = (
                (all_rank_lens_cpu + self.kv_cache_spec.block_size - 1)
                // self.kv_cache_spec.block_size
            ).sum(dim=1)
            dcp_padded_total_pages = max(
                int(total) for total in rank_page_totals.tolist()
            )
            dcp_padded_exact_pages = cm.num_reqs * 16
            dcp_ckv_gather_eligible = dcp_padded_total_tokens > 0

        # Per-token causal KV length. In pure decode the common metadata already
        # has exactly the graph-stable tensor both b12x consumers need, so bind it
        # directly instead of staging two identical D2D copies.
        if cm.max_query_len <= 1 and num_tokens == cm.num_reqs:
            if use_dcp:
                assert self.req_ids_arange is not None
                req_id_per_token_tensor = self.req_ids_arange[:num_tokens]
                self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                    seq_lens_for_req[:num_tokens], non_blocking=True
                )
                self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                    seq_lens_for_req[: cm.num_reqs], non_blocking=True
                )
                cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[
                    :num_tokens
                ]
                cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[
                    : cm.num_reqs
                ]
            else:
                cache_seq_lens_per_token = seq_lens_for_req[:num_tokens]
                cache_seq_lens_per_req = seq_lens_for_req[: cm.num_reqs]
        else:
            if cm.batch_topology is not None:
                starts = cm.batch_topology.query_start_loc_np[: cm.num_reqs + 1]
                query_lens = cm.batch_topology.query_lens_np
                req_id_per_token_np = cm.batch_topology.req_id_per_token_np
            else:
                starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
                query_lens = np.diff(starts)
                req_id_per_token_np = np.repeat(
                    np.arange(cm.num_reqs, dtype=np.int32), query_lens
                )
            num_query_tokens = int(starts[-1])
            if num_query_tokens > num_tokens:
                raise RuntimeError(
                    "B12X sparse MLA metadata received query_start_loc with "
                    f"{num_query_tokens} tokens, exceeding padded capacity "
                    f"{num_tokens}"
                )

            req_ids = None
            if use_dcp:
                req_ids = np.zeros((num_tokens,), dtype=np.int32)
                if num_query_tokens:
                    req_ids[:num_query_tokens] = req_id_per_token_np

            if not use_dcp and cm.positions is not None and cm.positions.ndim == 1:
                # Async scheduling intentionally exposes only an optimistic CPU
                # sequence-length bound. That bound can lag when a finished slot is
                # recycled for a shorter request, so it is not a valid causal mask
                # for multi-token verification. Positions are authoritative on the
                # GPU and give the exact per-token KV length for DCP1.
                per_token_lens_t = cm.positions[:num_tokens].to(torch.int32) + 1
            else:
                # DCP needs rank-local lengths rather than global positions. Avoid
                # the blocking lazy seq_lens D2H copy and convert the scheduler's
                # conservative CPU lengths to each rank's local interleaving.
                seq_lens_cpu_src = (
                    cm.seq_lens_cpu_upper_bound
                    if cm.seq_lens_cpu_upper_bound is not None
                    else cm.seq_lens_cpu
                )
                seq_lens_cpu = seq_lens_cpu_src.numpy().astype(np.int32, copy=False)
                per_token_lens = np.zeros((num_tokens,), dtype=np.int32)
                for req_id, q_len in enumerate(query_lens):
                    if q_len <= 0:
                        continue
                    start = int(starts[req_id])
                    end = int(starts[req_id + 1])
                    context_len = int(seq_lens_cpu[req_id]) - int(q_len)
                    if use_dcp:
                        global_per_token_lens = torch.arange(
                            context_len + 1,
                            context_len + int(q_len) + 1,
                            dtype=torch.int32,
                        )
                        per_token_lens[start:end] = get_dcp_local_seq_lens(
                            global_per_token_lens,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        ).numpy()
                    else:
                        per_token_lens[start:end] = np.arange(
                            context_len + 1,
                            context_len + int(q_len) + 1,
                            dtype=np.int32,
                        )

                per_token_lens_t = torch.from_numpy(per_token_lens)
                if per_token_lens_t.device.type == "cpu":
                    per_token_lens_t = per_token_lens_t.pin_memory()
            if req_ids is not None:
                assert self.req_id_per_token_buffer is not None
                req_ids_t = torch.from_numpy(req_ids)
                if req_ids_t.device.type == "cpu":
                    req_ids_t = req_ids_t.pin_memory()
                self.req_id_per_token_buffer[:num_tokens].copy_(
                    req_ids_t, non_blocking=True
                )
                req_id_per_token_tensor = self.req_id_per_token_buffer[:num_tokens]
            self.cache_seq_lens_per_token_buffer[:num_tokens].copy_(
                per_token_lens_t, non_blocking=True
            )
            self.cache_seq_lens_per_req_buffer[: cm.num_reqs].copy_(
                seq_lens_for_req[: cm.num_reqs], non_blocking=True
            )
            cache_seq_lens_per_token = self.cache_seq_lens_per_token_buffer[:num_tokens]
            cache_seq_lens_per_req = self.cache_seq_lens_per_req_buffer[: cm.num_reqs]

        return B12xMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            # cm.seq_lens_cpu performs a device->host copy of seq_lens into
            # pageable memory. During CUDA graph capture (the FixedMTP3
            # proposer supergraph builds draft attention metadata inside the
            # capture region) such a copy is illegal ("unless the CPU tensor
            # is pinned"), and an exact value cannot be synced under capture
            # anyway. min_seq_len only gates native-CKV-gather eligibility,
            # whose fallback path is always correct: capture with 0 (native
            # gather ineligible) and keep the exact eager value otherwise.
            min_seq_len=(
                0
                if torch.cuda.is_current_stream_capturing()
                else int(cm.seq_lens_cpu[: cm.num_reqs].min().item())
            ),
            num_actual_tokens=num_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            prefill_max_seq_len=cm.max_seq_len if num_prefills > 0 else 0,
            is_spec_decode=is_spec_decode,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token_tensor,
            page_table_1=(
                self.page_table_1_buffer[:num_tokens]
                if self.page_table_1_buffer is not None
                else None
            ),
            nsa_cache_seqlens=(
                self.nsa_cache_seqlens_buffer[:num_tokens]
                if self.nsa_cache_seqlens_buffer is not None
                else None
            ),
            seq_lens=cache_seq_lens_per_req,
            cache_seq_lens_per_req=cache_seq_lens_per_req,
            cache_seq_lens_per_token=cache_seq_lens_per_token,
            ckv_page_table_1=(
                self.ckv_page_table_1_buffer[:num_tokens]
                if self.ckv_page_table_1_buffer is not None
                else None
            ),
            ckv_nsa_cache_seqlens=(
                self.ckv_nsa_cache_seqlens_buffer[:num_tokens]
                if self.ckv_nsa_cache_seqlens_buffer is not None
                else None
            ),
            dcp_rank_req_starts=dcp_rank_req_starts,
            dcp_rank_req_lens=dcp_rank_req_lens,
            dcp_local_cu_seq_lens=dcp_local_cu_seq_lens,
            global_cache_seq_lens_per_req=(
                cm.seq_lens[: cm.num_reqs]
                if use_dcp and envs_mod.VLLM_B12X_MLA_CKV_GATHER
                else None
            ),
            dcp_local_total_tokens=dcp_local_total_tokens,
            dcp_padded_total_tokens=dcp_padded_total_tokens,
            dcp_rank_req_page_starts=dcp_rank_req_page_starts,
            dcp_rank_req_page_lens=dcp_rank_req_page_lens,
            dcp_padded_total_pages=dcp_padded_total_pages,
            dcp_padded_exact_pages=dcp_padded_exact_pages,
            dcp_ckv_gather_eligible=dcp_ckv_gather_eligible,
            ckv_prefetch_registry=self.ckv_prefetch_registry,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class B12xMLASparseImpl(MLAAttentionImpl[B12xMLASparseMetadata]):
    """b12x unified sparse-MLA implementation (decode + extend/prefill)."""

    is_sparse: ClassVar[bool] = True
    can_return_lse_for_decode: bool = True
    # B12X handles decode and extend inside its own top-k MQA kernels; the
    # generic dense-MHA prefill path assumes cache layouts it never validated.
    supports_mha_prefill: bool = False
    supports_dcp_project_before_merge: bool = True
    supports_dcp_gather_query_in_workspace: bool = True
    supports_dcp_project_before_merge_in_workspace: bool = True
    supports_dcp_reduce_scatter_output_in_workspace: bool = True
    # Cross-layer CKV prefetch state must exist before the first backend
    # instance so profile-cache cleanup is independent of construction order.
    _all_layer_kv_caches: list[torch.Tensor | None] = []
    _shared_gather_event: torch.cuda.Event | None = None
    _shared_gather_buf_idx: int = 0
    _kvarn_shared_dense: ClassVar[dict[tuple, torch.Tensor | None]] = {}
    _kvarn_shared_direct_stage: ClassVar[dict[tuple, torch.Tensor]] = {}
    _kvarn_shared_selected_stage: ClassVar[dict[tuple, torch.Tensor]] = {}
    _kvarn_shared_selected_physical_slots: ClassVar[
        dict[tuple, torch.Tensor]
    ] = {}
    _kvarn_shared_remapped: ClassVar[dict[tuple, torch.Tensor]] = {}
    _kvarn_shared_rotated: ClassVar[dict[tuple, torch.Tensor]] = {}
    _kvarn_shared_physical_slots: ClassVar[dict[tuple, torch.Tensor]] = {}
    _kvarn_instances: ClassVar[weakref.WeakSet] = weakref.WeakSet()
    _kvarn_hadamard: ClassVar[dict[torch.device, torch.Tensor]] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "B12X_MLA_SPARSE does not support alibi_slopes / sliding_window "
                "/ logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_MLA_SPARSE only supports decoder self-attention"
            )

        self.layer_name = ""
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self._is_kvarn_mla = is_kvarn_mla_cache_dtype(self.kv_cache_dtype)
        self._kvarn_config = None
        if self._is_kvarn_mla:
            from vllm.model_executor.layers.quantization.kvarn.config import (
                KVarNMLAConfig,
            )

            self._kvarn_config = KVarNMLAConfig.from_cache_dtype(self.kv_cache_dtype)
        self._kv_fp8_rope = bool(
            self.kv_cache_dtype == "nvfp4_ds_mla" and _kv_fp8_rope_enabled()
        )
        if _KV_FP8_ROPE_REQUESTED and not _is_glm_moe_dsa_model():
            logger.warning(
                "KV_FP8_ROPE=1 ignored: compact MLA records are restricted to "
                "model_type=glm_moe_dsa and its associated MTP draft"
            )
        if _kv_fp8_rope_enabled() and self.kv_cache_dtype != "nvfp4_ds_mla":
            logger.warning(
                "KV_FP8_ROPE=1 has no effect for kv_cache_dtype=%s; the compact "
                "record is GLM nvfp4_ds_mla-only",
                self.kv_cache_dtype,
            )
        if self._kv_fp8_rope:
            try:
                from sparkinfer.attention._shared.mla.kv_cache import (
                    concat_and_cache_nvfp4_mla_fp8_rope,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "KV_FP8_ROPE=1 requires a SparkInfer build with "
                    "concat_and_cache_nvfp4_mla_fp8_rope package API support"
                ) from exc
            self._concat_and_cache_nvfp4_mla_fp8_rope = (
                concat_and_cache_nvfp4_mla_fp8_rope
            )

        # MLA dims (absorbed: Q post-projection is [T, H, kv_lora_rank + rope]).
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.v_head_dim: int = mla_args.get("v_head_dim", 512)
        # GLM_NSA contract: q_head_dim = kv_lora_rank (512) + qk_rope (64) = 576.
        self.q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # Query absorption sees a head-major, non-contiguous Q view. Some
        # cuBLAS BF16 BMM kernels read past tight custom allocations, so route
        # this BMM through the safe query op instead of materializing Q.
        self.use_safe_mla_query_bmm = True
        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        assert self.topk_indices_buffer is not None, (
            "B12X_MLA_SPARSE requires sparse-MLA top-k indices "
            "(model with index_topk in its config)."
        )
        self.topk_tokens = int(self.topk_indices_buffer.shape[-1])

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self._vllm_config = vllm_config
        parallel_config = vllm_config.parallel_config
        self.dcp_workspace_non_dbo = not bool(parallel_config.enable_dbo)
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self._head_major_mla_output = True
        self.tp_world_size = int(parallel_config.tensor_parallel_size)
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.need_to_return_lse_for_decode = self.dcp_world_size > 1

        expects_physical_slots = self.dcp_world_size == 1
        if (
            indexer is not None
            and bool(indexer.output_physical_slots) != expects_physical_slots
        ):
            expected = "physical" if expects_physical_slots else "logical"
            raise RuntimeError(
                f"B12X_MLA_SPARSE requires {expected} sparse-indexer output "
                f"when dcp_world_size={self.dcp_world_size}"
            )

        scheduler_config = vllm_config.scheduler_config
        self.device = torch.device(f"cuda:{torch.accelerator.current_device_index()}")
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        self.block_size = 64
        # NVFP4 MLA record selection: ScaleFormat.NVFP4_E4M3 (2) rides every
        # decode/extend call so the CuTeDSL kernels specialize on the packed
        # E2M1+E4M3 record instead of the 656 B fp8_ds_mla record. KV_FP8_ROPE
        # only changes its RoPE tail; the latent format and outer-scale
        # correction are deliberately untouched.
        self._b12x_scale_format = 2 if self.kv_cache_dtype == "nvfp4_ds_mla" else None
        self._kv_record_bytes = (
            (368 if self._kv_fp8_rope else 432)
            if self.kv_cache_dtype == "nvfp4_ds_mla"
            else 656
        )
        logger.info(
            "B12X GLM MLA KV format: KV_FP8_ROPE=%d kv_gmem_stride=%d "
            "kv_cache_dtype=%s",
            int(self._kv_fp8_rope),
            self._kv_record_bytes,
            self.kv_cache_dtype,
        )
        # MLAAttention all-gathers the local query-head shard before entering a
        # DCP backend. The kernel must therefore plan for, and return, the full
        # gathered head set; the outer layer reduces/scatters it back afterward.
        self._input_num_heads = self.num_heads * self.dcp_world_size
        dcp_comm_backend = getattr(parallel_config, "dcp_comm_backend", None)
        model_config = vllm_config.model_config
        self._kvarn_direct_target_model = bool(
            model_config is not None
            and getattr(model_config.hf_config, "model_type", None)
            == "glm_moe_dsa"
        )
        native_ckv_mode = os.getenv(
            "VLLM_KVARN_MLA_NATIVE_CKV_GATHER", "0"
        ).strip()
        if native_ckv_mode not in {"0", "1"}:
            raise ValueError(
                "VLLM_KVARN_MLA_NATIVE_CKV_GATHER must be exactly 0 or 1"
            )
        native_ckv_requested = native_ckv_mode == "1"
        self._kvarn_precision_tail_tokens = int(
            os.getenv("KVARN_MLA_PRECISION_TAIL_TOKENS", "0")
        )
        if self._kvarn_precision_tail_tokens < 0:
            raise ValueError("KVARN_MLA_PRECISION_TAIL_TOKENS must be non-negative")
        native_ckv_supported = (
            self._is_kvarn_mla
            and self._kvarn_direct_target_model
            and self._kvarn_config is not None
            and self._kvarn_config.bits == 4
            and self._kvarn_config.group == 64
            and self._kvarn_config.boundary_tokens <= 128
            and self._kvarn_precision_tail_tokens <= 3072
            and self.block_size == 64
            and self.dcp_world_size == 4
            and torch.cuda.get_device_capability(self.device) == (12, 0)
        )
        hf_model_type = getattr(
            getattr(model_config, "hf_config", None), "model_type", None
        )
        # The MTP draft (deepseek_mtp) shares this backend and KVarN geometry
        # but never runs target CKV history gather; requesting native gather
        # there simply disables it instead of failing the worker.
        if (
            native_ckv_requested
            and not native_ckv_supported
            and hf_model_type != "deepseek_mtp"
        ):
            raise RuntimeError(
                "native CKV gather requires GLM K4/G64/block64/DCP4 on SM120: "
                f"is_kvarn={self._is_kvarn_mla} "
                f"direct_target={self._kvarn_direct_target_model} "
                f"model_type={hf_model_type} "
                f"bits={getattr(self._kvarn_config, 'bits', None)} "
                f"group={getattr(self._kvarn_config, 'group', None)} "
                f"boundary={getattr(self._kvarn_config, 'boundary_tokens', None)} "
                f"tail={self._kvarn_precision_tail_tokens} "
                f"block={self.block_size} dcp={self.dcp_world_size} "
                f"cap={torch.cuda.get_device_capability(self.device)}"
            )
        self._kvarn_native_ckv_gather = (
            native_ckv_requested and native_ckv_supported
        )
        native_mode = os.getenv("VLLM_KVARN_MLA_NATIVE_CUTE", "0").strip()
        if native_mode not in {"0", "1"}:
            raise ValueError("VLLM_KVARN_MLA_NATIVE_CUTE must be exactly 0 or 1")
        self._kvarn_native_packed_decode = None
        if native_mode == "1":
            from sparkinfer.attention.kvarn_mla import native_packed_k5_decode

            self._kvarn_native_packed_decode = native_packed_k5_decode
        direct_packed_mode = os.getenv(
            "VLLM_KVARN_MLA_DIRECT_PACKED_DECODE", "0"
        ).strip()
        if direct_packed_mode not in {"0", "1"}:
            raise ValueError(
                "VLLM_KVARN_MLA_DIRECT_PACKED_DECODE must be exactly 0 or 1"
            )
        direct_packed_requested = direct_packed_mode == "1"
        has_selected_buffer = _direct_packed_selected_buffer_valid(
            self.topk_indices_buffer, self.device
        )
        self._kvarn_direct_packed_decode = (
            _enable_direct_packed_kvarn_mla_instance(
                requested=direct_packed_requested,
                has_selected_buffer=has_selected_buffer,
                dcp_comm_backend=dcp_comm_backend,
            )
        )
        if direct_packed_requested and not self._kvarn_direct_packed_decode:
            logger.info(
                "Direct packed KVarN MLA decode is disabled without a valid "
                "selected-index buffer and target A2A ownership "
                "(selected_buffer_valid=%s, dcp_comm_backend=%r); the existing "
                "KVarN path remains active.",
                has_selected_buffer,
                dcp_comm_backend,
            )
        if self._kvarn_direct_packed_decode:
            config = self._kvarn_config
            geometry = None
            if config is not None:
                geometry = (
                    config.group,
                    config.latent_dim,
                    config.rope_dim,
                    config.bits,
                    config.tile_bytes,
                )
            capability = torch.cuda.get_device_capability(self.device)
            contract = {
                "kv_cache_dtype": self.kv_cache_dtype,
                "geometry": geometry,
                "tp_world_size": self.tp_world_size,
                "dcp_world_size": self.dcp_world_size,
                "dcp_comm_backend": dcp_comm_backend,
                "local_heads": self.num_heads,
                "gathered_heads": self._input_num_heads,
                "topk_tokens": self.topk_tokens,
                "device_capability": capability,
                "dcp_a2a_max_tokens": envs.VLLM_DCP_A2A_MAX_TOKENS,
                "dcp_a2a_large_backend": envs.VLLM_DCP_A2A_LARGE_BACKEND,
                "selected_buffer_valid": has_selected_buffer,
            }
            if (
                not self._is_kvarn_mla
                or geometry
                not in {
                    (64, 512, 64, 2, 18_560),
                    (64, 512, 64, 4, 26_752),
                    (64, 512, 64, 5, 30_848),
                }
                or self.tp_world_size != 4
                or self.dcp_world_size != 4
                or (
                    dcp_comm_backend != "a2a"
                    and envs.VLLM_DCP_A2A_LARGE_BACKEND != "a2a"
                )
                or not _direct_packed_kvarn_effective_a2a()
                or self._input_num_heads != 64
                or self.topk_tokens != 2048
                or capability != (12, 0)
                or not math.isfinite(self.scale)
                or self.scale <= 0
            ):
                raise ValueError(
                    "Direct packed KVarN MLA decode requires the exact GLM "
                    "K4/K5 G64 SM120 TP4/DCP4/A2A/H16->H64/topk2048 "
                    f"contract; got {contract}"
                )

        # KVarN prefill/decode serving is validated only on topologies where
        # one of the two gated mechanisms engages:
        #   * transient full-CKV gather (requires dcp_world_size > 1; see the
        #     _ckv_gather_enabled predicate below), or
        #   * direct packed decode (requires the exact TP4/DCP4/A2A contract
        #     above).
        # At dcp_world_size == 1 both reject and forward_mqa falls back to the
        # legacy composite (physical-slot indexer emission + full-arena
        # stage_physical_kvarn_mla_fp8 staging + the DCP1-only fused query
        # epilogue). That composite has no validated serving configuration:
        # every DCP4 run emits logical indexer slots and gathers CKV. It faulted
        # with cudaErrorIllegalAddress during the first real forward (engine
        # warmup prefill) in records/k2-dcp1-service.log, so fail closed at
        # configuration time instead of crashing the worker. The escape hatch
        # exists for single-process forensics probes only.
        if (
            self._is_kvarn_mla
            and self.dcp_world_size < 2
            and os.getenv("VLLM_KVARN_MLA_ALLOW_DCP1_FALLBACK", "0") != "1"
        ):
            raise RuntimeError(
                "kv_cache_dtype=kvarn_mla_* with B12X_MLA_SPARSE requires "
                "decode_context_parallel_size >= 2 on this build: at DCP1 both "
                "validated KVarN prefill mechanisms reject (CKV gather needs "
                "dcp>1; direct packed decode needs the TP4/DCP4/A2A contract) "
                "and the legacy physical-slot fallback faults on the first "
                "real forward. Re-run with --decode-context-parallel-size 4, or "
                "set VLLM_KVARN_MLA_ALLOW_DCP1_FALLBACK=1 to enter the "
                "unvalidated fallback for forensics."
            )
        if self._is_kvarn_mla and self.dcp_world_size == 2:
            logger.info_once(
                "KVarN MLA at decode_context_parallel_size=2 uses the canonical "
                "full-CKV gather path; this topology is unvalidated on this "
                "build (validated: DCP4)."
            )

        # Split-K cap: ceil(topk / tile). Bounds the borrowed mid_out/mid_lse
        # chunk dim and the workspace max_chunks_per_row.
        self._num_splits_cap = max(1, _cdiv(self.topk_tokens, _DECODE_SPLIT_TILE))
        self._kernel_num_heads = (
            _cdiv(self._input_num_heads, _HEAD_ALIGNMENT) * _HEAD_ALIGNMENT
        )
        self._pad_heads = self._kernel_num_heads != self._input_num_heads

        self.spec_decode_max_q = _env_int("VLLM_B12X_MLA_SPEC_DECODE_MAX_Q", 8)
        spec_decode_mode = (
            os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "auto").strip().lower()
        )
        disabled_modes = {"0", "false", "off", "no"}
        forced_modes = {"1", "true", "on", "yes"}
        if spec_decode_mode not in {"auto", *disabled_modes, *forced_modes}:
            raise ValueError(
                "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE must be auto, 0, or 1 "
                f"(got {spec_decode_mode!r})"
            )
        self.spec_extend_as_decode = spec_decode_mode not in disabled_modes
        self.spec_extend_as_decode_force = spec_decode_mode in forced_modes

        # Decode query rows per request (1, plus speculative draft tokens).
        q_per_req = 1
        spec = getattr(vllm_config, "speculative_config", None)
        if (
            self.spec_extend_as_decode
            and spec is not None
            and getattr(spec, "num_speculative_tokens", None)
        ):
            q_per_req = 1 + int(spec.num_speculative_tokens)
        # Auto mode only dispatches genuine verifier batches, whose maximum
        # row count is fixed by speculative_config. The explicit force mode
        # may also route arbitrary short extends and therefore reserves the
        # full operator limit.
        if self.spec_extend_as_decode_force:
            q_per_req = max(q_per_req, self.spec_decode_max_q)
        if self._kvarn_direct_packed_decode and max_num_seqs == 1:
            q_per_req = _direct_packed_kvarn_mla_rows(
                use_decode_kernel=True,
                direct_packed_enabled=True,
                num_actual_toks=q_per_req,
            )
        self._decode_max_rows = min(max_num_seqs * q_per_req, max_batched)
        if self._decode_max_rows < max_num_seqs:
            self._decode_max_rows = max_num_seqs

        self._max_batched = int(max_batched)
        self._kvarn_group_key: tuple[str, ...] | None = None
        self._kvarn_cache_ref: torch.Tensor | None = None
        self._kvarn_block_to_slot: torch.Tensor | None = None
        self._kvarn_block_to_logical: torch.Tensor | None = None
        if self._is_kvarn_mla:
            assert self._kvarn_config is not None
            kvarn_config = self._kvarn_config
            max_rollback_tokens = (
                int(spec.num_speculative_tokens)
                if vllm_config.scheduler_config.async_scheduling
                and spec is not None
                and getattr(spec, "num_speculative_tokens", None)
                else 0
            )
            self._kvarn_pool_size = kvarn_config.pool_slots(
                max_num_seqs,
                max_batched,
                max_rollback_tokens,
            )
            self._kvarn_latent_pool = torch.zeros(
                self._kvarn_pool_size,
                kvarn_config.group,
                kvarn_config.latent_dim,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            self._kvarn_rope_pool = torch.zeros(
                self._kvarn_pool_size,
                kvarn_config.group,
                kvarn_config.rope_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._kvarn_boundary_blocks = max_num_seqs
            self._kvarn_rollback_blocks = (
                _cdiv(
                    max_num_seqs * max_rollback_tokens,
                    kvarn_config.group,
                )
                + max_num_seqs
                if max_rollback_tokens
                else 0
            )
            self._kvarn_dense_cache: torch.Tensor | None = None
            self._kvarn_direct_stage: torch.Tensor | None = None
            self._kvarn_selected_stage: torch.Tensor | None = None
            self._kvarn_selected_physical_slots: torch.Tensor | None = None
            self._kvarn_remapped_indices: torch.Tensor | None = None
            self._kvarn_rotated_scratch: torch.Tensor | None = None
            self._kvarn_physical_slots: torch.Tensor | None = None
            cls = type(self)
            cls._kvarn_instances.add(self)
            if self.device not in cls._kvarn_hadamard:
                hadamard = torch.ones(1, 1, dtype=torch.float32)
                while hadamard.shape[0] < kvarn_config.latent_dim:
                    hadamard = torch.cat(
                        [
                            torch.cat([hadamard, hadamard], dim=1),
                            torch.cat([hadamard, -hadamard], dim=1),
                        ],
                        dim=0,
                    )
                cls._kvarn_hadamard[self.device] = (
                    hadamard.div(math.sqrt(kvarn_config.latent_dim))
                    .to(self.device, dtype=torch.bfloat16)
                    .contiguous()
                )
            self._kvarn_h = cls._kvarn_hadamard[self.device]
            from vllm.v1.attention.backends.mla.kvarn_mla_state import (
                KVarNMLAStateManager,
            )

            KVarNMLAStateManager.register(self)
        else:
            self._kvarn_pool_size = 0

        # Lazily import SparkInfer only on this opt-in path.
        from sparkinfer.attention.sparse_mla import (
            Caps as B12XSparseMLAScratchCaps,
        )
        from sparkinfer.attention.sparse_mla import (
            plan as plan_sparse_mla_scratch,
        )
        from sparkinfer.attention.sparse_mla import (
            run_decode as sparse_mla_decode_forward,
        )
        from sparkinfer.attention.sparse_mla import (
            run_extend as sparse_mla_extend_forward,
        )

        self._sparse_mla_decode_forward = sparse_mla_decode_forward
        self._sparse_mla_extend_forward = sparse_mla_extend_forward

        if self._b12x_scale_format is not None:
            required_kwargs = {"latent_scale", "scale_format"}
            unsupported_forwards = [
                mode
                for mode, forward in (
                    ("decode", sparse_mla_decode_forward),
                    ("extend", sparse_mla_extend_forward),
                )
                if not required_kwargs.issubset(inspect.signature(forward).parameters)
            ]
            if unsupported_forwards:
                raise RuntimeError(
                    "B12X_MLA_SPARSE with kv_cache_dtype='nvfp4_ds_mla' "
                    "requires a b12x build with NVFP4 sparse-MLA API support; "
                    "unsupported forwards: " + ", ".join(unsupported_forwards)
                )

        # Eager PLAN -> BIND -> KERNEL (no b12x workspace/arena, ever). We build a
        # caller-owned-scratch PLAN once per mode; each forward maps a vLLM
        # workspace-manager scratch tensor into a plain B12XSparseMLAScratch views
        # CONTAINER via plan.bind(). The unified SM120 sparse-MLA decode/extend
        # kernels duck-type the container's tmp_output/tmp_lse/output_buffer/
        # final_lse fields. The planner fixes the split count for each captured
        # graph and the merge specializes on that count, so no device-side control
        # scalar initialization is needed. final_lse is pre-materialized as a view
        # so the legacy lazy torch.empty(final_lse) never fires during capture.
        def _make_plan(
            mode: str, max_q_rows: int, num_q_heads: int, max_batch: int
        ) -> Any:
            return plan_sparse_mla_scratch(
                B12XSparseMLAScratchCaps(
                    device=self.device,
                    num_q_heads=int(num_q_heads),
                    max_q_rows=int(max_q_rows),
                    max_width=self.topk_tokens,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    head_dim=self.q_head_dim,
                    v_head_dim=self.kv_lora_rank,
                    mode=mode,
                    max_batch=int(max_batch),
                    max_chunks_per_row=self._num_splits_cap,
                    page_size=self.block_size,
                    head_major_output=self._head_major_mla_output,
                )
            )

        self._decode_plan = _make_plan(
            "decode",
            self._decode_max_rows,
            self._kernel_num_heads,
            self._decode_max_rows,
        )
        self._extend_plan = _make_plan(
            "extend", max_batched, self._kernel_num_heads, max_num_seqs
        )
        # One caller-owned uint8 scratch tensor covers either path (the larger
        # layout); the per-mode materializer carves its views from the prefix.
        self._scratch_nbytes = max(
            int(self._decode_plan.layout.nbytes),
            int(self._extend_plan.layout.nbytes),
        )

        # CKV gather setup (Fix B).
        from vllm import envs as envs_mod

        ckv_gather_requested = envs_mod.VLLM_B12X_MLA_CKV_GATHER
        self._ckv_gather_enabled = (
            ckv_gather_requested
            and (not self._is_kvarn_mla or self._kvarn_direct_target_model)
            and self.dcp_world_size > 1
            and self.num_heads % _HEAD_ALIGNMENT == 0
        )
        if self._kvarn_native_ckv_gather and not self._ckv_gather_enabled:
            raise RuntimeError(
                "native CKV gather requires VLLM_B12X_MLA_CKV_GATHER=1"
            )
        if ckv_gather_requested and not self._ckv_gather_enabled:
            logger.warning_once(
                "Ignoring VLLM_B12X_MLA_CKV_GATHER on unsupported "
                "topology: dcp=%d local_heads=%d DBO=%s",
                self.dcp_world_size,
                self.num_heads,
                not self.dcp_workspace_non_dbo,
            )
        self._ckv_kernel_num_heads = self.num_heads
        self._ckv_gather_max_tokens = envs_mod.VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
        self._ckv_gather_min_tokens = envs_mod.VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS
        configured_prefetch_depth = int(envs_mod.VLLM_B12X_MLA_CKV_PREFETCH_DEPTH)
        configured_prefetch_workspace_mib = int(
            envs_mod.VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB
        )
        if configured_prefetch_depth < 0:
            logger.warning_once(
                "Ignoring negative VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=%d; using 0",
                configured_prefetch_depth,
            )
        if configured_prefetch_workspace_mib < 0:
            logger.warning_once(
                "Ignoring negative VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB=%d; "
                "using 0 (unlimited)",
                configured_prefetch_workspace_mib,
            )
            configured_prefetch_workspace_mib = 0
        self._ckv_prefetch_supported = (
            self._ckv_gather_enabled
            and (not self._is_kvarn_mla or self._kvarn_direct_target_model)
            and _ckv_prefetch_supports_format(self.kv_cache_dtype)
        )
        self._ckv_local_capacity = (
            _cdiv(
                _cdiv(self._ckv_gather_max_tokens, max(1, self.dcp_world_size))
                + max_num_seqs * self.cp_kv_cache_interleave_size,
                self.block_size,
            )
            * self.block_size
        )
        requested_prefetch_depth = (
            max(0, configured_prefetch_depth) if self._ckv_prefetch_supported else 0
        )
        workspace_budget_bytes = configured_prefetch_workspace_mib * 1024 * 1024
        self._ckv_prefetch_depth = _ckv_prefetch_depth_within_budget(
            requested_prefetch_depth,
            workspace_budget_bytes,
            self.dcp_world_size,
            self._ckv_local_capacity,
            self._kv_record_bytes,
        )
        self._ckv_native_max_pages = _cdiv(
            self._ckv_local_capacity, self.block_size
        )
        self._ckv_native_max_exact_pages = max_num_seqs * 16
        if self._kvarn_native_ckv_gather:
            from sparkinfer.attention.kvarn_mla.api import (
                compact_kvarn_native_rank_nbytes,
            )

            self._ckv_native_max_rank_nbytes = (
                compact_kvarn_native_rank_nbytes(
                    self._ckv_native_max_pages,
                    self._ckv_native_max_exact_pages,
                )
            )
        else:
            self._ckv_native_max_rank_nbytes = 0

        def _lane_workspace_nbytes(depth: int) -> int:
            canonical = _ckv_prefetch_workspace_nbytes(
                depth,
                self.dcp_world_size,
                self._ckv_local_capacity,
                self._kv_record_bytes,
            )
            native = (
                _ckv_native_workspace_nbytes(
                    depth,
                    self.dcp_world_size,
                    self._ckv_native_max_rank_nbytes,
                )
                if self._kvarn_native_ckv_gather
                else 0
            )
            return canonical + native

        if workspace_budget_bytes:
            while (
                self._ckv_prefetch_depth > 0
                and _lane_workspace_nbytes(self._ckv_prefetch_depth)
                > workspace_budget_bytes
            ):
                self._ckv_prefetch_depth -= 1
        self._ckv_workspace_slots = _ckv_prefetch_ring_slots(
            self._ckv_prefetch_depth
        )
        self._ckv_canonical_workspace_nbytes = (
            _ckv_prefetch_workspace_nbytes(
                self._ckv_prefetch_depth,
                self.dcp_world_size,
                self._ckv_local_capacity,
                self._kv_record_bytes,
            )
            if self._ckv_gather_enabled
            else 0
        )
        self._ckv_workspace_nbytes = (
            _lane_workspace_nbytes(self._ckv_prefetch_depth)
            if self._ckv_gather_enabled
            else 0
        )
        execution_lanes = _env_int(
            "VLLM_B12X_MLA_CKV_EXECUTION_LANES",
            _ckv_prefetch_execution_lanes(
                parallel_config.num_ubatches,
                spec is not None,
            ),
        )
        self._ckv_workspace_pool = (
            _get_ckv_prefetch_workspace_pool(
                self.device,
                self._ckv_workspace_nbytes,
                execution_lanes,
            )
            if self._ckv_gather_enabled
            else None
        )
        if self._ckv_workspace_pool is not None:
            logger.info_once(
                "Preallocated %.1f MiB for %d persistent CKV execution lane(s)",
                self._ckv_workspace_pool.storage.numel() / (1024 * 1024),
                self._ckv_workspace_pool.max_slots,
            )
        if self._ckv_prefetch_depth < requested_prefetch_depth:
            logger.info_once(
                "Capping native CKV prefetch depth from %d to %d to fit the "
                "%d MiB per-lane workspace budget (actual %.1f MiB).",
                requested_prefetch_depth,
                self._ckv_prefetch_depth,
                configured_prefetch_workspace_mib,
                self._ckv_workspace_nbytes / (1024 * 1024),
            )

        # Separate extend plan for the gathered-cache path: full local heads
        # (no head all-gather), global seq lens.
        if self._ckv_gather_enabled:
            self._ckv_extend_plan = _make_plan(
                "extend", max_batched, self._ckv_kernel_num_heads, max_num_seqs
            )
            self._scratch_nbytes = max(
                self._scratch_nbytes,
                int(self._ckv_extend_plan.layout.nbytes),
            )
        else:
            self._ckv_extend_plan = None

        # Pre-touch q-concat and attention scratch together. Cross-layer CKV
        # data cannot live in WorkspaceManager: every caller borrows from
        # offset zero, so intervening indexer/MoE scratch would alias it. The
        # builder-owned state allocates a dedicated ring per workspace lane.
        workspace_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            (
                (max_batched, self._kernel_num_heads, self.q_head_dim),
                torch.bfloat16,
            )
        ]
        if self._pad_heads:
            workspace_specs.append(
                (
                    (max_batched, self._input_num_heads, self.kv_lora_rank),
                    torch.bfloat16,
                )
            )
        workspace_specs.append(((self._scratch_nbytes,), torch.uint8))
        self._workspace_specs = tuple(workspace_specs)
        self._borrow_workspaces()
        self._prewarm_extend_kernels_once(max_batched)

        # The builder-owned registry lazily creates one stream per workspace
        # lane after cache discovery finds the first lookahead target. CUDA
        # capture keeps the existing non-CKV fallback.
        if self._ckv_gather_enabled:
            self._ckv_current_chunk_kv_c: torch.Tensor | None = None
            self._ckv_current_chunk_kpe: torch.Tensor | None = None
            if not self._ckv_prefetch_supported:
                logger.warning_once(
                    "CKV gather prefetch disabled for kv_cache_dtype=%s "
                    "(KV_FP8_ROPE=%s); falling back to synchronous gather.",
                    self.kv_cache_dtype,
                    int(self._kv_fp8_rope),
                )
            elif self._ckv_prefetch_depth > 0:
                logger.info_once(
                    "Using native CKV layer prefetch with depth=%d and "
                    "%d workspace slots.",
                    self._ckv_prefetch_depth,
                    self._ckv_workspace_slots,
                )
        else:
            self._ckv_current_chunk_kv_c = None
            self._ckv_current_chunk_kpe = None

        # Q arrives BF16; the unified kernel quantizes inside.
        self.supports_quant_query_input = False

    @classmethod
    def initialize_kvarn_workspaces(
        cls, num_kv_pages: int, device: torch.device
    ) -> None:
        """Allocate graph-static KVarN staging after local pages are known."""
        matching = [
            impl
            for impl in cls._kvarn_instances
            if impl._is_kvarn_mla and impl.device == device
        ]
        for impl in matching:
            config = impl._kvarn_config
            assert config is not None
            envelope = config.workspace_envelope(impl._vllm_config, num_kv_pages)
            runtime_envelope = _kvarn_mla_workspace_envelope(
                num_kv_pages=num_kv_pages,
                group_size=config.group,
                latent_dim=config.latent_dim,
                rope_dim=config.rope_dim,
                max_batched_tokens=impl._max_batched,
                max_active_rows=impl._decode_max_rows,
                topk_tokens=impl.topk_tokens,
                boundary_blocks=impl._kvarn_boundary_blocks,
                rollback_blocks=impl._kvarn_rollback_blocks,
            )
            envelope_fields = (
                "dense_rows",
                "remap_elements",
                "rotation_rows",
                "physical_slot_rows",
                "dense_bytes",
                "total_bytes",
            )
            if any(
                getattr(runtime_envelope, field) != getattr(envelope, field)
                for field in envelope_fields
            ):
                raise ValueError(
                    "KVarN MLA runtime geometry does not match the graph-static "
                    "workspace geometry budgeted from VllmConfig"
                )
            selected_rows = impl._decode_max_rows * impl.topk_tokens
            selected_stage_bytes = selected_rows * 656
            selected_physical_bytes = selected_rows * 4
            direct_stage_bytes = envelope.physical_slot_rows * 656
            workspace_bytes = (
                envelope.total_bytes
                + selected_stage_bytes
                + selected_physical_bytes
            )
            if impl._kvarn_direct_packed_decode:
                workspace_bytes += direct_stage_bytes - envelope.dense_bytes
            key = (
                device,
                num_kv_pages,
                envelope,
                impl.topk_tokens,
                config.group,
                config.latent_dim,
                config.rope_dim,
                impl._kvarn_direct_packed_decode,
            )
            if key not in cls._kvarn_shared_dense:
                if device.type == "cuda":
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    reusable_bytes = max(
                        torch.cuda.memory_reserved(device)
                        - torch.cuda.memory_allocated(device),
                        0,
                    )
                    if workspace_bytes > free_bytes + reusable_bytes:
                        raise MemoryError(
                            "KVarN MLA graph workspace requires "
                            f"{workspace_bytes} bytes but only "
                            f"{free_bytes + reusable_bytes} bytes are available"
                        )
                dense = None
                if not impl._kvarn_direct_packed_decode:
                    dense = torch.empty(
                        envelope.dense_rows // config.group,
                        config.group,
                        config.latent_dim + config.rope_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                direct_stage = None
                if impl._kvarn_direct_packed_decode:
                    direct_stage = torch.empty(
                        envelope.physical_slot_rows,
                        656,
                        dtype=torch.uint8,
                        device=device,
                    )
                selected_stage = torch.empty(
                    impl._decode_max_rows,
                    impl.topk_tokens,
                    656,
                    dtype=torch.uint8,
                    device=device,
                )
                selected_physical_slots = torch.empty(
                    impl._decode_max_rows,
                    impl.topk_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                remapped = torch.empty(
                    impl._max_batched,
                    impl.topk_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                rotated = torch.empty(
                    envelope.rotation_rows,
                    config.latent_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                physical_slots = torch.arange(
                    envelope.physical_slot_rows,
                    dtype=torch.int32,
                    device=device,
                )
                cls._kvarn_shared_dense[key] = dense
                if direct_stage is not None:
                    cls._kvarn_shared_direct_stage[key] = direct_stage
                cls._kvarn_shared_selected_stage[key] = selected_stage
                cls._kvarn_shared_selected_physical_slots[key] = (
                    selected_physical_slots
                )
                cls._kvarn_shared_remapped[key] = remapped
                cls._kvarn_shared_rotated[key] = rotated
                cls._kvarn_shared_physical_slots[key] = physical_slots
            impl._kvarn_dense_cache = cls._kvarn_shared_dense[key]
            impl._kvarn_direct_stage = cls._kvarn_shared_direct_stage.get(key)
            impl._kvarn_selected_stage = cls._kvarn_shared_selected_stage[key]
            impl._kvarn_selected_physical_slots = (
                cls._kvarn_shared_selected_physical_slots[key]
            )
            impl._kvarn_remapped_indices = cls._kvarn_shared_remapped[key]
            impl._kvarn_rotated_scratch = cls._kvarn_shared_rotated[key]
            impl._kvarn_physical_slots = cls._kvarn_shared_physical_slots[key]

    @classmethod
    def reset_kv_cache_binding_state(cls) -> None:
        """Drop prefetch state whose pointers belong to an old KV cache.

        Layer prefetch learns each layer's cache tensor during forward and
        deliberately keeps those tensors across ordinary scheduler steps.
        They are not valid across a KV-cache replacement, however. In
        particular, MRV2 CUDA-graph memory profiling binds a temporary cache,
        destroys it, and later binds the production cache without rebuilding
        the attention implementations.

        The runner calls this hook before every cache binding and while tearing
        down the temporary profiling cache. The first forward on the new cache
        therefore primes the registry synchronously; later forwards recover
        normal layer prefetch.
        """
        cls._all_layer_kv_caches = []
        cls._shared_gather_event = None
        cls._shared_gather_buf_idx = 0
        for registry in tuple(_CKV_PREFETCH_STATE_REGISTRIES):
            registry.clear()
        for impl in tuple(cls._kvarn_instances):
            impl._kvarn_dense_cache = None
            impl._kvarn_direct_stage = None
            impl._kvarn_selected_stage = None
            impl._kvarn_selected_physical_slots = None
            impl._kvarn_remapped_indices = None
            impl._kvarn_rotated_scratch = None
            impl._kvarn_physical_slots = None
            impl._kvarn_block_to_logical = None
        cls._kvarn_shared_dense.clear()
        cls._kvarn_shared_direct_stage.clear()
        cls._kvarn_shared_selected_stage.clear()
        cls._kvarn_shared_selected_physical_slots.clear()
        cls._kvarn_shared_remapped.clear()
        cls._kvarn_shared_rotated.clear()
        cls._kvarn_shared_physical_slots.clear()
        from vllm.v1.attention.backends.mla.kvarn_mla_state import (
            KVarNMLAStateManager,
        )

        KVarNMLAStateManager.rebind_cache_pointers()

    def _ensure_kvarn_mla_cache(self, kv_cache: torch.Tensor) -> None:
        if not self._is_kvarn_mla:
            return
        if self._kvarn_group_key is None:
            raise RuntimeError(
                "MLA KVarN metadata must establish exact-block ownership before use"
            )
        from vllm.v1.attention.backends.mla.kvarn_mla_state import (
            KVarNMLAStateManager,
        )

        self._kvarn_block_to_slot = KVarNMLAStateManager.ensure_mirror(
            self._kvarn_group_key,
            kv_cache.device,
            kv_cache.shape[0],
        )
        if self._kvarn_physical_slots is None:
            raise RuntimeError(
                "KVarN MLA graph workspace was not initialized before cache use"
            )
        planned_pages = (
            self._kvarn_physical_slots.numel() // self._kvarn_config.group
        )
        if kv_cache.shape[0] != planned_pages:
            raise RuntimeError(
                "KVarN MLA cache page count changed after workspace allocation: "
                f"planned {planned_pages}, got {kv_cache.shape[0]}"
            )
        if self._kvarn_block_to_slot.shape[0] < planned_pages:
            raise RuntimeError(
                "KVarN MLA exact-slot map is smaller than the allocated cache"
            )
        KVarNMLAStateManager.validate_records_storage(kv_cache)
        self._kvarn_cache_ref = kv_cache

    def _flush_kvarn_mla_blocks(
        self, block_ids: torch.Tensor, pool_slots: torch.Tensor
    ) -> None:
        if self._kvarn_cache_ref is None or block_ids.numel() == 0:
            return
        assert self._kvarn_config is not None
        _ln_diag = os.environ.get("KVARN_MLA_DIAG_LAYER_NORMS", "")
        if _ln_diag:
            _n = getattr(self, "_kvarn_lnn_count", 0)
            if _n < int(_ln_diag):
                self._kvarn_lnn_count = _n + 1
                _rows = self._kvarn_latent_pool.index_select(0, pool_slots)
                _norms = _rows.float().reshape(_rows.shape[0], -1).norm(dim=1)
                logger.info(
                    "KVARN-LNN layer=%s flush#%d tiles=%d norm_mean=%.3f "
                    "norm_min=%.3f norm_max=%.3f",
                    self.layer_name,
                    _n,
                    _rows.shape[0],
                    _norms.mean().item(),
                    _norms.min().item(),
                    _norms.max().item(),
                )
        from vllm.v1.attention.ops.kvarn_mla import pack_kvarn_mla_blocks

        pack_kvarn_mla_blocks(
            self._kvarn_cache_ref,
            self._kvarn_latent_pool,
            self._kvarn_rope_pool,
            block_ids,
            pool_slots,
            self._kvarn_config,
        )

    def _materialize_kvarn_mla_cache(
        self,
        selected_indices: torch.Tensor,
        _attn_metadata: B12xMLASparseMetadata,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._kvarn_config is not None
        assert self._kvarn_block_to_slot is not None
        if (
            self._kvarn_dense_cache is None
            or self._kvarn_remapped_indices is None
            or self._kvarn_physical_slots is None
        ):
            raise RuntimeError("KVarN MLA graph workspace is not initialized")
        from vllm.v1.attention.ops.kvarn_mla import (
            materialize_physical_kvarn_mla,
            materialize_selected_kvarn_mla,
        )

        num_rows, width = selected_indices.shape
        remap_elements = num_rows * width
        if remap_elements > self._kvarn_remapped_indices.numel():
            raise ValueError(
                "KVarN MLA remap workspace has "
                f"{self._kvarn_remapped_indices.numel()} entries, "
                f"requires {remap_elements}"
            )
        remapped = self._kvarn_remapped_indices.view(-1)[:remap_elements].view(
            num_rows, width
        )
        if num_rows <= self._decode_max_rows:
            materialize_selected_kvarn_mla(
                selected_indices,
                kv_cache,
                self._kvarn_block_to_slot,
                self._kvarn_latent_pool,
                self._kvarn_rope_pool,
                self._kvarn_dense_cache,
                remapped,
                self._kvarn_config,
            )
        else:
            materialize_physical_kvarn_mla(
                self._kvarn_physical_slots,
                selected_indices,
                kv_cache,
                self._kvarn_block_to_slot,
                self._kvarn_latent_pool,
                self._kvarn_rope_pool,
                self._kvarn_dense_cache,
                remapped,
                self._kvarn_config,
            )
        return self._kvarn_dense_cache, remapped

    def _stage_selected_kvarn_mla_fp8_cache(
        self,
        selected_indices: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._kvarn_config is not None
        assert self._kvarn_block_to_slot is not None
        if (
            self._kvarn_selected_stage is None
            or self._kvarn_selected_physical_slots is None
            or self._kvarn_remapped_indices is None
        ):
            raise RuntimeError("KVarN MLA graph workspace is not initialized")

        num_rows, width = selected_indices.shape
        record_count = num_rows * width
        if num_rows > self._decode_max_rows:
            raise ValueError(
                "KVarN MLA selected staging exceeds its decode row envelope: "
                f"{num_rows} > {self._decode_max_rows}"
            )
        if record_count % self.block_size:
            raise ValueError(
                "KVarN MLA selected record count must be divisible by "
                f"block_size={self.block_size}; got {record_count}"
            )
        records_workspace = self._kvarn_selected_stage.view(-1, 656)
        physical_workspace = self._kvarn_selected_physical_slots.view(-1)
        remap_workspace = self._kvarn_remapped_indices.view(-1)
        if (
            record_count > records_workspace.shape[0]
            or record_count > physical_workspace.numel()
            or record_count > remap_workspace.numel()
        ):
            raise ValueError(
                "KVarN MLA selected staging workspace is smaller than the "
                f"required {record_count} records"
            )
        records = records_workspace[:record_count]
        remapped = remap_workspace[:record_count].view(num_rows, width)
        from vllm.v1.attention.ops.kvarn_mla import (
            stage_selected_kvarn_mla_fp8,
        )

        stage_selected_kvarn_mla_fp8(
            selected_indices,
            kv_cache,
            self._kvarn_block_to_slot,
            self._kvarn_latent_pool,
            self._kvarn_rope_pool,
            physical_workspace[:record_count],
            records,
            remapped,
            self._kvarn_config,
        )
        return records.view(-1, self.block_size, 656), remapped

    def _stage_kvarn_mla_fp8_cache(
        self,
        selected_indices: torch.Tensor,
        _attn_metadata: B12xMLASparseMetadata,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._kvarn_config is not None
        assert self._kvarn_block_to_slot is not None
        if (
            self._kvarn_remapped_indices is None
            or self._kvarn_physical_slots is None
        ):
            raise RuntimeError("KVarN MLA graph workspace is not initialized")
        stage_workspace = _select_kvarn_mla_stage_workspace(
            direct_packed=self._kvarn_direct_packed_decode,
            dense_cache=self._kvarn_dense_cache,
            direct_stage=self._kvarn_direct_stage,
        )
        from vllm.v1.attention.ops.kvarn_mla import (
            stage_physical_kvarn_mla_fp8,
        )

        num_rows, width = selected_indices.shape
        remap_elements = num_rows * width
        if remap_elements > self._kvarn_remapped_indices.numel():
            raise ValueError(
                "KVarN MLA remap workspace has "
                f"{self._kvarn_remapped_indices.numel()} entries, "
                f"requires {remap_elements}"
            )
        remapped = self._kvarn_remapped_indices.view(-1)[:remap_elements].view(
            num_rows, width
        )
        page_rows = kv_cache.shape[0] * self._kvarn_config.group
        required_bytes = page_rows * 656
        byte_workspace = stage_workspace.view(torch.uint8).view(-1)
        if byte_workspace.numel() < required_bytes:
            raise ValueError(
                f"KVarN MLA FP8 workspace has fewer than {required_bytes} bytes"
            )
        records = byte_workspace[:required_bytes].view(page_rows, 656)
        stage_physical_kvarn_mla_fp8(
            self._kvarn_physical_slots,
            selected_indices,
            kv_cache,
            self._kvarn_block_to_slot,
            self._kvarn_latent_pool,
            self._kvarn_rope_pool,
            records,
            remapped,
            self._kvarn_config,
        )
        return records.view(-1, self.block_size, 656), remapped

    def _forward_kvarn_mla(
        self,
        q: torch.Tensor,
        dense_cache: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self._kvarn_config is not None
        from vllm.v1.attention.ops.xpu_mla_sparse import (
            triton_bf16_mla_sparse_interface,
        )

        latent_dim = self._kvarn_config.latent_dim
        dense = dense_cache.view(-1, latent_dim + self._kvarn_config.rope_dim)
        out, _, lse = triton_bf16_mla_sparse_interface(
            q,
            dense.unsqueeze(1),
            selected_indices.unsqueeze(1),
            sm_scale=self.scale,
            d_v=latent_dim,
            block_dpe=self._kvarn_config.rope_dim,
            out=q[..., :latent_dim],
        )
        return (
            self._restore_kvarn_mla_output(out),
            lse if self.need_to_return_lse_for_decode else None,
        )

    def _kvarn_layer_alpha(self) -> tuple[float, float]:
        """(alpha, inv_alpha) per layer from KVARN_MLA_ALPHA (store-side lift).

        Format: "layer_idx:alpha,layer_idx:alpha,..." Only listed layers lift;
        unlisted layers return (1.0, 1.0). Parsed once per impl.
        """
        cached = getattr(self, "_kvarn_alpha_pair", None)
        if cached is not None:
            return cached
        pair = (1.0, 1.0)
        spec = os.environ.get("KVARN_MLA_ALPHA", "")
        if spec and self._is_kvarn_mla:
            m = re.search(r"layers\.(\d+)", self.layer_name or "")
            if m:
                li = int(m.group(1))
                for part in spec.split(","):
                    idx, _, a = part.partition(":")
                    if idx.strip().isdigit() and int(idx) == li:
                        try:
                            av = float(a)
                            pair = (av, 1.0 / av)
                        except ValueError:
                            pass
                        break
        self._kvarn_alpha_pair = pair
        return pair

    def _restore_kvarn_mla_output(self, output: torch.Tensor) -> torch.Tensor:
        if not self._is_kvarn_mla:
            return output
        out = kvarn_hadamard(output, self._kvarn_h)
        _, inv = self._kvarn_layer_alpha()
        if inv != 1.0:
            out = out * inv
        if os.environ.get("KVARN_MLA_LAYER_WATCH", "0") == "1":
            import pathlib

            with open("/tmp/layerwatch.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "layer": self.layer_name,
                            "sum": round(
                                float(out.detach().float().abs().sum().item()), 1
                            ),
                        }
                    )
                    + "\n"
                )
            _mp = pathlib.Path("/tmp/KVARN_LAYER_FLUSH")
            if _mp.exists():
                _mp.unlink()
                _mp.parent.mkdir(parents=True, exist_ok=True)
                _mp.with_suffix(".flushed").touch()
        return out

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Write the post-RoPE key using the selected runtime cache format.

        The disabled branch delegates to the shipped implementation unchanged,
        including its stock 432-byte NVFP4 writer. The enabled branch calls the
        b12x package writer API, which does not replace or perturb the stock
        writer used by KV_FP8_ROPE=0.
        """
        if self._is_kvarn_mla:
            if kv_cache.numel() == 0 or slot_mapping is None:
                return
            if self._kvarn_group_key is None:
                return
            assert self._kvarn_config is not None
            self._ensure_kvarn_mla_cache(kv_cache)
            assert self._kvarn_block_to_slot is not None
            num_tokens = slot_mapping.numel()
            if self._kvarn_rotated_scratch is None:
                raise RuntimeError("KVarN MLA graph workspace is not initialized")
            if num_tokens > self._kvarn_rotated_scratch.shape[0]:
                raise ValueError(
                    "KVarN MLA rotation workspace has "
                    f"{self._kvarn_rotated_scratch.shape[0]} rows, "
                    f"requires {num_tokens}"
                )
            if kv_c_normed.numel() < num_tokens * self._kvarn_config.latent_dim:
                raise ValueError("KVarN MLA latent input has too few elements")
            if k_pe.numel() < num_tokens * self._kvarn_config.rope_dim:
                raise ValueError("KVarN MLA RoPE input has too few elements")
            latent = kv_c_normed[:num_tokens].reshape(
                num_tokens, self._kvarn_config.latent_dim
            )
            rotated = self._kvarn_rotated_scratch[:num_tokens]
            kvarn_hadamard(latent, self._kvarn_h, out=rotated)
            alpha, _ = self._kvarn_layer_alpha()
            if alpha != 1.0:
                rotated.mul_(alpha)
            from vllm.v1.attention.ops.kvarn_mla import scatter_kvarn_mla_exact

            scatter_kvarn_mla_exact(
                rotated,
                k_pe[:num_tokens].reshape(num_tokens, self._kvarn_config.rope_dim),
                slot_mapping.reshape(-1),
                self._kvarn_block_to_slot,
                self._kvarn_latent_pool,
                self._kvarn_rope_pool,
            )
            return
        if not self._kv_fp8_rope:
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype != "nvfp4_ds_mla":
            raise RuntimeError(
                f"KV_FP8_ROPE writer reached a non-NVFP4 cache: {kv_cache_dtype!r}"
            )
        k_pe_flat = k_pe.squeeze(1)
        self._concat_and_cache_nvfp4_mla_fp8_rope(
            kv_c_normed,
            k_pe_flat,
            kv_cache,
            slot_mapping.flatten(),
            k_scale,
        )

    def _borrow_workspaces(self) -> list[torch.Tensor]:
        workspaces = current_workspace_manager().get_simultaneous(
            *self._workspace_specs
        )
        if self._pad_heads:
            dense_storage = workspaces[1]
            workspaces[1] = dense_storage.view(
                self._input_num_heads,
                self._max_batched,
                self.kv_lora_rank,
            ).transpose(0, 1)
        return workspaces

    def _borrow_workspace_parts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        workspace_tensors = self._borrow_workspaces()
        expected_count = 3 if self._pad_heads else 2
        if len(workspace_tensors) != expected_count:
            raise RuntimeError(
                "B12X DCP prefill borrowed an unexpected workspace count: "
                f"{len(workspace_tensors)} != {expected_count}"
            )
        q_workspace = workspace_tensors[0]
        dense_out_workspace = workspace_tensors[1] if self._pad_heads else None
        scratch_storage = workspace_tensors[-1]
        expected_q_shape = (
            self._max_batched,
            self._kernel_num_heads,
            self.q_head_dim,
        )
        if (
            tuple(q_workspace.shape) != expected_q_shape
            or q_workspace.dtype != torch.bfloat16
            or q_workspace.device != self.device
            or not q_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "B12X DCP prefill borrowed an invalid query workspace: "
                f"shape={tuple(q_workspace.shape)}, dtype={q_workspace.dtype}, "
                f"device={q_workspace.device}"
            )
        if dense_out_workspace is not None and (
            tuple(dense_out_workspace.shape)
            != (self._max_batched, self._input_num_heads, self.kv_lora_rank)
            or dense_out_workspace.dtype != torch.bfloat16
            or dense_out_workspace.device != self.device
            or not dense_out_workspace.movedim(0, 1).is_contiguous()
        ):
            raise RuntimeError("B12X DCP prefill borrowed an invalid dense output")
        if (
            tuple(scratch_storage.shape) != (self._scratch_nbytes,)
            or scratch_storage.dtype != torch.uint8
            or scratch_storage.device != self.device
            or not scratch_storage.is_contiguous()
        ):
            raise RuntimeError("B12X DCP prefill borrowed an invalid raw scratch")
        return q_workspace, dense_out_workspace, scratch_storage

    def supports_fused_mla_query_output(
        self,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> bool:
        """Whether fused query assembly can target the planned B12X layout."""
        return bool(
            self.dcp_world_size == 1
            and output_dtype == torch.bfloat16
            and num_heads == self._input_num_heads
            and self._kernel_num_heads == self._input_num_heads
            and self.q_head_dim == 576
        )

    def get_fused_mla_query_output(
        self,
        num_tokens: int,
        num_heads: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return the final B12X query view for fused DCP1 assembly.

        DCP query gathering needs a different local-head layout, so those
        configurations retain the ordinary temporary query path. For DCP1,
        writing the fused epilogue directly here lets ``forward_mqa`` consume
        the query without a concat or workspace copy.
        """
        if (
            not self.supports_fused_mla_query_output(num_heads, output_dtype)
            or num_tokens <= 0
            or num_tokens > self._max_batched
        ):
            return None
        q_workspace, _, _ = self._borrow_workspace_parts()
        output = q_workspace[:num_tokens, :num_heads]
        if not output.is_contiguous():
            raise RuntimeError("B12X fused MLA query output must be contiguous")
        return output

    def _validate_dcp_prefill_workspace_contract(self, num_tokens: int) -> None:
        supported_topologies = {
            (4, 2),
            (4, 4),
            (6, 2),
            (6, 3),
            (6, 6),
            (8, 2),
            (8, 4),
            (8, 8),
        }
        if (
            not 1025 <= num_tokens <= self._max_batched
            or not self.dcp_workspace_non_dbo
            or (self.tp_world_size, self.dcp_world_size) not in supported_topologies
            or self.dcp_world_size <= 1
            or self.num_heads <= 0
            or self._input_num_heads != self.num_heads * self.dcp_world_size
            or self._kernel_num_heads < self._input_num_heads
            or self._kernel_num_heads % _HEAD_ALIGNMENT != 0
            or self.q_head_dim != 576
            or self.kv_lora_rank != 512
            or self.v_head_dim != 256
        ):
            raise RuntimeError(
                "The DCP prefill workspace path received an unsupported "
                "topology or geometry: "
                f"tokens={num_tokens}/{self._max_batched}, "
                f"TP/DCP={self.tp_world_size}/{self.dcp_world_size}, "
                f"local/input/kernel heads={self.num_heads}/"
                f"{self._input_num_heads}/{self._kernel_num_heads}, "
                f"pad_heads={self._pad_heads}, dimensions="
                f"{self.q_head_dim}/{self.kv_lora_rank}/{self.v_head_dim}"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The DCP prefill workspace path is eager-only")

    def dcp_all_gather_query_in_workspace(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Gather local DCP query heads through the borrowed MLA workspaces."""
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            if ql_nope.ndim != 3 or q_pe.ndim != 3:
                raise ValueError("DCP workspace tuple queries must be rank-3")
            num_tokens, local_heads, nope_dim = ql_nope.shape
            if tuple(q_pe.shape) != (num_tokens, local_heads, 64) or nope_dim != 512:
                raise ValueError("DCP workspace requires noPE/RoPE dimensions 512/64")
            if ql_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
                raise TypeError("DCP workspace queries must be BF16")
            tuple_input = True
            query_device = ql_nope.device
        else:
            if q.ndim != 3:
                raise ValueError("DCP workspace query must be rank-3")
            num_tokens, local_heads, head_dim = q.shape
            if head_dim != self.q_head_dim or not q.is_contiguous():
                raise ValueError("DCP workspace tensor query has invalid layout")
            tuple_input = False
            query_device = q.device

        self._validate_dcp_prefill_workspace_contract(int(num_tokens))
        if local_heads != self.num_heads or query_device != self.device:
            raise ValueError("DCP workspace query does not match the MLA plan")

        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        process_group = dcp_group.device_group
        if (
            dcp_group.world_size != self.dcp_world_size
            or dcp_group.rank_in_group != self.dcp_rank
        ):
            raise RuntimeError("DCP workspace group does not match the MLA plan")

        q_workspace, _, scratch_storage = self._borrow_workspace_parts()
        q_begin = q_workspace.data_ptr()
        q_end = q_begin + q_workspace.numel() * q_workspace.element_size()
        scratch_begin = scratch_storage.data_ptr()
        scratch_end = scratch_begin + scratch_storage.numel()
        if q_begin < scratch_end and scratch_begin < q_end:
            raise RuntimeError("DCP query and scratch workspaces overlap")

        world_size = self.dcp_world_size
        head_dim = self.q_head_dim
        bytes_per_chunk_row = world_size * local_heads * head_dim * _BF16_BYTES
        chunk_capacity = scratch_storage.numel() // bytes_per_chunk_row
        if chunk_capacity <= 0:
            raise RuntimeError("DCP scratch cannot hold one gathered query row")

        q_workspace_flat = q_workspace.view(-1)
        chunk_start = 0
        while chunk_start < num_tokens:
            chunk_rows = min(chunk_capacity, num_tokens - chunk_start)
            if tuple_input:
                local_offset = chunk_start * self._kernel_num_heads * head_dim
                local_numel = chunk_rows * local_heads * head_dim
                local_chunk = q_workspace_flat.narrow(
                    0, local_offset, local_numel
                ).view(chunk_rows, local_heads, head_dim)
                ops.concat_mla_q(
                    ql_nope.narrow(0, chunk_start, chunk_rows),
                    q_pe.narrow(0, chunk_start, chunk_rows),
                    local_chunk,
                )
            else:
                local_chunk = cast(torch.Tensor, q).narrow(0, chunk_start, chunk_rows)

            gather_numel = world_size * chunk_rows * local_heads * head_dim
            gathered = (
                scratch_storage.narrow(0, 0, gather_numel * _BF16_BYTES)
                .view(torch.bfloat16)
                .view(world_size * chunk_rows, local_heads, head_dim)
            )
            dist.all_gather_into_tensor(
                gathered,
                local_chunk,
                group=process_group,
                async_op=False,
            )
            rank_major = gathered.view(world_size, chunk_rows, local_heads, head_dim)
            destination = q_workspace.narrow(0, chunk_start, chunk_rows)
            for source_rank in range(world_size):
                destination.narrow(1, source_rank * local_heads, local_heads).copy_(
                    rank_major[source_rank]
                )
            chunk_start += chunk_rows

        global_query = q_workspace[:num_tokens, : self._input_num_heads]
        expected_stride = (self._kernel_num_heads * head_dim, head_dim, 1)
        if (
            tuple(global_query.shape) != (num_tokens, self._input_num_heads, head_dim)
            or tuple(global_query.stride()) != expected_stride
        ):
            raise RuntimeError("DCP workspace produced an invalid query view")
        return global_query

    def dcp_decode_all_gather_query_in_workspace(
        self,
        q: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | None:
        """Gather decode DCP query heads directly into the borrowed q lane.

        Default-off companion of the prefill workspace gather
        (``VLLM_B12X_A2A_Q_DIRECT_WRITE``): the local (noPE, RoPE) tuple is
        concatenated by ``concat_mla_q`` into a scratch staging slice and the
        B12X PCIe channel writes the gathered heads straight into the decode
        q workspace lane that ``forward_mqa`` borrows afterwards. The returned
        view is an exact workspace alias, so ``forward_mqa`` skips its
        destination copy (``exact_workspace_alias``). Returns ``None`` whenever
        any precondition fails; the caller then keeps the incumbent
        cat -> gather -> workspace-copy chain (fail closed).
        """
        if not isinstance(q, tuple) or len(q) != 2:
            return None
        ql_nope, q_pe = q
        if (
            ql_nope.ndim != 3
            or q_pe.ndim != 3
            or ql_nope.dtype != torch.bfloat16
            or q_pe.dtype != torch.bfloat16
            or not ql_nope.is_cuda
            or ql_nope.device != self.device
            or q_pe.device != self.device
        ):
            return None
        num_tokens, local_heads, nope_dim = ql_nope.shape
        if (
            nope_dim != self.kv_lora_rank
            or tuple(q_pe.shape) != (num_tokens, local_heads, 64)
            or self.q_head_dim != self.kv_lora_rank + 64
            or local_heads != self.num_heads
            or num_tokens < 1
            or num_tokens > self._max_batched
            or self.dcp_world_size <= 1
            or self._input_num_heads != self.num_heads * self.dcp_world_size
            # The pool writes a contiguous [B, total_heads, 576] slab; only a
            # full-width prefix row slice of the lane qualifies.
            or self._kernel_num_heads != self._input_num_heads
        ):
            return None
        # CUDA-graph safety: pool acquisition performs IPC handle exchange
        # only on first use; under capture the accessor serves an
        # already-warmed channel from its cache or returns None (fail closed
        # to the incumbent chain), and all buffer shapes here are static.

        from vllm.distributed.parallel_state import get_dcp_group
        from vllm.v1.attention.ops.dcp_alltoall import (
            dcp_b12x_all_gather_heads_into,
        )

        dcp_group = get_dcp_group()
        if (
            dcp_group.world_size != self.dcp_world_size
            or dcp_group.rank_in_group != self.dcp_rank
        ):
            return None

        q_workspace, _, scratch_storage = self._borrow_workspace_parts()
        q_begin = q_workspace.data_ptr()
        q_end = q_begin + q_workspace.numel() * q_workspace.element_size()
        scratch_begin = scratch_storage.data_ptr()
        scratch_end = scratch_begin + scratch_storage.numel()
        if q_begin < scratch_end and scratch_begin < q_end:
            return None
        out_view = q_workspace[:num_tokens]
        if (
            tuple(out_view.shape) != (num_tokens, self._kernel_num_heads, self.q_head_dim)
            or not out_view.is_contiguous()
        ):
            return None
        staging_numel = num_tokens * local_heads * self.q_head_dim
        if scratch_storage.numel() < staging_numel * _BF16_BYTES:
            return None
        staging = (
            scratch_storage.narrow(0, 0, staging_numel * _BF16_BYTES)
            .view(torch.bfloat16)
            .view(num_tokens, local_heads, self.q_head_dim)
        )
        # Fold the torch.cat: same concat op forward_mqa's tuple branch uses,
        # writing raw (pre-Hadamard) values exactly as the incumbent chain
        # delivered them.
        ops.concat_mla_q(ql_nope, q_pe, staging)
        gathered = dcp_b12x_all_gather_heads_into(
            staging,
            dcp_group,
            out_view,
            max_batch_size=self._max_batched,
            pool_head_dim=self.kv_lora_rank,
        )
        if gathered is None:
            return None
        return out_view

    def dcp_project_before_merge_in_workspace(
        self,
        attn_out: torch.Tensor,
        lse: torch.Tensor,
        w_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Project DCP partials from 512 to 256 in borrowed MLA storage."""
        num_tokens = int(attn_out.shape[0])
        self._validate_dcp_prefill_workspace_contract(num_tokens)
        expected_head_major_stride = (
            self.kv_lora_rank,
            self._max_batched * self.kv_lora_rank,
            1,
        )
        expected_token_major_stride = (
            self._input_num_heads * self.kv_lora_rank,
            self.kv_lora_rank,
            1,
        )
        attn_stride = tuple(attn_out.stride())
        head_major_input = attn_stride == expected_head_major_stride
        token_major_input = attn_stride == expected_token_major_stride
        if (
            tuple(attn_out.shape)
            != (num_tokens, self._input_num_heads, self.kv_lora_rank)
            or not (head_major_input or token_major_input)
            or attn_out.dtype != torch.bfloat16
            or tuple(w_uv.shape)
            != (self._input_num_heads, self.kv_lora_rank, self.v_head_dim)
            or not w_uv.is_contiguous()
            or w_uv.dtype != torch.bfloat16
            or tuple(lse.shape) != (num_tokens, self._input_num_heads)
            or lse.dtype != torch.float32
        ):
            raise ValueError(
                "DCP workspace projection received an invalid tensor layout: "
                f"attn shape/stride={tuple(attn_out.shape)}/"
                f"{attn_stride}, expected stride="
                f"{expected_head_major_stride} or {expected_token_major_stride}"
            )

        q_workspace, dense_out_workspace, scratch_storage = (
            self._borrow_workspace_parts()
        )
        input_numel = self._input_num_heads * num_tokens * self.kv_lora_rank
        projected_numel = self._input_num_heads * num_tokens * self.v_head_dim
        projected_nbytes = projected_numel * _BF16_BYTES
        if scratch_storage.numel() < projected_nbytes or (
            head_major_input and q_workspace.numel() < input_numel
        ):
            raise RuntimeError("DCP projection workspace is too small")
        if token_major_input:
            if attn_out.untyped_storage().data_ptr() == (
                scratch_storage.untyped_storage().data_ptr()
            ):
                raise RuntimeError("DCP projection input aliases its output workspace")
            projection_input = attn_out.transpose(0, 1)
        else:
            expected_attn_storage = (
                dense_out_workspace if self._pad_heads else scratch_storage
            )
            assert expected_attn_storage is not None
            if attn_out.untyped_storage().data_ptr() != (
                expected_attn_storage.untyped_storage().data_ptr()
            ):
                raise RuntimeError(
                    "DCP attention output is not backed by the expected MLA workspace"
                )
            projection_input = q_workspace.view(-1)[:input_numel].view(
                self._input_num_heads, num_tokens, self.kv_lora_rank
            )
            projection_input.copy_(attn_out.transpose(0, 1))
        projected_head_major = (
            scratch_storage[:projected_nbytes]
            .view(torch.bfloat16)
            .view(self._input_num_heads, num_tokens, self.v_head_dim)
        )

        _run_dcp_project_bmm(projection_input, w_uv, projected_head_major)
        return projected_head_major.transpose(0, 1)

    def dcp_reduce_scatter_output_in_workspace(
        self,
        corrected_attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the dead query prefix as DCP reduce-scatter output."""
        num_tokens = int(corrected_attn_out.shape[0])
        self._validate_dcp_prefill_workspace_contract(num_tokens)
        input_head_major = corrected_attn_out.movedim(0, 1)
        if (
            tuple(corrected_attn_out.shape)
            != (num_tokens, self._input_num_heads, self.v_head_dim)
            or corrected_attn_out.dtype != torch.bfloat16
            or not input_head_major.is_contiguous()
        ):
            raise ValueError("DCP reduce-scatter input has an invalid layout")

        q_workspace, _, scratch_storage = self._borrow_workspace_parts()
        if (
            corrected_attn_out.untyped_storage().data_ptr()
            != scratch_storage.untyped_storage().data_ptr()
        ):
            raise RuntimeError(
                "DCP corrected input is not backed by the MLA scratch workspace"
            )
        output_numel = self.num_heads * num_tokens * self.v_head_dim
        output_head_major = q_workspace.view(-1)[:output_numel].view(
            self.num_heads, num_tokens, self.v_head_dim
        )
        output = output_head_major.transpose(0, 1)
        if not output_head_major.is_contiguous():
            raise RuntimeError("DCP reduce-scatter output is not contiguous")
        return output

    def _validate_ckv_workspace(self, ckv_workspace: torch.Tensor) -> None:
        if not self._ckv_gather_enabled:
            raise RuntimeError("CKV gather workspace requested while disabled")
        if (
            tuple(ckv_workspace.shape) != (self._ckv_workspace_nbytes,)
            or ckv_workspace.dtype != torch.uint8
            or ckv_workspace.device != self.device
            or not ckv_workspace.is_contiguous()
        ):
            raise RuntimeError("B12X CKV gather borrowed an invalid workspace")

    def _ckv_workspace_views(
        self, ckv_workspace: torch.Tensor, buf_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_ckv_workspace(ckv_workspace)
        if not 0 <= int(buf_idx) < self._ckv_workspace_slots:
            raise ValueError(
                f"CKV gather buffer index {buf_idx} is outside "
                f"[0, {self._ckv_workspace_slots})"
            )
        records = ckv_workspace[
            : self._ckv_canonical_workspace_nbytes
        ].view(-1, self._kv_record_bytes)
        local_buffer = records[: self._ckv_local_capacity]
        gathered_base = (
            self._ckv_local_capacity
            + buf_idx * self.dcp_world_size * self._ckv_local_capacity
        )
        gathered_buffer = records[
            gathered_base : gathered_base
            + self.dcp_world_size * self._ckv_local_capacity
        ]
        return local_buffer, gathered_buffer

    def _ckv_native_workspace_views(
        self,
        ckv_workspace: torch.Tensor,
        buf_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_ckv_workspace(ckv_workspace)
        if not self._kvarn_native_ckv_gather:
            raise RuntimeError("native CKV workspace requested while disabled")
        if not 0 <= int(buf_idx) < self._ckv_workspace_slots:
            raise ValueError("native CKV gather buffer index is out of range")
        local = ckv_workspace[
            : self._ckv_local_capacity * self._kv_record_bytes
        ]
        native = ckv_workspace[self._ckv_canonical_workspace_nbytes :]
        gathered_base = (
            buf_idx
            * self.dcp_world_size
            * self._ckv_native_max_rank_nbytes
        )
        gathered = native[
            gathered_base : gathered_base
            + self.dcp_world_size * self._ckv_native_max_rank_nbytes
        ]
        return local, gathered

    def _native_ckv_gather_eligible(
        self,
        attn_metadata: B12xMLASparseMetadata,
    ) -> bool:
        if not self._kvarn_native_ckv_gather:
            return False
        assert self._kvarn_config is not None
        if attn_metadata.min_seq_len <= (
            self._kvarn_config.boundary_tokens
            + self._kvarn_precision_tail_tokens
        ):
            return False
        if (
            attn_metadata.dcp_rank_req_page_starts is None
            or attn_metadata.dcp_rank_req_page_lens is None
            or attn_metadata.dcp_padded_total_pages <= 0
            or attn_metadata.dcp_padded_exact_pages <= 0
            or attn_metadata.dcp_padded_total_pages
            > self._ckv_native_max_pages
            or attn_metadata.dcp_padded_exact_pages
            > self._ckv_native_max_exact_pages
        ):
            return False
        from sparkinfer.attention.kvarn_mla.api import (
            compact_kvarn_native_rank_nbytes,
        )

        native_bytes = compact_kvarn_native_rank_nbytes(
            attn_metadata.dcp_padded_total_pages,
            attn_metadata.dcp_padded_exact_pages,
        )
        canonical_bytes = (
            attn_metadata.dcp_padded_total_tokens * self._kv_record_bytes
        )
        return native_bytes < canonical_bytes

    def dcp_prefill_ckv_gather_eligible(
        self,
        attn_metadata: B12xMLASparseMetadata,
        num_tokens: int,
    ) -> bool:
        if not self._ckv_gather_enabled:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if (
            not attn_metadata.dcp_ckv_gather_eligible
            or attn_metadata.num_decode_tokens != 0
            or attn_metadata.num_prefill_tokens != attn_metadata.num_actual_tokens
            or int(num_tokens) != attn_metadata.num_actual_tokens
            or int(num_tokens) <= self._ckv_gather_min_tokens
            or attn_metadata.dcp_padded_total_tokens > self._ckv_local_capacity
            or attn_metadata.dcp_local_total_tokens
            > attn_metadata.dcp_padded_total_tokens
        ):
            return False
        return all(
            tensor is not None
            for tensor in (
                attn_metadata.req_id_per_token,
                attn_metadata.ckv_page_table_1,
                attn_metadata.ckv_nsa_cache_seqlens,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                attn_metadata.dcp_local_cu_seq_lens,
            )
        )

    def _dcp_gather_ckv(
        self,
        kv_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        ckv_workspace: torch.Tensor,
        buf_idx: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if not self.dcp_prefill_ckv_gather_eligible(
            attn_metadata, attn_metadata.num_actual_tokens
        ):
            raise RuntimeError("CKV gather called for an ineligible attention batch")
        if (
            kv_cache.dtype != torch.uint8
            or kv_cache.ndim != 3
            or not kv_cache.is_contiguous()
            or (
                not self._is_kvarn_mla
                and tuple(kv_cache.shape[1:])
                != (self.block_size, self._kv_record_bytes)
            )
        ):
            raise ValueError("CKV gather requires contiguous paged KV cache pages")

        assert attn_metadata.dcp_local_cu_seq_lens is not None
        padded_tokens = attn_metadata.dcp_padded_total_tokens
        local_tokens = attn_metadata.dcp_local_total_tokens
        local_buffer, gathered_buffer = self._ckv_workspace_views(
            ckv_workspace, buf_idx
        )
        if stream is not None:
            ckv_workspace.record_stream(stream)
            # The side stream must observe the default stream's prior writes
            # to the paged KV cache (this and earlier steps' do_kv_cache_update)
            # before gathering history off it; there is otherwise no ordering
            # between the two streams in this direction. Enqueued before L's
            # attention/MLP, so the prefetch still overlaps that compute.
            stream.wait_stream(torch.cuda.current_stream())
            stream_ctx = torch.cuda.stream(stream)
        else:
            stream_ctx = torch.cuda.stream(torch.cuda.current_stream())
        with stream_ctx:
            if self._native_ckv_gather_eligible(attn_metadata):
                assert self._kvarn_block_to_slot is not None
                assert self._kvarn_latent_pool is not None
                assert self._kvarn_rope_pool is not None
                assert attn_metadata.dcp_rank_req_page_starts is not None
                assert attn_metadata.dcp_rank_req_page_lens is not None
                from sparkinfer.attention.kvarn_mla.api import (
                    compact_kvarn_native_rank_nbytes,
                    stage_compact_kvarn_native_history,
                )
                from sparkinfer.attention._shared.mla.kernel import (
                    native_kvarn_ckv_materialize_records,
                )
                from vllm.distributed.parallel_state import (
                    get_dcp_ckv_prefetch_group,
                    get_dcp_group,
                )

                padded_pages = attn_metadata.dcp_padded_total_pages
                padded_exact_pages = attn_metadata.dcp_padded_exact_pages
                rank_wire_bytes = compact_kvarn_native_rank_nbytes(
                    padded_pages, padded_exact_pages
                )
                native_local, native_gathered = (
                    self._ckv_native_workspace_views(ckv_workspace, buf_idx)
                )
                stage_compact_kvarn_native_history(
                    attn_metadata.block_table,
                    attn_metadata.dcp_rank_req_page_starts[self.dcp_rank],
                    attn_metadata.dcp_rank_req_page_lens[self.dcp_rank],
                    kv_cache,
                    self._kvarn_block_to_slot,
                    self._kvarn_latent_pool,
                    self._kvarn_rope_pool,
                    native_local[:rank_wire_bytes],
                    padded_pages=padded_pages,
                    padded_exact_pages=padded_exact_pages,
                )
                dcp_group = (
                    get_dcp_ckv_prefetch_group()
                    if stream is not None
                    else get_dcp_group()
                )
                _dcp_all_gather_current_stream(
                    dcp_group,
                    native_local[:rank_wire_bytes],
                    native_gathered[
                        : self.dcp_world_size * rank_wire_bytes
                    ],
                )
                native_kvarn_ckv_materialize_records(
                    native_gathered[
                        : self.dcp_world_size * rank_wire_bytes
                    ],
                    attn_metadata.dcp_rank_req_page_starts,
                    attn_metadata.dcp_rank_req_page_lens,
                    attn_metadata.dcp_rank_req_starts,
                    attn_metadata.dcp_rank_req_lens,
                    gathered_buffer[
                        : self.dcp_world_size * padded_tokens
                    ].view(-1, self._kv_record_bytes),
                    padded_tokens=padded_tokens,
                    padded_pages=padded_pages,
                    padded_exact_pages=padded_exact_pages,
                )
                gathered_tokens = gathered_buffer[
                    : self.dcp_world_size * self._ckv_local_capacity
                ]
                return gathered_tokens.view(
                    -1, self.block_size, self._kv_record_bytes
                )
            if local_tokens:
                dense_record_cache = (
                    kv_cache.stride(0)
                    == self.block_size * self._kv_record_bytes
                )
                if self._is_kvarn_mla and not dense_record_cache:
                    assert self._kvarn_block_to_slot is not None
                    assert self._kvarn_remapped_indices is not None
                    assert self._kvarn_config is not None
                    from sparkinfer.attention.kvarn_mla import (
                        stage_k5_as_fp8_records,
                    )
                    from vllm.v1.attention.ops.kvarn_mla import (
                        build_compact_kvarn_mla_physical_slots,
                    )

                    compact_slots = self._kvarn_remapped_indices.view(-1)[
                        :local_tokens
                    ]
                    build_compact_kvarn_mla_physical_slots(
                        attn_metadata.block_table,
                        attn_metadata.dcp_local_cu_seq_lens,
                        compact_slots,
                        batch_size=attn_metadata.num_reqs,
                        total_tokens=local_tokens,
                        page_size=self.block_size,
                    )
                    stage_k5_as_fp8_records(
                        compact_slots,
                        kv_cache,
                        self._kvarn_block_to_slot,
                        self._kvarn_latent_pool,
                        self._kvarn_rope_pool,
                        local_buffer[:local_tokens],
                    )
                else:
                    ops.cp_gather_cache(
                        src_cache=kv_cache,
                        dst=local_buffer[:local_tokens],
                        block_table=attn_metadata.block_table,
                        cu_seq_lens=attn_metadata.dcp_local_cu_seq_lens,
                        batch_size=attn_metadata.num_reqs,
                    )
            if local_tokens < padded_tokens:
                local_buffer[local_tokens:padded_tokens].zero_()

            from vllm.distributed.parallel_state import (
                get_dcp_ckv_prefetch_group,
                get_dcp_group,
            )

            # The prefetch (side stream) uses a dedicated communicator so it
            # cannot collide with the indexer's DCP merge on the default
            # stream; the synchronous path shares the default stream with the
            # merge and is safe on the main DCP communicator.
            dcp_group = (
                get_dcp_ckv_prefetch_group() if stream is not None else get_dcp_group()
            )
            _dcp_all_gather_current_stream(
                dcp_group,
                local_buffer[:padded_tokens].view(-1),
                gathered_buffer[: self.dcp_world_size * padded_tokens].view(-1),
            )
        # Keep the cache geometry stable across requests. CuTe/B12X caches the
        # compiled prefill launch, while ``padded_tokens`` grows with context;
        # exposing a differently sized first dimension on every request can
        # reuse a launch specialized for an earlier, smaller cache. The live
        # records remain packed in the prefix and selected indices are still
        # based on ``padded_tokens``, so the unused capacity is unreachable.
        gathered_tokens = gathered_buffer[
            : self.dcp_world_size * self._ckv_local_capacity
        ]
        return gathered_tokens.view(-1, self.block_size, self._kv_record_bytes)

    def set_ckv_current_chunk_kv(
        self, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> None:
        self._ckv_current_chunk_kv_c = kv_c_normed
        self._ckv_current_chunk_kpe = k_pe

    def _resolve_layer_index(self, layer) -> int | None:
        """Global layer index used to key the cross-layer prefetch registry.

        ``MLAAttention`` exposes ``layer_name`` (a dotted prefix), not
        ``layer_idx``; without this fallback the prefetch pipeline never
        engages and every layer gathers synchronously on the critical path.
        """
        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is not None:
            try:
                return int(layer_idx)
            except (TypeError, ValueError):
                return None
        layer_name = getattr(layer, "layer_name", None)
        if not layer_name:
            return None
        from vllm.model_executor.models.utils import extract_layer_index

        try:
            return extract_layer_index(layer_name)
        except (ValueError, AssertionError, IndexError):
            return None

    def _stage_current_kvarn_mla_fp8_records(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        gathered_buffer: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Stage full live rows with canonical exact-pool-equivalent rounding."""
        assert self._kvarn_config is not None
        if _KVARN_FUSED_CURRENT_STAGE_ENABLED:
            num_tokens = slots.numel()
            latent = kv_c[:num_tokens].reshape(
                num_tokens, self._kvarn_config.latent_dim
            )
            rope = k_pe[:num_tokens].reshape(
                num_tokens, self._kvarn_config.rope_dim
            )
            from sparkinfer.attention.kvarn_mla.api import (
                stage_bf16_sylvester_as_exact_pool_fp8_records,
            )

            stage_bf16_sylvester_as_exact_pool_fp8_records(
                latent,
                rope,
                slots,
                gathered_buffer.view(-1, self._kv_record_bytes),
            )
            return
        if self._kvarn_rotated_scratch is None:
            raise RuntimeError("KVarN MLA graph workspace is not initialized")
        num_tokens = slots.numel()
        if num_tokens > self._kvarn_rotated_scratch.shape[0]:
            raise ValueError(
                "KVarN MLA rotation workspace has "
                f"{self._kvarn_rotated_scratch.shape[0]} rows, "
                f"requires {num_tokens}"
            )
        latent = kv_c[:num_tokens].reshape(
            num_tokens, self._kvarn_config.latent_dim
        )
        rotated = self._kvarn_rotated_scratch[:num_tokens]
        kvarn_hadamard(latent, self._kvarn_h, out=rotated)
        from sparkinfer.attention.kvarn_mla.api import (
            stage_bf16_as_exact_pool_fp8_records,
        )

        stage_bf16_as_exact_pool_fp8_records(
            rotated,
            k_pe[:num_tokens].reshape(num_tokens, self._kvarn_config.rope_dim),
            slots,
            gathered_buffer.view(-1, self._kv_record_bytes),
        )


    def _append_current_chunk_to_gathered(
        self,
        gathered_buffer: torch.Tensor,
        attn_metadata: "B12xMLASparseMetadata",
        layer,
        num_actual_toks: int,
    ) -> None:
        """Write the current chunk's BF16 KV into the gathered buffer for
        all DCP ranks.  Every rank already holds the full BF16 latent; the
        normal ``do_kv_cache_update`` only writes this rank's interleaved
        subset to the paged cache.  The prefetch gathered history from the
        next layer's cache *before* that layer's ``do_kv_cache_update``
        ran, so the current chunk is missing.  This method writes all
        tokens — not just the local rank's share — into the correct slots
        of the rank-ordered gathered buffer.
        """
        if (
            self._ckv_current_chunk_kv_c is None
            or self._ckv_current_chunk_kpe is None
            or num_actual_toks == 0
        ):
            return
        kv_c = self._ckv_current_chunk_kv_c[:num_actual_toks]
        k_pe_flat = self._ckv_current_chunk_kpe[:num_actual_toks]
        if k_pe_flat.ndim == 3:
            k_pe_flat = k_pe_flat.squeeze(1)

        num_reqs = attn_metadata.num_reqs
        interleave = self.cp_kv_cache_interleave_size
        global_seq_lens = attn_metadata.global_cache_seq_lens_per_req
        if global_seq_lens is None:
            return
        global_seq_lens = global_seq_lens[:num_reqs]
        if attn_metadata.req_id_per_token is None:
            return
        req_ids = attn_metadata.req_id_per_token[:num_actual_toks].to(torch.int64)
        global_seq_per_token = global_seq_lens[req_ids].to(torch.int32)

        # Map each current-chunk token to its absolute position within its
        # own sequence. ``num_actual_toks`` spans the whole batch, so the
        # position must be computed per request: the current chunk holds the
        # last ``chunk_len_r`` positions of request r, and this token is the
        # ``(t - query_start_loc[r])``-th of them. Using the batch-global
        # ``t``/``num_actual_toks`` is only correct for a single request and
        # otherwise misplaces every request but the last.
        query_start_loc = attn_metadata.query_start_loc[: num_reqs + 1].to(torch.int32)
        req_chunk_start = query_start_loc[:-1][req_ids]
        req_chunk_len = (query_start_loc[1:] - query_start_loc[:-1])[req_ids]
        t = torch.arange(num_actual_toks, device=self.device, dtype=torch.int32)
        global_pos = global_seq_per_token - req_chunk_len + (t - req_chunk_start)
        owner = ((global_pos // interleave) % self.dcp_world_size).to(torch.int64)
        local_pos = (
            global_pos // (self.dcp_world_size * interleave) * interleave
            + global_pos % interleave
        ).to(torch.int64)

        rank_req_starts = attn_metadata.dcp_rank_req_starts
        if rank_req_starts is None:
            return
        flat_idx = owner * num_reqs + req_ids
        # ``dcp_rank_req_starts`` is a ``[dcp, :num_reqs]`` slice of a
        # ``[dcp, max_seqs]`` buffer, so it is non-contiguous whenever
        # ``num_reqs < max_seqs``; ``reshape`` compacts to row-length
        # ``num_reqs`` to match ``flat_idx``. ``view`` would raise here.
        rank_start = rank_req_starts.reshape(-1)[flat_idx].to(torch.int64)

        padded_tokens = attn_metadata.dcp_padded_total_tokens
        slots = owner * int(padded_tokens) + rank_start + local_pos

        k_scale = getattr(layer, "_k_scale", None)
        if self._kv_fp8_rope:
            self._concat_and_cache_nvfp4_mla_fp8_rope(
                kv_c,
                k_pe_flat,
                gathered_buffer,
                slots,
                k_scale,
            )
        elif self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"):
            ops.concat_and_cache_mla(
                kv_c,
                k_pe_flat,
                gathered_buffer,
                slots,
                self.kv_cache_dtype,
                k_scale,
            )
        elif self._is_kvarn_mla:
            self._stage_current_kvarn_mla_fp8_records(
                kv_c,
                k_pe_flat,
                gathered_buffer,
                slots,
            )

    def _sync_warmup(self) -> None:
        if self.device.type == "cuda":
            torch.accelerator.synchronize(self.device)
        if self.dcp_world_size <= 1:
            return
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            get_dcp_group().barrier()
        except Exception:
            return
        finally:
            if self.device.type == "cuda":
                torch.accelerator.synchronize(self.device)

    def _b12x_kernel_format_kwargs(self, latent_scale: float = 1.0) -> dict[str, Any]:
        if self._b12x_scale_format is None:
            return {}
        return {
            "latent_scale": float(latent_scale),
            "scale_format": self._b12x_scale_format,
        }

    def _prewarm_extend_kernels_once(self, max_batched: int) -> None:
        if self.device.type != "cuda":
            return
        key = (
            self.device.index,
            self.q_head_dim,
            self.kv_lora_rank,
            self._kernel_num_heads,
            int(self.topk_tokens),
            int(self.block_size),
            bool(self.need_to_return_lse_for_decode),
            self.kv_cache_dtype,
            bool(self._kv_fp8_rope),
        )
        if key in _EXTEND_PREWARM_DONE:
            return
        _EXTEND_PREWARM_DONE.add(key)
        kernel_format_kwargs = self._b12x_kernel_format_kwargs()

        rows_to_warm = (1, 2, 4, max(1, int(max_batched)))
        seen_rows: set[int] = set()
        # GLM fp8_ds_mla cache records are 656 B/token; the real KV cache is
        # laid out (num_blocks, block_size, 656) (see the allocator at the
        # block-shape branch above), so a page's stride(0) = block_size*656.
        # The prewarm dummy must match that layout -- (1, block_size, 656) --
        # so _cache_block_stride_bytes sees stride >= page_size*656. The prior
        # (block_size, 1, 656) shape put block_size in dim 0, giving stride(0)
        # = 656 < page_size*656, which tripped the SM120 stride assertion
        # whenever this prewarm ran (i.e. spec + cudagraphs, the first config
        # to reach here; verifier-only and eager-snap both skipped it).
        # One page is enough: prewarm top-k indices all point at slot zero.
        kv_cache = torch.zeros(
            (1, self.block_size, self._kv_record_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        for rows in rows_to_warm:
            rows = int(rows)
            if rows in seen_rows:
                continue
            seen_rows.add(rows)
            q = torch.zeros(
                (rows, self._kernel_num_heads, self.q_head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            selected_indices = torch.zeros(
                (rows, self.topk_tokens), dtype=torch.int32, device=self.device
            )
            cache_seqlens = torch.full(
                (1,), self.block_size, dtype=torch.int32, device=self.device
            )
            nsa_cache_seqlens = torch.ones(
                (rows,), dtype=torch.int32, device=self.device
            )
            scratch_storage = torch.empty(
                (self._scratch_nbytes,), dtype=torch.uint8, device=self.device
            )
            binding = self._extend_plan.bind(
                scratch=scratch_storage,
                q=q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    return_lse=True,
                    lse_scale="natural",
                    **kernel_format_kwargs,
                )
            else:
                self._sparse_mla_extend_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    **kernel_format_kwargs,
                )
            self._sync_warmup()

        _prewarm_dcp_project_bf16_bmm(
            self._input_num_heads,
            self.kv_lora_rank,
            self.v_head_dim,
            self.device.index if self.device.index is not None else 0,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: B12xMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._is_kvarn_mla and self._kvarn_group_key is None:
            q_tensor = q[0] if isinstance(q, tuple) else q
            output_shape = (*q_tensor.shape[:-1], self.kv_lora_rank)
            return torch.zeros(
                output_shape, dtype=q_tensor.dtype, device=q_tensor.device
            ), None
        # Stored by MultiHeadLatentAttentionWrapper as a host float. Avoid
        # reading device state or allocating per-call CUDA state; the CuTe
        # launch receives this as a runtime scalar.
        latent_scale = float(getattr(layer, "_nvfp4_mla_outer_scale", 1.0))
        kernel_format_kwargs = self._b12x_kernel_format_kwargs(latent_scale)
        query_rows = q[0].shape[0] if isinstance(q, tuple) else q.shape[0]
        use_spec_decode_kernel = self.spec_extend_as_decode and (
            self.spec_extend_as_decode_force or attn_metadata.is_spec_decode
        )
        use_decode_kernel = attn_metadata.max_query_len <= 1 or (
            use_spec_decode_kernel
            and attn_metadata.max_query_len <= self.spec_decode_max_q
            and query_rows <= attn_metadata.num_reqs * self.spec_decode_max_q
            and query_rows <= self._decode_max_rows
        )
        direct_native_rows = (
            _direct_packed_kvarn_mla_rows(
                use_decode_kernel=use_decode_kernel,
                direct_packed_enabled=self._kvarn_direct_packed_decode,
                num_actual_toks=int(query_rows),
            )
            if (
                self._is_kvarn_mla
                and self._kvarn_direct_target_model
                and self._kvarn_native_packed_decode is not None
            )
            else 0
        )
        use_direct_native = direct_native_rows > 0
        use_ckv_gather = self.dcp_prefill_ckv_gather_eligible(
            attn_metadata, int(query_rows)
        )
        workspace_tensors = self._borrow_workspaces()
        q_workspace = workspace_tensors[0]
        dense_out_workspace = workspace_tensors[1] if self._pad_heads else None
        scratch_storage = workspace_tensors[-1]
        expected_input_heads = (
            self.num_heads if use_ckv_gather else self._input_num_heads
        )
        if use_ckv_gather:
            local_q_numel = (
                self._max_batched * self._ckv_kernel_num_heads * self.q_head_dim
            )
            q_buffer = q_workspace.view(-1)[:local_q_numel].view(
                self._max_batched,
                self._ckv_kernel_num_heads,
                self.q_head_dim,
            )
        else:
            q_buffer = q_workspace
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            num_actual_toks = ql_nope.shape[0]
            num_input_heads = ql_nope.shape[1]
            if num_input_heads != expected_input_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {expected_input_heads}."
                )
            q_rows = direct_native_rows or num_actual_toks
            q_buffer = q_buffer[:q_rows]
            q_all = q_buffer[:, :num_input_heads]
            if q_rows > num_actual_toks:
                q_all[num_actual_toks:].zero_()
            if self._is_kvarn_mla:
                q_all[:num_actual_toks, ..., : self.kv_lora_rank].copy_(
                    ql_nope
                    if use_direct_native
                    else kvarn_hadamard(ql_nope, self._kvarn_h)
                )
                q_all[:num_actual_toks, ..., self.kv_lora_rank :].copy_(q_pe)
            else:
                ops.concat_mla_q(ql_nope, q_pe, q_all)
        else:
            num_actual_toks = q.shape[0]
            num_input_heads = q.shape[1]
            if num_input_heads != expected_input_heads:
                raise ValueError(
                    "B12X_MLA_SPARSE query heads do not match the planned "
                    f"head count: {num_input_heads} != {expected_input_heads}."
                )
            q_rows = direct_native_rows or num_actual_toks
            q_buffer = q_buffer[:q_rows]
            q_all = q_buffer[:, :num_input_heads]
            if q_rows > num_actual_toks:
                q_all[num_actual_toks:].zero_()
            q_actual = q_all[:num_actual_toks]
            exact_workspace_alias = (
                tuple(q.shape) == tuple(q_actual.shape)
                and tuple(q.stride()) == tuple(q_actual.stride())
                and q.dtype == q_actual.dtype
                and q.device == q_actual.device
                and q.untyped_storage().data_ptr()
                == q_actual.untyped_storage().data_ptr()
                and q.storage_offset() == q_actual.storage_offset()
            )
            if not exact_workspace_alias:
                q_actual.copy_(q.contiguous())
            if self._is_kvarn_mla and not use_direct_native:
                q_actual[..., : self.kv_lora_rank].copy_(
                    kvarn_hadamard(
                        q[..., : self.kv_lora_rank],
                        self._kvarn_h,
                    )
                )

        if self._is_kvarn_mla:
            _, q_inv = self._kvarn_layer_alpha()
            if q_inv != 1.0:
                q_all[:num_actual_toks, ..., : self.kv_lora_rank].mul_(q_inv)
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        per_token_cache = attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
        if use_ckv_gather:
            assert attn_metadata.req_id_per_token is not None
            assert attn_metadata.ckv_page_table_1 is not None
            assert attn_metadata.ckv_nsa_cache_seqlens is not None
            assert attn_metadata.dcp_rank_req_starts is not None
            assert attn_metadata.dcp_rank_req_lens is not None
            selected_indices = attn_metadata.ckv_page_table_1[
                :num_actual_toks, : topk_indices.shape[1]
            ]
            nsa_cache_seqlens = attn_metadata.ckv_nsa_cache_seqlens[:num_actual_toks]
            _map_global_topk_to_gathered_ckv(
                attn_metadata.req_id_per_token[:num_actual_toks],
                topk_indices,
                attn_metadata.dcp_rank_req_starts,
                attn_metadata.dcp_rank_req_lens,
                selected_indices,
                nsa_cache_seqlens,
                dcp_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                padded_rank_tokens=attn_metadata.dcp_padded_total_tokens,
            )
            # The gathered buffer holds the *global* sequence and the extend
            # kernel attends it in full (cache_seqlens = global lengths), so
            # the count of selected entries must be capped by the global
            # per-token causal length. ``per_token_cache`` is the *local*
            # per-rank length (~global/dcp); using it drops the other ranks'
            # selected tokens whenever it is the smaller bound
            # (local_len < min(topk, global_len)) -- i.e. for short contexts,
            # which shows up as small-context-only garbage.
            assert attn_metadata.global_cache_seq_lens_per_req is not None
            global_causal_len = _global_causal_lens_for_ckv_gather(
                attn_metadata.global_cache_seq_lens_per_req,
                attn_metadata.query_start_loc,
                attn_metadata.req_id_per_token,
                num_actual_toks,
            )
            if _SPARSE_META_FUSED_ENABLED:
                _mask_page_table_after_nsa_len(
                    selected_indices, nsa_cache_seqlens, clamp=global_causal_len
                )
            else:
                torch.minimum(
                    nsa_cache_seqlens,
                    global_causal_len,
                    out=nsa_cache_seqlens,
                )
                _mask_page_table_after_nsa_len(
                    selected_indices, nsa_cache_seqlens
                )
        elif self.dcp_world_size > 1:
            # The indexer globally merges logical top-k ids across DCP ranks.
            # Compact just this rank's winners into local physical cache slots;
            # the outer MLA layer combines the rank-local outputs using LSE.
            assert attn_metadata.req_id_per_token is not None
            assert attn_metadata.page_table_1 is not None
            assert attn_metadata.nsa_cache_seqlens is not None
            selected_indices = attn_metadata.page_table_1[
                :num_actual_toks, : topk_indices.shape[1]
            ]
            nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[:num_actual_toks]
            # Zero-copy: the kernel scatters directly into the persistent
            # CUDA-graph-stable views consumed by the b12x planned kernels.
            meta_fused = _SPARSE_META_FUSED_ENABLED
            if meta_fused:
                # One fused -1/0 reset replaces out.fill_(-1) +
                # valid_counts.zero_() inside the conversion wrapper; the
                # defensive -1 padding for shared page_table_1 consumers is
                # preserved bit-for-bit.
                _reset_page_table_and_counts(selected_indices, nsa_cache_seqlens)
            triton_filter_and_convert_dcp_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                dcp_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                out=selected_indices,
                valid_counts=nsa_cache_seqlens,
                out_preinitialized=meta_fused,
            )
            if meta_fused:
                _mask_page_table_after_nsa_len(
                    selected_indices, nsa_cache_seqlens, clamp=per_token_cache
                )
            else:
                torch.minimum(
                    nsa_cache_seqlens,
                    per_token_cache,
                    out=nsa_cache_seqlens,
                )
                _mask_page_table_after_nsa_len(
                    selected_indices, nsa_cache_seqlens
                )
        else:
            # Without DCP, the b12x indexer writes flat physical cache slots
            # directly into the shared top-k buffer.
            selected_indices = topk_indices
            nsa_cache_seqlens = per_token_cache
        if use_direct_native and direct_native_rows > num_actual_toks:
            if use_ckv_gather:
                assert attn_metadata.ckv_page_table_1 is not None
                assert attn_metadata.ckv_nsa_cache_seqlens is not None
                selected_indices = torch.as_strided(
                    attn_metadata.ckv_page_table_1,
                    (direct_native_rows, topk_indices.shape[1]),
                    attn_metadata.ckv_page_table_1.stride(),
                )
                nsa_cache_seqlens = torch.as_strided(
                    attn_metadata.ckv_nsa_cache_seqlens,
                    (direct_native_rows,),
                    attn_metadata.ckv_nsa_cache_seqlens.stride(),
                )
            elif self.dcp_world_size > 1:
                assert attn_metadata.page_table_1 is not None
                assert attn_metadata.nsa_cache_seqlens is not None
                selected_indices = torch.as_strided(
                    attn_metadata.page_table_1,
                    (direct_native_rows, topk_indices.shape[1]),
                    attn_metadata.page_table_1.stride(),
                )
                nsa_cache_seqlens = torch.as_strided(
                    attn_metadata.nsa_cache_seqlens,
                    (direct_native_rows,),
                    attn_metadata.nsa_cache_seqlens.stride(),
                )
            else:
                raise RuntimeError(
                    "Padded direct KVarN MLA decode requires DCP metadata"
                )
            selected_indices[num_actual_toks:].fill_(-1)
            nsa_cache_seqlens[num_actual_toks:].zero_()

        if self._is_kvarn_mla:
            self._ensure_kvarn_mla_cache(kv_c_and_k_pe_cache)
            if use_direct_native:
                if attn_metadata.max_query_len <= 1:
                    cache_seqlens = attn_metadata.cache_seq_lens_per_req
                else:
                    cache_seqlens = torch.as_strided(
                        attn_metadata.cache_seq_lens_per_token,
                        (direct_native_rows,),
                        attn_metadata.cache_seq_lens_per_token.stride(),
                    )
                    if direct_native_rows > num_actual_toks:
                        cache_seqlens[num_actual_toks:].zero_()
                binding = self._decode_plan.bind(
                    scratch=scratch_storage,
                    q=q_all[:, : self._input_num_heads],
                    selected_indices=selected_indices,
                    cache_seqlens_int32=cache_seqlens,
                    nsa_cache_seqlens_int32=nsa_cache_seqlens,
                )
                self._kvarn_diag_selected_indices = binding.selected_indices
                self._kvarn_diag_valid_counts = nsa_cache_seqlens
                direct_scratch = binding.scratch
                if (
                    direct_scratch.tmp_output is None
                    or direct_scratch.tmp_lse is None
                    or direct_scratch.output_buffer is None
                    or direct_scratch.final_lse is None
                ):
                    raise RuntimeError(
                        "Direct packed KVarN MLA decode scratch is incomplete"
                    )
                assert self._kvarn_block_to_slot is not None
                assert self._kvarn_config is not None
                from vllm.v1.attention.ops.kvarn_mla import (
                    direct_packed_kvarn_mla_decode,
                )

                out, lse = direct_packed_kvarn_mla_decode(
                    binding.q,
                    binding.selected_indices,
                    binding.nsa_cache_seqlens_int32,
                    kv_c_and_k_pe_cache,
                    self._kvarn_block_to_slot,
                    self._kvarn_latent_pool,
                    self._kvarn_rope_pool,
                    direct_scratch.tmp_output,
                    direct_scratch.tmp_lse,
                    direct_scratch.output_buffer,
                    direct_scratch.final_lse,
                    sm_scale=self.scale,
                    candidate_envelope=self.topk_tokens,
                    config=self._kvarn_config,
                    native_decode=self._kvarn_native_packed_decode,
                    exact_pool_only=False,
                    fuse_kvarn_hadamard=True,
                )
                _, dn_inv = self._kvarn_layer_alpha()
                return (
                    out[:num_actual_toks] * dn_inv if dn_inv != 1.0 else out[:num_actual_toks],
                    (
                        lse[:num_actual_toks]
                        if self.need_to_return_lse_for_decode
                        else None
                    ),
                )
            if use_decode_kernel:
                kv_cache, selected_indices = (
                    self._stage_selected_kvarn_mla_fp8_cache(
                        selected_indices,
                        kv_c_and_k_pe_cache,
                    )
                )
            else:
                if use_ckv_gather:
                    kv_cache = kv_c_and_k_pe_cache
                else:
                    kv_cache, selected_indices = self._stage_kvarn_mla_fp8_cache(
                        selected_indices,
                        attn_metadata,
                        kv_c_and_k_pe_cache,
                    )
            kernel_format_kwargs = {"latent_scale": 1.0, "scale_format": 1}
        else:
            # KV cache -> paged rank-3 uint8. B12X unified SM120 kernels consume
            # flat slot ids in selected_indices, but compute raw byte offsets as:
            #   block = slot // page_size, local = slot % page_size
            # so the cache tensor itself must expose a per-block stride of
            # block_size * record_bytes. The older split path used a token-flat
            # (num_slots, 1, bytes) view; that makes stride(0) one record and breaks
            # the unified block-stride contract.
            kv_u8 = kv_c_and_k_pe_cache.view(torch.uint8)
            if kv_u8.ndim == 3 and kv_u8.shape[1] == self.block_size:
                kv_cache = kv_u8
            elif kv_u8.ndim == 3 and kv_u8.shape[1] == 1:
                if kv_u8.shape[0] % self.block_size != 0:
                    raise ValueError(
                        "B12X_MLA_SPARSE flat KV cache rows must be divisible by "
                        f"block_size={self.block_size}; got {kv_u8.shape[0]}"
                    )
                kv_cache = kv_u8.reshape(-1, self.block_size, kv_u8.shape[-1])
            else:
                raise ValueError(
                    f"B12X_MLA_SPARSE expected {self.kv_cache_dtype} KV cache as "
                    f"(blocks,{self.block_size},bytes) or (slots,1,bytes), got "
                    f"{tuple(kv_u8.shape)}"
                )
        if not kv_cache.is_contiguous():
            raise ValueError(
                "B12X_MLA_SPARSE requires a contiguous native paged KV cache; "
                f"got stride={tuple(kv_cache.stride())}"
            )
        layer_idx = self._resolve_layer_index(layer)
        prefetch_registry = attn_metadata.ckv_prefetch_registry
        if (
            self._ckv_gather_enabled
            and self._ckv_prefetch_depth > 0
            and layer_idx is not None
            and prefetch_registry is not None
            and self._ckv_workspace_pool is not None
        ):
            # The prefetch registry must hold the layer's real paged cache.
            # ``kv_cache`` may have been rebound above to a per-step staging
            # view (decode/extend selected-record staging) that is shared
            # across all MLA layers; registering it poisons every
            # ``layer_caches`` entry, and the next gather-eligible request's
            # side-stream prefetches then gather all layers from that one
            # stale shared tensor (identical wrong KV for every layer).
            cache_state = prefetch_registry.for_workspace(
                q_workspace,
                layer_idx,
                kv_c_and_k_pe_cache,
                workspace_pool=self._ckv_workspace_pool,
            )
            cache_state.register_cache(layer_idx, kv_c_and_k_pe_cache, self)
        if use_ckv_gather:
            if prefetch_registry is None:
                raise RuntimeError("CKV gather requires a prefetch state registry")
            if self._ckv_workspace_pool is None:
                raise RuntimeError("CKV gather requires a persistent workspace pool")
            prefetch_state = prefetch_registry.for_workspace(
                q_workspace,
                layer_idx,
                kv_c_and_k_pe_cache,
                workspace_pool=self._ckv_workspace_pool,
            )
            ckv_workspace = prefetch_state.get_ckv_workspace(self._ckv_workspace_nbytes)
            if layer_idx is not None and self._ckv_prefetch_depth > 0:
                prefetch_state.enter_layer(layer_idx)
                prefetch_state.register_cache(layer_idx, kv_c_and_k_pe_cache, self)
            pending = (
                prefetch_state.pending_layers.pop(layer_idx, None)
                if layer_idx is not None and self._ckv_prefetch_depth > 0
                else None
            )
            if pending is not None:
                gather_event, current_buf_idx = pending
                gather_event.wait()
                _, gathered_buffer = self._ckv_workspace_views(
                    ckv_workspace, current_buf_idx
                )
                kv_cache = gathered_buffer[
                    : self.dcp_world_size * self._ckv_local_capacity
                ].view(-1, self.block_size, self._kv_record_bytes)
                self._append_current_chunk_to_gathered(
                    kv_cache,
                    attn_metadata,
                    layer,
                    num_actual_toks,
                )
            else:
                # The ring shares one local staging region across gathered
                # slots. An irregular fallback can occur while a future layer
                # is pending, so order this main-stream write after all current
                # side-stream users without increasing the persistent pool.
                prefetch_state.wait_for_pending_writes()
                current_buf_idx = (
                    layer_idx % self._ckv_workspace_slots
                    if layer_idx is not None
                    else 0
                )
                kv_cache = self._dcp_gather_ckv(
                    kv_cache,
                    attn_metadata,
                    ckv_workspace,
                    buf_idx=current_buf_idx,
                )
            logger.info_once(
                "Using transient full-CKV gather for B12X sparse MLA prefill "
                "(capacity=%d logical tokens)",
                self._ckv_gather_max_tokens,
            )
            if (
                self._ckv_prefetch_supported
                and self._ckv_prefetch_depth > 0
                and layer_idx is not None
            ):
                # The first eligible request only discovers one layer cache at
                # a time. It intentionally has no lookahead and primes the
                # cache registry for subsequent requests.
                targets = _ckv_prefetch_target_indices(
                    layer_idx,
                    self._ckv_prefetch_depth,
                    prefetch_state.layer_caches,
                    prefetch_state.pending_layers,
                )
                prefetch_stream = (
                    prefetch_state.get_gather_stream() if targets else None
                )
                for target_idx in targets:
                    assert prefetch_stream is not None
                    target_kv = prefetch_state.layer_caches[target_idx]
                    if target_kv is None:
                        raise RuntimeError(
                            f"CKV prefetch target {target_idx} has no cache"
                        )
                    if target_idx >= len(prefetch_state.layer_owners):
                        raise RuntimeError(
                            f"CKV prefetch target {target_idx} has no staging owner"
                        )
                    target_owner_ref = prefetch_state.layer_owners[target_idx]
                    target_owner = (
                        target_owner_ref() if target_owner_ref is not None else None
                    )
                    if target_owner is None:
                        raise RuntimeError(
                            f"CKV prefetch target {target_idx} staging owner expired"
                        )
                    if (
                        getattr(target_owner, "kv_cache_dtype", None)
                        != self.kv_cache_dtype
                    ):
                        # Layer-wise mixed KV precision: the target layer packs a
                        # different record geometry. Staging across the dtype
                        # boundary reinterprets foreign bytes (the eviction-mode
                        # corruption); skip the target instead - it gathers
                        # synchronously on its own forward.
                        continue
                    for attribute in (
                        "block_size",
                        "dcp_world_size",
                        "_ckv_local_capacity",
                        "_kv_record_bytes",
                    ):
                        if getattr(target_owner, attribute) != getattr(self, attribute):
                            raise RuntimeError(
                                "CKV prefetch target staging geometry mismatch: "
                                f"{attribute} target={getattr(target_owner, attribute)} "
                                f"source={getattr(self, attribute)}"
                            )
                    if bool(target_owner._is_kvarn_mla) != bool(self._is_kvarn_mla):
                        raise RuntimeError(
                            "CKV prefetch target changed KVarN staging format"
                        )
                    target_buf_idx = target_idx % self._ckv_workspace_slots
                    target_owner._dcp_gather_ckv(
                        target_kv,
                        attn_metadata,
                        ckv_workspace,
                        buf_idx=target_buf_idx,
                        stream=prefetch_stream,
                    )
                    target_event = torch.cuda.Event(blocking=False)
                    target_event.record(prefetch_stream)
                    prefetch_state.pending_layers[target_idx] = (
                        target_event,
                        target_buf_idx,
                    )

        use_spec_decode_kernel = self.spec_extend_as_decode and (
            self.spec_extend_as_decode_force or attn_metadata.is_spec_decode
        )
        use_decode_kernel = attn_metadata.max_query_len <= 1 or (
            use_spec_decode_kernel
            and attn_metadata.max_query_len <= self.spec_decode_max_q
            and num_actual_toks <= attn_metadata.num_reqs * self.spec_decode_max_q
            and num_actual_toks <= self._decode_max_rows
        )
        if use_decode_kernel:
            cache_seqlens = (
                attn_metadata.cache_seq_lens_per_req
                if attn_metadata.max_query_len <= 1
                else attn_metadata.cache_seq_lens_per_token[:num_actual_toks]
            )
            decode_q = q_all
            if self._pad_heads:
                decode_q = q_buffer[:, : self._kernel_num_heads]
                decode_q[:, self._input_num_heads :, :].zero_()
            # Eager bind maps caller-owned scratch into views. forced_num_splits
            # pins the planner choice for this captured graph; the merge kernel is
            # specialized on that count and needs no device-side control fill.
            binding = self._decode_plan.bind(
                scratch=scratch_storage,
                q=decode_q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            if self.need_to_return_lse_for_decode:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_decode_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        forced_num_splits=self._num_splits_cap,
                        return_lse=True,
                        lse_scale="natural",
                        **kernel_format_kwargs,
                    ),
                )
                if self._pad_heads:
                    assert dense_out_workspace is not None
                    dense_out = dense_out_workspace[:num_actual_toks]
                    dense_out.copy_(out[:, : self._input_num_heads, :])
                    out = dense_out
                    lse = lse[:, : self._input_num_heads]
                return self._restore_kvarn_mla_output(out), lse
            out = cast(
                torch.Tensor,
                self._sparse_mla_decode_forward(
                    binding=binding,
                    kv_cache=kv_cache,
                    sm_scale=self.scale,
                    v_head_dim=self.kv_lora_rank,
                    forced_num_splits=self._num_splits_cap,
                    **kernel_format_kwargs,
                ),
            )
            if self._pad_heads:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
            return self._restore_kvarn_mla_output(out), None
        else:
            # Extend / prefill -> single-pass unified prefill (no split-K
            # scratch needed; only output_buffer is read). b12x supports 8-head
            # granularity, so only a non-aligned local tail is padded here.
            if use_ckv_gather:
                if attn_metadata.global_cache_seq_lens_per_req is None:
                    raise RuntimeError("CKV gather is missing global sequence lengths")
                cache_seqlens = attn_metadata.global_cache_seq_lens_per_req
            else:
                cache_seqlens = attn_metadata.cache_seq_lens_per_req
            prefill_q = q_all
            if self._pad_heads and not use_ckv_gather:
                prefill_q = q_buffer[:, : self._kernel_num_heads]
                prefill_q[:, self._input_num_heads :, :].zero_()

            extend_plan = self._ckv_extend_plan if use_ckv_gather else self._extend_plan
            if extend_plan is None:
                raise RuntimeError("CKV gather extend plan was not initialized")
            binding = extend_plan.bind(
                scratch=scratch_storage,
                q=prefill_q,
                selected_indices=selected_indices,
                cache_seqlens_int32=cache_seqlens,
                nsa_cache_seqlens_int32=nsa_cache_seqlens,
            )
            lse = None
            if self.need_to_return_lse_for_decode and not use_ckv_gather:
                out, lse = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        return_lse=True,
                        lse_scale="natural",
                        **kernel_format_kwargs,
                    ),
                )
            else:
                out = cast(
                    torch.Tensor,
                    self._sparse_mla_extend_forward(
                        binding=binding,
                        kv_cache=kv_cache,
                        sm_scale=self.scale,
                        v_head_dim=self.kv_lora_rank,
                        **kernel_format_kwargs,
                    ),
                )
            if self._pad_heads and not use_ckv_gather:
                assert dense_out_workspace is not None
                dense_out = dense_out_workspace[:num_actual_toks]
                dense_out.copy_(out[:, : self._input_num_heads, :])
                out = dense_out
                if lse is not None:
                    lse = lse[:, : self._input_num_heads]
        return self._restore_kvarn_mla_output(out), lse
