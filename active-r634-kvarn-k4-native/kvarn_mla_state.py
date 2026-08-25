# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle management for MLA KVarN exact-block pool slots."""

from __future__ import annotations

import math
import os
import time
_BOOT_T0 = time.monotonic()
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from weakref import WeakSet

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kvarn.config import KVarNMLAConfig

logger = init_logger(__name__)


class KVarNMLAExactPoolPressure(RuntimeError):
    """Conservative ownership exceeds the graph-static exact-slot pool."""


class KVarNMLAImpl(Protocol):
    layer_name: str
    _is_kvarn_mla: bool
    _kvarn_group_key: tuple[str, ...] | None
    _kvarn_pool_size: int
    device: torch.device

    def _flush_kvarn_mla_blocks(
        self, block_ids: torch.Tensor, pool_slots: torch.Tensor
    ) -> None: ...


class KVarNMLARequestState(Protocol):
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


class _BlockAudit:
    """Ring-buffered per-block transition log for corruption forensics."""

    def __init__(self, capacity: int = 20000):
        self.events: list[tuple] = []
        self.capacity = capacity

    def log(self, block_id, event: str, detail: str = "") -> None:
        self.events.append((time.monotonic(), block_id, event, detail))
        if len(self.events) > self.capacity:
            del self.events[: self.capacity // 2]

    def history(self, block_id, last: int = 40):
        return [e for e in self.events if e[1] == block_id][-last:]


@dataclass
class KVarNMLALiveBlockTracker:
    """Tracks persistent and per-step exact blocks from scheduler CPU state."""

    group_configs: dict[int, tuple[int, int, int, int, int, int, int]]
    request_blocks: dict[str, dict[int, dict[int, int | None]]] = field(
        default_factory=dict
    )
    step_blocks: dict[int, dict[int, int | None]] = field(default_factory=dict)
    pending_blocks: dict[str, dict[int, dict[int, int]]] = field(default_factory=dict)
    resolved_blocks: dict[str, dict[int, dict[int, int]]] = field(default_factory=dict)
    epoch_blocks: dict[str, dict[int, dict[int, int | None]]] = field(
        default_factory=dict
    )
    epoch_owners: dict[str, str] = field(default_factory=dict)
    latest_epoch: dict[str, str] = field(default_factory=dict)
    request_full_context_groups: dict[str, set[int]] = field(default_factory=dict)
    epoch_full_context_groups: dict[str, set[int]] = field(default_factory=dict)
    def __post_init__(self) -> None:
        precision_tail_tokens = int(
            os.environ.get("KVARN_MLA_PRECISION_TAIL_TOKENS", "0")
        )
        if precision_tail_tokens < 0:
            raise ValueError("KVARN_MLA_PRECISION_TAIL_TOKENS must be nonnegative")
        normalized: dict[int, tuple[int, int, int, int, int, int, int]] = {}
        for group_id, config in self.group_configs.items():
            if len(config) == 6:
                (
                    group_size,
                    boundary_tokens,
                    blocks_per_manager_block,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                ) = config
                normalized[group_id] = (
                    group_size,
                    boundary_tokens,
                    precision_tail_tokens,
                    blocks_per_manager_block,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
            elif len(config) == 7:
                normalized[group_id] = config
            else:
                raise ValueError(
                    f"Invalid KVarN MLA group config for {group_id}: {config}"
                )
        self.group_configs = normalized


    @staticmethod
    def _physical_block_id(
        block_ids: list[int],
        logical_block: int,
        blocks_per_manager_block: int,
    ) -> int:
        manager_block, subblock = divmod(logical_block, blocks_per_manager_block)
        return block_ids[manager_block] * blocks_per_manager_block + subblock

    @staticmethod
    def _local_num_tokens(
        num_tokens: int,
        dcp_world_size: int,
        dcp_rank: int,
        dcp_interleave: int,
    ) -> int:
        cycle = dcp_world_size * dcp_interleave
        full_cycles, remainder = divmod(num_tokens, cycle)
        return full_cycles * dcp_interleave + min(
            max(remainder - dcp_rank * dcp_interleave, 0),
            dcp_interleave,
        )

    @staticmethod
    def _merge_fill(
        blocks: dict[int, int | None],
        block_id: int,
        fill: int | None,
    ) -> None:
        if block_id not in blocks:
            blocks[block_id] = fill
            return
        current = blocks[block_id]
        if fill is not None and (current is None or fill > current):
            blocks[block_id] = fill

    def _persistent_group_blocks(
        self,
        block_ids: list[int],
        group_size: int,
        boundary_tokens: int,
        precision_tail_tokens: int,
        blocks_per_manager_block: int,
        local_min_end: int,
        local_max_end: int,
        local_tail_start: int,
    ) -> dict[int, int | None]:
        num_logical_blocks = min(
            math.ceil(local_max_end / group_size),
            len(block_ids) * blocks_per_manager_block,
        )
        if num_logical_blocks == 0:
            return {}

        blocks: dict[int, int | None] = {}
        boundary_blocks = min(
            math.ceil(boundary_tokens / group_size),
            num_logical_blocks,
        )
        for logical_block in range(boundary_blocks):
            fill = min(
                group_size,
                max(local_min_end - logical_block * group_size, 0),
            )
            blocks[
                self._physical_block_id(
                    block_ids,
                    logical_block,
                    blocks_per_manager_block,
                )
            ] = fill or None
        if precision_tail_tokens:
            first_tail = min(local_tail_start // group_size, num_logical_blocks)
            for logical_block in range(first_tail, num_logical_blocks):
                fill = min(
                    group_size,
                    max(local_min_end - logical_block * group_size, 0),
                )
                self._merge_fill(
                    blocks,
                    self._physical_block_id(
                        block_ids,
                        logical_block,
                        blocks_per_manager_block,
                    ),
                    fill or None,
                )

        first_current = max(math.ceil(local_min_end / group_size) - 1, 0)
        for logical_block in range(first_current, num_logical_blocks):
            fill = min(
                group_size,
                max(local_min_end - logical_block * group_size, 0),
            )
            self._merge_fill(
                blocks,
                self._physical_block_id(
                    block_ids,
                    logical_block,
                    blocks_per_manager_block,
                ),
                fill or None,
            )
        return blocks

    def _consume_resolved(self, req_id: str) -> None:
        owner = req_id.split(":")[0]
        for group_id, blocks in self.resolved_blocks.pop(req_id, {}).items():
            step_blocks = self.step_blocks[group_id]
            for block_id, fill in blocks.items():
                holder = None
                for rid, rgroups in self.request_blocks.items():
                    if block_id in rgroups.get(group_id, {}):
                        holder = rid
                        break
                if holder is not None and holder != owner:
                    # Ghost: the physical block id was reused by another
                    # request; a dead owner's correction must not max-merge
                    # into the new owner's fills.
                    continue
                self._merge_fill(step_blocks, block_id, fill)

    def resolve_async(
        self,
        epoch_id: str,
        request: KVarNMLARequestState,
        actual_end_tokens: int,
    ) -> None:
        """Resolve one conservative ownership epoch after acceptance is known."""
        pending = self.pending_blocks.pop(epoch_id, None)
        owner = self.epoch_owners.pop(epoch_id, epoch_id)
        self.epoch_blocks.pop(epoch_id, None)
        full_context_groups = self.epoch_full_context_groups.pop(epoch_id, set())
        if pending is None:
            return

        is_latest = self.latest_epoch.get(owner) == epoch_id
        owner_groups = self.request_blocks.get(owner) if is_latest else None
        live_full_context_groups = (
            self.request_full_context_groups.get(owner, set())
            if owner_groups is not None
            else set()
        )
        resolved: dict[int, dict[int, int]] = {}
        for (
            group_id,
            (
                group_size,
                boundary_tokens,
                precision_tail_tokens,
                blocks_per_manager_block,
                dcp_world_size,
                dcp_rank,
                dcp_interleave,
            ),
        ) in self.group_configs.items():
            pending_group = pending.get(group_id, {})
            trim_pending_only = (
                group_id in full_context_groups and group_id in live_full_context_groups
            )
            if group_id >= len(request.block_ids) or actual_end_tokens <= 0:
                if owner_groups is not None:
                    if trim_pending_only:
                        owned_group = owner_groups.get(group_id, {})
                        for block_id in pending_group:
                            owned_group.pop(block_id, None)
                    else:
                        owner_groups.pop(group_id, None)
                continue

            local_end = self._local_num_tokens(
                actual_end_tokens,
                dcp_world_size,
                dcp_rank,
                dcp_interleave,
            )
            if trim_pending_only:
                if owner_groups is not None:
                    owned_group = owner_groups.setdefault(group_id, {})
                    for block_id, logical_block in pending_group.items():
                        fill = min(
                            group_size,
                            max(local_end - logical_block * group_size, 0),
                        )
                        if fill:
                            owned_group[block_id] = fill
                        else:
                            owned_group.pop(block_id, None)
            elif owner_groups is not None:
                local_tail_start = self._local_num_tokens(
                    max(actual_end_tokens - precision_tail_tokens, 0),
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
                owner_groups[group_id] = self._persistent_group_blocks(
                    request.block_ids[group_id],
                    group_size,
                    boundary_tokens,
                    precision_tail_tokens,
                    blocks_per_manager_block,
                    local_end,
                    local_end,
                    local_tail_start,
                )

            resolved_group: dict[int, int] = {}
            for block_id, logical_block in pending_group.items():
                fill = min(
                    group_size,
                    max(local_end - logical_block * group_size, 0),
                )
                if fill:
                    resolved_group[block_id] = fill
            if resolved_group:
                resolved[group_id] = resolved_group

        if is_latest:
            self.latest_epoch.pop(owner, None)
        if resolved:
            self.resolved_blocks[epoch_id] = resolved

    def discard_epoch(self, epoch_id: str) -> None:
        """Drop a settled stale epoch without publishing obsolete fill counts."""
        self.pending_blocks.pop(epoch_id, None)
        self.epoch_blocks.pop(epoch_id, None)
        self.epoch_full_context_groups.pop(epoch_id, None)
        owner = self.epoch_owners.pop(epoch_id, None)
        if owner is not None and self.latest_epoch.get(owner) == epoch_id:
            self.latest_epoch.pop(owner, None)

    def mark_request_unknown(self, req_id: str) -> None:
        """Prevent stale fills from surviving scheduler block-ID reuse."""
        for blocks in self.request_blocks.get(req_id, {}).values():
            for block_id in blocks:
                blocks[block_id] = None
        for epoch_id, owner in self.epoch_owners.items():
            if owner != req_id:
                continue
            for blocks in self.epoch_blocks[epoch_id].values():
                for block_id in blocks:
                    blocks[block_id] = None

    def update(
        self,
        requests: Mapping[str, KVarNMLARequestState],
        scheduled_tokens: Mapping[str, int],
        finished_req_ids: Iterable[str],
        preempted_req_ids: Iterable[str],
        rollback_tokens: Mapping[str, int] | None = None,
        ownership_epochs: Mapping[str, str] | None = None,
    ) -> None:
        self.step_blocks = {group_id: {} for group_id in self.group_configs}
        removed = set(finished_req_ids)
        removed.update(preempted_req_ids)
        for req_id in removed:
            self.request_blocks.pop(req_id, None)
            self.request_full_context_groups.pop(req_id, None)
            self.latest_epoch.pop(req_id, None)
            for epoch_id, owner in tuple(self.epoch_owners.items()):
                if owner == req_id:
                    self.discard_epoch(epoch_id)
        for epoch_id in tuple(self.resolved_blocks):
            self._consume_resolved(epoch_id)

        rollback_tokens = rollback_tokens or {}
        ownership_epochs = ownership_epochs or {}
        for req_id, num_scheduled_tokens in scheduled_tokens.items():
            request = requests.get(req_id)
            if request is None:
                continue
            epoch_id = ownership_epochs.get(req_id, req_id)
            start_tokens = request.num_computed_tokens
            if epoch_id in self.pending_blocks:
                self.resolve_async(epoch_id, request, start_tokens)
                self._consume_resolved(epoch_id)

            rollback = rollback_tokens.get(req_id, 0)
            min_start_tokens = max(start_tokens - rollback, 0)
            min_end_tokens = min_start_tokens + num_scheduled_tokens
            max_end_tokens = start_tokens + num_scheduled_tokens
            previous_groups = self.request_blocks.get(req_id, {})
            previous_full_context = self.request_full_context_groups.get(req_id, set())
            groups: dict[int, dict[int, int | None]] = {}
            full_context_groups: set[int] = set()
            pending: dict[int, dict[int, int]] = {}
            for (
                group_id,
                (
                    group_size,
                    boundary_tokens,
                    precision_tail_tokens,
                    blocks_per_manager_block,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                ),
            ) in self.group_configs.items():
                if group_id >= len(request.block_ids) or max_end_tokens <= 0:
                    continue
                local_min_start = self._local_num_tokens(
                    min_start_tokens,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
                local_min_end = self._local_num_tokens(
                    min_end_tokens,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
                local_max_end = self._local_num_tokens(
                    max_end_tokens,
                    dcp_world_size,
                    dcp_rank,
                    dcp_interleave,
                )
                block_ids = request.block_ids[group_id]
                num_logical_blocks = min(
                    math.ceil(local_max_end / group_size),
                    len(block_ids) * blocks_per_manager_block,
                )
                if num_logical_blocks == 0:
                    continue

                first_touched: int | None = None
                last_touched = -1
                if local_min_start < local_max_end:
                    first_touched = local_min_start // group_size
                    last_touched = min(
                        (local_max_end - 1) // group_size,
                        num_logical_blocks - 1,
                    )

                full_context = (
                    boundary_tokens + precision_tail_tokens >= max_end_tokens
                )
                if full_context:
                    full_context_groups.add(group_id)
                if (
                    full_context
                    and group_id in previous_full_context
                    and group_id in previous_groups
                ):
                    group_blocks = previous_groups[group_id]
                    first_current = max(math.ceil(local_min_end / group_size) - 1, 0)
                    first_update = (
                        first_current
                        if first_touched is None
                        else min(first_current, first_touched)
                    )
                    for logical_block in range(first_update, num_logical_blocks):
                        block_id = self._physical_block_id(
                            block_ids,
                            logical_block,
                            blocks_per_manager_block,
                        )
                        fill = min(
                            group_size,
                            max(local_min_end - logical_block * group_size, 0),
                        )
                        group_blocks[block_id] = fill or None
                    groups[group_id] = group_blocks
                else:
                    local_tail_start = self._local_num_tokens(
                        max(min_end_tokens - precision_tail_tokens, 0),
                        dcp_world_size,
                        dcp_rank,
                        dcp_interleave,
                    )
                    groups[group_id] = self._persistent_group_blocks(
                        block_ids,
                        group_size,
                        boundary_tokens,
                        precision_tail_tokens,
                        blocks_per_manager_block,
                        local_min_end,
                        local_max_end,
                        local_tail_start,
                    )

                if first_touched is not None:
                    step_blocks = self.step_blocks[group_id]
                    pending_group: dict[int, int] = {}
                    for logical_block in range(first_touched, last_touched + 1):
                        block_id = self._physical_block_id(
                            block_ids,
                            logical_block,
                            blocks_per_manager_block,
                        )
                        fill = min(
                            group_size,
                            max(local_min_end - logical_block * group_size, 0),
                        )
                        self._merge_fill(step_blocks, block_id, fill or None)
                        pending_group[block_id] = logical_block
                    if rollback and pending_group:
                        pending[group_id] = pending_group
            self.request_blocks[req_id] = groups
            self.request_full_context_groups[req_id] = full_context_groups
            if pending:
                retained = {
                    group_id: dict(blocks)
                    for group_id, blocks in groups.items()
                    if group_id not in full_context_groups
                }
                epoch_full_context: set[int] = set()
                for group_id, pending_group in pending.items():
                    if group_id in full_context_groups:
                        epoch_full_context.add(group_id)
                    retained_group = retained.setdefault(group_id, {})
                    step_group = self.step_blocks[group_id]
                    for block_id in pending_group:
                        self._merge_fill(
                            retained_group, block_id, step_group.get(block_id)
                        )
                self.pending_blocks[epoch_id] = pending
                self.epoch_blocks[epoch_id] = retained
                if epoch_full_context:
                    self.epoch_full_context_groups[epoch_id] = epoch_full_context
                self.epoch_owners[epoch_id] = req_id
                self.latest_epoch[req_id] = epoch_id

    def block_fills(self, group_id: int) -> dict[int, int | None]:
        sources = [self.step_blocks.get(group_id, {})]
        sources.extend(
            groups.get(group_id, {}) for groups in self.request_blocks.values()
        )
        sources.extend(
            groups.get(group_id, {}) for groups in self.epoch_blocks.values()
        )
        sources.extend(
            groups.get(group_id, {}) for groups in self.resolved_blocks.values()
        )
        if not sources:
            return {}

        largest_index = max(range(len(sources)), key=lambda index: len(sources[index]))
        fills = dict(sources[largest_index])
        for index, blocks in enumerate(sources):
            if index == largest_index or not blocks:
                continue
            overlap = fills.keys() & blocks.keys()
            previous = {block_id: fills[block_id] for block_id in overlap}
            fills.update(blocks)
            for block_id, fill in previous.items():
                self._merge_fill(fills, block_id, fill)
        return fills


@dataclass
class _GroupState:
    pool_size: int
    packed_format: str = "k5_g64"
    impls: tuple[KVarNMLAImpl, ...] = ()
    mapping: dict[int, int] = field(default_factory=dict)
    free_slots: list[int] = field(default_factory=list)
    block_fill: dict[int, int] = field(default_factory=dict)
    mirrors: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    flushed: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.free_slots = list(range(self.pool_size - 1, -1, -1))


class KVarNMLAStateManager:
    """Shares physical-block-to-exact-slot ownership across MLA layers."""

    _impls: WeakSet[KVarNMLAImpl] = WeakSet()
    _audit = _BlockAudit()
    _diag_steps: int = 0
    _groups: dict[tuple[str, ...], _GroupState] = {}

    @classmethod
    def register(cls, impl: KVarNMLAImpl) -> None:
        cls._impls.add(impl)

    @classmethod
    def validate_records_storage(cls, kv_cache) -> None:
        """Drop flushed membership if the paged record tensor was reallocated.

        Mapping (exact pool rows, impl-side) survives reallocation; packed
        records do not. Restoring from wiped records yields zeros - detect
        the storage change at first re-bind and retire only the record
        provenance.
        """
        ptr = kv_cache.data_ptr()
        for state in cls._groups.values():
            prev = getattr(state, "records_ptr", None)
            if prev is not None and prev != ptr:
                state.flushed.clear()
            state.records_ptr = ptr

    @classmethod
    def rebind_cache_pointers(cls) -> None:
        """Drop impl->tensor bindings only; ownership state survives.

        Safe mid-serving (e.g. cudagraph memory profiling re-runs): the KV
        tensors are NOT reallocated, so mappings/flushed sets must persist.
        """
        cls._diag_steps = 0
        for impl in cls._impls:
            impl._kvarn_group_key = None
            impl._kvarn_cache_ref = None  # type: ignore[attr-defined]
            impl._kvarn_block_to_slot = None  # type: ignore[attr-defined]
            impl._kvarn_block_to_logical = None  # type: ignore[attr-defined]

    @classmethod
    def reset_cache_bindings(cls) -> None:
        if cls._groups:
            import traceback
            logger.warning(
                "KVarN reset_cache_bindings clearing %d group states; "
                "uptime=%.1fs; caller:\n%s",
                len(cls._groups),
                time.monotonic() - _BOOT_T0,
                len(cls._groups),
                time.monotonic() - _BOOT_T0,
                "".join(traceback.format_stack()[-14:-1]),
            )
        cls._groups.clear()
        cls.rebind_cache_pointers()
        cls._diag_steps = 0
        for impl in cls._impls:
            impl._kvarn_group_key = None
            impl._kvarn_cache_ref = None  # type: ignore[attr-defined]
            impl._kvarn_block_to_slot = None  # type: ignore[attr-defined]
            impl._kvarn_block_to_logical = None  # type: ignore[attr-defined]

    @classmethod
    def ensure_mirror(
        cls,
        group_key: tuple[str, ...],
        device: torch.device,
        num_blocks: int,
    ) -> torch.Tensor:
        state = cls._groups[group_key]
        mirror = state.mirrors.get(device)
        if mirror is None or mirror.shape[0] < num_blocks:
            mirror = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
            state.mirrors[device] = mirror
            cls._sync_mirror(state, mirror)
        return mirror

    @staticmethod
    def _sync_mirror(state: _GroupState, mirror: torch.Tensor) -> None:
        mirror.fill_(-1)
        if not state.mapping:
            return
        block_ids = torch.tensor(
            list(state.mapping), dtype=torch.long, device=mirror.device
        )
        slots = torch.tensor(
            list(state.mapping.values()), dtype=torch.int32, device=mirror.device
        )
        valid = block_ids < mirror.shape[0]
        mirror[block_ids[valid]] = slots[valid]

    @staticmethod
    def _update_mirror(mirror: torch.Tensor, updates: dict[int, int]) -> None:
        if not updates:
            return
        block_ids = torch.tensor(list(updates), dtype=torch.long, device=mirror.device)
        slots = torch.tensor(
            list(updates.values()), dtype=torch.int32, device=mirror.device
        )
        valid = block_ids < mirror.shape[0]
        mirror[block_ids[valid]] = slots[valid]

    @classmethod
    def prepare_step(
        cls,
        group_key: tuple[str, ...],
        layer_names: list[str],
        common_metadata,
        config: KVarNMLAConfig,
        dcp_world_size: int,
    ) -> None:
        impls = tuple(
            impl
            for impl in cls._impls
            if impl._is_kvarn_mla and impl.layer_name in layer_names
        )
        if not impls:
            return
        # Key the shared state by SPEC IDENTITY, not the calling backend
        # instance's layer-name tuple: the target instance and the MTP draft
        # instance serve the SAME cache spec and must share one _GroupState,
        # or each allocates the same physical blocks in separate states
        # (provenance-lost orphans; audit 2026-08-25).
        group_key = (
            config.bits,
            config.group,
            config.latent_dim,
            config.rope_dim,
        )
        for impl in impls:
            impl._kvarn_group_key = group_key

        pool_size = min(impl._kvarn_pool_size for impl in impls)
        state = cls._groups.get(group_key)
        if state is None:
            state = _GroupState(pool_size=pool_size, impls=impls)
            cls._groups[group_key] = state
        elif state.pool_size != pool_size:
            raise RuntimeError(
                "MLA KVarN layers in one cache group must share pool size"
            )

        block_fills = common_metadata.kvarn_mla_block_fills
        if block_fills is None:
            raise RuntimeError("MLA KVarN requires exact-block ownership metadata")
        needed = set(block_fills)
        for block_id, fill in block_fills.items():
            if fill is None:
                state.block_fill.pop(block_id, None)
            else:
                # The current ownership snapshot is authoritative. A physical
                # block can be reused by a new request without an intervening
                # empty step, and rollback can also lower its live fill.
                state.block_fill[block_id] = fill

        retired = [block_id for block_id in state.mapping if block_id not in needed]
        flush_ids = [
            block_id
            for block_id in retired
            if state.block_fill.get(block_id, 0) >= config.group
            or block_id in state.flushed
        ]
        for _fb in flush_ids:
            cls._audit.log(_fb, "flush_pack", f"fill={state.block_fill.get(_fb)} st={group_key[-1] if isinstance(group_key, tuple) else group_key}")
        if flush_ids:
            device = impls[0].device
            block_ids = torch.tensor(flush_ids, dtype=torch.long, device=device)
            pool_slots = torch.tensor(
                [state.mapping[block_id] for block_id in flush_ids],
                dtype=torch.long,
                device=device,
            )
            for impl in impls:
                impl._flush_kvarn_mla_blocks(block_ids, pool_slots)
            # The paged packed copy of these blocks is now valid, so a later
            # prefix-cache hit can restore exact rows from it.
            state.flushed.update(flush_ids)

        # VALUE FORENSICS: a mapped block's pool content may only change
        # when that block is scheduled (scattered) this step. Any checksum
        # change on an unscheduled block is the corruption in the act.
        if os.environ.get("KVARN_MLA_VALUE_WATCH", "0") == "1":
            _pool = next(
                (
                    impl._kvarn_latent_pool
                    for impl in state.impls
                    if getattr(impl, "_kvarn_latent_pool", None) is not None
                ),
                None,
            )
            if _pool is not None:
                _prev = getattr(state, "slot_sums", None)
                _sums = {}
                with torch.no_grad():
                    _flat = _pool.float().reshape(_pool.shape[0], -1).sum(dim=1)
                for b, s in state.mapping.items():
                    _sums[b] = round(_flat[s].item(), 3)
                state.slot_sums = _sums
                if _prev is not None:
                    _bad = [
                        (b, _prev.get(b), _sums[b])
                        for b in _sums
                        if b in _prev
                        and b not in block_fills
                        and _prev[b] != _sums[b]
                    ]
                    if _bad:
                        logger.warning(
                            "KVARN-SILENT-WRITE n=%d ex=%s",
                            len(_bad),
                            _bad[:5],
                        )
        _lost_flushed = [b for b in retired if b in state.flushed]
        if _lost_flushed:
            cls._audit.log(
                _lost_flushed[0],
                "retired_still_flushed",
                f"n={len(_lost_flushed)} ids={_lost_flushed[:6]}",
            )
        for block_id in retired:
            state.free_slots.append(state.mapping.pop(block_id))
            fill = state.block_fill.pop(block_id, None)
            cls._audit.log(
                block_id,
                "retire",
                f"fill={fill} flushed={block_id in state.flushed}",
            )
            if fill is not None and fill < config.group and block_id not in (
                state.flushed
            ):
                # A never-flushed block retired below full fill has no packed
                # copy of its current content. Blocks already in flushed are
                # re-packed above (pack is fill-unbounded; readers are
                # fill-bounded), so their membership stays valid.
                pass
            # A block with NO fill entry (accounting dropped on request
            # removal or stale-fill guard) that was previously flushed-full
            # still has a valid packed copy: nothing rewrote it. Keep the
            # flushed membership so future re-entries restore from it.

        missing = sorted(needed.difference(state.mapping))
        _orphans = [
            block_id
            for block_id in missing
            if block_id not in state.flushed
            and (block_fills.get(block_id) or 0) >= config.group
        ]
        if len(missing) > len(state.free_slots):
            raise RuntimeError(
                "MLA KVarN exact-block pool exhausted: "
                f"need {len(missing)} slots, have {len(state.free_slots)}. "
                "Reduce max_num_seqs or max_num_batched_tokens."
            )
        mirror_updates = {block_id: -1 for block_id in retired}
        _orphan_set = set(_orphans) if _orphans else set()
        for block_id in missing:
            slot = state.free_slots.pop()
            state.mapping[block_id] = slot
            mirror_updates[block_id] = slot
            if block_id in _orphan_set:
                # Provenance-lost block: its recycled slot holds ANOTHER
                # block's rows. Serving them cross-contaminates attention
                # and spreads (state-accumulating corruption). Zero the
                # slot: the block degrades (empty context) but cannot
                # poison or be poisoned by foreign content.
                state_pools = [
                    impl._kvarn_latent_pool
                    for impl in state.impls
                    if getattr(impl, "_kvarn_latent_pool", None) is not None
                ]
                for pool in state_pools:
                    pool[slot].zero_()
                cls._audit.log(block_id, "orphan_zero", f"slot={slot}")
            cls._audit.log(block_id, "alloc", f"slot={slot} st={group_key[-1] if isinstance(group_key, tuple) else group_key}")

        # A missing block whose rows were persisted by a retire-flush is a
        # prefix-cache hit (or an equivalent re-entry): no step will ever
        # scatter its pre-computed rows again, but the freshly popped slot
        # still holds another block's exact rows. Restore this block's rows
        # from its packed paged record so exact-pool readers see its own KV.
        # Blocks without a packed copy are genuinely fresh: every row they
        # will ever expose is written by scatter in the acquiring step, which
        # overwrites whatever the recycled slot held.
        rehydrate_ids = [
            block_id for block_id in missing if block_id in state.flushed
        ]
        for _rb in rehydrate_ids:
            cls._audit.log(_rb, "rehydrate", f"st={group_key[-1] if isinstance(group_key, tuple) else group_key}")
        if _orphans:
            if os.environ.get("KVARN_MLA_ORPHAN_RAISE", "0") == "1":
                _hist = {b: cls._audit.history(b) for b in _orphans[:8]}
                raise RuntimeError(
                    "KVarN provenance-lost re-entry (fatal): blocks "
                    f"{_orphans[:8]} missing AND not flushed with "
                    f"fill>=group. st={group_key}. Audit history: {_hist}"
                )
            logger.warning(
                "KVARN-ORPHAN t=%.1f n=%d ids=%s",
                time.monotonic() - _BOOT_T0,
                len(_orphans),
                _orphans[:8],
            )
        _rh_dbg = os.environ.get("KVARN_MLA_DIAG_RESTORE_STATE", "")
        if _rh_dbg and len(missing) >= 4:
            _skipped = [b for b in missing if b not in state.flushed]
            logger.warning(
                "KVARN-RESTORE group=%s missing=%d rehydrate=%d SKIPPED=%d "
                "skipped_ids=%s fills_known=%d flushed_total=%d",
                group_key[1] if len(group_key) > 1 else group_key,
                len(missing),
                len(rehydrate_ids),
                len(_skipped),
                _skipped[:16],
                len(block_fills),
                len(state.flushed),
            )
        if rehydrate_ids:
            from vllm.v1.attention.ops.kvarn_mla import (
                rehydrate_kvarn_mla_blocks,
            )

            device = impls[0].device
            block_ids = torch.tensor(rehydrate_ids, dtype=torch.long, device=device)
            pool_slots = torch.tensor(
                [state.mapping[block_id] for block_id in rehydrate_ids],
                dtype=torch.long,
                device=device,
            )
            for impl in impls:
                cache_ref = getattr(impl, "_kvarn_cache_ref", None)
                latent_pool = getattr(impl, "_kvarn_latent_pool", None)
                rope_pool = getattr(impl, "_kvarn_rope_pool", None)
                if cache_ref is None or latent_pool is None or rope_pool is None:
                    continue
                rehydrate_kvarn_mla_blocks(
                    cache_ref,
                    latent_pool,
                    rope_pool,
                    block_ids,
                    pool_slots,
                    config,
                )

        for mirror in state.mirrors.values():
            cls._update_mirror(mirror, mirror_updates)


    @classmethod
    def maybe_log_exact_row_coverage(cls) -> None:
        """Log per-step KV-row sourcing (exact/packed/invalid) when enabled.

        KVARN_MLA_DIAG_EXACT_ROWS caps the number of STEPS logged (0 = off);
        every impl with diag buffers is logged (target and draft).
        """
        raw_limit = os.getenv("KVARN_MLA_DIAG_EXACT_ROWS", "0")
        try:
            limit = int(raw_limit)
        except ValueError:
            limit = 0
        if limit <= 0 or cls._diag_steps >= limit:
            return
        cls._diag_steps += 1
        for group_key, state in cls._groups.items():
            for impl in state.impls:
                selected = getattr(impl, "_kvarn_diag_selected_indices", None)
                if selected is None:
                    continue
                block_to_slot = getattr(impl, "_kvarn_block_to_slot", None)
                if block_to_slot is None:
                    continue
                rows = selected.shape[0]
                width = selected.shape[1]
                valid_counts = getattr(impl, "_kvarn_diag_valid_counts", None)
                counts = (
                    valid_counts[:rows].clamp(min=0, max=width)
                    if valid_counts is not None
                    else (selected >= 0).sum(dim=1)
                )
                active = (
                    torch.arange(width, device=selected.device)[None, :]
                    < counts[:, None]
                )
                physical_valid = (
                    (selected >= 0) & (selected < block_to_slot.numel() * 64)
                )
                safe_blocks = selected.clamp(0, block_to_slot.numel() * 64 - 1) // 64
                pool_slots = block_to_slot[safe_blocks]
                exact = (active & physical_valid & (pool_slots >= 0)).sum(dim=1)
                packed = (active & physical_valid & (pool_slots < 0)).sum(dim=1)
                invalid = (active & ~physical_valid).sum(dim=1)
                detail = (
                    torch.stack((counts, exact, packed, invalid), dim=1)[
                        : min(rows, 32)
                    ]
                    .cpu()
                    .tolist()
                )
                totals = (
                    torch.stack(
                        (counts.sum(), exact.sum(), packed.sum(), invalid.sum())
                    )
                    .cpu()
                    .tolist()
                )
                is_draft = getattr(impl, "_is_deepseek_mtp_draft", False)
                logger.warning(
                    "KVarN exact-row coverage step=%d group=%s impl=%s rows=%d "
                    "tot=[valid,exact,packed,invalid]=%s per_row[valid,exact,"
                    "packed,invalid]=%s mapped_blocks=%d selsum=%d poolsum=%d",
                    cls._diag_steps,
                    group_key,
                    "draft" if is_draft else "target",
                    rows,
                    totals,
                    detail,
                    len(state.mapping),
                    (
                        int(selected.to(torch.int64).sum().item())
                        if selected is not None
                        else -1
                    ),
                    (
                        int(
                            impl._kvarn_latent_pool.float().abs().sum().item()
                        )
                        if getattr(impl, "_kvarn_latent_pool", None) is not None
                        else -1
                    ),
                )
