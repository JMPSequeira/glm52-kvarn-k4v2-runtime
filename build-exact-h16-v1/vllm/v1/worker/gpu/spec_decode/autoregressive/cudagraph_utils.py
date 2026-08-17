# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class SpeculatorCudaGraphManager(CudaGraphManager):
    """CudaGraphManager for draft prefill and decode.

    Builds fresh dummy inputs and attention metadata for every warmup and
    capture pass so that the contents of the shared persistent buffers
    (e.g. query_start_loc, seq_lens, FA3 scheduler metadata) always match
    the batch descriptor being captured. Reusing metadata built during an
    earlier capture would execute kernels with stale buffer contents.
    """

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                skip_attn=(
                    desc.cg_mode == CUDAGraphMode.PIECEWISE
                    and not self.use_breakable_cg
                ),
            )

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)


@dataclass(frozen=True)
class FixedMTP3GraphKey:
    """Identity of a fixed-depth proposer graph.

    The cache-owner identities deliberately participate in the key: a graph
    records writes through the exact block-table and KV-cache objects supplied
    during capture and must never be replayed after those objects are replaced.
    """

    depth: int
    num_tokens: int
    num_reqs: int
    uniform_token_count: int
    block_tables_id: int
    kv_cache_config_id: int
    model_state_id: int
    input_buffers_id: int
    static_buffer_ids: tuple[int, ...]
    recurrent_owner_id: int


class FixedMTP3CudaGraphManager(SpeculatorCudaGraphManager):
    """FULL-graph manager for the recurrent portion of a fixed MTP3 proposal."""

    DEPTH = 3

    def __init__(
        self,
        *args,
        block_tables: BlockTables,
        kv_cache_config: KVCacheConfig,
        model_state: ModelState,
        input_buffers: InputBuffers,
        static_buffers: tuple[torch.Tensor, ...],
        recurrent_owner: object,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._fixed_mtp3_block_tables = block_tables
        self._fixed_mtp3_kv_cache_config = kv_cache_config
        self._fixed_mtp3_model_state = model_state
        self._fixed_mtp3_input_buffers = input_buffers
        self._fixed_mtp3_static_buffers = static_buffers
        self._fixed_mtp3_recurrent_owner = recurrent_owner
        self.fixed_mtp3_graphs: dict[
            FixedMTP3GraphKey, torch.cuda.CUDAGraph
        ] = {}

    def _fixed_mtp3_key(
        self,
        desc: BatchExecutionDescriptor,
        *,
        depth: int = DEPTH,
    ) -> FixedMTP3GraphKey:
        if desc.num_reqs is None or desc.uniform_token_count is None:
            raise RuntimeError(
                "Fixed MTP3 requires a statically shaped uniform decode descriptor."
            )
        return FixedMTP3GraphKey(
            depth=depth,
            num_tokens=desc.num_tokens,
            num_reqs=desc.num_reqs,
            uniform_token_count=desc.uniform_token_count,
            block_tables_id=id(self._fixed_mtp3_block_tables),
            kv_cache_config_id=id(self._fixed_mtp3_kv_cache_config),
            model_state_id=id(self._fixed_mtp3_model_state),
            input_buffers_id=id(self._fixed_mtp3_input_buffers),
            static_buffer_ids=tuple(map(id, self._fixed_mtp3_static_buffers)),
            recurrent_owner_id=id(self._fixed_mtp3_recurrent_owner),
        )

    def capture_fixed_mtp3(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        static_buffers: tuple[torch.Tensor, ...],
    ) -> None:
        if (
            model_state is not self._fixed_mtp3_model_state
            or input_buffers is not self._fixed_mtp3_input_buffers
            or block_tables is not self._fixed_mtp3_block_tables
            or kv_cache_config is not self._fixed_mtp3_kv_cache_config
            or getattr(forward_fn, "__self__", None)
            is not self._fixed_mtp3_recurrent_owner
            or len(static_buffers) != len(self._fixed_mtp3_static_buffers)
            or any(
                current is not captured
                for current, captured in zip(
                    static_buffers,
                    self._fixed_mtp3_static_buffers,
                    strict=True,
                )
            )
        ):
            raise RuntimeError(
                "Fixed MTP3 capture was given different recurrent, cache, or "
                "static-buffer ownership."
            )
        super().capture(
            forward_fn,
            model_state,
            input_buffers,
            block_tables,
            attn_groups,
            kv_cache_config,
            progress_bar_desc="Capturing fixed MTP3 proposer supergraphs",
        )
        self.fixed_mtp3_graphs = {
            self._fixed_mtp3_key(desc): graph
            for desc, graph in self.graphs.items()
            if desc.cg_mode == CUDAGraphMode.FULL
        }

    def dispatch_fixed_mtp3(
        self,
        *,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int,
    ) -> BatchExecutionDescriptor:
        desc = self.dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras=0,
        )
        if (
            desc.cg_mode != CUDAGraphMode.FULL
            or desc.num_reqs != num_reqs
            or desc.num_tokens != num_tokens
            or desc.uniform_token_count != uniform_token_count
        ):
            raise RuntimeError(
                "Fixed MTP3 has no exact FULL proposer graph for "
                f"num_reqs={num_reqs}, num_tokens={num_tokens}, "
                f"uniform_token_count={uniform_token_count}; eager or padded "
                "fallback is disabled."
            )
        return desc

    def run_fixed_mtp3(
        self,
        desc: BatchExecutionDescriptor,
        *,
        depth: int,
        block_tables: BlockTables,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        input_buffers: InputBuffers,
        static_buffers: tuple[torch.Tensor, ...],
        recurrent_owner: object,
    ) -> None:
        if (
            model_state is not self._fixed_mtp3_model_state
            or input_buffers is not self._fixed_mtp3_input_buffers
            or block_tables is not self._fixed_mtp3_block_tables
            or kv_cache_config is not self._fixed_mtp3_kv_cache_config
            or recurrent_owner is not self._fixed_mtp3_recurrent_owner
            or len(static_buffers) != len(self._fixed_mtp3_static_buffers)
            or any(
                current is not captured
                for current, captured in zip(
                    static_buffers,
                    self._fixed_mtp3_static_buffers,
                    strict=True,
                )
            )
        ):
            raise RuntimeError(
                "Fixed MTP3 graph cannot be replayed with different recurrent, "
                "cache, or static-buffer ownership."
            )
        key = self._fixed_mtp3_key(desc, depth=depth)
        graph = self.fixed_mtp3_graphs.get(key)
        if graph is None or graph is not self.graphs.get(desc):
            raise RuntimeError(f"Fixed MTP3 proposer graph key mismatch: {key}.")
        super().run_fullgraph(desc)

    def clear(self) -> None:
        self.fixed_mtp3_graphs.clear()
        super().clear()
