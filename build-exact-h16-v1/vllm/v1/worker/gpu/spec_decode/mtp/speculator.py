# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    FixedMTP3CudaGraphManager,
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.mtp.table_extension import MTPTableExtension

logger = init_logger(__name__)

_FIXED_MTP3_ENV = "VLLM_FIXED_MTP3_PROPOSER_SUPERGRAPH"


def fixed_mtp3_supergraph_enabled() -> bool:
    raw = os.getenv(_FIXED_MTP3_ENV, "0").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{_FIXED_MTP3_ENV} must be a boolean, got {os.getenv(_FIXED_MTP3_ENV)!r}."
    )


class MTPSpeculator(AutoRegressiveSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.fixed_mtp3_supergraph = fixed_mtp3_supergraph_enabled()
        # Core mutable input/output buffers recorded by the recurrent graph.
        # The base speculator allocates them once so their addresses remain
        # stable across every replay.
        self.fixed_mtp3_static_buffers = (
            self._fixed_mtp3_current_static_buffers()
        )
        if self.fixed_mtp3_supergraph:
            # Capture can precede the first real proposal. Use the conservative
            # model limit while constructing its static attention schedule;
            # runtime proposals still publish their tighter CPU upper bound.
            # Pinned: the supergraph builds draft attention metadata INSIDE
            # the CUDA graph capture region, and this CPU upper bound flows
            # into the attention metadata builders there — any host->device
            # staging of it must come from pinned memory to be capture-legal.
            self._draft_decode_seq_lens_upper_bound = torch.full(
                (self.max_num_reqs,),
                self.max_model_len,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
        # Table extension (default-off env knobs): extend the draft block
        # past the MTP head with n-gram table tokens. The engine is
        # configured with num_speculative_tokens = head + extension so the
        # whole downstream (scheduler padding, verify rows, CUDA graph
        # capture sizes) sees one static width; this speculator stops the
        # MTP head after its own steps and fills the remaining slots
        # host-side from the per-request table. Incompatible with the
        # FixedMTP3 proposer supergraph (fixed depth 3): fail closed.
        self.table_extension: MTPTableExtension | None = None
        slots = envs.VLLM_MTP_TABLE_EXTENSION
        if slots > 0:
            if self.fixed_mtp3_supergraph:
                logger.warning(
                    "MTP table extension is incompatible with the "
                    "FixedMTP3 proposer supergraph; extension disabled."
                )
            elif self.draft_logits is not None:
                # Extension slots have no draft logits; probabilistic
                # rejection sampling cannot verify them. Fail closed to
                # the unextended arm (all head steps, full width).
                logger.warning(
                    "MTP table extension requires greedy draft sampling "
                    "(draft_sample_method='greedy'); extension disabled."
                )
            else:
                self.table_extension = MTPTableExtension(
                    slots=slots,
                    k=envs.VLLM_MTP_TABLE_K,
                    log_path=envs.VLLM_MTP_TABLE_LOG,
                    device=device,
                    max_num_reqs=self.max_num_reqs,
                    num_speculative_steps=self.num_speculative_steps,
                )

    def _validate_fixed_mtp3_config(self, cudagraph_mode: CUDAGraphMode) -> None:
        if not self.fixed_mtp3_supergraph:
            return
        if self.num_speculative_steps != FixedMTP3CudaGraphManager.DEPTH:
            raise ValueError(
                "Fixed MTP3 proposer supergraph requires "
                "num_speculative_tokens=3."
            )
        if self.draft_logits is not None:
            raise ValueError(
                "Fixed MTP3 proposer supergraph requires deterministic greedy "
                "draft sampling; probabilistic sampling is unsupported."
            )
        if self.supports_mm_inputs:
            raise ValueError(
                "Fixed MTP3 proposer supergraph does not support multimodal inputs."
            )
        if (
            self.speculative_config.uses_batch_size_dynamic_speculative_decoding()
            or self.speculative_config.uses_acceptance_length_adaptation()
        ):
            raise ValueError(
                "Fixed MTP3 proposer supergraph does not support mixed or "
                "adaptive draft depth."
            )
        if self.dp_size != 1:
            raise ValueError(
                "Fixed MTP3 proposer supergraph currently requires data-parallel "
                "size 1."
            )
        if not cudagraph_mode.has_full_cudagraphs():
            raise ValueError(
                "Fixed MTP3 proposer supergraph requires FULL CUDA graphs."
            )

    def _fixed_mtp3_current_static_buffers(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.input_buffers.query_start_loc,
            self.input_buffers.seq_lens,
            self.hidden_states,
            self.draft_tokens,
            self.current_draft_step,
            self.idx_mapping,
            self.temperature,
            self.seeds,
            self.active_num_reqs,
        )


    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        self._validate_fixed_mtp3_config(cudagraph_mode)
        if not self.fixed_mtp3_supergraph:
            super().init_cudagraph_manager(cudagraph_mode)
            return

        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )
        self.decode_cudagraph_manager = FixedMTP3CudaGraphManager(
            self.vllm_config,
            self.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_len=1,
            block_tables=self.block_tables,
            kv_cache_config=self.kv_cache_config,
            model_state=self.model_state,
            input_buffers=self.input_buffers,
            static_buffers=self.fixed_mtp3_static_buffers,
            recurrent_owner=self,
        )

    def _validate_fixed_mtp3_runtime(
        self,
        *,
        num_speculative_tokens: int,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> None:
        if num_speculative_tokens != FixedMTP3CudaGraphManager.DEPTH:
            raise RuntimeError(
                "Fixed MTP3 proposer supergraph cannot replay a mixed draft depth."
            )
        if mm_inputs is not None:
            raise RuntimeError(
                "Fixed MTP3 proposer supergraph cannot replay multimodal inputs."
            )

    def _fixed_mtp3_decode(
        self,
        num_reqs: int,
        num_tokens: int,
        _attn_metadata: dict[str, Any] | None,
        _slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        _cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        if num_tokens != num_reqs:
            raise RuntimeError(
                "Fixed MTP3 capture requires one recurrent token per request."
            )
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform_token_count=1,
        )
        self._multi_step_decode(
            num_reqs,
            False,
            batch_desc,
            num_tokens_across_dp,
            FixedMTP3CudaGraphManager.DEPTH,
        )

    def _run_fixed_mtp3_supergraph(self, num_reqs: int) -> None:
        manager = self.decode_cudagraph_manager
        if not isinstance(manager, FixedMTP3CudaGraphManager):
            raise RuntimeError("Fixed MTP3 proposer graph manager is not initialized.")
        desc = manager.dispatch_fixed_mtp3(
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            uniform_token_count=1,
        )
        manager.run_fixed_mtp3(
            desc,
            depth=FixedMTP3CudaGraphManager.DEPTH,
            block_tables=self.block_tables,
            model_state=self.model_state,
            input_buffers=self.input_buffers,
            recurrent_owner=self,
            static_buffers=self._fixed_mtp3_current_static_buffers(),
            kv_cache_config=self.kv_cache_config,
        )

    @property
    def model_returns_tuple(self) -> bool:
        # DeepSeek MTP recycles the post-final-norm hidden state between
        # draft steps, so forward() returns (logit_hidden, recycle_hidden).
        return "DeepSeekMTPModel" in (
            self.draft_model_config.hf_config.architectures or []
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)


    def set_token_stream(self, all_token_ids, total_len_gpu) -> None:
        """Forward the runner's authoritative token stream (UVA-backed
        all_token_ids + GPU total_len) to the table extension. No-op
        unless the extension is enabled (the runner's generic
        set_token_stream hook calls this)."""
        if self.table_extension is not None:
            self.table_extension.set_token_stream(all_token_ids, total_len_gpu)

    def propose(
        self,
        input_batch,
        *args,
        num_speculative_tokens: int | None = None,
        **kwargs,
    ):
        table = self.table_extension
        if table is None:
            return super().propose(
                input_batch,
                *args,
                num_speculative_tokens=num_speculative_tokens,
                **kwargs,
            )
        width = (
            num_speculative_tokens
            if num_speculative_tokens is not None
            else self.num_speculative_steps
        )
        # The MTP head runs only its own steps; the table fills the rest.
        head = max(1, width - table.slots)
        # Stage the per-request stream-length readback BEFORE the draft
        # decode graphs replay (NgramAssist begin_step pattern).
        table.begin_step(input_batch)
        super().propose(
            input_batch,
            *args,
            num_speculative_tokens=head,
            **kwargs,
        )
        # Host-side extension of the persistent draft block to full width.
        # Same tensor/row contract as the unextended path: a
        # [num_reqs, width] view of the persistent draft-token buffer.
        table.extend(
            input_batch,
            self.draft_tokens,
            dummy_run=bool(kwargs.get("dummy_run", False)),
            is_profile=bool(kwargs.get("is_profile", False)),
        )
        return self.draft_tokens[: input_batch.num_reqs, :width]