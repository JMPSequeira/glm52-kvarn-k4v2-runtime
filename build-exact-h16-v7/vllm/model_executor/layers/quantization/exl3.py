# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EXL3 (ExLlamaV3 trellis) quantization support.

Rank-sliced routed-expert checkpoints use Sparkinfer's planned full-rotation
Trellis MoE API for the decode window and the ExLlamaV3 extension for larger
prefill batches. Generic dense and non-rank-sliced MoE checkpoints use the
bit-faithful ``exllamav3_ext.exl3_gemm`` parity path. Every logical checkpoint
matrix is dispatched independently: vLLM's packed QKV and gate/up modules are
not treated as one EXL3 matrix because each source matrix owns its Hadamard
vectors and codebook marker.

Both dependencies are imported lazily. Importing this module, parsing
checkpoint metadata, or compiling it with ``py_compile`` does not load either
one or initialize CUDA.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import os
import re
import sys
import weakref
from collections.abc import Iterable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from transformers import PretrainedConfig

from vllm.config import get_current_vllm_config_or_none
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    MoEActivation,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_MCG_SENTINEL = 0xCBAC1FED
_MUL1_SENTINEL = 0x83DCD12D
_HADAMARD_BLOCK = 128
_EXL3_EXT: Any | None = None
_SPARKINFER_TRELLIS_API: Any | None = None
_SPARKINFER_MIXED_TRELLIS_API: Any | None = None
_RANK_SLICED_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}
_EXACT_RANK_SLICED_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}
_MIXED_TRELLIS_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}
_MIXED_TRELLIS_BUFFERS: dict[tuple[Any, ...], Any] = {}
_NEXT_RUNTIME_SCOPE_ID = 0
_MIXED_TRELLIS_ROUTE_BLOCK_SIZE = int(
    os.getenv("VLLM_EXL3_TRELLIS_ROUTE_BLOCK_SIZE", "8")
)
_GLM52_MIXED_TRELLIS_PREFILL_BLOCK_SIZE = 32
_GLM52_MIXED_TRELLIS_BLOCK32_SIGNATURES = frozenset({((3, 192), (4, 64))})


class _SharedHRotationArena:
    __slots__ = ("data", "__weakref__")

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


_SHARED_H_ROTATION_ARENAS: weakref.WeakValueDictionary[
    tuple[Any, ...], _SharedHRotationArena
] = weakref.WeakValueDictionary()


# Smallest m the Trellis kernel path can service, and therefore the smallest
# row count an EXL3 rank-sliced MoE layer can be CUDA-graph captured at. A
# capture-size selector may read this to align its sizes with the backend
# instead of failing at capture time.
MIN_CAPTURABLE_TRELLIS_M = 1

# Historical default for non-captured (target) layers.
_DEFAULT_TRELLIS_MIN_M = 4


def _is_shared_expert_projection(prefix: str) -> bool:
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        is_shared_expert_projection,
    )

    return is_shared_expert_projection(prefix)


def _new_mxfp8_online_linear_method() -> QuantizeMethodBase:
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )

    return Mxfp8OnlineLinearMethod()


def _is_draft_layer(layer: Any) -> bool:
    """True for a rank-sliced MTP/EAGLE draft MoE layer.

    Derived from the layer prefix rather than the quant config, because config
    identity is not a reliable role signal: only some MTP model files build the
    draft a fresh ``Exl3Config``. Several (e.g. ``glm4_moe_mtp``, ``qwen3_next_mtp``,
    ``deepseek_eagle``) pass ``vllm_config.quant_config`` straight through, which
    makes the draft's scope identical to the target's and silently restores the
    shared-scratch corruption this scoping exists to prevent.
    """
    name = str(getattr(layer, "layer_name", "") or getattr(layer, "prefix", ""))
    return any(
        t in name for t in (".mtp", "mtp.", "nextn", "eagle", "draft", "speculator")
    )


def _runtime_owner_token(quant_config: Any, layer: Any) -> tuple[int, bool]:
    """Runtime-cache owner identity: (config scope, is_draft).

    Adding the role makes target/draft isolation independent of whether the model
    file happened to mint a separate quant config.
    """
    return (_runtime_scope_id(quant_config), _is_draft_layer(layer))


def _runtime_scope_id(quant_config: Any) -> int:
    """Stable identity for the model that owns a rank-sliced runtime.

    A cached runtime owns mutable Trellis/prefill scratch plus parity staging and
    sort buffers, so an entry must never be shared across models. A target MoE
    layer and a rank-sliced MTP draft layer have identical shapes, topk and
    planner settings -- both read ``max_num_batched_tokens`` from the same
    scheduler config -- so a shape-only key makes the draft reuse the target's
    scratch. That defeats the target/draft resource isolation their
    independently captured CUDA graphs rely on.

    Scoping by the owning quant config is deliberately coarser than per-layer:
    the draft is built with its own ``Exl3Config`` while every layer of one model
    shares a single config, so each model gets exactly one runtime. The prefill
    arena alone is ~1 GiB, so per-layer runtimes would cost tens of GiB per rank
    on a 75+ layer model and are not affordable.
    """
    global _NEXT_RUNTIME_SCOPE_ID
    scope = getattr(quant_config, "_exl3_runtime_scope_id", None)
    if scope is not None:
        return scope
    scope = _NEXT_RUNTIME_SCOPE_ID
    _NEXT_RUNTIME_SCOPE_ID += 1
    try:
        quant_config._exl3_runtime_scope_id = scope  # noqa: SLF001
    except AttributeError:
        # Frozen/slotted config: fall back to object identity. Configs live for
        # the process lifetime, so reuse-after-GC aliasing is not a concern here.
        return id(quant_config)
    return scope


_RANK_SLICED_FORMAT = "exl3-trellis"
_PROTECTED_POLICY_FORMAT = "exl3-protected-v1"
_PER_EXPERT_ROTATION_LAYOUT = "per_expert_v1"
_SHARED_H_ROTATION_LAYOUT = "shared_h_v1"
_SHARED_H_TENSOR_SCHEMA = (
    "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
)
_RANK_SLICED_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+)\.rank(?P<rank>\d+)\."
    r"(?P<field>trellis|suh|svh|mcg|mul1)$"
)
_SHARED_H_WEIGHT_RE = re.compile(
    r"^(?P<experts_prefix>.+\.experts)\.shared_h\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.rank(?P<rank>\d+)\."
    r"(?P<field>suh|svh)$"
)
_EXPERT_PROJECTION_RE = re.compile(
    r"^.+\.experts\.\d+\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
_RANK_SLICED_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)(?:\.|$)")

ShardId = str | int | tuple[int, ...] | None


def _load_exl3_ext() -> Any:
    """Load the existing ExLlamaV3 extension only from an actual CUDA call."""

    global _EXL3_EXT
    if _EXL3_EXT is not None:
        return _EXL3_EXT

    shim = os.environ.get("VLLM_EXL3_ABI_SHIM")
    if shim:
        ctypes.CDLL(shim, mode=ctypes.RTLD_GLOBAL)

    ext_path = os.environ.get("VLLM_EXL3_EXT_PATH")
    if ext_path:
        search_dir = ext_path if os.path.isdir(ext_path) else os.path.dirname(ext_path)
        if search_dir and search_dir not in sys.path:
            sys.path.insert(0, search_dir)

    try:
        ext = importlib.import_module("exllamav3_ext")
    except Exception as exc:
        hint = (
            "Set VLLM_EXL3_EXT_PATH to the directory containing "
            "exllamav3_ext*.so (and VLLM_EXL3_ABI_SHIM when the local "
            "PyTorch ABI shim is required)."
        )
        raise RuntimeError(f"Unable to import exllamav3_ext. {hint}") from exc

    if not hasattr(ext, "exl3_gemm"):
        raise RuntimeError(
            "The imported exllamav3_ext does not export exl3_gemm; rebuild the "
            "track_a_retile extension used by this overlay."
        )
    _EXL3_EXT = ext
    return ext


def _load_sparkinfer_trellis() -> Any:
    """Resolve the compatible planned Trellis MoE API lazily."""

    global _SPARKINFER_TRELLIS_API
    if _SPARKINFER_TRELLIS_API is not None:
        return _SPARKINFER_TRELLIS_API
    try:
        from sparkinfer.moe import trellis_moe
    except Exception as exc:
        raise RuntimeError(
            "Rank-sliced EXL3 requires sparkinfer.moe.trellis_moe. "
            "Install the runtime overlay built for this vLLM image."
        ) from exc
    _SPARKINFER_TRELLIS_API = trellis_moe
    return trellis_moe


def _load_sparkinfer_mixed_trellis() -> Any:
    """Resolve the pinned one-grid mixed-bitrate Trellis API lazily."""

    global _SPARKINFER_MIXED_TRELLIS_API
    if _SPARKINFER_MIXED_TRELLIS_API is not None:
        return _SPARKINFER_MIXED_TRELLIS_API
    try:
        module = importlib.import_module(
            "sparkinfer.moe._shared.kernels.w4a16.mixed_trellis"
        )
        prepare = importlib.import_module(
            "sparkinfer.moe._shared.kernels.w4a16.prepare"
        )
        host = importlib.import_module("sparkinfer.moe._shared.kernels.w4a16.host")
    except Exception as exc:
        raise RuntimeError(
            "Mixed-bitrate rank-sliced EXL3 requires the pinned SparkInfer "
            "mixed_trellis implementation."
        ) from exc
    api = SimpleNamespace(
        build_tiered_maps=module.build_tiered_maps,
        compile_mixed_trellis=module.compile_mixed_trellis,
        make_mixed_trellis_buffers=module.make_mixed_trellis_buffers,
        max_packed_route_slots=host.max_packed_route_slots,
        prepare_weights=prepare.prepare_trellis256_moe_weights,
        run_mixed_trellis=module.run_mixed_trellis,
        run_mixed_trellis_monolithic=module.run_mixed_trellis_monolithic,
        warmup_mixed_trellis_route_pack=module.warmup_mixed_trellis_route_pack,
    )
    _SPARKINFER_MIXED_TRELLIS_API = api
    return api


def _resolve_mixed_trellis_prefill_block_m(
    *,
    configured_block_m: int,
    explicit_override: bool,
    hidden_size: int,
    intermediate_size: int,
    tier_signature: tuple[tuple[int, int], ...],
    topk: int,
    device_major: int,
    prefill_tile_config: tuple[int, int, int, int],
) -> int:
    qualified = (
        not explicit_override
        and device_major == 12
        and hidden_size == 6144
        and intermediate_size == 512
        and tier_signature in _GLM52_MIXED_TRELLIS_BLOCK32_SIGNATURES
        and topk == 8
        and prefill_tile_config == (128, 128, 32, 512)
    )
    if qualified:
        return _GLM52_MIXED_TRELLIS_PREFILL_BLOCK_SIZE
    return configured_block_m


def _unique_tensor_storage_bytes(buffers: Any) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for value in vars(buffers).values():
        if not isinstance(value, torch.Tensor):
            continue
        storage = value.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total

def _shared_mixed_buffers(
    owner_token: tuple[int, bool],
    mixed_api: Any,
    launch: Any,
    device: torch.device,
    sms: int,
) -> Any:
    total_experts = sum(
        int(getattr(launch, f"tier{tier}_num_experts", 0)) for tier in range(4)
    )
    key = (
        owner_token,
        device.type,
        device.index,
        int(launch.size_m),
        int(launch.top_k),
        int(launch.hidden_size),
        int(launch.intermediate_size),
        int(launch.moe_block_size),
        total_experts,
        int(launch.blocks_per_sm),
        int(sms),
    )
    buffers = _MIXED_TRELLIS_BUFFERS.get(key)
    if buffers is None:
        buffers = mixed_api.make_mixed_trellis_buffers(
            launch,
            device=device,
            sms=sms,
        )
        _MIXED_TRELLIS_BUFFERS[key] = buffers
    return buffers


def _resolve_prefill_capacity(max_batched_tokens: int) -> int:
    capacity = _positive_env_int("VLLM_EXL3_PREFILL_CAPACITY", max_batched_tokens)
    if capacity > max_batched_tokens:
        raise ValueError(
            "VLLM_EXL3_PREFILL_CAPACITY cannot exceed "
            f"max_num_batched_tokens: {capacity} > {max_batched_tokens}"
        )
    return capacity


def _resolve_mixed_trellis_prefill_tile_config(
    decode_tile_config: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    raw = os.environ.get("VLLM_EXL3_PREFILL_TILE_CONFIG", "").strip()
    if not raw:
        return decode_tile_config
    try:
        config = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError(
            "VLLM_EXL3_PREFILL_TILE_CONFIG must contain four comma-separated "
            f"integers, got {raw!r}"
        ) from exc
    if len(config) != 4 or any(value <= 0 or value % 16 for value in config):
        raise ValueError(
            "VLLM_EXL3_PREFILL_TILE_CONFIG must be four positive multiples of "
            f"16, got {raw!r}"
        )
    fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n = config
    fc1_threads = fc1_tile_k * fc1_tile_n // 64
    fc2_threads = fc2_tile_k * fc2_tile_n // 64
    if fc1_threads != fc2_threads:
        raise ValueError(
            "VLLM_EXL3_PREFILL_TILE_CONFIG FC1/FC2 thread counts must match, "
            f"got {fc1_threads} and {fc2_threads}"
        )
    return config


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_exl3_mcg8_mla_common(
    source: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    out: torch.Tensor,
    head0_parity: int,
    *,
    source_width: int,
    output_shape: tuple[int, ...],
    op_name: str,
) -> None:
    if source.ndim != 3 or tuple(source.shape[:1]) != (16,):
        raise ValueError(
            f"{op_name} source must have shape [16, M, {source_width}], "
            f"got {tuple(source.shape)}."
        )
    if source.shape[2] != source_width:
        raise ValueError(
            f"{op_name} source must have shape [16, M, {source_width}], "
            f"got {tuple(source.shape)}."
        )
    rows = source.shape[1]
    if not 1 <= rows <= 4096:
        raise ValueError(f"{op_name} requires 1 <= M <= 4096, got {rows}.")
    expected_output_shape = tuple(
        rows if dimension == -1 else dimension for dimension in output_shape
    )
    if tuple(out.shape) != expected_output_shape:
        raise ValueError(
            f"{op_name} output must have shape {expected_output_shape}, "
            f"got {tuple(out.shape)}."
        )
    expected_packed = (
        (trellis, (32, 448, 128), torch.int16, "trellis"),
        (suh, (512,), torch.float16, "suh"),
        (svh, (7168,), torch.float16, "svh"),
    )
    for tensor, shape, dtype, name in expected_packed:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{op_name} {name} must be {dtype} with shape {shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{op_name} {name} must be contiguous.")
    if source.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError(f"{op_name} requires BF16 source and output tensors.")
    if not source.is_contiguous() or not out.is_contiguous():
        raise ValueError(f"{op_name} requires contiguous source and output tensors.")
    if head0_parity != 0:
        raise ValueError(
            f"{op_name} fixed local pack requires head0_parity=0, got {head0_parity}."
        )
    tensors = (source, trellis, suh, svh, out)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError(f"{op_name} requires CUDA tensors.")
    if any(tensor.device != source.device for tensor in tensors[1:]):
        raise ValueError(f"{op_name} tensors must be on one CUDA device.")


def _exl3_mcg8_mla_query_adj_impl(
    q: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    out: torch.Tensor,
    head0_parity: int,
) -> None:
    _load_exl3_ext().exl3_mcg8_mla_query_adj(q, trellis, suh, svh, out, head0_parity)


def _exl3_mcg8_mla_value_impl(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    out: torch.Tensor,
    head0_parity: int,
) -> None:
    _load_exl3_ext().exl3_mcg8_mla_value(x, trellis, suh, svh, out, head0_parity)


def _exl3_mcg8_mla_query_adj_fake(
    q: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    out: torch.Tensor,
    head0_parity: int,
) -> None:
    del q, trellis, suh, svh, out, head0_parity


def _exl3_mcg8_mla_value_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    out: torch.Tensor,
    head0_parity: int,
) -> None:
    del x, trellis, suh, svh, out, head0_parity


direct_register_custom_op(
    op_name="exl3_mcg8_mla_query_adj",
    op_func=_exl3_mcg8_mla_query_adj_impl,
    mutates_args=["out"],
    fake_impl=_exl3_mcg8_mla_query_adj_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
direct_register_custom_op(
    op_name="exl3_mcg8_mla_value",
    op_func=_exl3_mcg8_mla_value_impl,
    mutates_args=["out"],
    fake_impl=_exl3_mcg8_mla_value_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def exl3_mcg8_mla_query_adj(
    q: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    out: torch.Tensor,
    head0_parity: int,
) -> None:
    """Run the fixed packed-MCG8 MLA query adjoint into caller-owned storage."""
    _validate_exl3_mcg8_mla_common(
        q,
        trellis,
        suh,
        svh,
        out,
        head0_parity,
        source_width=192,
        output_shape=(16, -1, 512),
        op_name="exl3_mcg8_mla_query_adj",
    )
    torch.ops.vllm.exl3_mcg8_mla_query_adj(q, trellis, suh, svh, out, head0_parity)


def exl3_mcg8_mla_value(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    out: torch.Tensor,
    head0_parity: int,
) -> None:
    """Run the fixed packed-MCG8 MLA value projection into caller-owned storage."""
    _validate_exl3_mcg8_mla_common(
        x,
        trellis,
        suh,
        svh,
        out,
        head0_parity,
        source_width=512,
        output_shape=(-1, 16, 256),
        op_name="exl3_mcg8_mla_value",
    )
    torch.ops.vllm.exl3_mcg8_mla_value(x, trellis, suh, svh, out, head0_parity)


@torch.library.custom_op(
    "vllm::exl3_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _exl3_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    """Opaque torch op around the bit-faithful ExLlamaV3 dense call."""

    ext = _load_exl3_ext()
    output = torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )
    x_had = torch.empty_like(x)
    ext.exl3_gemm(
        x,
        trellis,
        output,
        suh,
        x_had,
        svh,
        -1,
        mcg,
        mul1,
        0,
    )
    return output


@_exl3_gemm.register_fake
def _exl3_gemm_fake(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: bool,
    mul1: bool,
) -> torch.Tensor:
    del suh, svh, mcg, mul1
    return torch.empty(
        (x.shape[0], trellis.shape[1] * 16),
        dtype=torch.float16,
        device=x.device,
    )


@torch.library.custom_op(
    "vllm::exl3_moe_fused",
    mutates_args=(
        "expert_count",
        "expert_offsets",
        "token_sorted",
        "weight_sorted",
        "tg",
        "tu",
        "ig",
        "iu",
    ),
)
def _exl3_moe_fused(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    expert_count: torch.Tensor,
    expert_offsets: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    tg: torch.Tensor,
    tu: torch.Tensor,
    ig: torch.Tensor,
    iu: torch.Tensor,
    gate_trellis_ptrs: torch.Tensor,
    gate_suh_ptrs: torch.Tensor,
    gate_svh_ptrs: torch.Tensor,
    up_trellis_ptrs: torch.Tensor,
    up_suh_ptrs: torch.Tensor,
    up_svh_ptrs: torch.Tensor,
    down_trellis_ptrs: torch.Tensor,
    down_suh_ptrs: torch.Tensor,
    down_svh_ptrs: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Opaque exact-MCG8 MoE op with explicit scratch mutation."""

    hidden = x.to(torch.float16)
    output = torch.zeros(
        (x.shape[0], x.shape[1]),
        dtype=torch.float32,
        device=x.device,
    )
    _load_exl3_ext().exl3_moe_fused(
        hidden,
        output,
        topk_ids,
        topk_weights,
        expert_map,
        expert_count,
        expert_offsets,
        token_sorted,
        weight_sorted,
        tg,
        tu,
        ig,
        iu,
        0,
        bits,
        bits,
        bits,
        gate_trellis_ptrs,
        gate_suh_ptrs,
        gate_svh_ptrs,
        up_trellis_ptrs,
        up_suh_ptrs,
        up_svh_ptrs,
        down_trellis_ptrs,
        down_suh_ptrs,
        down_svh_ptrs,
        True,
        False,
        True,
        False,
        True,
        False,
        0.0,
        0,
    )
    return output.to(x.dtype)


@_exl3_moe_fused.register_fake
def _exl3_moe_fused_fake(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    expert_count: torch.Tensor,
    expert_offsets: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    tg: torch.Tensor,
    tu: torch.Tensor,
    ig: torch.Tensor,
    iu: torch.Tensor,
    gate_trellis_ptrs: torch.Tensor,
    gate_suh_ptrs: torch.Tensor,
    gate_svh_ptrs: torch.Tensor,
    up_trellis_ptrs: torch.Tensor,
    up_suh_ptrs: torch.Tensor,
    up_svh_ptrs: torch.Tensor,
    down_trellis_ptrs: torch.Tensor,
    down_suh_ptrs: torch.Tensor,
    down_svh_ptrs: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    del (
        topk_ids,
        topk_weights,
        expert_map,
        expert_count,
        expert_offsets,
        token_sorted,
        weight_sorted,
        tg,
        tu,
        ig,
        iu,
        gate_trellis_ptrs,
        gate_suh_ptrs,
        gate_svh_ptrs,
        up_trellis_ptrs,
        up_suh_ptrs,
        up_svh_ptrs,
        down_trellis_ptrs,
        down_suh_ptrs,
        down_svh_ptrs,
        bits,
    )
    return torch.empty_like(x)


class Exl3Config(QuantizationConfig):
    """Configuration for modern and legacy EXL3 trellis checkpoints."""

    def __init__(
        self,
        bits: float | None = None,
        head_bits: float | None = None,
        codebook: str | None = None,
        version: str | None = None,
        tensor_storage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.codebook = codebook
        self.version = version
        self.tensor_storage = tensor_storage or {}
        self._eager_checked = False
        self.rank_sliced_metadata: dict[str, Any] | None = None
        self.rank_sliced_rotation_layout = _PER_EXPERT_ROTATION_LAYOUT
        self.trellis_bits_by_layer: dict[int, int] | None = None
        self.trellis_bits_by_expert: dict[int, tuple[int, ...]] | None = None
        self.rank_sliced_k_values: tuple[int, ...] | None = None
        self.protected_bits_by_prefix: dict[str, int] | None = None

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The kernel boundary is always fp16.  BF16 model activations are cast
        # in apply() and converted back after the fp16 bias addition.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Exl3Config:
        return cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            codebook=config.get("codebook"),
            version=config.get("version"),
            tensor_storage=config.get("tensor_storage"),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: PretrainedConfig | None = None,
    ) -> str | None:
        del hf_quant_cfg
        if user_quant is not None and user_quant != "exl3":
            return None
        metadata = getattr(hf_config, "hybrid_tr3_tail", None)
        if isinstance(metadata, dict) and metadata.get("format") == _RANK_SLICED_FORMAT:
            return "exl3"
        return None

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        rank_sliced = getattr(hf_config, "hybrid_tr3_tail", None)
        if (
            isinstance(rank_sliced, dict)
            and "protected_tensor_policy" in rank_sliced
            and rank_sliced.get("format") != _RANK_SLICED_FORMAT
        ):
            raise ValueError(
                "protected_tensor_policy requires hybrid_tr3_tail format "
                f"{_RANK_SLICED_FORMAT!r}"
            )
        if (
            isinstance(rank_sliced, dict)
            and rank_sliced.get("format") == _RANK_SLICED_FORMAT
        ):
            self._configure_rank_sliced(rank_sliced)
            if self.rank_sliced_k_values is not None:
                self._load_rank_sliced_bitrates(
                    model_name,
                    revision=revision or getattr(hf_config, "_commit_hash", None),
                )
            if "protected_tensor_policy" not in rank_sliced:
                return

            protected_bits = self._parse_protected_tensor_policy(
                rank_sliced["protected_tensor_policy"]
            )
            self._hydrate_tensor_storage(model_name, hf_config, revision)
            self._validate_storage_metadata()
            self._validate_protected_tensor_storage(protected_bits)
            self.protected_bits_by_prefix = dict(sorted(protected_bits.items()))
            counts = {
                bits: sum(value == bits for value in protected_bits.values())
                for bits in (6, 8)
            }
            logger.info_once(
                "EXL3 protected tensor policy validated: %d tensors "
                "(MCG6=%d, MCG8=%d).",
                len(protected_bits),
                counts[6],
                counts[8],
            )
            self._force_independent_lm_head(hf_config)
            return

        self._hydrate_tensor_storage(model_name, hf_config, revision)
        self._validate_storage_metadata()
        self._force_independent_lm_head(hf_config)

    def _hydrate_tensor_storage(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None,
        revision: str | None,
    ) -> None:
        # vLLM returns the summary embedded in config.json without consulting
        # get_config_filenames(). Hydrate the per-module records explicitly.
        if self.tensor_storage:
            if not isinstance(self.tensor_storage, dict):
                raise ValueError("EXL3 tensor_storage must be an object")
            return

        resolved_revision = revision
        if resolved_revision is None and hf_config is not None:
            resolved_revision = getattr(hf_config, "_commit_hash", None)
        config = get_hf_file_to_dict(
            "quantization_config.json",
            model_name,
            revision=resolved_revision,
        )
        storage = config.get("tensor_storage") if isinstance(config, dict) else None
        if not isinstance(storage, dict) or not storage:
            raise ValueError(
                "EXL3 requires quantization_config.json with a non-empty "
                "tensor_storage map. For branch-indexed Hugging Face repos, "
                "download/serve an actual bpw revision rather than main."
            )
        if self.rank_sliced_metadata is None:
            self.bits = config.get("bits", self.bits)
            self.head_bits = config.get("head_bits", self.head_bits)
            self.codebook = config.get("codebook", self.codebook)
            self.version = config.get("version", self.version)
        self.tensor_storage = storage

    @staticmethod
    def _parse_protected_tensor_policy(policy: Any) -> dict[str, int]:
        if not isinstance(policy, dict):
            raise ValueError("protected_tensor_policy must be an object")

        expected_fields = {"format", "tensors"}
        actual_fields = set(policy)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(repr(field) for field in actual_fields - expected_fields)
            raise ValueError(
                "protected_tensor_policy fields mismatch: "
                f"missing={missing}, extra={extra}"
            )
        if policy["format"] != _PROTECTED_POLICY_FORMAT:
            raise ValueError(
                f"unsupported protected_tensor_policy format: {policy['format']!r}"
            )

        tensors = policy["tensors"]
        if not isinstance(tensors, dict) or not tensors:
            raise ValueError(
                "protected_tensor_policy tensors must be a non-empty object"
            )

        protected_bits: dict[str, int] = {}
        expected_declaration_fields = {"bits", "codebook"}
        for prefix, declaration in tensors.items():
            if not isinstance(prefix, str) or not prefix:
                raise ValueError(
                    "protected_tensor_policy tensor prefixes must be non-empty strings"
                )
            if not isinstance(declaration, dict):
                raise ValueError(
                    f"protected_tensor_policy entry {prefix!r} must be an object"
                )
            declaration_fields = set(declaration)
            if declaration_fields != expected_declaration_fields:
                missing = sorted(expected_declaration_fields - declaration_fields)
                extra = sorted(
                    repr(field)
                    for field in declaration_fields - expected_declaration_fields
                )
                raise ValueError(
                    f"protected_tensor_policy entry {prefix!r} fields mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            if declaration["codebook"] != "mcg":
                raise ValueError(
                    f"protected_tensor_policy entry {prefix!r} must use "
                    f"codebook 'mcg', got {declaration['codebook']!r}"
                )
            bits = declaration["bits"]
            if (
                not isinstance(bits, int)
                or isinstance(bits, bool)
                or bits not in (6, 8)
            ):
                raise ValueError(
                    f"protected_tensor_policy entry {prefix!r} bits must be 6 or 8, "
                    f"got {bits!r}"
                )
            protected_bits[prefix] = bits
        return protected_bits

    def _validate_protected_tensor_storage(
        self, protected_bits: dict[str, int]
    ) -> None:
        actual_prefixes = {
            prefix
            for prefix, entry in self.tensor_storage.items()
            if entry.get("quant_format") == "exl3"
        }
        expected_prefixes = set(protected_bits)
        if actual_prefixes != expected_prefixes:
            missing = sorted(expected_prefixes - actual_prefixes)
            extra = sorted(actual_prefixes - expected_prefixes)
            raise ValueError(
                "protected EXL3 tensor_storage coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )

        for prefix, bits in protected_bits.items():
            entry = self.tensor_storage[prefix]
            stored = entry["stored_tensors"]
            mcg_name = f"{prefix}.mcg"
            mul1_name = f"{prefix}.mul1"
            if mcg_name not in stored:
                raise ValueError(
                    f"protected EXL3 tensor {prefix!r} is missing its MCG marker"
                )
            if mul1_name in stored:
                raise ValueError(
                    f"protected EXL3 tensor {prefix!r} has a forbidden MUL1 marker"
                )

            stored_bits = entry.get("bits_per_weight")
            if (
                not isinstance(stored_bits, int)
                or isinstance(stored_bits, bool)
                or stored_bits != bits
            ):
                raise ValueError(
                    f"protected EXL3 tensor {prefix!r} bits_per_weight "
                    f"{stored_bits!r} does not match declared bits {bits}"
                )

            trellis = stored.get(f"{prefix}.trellis")
            shape = trellis.get("shape") if isinstance(trellis, dict) else None
            if (
                not isinstance(shape, list)
                or not shape
                or not isinstance(shape[-1], int)
                or isinstance(shape[-1], bool)
                or shape[-1] != bits * 16
            ):
                raise ValueError(
                    f"protected EXL3 tensor {prefix!r} Trellis shape {shape!r} "
                    f"does not encode {bits} bits per weight"
                )

    @staticmethod
    def _normalize_rank_sliced_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        if "trellis_bits_by_layer" in metadata:
            return metadata
        layers = metadata.get("moe_layers")
        bits = metadata.get("bits")
        experts = metadata.get("experts_per_layer")
        if (
            not isinstance(layers, list)
            or len(layers) != 2
            or not all(
                isinstance(layer, int) and not isinstance(layer, bool)
                for layer in layers
            )
            or not isinstance(bits, (int, float))
            or isinstance(bits, bool)
            or int(bits) != bits
            or metadata.get("tr3_tail_per_layer") != experts
            or metadata.get("nvfp4_keep_per_layer") != 0
        ):
            return metadata
        normalized = dict(metadata)
        normalized["trellis_bits_by_layer"] = {
            str(layer): int(bits) for layer in range(layers[0], layers[1] + 1)
        }
        logger.info_once(
            "Normalized legacy uniform rank-sliced EXL3 metadata for layers %d-%d.",
            layers[0],
            layers[1],
        )
        return normalized

    def _configure_rank_sliced(self, metadata: dict[str, Any]) -> None:
        metadata = self._normalize_rank_sliced_metadata(metadata)
        required = {
            "codebook",
            "experts_per_layer",
            "moe_layers",
            "tensor_schema",
            "tp",
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(
                "rank-sliced EXL3 metadata is missing: " + ", ".join(missing)
            )
        if metadata["codebook"] != "mcg":
            raise ValueError(
                "rank-sliced EXL3 currently requires the MCG codebook, got "
                f"{metadata['codebook']!r}"
            )
        layers = metadata["moe_layers"]
        if (
            not isinstance(layers, list)
            or len(layers) != 2
            or not all(
                isinstance(layer, int) and not isinstance(layer, bool)
                for layer in layers
            )
            or layers[0] < 0
            or layers[1] < layers[0]
        ):
            raise ValueError("rank-sliced EXL3 moe_layers must be [first, last]")
        expected_schema = (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        )
        if metadata["tensor_schema"] != expected_schema:
            raise ValueError(
                "unsupported rank-sliced EXL3 tensor schema: "
                f"{metadata['tensor_schema']!r}"
            )
        rotation_layout = str(
            metadata.get("rotation_layout", _PER_EXPERT_ROTATION_LAYOUT)
        )
        if rotation_layout not in {
            _PER_EXPERT_ROTATION_LAYOUT,
            _SHARED_H_ROTATION_LAYOUT,
        }:
            raise ValueError(
                f"unsupported rank-sliced EXL3 rotation_layout: {rotation_layout!r}"
            )
        shared_schema = metadata.get("shared_h_tensor_schema")
        if rotation_layout == _SHARED_H_ROTATION_LAYOUT:
            if shared_schema != _SHARED_H_TENSOR_SCHEMA:
                raise ValueError(
                    "shared_h_v1 rank-sliced EXL3 requires "
                    f"shared_h_tensor_schema={_SHARED_H_TENSOR_SCHEMA!r}"
                )
        elif shared_schema is not None:
            raise ValueError(
                "shared_h_tensor_schema is only valid with "
                "rotation_layout='shared_h_v1'"
            )
        for field in ("experts_per_layer", "tp"):
            value = metadata[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"rank-sliced EXL3 {field} must be a positive integer")

        mixed = metadata.get("bits") == "mixed"
        if mixed:
            if "trellis_bits_by_layer" in metadata:
                raise ValueError(
                    "mixed rank-sliced EXL3 cannot declare trellis_bits_by_layer"
                )
            k_values = tuple(
                sorted({int(value) for value in metadata.get("k_values", ())})
            )
            if len(k_values) != 2 or any(
                value not in (3, 4, 5, 6) for value in k_values
            ):
                raise ValueError(
                    "mixed rank-sliced EXL3 currently requires exactly two "
                    f"k_values within 3..6, got {metadata.get('k_values')!r}"
                )
            if not isinstance(metadata.get("bits_per_expert"), str):
                raise ValueError(
                    "mixed rank-sliced EXL3 requires a bits_per_expert JSON reference"
                )
            bits_by_layer = None
            self.rank_sliced_k_values = k_values
        else:
            raw_bits = metadata.get("trellis_bits_by_layer")
            if not isinstance(raw_bits, dict):
                raise ValueError(
                    "rank-sliced EXL3 trellis_bits_by_layer must be an object"
                )
            invalid_keys = [
                repr(key)
                for key in raw_bits
                if not isinstance(key, str) or not key.isdecimal()
            ]
            if invalid_keys:
                raise ValueError(
                    "rank-sliced EXL3 trellis_bits_by_layer keys must be "
                    f"decimal layer strings, got {invalid_keys[:8]}"
                )
            expected_keys = {str(layer) for layer in range(layers[0], layers[1] + 1)}
            actual_keys = set(raw_bits)
            if actual_keys != expected_keys:
                missing_layers = sorted(expected_keys - actual_keys, key=int)
                extra_layers = sorted(actual_keys - expected_keys)
                raise ValueError(
                    "rank-sliced EXL3 trellis_bits_by_layer coverage mismatch: "
                    f"missing={missing_layers[:8]}, extra={extra_layers[:8]}"
                )
            allowed_bits = {3, 4, 5, 6, 8}
            bits_by_layer = {}
            for layer_key, bits in raw_bits.items():
                if (
                    not isinstance(bits, int)
                    or isinstance(bits, bool)
                    or bits not in allowed_bits
                ):
                    raise ValueError(
                        "rank-sliced EXL3 layer bitrates must be integral "
                        "MCG3/4/5/6/8 values, got layer "
                        f"{layer_key}={bits!r}"
                    )
                bits_by_layer[int(layer_key)] = bits
            self.rank_sliced_k_values = None
            self.trellis_bits_by_expert = None

        self.rank_sliced_metadata = dict(metadata)
        self.rank_sliced_rotation_layout = rotation_layout
        self.trellis_bits_by_layer = bits_by_layer
        self.protected_bits_by_prefix = None
        self.bits = None
        self.codebook = str(metadata["codebook"])
        self.version = str(metadata.get("exllamav3_version", "rank-sliced-mixed"))

    def _load_rank_sliced_bitrates(
        self, model_name: str, *, revision: str | None
    ) -> None:
        assert self.rank_sliced_metadata is not None
        reference = str(self.rank_sliced_metadata["bits_per_expert"])
        try:
            filename, field = reference.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(
                "rank-sliced EXL3 bits_per_expert must use "
                f"'file.json:field' syntax, got {reference!r}"
            ) from exc
        payload = get_hf_file_to_dict(filename, model_name, revision=revision)
        if not isinstance(payload, dict):
            raise ValueError(f"rank-sliced EXL3 could not load {filename!r}")
        experts = int(self.rank_sliced_metadata["experts_per_layer"])
        first, last = (int(value) for value in self.rank_sliced_metadata["moe_layers"])
        allowed = set(self.rank_sliced_k_values or ())
        by_layer: dict[int, tuple[int, ...]] = {}
        for layer in range(first, last + 1):
            entry = payload.get(str(layer))
            raw = entry.get(field) if isinstance(entry, dict) else None
            if not isinstance(raw, list) or len(raw) != experts:
                raise ValueError(
                    "rank-sliced EXL3 bitrate map must contain one entry per "
                    f"expert: layer={layer}, field={field!r}, expected={experts}"
                )
            bitrates = tuple(int(value) for value in raw)
            unexpected = sorted(set(bitrates).difference(allowed))
            if unexpected:
                raise ValueError(
                    f"rank-sliced EXL3 layer {layer} uses undeclared bitrates "
                    f"{unexpected}; declared={sorted(allowed)}"
                )
            if set(bitrates) != allowed:
                raise ValueError(
                    f"rank-sliced EXL3 layer {layer} must use every declared "
                    f"bitrate {sorted(allowed)}"
                )
            by_layer[layer] = bitrates
        self.trellis_bits_by_expert = by_layer

    def trellis_bitrates_for_layer(self, prefix: str) -> tuple[int, ...]:
        match = _RANK_SLICED_LAYER_RE.search(prefix)
        if match is None:
            raise ValueError(f"Cannot resolve EXL3 layer index from {prefix!r}")
        layer = int(match.group("layer"))
        if self.trellis_bits_by_expert is not None:
            try:
                return self.trellis_bits_by_expert[layer]
            except KeyError as exc:
                raise ValueError(
                    f"EXL3 layer {layer} is outside the declared precision map"
                ) from exc
        if self.trellis_bits_by_layer is None:
            raise ValueError("EXL3 checkpoint has no Trellis precision map")
        bits = self.trellis_bits_by_layer.get(layer)
        if bits is None:
            raise ValueError(
                f"EXL3 layer {layer} is outside the declared precision map"
            )
        experts = int(self.rank_sliced_metadata["experts_per_layer"])
        return (bits,) * experts

    def trellis_bits_for_layer(self, prefix: str) -> int:
        bitrates = self.trellis_bitrates_for_layer(prefix)
        if len(set(bitrates)) != 1:
            raise ValueError(f"EXL3 layer {prefix!r} has mixed per-expert precision")
        return bitrates[0]

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        # Keep both spellings: loader prefixes use vLLM names, while packed
        # source-matrix discovery intentionally refers to the unstacked HF name.
        mapped = hf_to_vllm_mapper.apply_dict(self.tensor_storage)
        self.tensor_storage = {**self.tensor_storage, **mapped}

    def _validate_storage_metadata(self) -> None:
        bad: list[str] = []
        exl3_count = 0
        for prefix, entry in self.tensor_storage.items():
            if entry.get("quant_format") != "exl3":
                continue
            exl3_count += 1
            stored = entry.get("stored_tensors", {})
            suffixes = {name.rsplit(".", 1)[-1] for name in stored}
            required = {"trellis"}
            if not ({"suh", "su"} & suffixes):
                required.add("suh|su")
            if not ({"svh", "sv"} & suffixes):
                required.add("svh|sv")
            missing = [name for name in required if name not in suffixes]
            if missing:
                bad.append(f"{prefix}: missing {','.join(sorted(missing))}")
            if {"mcg", "mul1"} <= suffixes:
                bad.append(f"{prefix}: both mcg and mul1 are present")
        if not exl3_count:
            raise ValueError("quantization_config.json has no EXL3 tensor records")
        if bad:
            raise ValueError("Invalid EXL3 tensor metadata: " + "; ".join(bad[:16]))

    def _force_independent_lm_head(self, hf_config: PretrainedConfig | None) -> None:
        if hf_config is None or not self.has_quantized_lm_head():
            return
        configs: list[Any] = [hf_config]
        try:
            text_config = hf_config.get_text_config()
        except (AttributeError, TypeError):
            text_config = None
        if text_config is not None and text_config is not hf_config:
            configs.append(text_config)
        changed = False
        for config in configs:
            if getattr(config, "tie_word_embeddings", False):
                config.tie_word_embeddings = False
                changed = True
        if changed:
            logger.warning_once(
                "EXL3 metadata contains an independently quantized lm_head; "
                "overriding tie_word_embeddings so vLLM instantiates it."
            )

    def _require_enforce_eager(self) -> None:
        if self.rank_sliced_metadata is not None:
            # The routed-expert fast path is eagerly planned before graph
            # capture. Only its large-M parity fallback remains eager.
            return
        # exllamav3_ext's exl3_gemm autotunes with timing launches on the first
        # call per (m-bucket, k, n, K) shape hash; under CUDA-graph capture
        # those launches fault, and m-bucketing means a warmup pass cannot
        # reliably cover every bucket. Fail fast at build time instead of
        # faulting mid-capture.
        if self._eager_checked:
            return
        self._eager_checked = True
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return
        if not vllm_config.model_config.enforce_eager:
            raise ValueError(
                "The EXL3 quantization backend requires eager execution: "
                "pass --enforce-eager (enforce_eager=True). exl3_gemm "
                "autotunes with timing launches on first use per shape "
                "bucket, which is incompatible with CUDA-graph capture."
            )

    def _get_online_mxfp8_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if layer.__class__.__name__ == "ParallelLMHead":
            return None
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return None
        args = vllm_config.model_config.quantization_config
        if not isinstance(args, QuantizationConfigArgs):
            return None
        shared_expert = _is_shared_expert_projection(prefix)
        spec = args.shared_experts if shared_expert else args.linear
        if spec is None:
            return None
        if (
            not shared_expert
            and args.ignore
            and should_ignore_layer(
                prefix,
                ignore=args.ignore,
                fused_mapping=self.packed_modules_mapping,
            )
        ):
            return None
        if spec.weight != kMxfp8Dynamic or spec.activation is not None:
            raise ValueError(
                "EXL3 online overlay only supports weight='mxfp8' with no "
                "activation override."
            )
        logger.info_once(
            "EXL3 online overlay: quantizing checkpoint-unowned BF16 linears "
            "to MXFP8 at load time."
        )
        logger.debug("MXFP8 EXL3 overlay applied to %s", prefix)
        return _new_mxfp8_online_linear_method()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        self._require_enforce_eager()
        is_lm_head = layer.__class__.__name__ == "ParallelLMHead"
        if is_lm_head and not prefix:
            prefix = "lm_head"
        if isinstance(layer, LinearBase) or is_lm_head:
            if self._linear_prefix_is_exl3(prefix):
                return Exl3LinearMethod(self)
            method = self._get_online_mxfp8_method(layer, prefix)
            return method if method is not None else UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            if not self._moe_prefix_is_exl3(prefix, layer):
                return None
            return Exl3MoEMethod(self, layer.moe_config)
        return None

    def _storage_entry(self, prefix: str) -> dict[str, Any] | None:
        candidates = [prefix]
        if prefix.startswith("model."):
            candidates.append(prefix.removeprefix("model."))
        else:
            candidates.append(f"model.{prefix}")

        # Multimodal wrappers often add an extra `model` or `language_model`
        # segment relative to vLLM's text-only module — interior
        # (`model.language_model.layers...`) or leading
        # (`language_model.lm_head`), so leading segments collapse too.
        parts = prefix.split(".")
        for removable in ("model", "language_model"):
            for idx in range(0, len(parts) - 1):
                if parts[idx] != removable:
                    continue
                collapsed = ".".join(parts[:idx] + parts[idx + 1 :])
                candidates.extend((collapsed, f"model.{collapsed}"))
                if collapsed.startswith("model."):
                    candidates.append(collapsed.removeprefix("model."))

        for candidate in dict.fromkeys(candidates):
            entry = self.tensor_storage.get(candidate)
            if entry is not None:
                return entry
        return None

    def _is_exl3_prefix(self, prefix: str) -> bool:
        entry = self._storage_entry(prefix)
        return entry is not None and entry.get("quant_format") == "exl3"

    def has_serialized_linear(self, prefix: str) -> bool:
        """Return whether a linear is backed by serialized EXL3 tensors."""
        return self._linear_prefix_is_exl3(prefix)

    def _linear_prefix_is_exl3(self, prefix: str) -> bool:
        if self._is_exl3_prefix(prefix):
            return True
        leaf = prefix.rsplit(".", 1)[-1]
        source_leaves = self.packed_modules_mapping.get(leaf)
        if not source_leaves:
            return False
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        return all(
            self._is_exl3_prefix(f"{base}.{source}" if base else source)
            for source in source_leaves
        )

    def _moe_prefix_is_exl3(
        self, prefix: str, layer: torch.nn.Module | None = None
    ) -> bool:
        if self.rank_sliced_metadata is not None:
            match = re.search(r"layers\.(\d+)\b", prefix)
            if match is None:
                return False
            first, last = (int(v) for v in self.rank_sliced_metadata["moe_layers"])
            return first <= int(match.group(1)) <= last
        # Use the layer's checkpoint projection names (the same fields
        # _validate_codebooks keys off) so remapped-projection MoE
        # checkpoints are still detected; fall back to the defaults when the
        # layer variant does not carry them.
        projections = tuple(
            getattr(layer, attr, default)
            for attr, default in (
                ("ckpt_gate_proj_name", "gate_proj"),
                ("ckpt_up_proj_name", "up_proj"),
                ("ckpt_down_proj_name", "down_proj"),
            )
        )
        expert_prefixes = (f"{prefix}.0", f"{prefix}.experts.0")
        return any(
            all(
                self._is_exl3_prefix(f"{expert}.{projection}")
                for projection in projections
            )
            for expert in expert_prefixes
        )

    def codebook_for_prefix(self, prefix: str) -> str | None:
        entry = self._storage_entry(prefix)
        if entry is not None:
            suffixes = {
                name.rsplit(".", 1)[-1] for name in entry.get("stored_tensors", {})
            }
            if "mcg" in suffixes:
                return "mcg"
            if "mul1" in suffixes:
                return "mul1"
        if self.rank_sliced_metadata is not None:
            match = re.search(r"layers\.(\d+)\b", prefix)
            if match is None:
                return None
            first, last = (int(v) for v in self.rank_sliced_metadata["moe_layers"])
            return "mcg" if first <= int(match.group(1)) <= last else None
        return None

    def has_quantized_lm_head(self) -> bool:
        return self._is_exl3_prefix("lm_head")

    def normalize_rank_sliced_weight_name(self, name: str) -> str | None:
        """Drop non-local TP payloads and remove the serialized rank segment."""
        if self.rank_sliced_metadata is None:
            return name
        shared_match = _SHARED_H_WEIGHT_RE.match(name)
        if shared_match is not None:
            if self.rank_sliced_rotation_layout != _SHARED_H_ROTATION_LAYOUT:
                raise ValueError(
                    "rank-sliced EXL3 contains shared-H tensors but metadata "
                    "does not declare rotation_layout='shared_h_v1'"
                )
            projection = shared_match.group("projection")
            field = shared_match.group("field")
            expected_field = "svh" if projection == "down_proj" else "suh"
            if field != expected_field:
                raise ValueError(
                    "invalid shared-H EXL3 tensor: "
                    f"projection={projection!r} requires {expected_field}, got {field}"
                )
            if int(shared_match.group("rank")) != get_tensor_model_parallel_rank():
                return None
            return f"{shared_match.group('experts_prefix')}.0.{projection}.{field}"
        match = _RANK_SLICED_WEIGHT_RE.match(name)
        if match is None:
            return name
        if self.rank_sliced_rotation_layout == _SHARED_H_ROTATION_LAYOUT:
            projection_match = _EXPERT_PROJECTION_RE.match(match.group("prefix"))
            if projection_match is not None:
                projection = projection_match.group("projection")
                field = match.group("field")
                is_h_side = (
                    projection in {"gate_proj", "up_proj"} and field == "suh"
                ) or (projection == "down_proj" and field == "svh")
                if is_h_side:
                    raise ValueError(
                        "shared_h_v1 must store H-side rotations under "
                        "experts.shared_h, not under an expert id"
                    )
        if int(match.group("rank")) != get_tensor_model_parallel_rank():
            return None
        return f"{match.group('prefix')}.{match.group('field')}"


class Exl3Parameter(BasevLLMParameter):
    """Zero-sized parameter holding independently shaped EXL3 components."""

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[ShardId, torch.Tensor] = {}
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        self.exl3_tensors[shard_id] = loaded_weight.contiguous()


def _exl3_weight_loader(
    param: Exl3Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: ShardId = None,
) -> None:
    param.load_exl3_weight(loaded_weight, loaded_shard_id)


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype, extra_weight_attrs
        if layer.__class__.__name__ == "ParallelLMHead":
            org = getattr(layer, "org_vocab_size", None)
            total = getattr(layer, "num_embeddings", None)
            if org is not None and total is not None and org != total:
                raise NotImplementedError(
                    "EXL3 lm_head with added vocabulary is unsupported: the "
                    f"trellis tensor covers the original {org} rows but the "
                    f"layer allocates {total}; TP slicing would silently "
                    "misalign. Strip --lora-extra-vocab-size / added tokens "
                    "or leave lm_head unquantized."
                )
        # Respect the layer's effective topology. disable_tp linears set their
        # own tp_size=1, while ReplicatedLinear carries full weights even when
        # the process-wide tensor group is larger than one.
        if isinstance(layer, ReplicatedLinear):
            layer.exl3_tp_rank = 0
            layer.exl3_tp_size = 1
        else:
            layer.exl3_tp_rank = getattr(
                layer, "tp_rank", get_tensor_model_parallel_rank()
            )
            layer.exl3_tp_size = getattr(
                layer, "tp_size", get_tensor_model_parallel_world_size()
            )
        layer.exl3_input_size = input_size
        layer.exl3_input_size_per_partition = input_size_per_partition
        layer.exl3_output_size = output_size
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer.exl3_rank_sliced_linear = (
            self.quant_config.rank_sliced_metadata is not None
        )
        layer.exl3_shard_ids = self._shard_ids_for_layer(layer, output_partition_sizes)
        layer.exl3_parallel_mode = (
            "row" if input_size_per_partition != input_size else "column"
        )
        source_prefixes = self._source_prefixes_for_layer(layer, layer.exl3_shard_ids)
        layer.exl3_expected_codebooks = {
            shard_id: self.quant_config.codebook_for_prefix(source_prefix)
            for shard_id, source_prefix in zip(
                layer.exl3_shard_ids, source_prefixes, strict=True
            )
        }

        # su/sv are legacy packed sign bitfields.  Modern checkpoints load
        # suh/svh directly.
        for name in ("suh", "svh", "su", "sv", "trellis", "mcg", "mul1"):
            layer.register_parameter(
                name,
                Exl3Parameter(weight_loader=_exl3_weight_loader),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._materialize_legacy_hadamard(layer)
        missing: list[str] = []
        for attr in ("suh", "svh", "trellis"):
            param = getattr(layer, attr)
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in param.exl3_tensors:
                    missing.append(f"{attr}[{shard_id!r}]")
        for shard_id in layer.exl3_shard_ids:
            expected = layer.exl3_expected_codebooks[shard_id]
            has_mcg = shard_id in layer.mcg.exl3_tensors
            has_mul1 = shard_id in layer.mul1.exl3_tensors
            if has_mcg and has_mul1:
                missing.append(f"codebook[{shard_id!r}]=both mcg and mul1")
            elif expected == "mcg" and not has_mcg:
                missing.append(f"mcg[{shard_id!r}]")
            elif expected == "mul1" and not has_mul1:
                missing.append(f"mul1[{shard_id!r}]")
            elif expected is None and (has_mcg or has_mul1):
                missing.append(f"unexpected codebook[{shard_id!r}]")
        if missing:
            prefix = getattr(layer, "prefix", layer.__class__.__name__)
            raise ValueError(
                f"Missing or inconsistent EXL3 tensors for {prefix}: "
                + ", ".join(missing)
            )

        self._pad_lm_head_tensors_for_tensor_parallel(layer)
        self._validate_loaded_tensors(layer)
        self._shard_tensors_for_tensor_parallel(layer)
        self._validate_loaded_tensors(layer)

        # device_loading_context has moved the zero-sized registered parameter
        # to the model target device.  Its device is the safest destination for
        # the tensors kept in the side dictionaries.
        device = layer.trellis.device
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            param = getattr(layer, attr)
            for shard_id, tensor in list(param.exl3_tensors.items()):
                param.exl3_tensors[shard_id] = tensor.to(
                    device=device, non_blocking=True
                ).contiguous()

    @staticmethod
    def _pad_lm_head_tensors_for_tensor_parallel(layer: torch.nn.Module) -> None:
        """Append isolated zero-output blocks to an EXL3 ParallelLMHead."""
        if layer.__class__.__name__ != "ParallelLMHead":
            return

        prefix = getattr(layer, "prefix", "lm_head")
        original_n = getattr(layer, "org_vocab_size", None)
        logical_n = getattr(layer, "num_embeddings", None)
        padded_n = getattr(layer, "num_embeddings_padded", None)
        packed_target_n = getattr(layer, "exl3_output_size", None)
        tp_size = getattr(layer, "exl3_tp_size", None)
        sizes = (
            ("original vocabulary", original_n),
            ("logical vocabulary", logical_n),
            ("padded vocabulary", padded_n),
            ("packed output", packed_target_n),
            ("TP size", tp_size),
        )
        invalid = [
            name
            for name, value in sizes
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ]
        if invalid:
            raise ValueError(
                f"Invalid EXL3 lm_head padding metadata for {prefix}: "
                + ", ".join(invalid)
            )
        assert isinstance(original_n, int)
        assert isinstance(logical_n, int)
        assert isinstance(padded_n, int)
        assert isinstance(packed_target_n, int)
        assert isinstance(tp_size, int)

        if logical_n != original_n:
            raise ValueError(
                "EXL3 lm_head padding cannot represent added vocabulary: "
                f"original N={original_n}, logical N={logical_n}"
            )
        if packed_target_n != padded_n:
            raise ValueError(
                f"EXL3 lm_head packed target N={packed_target_n} does not match "
                f"the layer's padded N={padded_n}"
            )
        if original_n % _HADAMARD_BLOCK:
            raise ValueError(
                f"EXL3 lm_head original N={original_n} is not "
                f"{_HADAMARD_BLOCK}-aligned; padding would mix stored rows "
                "with a new output Hadamard block"
            )
        topology_alignment = tp_size * _HADAMARD_BLOCK
        if padded_n % topology_alignment:
            raise ValueError(
                f"EXL3 lm_head padded N={padded_n} must be a multiple of "
                f"TP size {tp_size} * {_HADAMARD_BLOCK}"
            )
        if padded_n < original_n:
            raise ValueError(
                f"EXL3 lm_head padded N={padded_n} is below original N={original_n}"
            )
        padding_n = padded_n - original_n
        if padding_n % _HADAMARD_BLOCK:
            raise ValueError(
                f"EXL3 lm_head padding N={padding_n} must contain only whole "
                f"{_HADAMARD_BLOCK}-row Hadamard blocks"
            )

        partition_sizes = getattr(layer, "exl3_output_partition_sizes", None)
        expected_partition_n = padded_n // tp_size
        if (
            not isinstance(partition_sizes, list)
            or len(partition_sizes) != 1
            or partition_sizes[0] != expected_partition_n
            or expected_partition_n % _HADAMARD_BLOCK
        ):
            raise ValueError(
                "EXL3 lm_head output partition must be one equal, "
                f"{_HADAMARD_BLOCK}-aligned TP slice; got "
                f"{partition_sizes!r} for padded N={padded_n}, TP size={tp_size}"
            )

        shard_ids = getattr(layer, "exl3_shard_ids", None)
        if not isinstance(shard_ids, list) or not shard_ids:
            raise ValueError(
                f"EXL3 lm_head {prefix} must declare at least one storage shard"
            )

        loaded: list[tuple[ShardId, torch.Tensor, torch.Tensor]] = []
        for shard_id in shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            if trellis.ndim != 3:
                raise ValueError(
                    f"EXL3 lm_head trellis[{shard_id!r}] must be rank 3 before padding"
                )
            if svh.ndim != 1:
                raise ValueError(
                    f"EXL3 lm_head svh[{shard_id!r}] must be rank 1 before padding"
                )
            stored_n = trellis.shape[1] * 16
            if stored_n != original_n or svh.numel() != original_n:
                raise ValueError(
                    f"EXL3 lm_head storage[{shard_id!r}] must have exact "
                    f"original N={original_n} before padding; got "
                    f"trellis N={stored_n}, svh N={svh.numel()}"
                )
            loaded.append((shard_id, trellis, svh))

        if not padding_n:
            return

        replacements: list[tuple[ShardId, torch.Tensor, torch.Tensor]] = []
        for shard_id, trellis, svh in loaded:
            trellis_padding = trellis.new_zeros(
                (trellis.shape[0], padding_n // 16, trellis.shape[2])
            )
            # Output Hadamard transforms are independent per 128-row block and
            # svh is multiplied elementwise afterwards. A whole new block with
            # zero svh produces exact zero without mixing stored output rows.
            svh_padding = svh.new_zeros(padding_n)
            replacements.append(
                (
                    shard_id,
                    torch.cat((trellis, trellis_padding), dim=1),
                    torch.cat((svh, svh_padding), dim=0),
                )
            )
        for shard_id, trellis, svh in replacements:
            layer.trellis.exl3_tensors[shard_id] = trellis
            layer.svh.exl3_tensors[shard_id] = svh

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
        outputs = [
            self._apply_one(layer, x_2d, shard_id) for shard_id in layer.exl3_shard_ids
        ]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output = output + bias.to(dtype=output.dtype)
        output = output.reshape(*original_shape, output.shape[-1])
        return output if output.dtype == original_dtype else output.to(original_dtype)

    @staticmethod
    def _unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
        words = bitfield.contiguous().view(torch.uint16).to(torch.int32)
        masks = 1 << torch.arange(16, device=words.device, dtype=torch.int32)
        negative = (words.unsqueeze(-1) & masks) != 0
        return (
            torch.where(
                negative,
                torch.tensor(-1.0, device=words.device, dtype=torch.float16),
                torch.tensor(1.0, device=words.device, dtype=torch.float16),
            )
            .flatten()
            .contiguous()
        )

    @classmethod
    def _materialize_legacy_hadamard(cls, layer: torch.nn.Module) -> None:
        for packed_name, half_name in (("su", "suh"), ("sv", "svh")):
            packed = getattr(layer, packed_name).exl3_tensors
            half = getattr(layer, half_name).exl3_tensors
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in half and shard_id in packed:
                    half[shard_id] = cls._unpack_signs(packed[shard_id])

    @staticmethod
    def _validate_marker(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.dtype != torch.int32 or tensor.numel() != 1:
            raise ValueError(f"EXL3 {name} must be a scalar int32 sentinel")
        value = int(tensor.reshape(()).item()) & 0xFFFFFFFF
        if value != expected:
            raise ValueError(
                f"Invalid EXL3 {name} sentinel 0x{value:08x}; expected 0x{expected:08x}"
            )

    @classmethod
    def _validate_loaded_tensors(cls, layer: torch.nn.Module) -> None:
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            if trellis.dtype != torch.int16 or trellis.ndim != 3:
                raise ValueError("EXL3 trellis must be rank-3 int16")
            if trellis.shape[2] % 16 or not 1 <= trellis.shape[2] // 16 <= 8:
                raise ValueError(
                    f"Invalid EXL3 trellis bit width {trellis.shape[2]} / 16"
                )
            if suh.dtype != torch.float16 or suh.ndim != 1:
                raise ValueError("EXL3 suh must be rank-1 float16")
            if svh.dtype != torch.float16 or svh.ndim != 1:
                raise ValueError("EXL3 svh must be rank-1 float16")
            k = trellis.shape[0] * 16
            n = trellis.shape[1] * 16
            if suh.numel() != k or svh.numel() != n:
                raise ValueError(
                    "EXL3 dimensions disagree: "
                    f"trellis={tuple(trellis.shape)}, suh={suh.numel()}, "
                    f"svh={svh.numel()}"
                )
            if k % _HADAMARD_BLOCK or n % _HADAMARD_BLOCK:
                raise ValueError(
                    f"EXL3 kernel dimensions must be {_HADAMARD_BLOCK}-aligned, "
                    f"got K={k}, N={n}"
                )
            if shard_id in layer.mcg.exl3_tensors:
                cls._validate_marker(
                    layer.mcg.exl3_tensors[shard_id], _MCG_SENTINEL, "mcg"
                )
            if shard_id in layer.mul1.exl3_tensors:
                cls._validate_marker(
                    layer.mul1.exl3_tensors[shard_id], _MUL1_SENTINEL, "mul1"
                )

    @staticmethod
    def _slice_exl3_tensor(
        tensor: torch.Tensor,
        *,
        dim: int,
        start: int,
        size: int,
    ) -> torch.Tensor:
        if start % _HADAMARD_BLOCK or size % _HADAMARD_BLOCK:
            axis = "output" if dim == 1 else "input"
            raise ValueError(
                f"EXL3 TP {axis} slice must be {_HADAMARD_BLOCK}-aligned, "
                f"got start={start}, size={size}"
            )
        return tensor.narrow(dim, start // 16, size // 16).contiguous()

    @staticmethod
    def _output_shard_size(layer: torch.nn.Module, shard_id: ShardId) -> int:
        if shard_id is None:
            return layer.exl3_output_partition_sizes[0]
        if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
            return layer.exl3_output_partition_sizes[{"q": 0, "k": 1, "v": 2}[shard_id]]
        if isinstance(shard_id, tuple):
            return sum(layer.exl3_output_partition_sizes[idx] for idx in shard_id)
        if isinstance(shard_id, int):
            return layer.exl3_output_partition_sizes[shard_id]
        return layer.exl3_output_partition_sizes[layer.exl3_shard_ids.index(shard_id)]

    @staticmethod
    def _qkv_output_start(
        layer: torch.nn.Module, shard_id: ShardId, shard_size: int
    ) -> int:
        if shard_id in ("k", "v"):
            shard_rank = layer.exl3_tp_rank // layer.num_kv_head_replicas
        else:
            shard_rank = layer.exl3_tp_rank
        return shard_rank * shard_size

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: torch.nn.Module) -> None:
        if layer.exl3_tp_size == 1:
            return
        if layer.exl3_parallel_mode == "row":
            start = layer.exl3_tp_rank * layer.exl3_input_size_per_partition
            size = layer.exl3_input_size_per_partition
            for shard_id in layer.exl3_shard_ids:
                layer.suh.exl3_tensors[shard_id] = (
                    layer.suh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[shard_id],
                    dim=0,
                    start=start,
                    size=size,
                )
            return

        already_sharded = cls._expand_tuple_output_shards(layer)
        for shard_id in layer.exl3_shard_ids:
            if shard_id in already_sharded:
                continue
            size = cls._output_shard_size(layer, shard_id)
            start = cls._qkv_output_start(layer, shard_id, size)
            layer.svh.exl3_tensors[shard_id] = (
                layer.svh.exl3_tensors[shard_id].narrow(0, start, size).contiguous()
            )
            layer.trellis.exl3_tensors[shard_id] = cls._slice_exl3_tensor(
                layer.trellis.exl3_tensors[shard_id],
                dim=1,
                start=start,
                size=size,
            )

    @classmethod
    def _expand_tuple_output_shards(cls, layer: torch.nn.Module) -> set[int]:
        tuples = [sid for sid in layer.exl3_shard_ids if isinstance(sid, tuple)]
        if not tuples:
            return set()

        expanded_ids: list[ShardId] = []
        component_ids: set[int] = set()
        for shard_id in layer.exl3_shard_ids:
            if isinstance(shard_id, tuple):
                expanded_ids.extend(shard_id)
                component_ids.update(shard_id)
            else:
                expanded_ids.append(shard_id)

        for tuple_id in tuples:
            full_offsets: dict[int, int] = {}
            offset = 0
            for idx in tuple_id:
                full_offsets[idx] = offset
                offset += layer.exl3_output_partition_sizes[idx] * layer.exl3_tp_size
            for idx in tuple_id:
                size = layer.exl3_output_partition_sizes[idx]
                start = full_offsets[idx] + layer.exl3_tp_rank * size
                layer.suh.exl3_tensors[idx] = layer.suh.exl3_tensors[tuple_id]
                layer.svh.exl3_tensors[idx] = (
                    layer.svh.exl3_tensors[tuple_id].narrow(0, start, size).contiguous()
                )
                layer.trellis.exl3_tensors[idx] = cls._slice_exl3_tensor(
                    layer.trellis.exl3_tensors[tuple_id],
                    dim=1,
                    start=start,
                    size=size,
                )
                layer.exl3_expected_codebooks[idx] = layer.exl3_expected_codebooks[
                    tuple_id
                ]
                for marker in ("mcg", "mul1"):
                    tensors = getattr(layer, marker).exl3_tensors
                    if tuple_id in tensors:
                        tensors[idx] = tensors[tuple_id]
            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                getattr(layer, attr).exl3_tensors.pop(tuple_id, None)
            layer.exl3_expected_codebooks.pop(tuple_id, None)

        layer.exl3_shard_ids = expanded_ids
        return component_ids

    @staticmethod
    def _shard_ids_for_layer(
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
    ) -> list[ShardId]:
        if len(output_partition_sizes) == 1:
            return [None]
        prefix = getattr(layer, "prefix", "")
        if isinstance(layer, QKVParallelLinear) and len(output_partition_sizes) == 3:
            return ["q", "k", "v"]
        if prefix.endswith("in_proj_qkvz"):
            return [(0, 1, 2), 3]
        return list(range(len(output_partition_sizes)))

    def _source_prefixes_for_layer(
        self, layer: torch.nn.Module, shard_ids: list[ShardId]
    ) -> list[str]:
        prefix = getattr(layer, "prefix", "")
        if len(shard_ids) == 1:
            return [prefix or "lm_head"]
        leaf = prefix.rsplit(".", 1)[-1]
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        sources = self.quant_config.packed_modules_mapping.get(leaf)
        if sources and len(sources) == len(shard_ids):
            return [f"{base}.{source}" if base else source for source in sources]
        raise ValueError(
            f"EXL3 does not know the source matrices for packed layer {prefix}; "
            "add it to the model's packed_modules_mapping."
        )

    @staticmethod
    def _apply_one(
        layer: torch.nn.Module, x: torch.Tensor, shard_id: ShardId
    ) -> torch.Tensor:
        trellis = layer.trellis.exl3_tensors[shard_id]
        packed_k = trellis.shape[0] * 16
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"EXL3 input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        if x.shape[-1] < packed_k:
            x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        output = _exl3_gemm(
            x,
            trellis,
            layer.suh.exl3_tensors[shard_id],
            layer.svh.exl3_tensors[shard_id],
            shard_id in layer.mcg.exl3_tensors,
            shard_id in layer.mul1.exl3_tensors,
        )
        logical_n = Exl3LinearMethod._output_shard_size(layer, shard_id)
        if output.shape[-1] < logical_n:
            raise ValueError(
                f"EXL3 packed N={output.shape[-1]} is below logical N={logical_n}"
            )
        return output[..., :logical_n]


def warmup_exl3_linear(
    model: torch.nn.Module,
    *,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> int:
    """Autotune mixed-checkpoint dense GEMMs before CUDA graph capture."""

    token_counts = {1}
    token_counts.update(int(size) for size in cudagraph_capture_sizes if int(size) > 0)
    seen_signatures: set[
        tuple[torch.device, int, int, int, bool, bool, tuple[int, ...]]
    ] = set()
    warmed = 0
    last_device: torch.device | None = None
    with torch.inference_mode():
        for layer in model.modules():
            if not getattr(layer, "exl3_rank_sliced_linear", False):
                continue
            for shard_id in layer.exl3_shard_ids:
                trellis = layer.trellis.exl3_tensors[shard_id]
                device = trellis.device
                signature = (
                    device,
                    int(trellis.shape[0] * 16),
                    int(trellis.shape[1] * 16),
                    int(trellis.shape[2] // 16),
                    shard_id in layer.mcg.exl3_tensors,
                    shard_id in layer.mul1.exl3_tensors,
                    tuple(sorted(token_counts)),
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                source_width = int(layer.exl3_input_size_per_partition)
                for tokens in sorted(token_counts):
                    source = torch.zeros(
                        (tokens, source_width),
                        dtype=torch.float16,
                        device=device,
                    )
                    Exl3LinearMethod._apply_one(layer, source, shard_id)
                    warmed += 1
                last_device = device
        if warmed and last_device is not None and last_device.type == "cuda":
            torch.cuda.synchronize(last_device)
    return warmed


class Exl3FusedMlpMethod:
    """Use the exact fused MoE kernel for an eligible MCG8 MLP."""

    def __init__(
        self,
        quant_config: Exl3Config,
        prefix: str,
        *,
        is_shared_expert: bool,
    ) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        kind = "SHARED_EXPERT" if is_shared_expert else "DENSE_MLP"
        default_min_rows = 512 if is_shared_expert else 2048
        self.enabled = (
            quant_config.rank_sliced_metadata is not None
            and os.environ.get(f"VLLM_EXL3_FUSED_{kind}", "0") == "1"
        )
        self.min_rows = _positive_env_int(
            f"VLLM_EXL3_FUSED_{kind}_MIN_M", default_min_rows
        )
        self._prepared = False
        self._ineligible = False
        self._layer: SimpleNamespace | None = None
        self._method: Exl3MoEMethod | None = None
        self._ids: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None

    @staticmethod
    def _bits(trellis: torch.Tensor) -> int:
        if trellis.ndim != 3 or trellis.shape[2] % 16:
            raise ValueError(
                f"Invalid EXL3 fused-MLP Trellis shape {tuple(trellis.shape)}"
            )
        return int(trellis.shape[2] // 16)

    def _prepare(
        self,
        gate_up_proj: torch.nn.Module,
        down_proj: torch.nn.Module,
        device: torch.device,
    ) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "EXL3 fused MLP must be prepared during the eager profile pass "
                "before CUDA graph capture"
            )
        if not isinstance(
            getattr(gate_up_proj, "quant_method", None), Exl3LinearMethod
        ) or not isinstance(getattr(down_proj, "quant_method", None), Exl3LinearMethod):
            self._ineligible = True
            return
        if tuple(gate_up_proj.exl3_shard_ids) != (0, 1) or tuple(
            down_proj.exl3_shard_ids
        ) != (None,):
            self._ineligible = True
            return

        gate = gate_up_proj.trellis.exl3_tensors[0]
        up = gate_up_proj.trellis.exl3_tensors[1]
        down = down_proj.trellis.exl3_tensors[None]
        if (
            not all(shard in gate_up_proj.mcg.exl3_tensors for shard in (0, 1))
            or None not in down_proj.mcg.exl3_tensors
        ):
            self._ineligible = True
            return
        bits = {self._bits(gate), self._bits(up), self._bits(down)}
        if bits != {8}:
            self._ineligible = True
            return

        hidden_size = int(gate.shape[0] * 16)
        intermediate_size = int(gate.shape[1] * 16)
        if (
            tuple(up.shape[:2]) != tuple(gate.shape[:2])
            or int(down.shape[0] * 16) != intermediate_size
            or int(down.shape[1] * 16) != hidden_size
        ):
            raise ValueError("EXL3 fused-MLP gate/up/down geometry is inconsistent")
        pointer_slabs = (
            gate.unsqueeze(0),
            gate_up_proj.suh.exl3_tensors[0].unsqueeze(0),
            gate_up_proj.svh.exl3_tensors[0].unsqueeze(0),
            up.unsqueeze(0),
            gate_up_proj.suh.exl3_tensors[1].unsqueeze(0),
            gate_up_proj.svh.exl3_tensors[1].unsqueeze(0),
            down.unsqueeze(0),
            down_proj.suh.exl3_tensors[None].unsqueeze(0),
            down_proj.svh.exl3_tensors[None].unsqueeze(0),
        )
        vllm_config = get_current_vllm_config_or_none()
        scheduler_config = None if vllm_config is None else vllm_config.scheduler_config
        if (
            scheduler_config is None
            or getattr(scheduler_config, "max_num_batched_tokens", None) is None
        ):
            raise ValueError(
                "EXL3 fused MLP requires scheduler_config.max_num_batched_tokens"
            )
        capacity = int(scheduler_config.max_num_batched_tokens)
        self._layer = SimpleNamespace(
            layer_name=self.prefix,
            exl3_max_num_batched_tokens=capacity,
            exl3_hidden_size=hidden_size,
            exl3_intermediate_size_per_partition=intermediate_size,
            exl3_trellis_bits=8,
            exl3_exact_rank_sliced_moe=True,
            exl3_pointer_tables=tuple(
                Exl3MoEMethod._pointer_table(slab) for slab in pointer_slabs
            ),
            exl3_expert_map=torch.zeros(1, dtype=torch.int64, device=device),
            local_num_experts=1,
        )
        self._method = object.__new__(Exl3MoEMethod)
        self._method.quant_config = self.quant_config
        self._ids = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self._weights = torch.ones((capacity, 1), dtype=torch.float32, device=device)
        self._prepared = True

    def apply(
        self,
        gate_up_proj: torch.nn.Module,
        down_proj: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        rows = int(x_2d.shape[0])
        if rows < self.min_rows:
            return None
        if not self._prepared and not self._ineligible:
            self._prepare(gate_up_proj, down_proj, x.device)
        if self._ineligible:
            return None
        assert self._layer is not None
        assert self._method is not None
        assert self._ids is not None
        assert self._weights is not None
        if rows > self._ids.shape[0]:
            raise ValueError(
                "EXL3 fused-MLP rows exceed scheduler capacity: "
                f"rows={rows}, capacity={self._ids.shape[0]}"
            )
        output = self._method._apply_exact_rank_sliced_moe(
            self._layer,
            x_2d,
            self._weights[:rows],
            self._ids[:rows],
        )
        if down_proj.reduce_results and down_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output.reshape(*original_shape[:-1], output.shape[-1])


class Exl3MoEParameter(BasevLLMParameter):
    """EXL3 tensors keyed by expert/projection, optionally in one GPU slab."""

    def __new__(
        cls,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
    ):
        del num_experts, shard_ids, preallocate
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(
        self,
        *,
        weight_loader,
        num_experts: int = 0,
        shard_ids: tuple[str, ...] = (),
        preallocate: bool = False,
    ):
        self.exl3_tensors: dict[tuple[int, str], torch.Tensor] = {}
        self.exl3_backing: torch.Tensor | None = None
        self.exl3_num_experts = int(num_experts)
        self.exl3_shard_ids = tuple(shard_ids)
        self.exl3_preallocate = bool(preallocate)
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        *,
        expert_id: int,
        shard_id: str,
    ) -> None:
        key = (int(expert_id), str(shard_id))
        if not self.exl3_preallocate:
            self.exl3_tensors[key] = loaded_weight.contiguous()
            return
        if self.exl3_num_experts <= 0 or shard_id not in self.exl3_shard_ids:
            raise ValueError(
                f"invalid EXL3 slab key expert={expert_id}, shard={shard_id!r}"
            )
        if not 0 <= int(expert_id) < self.exl3_num_experts:
            raise ValueError(
                f"EXL3 expert {expert_id} is outside [0, {self.exl3_num_experts})"
            )
        if self.device.type == "meta":
            raise RuntimeError("rank-sliced EXL3 slabs cannot be allocated on meta")
        if self.exl3_backing is None:
            prefix = (
                (len(self.exl3_shard_ids), self.exl3_num_experts)
                if len(self.exl3_shard_ids) > 1
                else (self.exl3_num_experts,)
            )
            self.exl3_backing = torch.empty(
                prefix + tuple(loaded_weight.shape),
                dtype=loaded_weight.dtype,
                device=self.device,
            )
        shard_index = self.exl3_shard_ids.index(shard_id)
        target = (
            self.exl3_backing[shard_index, expert_id]
            if len(self.exl3_shard_ids) > 1
            else self.exl3_backing[expert_id]
        )
        if tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                "rank-sliced EXL3 tensor shape changed within one slab: "
                f"expected={tuple(target.shape)}, got={tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight, non_blocking=True)
        self.exl3_tensors[key] = target


def _exl3_moe_weight_loader(
    param: Exl3MoEParameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    del weight_name
    param.load_exl3_weight(
        loaded_weight,
        expert_id=expert_id,
        shard_id=shard_id,
    )
    return True if return_success else None


# Model loaders (e.g. llama4-style paths) check this attribute before routing
# expert tensors through a param's weight_loader with MoE kwargs.
_exl3_moe_weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def _valid_fused_route_pack_contract(
    rank_sliced_metadata: dict[str, Any] | None,
    layer: RoutedExperts,
) -> bool:
    correction_bias = layer.e_score_correction_bias
    return (
        rank_sliced_metadata is not None
        and getattr(layer, "exl3_mixed_bitrate", False)
        and layer.activation == MoEActivation.SILU
        and layer.expert_map is None
        and not layer.apply_router_weight_on_input
        and layer.custom_routing_function is None
        and layer.use_grouped_topk
        and layer.num_expert_group == 1
        and layer.topk_group == 1
        and layer.scoring_func == "sigmoid"
        and layer.renormalize
        and layer.global_num_experts == 256
        and layer.local_num_experts == 256
        and layer.top_k == 8
        and float(layer.routed_scaling_factor) == 1.0
        and correction_bias is not None
        and correction_bias.dtype == torch.float32
        and tuple(correction_bias.shape) == (256,)
    )


class Exl3MoEMethod(FusedMoEMethodBase):
    """Correctness MoE path: route, then use three dense EXL3 GEMMs/expert."""

    def __init__(self, quant_config: Exl3Config, moe) -> None:
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del extra_weight_attrs
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"EXL3 MoE requires BF16 or FP16 activations, got {params_dtype}"
            )
        if self.moe.moe_parallel_config.use_ep:
            raise NotImplementedError(
                "EXL3 correctness MoE currently supports TP but not expert parallelism"
            )
        if self.moe.has_bias:
            raise NotImplementedError(
                "EXL3 correctness MoE does not yet support expert biases"
            )
        layer.exl3_tp_rank = self.moe.moe_parallel_config.tp_rank
        layer.exl3_tp_size = self.moe.moe_parallel_config.tp_size
        layer.exl3_hidden_size = hidden_size
        layer.exl3_intermediate_size_per_partition = intermediate_size_per_partition
        layer.exl3_params_dtype = params_dtype
        rank_sliced = self.quant_config.rank_sliced_metadata is not None
        shared_h = (
            rank_sliced
            and self.quant_config.rank_sliced_rotation_layout
            == _SHARED_H_ROTATION_LAYOUT
        )
        layer.exl3_shared_h_rotations = shared_h
        if rank_sliced:
            layer.exl3_layer_bitrates = self.quant_config.trellis_bitrates_for_layer(
                layer.layer_name
            )
            layer.exl3_mixed_bitrate = len(set(layer.exl3_layer_bitrates)) > 1
            if not layer.exl3_mixed_bitrate:
                layer.exl3_trellis_bits = layer.exl3_layer_bitrates[0]
            checkpoint_tp = int(self.quant_config.rank_sliced_metadata["tp"])
            if checkpoint_tp != layer.exl3_tp_size:
                raise ValueError(
                    "rank-sliced EXL3 checkpoint TP does not match runtime: "
                    f"checkpoint={checkpoint_tp}, runtime={layer.exl3_tp_size}"
                )
            expected_experts = int(
                self.quant_config.rank_sliced_metadata["experts_per_layer"]
            )
            if expected_experts != num_experts:
                raise ValueError(
                    "rank-sliced EXL3 expert count does not match the model: "
                    f"checkpoint={expected_experts}, model={num_experts}"
                )
            vllm_config = get_current_vllm_config_or_none()
            scheduler_config = (
                vllm_config.scheduler_config if vllm_config is not None else None
            )
            # No silent fallback: a wrong capacity here puts the target and the
            # rank-sliced MTP draft on different plans with no error, which is
            # exactly the class of mismatch that corrupts only at scale.
            if (
                scheduler_config is None
                or getattr(scheduler_config, "max_num_batched_tokens", None) is None
            ):
                raise ValueError(
                    "EXL3 rank-sliced MoE requires scheduler_config."
                    "max_num_batched_tokens to plan its Trellis arena; refusing to "
                    "guess a capacity."
                )
            layer.exl3_max_num_batched_tokens = int(
                scheduler_config.max_num_batched_tokens
            )
        for prefix, shard_ids in (("w13", ("w1", "w3")), ("w2", ("w2",))):
            for suffix in ("suh", "svh", "trellis", "mcg", "mul1"):
                shared_parameter = shared_h and (
                    (prefix == "w13" and suffix == "suh")
                    or (prefix == "w2" and suffix == "svh")
                )
                layer.register_parameter(
                    f"{prefix}_{suffix}",
                    Exl3MoEParameter(
                        weight_loader=_exl3_moe_weight_loader,
                        num_experts=1 if shared_parameter else num_experts,
                        shard_ids=shard_ids,
                        preallocate=rank_sliced
                        and suffix
                        in (
                            {"suh", "svh"}
                            if getattr(layer, "exl3_mixed_bitrate", False)
                            else {"suh", "svh", "trellis"}
                        ),
                    ),
                )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        required = {"w13": ("w1", "w3"), "w2": ("w2",)}
        missing: list[str] = []
        for prefix, shard_ids in required.items():
            for attr in ("suh", "svh", "trellis"):
                tensors = getattr(layer, f"{prefix}_{attr}").exl3_tensors
                shared_parameter = getattr(
                    layer, "exl3_shared_h_rotations", False
                ) and (
                    (prefix == "w13" and attr == "suh")
                    or (prefix == "w2" and attr == "svh")
                )
                expert_ids = (
                    (0,) if shared_parameter else range(layer.local_num_experts)
                )
                for expert_id in expert_ids:
                    for shard_id in shard_ids:
                        if (expert_id, shard_id) not in tensors:
                            missing.append(f"{prefix}_{attr}[{expert_id},{shard_id}]")
        if missing:
            raise ValueError(
                f"Missing EXL3 MoE tensors for {layer.layer_name}: "
                + ", ".join(missing[:32])
                + (" ..." if len(missing) > 32 else "")
            )
        self._validate_codebooks(layer)
        if self.quant_config.rank_sliced_metadata is None:
            self._shard_tensors_for_tensor_parallel(layer)
        device = layer.w13_trellis.device
        for prefix in ("w13", "w2"):
            for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
                param = getattr(layer, f"{prefix}_{attr}")
                for key, tensor in list(param.exl3_tensors.items()):
                    param.exl3_tensors[key] = tensor.to(
                        device=device, non_blocking=True
                    ).contiguous()
        self._validate_moe_shapes(layer)

        if self.quant_config.rank_sliced_metadata is not None:
            self._prepare_rank_sliced_weights(layer)
            return

    def _validate_codebooks(self, layer: RoutedExperts) -> None:
        projections = {
            "w1": layer.ckpt_gate_proj_name,
            "w2": layer.ckpt_down_proj_name,
            "w3": layer.ckpt_up_proj_name,
        }
        for expert_id in range(layer.local_num_experts):
            for shard_id, projection in projections.items():
                prefix = f"{layer.layer_name}.{expert_id}.{projection}"
                expected = self.quant_config.codebook_for_prefix(prefix)
                group = "w2" if shard_id == "w2" else "w13"
                key = (expert_id, shard_id)
                has_mcg = key in getattr(layer, f"{group}_mcg").exl3_tensors
                has_mul1 = key in getattr(layer, f"{group}_mul1").exl3_tensors
                if has_mcg and has_mul1:
                    raise ValueError(f"EXL3 MoE {prefix} has both codebooks")
                if expected == "mcg" and not has_mcg:
                    raise ValueError(f"EXL3 MoE {prefix} is missing mcg")
                if expected == "mul1" and not has_mul1:
                    raise ValueError(f"EXL3 MoE {prefix} is missing mul1")
                if expected is None and (has_mcg or has_mul1):
                    raise ValueError(
                        f"EXL3 MoE {prefix} has an unexpected codebook marker"
                    )
                if has_mcg:
                    Exl3LinearMethod._validate_marker(
                        getattr(layer, f"{group}_mcg").exl3_tensors[key],
                        _MCG_SENTINEL,
                        "mcg",
                    )
                if has_mul1:
                    Exl3LinearMethod._validate_marker(
                        getattr(layer, f"{group}_mul1").exl3_tensors[key],
                        _MUL1_SENTINEL,
                        "mul1",
                    )

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: RoutedExperts) -> None:
        if layer.exl3_tp_size == 1:
            return
        start = layer.exl3_tp_rank * layer.exl3_intermediate_size_per_partition
        size = layer.exl3_intermediate_size_per_partition
        for expert_id in range(layer.local_num_experts):
            for shard_id in ("w1", "w3"):
                key = (expert_id, shard_id)
                layer.w13_svh.exl3_tensors[key] = (
                    layer.w13_svh.exl3_tensors[key].narrow(0, start, size).contiguous()
                )
                layer.w13_trellis.exl3_tensors[key] = (
                    Exl3LinearMethod._slice_exl3_tensor(
                        layer.w13_trellis.exl3_tensors[key],
                        dim=1,
                        start=start,
                        size=size,
                    )
                )
            key = (expert_id, "w2")
            layer.w2_suh.exl3_tensors[key] = (
                layer.w2_suh.exl3_tensors[key].narrow(0, start, size).contiguous()
            )
            layer.w2_trellis.exl3_tensors[key] = Exl3LinearMethod._slice_exl3_tensor(
                layer.w2_trellis.exl3_tensors[key],
                dim=0,
                start=start,
                size=size,
            )

    @staticmethod
    def _validate_moe_shapes(layer: RoutedExperts) -> None:
        for expert_id in range(layer.local_num_experts):
            layer_bitrates = tuple(getattr(layer, "exl3_layer_bitrates", ()))
            expected_bits = int(layer_bitrates[expert_id]) if layer_bitrates else None
            for group, shard_ids in (("w13", ("w1", "w3")), ("w2", ("w2",))):
                for shard_id in shard_ids:
                    key = (expert_id, shard_id)
                    trellis = getattr(layer, f"{group}_trellis").exl3_tensors[key]
                    suh_key = (
                        (0, shard_id)
                        if getattr(layer, "exl3_shared_h_rotations", False)
                        and group == "w13"
                        else key
                    )
                    svh_key = (
                        (0, shard_id)
                        if getattr(layer, "exl3_shared_h_rotations", False)
                        and group == "w2"
                        else key
                    )
                    suh = getattr(layer, f"{group}_suh").exl3_tensors[suh_key]
                    svh = getattr(layer, f"{group}_svh").exl3_tensors[svh_key]
                    if (
                        trellis.dtype != torch.int16
                        or trellis.ndim != 3
                        or trellis.shape[2] % 16
                        or not 1 <= trellis.shape[2] // 16 <= 8
                        or (
                            expected_bits is not None
                            and trellis.shape[2] != 16 * expected_bits
                        )
                        or suh.dtype != torch.float16
                        or suh.ndim != 1
                        or svh.dtype != torch.float16
                        or svh.ndim != 1
                        or suh.numel() != trellis.shape[0] * 16
                        or svh.numel() != trellis.shape[1] * 16
                        or (trellis.shape[0] * 16) % _HADAMARD_BLOCK
                        or (trellis.shape[1] * 16) % _HADAMARD_BLOCK
                    ):
                        raise ValueError(
                            f"Invalid EXL3 MoE tensors for expert={expert_id}, "
                            f"projection={shard_id}"
                        )

    @staticmethod
    def _rank_sliced_backing(
        layer: RoutedExperts,
        param_name: str,
    ) -> torch.Tensor:
        param = getattr(layer, param_name)
        backing = param.exl3_backing
        if backing is None or not backing.is_contiguous():
            raise RuntimeError(
                f"rank-sliced EXL3 parameter {param_name} has no contiguous slab"
            )
        for (expert_id, shard_id), tensor in param.exl3_tensors.items():
            shard_index = param.exl3_shard_ids.index(shard_id)
            expected = (
                backing[shard_index, expert_id]
                if len(param.exl3_shard_ids) > 1
                else backing[expert_id]
            )
            if tensor.data_ptr() != expected.data_ptr():
                raise RuntimeError(
                    "rank-sliced EXL3 expert payload lost its slab alias: "
                    f"{param_name}[{expert_id},{shard_id}]"
                )
        return backing

    @staticmethod
    def _pointer_table(
        slab: torch.Tensor,
        *,
        num_experts: int | None = None,
    ) -> torch.Tensor:
        if slab.ndim < 2 or not slab[0].is_contiguous():
            raise RuntimeError("EXL3 pointer-table rows must be contiguous")
        rows = int(slab.shape[0])
        entries = rows if num_experts is None else int(num_experts)
        if rows not in {1, entries}:
            raise RuntimeError(
                "EXL3 pointer-table rows must be per-expert or broadcast: "
                f"rows={rows}, experts={entries}"
            )
        step = slab.stride(0) * slab.element_size()
        base = slab.data_ptr()
        return torch.tensor(
            [
                base + (0 if rows == 1 else expert_id) * step
                for expert_id in range(entries)
            ],
            dtype=torch.int64,
            device=slab.device,
        )

    @staticmethod
    def _trellis_tile_config(hidden_size: int, intermediate_size: int):
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError(
                "rank-sliced EXL3 full rotations require hidden and "
                "intermediate dimensions divisible by 128"
            )
        if hidden_size % 256 == 0 and intermediate_size % 256 == 0:
            return (64, 256, 64, 256)
        return (64, 128, 64, 128)

    @staticmethod
    def _select_rotation_rows(
        rotations: torch.Tensor, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        if int(rotations.shape[0]) == 1:
            return rotations
        return rotations.index_select(0, expert_ids).contiguous()

    @staticmethod
    def _mixed_trellis_tile_config(
        hidden_size: int, intermediate_size: int
    ) -> tuple[int, int, int, int]:
        if hidden_size % 128 or intermediate_size % 128:
            raise ValueError(
                "mixed rank-sliced EXL3 requires hidden and intermediate "
                "dimensions divisible by 128"
            )
        if hidden_size % 512 == 0:
            return (128, 128, 32, 512)
        if hidden_size % 256 == 0:
            return (128, 128, 64, 256)
        return (128, 128, 128, 128)

    def _prepare_mixed_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        mixed_api = _load_sparkinfer_mixed_trellis()
        num_experts = int(layer.local_num_experts)
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        bitrates = tuple(int(value) for value in layer.exl3_layer_bitrates)
        if len(bitrates) != num_experts:
            raise ValueError(
                "mixed rank-sliced EXL3 bitrate count does not match experts: "
                f"{len(bitrates)} != {num_experts}"
            )
        tiers = {
            bits: tuple(
                expert
                for expert, expert_bits in enumerate(bitrates)
                if expert_bits == bits
            )
            for bits in sorted(set(bitrates))
        }
        if len(tiers) != 2:
            raise ValueError(
                "one-grid mixed Trellis requires exactly two bitrates, got "
                f"{tuple(tiers)}"
            )

        w13_param = layer.w13_trellis
        w2_param = layer.w2_trellis
        if tuple(w13_param.exl3_shard_ids) != ("w1", "w3") or tuple(
            w2_param.exl3_shard_ids
        ) != ("w2",):
            raise ValueError("mixed rank-sliced EXL3 shard layout changed")
        gate_suh, up_suh = self._rank_sliced_backing(layer, "w13_suh")
        gate_svh, up_svh = self._rank_sliced_backing(layer, "w13_svh")
        down_suh = self._rank_sliced_backing(layer, "w2_suh")
        down_svh = self._rank_sliced_backing(layer, "w2_svh")
        device = gate_suh.device
        tile_config = self._mixed_trellis_tile_config(hidden_size, intermediate_size)
        tier_order = tuple(expert for experts in tiers.values() for expert in experts)
        tier_index = torch.tensor(tier_order, dtype=torch.long, device=device)
        combined_gate_suh = self._select_rotation_rows(gate_suh, tier_index)
        combined_up_suh = self._select_rotation_rows(up_suh, tier_index)
        broadcast_suh = int(combined_gate_suh.shape[0]) == 1
        if broadcast_suh != (int(combined_up_suh.shape[0]) == 1):
            raise ValueError(
                "mixed EXL3 gate/up SUH rotations must both be per-expert "
                "or both broadcast"
            )
        combined_intermediate_rotations = torch.cat(
            (
                gate_svh.index_select(0, tier_index),
                up_svh.index_select(0, tier_index),
                down_suh.index_select(0, tier_index),
            ),
            dim=1,
        ).contiguous()
        combined_down_svh = self._select_rotation_rows(down_svh, tier_index)
        broadcast_svh = int(combined_down_svh.shape[0]) == 1
        rotations = SimpleNamespace(
            intermediate=combined_intermediate_rotations,
            gate_suh=combined_gate_suh,
            up_suh=combined_up_suh,
            down_svh=combined_down_svh,
        )

        prepared_tiers = []
        tier_ids = []
        tier_offset = 0
        for bits, expert_ids in tiers.items():
            w13 = torch.stack(
                tuple(
                    torch.stack(
                        tuple(
                            w13_param.exl3_tensors[(expert, shard)]
                            for expert in expert_ids
                        )
                    )
                    for shard in ("w1", "w3")
                )
            ).contiguous()
            w2 = torch.stack(
                tuple(w2_param.exl3_tensors[(expert, "w2")] for expert in expert_ids)
            ).contiguous()
            expected_w13 = (
                2,
                len(expert_ids),
                hidden_size // 16,
                intermediate_size // 16,
                16 * bits,
            )
            expected_w2 = (
                len(expert_ids),
                intermediate_size // 16,
                hidden_size // 16,
                16 * bits,
            )
            if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
                raise ValueError(
                    f"mixed EXL3 K{bits} slab geometry mismatch: "
                    f"{tuple(w13.shape)}/{tuple(w2.shape)} != "
                    f"{expected_w13}/{expected_w2}"
                )
            tier_slice = slice(tier_offset, tier_offset + len(expert_ids))
            tier_gate_suh = (
                combined_gate_suh if broadcast_suh else combined_gate_suh[tier_slice]
            )
            tier_up_suh = (
                combined_up_suh if broadcast_suh else combined_up_suh[tier_slice]
            )
            tier_down_svh = (
                combined_down_svh if broadcast_svh else combined_down_svh[tier_slice]
            )
            prepared_tiers.append(
                mixed_api.prepare_weights(
                    w13=w13,
                    w2=w2,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=len(expert_ids),
                    activation=layer.activation.value,
                    fc1_tile_n=tile_config[1],
                    fc2_tile_n=tile_config[3],
                    params_dtype=torch.float16,
                    w13_layout="trellis3_t256_proj",
                    trellis_bits=bits,
                    codebook="mcg",
                    gate_suh=tier_gate_suh,
                    up_suh=tier_up_suh,
                    intermediate_rotations=combined_intermediate_rotations[tier_slice],
                    down_svh=tier_down_svh,
                    tile_config=tile_config,
                    workspace=w13.view(torch.int32).reshape(-1)[:1],
                )
            )
            tier_ids.append(expert_ids)
            tier_offset += len(expert_ids)

        global_to_combined, descriptor_map = mixed_api.build_tiered_maps(
            tier_ids[0], tier_ids[1], device=device
        )
        layer.exl3_mixed_trellis = {
            "tiers": tuple(prepared_tiers),
            "tier_ids": tuple(tier_ids),
            "tier_bits": tuple(tiers),
            "global_to_combined": global_to_combined,
            "descriptor_map": descriptor_map,
            "rotations": rotations,
            "broadcast_suh": broadcast_suh,
            "broadcast_svh": broadcast_svh,
            "tile_config": tile_config,
        }
        layer.exl3_trellis_tile_config = tile_config
        layer.exl3_exact_rank_sliced_moe = False
        for prefix in ("w13", "w2"):
            for suffix in ("suh", "svh", "trellis", "mcg", "mul1"):
                param = getattr(layer, f"{prefix}_{suffix}")
                param.exl3_tensors.clear()
                param.exl3_backing = None
        logger.info(
            "EXL3 mixed Trellis %s: tiers=%s",
            layer.layer_name,
            tuple((bits, len(ids)) for bits, ids in zip(tiers, tier_ids, strict=True)),
        )

    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        if getattr(layer, "exl3_mixed_bitrate", False):
            self._prepare_mixed_rank_sliced_weights(layer)
            return
        num_experts = int(layer.local_num_experts)
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        bits = int(layer.exl3_trellis_bits)
        if bits not in (3, 4, 5, 6, 8):
            raise ValueError(
                f"native rank-sliced EXL3 has unsupported MCG{bits} bitrate"
            )

        w13 = self._rank_sliced_backing(layer, "w13_trellis")
        w2 = self._rank_sliced_backing(layer, "w2_trellis")
        gate_suh, up_suh = self._rank_sliced_backing(layer, "w13_suh")
        gate_svh, up_svh = self._rank_sliced_backing(layer, "w13_svh")
        down_suh = self._rank_sliced_backing(layer, "w2_suh")
        down_svh = self._rank_sliced_backing(layer, "w2_svh")
        expected_w13 = (
            2,
            num_experts,
            hidden_size // 16,
            intermediate_size // 16,
            16 * bits,
        )
        expected_w2 = (
            num_experts,
            intermediate_size // 16,
            hidden_size // 16,
            16 * bits,
        )
        if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
            raise ValueError(
                "rank-sliced EXL3 slab geometry mismatch: "
                f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}, "
                f"expected={expected_w13}/{expected_w2}"
            )

        tile_config = self._trellis_tile_config(hidden_size, intermediate_size)
        layer.exl3_trellis_tile_config = tile_config
        layer.exl3_exact_rank_sliced_moe = bits == 8
        if layer.exl3_exact_rank_sliced_moe:
            slabs = (
                w13[0],
                gate_suh,
                gate_svh,
                w13[1],
                up_suh,
                up_svh,
                w2,
                down_suh,
                down_svh,
            )
            layer.exl3_pointer_tables = tuple(
                self._pointer_table(slab, num_experts=num_experts) for slab in slabs
            )
            ext = _load_exl3_ext()
            required = {"exl3_moe_fused", "exl3_moe_max_concurrency"}
            missing = sorted(name for name in required if not hasattr(ext, name))
            if missing:
                raise RuntimeError(
                    "The EXL3 extension lacks exact MCG8 MoE entry points: "
                    + ", ".join(missing)
                )
            logger.info_once(
                "EXL3 rank-sliced MCG8 route=exact-exllamav3 layer=%s",
                layer.layer_name,
            )
        else:
            intermediate_rotations = torch.empty(
                (num_experts, 3 * intermediate_size),
                dtype=torch.float16,
                device=w13.device,
            )
            gate_svh_view = intermediate_rotations[:, :intermediate_size]
            up_svh_view = intermediate_rotations[
                :, intermediate_size : 2 * intermediate_size
            ]
            down_suh_view = intermediate_rotations[:, 2 * intermediate_size :]
            gate_svh_view.copy_(gate_svh)
            up_svh_view.copy_(up_svh)
            down_suh_view.copy_(down_suh)

            marker = layer.w13_mcg.exl3_tensors[(0, "w1")]
            api = _load_sparkinfer_trellis()
            prepared_gate_suh = gate_suh
            prepared_up_suh = up_suh
            prepared_down_svh = down_svh
            if getattr(layer, "exl3_shared_h_rotations", False):
                arena_key = (
                    _runtime_owner_token(self.quant_config, layer),
                    w13.device.type,
                    w13.device.index,
                    num_experts,
                    hidden_size,
                )
                arena = _SHARED_H_ROTATION_ARENAS.get(arena_key)
                if arena is None:
                    arena = _SharedHRotationArena(
                        torch.empty(
                            (num_experts, 3 * hidden_size),
                            dtype=torch.float16,
                            device=w13.device,
                        )
                    )
                    _SHARED_H_ROTATION_ARENAS[arena_key] = arena
                layer.exl3_shared_h_arena = arena
                layer.exl3_shared_gate_suh = gate_suh
                layer.exl3_shared_up_suh = up_suh
                layer.exl3_shared_down_svh = down_svh
                prepared_gate_suh = arena.data[:, :hidden_size]
                prepared_up_suh = arena.data[:, hidden_size : 2 * hidden_size]
                prepared_down_svh = arena.data[:, 2 * hidden_size :]
            layer.exl3_trellis_weights = api.prepare_weights(
                w13,
                w2,
                gate_suh=prepared_gate_suh,
                up_suh=prepared_up_suh,
                intermediate_rotations=intermediate_rotations,
                down_svh=prepared_down_svh,
                codebook="mcg",
                mcg=int(marker.reshape(()).item()) & 0xFFFFFFFF,
                tile_config=tile_config,
            )
            prepared_bits = int(layer.exl3_trellis_weights.trellis_bits)
            if prepared_bits != bits:
                raise RuntimeError(
                    "Sparkinfer prepared a bitrate inconsistent with checkpoint "
                    f"metadata: expected={bits}, prepared={prepared_bits}"
                )

            slabs = (
                w13[0],
                prepared_gate_suh,
                gate_svh_view,
                w13[1],
                prepared_up_suh,
                up_svh_view,
                w2,
                down_suh_view,
                prepared_down_svh,
            )
            layer.exl3_pointer_tables = tuple(
                self._pointer_table(slab, num_experts=num_experts) for slab in slabs
            )
            layer.exl3_intermediate_rotations = intermediate_rotations
            for param_name in ("w13_svh", "w2_suh"):
                param = getattr(layer, param_name)
                param.exl3_tensors.clear()
                param.exl3_backing = None

        layer.exl3_expert_map = torch.arange(
            num_experts,
            dtype=torch.int64,
            device=w13.device,
        )
        if getattr(layer, "exl3_shared_h_rotations", False):
            saved_bytes = (num_experts - 1) * 3 * hidden_size * 2
            logger.info_once(
                "EXL3 shared-H rotation layout active; saving %.2f MiB per "
                "routed-expert layer and TP rank",
                saved_bytes / (1024 * 1024),
            )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.long

    @property
    def is_monolithic(self) -> bool:
        return os.environ.get("VLLM_EXL3_FUSED_ROUTE_PACK", "0") == "1"

    def _exact_rank_sliced_runtime(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> dict[str, Any]:
        max_batched_tokens = int(layer.exl3_max_num_batched_tokens)
        chunk = _positive_env_int("VLLM_EXL3_EXACT_CHUNK", 64)
        topk = int(topk_ids.shape[1])
        key = (
            _runtime_owner_token(self.quant_config, layer),
            x.device.index,
            x.dtype,
            int(layer.exl3_hidden_size),
            int(layer.exl3_intermediate_size_per_partition),
            int(layer.local_num_experts),
            topk,
            max_batched_tokens,
            chunk,
        )
        runtime = _EXACT_RANK_SLICED_RUNTIMES.get(key)
        if runtime is not None:
            return runtime
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Exact MCG8 EXL3 runtime must be allocated during the eager "
                "profile pass before CUDA graph capture"
            )
        ext = _load_exl3_ext()
        concurrency = int(ext.exl3_moe_max_concurrency(x.device.index))
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        num_experts = int(layer.local_num_experts)
        runtime = {
            "ext": ext,
            "max_batched_tokens": max_batched_tokens,
            "chunk": chunk,
            "topk": topk,
            "tg": torch.empty(
                (concurrency, chunk, hidden_size),
                dtype=torch.float16,
                device=x.device,
            ),
            "tu": torch.empty(
                (concurrency, chunk, hidden_size),
                dtype=torch.float16,
                device=x.device,
            ),
            "ig": torch.empty(
                (concurrency, chunk, intermediate_size),
                dtype=torch.float16,
                device=x.device,
            ),
            "iu": torch.empty(
                (concurrency, chunk, intermediate_size),
                dtype=torch.float16,
                device=x.device,
            ),
            "expert_count": torch.empty(
                num_experts + 1,
                dtype=torch.int64,
                device=x.device,
            ),
            "expert_offsets": torch.empty(
                num_experts + 1,
                dtype=torch.int64,
                device=x.device,
            ),
            "token_sorted": torch.empty(
                max_batched_tokens * topk,
                dtype=torch.int64,
                device=x.device,
            ),
            "weight_sorted": torch.empty(
                max_batched_tokens * topk,
                dtype=torch.float16,
                device=x.device,
            ),
        }
        _EXACT_RANK_SLICED_RUNTIMES[key] = runtime
        logger.info_once(
            "EXL3 exact MCG8 runtime allocated: capacity=%d chunk=%d "
            "concurrency=%d shared_across_layers=true",
            max_batched_tokens,
            chunk,
            concurrency,
        )
        return runtime

    def _apply_exact_rank_sliced_moe(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        bits = int(layer.exl3_trellis_bits)
        if bits != 8:
            raise RuntimeError(f"exact EXL3 MoE route requires MCG8, got MCG{bits}")
        runtime = self._exact_rank_sliced_runtime(layer, x, topk_ids)
        rows = int(x.shape[0])
        if rows > runtime["max_batched_tokens"]:
            raise ValueError(
                "Exact MCG8 EXL3 batch exceeds scheduler capacity: "
                f"m={rows}, capacity={runtime['max_batched_tokens']}"
            )
        return _exl3_moe_fused(
            x,
            topk_ids,
            topk_weights,
            layer.exl3_expert_map,
            runtime["expert_count"],
            runtime["expert_offsets"],
            runtime["token_sorted"],
            runtime["weight_sorted"],
            runtime["tg"],
            runtime["tu"],
            runtime["ig"],
            runtime["iu"],
            *layer.exl3_pointer_tables,
            bits,
        )

    def _mixed_rank_sliced_runtime(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        *,
        topk: int | None = None,
        route_ids_dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        mixed = layer.exl3_mixed_trellis
        if topk_ids is not None:
            topk = int(topk_ids.shape[1])
            route_ids_dtype = topk_ids.dtype
        if topk is None or route_ids_dtype is None:
            raise ValueError("mixed-bitrate EXL3 runtime requires route geometry")
        policy = mixed.get("runtime_policy")
        if policy is None:
            max_batched_tokens = int(layer.exl3_max_num_batched_tokens)
            max_decode_m = min(
                _positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32),
                max_batched_tokens,
            )
            prefill_capacity = _resolve_prefill_capacity(max_batched_tokens)
            configured_block_m = _positive_env_int("VLLM_EXL3_PREFILL_BLOCK_M", 64)
            explicit_block_m = bool(
                os.environ.get("VLLM_EXL3_PREFILL_BLOCK_M", "").strip()
            )
            tier_signature = tuple(
                (int(bits), len(ids))
                for bits, ids in zip(mixed["tier_bits"], mixed["tier_ids"], strict=True)
            )
            props = torch.cuda.get_device_properties(x.device)
            prefill_block_m = _resolve_mixed_trellis_prefill_block_m(
                configured_block_m=configured_block_m,
                explicit_override=explicit_block_m,
                hidden_size=int(layer.exl3_hidden_size),
                intermediate_size=int(layer.exl3_intermediate_size_per_partition),
                tier_signature=tier_signature,
                topk=topk,
                device_major=int(getattr(props, "major", 0)),
                prefill_tile_config=mixed["tile_config"],
            )
            prefill_tile_config = _resolve_mixed_trellis_prefill_tile_config(
                mixed["tile_config"]
            )
            policy = {
                "device_index": x.device.index,
                "topk": topk,
                "max_decode_m": max_decode_m,
                "max_batched_tokens": max_batched_tokens,
                "prefill_capacity": prefill_capacity,
                "prefill_block_m": prefill_block_m,
                "prefill_tile_config": prefill_tile_config,
                "tier_signature": tier_signature,
                "sms": int(props.multi_processor_count),
                "max_shared_mem": int(props.shared_memory_per_block_optin),
            }
            mixed["runtime_policy"] = policy
        elif policy["device_index"] != x.device.index or policy["topk"] != topk:
            raise RuntimeError(
                "mixed-bitrate EXL3 runtime geometry changed after planning"
            )

        tier_signature = policy["tier_signature"]
        owner_token = _runtime_owner_token(self.quant_config, layer)
        key = (
            owner_token,
            x.device.index,
            x.dtype,
            route_ids_dtype,
            int(layer.exl3_hidden_size),
            int(layer.exl3_intermediate_size_per_partition),
            tier_signature,
            topk,
            policy["max_decode_m"],
            policy["max_batched_tokens"],
            policy["prefill_capacity"],
            mixed["tile_config"],
            policy["prefill_tile_config"],
            policy["prefill_block_m"],
            bool(mixed["broadcast_suh"]),
            bool(mixed["broadcast_svh"]),
        )
        runtime = _MIXED_TRELLIS_RUNTIMES.get(key)
        if runtime is not None:
            mixed["runtime"] = runtime
            return runtime
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mixed-bitrate EXL3 runtime must be compiled during the eager "
                "profile pass before CUDA graph capture"
            )
        if route_ids_dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "mixed-bitrate EXL3 requires int32/int64 route IDs, got "
                f"{route_ids_dtype}"
            )

        mixed_api = _load_sparkinfer_mixed_trellis()
        total_experts = sum(experts for _, experts in tier_signature)

        def make_state(
            capacity: int,
            block_m: int,
            tile_config: tuple[int, int, int, int],
        ) -> dict[str, Any]:
            route_slots = mixed_api.max_packed_route_slots(
                capacity * topk, block_m, total_experts
            )
            launch = mixed_api.compile_mixed_trellis(
                size_m=capacity,
                hidden_size=int(layer.exl3_hidden_size),
                intermediate_size=int(layer.exl3_intermediate_size_per_partition),
                tier0_num_experts=tier_signature[0][1],
                tier1_num_experts=tier_signature[1][1],
                tier0_bits=tier_signature[0][0],
                tier1_bits=tier_signature[1][0],
                top_k=topk,
                max_m_blocks=(route_slots + block_m - 1) // block_m,
                moe_block_size=block_m,
                sms=policy["sms"],
                max_shared_mem=policy["max_shared_mem"],
                force_tile_config=tile_config,
                rotation_input_dtype=("bf16" if x.dtype == torch.bfloat16 else "fp16"),
                route_ids_dtype=route_ids_dtype,
                broadcast_suh=bool(mixed["broadcast_suh"]),
                broadcast_svh=bool(mixed["broadcast_svh"]),
            )
            return {
                "launch": launch,
                "buffers": _shared_mixed_buffers(
                    owner_token,
                    mixed_api,
                    launch,
                    x.device,
                    policy["sms"],
                ),
            }

        decode = make_state(
            policy["max_decode_m"],
            _MIXED_TRELLIS_ROUTE_BLOCK_SIZE,
            mixed["tile_config"],
        )
        prefill = None
        if policy["max_batched_tokens"] > policy["max_decode_m"]:
            if os.environ.get("VLLM_EXL3_PREFILL_TRELLIS", "1") != "1":
                raise ValueError("mixed-K EXL3 requires VLLM_EXL3_PREFILL_TRELLIS=1")
            prefill = make_state(
                policy["prefill_capacity"],
                policy["prefill_block_m"],
                policy["prefill_tile_config"],
            )
        runtime = {
            "mixed_api": mixed_api,
            "decode": decode,
            "prefill": prefill,
            **policy,
        }
        _MIXED_TRELLIS_RUNTIMES[key] = runtime
        mixed["runtime"] = runtime
        logger.info_once(
            "EXL3 mixed Trellis runtime planned: tiers=%s decode=%d tile=%s "
            "prefill=%d/%d tile=%s buffers=%.1f+%.1f MiB",
            tier_signature,
            policy["max_decode_m"],
            mixed["tile_config"],
            policy["prefill_capacity"],
            policy["max_batched_tokens"],
            policy["prefill_tile_config"],
            _unique_tensor_storage_bytes(decode["buffers"]) / (1 << 20),
            (
                0.0
                if prefill is None
                else _unique_tensor_storage_bytes(prefill["buffers"]) / (1 << 20)
            ),
        )
        return runtime

    def _apply_mixed_rank_sliced(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        runtime = self._mixed_rank_sliced_runtime(layer, x, topk_ids)
        rows = int(x.shape[0])
        if rows > runtime["max_batched_tokens"]:
            raise ValueError(
                "mixed-bitrate EXL3 batch exceeds planned capacity: "
                f"{rows} > {runtime['max_batched_tokens']}"
            )
        mixed = layer.exl3_mixed_trellis

        def run_state(
            slice_x: torch.Tensor,
            slice_weights: torch.Tensor,
            slice_ids: torch.Tensor,
            state: dict[str, Any],
        ) -> torch.Tensor:
            return (
                runtime["mixed_api"]
                .run_mixed_trellis(
                    slice_x,
                    mixed["tiers"][0],
                    mixed["tiers"][1],
                    slice_weights,
                    slice_ids,
                    mixed["global_to_combined"],
                    mixed["descriptor_map"],
                    mixed["rotations"],
                    state["launch"],
                    state["buffers"],
                )
                .to(slice_x.dtype)
            )

        if rows <= runtime["max_decode_m"]:
            return run_state(x, topk_weights, topk_ids, runtime["decode"])
        if runtime["prefill"] is None:
            raise RuntimeError("mixed-K EXL3 one-grid prefill plan is unavailable")
        capacity = int(runtime["prefill_capacity"])
        if rows <= capacity:
            return run_state(x, topk_weights, topk_ids, runtime["prefill"])
        output = torch.empty_like(x)
        for start in range(0, rows, capacity):
            stop = min(start + capacity, rows)
            output[start:stop].copy_(
                run_state(
                    x[start:stop],
                    topk_weights[start:stop],
                    topk_ids[start:stop],
                    runtime["prefill"],
                )
            )
        return output

    def _rank_sliced_runtime(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> dict[str, Any]:
        # A rank-sliced draft layer is CUDA-graph captured at small row counts
        # (m = draft rows per step, typically 1..3). If m falls outside the
        # Trellis window the eager parity path is reached, which is illegal
        # during capture -- the engine then cannot start at all:
        #
        #   RuntimeError: EXL3 eager parity path entered during CUDA graph
        #   capture (m=3); capture sizes must lie inside the Trellis window
        #   [4, 32]
        #
        # That was previously worked around by asking every operator to set
        # VLLM_EXL3_TRELLIS_MIN_M=1 by hand. A backend should satisfy its own
        # capture contract instead, so draft layers default the window down to
        # MIN_CAPTURABLE_TRELLIS_M. An explicit env value still wins, and the
        # target path keeps its original default.
        default_min_m = (
            MIN_CAPTURABLE_TRELLIS_M
            if _is_draft_layer(layer)
            else _DEFAULT_TRELLIS_MIN_M
        )
        min_trellis_m = _positive_env_int("VLLM_EXL3_TRELLIS_MIN_M", default_min_m)
        max_trellis_m = _positive_env_int("VLLM_EXL3_TRELLIS_MAX_M", 32)
        block_m = _positive_env_int("VLLM_EXL3_TRELLIS_BLOCK_M", 8)
        chunk = _positive_env_int("VLLM_EXL3_PREFILL_CHUNK", 128)
        prefill_trellis = os.environ.get("VLLM_EXL3_PREFILL_TRELLIS", "1") == "1"
        prefill_block_m = _positive_env_int("VLLM_EXL3_PREFILL_BLOCK_M", 64)
        mid_prefill_max_m = _positive_env_int("VLLM_EXL3_PREFILL_MID_MAX_M", 128)
        mid_prefill_block_m = _positive_env_int("VLLM_EXL3_PREFILL_MID_BLOCK_M", 8)
        small_prefill_max_m = _positive_env_int("VLLM_EXL3_PREFILL_SMALL_MAX_M", 512)
        small_prefill_block_m = _positive_env_int("VLLM_EXL3_PREFILL_SMALL_BLOCK_M", 32)
        if min_trellis_m > max_trellis_m:
            raise ValueError(
                "VLLM_EXL3_TRELLIS_MIN_M cannot exceed VLLM_EXL3_TRELLIS_MAX_M"
            )
        # Batch-INVARIANT capacity. Including x.shape[0] here made the runtime
        # cache key depend on the live batch, so any m above the planned capacity
        # produced a cache MISS -- silently planning a fresh ~1 GiB arena mid-serve
        # and making the capacity guard in _apply_rank_sliced unreachable. The
        # planned capacity is a property of the layer, not of one forward pass.
        max_batched_tokens = int(layer.exl3_max_num_batched_tokens)
        topk = int(topk_ids.shape[1])
        device_index = x.device.index
        key = (
            # Owning model scope first: the cached runtime holds mutable scratch,
            # so a target layer and a same-shape rank-sliced MTP draft layer must
            # not share an entry (see _runtime_scope_id).
            _runtime_owner_token(self.quant_config, layer),
            device_index,
            x.dtype,
            int(layer.exl3_hidden_size),
            int(layer.exl3_intermediate_size_per_partition),
            int(layer.local_num_experts),
            topk,
            max_batched_tokens,
            min_trellis_m,
            max_trellis_m,
            block_m,
            chunk,
            prefill_trellis,
            prefill_block_m,
            mid_prefill_max_m,
            mid_prefill_block_m,
            small_prefill_max_m,
            small_prefill_block_m,
            layer.exl3_trellis_tile_config,
        )
        bits = int(layer.exl3_trellis_bits)
        runtime = _RANK_SLICED_RUNTIMES.get(key)
        api = _load_sparkinfer_trellis()

        def _make_plan(plan_max_tokens: int, plan_block_m: int):
            caps = api.Caps(
                max_tokens=plan_max_tokens,
                num_topk=topk,
                num_experts=int(layer.local_num_experts),
                hidden_size=int(layer.exl3_hidden_size),
                intermediate_size=int(layer.exl3_intermediate_size_per_partition),
                route_num_experts=0,
                block_size_m=plan_block_m,
                input_dtype=x.dtype,
                device=x.device,
                trellis_bits=bits,
                tile_config=layer.exl3_trellis_tile_config,
            )
            return api.plan(caps)

        def _allocate_scratch(plan):
            scratch_spec = plan.scratch_specs()[0]
            return torch.empty(
                scratch_spec.shape,
                dtype=scratch_spec.dtype,
                device=scratch_spec.device,
            )

        def _build_plans() -> dict[str, Any]:
            if max_batched_tokens <= mid_prefill_max_m:
                main_prefill_block_m = mid_prefill_block_m
            elif max_batched_tokens <= small_prefill_max_m:
                main_prefill_block_m = small_prefill_block_m
            else:
                main_prefill_block_m = prefill_block_m
            plans: dict[str, Any] = {
                "trellis_plan": _make_plan(max_trellis_m, block_m),
                "mid_prefill_plan": None,
                "small_prefill_plan": None,
                "prefill_plan": None,
                "main_prefill_block_m": main_prefill_block_m,
            }
            if prefill_trellis and max_batched_tokens > max_trellis_m:
                plans["prefill_plan"] = _make_plan(
                    max_batched_tokens, main_prefill_block_m
                )
                if max_trellis_m < small_prefill_max_m < max_batched_tokens:
                    plans["small_prefill_plan"] = _make_plan(
                        small_prefill_max_m, small_prefill_block_m
                    )
                if max_trellis_m < mid_prefill_max_m < max_batched_tokens:
                    plans["mid_prefill_plan"] = _make_plan(
                        mid_prefill_max_m, mid_prefill_block_m
                    )
            return plans

        def _validate_shared_scratch(plan, scratch: torch.Tensor, label: str) -> None:
            if plan is None:
                return
            scratch_spec = plan.scratch_specs()[0]
            if (
                scratch_spec.dtype != scratch.dtype
                or torch.device(scratch_spec.device) != scratch.device
                or math.prod(scratch_spec.shape) > scratch.numel()
            ):
                raise RuntimeError(
                    f"EXL3 {label} plan cannot share its persistent scratch arena"
                )

        if runtime is not None:
            if bits not in runtime["plans_by_bits"]:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Every mixed-precision EXL3 bitrate must be planned "
                        "during the eager profile pass before CUDA graph capture"
                    )
                plans = _build_plans()
                _validate_shared_scratch(
                    plans["trellis_plan"], runtime["trellis_scratch"], "decode"
                )
                _validate_shared_scratch(
                    plans["prefill_plan"], runtime["prefill_scratch"], "prefill"
                )
                _validate_shared_scratch(
                    plans["small_prefill_plan"],
                    runtime["prefill_scratch"],
                    "small-prefill",
                )
                _validate_shared_scratch(
                    plans["mid_prefill_plan"],
                    runtime["prefill_scratch"],
                    "mid-prefill",
                )
                runtime["plans_by_bits"][bits] = plans
            return runtime

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Rank-sliced EXL3 runtime must be planned during the eager "
                "profile pass before CUDA graph capture"
            )
        plans = _build_plans()
        trellis_plan = plans["trellis_plan"]
        mid_prefill_plan = plans["mid_prefill_plan"]
        small_prefill_plan = plans["small_prefill_plan"]
        prefill_plan = plans["prefill_plan"]
        main_prefill_block_m = plans["main_prefill_block_m"]
        prefill_scratch = (
            None if prefill_plan is None else _allocate_scratch(prefill_plan)
        )
        if prefill_scratch is None:
            trellis_scratch = _allocate_scratch(trellis_plan)
        else:
            _validate_shared_scratch(trellis_plan, prefill_scratch, "decode")
            _validate_shared_scratch(
                small_prefill_plan, prefill_scratch, "small-prefill"
            )
            _validate_shared_scratch(mid_prefill_plan, prefill_scratch, "mid-prefill")
            trellis_scratch = prefill_scratch

        ext = _load_exl3_ext()
        required_ext = {
            "exl3_moe",
            "exl3_moe_max_concurrency",
        }
        missing_ext = sorted(name for name in required_ext if not hasattr(ext, name))
        if missing_ext:
            raise RuntimeError(
                "The EXL3 extension lacks routed-expert entry points: "
                + ", ".join(missing_ext)
            )
        concurrency = int(ext.exl3_moe_max_concurrency(torch.cuda.current_device()))
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        num_experts = int(layer.local_num_experts)
        device = x.device
        # With the prefill plan live, the parity path only ever serves
        # m < min_trellis_m, so its persistent staging shrinks to one chunk.
        parity_rows = (
            max_batched_tokens
            if prefill_plan is None
            else min(chunk, max_batched_tokens)
        )
        runtime = {
            "api": api,
            "plans_by_bits": {bits: plans},
            "trellis_plan": trellis_plan,
            "trellis_scratch": trellis_scratch,
            "mid_prefill_plan": mid_prefill_plan,
            "small_prefill_plan": small_prefill_plan,
            "prefill_plan": prefill_plan,
            "prefill_scratch": prefill_scratch,
            "ext": ext,
            "min_trellis_m": min_trellis_m,
            "max_trellis_m": max_trellis_m,
            "max_batched_tokens": max_batched_tokens,
            "small_prefill_max_m": small_prefill_max_m,
            "mid_prefill_max_m": mid_prefill_max_m,
            "parity_rows": parity_rows,
            "topk": topk,
            "chunk": chunk,
            "xh": torch.empty(
                (parity_rows, hidden_size),
                dtype=torch.float16,
                device=device,
            ),
            "out32": torch.empty(
                (parity_rows, hidden_size),
                dtype=torch.float32,
                device=device,
            ),
            "tg": torch.empty(
                (concurrency, chunk, hidden_size),
                dtype=torch.float16,
                device=device,
            ),
            "tu": torch.empty(
                (concurrency, chunk, hidden_size),
                dtype=torch.float16,
                device=device,
            ),
            "ig": torch.empty(
                (concurrency, chunk, intermediate_size),
                dtype=torch.float16,
                device=device,
            ),
            "iu": torch.empty(
                (concurrency, chunk, intermediate_size),
                dtype=torch.float16,
                device=device,
            ),
            "expert_count": torch.empty(
                num_experts + 1,
                dtype=torch.int64,
                device=device,
            ),
            "expert_offsets": torch.empty(
                num_experts + 1,
                dtype=torch.int64,
                device=device,
            ),
            "token_sorted": torch.empty(
                parity_rows * topk,
                dtype=torch.int64,
                device=device,
            ),
            "weight_sorted": torch.empty(
                parity_rows * topk,
                dtype=torch.float16,
                device=device,
            ),
            "flat_token": torch.arange(
                chunk,
                dtype=torch.int64,
                device=device,
            ).repeat_interleave(topk),
            "ones": torch.ones(
                chunk * topk,
                dtype=torch.int64,
                device=device,
            ),
        }
        _RANK_SLICED_RUNTIMES[key] = runtime
        prefill_arena_mib = (
            0.0
            if prefill_scratch is None
            else prefill_scratch.numel() * prefill_scratch.element_size() / (1 << 20)
        )
        logger.info_once(
            "EXL3 rank-sliced runtime planned: Trellis m=%d..%d block_m=%d, "
            "mid-prefill %s, small-prefill %s, prefill %s "
            "capacity=%d chunk=%d topk=%d",
            min_trellis_m,
            max_trellis_m,
            block_m,
            (
                f"m<={mid_prefill_max_m} block_m={mid_prefill_block_m}"
                if mid_prefill_plan is not None
                else "folded"
            ),
            (
                f"m<={small_prefill_max_m} block_m={small_prefill_block_m}"
                if small_prefill_plan is not None
                else "folded"
            ),
            (
                f"trellis block_m={main_prefill_block_m} "
                f"arena={prefill_arena_mib:.1f}MiB"
                if prefill_plan is not None
                else "parity"
            ),
            max_batched_tokens,
            chunk,
            topk,
        )
        return runtime

    def _apply_rank_sliced(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(layer, "exl3_shared_h_rotations", False):
            arena = layer.exl3_shared_h_arena.data
            hidden_size = int(layer.exl3_hidden_size)
            arena[:, :hidden_size].copy_(
                layer.exl3_shared_gate_suh.expand(-1, hidden_size)
            )
            arena[:, hidden_size : 2 * hidden_size].copy_(
                layer.exl3_shared_up_suh.expand(-1, hidden_size)
            )
            arena[:, 2 * hidden_size :].copy_(
                layer.exl3_shared_down_svh.expand(-1, hidden_size)
            )
        runtime = self._rank_sliced_runtime(layer, x, topk_ids)
        plans = runtime["plans_by_bits"][int(layer.exl3_trellis_bits)]
        if int(layer.exl3_trellis_weights.trellis_bits) != int(layer.exl3_trellis_bits):
            raise RuntimeError("EXL3 planned runtime bitrate drift")
        m = int(x.shape[0])
        if runtime["min_trellis_m"] <= m <= runtime["max_trellis_m"]:
            binding = runtime["api"].bind(
                plans["trellis_plan"],
                scratch=runtime["trellis_scratch"],
                a=x,
                weights=layer.exl3_trellis_weights,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
            output = runtime["api"].run(binding=binding)
            return output.to(x.dtype)

        if plans["prefill_plan"] is not None and m > runtime["max_trellis_m"]:
            if m > runtime["max_batched_tokens"]:
                raise ValueError(
                    "EXL3 batch exceeds its planned capacity: "
                    f"m={m}, capacity={runtime['max_batched_tokens']}"
                )
            prefill_plan = plans["prefill_plan"]
            if (
                plans["mid_prefill_plan"] is not None
                and m <= runtime["mid_prefill_max_m"]
            ):
                prefill_plan = plans["mid_prefill_plan"]
            elif (
                plans["small_prefill_plan"] is not None
                and m <= runtime["small_prefill_max_m"]
            ):
                prefill_plan = plans["small_prefill_plan"]
            binding = runtime["api"].bind(
                prefill_plan,
                scratch=runtime["prefill_scratch"],
                a=x,
                weights=layer.exl3_trellis_weights,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            )
            output = runtime["api"].run(binding=binding)
            return output.to(x.dtype)

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "EXL3 eager parity path entered during CUDA graph capture "
                f"(m={m}); capture sizes must lie inside the Trellis window "
                f"[{runtime['min_trellis_m']}, {runtime['max_trellis_m']}]. "
                f"Layer {getattr(layer, 'layer_name', '<unknown>')!r} was "
                f"classified as {'draft' if _is_draft_layer(layer) else 'target'}. "
                "A rank-sliced draft layer should have had its window widened to "
                f"MIN_CAPTURABLE_TRELLIS_M={MIN_CAPTURABLE_TRELLIS_M} "
                "automatically; if VLLM_EXL3_TRELLIS_MIN_M is set explicitly, "
                f"lower it to {MIN_CAPTURABLE_TRELLIS_M} or unset it."
            )
        if m > runtime["parity_rows"]:
            raise ValueError(
                "EXL3 batch exceeds its planned parity capacity: "
                f"m={m}, capacity={runtime['parity_rows']}"
            )
        bits = int(layer.exl3_trellis_bits)
        ext = runtime["ext"]
        xh = runtime["xh"][:m]
        xh.copy_(x)
        out32 = runtime["out32"][:m]
        out32.zero_()
        chunk = int(runtime["chunk"])
        pointer_args = layer.exl3_pointer_tables
        if m > chunk and hasattr(ext, "exl3_moe_fused"):
            ext.exl3_moe_fused(
                xh,
                out32,
                topk_ids,
                topk_weights,
                layer.exl3_expert_map,
                runtime["expert_count"],
                runtime["expert_offsets"],
                runtime["token_sorted"],
                runtime["weight_sorted"],
                runtime["tg"],
                runtime["tu"],
                runtime["ig"],
                runtime["iu"],
                0,
                bits,
                bits,
                bits,
                *pointer_args,
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
                0,
            )
            return out32.to(x.dtype)

        local_ids = layer.exl3_expert_map[topk_ids.long()]
        half_weights = topk_weights.to(torch.float16)
        topk = int(runtime["topk"])
        for start in range(0, m, chunk):
            current_m = min(chunk, m - start)
            flat = local_ids[start : start + current_m].reshape(-1)
            order = torch.argsort(flat)
            route_count = current_m * topk
            token_ids = runtime["flat_token"][:route_count].index_select(0, order)
            route_weights = (
                half_weights[start : start + current_m]
                .reshape(-1)
                .index_select(0, order)
            )
            counts = runtime["expert_count"]
            counts.zero_()
            counts.scatter_add_(0, flat, runtime["ones"][:route_count])
            ext.exl3_moe(
                xh[start : start + current_m],
                out32[start : start + current_m],
                counts,
                token_ids,
                route_weights,
                runtime["tg"],
                runtime["tu"],
                runtime["ig"],
                runtime["iu"],
                0,
                bits,
                bits,
                bits,
                *pointer_args,
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
                1,
            )
        return out32.to(x.dtype)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.activation != MoEActivation.SILU:
            raise NotImplementedError(
                f"EXL3 correctness MoE supports SiLU only, got {layer.activation}"
            )
        if layer.expert_map is not None:
            raise NotImplementedError("EXL3 MoE expert maps/EPLB are not supported")
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "EXL3 MoE does not support router weights applied on input"
            )

        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        ids = topk_ids.reshape(x_2d.shape[0], -1).contiguous()
        weights = topk_weights.reshape_as(ids).to(torch.float32).contiguous()
        if self.quant_config.rank_sliced_metadata is not None:
            if getattr(layer, "exl3_mixed_bitrate", False):
                output = self._apply_mixed_rank_sliced(layer, x_2d, weights, ids)
            elif layer.exl3_exact_rank_sliced_moe:
                output = self._apply_exact_rank_sliced_moe(layer, x_2d, weights, ids)
            else:
                output = self._apply_rank_sliced(layer, x_2d, weights, ids)
            return output.reshape(*original_shape, output.shape[-1])

        x_2d = x_2d.to(torch.float16)
        ids = ids.to(torch.long)
        weights = weights.to(torch.float16)
        output = torch.zeros(
            (x_2d.shape[0], layer.hidden_size),
            dtype=torch.float32,
            device=x.device,
        )
        for expert_id in range(layer.local_num_experts):
            positions = (ids == expert_id).nonzero(as_tuple=False)
            if positions.shape[0] == 0:
                continue
            token_ids = positions[:, 0]
            route_ids = positions[:, 1]
            expert_input = x_2d.index_select(0, token_ids)
            gate = self._apply_expert(layer, "w13", expert_input, expert_id, "w1")
            up = self._apply_expert(layer, "w13", expert_input, expert_id, "w3")
            hidden = torch.nn.functional.silu(gate) * up
            expert_output = self._apply_expert(layer, "w2", hidden, expert_id, "w2")
            route_weight = weights[token_ids, route_ids].unsqueeze(-1)
            output.index_add_(
                0,
                token_ids,
                (expert_output * route_weight).to(torch.float32),
            )
        output = output.reshape(*original_shape, output.shape[-1])
        return output if output.dtype == original_dtype else output.to(original_dtype)

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        if not self.is_monolithic:
            raise RuntimeError("EXL3 monolithic route-pack gate is disabled")
        correction_bias = layer.e_score_correction_bias
        valid_contract = _valid_fused_route_pack_contract(
            self.quant_config.rank_sliced_metadata,
            layer,
        )
        if not valid_contract:
            raise RuntimeError(
                "VLLM_EXL3_FUSED_ROUTE_PACK requires the exact mixed-K "
                "GLM E256/K8 sigmoid, bias, renormalize contract with "
                "runner-owned post-output scale=2.5"
            )
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        rows = int(x_2d.shape[0])
        if router_logits.shape != (rows, 256):
            raise ValueError(
                "EXL3 monolithic router logits must be [M,256], got "
                f"{tuple(router_logits.shape)}"
            )
        if rows > 9:
            from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
                grouped_topk,
            )

            topk_weights, topk_ids = grouped_topk(
                hidden_states=x_2d,
                gating_output=router_logits,
                topk=8,
                renormalize=True,
                num_expert_group=1,
                topk_group=1,
                scoring_func="sigmoid",
                routed_scaling_factor=1.0,
                e_score_correction_bias=correction_bias,
            )
            output = self.apply(
                layer,
                x_2d,
                topk_weights,
                topk_ids.to(torch.long),
                None,
                None,
            )
            return output.reshape(*original_shape, output.shape[-1])
        runtime = self._mixed_rank_sliced_runtime(
            layer,
            x_2d,
            topk=8,
            route_ids_dtype=torch.int64,
        )
        mixed = layer.exl3_mixed_trellis
        output = runtime["mixed_api"].run_mixed_trellis_monolithic(
            x_2d,
            router_logits,
            correction_bias,
            mixed["tiers"][0],
            mixed["tiers"][1],
            mixed["global_to_combined"],
            mixed["descriptor_map"],
            mixed["rotations"],
            runtime["decode"]["launch"],
            runtime["decode"]["buffers"],
            routed_scale=1.0,
        ).to(x_2d.dtype)
        return output.reshape(*original_shape, output.shape[-1])

    @staticmethod
    def _apply_expert(
        layer: RoutedExperts,
        group: str,
        x: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ) -> torch.Tensor:
        key = (expert_id, shard_id)
        trellis = getattr(layer, f"{group}_trellis").exl3_tensors[key]
        packed_k = trellis.shape[0] * 16
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"EXL3 MoE input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        if x.shape[-1] < packed_k:
            x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        output = _exl3_gemm(
            x,
            trellis,
            getattr(layer, f"{group}_suh").exl3_tensors[key],
            getattr(layer, f"{group}_svh").exl3_tensors[key],
            key in getattr(layer, f"{group}_mcg").exl3_tensors,
            key in getattr(layer, f"{group}_mul1").exl3_tensors,
        )
        logical_n = (
            layer.hidden_size
            if shard_id == "w2"
            else layer.exl3_intermediate_size_per_partition
        )
        if output.shape[-1] < logical_n:
            raise ValueError(
                f"EXL3 MoE packed N={output.shape[-1]} is below logical N={logical_n}"
            )
        return output[..., :logical_n]


def get_exl3_mcg8_mla_packed_weights(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Return the one supported local packed MLA projection or fail closed."""
    method = getattr(layer, "quant_method", None)
    if not isinstance(method, Exl3LinearMethod):
        return None

    prefix = getattr(layer, "prefix", layer.__class__.__name__)
    metadata = {
        "input size": getattr(layer, "exl3_input_size", None),
        "local input size": getattr(layer, "exl3_input_size_per_partition", None),
        "local output partitions": getattr(layer, "exl3_output_partition_sizes", None),
        "parallel mode": getattr(layer, "exl3_parallel_mode", None),
        "storage shards": getattr(layer, "exl3_shard_ids", None),
        "codebook": getattr(layer, "exl3_expected_codebooks", {}).get(None),
    }
    expected_metadata = {
        "input size": 512,
        "local input size": 512,
        "local output partitions": [7168],
        "parallel mode": "column",
        "storage shards": [None],
        "codebook": "mcg",
    }
    mismatches = [
        f"{name}={value!r} (expected {expected_metadata[name]!r})"
        for name, value in metadata.items()
        if value != expected_metadata[name]
    ]
    if mismatches:
        raise ValueError(
            f"Unsupported EXL3 MCG8 MLA geometry for {prefix}: " + ", ".join(mismatches)
        )

    stores: dict[str, dict[ShardId, torch.Tensor]] = {}
    for name in ("trellis", "suh", "svh", "mcg", "mul1"):
        store = getattr(getattr(layer, name, None), "exl3_tensors", None)
        if not isinstance(store, dict):
            raise ValueError(
                f"Unsupported EXL3 MCG8 MLA geometry for {prefix}: "
                f"missing {name} tensor store."
            )
        stores[name] = store
    expected_store_keys = {
        "trellis": {None},
        "suh": {None},
        "svh": {None},
        "mcg": {None},
        "mul1": set(),
    }
    invalid_store_keys = [
        f"{name}={set(store)!r}"
        for name, store in stores.items()
        if set(store) != expected_store_keys[name]
    ]
    if invalid_store_keys:
        raise ValueError(
            f"Unsupported EXL3 MCG8 MLA local shards for {prefix}: "
            + ", ".join(invalid_store_keys)
        )
    Exl3LinearMethod._validate_marker(stores["mcg"][None], _MCG_SENTINEL, "mcg")

    trellis = stores["trellis"][None]
    suh = stores["suh"].get(None)
    svh = stores["svh"].get(None)
    expected_tensors = (
        (trellis, (32, 448, 128), torch.int16, "trellis"),
        (suh, (512,), torch.float16, "suh"),
        (svh, (7168,), torch.float16, "svh"),
    )
    for tensor, shape, dtype, name in expected_tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            actual = (
                f"{getattr(tensor, 'dtype', None)} "
                f"{tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None}"
            )
            raise ValueError(
                f"Unsupported EXL3 MCG8 MLA {name} for {prefix}: expected "
                f"contiguous {dtype} {shape}, got {actual}."
            )
    assert isinstance(suh, torch.Tensor)
    assert isinstance(svh, torch.Tensor)
    if suh.device != trellis.device or svh.device != trellis.device:
        raise ValueError(
            f"Unsupported EXL3 MCG8 MLA placement for {prefix}: packed tensors "
            "must share one device."
        )

    tp_rank = getattr(layer, "exl3_tp_rank", None)
    if not isinstance(tp_rank, int) or isinstance(tp_rank, bool) or tp_rank < 0:
        raise ValueError(
            f"Unsupported EXL3 MCG8 MLA TP rank for {prefix}: {tp_rank!r}."
        )
    ext = _load_exl3_ext()
    missing_symbols = [
        name
        for name in ("exl3_mcg8_mla_query_adj", "exl3_mcg8_mla_value")
        if not hasattr(ext, name)
    ]
    if missing_symbols:
        raise RuntimeError(
            "The imported exllamav3_ext lacks the fixed packed MLA ABI: "
            + ", ".join(missing_symbols)
            + ". Rebuild the canonical extension."
        )
    return trellis, suh, svh, 0


def warmup_exl3_mixed_trellis_route_pack(model: torch.nn.Module) -> int:
    """Materialize mixed-Trellis route-pack modules before KV sizing."""
    warmed = 0
    seen_layers: set[int] = set()
    for module in model.modules():
        routed_experts = getattr(module, "routed_experts", module)
        mixed = getattr(routed_experts, "exl3_mixed_trellis", None)
        if not isinstance(mixed, dict) or id(routed_experts) in seen_layers:
            continue
        seen_layers.add(id(routed_experts))

        runtime = mixed.get("runtime")
        if runtime is None:
            layer_name = getattr(routed_experts, "layer_name", "<unknown>")
            raise RuntimeError(
                "mixed-bitrate EXL3 route-pack warmup found no runtime for "
                f"{layer_name}; the eager profile pass must plan the runtime "
                "before kernel warmup"
            )
        api = runtime["mixed_api"]
        warmup = getattr(api, "warmup_mixed_trellis_route_pack", None)
        if not callable(warmup):
            raise RuntimeError(
                "mixed-bitrate EXL3 route-pack warmup requires a matching "
                "SparkInfer build"
            )

        for state_name in ("decode", "prefill"):
            state = runtime.get(state_name)
            if state is None:
                continue
            fused_kwargs: dict[str, Any] = {}
            if (
                state_name == "decode"
                and os.environ.get("VLLM_EXL3_FUSED_ROUTE_PACK", "0") == "1"
            ):
                if routed_experts.e_score_correction_bias is None:
                    raise RuntimeError(
                        "enabled EXL3 fused route-pack warmup requires correction bias"
                    )
                fused_kwargs = {
                    "correction_bias": routed_experts.e_score_correction_bias,
                    "routed_scale": float(routed_experts.routed_scaling_factor),
                }
            warmed += int(
                warmup(
                    state["launch"],
                    state["buffers"],
                    expert_map=mixed["global_to_combined"],
                    **fused_kwargs,
                )
            )
    return warmed


__all__ = [
    "Exl3Config",
    "Exl3LinearMethod",
    "Exl3MoEMethod",
    "warmup_exl3_mixed_trellis_route_pack",
]
