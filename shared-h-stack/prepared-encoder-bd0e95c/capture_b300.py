#!/usr/bin/env python3
"""TP8 BF16 calibration capture for the all-EXL3 GLM-5.2 B300 build.

The payload ABI intentionally matches the proven TR3 encoder (BF16 x rows plus
natural top-8 expert IDs), but the output directory MUST be a RAM-backed tmpfs.
At most eight capture-data layers are striped across TP ranks per window.
No CPU model offload, low-memory mode, forced expert activation, or transformers
model loading is used.  Transformers is used only for tokenization in this
capture environment; the encoder has no transformers dependency.
"""

from __future__ import annotations

import argparse
from collections import Counter
import faulthandler
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
import time
import uuid


try:
    faulthandler.register(signal.SIGUSR2, all_threads=True)
except (AttributeError, RuntimeError):
    pass


CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
CORPUS_ROWS = 12_228
AXES = (
    "axis1_general",
    "axis2_legal",
    "axis3_code_agentic",
    "axis4_reasoning_termination",
)
AXIS_ROWS = 3_057

SEED = 20260711
TARGET_TOKENS = 1_048_576  # one GLOBAL budget across all passes
MAX_SAMPLE_TOKENS = 4_096
MIN_SAMPLE_TOKENS = 8

HIDDEN = 6_144
NUM_EXPERTS = 256
TOPK = 8
FIRST_MOE_LAYER = 3
NUM_LAYERS = 78
MOE_LAYERS = list(range(FIRST_MOE_LAYER, NUM_LAYERS))

CAPTURE_TP = 8             # BF16 model residency on 8x B300
OUTPUT_TP = 4              # manifest reminder; encoded format remains TP4
MAX_MODEL_LEN = 4_352
MAX_NUM_BATCHED_TOKENS = 8_192
DEFAULT_KV_CACHE_BYTES = 4 << 30
DEFAULT_MAX_NUM_SEQS = 8
DEFAULT_MIN_POST_LOAD_FREE_BYTES = 48 << 30
GIB = 1 << 30
DEFAULT_DISK_RESERVE_BYTES = 256 * GIB
DEFAULT_RAMFS_RESERVE_BYTES = 64 * GIB
DEFAULT_HOST_RAM_RESERVE_BYTES = 128 * GIB
DEFAULT_MAX_CAPTURE_WINDOW_LAYERS = 8
METADATA_WRITE_ALLOWANCE_BYTES = 1 * GIB
RAM_FILESYSTEMS = {"tmpfs", "ramfs"}


def _existing_ancestor(path: Path) -> Path:
    current = path.expanduser().resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"no existing filesystem ancestor for {path}")
        current = current.parent
    return current


def filesystem_type(path: Path) -> str:
    anchor = _existing_ancestor(path)
    best = (0, "unknown")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        mountpoint = fields[4]
        for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            mountpoint = mountpoint.replace(escaped, plain)
        mount = Path(mountpoint)
        try:
            anchor.relative_to(mount)
        except ValueError:
            continue
        if len(mountpoint) >= best[0]:
            best = (len(mountpoint), fields[separator + 1])
    return best[1]


def assert_disk_free(
    path: Path, write_bytes: int, label: str, *, quiet: bool = False
) -> dict:
    """Preserve overlay headroom before every metadata/log write."""
    anchor = _existing_ancestor(path)
    fs_type = filesystem_type(anchor)
    reserve = int(os.environ.get("B300_DISK_RESERVE_BYTES", DEFAULT_DISK_RESERVE_BYTES))
    usage = shutil.disk_usage(anchor)
    required = int(write_bytes) + (0 if fs_type in RAM_FILESYSTEMS else reserve)
    if usage.free < required:
        raise RuntimeError(
            f"DISK GUARD {label}: {usage.free} bytes free on {fs_type} at {anchor}; "
            f"need {write_bytes} write bytes + {required - write_bytes} reserve"
        )
    result = {
        "target": str(path),
        "filesystem": fs_type,
        "free_bytes": usage.free,
        "write_allowance_bytes": int(write_bytes),
        "reserve_bytes": required - int(write_bytes),
    }
    if not quiet:
        print(f"DISK GUARD PASS: {label}: {result}", flush=True)
    return result


def _meminfo_bytes(key: str) -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"{key} absent from /proc/meminfo")


def _cgroup_memory_headroom() -> int | None:
    root = Path("/sys/fs/cgroup")
    maximum, current = root / "memory.max", root / "memory.current"
    if not maximum.is_file() or not current.is_file():
        return None
    raw = maximum.read_text().strip()
    if raw == "max":
        return None
    return max(0, int(raw) - int(current.read_text().strip()))


def active_swap_bytes() -> int:
    lines = Path("/proc/swaps").read_text().splitlines()[1:]
    return sum(int(line.split()[3]) * 1024 for line in lines if line.split())


def assert_ram_capture_target(path: Path, write_bytes: int, label: str) -> dict:
    anchor = _existing_ancestor(path)
    fs_type = filesystem_type(anchor)
    if fs_type not in RAM_FILESYSTEMS:
        raise RuntimeError(
            f"RAM CAPTURE GUARD {label}: {path} resolves to {fs_type}, not tmpfs/ramfs"
        )
    ramfs_reserve = int(
        os.environ.get("B300_RAMFS_RESERVE_BYTES", DEFAULT_RAMFS_RESERVE_BYTES)
    )
    host_reserve = int(
        os.environ.get("B300_HOST_RAM_RESERVE_BYTES", DEFAULT_HOST_RAM_RESERVE_BYTES)
    )
    fs_free = shutil.disk_usage(anchor).free
    mem_available = _meminfo_bytes("MemAvailable")
    required_fs = int(write_bytes) + ramfs_reserve
    required_host = int(write_bytes) + host_reserve
    if fs_free < required_fs:
        raise RuntimeError(
            f"RAM CAPTURE GUARD {label}: tmpfs free={fs_free}, "
            f"need payload={write_bytes} + reserve={ramfs_reserve}"
        )
    if mem_available < required_host:
        raise RuntimeError(
            f"RAM CAPTURE GUARD {label}: MemAvailable={mem_available}, "
            f"need payload={write_bytes} + reserve={host_reserve}"
        )
    cgroup_headroom = _cgroup_memory_headroom()
    if cgroup_headroom is not None and cgroup_headroom < required_host:
        raise RuntimeError(
            f"RAM CAPTURE GUARD {label}: cgroup headroom={cgroup_headroom}, "
            f"need payload={write_bytes} + reserve={host_reserve}"
        )
    swap_used = active_swap_bytes()
    result = {
        "target": str(path),
        "filesystem": fs_type,
        "tmpfs_free_bytes": fs_free,
        "payload_write_bytes": int(write_bytes),
        "tmpfs_reserve_bytes": ramfs_reserve,
        "mem_available_bytes": mem_available,
        "host_reserve_bytes": host_reserve,
        "cgroup_headroom_bytes": cgroup_headroom,
        "active_swap_bytes": swap_used,
        "swap_policy": "small capture-data windows plus host-RAM reserve; no overlay capture",
    }
    print(f"RAM CAPTURE GUARD PASS: {label}: {result}", flush=True)
    return result


def log(message: str, logfile: str | None = None) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    if logfile:
        assert_disk_free(Path(logfile), METADATA_WRITE_ALLOWANCE_BYTES, "capture log", quiet=True)
        with open(logfile, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_file(path: str | Path, chunk: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    encoded_size = len(json.dumps(payload, sort_keys=True).encode()) + 1
    assert_disk_free(path, max(encoded_size * 2, 1 << 20), "capture JSON", quiet=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    out = sorted(set(layers))
    if not out or any(layer not in MOE_LAYERS for layer in out):
        raise ValueError(f"capture layers must be a nonempty subset of 3..77, got {spec!r}")
    return out


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tokenizer_identity(src: Path, tokenizer: object) -> dict:
    files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ):
        path = src / name
        if path.is_file():
            files[name] = sha256_file(path)
    return {
        "class": type(tokenizer).__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "files_sha256": files,
    }


def _waterfill_quotas(capacities: dict[str, int], target: int) -> dict[str, int]:
    """Equal-token quotas with no sample repetition; redistribute short-axis slack."""
    if sum(capacities.values()) < target:
        raise RuntimeError(
            f"owner corpus has only {sum(capacities.values())} eligible truncated tokens, "
            f"below target {target}"
        )
    quotas = {axis: 0 for axis in capacities}
    remaining_axes = list(capacities)
    remaining = target
    while remaining_axes:
        share = math.ceil(remaining / len(remaining_axes))
        short = [axis for axis in remaining_axes if capacities[axis] < share]
        if not short:
            base, extra = divmod(remaining, len(remaining_axes))
            for index, axis in enumerate(remaining_axes):
                quotas[axis] = base + int(index < extra)
            remaining = 0
            break
        for axis in short:
            quotas[axis] = capacities[axis]
            remaining -= quotas[axis]
            remaining_axes.remove(axis)
    if remaining != 0 or sum(quotas.values()) != target:
        raise AssertionError((remaining, quotas, target))
    return quotas


def build_capture_plan(src: Path, corpus: Path, target_tokens: int, logfile: str) -> dict:
    """Build the owner-corpus-only deterministic Luke-style multi-pass manifest."""
    from transformers import AutoTokenizer

    corpus_digest = sha256_file(corpus)
    if corpus_digest != CORPUS_SHA256:
        raise RuntimeError(f"owner corpus sha256 mismatch: {corpus_digest}")

    records: list[dict] = []
    counts: Counter[str] = Counter()
    with corpus.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            axis = record.get("axis")
            text = record.get("text")
            if axis not in AXES or not isinstance(text, str):
                raise RuntimeError(f"corpus line {line_no + 1}: invalid axis/text")
            records.append({"line": line_no, "axis": axis, "text": text})
            counts[axis] += 1
    if len(records) != CORPUS_ROWS or counts != Counter({axis: AXIS_ROWS for axis in AXES}):
        raise RuntimeError(f"owner corpus row/axis mismatch: rows={len(records)} axes={dict(counts)}")

    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=False)
    by_axis: dict[str, list[tuple[int, int]]] = {axis: [] for axis in AXES}
    skipped_short = 0
    t0 = time.time()
    for record in records:
        length = min(len(tokenizer.encode(record["text"])), MAX_SAMPLE_TOKENS)
        if length < MIN_SAMPLE_TOKENS:
            skipped_short += 1
            continue
        by_axis[record["axis"]].append((record["line"], length))
    capacities = {axis: sum(length for _, length in rows) for axis, rows in by_axis.items()}
    quotas = _waterfill_quotas(capacities, target_tokens)

    passes = []
    selected_lines: set[int] = set()
    total_tokens = 0
    for axis_index, axis in enumerate(AXES):
        candidates = list(by_axis[axis])
        random.Random(SEED + 1_000_003 * axis_index).shuffle(candidates)
        chosen = []
        pass_tokens = 0
        for line_no, length in candidates:
            if pass_tokens >= quotas[axis]:
                break
            if line_no in selected_lines:
                raise AssertionError(f"line selected twice: {line_no}")
            selected_lines.add(line_no)
            chosen.append({"line": line_no, "ntok": length})
            pass_tokens += length
        if pass_tokens < quotas[axis]:
            raise RuntimeError(f"axis {axis} exhausted at {pass_tokens} < quota {quotas[axis]}")
        passes.append(
            {
                "name": axis,
                "axis": axis,
                "quota_tokens": quotas[axis],
                "tokens": pass_tokens,
                "samples": chosen,
            }
        )
        total_tokens += pass_tokens

    source_identity = {
        "config_sha256": sha256_file(src / "config.json"),
        "index_sha256": sha256_file(src / "model.safetensors.index.json"),
    }
    plan = {
        "schema": "glm52-b300-capture-plan-v1",
        "selection_policy": "owner-corpus-axis-separated-luke-multipass-no-repeat-v1",
        "selection_note": (
            "Only the owner-pinned 12,228-row corpus is eligible. One natural-routing pass "
            "is sealed per owner axis; rows never repeat and no stock dataset is allowed."
        ),
        "owner_corpus_only": True,
        "calibration_baseline": True,
        "corpus_sha256": corpus_digest,
        "corpus_rows": len(records),
        "axis_rows": dict(sorted(counts.items())),
        "seed": SEED,
        "target_tokens": target_tokens,
        "max_sample_tokens": MAX_SAMPLE_TOKENS,
        "min_sample_tokens": MIN_SAMPLE_TOKENS,
        "skipped_short": skipped_short,
        "passes": passes,
        "total_tokens": total_tokens,
        "capture_bytes": total_tokens * len(MOE_LAYERS) * (HIDDEN * 2 + TOPK),
        "source": source_identity,
        "tokenizer": _tokenizer_identity(src, tokenizer),
        "capture_tp": CAPTURE_TP,
        "output_tp": OUTPUT_TP,
        "routing": {
            "natural": True,
            "forced_expert_activation": False,
            "scoring_func": "sigmoid",
            "top_k": TOPK,
            "n_group": 1,
            "topk_group": 1,
        },
    }
    plan["capture_fingerprint"] = canonical_hash(plan)
    log(
        f"capture plan: {len(passes)} axis passes, {sum(len(p['samples']) for p in passes)} "
        f"samples, {total_tokens} tokens (global target {target_tokens}), "
        f"{plan['capture_bytes']/2**30:.2f} GiB for all 75 layers; "
        f"capacities={capacities}, quotas={quotas}, tokenize={time.time()-t0:.1f}s",
        logfile,
    )
    return plan


def validate_plan(plan: dict, src: Path, corpus: Path) -> None:
    if plan.get("schema") != "glm52-b300-capture-plan-v1":
        raise RuntimeError(f"unexpected capture plan schema: {plan.get('schema')!r}")
    claimed = plan.get("capture_fingerprint")
    canonical = dict(plan)
    canonical.pop("capture_fingerprint", None)
    actual = canonical_hash(canonical)
    if claimed != actual:
        raise RuntimeError(f"capture plan fingerprint mismatch: {claimed} != {actual}")
    if sha256_file(corpus) != plan.get("corpus_sha256") or claimed is None:
        raise RuntimeError("capture plan does not match the supplied owner corpus")
    current_source = {
        "config_sha256": sha256_file(src / "config.json"),
        "index_sha256": sha256_file(src / "model.safetensors.index.json"),
    }
    if current_source != plan.get("source"):
        raise RuntimeError(f"capture plan source mismatch: {current_source} != {plan.get('source')}")
    if int(plan.get("capture_tp", -1)) != CAPTURE_TP or int(plan.get("output_tp", -1)) != OUTPUT_TP:
        raise RuntimeError("capture/output TP in plan does not match TP8 capture / TP4 format")
    if plan.get("selection_policy") != "owner-corpus-axis-separated-luke-multipass-no-repeat-v1":
        raise RuntimeError("capture plan is not the owner-corpus Luke-style baseline")
    if plan.get("owner_corpus_only") is not True:
        raise RuntimeError("capture plan does not declare owner-corpus-only calibration")
    if plan.get("calibration_baseline") is not True:
        raise RuntimeError("capture plan does not declare mandatory baseline calibration")
    routing = plan.get("routing", {})
    expected_routing = {
        "natural": True,
        "forced_expert_activation": False,
        "scoring_func": "sigmoid",
        "top_k": TOPK,
        "n_group": 1,
        "topk_group": 1,
    }
    if routing != expected_routing:
        raise RuntimeError(f"capture plan routing {routing} != mandatory {expected_routing}")
    passes = plan.get("passes", [])
    if not passes:
        raise RuntimeError("capture plan contains no passes")
    if [item.get("axis") for item in passes] != list(AXES):
        raise RuntimeError("Luke-style baseline must contain one ordered pass per owner axis")
    lines = [sample["line"] for item in passes for sample in item["samples"]]
    if len(lines) != len(set(lines)):
        raise RuntimeError("capture plan repeats corpus rows")
    if sum(int(item["tokens"]) for item in passes) != int(plan["total_tokens"]):
        raise RuntimeError("capture plan per-pass token sum mismatch")


def load_plan_tokens(plan: dict, src: Path, corpus: Path, logfile: str) -> list[tuple[str, list[list[int]]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=False)
    if _tokenizer_identity(src, tokenizer) != plan.get("tokenizer"):
        raise RuntimeError("tokenizer identity changed since capture plan was built")
    raw_lines = corpus.read_text(encoding="utf-8").splitlines()
    out = []
    for pass_info in plan["passes"]:
        token_lists = []
        for sample in pass_info["samples"]:
            line_no = int(sample["line"])
            record = json.loads(raw_lines[line_no])
            ids = tokenizer.encode(record["text"])[:MAX_SAMPLE_TOKENS]
            if len(ids) != int(sample["ntok"]):
                raise RuntimeError(
                    f"tokenization drift at corpus line {line_no}: {len(ids)} != {sample['ntok']}"
                )
            token_lists.append(ids)
        tokens = sum(map(len, token_lists))
        if tokens != int(pass_info["tokens"]):
            raise RuntimeError(f"pass {pass_info['name']}: token sum drift {tokens}")
        log(f"prepared pass {pass_info['name']}: {len(token_lists)} samples, {tokens} tokens", logfile)
        out.append((str(pass_info["name"]), token_lists))
    return out


class CaptureWorkerExtension:
    """vLLM worker extension.  Every layer is owned by exactly one TP rank."""

    def _tr3_rank(self) -> int:
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            return int(get_tensor_model_parallel_rank())
        except Exception:
            return int(getattr(self, "rank", 0))

    def _tr3_world(self) -> int:
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

            return int(get_tensor_model_parallel_world_size())
        except Exception:
            return CAPTURE_TP

    def _tr3_model(self):
        runner = getattr(self, "model_runner", None)
        if runner is None:
            raise RuntimeError("vLLM worker has no model_runner")
        return runner.get_model() if hasattr(runner, "get_model") else runner.model

    def tr3_cuda_memory_status(self) -> dict:
        import torch

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "rank": self._tr3_rank(),
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }

    def tr3_capture_init(
        self,
        out_dir: str,
        verify_mode: bool,
        active_layers: list[int],
        capture_fingerprint: str,
        expected_tokens: int,
    ) -> dict:
        import torch
        import torch.nn.functional as torch_f

        rank = self._tr3_rank()
        world = self._tr3_world()
        if world != CAPTURE_TP:
            raise RuntimeError(f"capture world size {world} != required TP{CAPTURE_TP}")
        active_layers = sorted(int(layer) for layer in active_layers)
        assigned = active_layers[rank::world]
        assert_ram_capture_target(
            Path(out_dir),
            int(expected_tokens) * len(assigned) * (HIDDEN * 2 + TOPK),
            f"rank {rank} assigned capture",
        )
        model = self._tr3_model()

        experts_mods: dict[int, torch.nn.Module] = {}
        mlp_mods: dict[int, torch.nn.Module] = {}
        pat = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
        for name, module in model.named_modules():
            match = pat.search(name)
            if match:
                experts_mods[int(match.group(1))] = module
        pat_mlp = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp$")
        for name, module in model.named_modules():
            match = pat_mlp.search(name)
            if match and int(match.group(1)) in experts_mods:
                mlp_mods[int(match.group(1))] = module
        missing = [layer for layer in active_layers if layer not in experts_mods or layer not in mlp_mods]
        if missing:
            raise RuntimeError(f"MoE modules missing for capture layers {missing}")

        config_obj = getattr(model, "config", None) or getattr(model, "model_config", None)
        hf_config = getattr(config_obj, "hf_config", config_obj)
        routing = {
            "n_group": getattr(hf_config, "n_group", None),
            "topk_group": getattr(hf_config, "topk_group", None),
            "scoring_func": getattr(hf_config, "scoring_func", None),
            "top_k": getattr(
                hf_config,
                "num_experts_per_tok",
                getattr(hf_config, "top_k", None),
            ),
        }
        expected_routing = {"n_group": 1, "topk_group": 1, "scoring_func": "sigmoid", "top_k": TOPK}
        if routing != expected_routing:
            raise RuntimeError(f"routing config {routing} != {expected_routing}")

        self._tr3_gate_w = {}
        self._tr3_gate_b = {}
        for layer in assigned:
            mlp = mlp_mods[layer]
            if bool(getattr(mlp, "is_sequence_parallel", False)):
                raise RuntimeError(
                    f"layer {layer}: sequence-parallel MoE would expose only a token shard; disable it"
                )
            gate = getattr(mlp, "gate", None)
            if gate is None or not hasattr(gate, "weight"):
                raise RuntimeError(f"layer {layer}: mlp.gate.weight absent")
            weight = gate.weight
            bias = getattr(gate, "e_score_correction_bias", None)
            if tuple(weight.shape) != (NUM_EXPERTS, HIDDEN) or bias is None:
                raise RuntimeError(f"layer {layer}: unexpected router weight/bias shape")
            self._tr3_gate_w[layer] = weight.detach().to(torch.float32)
            self._tr3_gate_b[layer] = bias.detach().to(torch.float32).flatten()
            if tuple(self._tr3_gate_b[layer].shape) != (NUM_EXPERTS,):
                raise RuntimeError(f"layer {layer}: router correction bias shape mismatch")

        self._tr3_files_x = {}
        self._tr3_files_i = {}
        self._tr3_hash_x = {}
        self._tr3_hash_i = {}
        self._tr3_counts = {layer: 0 for layer in assigned}
        self._tr3_routed = {layer: [0] * NUM_EXPERTS for layer in assigned}
        self._tr3_verify = bool(verify_mode)
        self._tr3_verify_cmp = {"checked_tokens": 0, "mismatch_tokens": 0, "kwarg_logits_layers": 0}
        self._tr3_capture_fingerprint = capture_fingerprint
        for layer in assigned:
            layer_dir = Path(out_dir) / f"layer_{layer:03d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            self._tr3_files_x[layer] = open(layer_dir / "x.bin.partial", "wb", buffering=16 << 20)
            self._tr3_files_i[layer] = open(layer_dir / "ids.bin.partial", "wb", buffering=1 << 20)
            self._tr3_hash_x[layer] = hashlib.sha256()
            self._tr3_hash_i[layer] = hashlib.sha256()

        self._tr3_handles = []

        def make_hook(layer: int):
            gate_w = self._tr3_gate_w[layer]
            gate_b = self._tr3_gate_b[layer]

            def hook(module, args, kwargs):
                hidden = kwargs.get("hidden_states")
                if hidden is None and args:
                    hidden = args[0]
                if hidden is None or hidden.dim() != 2 or hidden.shape[1] != HIDDEN:
                    raise RuntimeError(
                        f"layer {layer}: unexpected MoE input "
                        f"{None if hidden is None else tuple(hidden.shape)}"
                    )
                rows = int(hidden.shape[0])
                logits = torch_f.linear(hidden.to(torch.float32), gate_w)
                scores = torch.sigmoid(logits) + gate_b
                ids = torch.topk(scores, TOPK, dim=-1, sorted=False).indices

                if self._tr3_verify:
                    router_logits = kwargs.get("router_logits")
                    if router_logits is None and len(args) > 1:
                        router_logits = args[1]
                    if (
                        router_logits is not None
                        and router_logits.dim() == 2
                        and router_logits.shape[1] == NUM_EXPERTS
                        and router_logits.data_ptr() != hidden.data_ptr()
                    ):
                        scores_2 = torch.sigmoid(router_logits.to(torch.float32)) + gate_b
                        ids_2 = torch.topk(scores_2, TOPK, dim=-1, sorted=False).indices
                        bad = (ids.sort(dim=-1).values != ids_2.sort(dim=-1).values).any(dim=-1)
                        self._tr3_verify_cmp["checked_tokens"] += rows
                        self._tr3_verify_cmp["mismatch_tokens"] += int(bad.sum().item())
                        self._tr3_verify_cmp["kwarg_logits_layers"] += 1

                ids_cpu = ids.to(torch.uint8).cpu().contiguous()
                hidden_cpu = hidden.detach().to(torch.bfloat16).cpu().contiguous()
                x_bytes = hidden_cpu.view(torch.int16).numpy().tobytes()
                id_bytes = ids_cpu.numpy().tobytes()
                self._tr3_hash_x[layer].update(x_bytes)
                self._tr3_hash_i[layer].update(id_bytes)
                self._tr3_files_x[layer].write(x_bytes)
                self._tr3_files_i[layer].write(id_bytes)
                self._tr3_counts[layer] += rows

                bincount = torch.bincount(ids.flatten().to(torch.int64), minlength=NUM_EXPERTS)
                routed = bincount.cpu().tolist()
                for expert, value in enumerate(routed):
                    self._tr3_routed[layer][expert] += int(value)
                return None

            return hook

        for layer in assigned:
            self._tr3_handles.append(
                experts_mods[layer].register_forward_pre_hook(make_hook(layer), with_kwargs=True)
            )
        return {
            "rank": rank,
            "world": world,
            "hooks": len(self._tr3_handles),
            "assigned_layers": assigned,
            "routing": routing,
            "gate_dtype": str(mlp_mods[assigned[0]].gate.weight.dtype) if assigned else None,
        }

    def tr3_capture_status(self) -> dict:
        return {
            "rank": self._tr3_rank(),
            "counts": {str(layer): count for layer, count in getattr(self, "_tr3_counts", {}).items()},
        }

    def tr3_capture_finalize(self) -> dict:
        if not hasattr(self, "_tr3_counts"):
            return {"rank": self._tr3_rank(), "layers": {}}
        for handle in self._tr3_handles:
            handle.remove()
        layers = {}
        for layer in sorted(self._tr3_counts):
            x_handle = self._tr3_files_x[layer]
            i_handle = self._tr3_files_i[layer]
            x_handle.flush()
            i_handle.flush()
            x_handle.close()
            i_handle.close()
            layer_dir = Path(x_handle.name).parent
            os.replace(layer_dir / "x.bin.partial", layer_dir / "x.bin")
            os.replace(layer_dir / "ids.bin.partial", layer_dir / "ids.bin")
            layers[str(layer)] = {
                "tokens": self._tr3_counts[layer],
                "routed": self._tr3_routed[layer],
                "sha256_x": self._tr3_hash_x[layer].hexdigest(),
                "sha256_ids": self._tr3_hash_i[layer].hexdigest(),
            }
        return {
            "rank": self._tr3_rank(),
            "layers": layers,
            "verify_cmp": self._tr3_verify_cmp,
            "capture_fingerprint": self._tr3_capture_fingerprint,
        }


def make_llm(args):
    here = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    from vllm import LLM

    return LLM(
        model=str(args.src),
        tensor_parallel_size=CAPTURE_TP,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=args.max_num_seqs,
        kv_cache_memory_bytes=args.kv_cache_bytes,
        # The B300 TP8 post-load profile deadlocks in vLLM's custom/SYMM_MEM
        # dispatch path. PyNCCL is already used for EP and is reliable here.
        disable_custom_all_reduce=True,
        worker_extension_cls="capture_b300.CaptureWorkerExtension",
        seed=SEED,
        trust_remote_code=False,
    )


def _merge_status(results: list[dict], active_layers: list[int]) -> dict[int, int]:
    merged: dict[int, int] = {}
    for result in results:
        for layer, count in result.get("counts", {}).items():
            layer_i = int(layer)
            if layer_i in merged:
                raise RuntimeError(f"layer {layer_i} captured by multiple TP ranks")
            merged[layer_i] = int(count)
    if set(merged) != set(active_layers):
        raise RuntimeError(f"capture status layers {sorted(merged)} != active {active_layers}")
    return merged


def verify_engine_outputs(llm, logfile: str) -> None:
    from vllm import SamplingParams

    prompts = [
        "The capital of France is",
        "Under Kentucky law, a claim for negligence requires proof of duty, breach,",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
    )
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        if not text.strip():
            raise RuntimeError(f"empty greedy continuation for {prompt!r}")
        log(f"verify greedy: {prompt!r} -> {text!r}", logfile)


def check_post_load_memory(llm, minimum_free_bytes: int, logfile: str) -> list[dict]:
    statuses = sorted(llm.collective_rpc("tr3_cuda_memory_status"), key=lambda item: item["rank"])
    if len(statuses) != CAPTURE_TP:
        raise RuntimeError(f"expected {CAPTURE_TP} CUDA memory reports, got {len(statuses)}")
    for status in statuses:
        log(
            f"post-load GPU rank {status['rank']}: {status['device']} capability="
            f"{tuple(status['capability'])} free={status['free_bytes']/2**30:.2f} GiB / "
            f"{status['total_bytes']/2**30:.2f} GiB, allocated="
            f"{status['allocated_bytes']/2**30:.2f} GiB reserved="
            f"{status['reserved_bytes']/2**30:.2f} GiB",
            logfile,
        )
        if tuple(status["capability"]) != (10, 3):
            raise RuntimeError(
                f"rank {status['rank']}: expected B300 capability (10, 3), "
                f"got {tuple(status['capability'])}"
            )
        if int(status["free_bytes"]) < minimum_free_bytes:
            raise RuntimeError(
                f"rank {status['rank']}: only {status['free_bytes']/2**30:.2f} GiB free "
                f"after BF16 load; gate is {minimum_free_bytes/2**30:.2f} GiB"
            )
    return statuses


def run_capture(llm, passes, out_dir: Path, plan: dict, active_layers: list[int], args, logfile: str):
    from vllm import SamplingParams, TokensPrompt

    init_results = llm.collective_rpc(
        "tr3_capture_init",
        args=(
            str(out_dir),
            bool(args.verify_engine),
            active_layers,
            plan["capture_fingerprint"],
            int(plan["total_tokens"]),
        ),
    )
    assignments = {}
    for result in init_results:
        for layer in result.get("assigned_layers", []):
            if layer in assignments:
                raise RuntimeError(f"layer {layer} assigned to two ranks")
            assignments[layer] = int(result["rank"])
    if set(assignments) != set(active_layers) or sum(int(r.get("hooks", 0)) for r in init_results) != len(active_layers):
        raise RuntimeError(f"hook assignment invalid: {init_results}")
    log(f"hooks installed: {len(active_layers)} layers striped TP8; assignment={assignments}", logfile)

    sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    cumulative = 0
    pass_audit = []
    for pass_index, (pass_name, token_lists) in enumerate(passes):
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_lists]
        expected = sum(map(len, token_lists))
        remaining_tokens = int(plan["total_tokens"]) - cumulative
        assert_ram_capture_target(
            out_dir,
            remaining_tokens * len(active_layers) * (HIDDEN * 2 + TOPK),
            f"before pass {pass_name}",
        )
        started = time.time()
        outputs = llm.generate(prompts, sampling)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"pass {pass_name}: output count {len(outputs)} != {len(prompts)}")
        cumulative += expected
        counts = _merge_status(llm.collective_rpc("tr3_capture_status"), active_layers)
        bad = {layer: value for layer, value in counts.items() if value != cumulative}
        if bad:
            raise RuntimeError(
                f"pass {pass_name}: token accounting expected cumulative {cumulative}: {bad}"
            )
        elapsed = time.time() - started
        pass_audit.append(
            {"index": pass_index, "name": pass_name, "samples": len(prompts), "tokens": expected}
        )
        log(
            f"pass {pass_index + 1}/{len(passes)} {pass_name}: {expected} tokens, "
            f"cumulative={cumulative}, {elapsed:.1f}s",
            logfile,
        )

    if cumulative != int(plan["total_tokens"]):
        raise RuntimeError(f"captured {cumulative} tokens != plan {plan['total_tokens']}")
    final_results = llm.collective_rpc("tr3_capture_finalize")
    merged_layers = {}
    verify = {"checked_tokens": 0, "mismatch_tokens": 0, "kwarg_logits_layers": 0}
    for result in final_results:
        if result.get("capture_fingerprint") not in (None, plan["capture_fingerprint"]):
            raise RuntimeError("worker returned wrong capture fingerprint")
        for layer, info in result.get("layers", {}).items():
            if layer in merged_layers:
                raise RuntimeError(f"duplicate finalized layer {layer}")
            merged_layers[layer] = info
        for key, value in result.get("verify_cmp", {}).items():
            verify[key] = verify.get(key, 0) + int(value)
    if {int(layer) for layer in merged_layers} != set(active_layers):
        raise RuntimeError("finalized layer set does not match requested capture")
    return merged_layers, pass_audit, assignments, verify


def write_manifests(
    out_dir: Path,
    plan: dict,
    layer_results: dict,
    pass_audit: list[dict],
    assignments: dict[int, int],
    verify: dict,
    args,
    logfile: str,
) -> dict:
    tokens = int(plan["total_tokens"])
    run_uuid = str(uuid.uuid4())
    for layer_s, info in layer_results.items():
        layer = int(layer_s)
        layer_dir = out_dir / f"layer_{layer:03d}"
        x_path = layer_dir / "x.bin"
        ids_path = layer_dir / "ids.bin"
        if int(info["tokens"]) != tokens:
            raise RuntimeError(f"layer {layer}: finalized token count mismatch")
        if x_path.stat().st_size != tokens * HIDDEN * 2:
            raise RuntimeError(f"layer {layer}: x payload size mismatch")
        if ids_path.stat().st_size != tokens * TOPK:
            raise RuntimeError(f"layer {layer}: ids payload size mismatch")
        routed = [int(value) for value in info["routed"]]
        if len(routed) != NUM_EXPERTS or sum(routed) != tokens * TOPK:
            raise RuntimeError(f"layer {layer}: routed count audit failed")
        manifest = {
            "schema": "glm52-b300-layer-capture-v1",
            "layer": layer,
            "capture_fingerprint": plan["capture_fingerprint"],
            "capture_run_uuid": run_uuid,
            "capture_tp": CAPTURE_TP,
            "owner_rank": assignments[layer],
            "tokens": tokens,
            "hidden": HIDDEN,
            "x_dtype": "bfloat16",
            "x_bytes": tokens * HIDDEN * 2,
            "ids_topk": TOPK,
            "ids_bytes": tokens * TOPK,
            "sha256_x": info["sha256_x"],
            "sha256_ids": info["sha256_ids"],
            "routed_counts": routed,
            "routed_min": min(routed),
            "routed_max": max(routed),
            "cold_experts_lt1024": [expert for expert, count in enumerate(routed) if count < 1024],
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(layer_dir / "layer_manifest.json", manifest)

    if verify["checked_tokens"]:
        fraction = 1.0 - verify["mismatch_tokens"] / verify["checked_tokens"]
        if fraction < 0.99:
            raise RuntimeError(f"routing recompute vs router_logits agreement only {fraction:.6f}")
    run_manifest = {
        "schema": "glm52-b300-capture-run-v1",
        "capture_fingerprint": plan["capture_fingerprint"],
        "capture_run_uuid": run_uuid,
        "active_layers": sorted(int(layer) for layer in layer_results),
        "tokens_per_layer": tokens,
        "capture_payload_bytes": tokens * len(layer_results) * (HIDDEN * 2 + TOPK),
        "pass_audit": pass_audit,
        "rank_assignment": {str(layer): rank for layer, rank in sorted(assignments.items())},
        "verify_routing": verify,
        "capture_tp": CAPTURE_TP,
        "output_tp": OUTPUT_TP,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_tokens": 1,
        "cpu_offload_gb": 0,
        "low_memory_mode": False,
        "kv_cache_memory_bytes": args.kv_cache_bytes,
        "max_num_seqs": args.max_num_seqs,
        "vllm": vllm_version(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(out_dir / "capture_run_manifest.json", run_manifest)
    log(
        f"CAPTURE SEALED: {len(layer_results)} layer(s), {tokens} tokens/layer, "
        f"{run_manifest['capture_payload_bytes']/2**30:.2f} GiB RAM payload, "
        f"fingerprint={plan['capture_fingerprint']}",
        logfile,
    )
    return run_manifest


def vllm_version() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:
        return "unknown"


def clean_active_capture(out_dir: Path, layers: list[int]) -> None:
    for layer in layers:
        layer_dir = out_dir / f"layer_{layer:03d}"
        for name in ("x.bin", "ids.bin", "x.bin.partial", "ids.bin.partial", "layer_manifest.json"):
            path = layer_dir / name
            if path.exists():
                path.unlink()
        try:
            layer_dir.rmdir()
        except OSError:
            pass
    for name in ("capture_run_manifest.json", "capture_run_manifest.json.tmp"):
        path = out_dir / name
        if path.exists():
            path.unlink()


def capture_ready(out_dir: Path, plan: dict, layers: list[int]) -> tuple[bool, str]:
    tokens = int(plan["total_tokens"])
    for layer in layers:
        layer_dir = out_dir / f"layer_{layer:03d}"
        manifest_path = layer_dir / "layer_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            return False, f"layer {layer}: manifest unavailable ({exc})"
        if manifest.get("capture_fingerprint") != plan["capture_fingerprint"]:
            return False, f"layer {layer}: capture fingerprint mismatch"
        if int(manifest.get("tokens", -1)) != tokens:
            return False, f"layer {layer}: token count mismatch"
        x_path, ids_path = layer_dir / "x.bin", layer_dir / "ids.bin"
        if not x_path.is_file() or x_path.stat().st_size != tokens * HIDDEN * 2:
            return False, f"layer {layer}: x RAM payload absent/incomplete"
        if not ids_path.is_file() or ids_path.stat().st_size != tokens * TOPK:
            return False, f"layer {layer}: ids RAM payload absent/incomplete"
    return True, "ready"


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-5.2 TP8 BF16 RAM capture for B300")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true", help="build deterministic capture plan; no model load")
    modes.add_argument("--capture", action="store_true", help="run TP8 BF16 natural-routing capture")
    modes.add_argument("--status", action="store_true", help="validate RAM payloads for --layers")
    parser.add_argument("--src", default="/workspace/bf16", type=Path)
    parser.add_argument(
        "--corpus",
        default="/workspace/tr3/calib/reap_recall_calib.jsonl",
        type=Path,
    )
    parser.add_argument("--plan-file", default="/workspace/tr3/capture_plan.json", type=Path)
    parser.add_argument(
        "--capture-dir", default="/dev/shm/glm52-tr3-capture", type=Path
    )
    parser.add_argument("--layers", default="3-77")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--kv-cache-bytes", type=int, default=DEFAULT_KV_CACHE_BYTES)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument(
        "--min-post-load-free-bytes",
        type=int,
        default=DEFAULT_MIN_POST_LOAD_FREE_BYTES,
    )
    parser.add_argument(
        "--max-window-layers",
        type=int,
        default=DEFAULT_MAX_CAPTURE_WINDOW_LAYERS,
        help="fail closed above this capture-data window size",
    )
    parser.add_argument("--fresh", action="store_true", help="replace incomplete RAM payloads for active layers")
    parser.add_argument(
        "--reuse-existing-plan",
        action="store_true",
        help="accept a pre-staged, fingerprint-valid owner plan instead of regenerating boundaries",
    )
    parser.add_argument("--verify-engine", action="store_true", help="run three greedy checks before hooks")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    args.src = args.src.resolve()
    args.corpus = args.corpus.resolve()
    args.plan_file = args.plan_file.resolve()
    logfile = args.log or str(args.plan_file.with_suffix(".capture.log"))
    assert_disk_free(
        args.plan_file,
        METADATA_WRITE_ALLOWANCE_BYTES,
        "capture plan/log root",
    )

    if args.plan:
        if args.reuse_existing_plan and args.plan_file.exists():
            existing = json.loads(args.plan_file.read_text())
            validate_plan(existing, args.src, args.corpus)
            log(
                f"reusing sealed capture plan: {args.plan_file} "
                f"({len(existing['passes'])} passes, fingerprint={existing['capture_fingerprint']})",
                logfile,
            )
            return
        plan = build_capture_plan(args.src, args.corpus, args.target_tokens, logfile)
        if args.plan_file.exists():
            existing = json.loads(args.plan_file.read_text())
            validate_plan(existing, args.src, args.corpus)
            if existing["capture_fingerprint"] != plan["capture_fingerprint"]:
                raise SystemExit(
                    f"existing plan {args.plan_file} has a different fingerprint; move it explicitly "
                    "before changing calibration selection"
                )
            log(f"capture plan already exists and matches: {args.plan_file}", logfile)
        else:
            atomic_json(args.plan_file, plan)
            log(f"wrote capture plan: {args.plan_file}", logfile)
        return

    plan = json.loads(args.plan_file.read_text())
    validate_plan(plan, args.src, args.corpus)
    layers = parse_layers(args.layers)
    args.capture_dir = args.capture_dir.resolve()

    if args.status:
        ready, reason = capture_ready(args.capture_dir, plan, layers)
        print(f"{'READY' if ready else 'NOT READY'}: {reason}")
        raise SystemExit(0 if ready else 1)

    if not (1 <= args.max_window_layers <= len(MOE_LAYERS)):
        raise SystemExit("--max-window-layers must be in 1..75")
    if len(layers) > args.max_window_layers:
        raise SystemExit(
            f"RAM streaming requires windows of at most {args.max_window_layers} layers; "
            f"requested {len(layers)} ({args.layers}). Capture, encode with "
            "--consume-capture, then advance the window."
        )

    capture_bytes = int(plan["total_tokens"]) * len(layers) * (HIDDEN * 2 + TOPK)
    assert_ram_capture_target(args.capture_dir, capture_bytes, "capture-data window")

    if args.fresh:
        clean_active_capture(args.capture_dir, layers)
    else:
        ready, reason = capture_ready(args.capture_dir, plan, layers)
        if ready:
            log(f"RAM capture already ready for layers {args.layers}; skipping", logfile)
            return
        active_dirs = [args.capture_dir / f"layer_{layer:03d}" for layer in layers]
        if any(path.exists() and any(path.iterdir()) for path in active_dirs):
            raise SystemExit(f"RAM capture incomplete ({reason}); rerun with --fresh")
    args.capture_dir.mkdir(parents=True, exist_ok=True)

    passes = load_plan_tokens(plan, args.src, args.corpus, logfile)
    log(
        f"loading BF16 source TP{CAPTURE_TP}, no CPU offload/low-memory mode; "
        f"capture layers={args.layers}, RAM dir={args.capture_dir}",
        logfile,
    )
    llm = make_llm(args)
    check_post_load_memory(llm, args.min_post_load_free_bytes, logfile)
    if args.verify_engine:
        verify_engine_outputs(llm, logfile)
    layer_results, pass_audit, assignments, verify = run_capture(
        llm, passes, args.capture_dir, plan, layers, args, logfile
    )
    write_manifests(
        args.capture_dir,
        plan,
        layer_results,
        pass_audit,
        assignments,
        verify,
        args,
        logfile,
    )
    del llm
    gc.collect()
    log("vLLM object released; process exit returns all BF16 HBM before encode", logfile)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
