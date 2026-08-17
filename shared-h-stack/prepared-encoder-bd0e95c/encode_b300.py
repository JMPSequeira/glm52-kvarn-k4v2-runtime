#!/usr/bin/env python3
"""Strict BF16 -> all-routed-expert EXL3 adapter for encode_tr3_v31.py.

The calibrated LDLQ/trellis implementation is imported byte-for-byte from the
owner-designated production encoder and pinned by SHA-256.  This adapter changes
only source IO, RAM capture IO, all-256 tiering, resume identity, and assembly.
It never imports transformers or the exllamav3 Python package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback


BASE_ENCODER_SHA256 = "400c0df1c95c81c30a2ce31e060f0445a798fd29ad9339923d4e02e3ee40f6f7"
CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
EXPECTED_SHARDS = 282
EXPECTED_LAYER_TENSORS = 256 * 3 * 4 * 3 + 3 * 4
EXPECTED_TOTAL_TR3_TENSORS = 75 * EXPECTED_LAYER_TENSORS
EXPECTED_REPLACED_WEIGHTS = 75 * 256 * 3
CAPTURE_PLAN_SCHEMA = "glm52-b300-capture-plan-v1"
DONE_SCHEMA = "glm52-b300-exl3-layer-v1"
ADAPTER_VERSION = "shared-h-v1"
RAM_FILESYSTEMS = {"tmpfs", "ramfs"}
GIB = 1 << 30
DEFAULT_DISK_RESERVE_BYTES = 256 * GIB
ENCODE_LAYER_ALLOWANCE_BYTES = 4 * GIB
ASSEMBLY_ALLOWANCE_BYTES = 512 * GIB
METADATA_WRITE_ALLOWANCE_BYTES = 1 * GIB

BASE = None
ACTIVE_SOURCE_HASHER = None
ACTIVE_SOURCE_TENSORS = 0
CURRENT_EXPECTED_RECIPE = None


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


def assert_ram_capture_target(path: Path) -> dict:
    anchor = _existing_ancestor(path)
    fs_type = filesystem_type(anchor)
    if fs_type not in RAM_FILESYSTEMS:
        raise RuntimeError(
            f"RAM CAPTURE GUARD: {path} resolves to {fs_type}; refusing disk-backed capture"
        )
    result = {"target": str(path), "filesystem": fs_type}
    print(f"RAM CAPTURE GUARD PASS: {result}", flush=True)
    return result


def sha256_file(path: str | Path, chunk: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    encoded_size = len(json.dumps(payload, sort_keys=True).encode()) + 1
    assert_disk_free(path, max(encoded_size * 2, 1 << 20), "encoder JSON", quiet=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def load_base_encoder(path: Path):
    global BASE
    path = path.resolve()
    digest = sha256_file(path)
    if digest != BASE_ENCODER_SHA256:
        raise RuntimeError(
            f"production encoder SHA mismatch: {path} is {digest}, expected {BASE_ENCODER_SHA256}"
        )
    module_name = "_b300_encode_tr3_v31"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import production encoder from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    BASE = module

    # Format constants stay owner-pinned.  Only the tier count changes.
    if not (
        module.HIDDEN == 6144
        and module.MOE_INTER == 2048
        and module.NUM_LAYERS == 78
        and module.FIRST_MOE_LAYER == 3
        and module.NUM_EXPERTS == 256
        and module.BITS == 3
        and module.TP == 4
    ):
        raise RuntimeError("production encoder constants no longer match the owner-pinned recipe")
    module.KEEP_NVFP4 = 0
    module._lazy_torch = lazy_torch_b300
    module.load_expert_bf16 = load_expert_bf16_direct
    module.LayerCalib = LayerCalibRAM
    module.layer_done = layer_done
    return module


def lazy_torch_b300():
    import torch

    # Never accept the wheel's prebuilt top-level extension implicitly.  Every
    # spawned worker enters the sm_100 bootstrap so the cached JIT artifact is
    # provenance-checked and registered in this process.
    from bootstrap_ext_b300 import build

    ext = build()
    return torch, ext


def source_identity(src: Path) -> dict:
    index_path = src / "model.safetensors.index.json"
    config_path = src / "config.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(f"source has {len(shards)} indexed shards, expected {EXPECTED_SHARDS}")
    missing = [name for name in shards if not (src / name).is_file() or (src / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"source shards missing/empty: {missing[:8]}")
    shard_stats = [
        {
            "name": name,
            "bytes": (src / name).stat().st_size,
            "mtime_ns": (src / name).stat().st_mtime_ns,
        }
        for name in shards
    ]
    return {
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "unique_shards": len(shards),
        "shard_bytes": sum((src / name).stat().st_size for name in shards),
        # Fast fail-closed resume identity.  Final assembly additionally
        # re-hashes every routed BF16 tensor against the digest recorded at
        # encode time, so metadata alone is never the final integrity check.
        "shard_stat_fingerprint": canonical_hash({"shards": shard_stats}),
    }


def read_capture_plan(path: Path, src: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan.get("schema") != CAPTURE_PLAN_SCHEMA:
        raise RuntimeError(f"unexpected capture plan schema: {plan.get('schema')!r}")
    canonical = dict(plan)
    claimed = canonical.pop("capture_fingerprint", None)
    if claimed != canonical_hash(canonical):
        raise RuntimeError("capture plan fingerprint is invalid")
    if plan.get("corpus_sha256") != CORPUS_SHA256:
        raise RuntimeError("capture plan does not use the owner-pinned corpus")
    current = source_identity(src)
    planned = plan.get("source", {})
    if current["config_sha256"] != planned.get("config_sha256") or current["index_sha256"] != planned.get("index_sha256"):
        raise RuntimeError("capture plan and BF16 source config/index do not match")
    if int(plan.get("capture_tp", -1)) != 8 or int(plan.get("output_tp", -1)) != 4:
        raise RuntimeError("capture plan must declare TP8 capture and TP4 output")
    if plan.get("selection_policy") != "owner-corpus-axis-separated-luke-multipass-no-repeat-v1":
        raise RuntimeError("capture plan is not the owner-corpus Luke-style baseline")
    if plan.get("owner_corpus_only") is not True:
        raise RuntimeError("capture plan does not declare owner-corpus-only calibration")
    if plan.get("calibration_baseline") is not True:
        raise RuntimeError("capture plan does not declare mandatory baseline calibration")
    expected_routing = {
        "natural": True,
        "forced_expert_activation": False,
        "scoring_func": "sigmoid",
        "top_k": 8,
        "n_group": 1,
        "topk_group": 1,
    }
    if plan.get("routing") != expected_routing:
        raise RuntimeError("capture plan routing is not natural sigmoid top-8")
    return plan


def recipe_material(args, plan: dict) -> dict:
    base = BASE
    assert base is not None
    bootstrap_path = Path(__file__).resolve().with_name("bootstrap_ext_b300.py")
    return {
        "schema": "glm52-b300-exl3-recipe-v1",
        "capture_fingerprint": plan["capture_fingerprint"],
        "corpus_sha256": plan["corpus_sha256"],
        "selection_policy": plan["selection_policy"],
        "tokens_per_layer": int(plan["total_tokens"]),
        "source": source_identity(Path(args.src).resolve()),
        "production_encoder_sha256": BASE_ENCODER_SHA256,
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "adapter_version": ADAPTER_VERSION,
        "exllamav3": "0.0.43",
        "scope": {
            "layers": [3, 77],
            "experts_per_layer": 256,
            "projections": ["gate_proj", "up_proj", "down_proj"],
            "keep_nvfp4": 0,
            "tr3_experts": 256,
        },
        "format": {
            "bits": 3,
            "codebook": "mcg",
            "mcg_multiplier": base.MCG_MULT,
            "tp": 4,
            "hidden": 6144,
            "moe_intermediate": 2048,
        },
        "calibration_math": {
            "hessian": "ldlq-calibrated",
            "sigma_reg": base.SIGMA_REG,
            "min_routed": int(args.min_routed),
            "seed_base": base.SEED_BASE,
            "out_scales": args.out_scales,
            "lockstep": base.resolve_lockstep(args.lockstep),
            "gss_lockstep": True,
            "rotation_layout": "shared_h_v1",
            "shared_profile": "signed-geometric-mean over all 256 experts",
            "shared_profile_passes": 2,
        },
    }


def recipe_fingerprint(args, plan: dict) -> str:
    return canonical_hash(recipe_material(args, plan))


def _expected_layer_entries(layer: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    base = BASE
    assert base is not None
    expected = {}
    for expert in range(base.NUM_EXPERTS):
        for proj in base.PROJS:
            for rank in range(base.TP):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.rank{rank}"
                if proj == "down_proj":
                    k, n = base.SLICE, base.HIDDEN
                else:
                    k, n = base.HIDDEN, base.SLICE
                if proj == "down_proj":
                    expected[f"{prefix}.suh"] = ("F16", (k,))
                else:
                    expected[f"{prefix}.svh"] = ("F16", (n,))
                expected[f"{prefix}.trellis"] = ("I16", (k // 16, n // 16, 48))
                expected[f"{prefix}.mcg"] = ("I32", ())
    for proj in base.PROJS:
        for rank in range(base.TP):
            field = "svh" if proj == "down_proj" else "suh"
            expected[
                f"model.layers.{layer}.mlp.experts.shared_h.{proj}.rank{rank}.{field}"
            ] = ("F16", (base.HIDDEN,))
    return expected


def validate_layer_schema(path: Path, layer: int) -> None:
    base = BASE
    assert base is not None
    reader = base.STReader(str(path))
    expected = _expected_layer_entries(layer)
    if len(reader.tensors) != EXPECTED_LAYER_TENSORS or set(reader.tensors) != set(expected):
        missing = sorted(set(expected) - set(reader.tensors))[:4]
        extra = sorted(set(reader.tensors) - set(expected))[:4]
        raise RuntimeError(
            f"layer {layer} EXL3 schema mismatch: count={len(reader.tensors)} "
            f"missing={missing} extra={extra}"
        )
    for name, (dtype, shape) in expected.items():
        actual_dtype, actual_shape, _, _ = reader.tensors[name]
        if actual_dtype != dtype or tuple(actual_shape) != shape:
            raise RuntimeError(
                f"layer {layer} tensor {name}: {actual_dtype}{actual_shape} != {dtype}{shape}"
            )


def layer_done(work: str, layer: int, expected_recipe: str | None = None) -> bool:
    base = BASE
    assert base is not None
    expected_recipe = expected_recipe or CURRENT_EXPECTED_RECIPE
    st_path_s, done_path_s = base.layer_paths(work, layer)
    st_path, done_path = Path(st_path_s), Path(done_path_s)
    if not st_path.is_file() or not done_path.is_file():
        return False
    try:
        done = json.loads(done_path.read_text())
        checks = (
            done.get("schema") == DONE_SCHEMA,
            int(done.get("layer", -1)) == layer,
            int(done.get("bits", -1)) == 3,
            int(done.get("tp", -1)) == 4,
            done.get("keep_nvfp4") == [],
            done.get("tail_tr3") == list(range(256)),
            done.get("rotation_layout") == "shared_h_v1",
            int(done.get("tensor_count", -1)) == EXPECTED_LAYER_TENSORS,
            expected_recipe is None or done.get("recipe_fingerprint") == expected_recipe,
            done.get("file_sha256") == sha256_file(st_path),
        )
        if not all(checks):
            return False
        validate_layer_schema(st_path, layer)
        return True
    except Exception:
        return False


class LayerCalibRAM:
    """Read-only mmap view over one sealed tmpfs capture layer."""

    def __init__(self, capture_dir: str, layer: int, logfile: str | None = None):
        base = BASE
        assert base is not None
        torch, _ = lazy_torch_b300()
        import numpy as np

        layer_dir = Path(capture_dir) / f"layer_{layer:03d}"
        manifest = json.loads((layer_dir / "layer_manifest.json").read_text())
        if manifest.get("capture_fingerprint") != getattr(self.__class__, "expected_fingerprint", None):
            raise RuntimeError(f"layer {layer}: RAM capture fingerprint mismatch")
        self.manifest = manifest
        self.L = layer
        self.tokens = int(manifest["tokens"])
        if manifest.get("hidden") != base.HIDDEN or manifest.get("x_dtype") != "bfloat16":
            raise RuntimeError(f"layer {layer}: capture geometry/dtype mismatch")
        x_path, ids_path = layer_dir / "x.bin", layer_dir / "ids.bin"
        if x_path.stat().st_size != self.tokens * base.HIDDEN * 2:
            raise RuntimeError(f"layer {layer}: x RAM payload size mismatch")
        if ids_path.stat().st_size != self.tokens * 8:
            raise RuntimeError(f"layer {layer}: ids RAM payload size mismatch")

        started = time.time()
        if sha256_file(x_path) != manifest.get("sha256_x"):
            raise RuntimeError(f"layer {layer}: x RAM payload SHA mismatch")
        if sha256_file(ids_path) != manifest.get("sha256_ids"):
            raise RuntimeError(f"layer {layer}: ids RAM payload SHA mismatch")
        self._x_np = np.memmap(x_path, dtype=np.int16, mode="c", shape=(self.tokens, base.HIDDEN))
        self.x = torch.from_numpy(self._x_np).view(torch.bfloat16)
        self._ids_np = np.memmap(ids_path, dtype=np.uint8, mode="r", shape=(self.tokens, 8))
        if self._ids_np.min() < 0 or self._ids_np.max() >= base.NUM_EXPERTS:
            raise RuntimeError(f"layer {layer}: routed id out of range")
        sorted_ids = np.sort(self._ids_np, axis=1)
        if not (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all():
            raise RuntimeError(f"layer {layer}: duplicate routed expert within a token")
        flat = self._ids_np.reshape(-1)
        order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=base.NUM_EXPERTS)
        if counts.tolist() != manifest.get("routed_counts") or int(counts.sum()) != self.tokens * 8:
            raise RuntimeError(f"layer {layer}: routed-count manifest mismatch")
        self._starts = np.concatenate([[0], np.cumsum(counts)])
        self._token_of = (order // 8).astype(np.int64)
        self.routed_counts = counts.tolist()
        self._H_layer_cpu = None
        self._fallback_rows = None
        if logfile:
            base.log(
                f"layer {layer}: RAM mmap verified — {self.tokens} tokens, "
                f"routed min/max={counts.min()}/{counts.max()}, "
                f"cold(<{base.MIN_ROUTED})={(counts < base.MIN_ROUTED).sum()}, "
                f"sha+mmap={time.time()-started:.1f}s",
                logfile,
            )

    def expert_rows(self, expert: int):
        torch, _ = lazy_torch_b300()
        return torch.from_numpy(
            self._token_of[self._starts[expert] : self._starts[expert + 1]].copy()
        )

    def gather_chunks(self, rows, device, chunk=None):
        torch, _ = lazy_torch_b300()
        chunk = chunk or BASE.HCHUNK
        for start in range(0, rows.numel(), chunk):
            selection = rows[start : start + chunk]
            yield torch.index_select(self.x, 0, selection).to(device).float()

    def all_rows(self):
        torch, _ = lazy_torch_b300()
        return torch.arange(self.tokens, dtype=torch.int64)

    def fallback_rows(self):
        torch, _ = lazy_torch_b300()
        if self._fallback_rows is None:
            generator = torch.Generator().manual_seed(
                BASE.SEED_BASE ^ (self.L * 2654435761 & 0x7FFFFFFF)
            )
            count = min(self.tokens, BASE.FALLBACK_ROWS)
            self._fallback_rows = torch.randperm(self.tokens, generator=generator)[:count].sort().values
        return self._fallback_rows

    def layer_H(self, device):
        torch, _ = lazy_torch_b300()
        if self._H_layer_cpu is None:
            hessian = torch.zeros(BASE.HIDDEN, BASE.HIDDEN, dtype=torch.float32, device=device)
            for chunk in self.gather_chunks(self.all_rows(), device):
                hessian.addmm_(chunk.T, chunk)
            if not torch.isfinite(hessian).all():
                raise RuntimeError(f"layer {self.L}: non-finite layer Hessian")
            self._H_layer_cpu = hessian.cpu()
            del hessian
        return self._H_layer_cpu.to(device), self.tokens


def load_expert_bf16_direct(src, layer: int, expert: int, proj: str, device, logfile):
    global ACTIVE_SOURCE_HASHER, ACTIVE_SOURCE_TENSORS
    torch, _ = lazy_torch_b300()
    key = src_key = BASE.expert_key(layer, expert, proj, "weight")
    reader = src.reader_for(src_key, logfile)
    dtype, shape, _, _ = reader.tensors[src_key]
    expected_shape = (
        (BASE.MOE_INTER, BASE.HIDDEN) if proj != "down_proj" else (BASE.HIDDEN, BASE.MOE_INTER)
    )
    if dtype != "BF16" or tuple(shape) != expected_shape:
        raise RuntimeError(f"{key}: expected BF16 {expected_shape}, got {dtype} {shape}")
    raw = reader.read_bytes(src_key)
    if len(raw) != BASE.MOE_INTER * BASE.HIDDEN * 2:
        raise RuntimeError(f"{key}: BF16 payload length mismatch")
    if ACTIVE_SOURCE_HASHER is not None:
        ACTIVE_SOURCE_HASHER.update(key.encode())
        ACTIVE_SOURCE_HASHER.update(b"\0")
        ACTIVE_SOURCE_HASHER.update(raw)
        ACTIVE_SOURCE_TENSORS += 1
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(*shape)
    return tensor.to(device)


def check_capture(capture_dir: str, layers: list[int], plan: dict) -> None:
    assert_ram_capture_target(Path(capture_dir))
    tokens = int(plan["total_tokens"])
    for layer in layers:
        layer_dir = Path(capture_dir) / f"layer_{layer:03d}"
        try:
            manifest = json.loads((layer_dir / "layer_manifest.json").read_text())
        except Exception as exc:
            raise RuntimeError(f"layer {layer}: RAM capture manifest missing ({exc})") from exc
        if manifest.get("capture_fingerprint") != plan["capture_fingerprint"]:
            raise RuntimeError(f"layer {layer}: RAM capture belongs to a different plan")
        if int(manifest.get("tokens", -1)) != tokens:
            raise RuntimeError(f"layer {layer}: RAM capture token count mismatch")
        if (layer_dir / "x.bin").stat().st_size != tokens * BASE.HIDDEN * 2:
            raise RuntimeError(f"layer {layer}: x RAM payload incomplete")
        if (layer_dir / "ids.bin").stat().st_size != tokens * 8:
            raise RuntimeError(f"layer {layer}: ids RAM payload incomplete")


def consume_capture(capture_dir: str, layer: int, logfile: str) -> None:
    layer_dir = Path(capture_dir) / f"layer_{layer:03d}"
    removed = 0
    for name in ("x.bin", "ids.bin"):
        path = layer_dir / name
        if path.exists():
            removed += path.stat().st_size
            path.unlink()
    BASE.log(f"layer {layer}: released {removed/2**30:.2f} GiB RAM capture payload", logfile)


def prune_completed_capture(args, layers: list[int]) -> None:
    """Release only payloads whose matching EXL3 layer passes every done check."""
    logfile = str(Path(args.work) / "logs" / "capture-prune.log")
    assert_disk_free(Path(args.work), METADATA_WRITE_ALLOWANCE_BYTES, "capture prune log")
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    pruned = 0
    for layer in layers:
        if layer_done(args.work, layer, args.expected_recipe):
            consume_capture(args.capture_dir, layer, logfile)
            pruned += 1
    BASE.log(f"capture prune: {pruned} validated completed layer(s) inspected", logfile)


def process_layer_b300(args, src, layer: int, logfile: str, expected_recipe: str) -> None:
    global ACTIVE_SOURCE_HASHER, ACTIVE_SOURCE_TENSORS
    assert_disk_free(Path(args.work), ENCODE_LAYER_ALLOWANCE_BYTES, f"encode layer {layer}")
    ACTIVE_SOURCE_HASHER = None
    ACTIVE_SOURCE_TENSORS = 0
    LayerCalibRAM.expected_fingerprint = args.capture_plan["capture_fingerprint"]
    layer_manifest = json.loads(
        (Path(args.capture_dir) / f"layer_{layer:03d}" / "layer_manifest.json").read_text()
    )
    profile_identity = canonical_hash(
        {
            "schema": "glm52-exl3-shared-h-profile-identity-v1",
            "recipe_fingerprint": expected_recipe,
            "layer": layer,
            "capture_sha256_x": layer_manifest["sha256_x"],
            "capture_sha256_ids": layer_manifest["sha256_ids"],
        }
    )
    BASE.profile_layer_shared_h(
        src,
        args.work,
        layer,
        {"auto": None, "always": True, "never": False}[args.out_scales],
        args.capture_dir,
        args.min_routed,
        logfile,
        profile_identity,
    )
    shared_h_profiles, shared_h_profile_done = BASE.load_layer_shared_h_profiles(
        args.work, layer, expected_identity=profile_identity
    )
    # The production source digest covers the actual LDLQ encode pass. The
    # profile pass reads the same tensors but is independently hash-pinned.
    ACTIVE_SOURCE_HASHER = hashlib.sha256()
    ACTIVE_SOURCE_TENSORS = 0
    BASE.process_layer(
        src,
        args.work,
        layer,
        {"auto": None, "always": True, "never": False}[args.out_scales],
        args.capture_dir,
        args.min_routed,
        logfile,
        BASE.resolve_lockstep(args.lockstep),
        shared_h_profiles,
    )
    if ACTIVE_SOURCE_TENSORS != 256 * 3:
        raise RuntimeError(
            f"layer {layer}: read {ACTIVE_SOURCE_TENSORS} BF16 expert tensors, expected 768"
        )
    st_path, done_path = BASE.layer_paths(args.work, layer)
    done = json.loads(Path(done_path).read_text())
    if done.get("keep_nvfp4") != [] or done.get("tail_tr3") != list(range(256)):
        raise RuntimeError(f"layer {layer}: encoder tier drifted from all-256 EXL3")
    done.update(
        {
            "schema": DONE_SCHEMA,
            "recipe_fingerprint": expected_recipe,
            "recipe": recipe_material(args, args.capture_plan),
            "source_expert_payload_sha256": ACTIVE_SOURCE_HASHER.hexdigest(),
            "source_expert_tensor_count": ACTIVE_SOURCE_TENSORS,
            "tensor_count": EXPECTED_LAYER_TENSORS,
            "capture": {
                "fingerprint": args.capture_plan["capture_fingerprint"],
                "tokens": int(args.capture_plan["total_tokens"]),
                "sha256_x": layer_manifest["sha256_x"],
                "sha256_ids": layer_manifest["sha256_ids"],
            },
            "source_format": "BF16-direct",
            "rotation_layout": "shared_h_v1",
            "shared_h_profile_sha256": shared_h_profile_done["file_sha256"],
            "keep_nvfp4": [],
            "tail_tr3": list(range(256)),
        }
    )
    atomic_json(done_path, done)
    if not layer_done(args.work, layer, expected_recipe):
        raise RuntimeError(f"layer {layer}: post-write done/schema verification failed")
    if args.consume_capture:
        consume_capture(args.capture_dir, layer, logfile)
    ACTIVE_SOURCE_HASHER = None


def worker_main(args) -> None:
    logfile = str(Path(args.work) / "logs" / f"worker{args.worker_rank}.log")
    assert_disk_free(
        Path(args.work), ENCODE_LAYER_ALLOWANCE_BYTES, f"worker {args.worker_rank} startup"
    )
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    torch, _ = lazy_torch_b300()
    BASE.log(
        f"B300 worker {args.worker_rank} on {torch.cuda.get_device_name(0)}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        logfile,
    )
    src = BASE.SourceModel(args.src)
    layers = BASE.parse_layers(args.layers)
    mine = layers[args.worker_rank :: args.workers]
    todo = [layer for layer in mine if not layer_done(args.work, layer, args.expected_recipe)]
    check_capture(args.capture_dir, todo, args.capture_plan)
    for layer in todo:
        process_layer_b300(args, src, layer, logfile, args.expected_recipe)
    BASE.log(f"worker {args.worker_rank}: {len(todo)} layer(s) encoded", logfile)


def orchestrate_encode(args) -> None:
    if not (1 <= args.workers <= args.gpus):
        raise RuntimeError(f"workers must be in [1,gpus], got {args.workers}/{args.gpus}")
    layers = BASE.parse_layers(args.layers)
    todo = [layer for layer in layers if not layer_done(args.work, layer, args.expected_recipe)]
    assert_disk_free(
        Path(args.work),
        len(todo) * ENCODE_LAYER_ALLOWANCE_BYTES + METADATA_WRITE_ALLOWANCE_BYTES,
        f"aggregate encode work for {len(todo)} pending layers",
    )
    Path(args.work, "logs").mkdir(parents=True, exist_ok=True)
    logfile = str(Path(args.work) / "logs" / "driver.log")
    check_capture(args.capture_dir, todo, args.capture_plan)
    BASE.log(
        f"BF16-direct all-EXL3 encode: {len(layers)} requested, {len(todo)} pending, "
        f"workers={args.workers}, GPUs={args.gpus}, bits=3, keep=0, tail=256, "
        f"encoded_tp=4, recipe={args.expected_recipe}",
        logfile,
    )
    if not todo:
        BASE.log("nothing to encode", logfile)
        return
    processes = []
    for rank in range(args.workers):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(rank % args.gpus)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-rank",
            str(rank),
            "--base-encoder",
            args.base_encoder,
            "--src",
            args.src,
            "--work",
            args.work,
            "--layers",
            args.layers,
            "--workers",
            str(args.workers),
            "--gpus",
            str(args.gpus),
            "--capture-dir",
            args.capture_dir,
            "--capture-manifest",
            args.capture_manifest,
            "--min-routed",
            str(args.min_routed),
            "--out-scales",
            args.out_scales,
            "--lockstep",
            str(args.lockstep),
        ]
        if args.consume_capture:
            command.append("--consume-capture")
        process = subprocess.Popen(command, env=env)
        processes.append(process)
        BASE.log(f"spawned worker {rank} pid {process.pid} on GPU {rank % args.gpus}", logfile)
    rc = 0
    try:
        for rank, process in enumerate(processes):
            code = process.wait()
            rc |= code
            BASE.log(f"worker {rank} exited {code}", logfile)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise
    complete = [layer for layer in layers if layer_done(args.work, layer, args.expected_recipe)]
    BASE.log(f"encode pass: {len(complete)}/{len(layers)} valid layers", logfile)
    if rc or len(complete) != len(layers):
        raise SystemExit(1)


def modelopt_dispatch_config() -> dict:
    """Compatibility metadata that selects the b12x ModelOpt interception.

    There are zero NVFP4 payload tensors.  The ignore list leaves only routed
    experts in layers 3..77 on the intercepted path; MTP layer 78 is wholly
    ignored so its BF16 experts are loaded normally.
    """
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": [
            "lm_head",
            "*embed_tokens*",
            "model.norm*",
            "*self_attn*",
            "model.layers.0.mlp*",
            "model.layers.1.mlp*",
            "model.layers.2.mlp*",
            "*shared_experts*",
            "*mlp.gate*",
            "model.layers.78*",
        ],
        "quant_algo": "NVFP4",
        "producer": {"name": "b300-exl3-modelopt-dispatch-shim", "version": ADAPTER_VERSION},
        "quant_method": "modelopt",
    }


def audit_output(out: Path, source_keys: set[str], dropped: set[str], logfile: str) -> None:
    index = json.loads((out / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    indexed_files = set(weight_map.values())
    actual_files = {path.name for path in out.glob("*.safetensors")}
    if indexed_files != actual_files:
        raise RuntimeError(
            f"output shard/index file set mismatch: missing={indexed_files-actual_files}, "
            f"stale={actual_files-indexed_files}"
        )
    if dropped & set(weight_map):
        raise RuntimeError("BF16 routed expert weights survived output assembly")
    tr3_keys = [key for key in weight_map if ".rank" in key and key.rsplit(".", 1)[-1] in {"trellis", "suh", "svh", "mcg"}]
    if len(tr3_keys) != EXPECTED_TOTAL_TR3_TENSORS:
        raise RuntimeError(f"output has {len(tr3_keys)} EXL3 tensors, expected {EXPECTED_TOTAL_TR3_TENSORS}")
    carried_expected = source_keys - dropped
    carried_actual = set(weight_map) & source_keys
    if carried_actual != carried_expected:
        raise RuntimeError(
            f"carried source scope mismatch: missing={list(carried_expected-carried_actual)[:4]} "
            f"unexpected={list(carried_actual-carried_expected)[:4]}"
        )
    BASE.audit_index(str(out), logfile)


def _hash_source_expert_layer(job: tuple[str, int]) -> tuple[int, str, int]:
    """Recreate the exact per-layer source digest recorded during encode."""
    src_path, layer = job
    src = BASE.SourceModel(src_path)
    digest = hashlib.sha256()
    count = 0
    for expert in range(BASE.NUM_EXPERTS):
        for proj in BASE.PROJS:
            key = BASE.expert_key(layer, expert, proj, "weight")
            reader = src.reader_for(key)
            dtype, shape, _, _ = reader.tensors[key]
            expected_shape = (
                (BASE.MOE_INTER, BASE.HIDDEN)
                if proj != "down_proj"
                else (BASE.HIDDEN, BASE.MOE_INTER)
            )
            if dtype != "BF16" or tuple(shape) != expected_shape:
                raise RuntimeError(
                    f"{key}: final source audit expected BF16 {expected_shape}, got {dtype} {shape}"
                )
            raw = reader.read_bytes(key)
            digest.update(key.encode())
            digest.update(b"\0")
            digest.update(raw)
            count += 1
    return layer, digest.hexdigest(), count


def verify_encoded_source_payloads(
    src_path: str, dones: dict[int, dict], layers: list[int], workers: int, logfile: str
) -> None:
    """Fail if any BF16 expert payload changed after it was encoded."""
    from multiprocessing import get_context

    BASE.log(
        "assemble: re-hashing all 57,600 routed BF16 source tensors against encode-time digests",
        logfile,
    )
    jobs = [(src_path, layer) for layer in layers]
    with get_context("fork").Pool(processes=min(workers, len(jobs))) as pool:
        results = pool.map(_hash_source_expert_layer, jobs)
    for layer, digest, count in results:
        done = dones[layer]
        if count != 256 * 3 or int(done.get("source_expert_tensor_count", -1)) != count:
            raise RuntimeError(f"layer {layer}: source expert tensor-count audit failed")
        if digest != done.get("source_expert_payload_sha256"):
            raise RuntimeError(f"layer {layer}: BF16 expert payload changed since encode")
    BASE.log("assemble: routed BF16 source payload audit PASSED", logfile)


def assemble(args) -> None:
    out = Path(args.out)
    assert_disk_free(out, ASSEMBLY_ALLOWANCE_BYTES, "assembled checkpoint")
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise RuntimeError(f"assembly output must be absent/empty (fail closed): {out}")
    out.mkdir(parents=True, exist_ok=True)
    Path(args.work, "logs").mkdir(parents=True, exist_ok=True)
    logfile = str(Path(args.work) / "logs" / "assemble.log")
    src = BASE.SourceModel(args.src)
    layers = src.moe_layers()
    dones = {}
    for layer in layers:
        if not layer_done(args.work, layer, args.expected_recipe):
            raise RuntimeError(f"layer {layer}: missing/stale/invalid done artifact")
        _, done_path = BASE.layer_paths(args.work, layer)
        dones[layer] = json.loads(Path(done_path).read_text())
    BASE.log(f"assemble: all {len(layers)} all-EXL3 layers validated", logfile)
    verify_encoded_source_payloads(args.src, dones, layers, args.io_workers, logfile)

    all_src = sorted(src.weight_map)
    source_keys = set(all_src)
    dropped = {
        BASE.expert_key(layer, expert, proj, "weight")
        for layer in layers
        for expert in range(BASE.NUM_EXPERTS)
        for proj in BASE.PROJS
    }
    if len(dropped) != EXPECTED_REPLACED_WEIGHTS or not dropped <= source_keys:
        raise RuntimeError(
            f"BF16 replacement scope invalid: {len(dropped)} planned, "
            f"{len(dropped & source_keys)} present"
        )

    def carried_for_prefix(prefix: str) -> list[str]:
        return [key for key in all_src if key.startswith(prefix) and key not in dropped]

    shard_jobs = [("model-embed.safetensors", [("src", "model.embed_tokens.weight")])]
    head = [key for key in ("lm_head.weight", "model.norm.weight") if key in src.weight_map]
    shard_jobs.append(("model-head.safetensors", [("src", key) for key in head]))
    for layer in range(BASE.NUM_LAYERS + 1):
        items = [("src", key) for key in carried_for_prefix(f"model.layers.{layer}.")]
        if layer in dones:
            st_path, _ = BASE.layer_paths(args.work, layer)
            reader = BASE.STReader(st_path)
            if len(reader.tensors) != EXPECTED_LAYER_TENSORS:
                raise RuntimeError(f"layer {layer}: unexpected work tensor count")
            items += [("tr3", layer, key) for key in sorted(reader.tensors)]
        shard_jobs.append((f"model-layer-{layer:03d}.safetensors", items))

    accounted = {item[1] for _, items in shard_jobs for item in items if item[0] == "src"}
    if accounted != source_keys - dropped:
        raise RuntimeError("source tensor accounting is not bijective before assembly")

    from multiprocessing import get_context

    jobs = [(args.src, args.work, str(out), name, items) for name, items in shard_jobs]
    with get_context("fork").Pool(processes=min(args.io_workers, len(jobs))) as pool:
        results = pool.map(BASE._assemble_shard, jobs)
    weight_map = {}
    total_size = 0
    for (name, _), (keys, nbytes) in zip(shard_jobs, results):
        for key in keys:
            if key in weight_map:
                raise RuntimeError(f"duplicate output tensor key: {key}")
            weight_map[key] = name
        total_size += nbytes
    BASE.log(
        f"assemble: {len(shard_jobs)} shards, {total_size/2**30:.2f} GiB payload; "
        "all carried tensors re-read and byte-verified",
        logfile,
    )

    plan = args.capture_plan
    per_x = {str(layer): dones[layer]["capture"]["sha256_x"] for layer in layers}
    per_ids = {str(layer): dones[layer]["capture"]["sha256_ids"] for layer in layers}
    if {done["recipe_fingerprint"] for done in dones.values()} != {args.expected_recipe}:
        raise RuntimeError("mixed recipe fingerprints in layer done files")
    config = dict(src.config)
    config["quantization_config"] = modelopt_dispatch_config()
    config["hybrid_tr3_tail"] = {
        "producer": "encode_b300.py",
        "producer_version": ADAPTER_VERSION,
        "source_format": "BF16",
        "source_config_sha256": plan["source"]["config_sha256"],
        "source_index_sha256": plan["source"]["index_sha256"],
        "format": "exl3-trellis",
        "bits": 3.0,
        "codebook": "mcg",
        "mcg_multiplier": BASE.MCG_MULT,
        "hessian": "ldlq-calibrated",
        "exllamav3_version": "0.0.43",
        "moe_layers": [3, 77],
        "experts_per_layer": 256,
        "nvfp4_keep_per_layer": 0,
        "tr3_tail_per_layer": 256,
        "tp": 4,
        "rotation_layout": "shared_h_v1",
        "shared_h_tensor_schema": (
            "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}"
        ),
        "recipe_fingerprint": args.expected_recipe,
        "calibration": {
            "corpus_sha256": plan["corpus_sha256"],
            "corpus_rows": plan["corpus_rows"],
            "axis_rows": plan["axis_rows"],
            "selection_policy": plan["selection_policy"],
            "capture_fingerprint": plan["capture_fingerprint"],
            "passes": [
                {
                    "name": item["name"],
                    "axis": item.get("axis"),
                    "samples": len(item["samples"]),
                    "tokens": item["tokens"],
                }
                for item in plan["passes"]
            ],
            "tokens_per_layer": plan["total_tokens"],
            "min_routed_floor": args.min_routed,
            "sigma_reg": BASE.SIGMA_REG,
            "layer_h_fallback_experts_total": sum(
                len(done["experts_layer_h_fallback"]) for done in dones.values()
            ),
            "q_fallback_slices_total": sum(len(done["q_fallback_slices"]) for done in dones.values()),
            "per_layer_x_sha256": per_x,
            "per_layer_ids_sha256": per_ids,
        },
        "scope": {
            "quantized": "routed MoE expert gate/up/down projections, all 256 experts, layers 3..77",
            "bf16_byte_exact": [
                "attention",
                "dense MLPs layers 0..2",
                "shared experts",
                "router/gates",
                "MTP layer 78",
                "embeddings",
                "lm_head",
            ],
        },
        "slicing": {
            "gate_proj": "TP4 N-slice: rank r owns output rows [512r,512r+512)",
            "up_proj": "TP4 N-slice: rank r owns output rows [512r,512r+512)",
            "down_proj": "TP4 K-slice: rank r owns input columns [512r,512r+512)",
        },
        "tensor_schema": "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}",
        "tier_bitmap": "tier_bitmap.json",
        "modelopt_dispatch_note": (
            "quantization_config selects the b12x ModelOpt interception only; this artifact "
            "contains zero NVFP4 expert payloads. Layer 78 is ignored and remains BF16."
        ),
    }
    atomic_json(out / "config.json", config)
    atomic_json(
        out / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    tier_bitmap = {
        str(layer): {
            "keep_nvfp4": [],
            "tail_tr3": list(range(256)),
            "expert_rel_rt_mse": dones[layer]["expert_rel_rt_mse"],
        }
        for layer in layers
    }
    atomic_json(out / "tier_bitmap.json", tier_bitmap)
    atomic_json(out / "calibration_manifest.json", plan)

    generated_names = {
        "calibration_manifest.json",
        "tier_bitmap.json",
        "model.safetensors.index.json",
        "config.json",
        "MANIFEST.sha256",
    }
    for filename in BASE.aux_files(args.src):
        if filename in generated_names:
            raise RuntimeError(f"source auxiliary name collides with generated output: {filename}")
        source = Path(args.src) / filename
        destination = out / filename
        with source.open("rb") as src_handle, destination.open("wb") as dst_handle:
            while block := src_handle.read(BASE.CHUNK):
                dst_handle.write(block)
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"auxiliary byte-copy mismatch: {filename}")

    audit_output(out, source_keys, dropped, logfile)
    names = sorted(path.name for path in out.iterdir() if path.is_file() and path.name != "MANIFEST.sha256")
    with (out / "MANIFEST.sha256").open("w", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{sha256_file(out / name)}  {name}\n")
    BASE.log(
        f"assemble COMPLETE: {out}, {len(names)} files + MANIFEST, "
        f"replaced={len(dropped)} BF16 weights, EXL3 tensors={EXPECTED_TOTAL_TR3_TENSORS}",
        logfile,
    )


def configure_args(args) -> None:
    global CURRENT_EXPECTED_RECIPE
    args.base_encoder = str(Path(args.base_encoder).resolve())
    args.src = str(Path(args.src).resolve())
    args.work = str(Path(args.work).resolve())
    args.capture_manifest = str(Path(args.capture_manifest).resolve())
    if args.capture_dir:
        args.capture_dir = str(Path(args.capture_dir).resolve())
    load_base_encoder(Path(args.base_encoder))
    args.capture_plan = read_capture_plan(Path(args.capture_manifest), Path(args.src))
    args.expected_recipe = recipe_fingerprint(args, args.capture_plan)
    CURRENT_EXPECTED_RECIPE = args.expected_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-5.2 BF16 -> all-EXL3 B300 encoder")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--encode", action="store_true")
    actions.add_argument("--assemble", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--oracle", action="store_true")
    actions.add_argument("--pending", action="store_true", help="print comma-separated invalid/pending layers")
    actions.add_argument(
        "--prune-completed-capture",
        action="store_true",
        help="delete tmpfs x/ids only for layers with fully validated done artifacts",
    )
    parser.add_argument("--base-encoder", default="/workspace/tr3/encode_tr3_v31.py")
    parser.add_argument("--src", default="/workspace/bf16")
    parser.add_argument("--work", default="/workspace/tr3/encode-work")
    parser.add_argument("--out", default="")
    parser.add_argument("--capture-dir", default="/dev/shm/glm52-tr3-capture")
    parser.add_argument("--capture-manifest", default="/workspace/tr3/capture_plan.json")
    parser.add_argument("--layers", default="3-77")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--min-routed", type=int, default=1024)
    parser.add_argument("--out-scales", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--lockstep", default="auto")
    parser.add_argument("--consume-capture", action="store_true")
    parser.add_argument("--oracle-experts", default="0,1")
    parser.add_argument("--oracle-log", default="/tmp/b300_oracle.log")
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_args(args)
    assert_disk_free(Path(args.work), METADATA_WRITE_ALLOWANCE_BYTES, "encoder startup")

    if args.worker_rank is not None:
        worker_main(args)
        return
    layers = BASE.parse_layers(args.layers)
    if args.pending:
        pending = [layer for layer in layers if not layer_done(args.work, layer, args.expected_recipe)]
        print(",".join(map(str, pending)) if pending else "NONE")
        return
    if args.prune_completed_capture:
        if not args.capture_dir:
            raise RuntimeError("--prune-completed-capture requires --capture-dir")
        prune_completed_capture(args, layers)
        return
    if args.smoke:
        smoke_args = argparse.Namespace()
        BASE.smoke(smoke_args)
        return
    if args.oracle:
        if not args.capture_dir:
            raise RuntimeError("--oracle requires --capture-dir")
        LayerCalibRAM.expected_fingerprint = args.capture_plan["capture_fingerprint"]
        assert_disk_free(Path(args.oracle_log), METADATA_WRITE_ALLOWANCE_BYTES, "oracle log")
        oracle_args = argparse.Namespace(
            src=args.src,
            layers=args.layers,
            capture_dir=args.capture_dir,
            lockstep=args.lockstep,
            out_scales=args.out_scales,
            oracle_experts=args.oracle_experts,
            oracle_log=args.oracle_log,
            min_routed=args.min_routed,
        )
        BASE.check_capture = lambda capture_dir, requested: check_capture(
            capture_dir, requested, args.capture_plan
        )
        BASE.oracle(oracle_args)
        return
    if args.assemble:
        if not args.out:
            raise RuntimeError("--assemble requires --out")
        assemble(args)
        return
    if not args.capture_dir:
        raise RuntimeError("--encode requires --capture-dir")
    orchestrate_encode(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
