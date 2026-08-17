#!/usr/bin/env python3
"""Measured-host preflight for the GLM-5.2 BF16 -> EXL3 B300 conversion.

``--smoke`` never hides a contradiction, but it treats an absent/in-flight BF16
download and a not-yet-created capture plan as PENDING so every independent box
check still runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys


CORPUS_SHA256 = "cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4"
CORPUS_ROWS = 12_228
AXIS_COUNTS = {
    "axis1_general": 3_057,
    "axis2_legal": 3_057,
    "axis3_code_agentic": 3_057,
    "axis4_reasoning_termination": 3_057,
}
AXES = tuple(AXIS_COUNTS)
EXPECTED_SHARDS = 282
FIRST_MOE_LAYER = 3
NUM_LAYERS = 78
NUM_EXPERTS = 256
HIDDEN = 6144
MOE_INTER = 2048
PROJS = ("gate_proj", "up_proj", "down_proj")

EXPECTED_GPUS = 8
EXPECTED_GPU_MIB = 275_040
EXPECTED_GPU_NAME = "NVIDIA B300 SXM6 AC"
EXPECTED_NVLINK_LINES = 144
OWNER_BUILD_ARCH = "10.0"
GIB = 1 << 30
DEFAULT_DISK_RESERVE_BYTES = 256 * GIB
SOURCE_EXPECTED_BYTES = 1_507_000_000_000
ENCODE_WORK_ALLOWANCE_BYTES = 320_000_000_000
ASSEMBLED_OUTPUT_ALLOWANCE_BYTES = 400_000_000_000
PREFLIGHT_WRITE_ALLOWANCE_BYTES = 1 * GIB
NOMINAL_CAPTURE_BYTES = 1_048_576 * 75 * (HIDDEN * 2 + 8)


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, status: str, check: str, evidence) -> None:
        item = {"status": status, "check": check, "evidence": evidence}
        self.items.append(item)
        rendered = evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True)
        print(f"{status:7s} {check}: {rendered}", flush=True)

    def passed(self, check: str, evidence) -> None:
        self.add("PASS", check, evidence)

    def pending(self, check: str, evidence) -> None:
        self.add("PENDING", check, evidence)

    def failed(self, check: str, evidence) -> None:
        self.add("FAIL", check, evidence)

    def counts(self) -> Counter[str]:
        return Counter(item["status"] for item in self.items)


def _existing_ancestor(path: Path) -> Path:
    current = path.expanduser().resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"no existing filesystem ancestor for {path}")
        current = current.parent
    return current


def assert_disk_free(path: Path, write_bytes: int = PREFLIGHT_WRITE_ALLOWANCE_BYTES) -> dict:
    """The preflight itself writes nothing, but it must exercise the disk gate."""
    anchor = _existing_ancestor(path)
    reserve = int(os.environ.get("B300_DISK_RESERVE_BYTES", DEFAULT_DISK_RESERVE_BYTES))
    usage = shutil.disk_usage(anchor)
    required = int(write_bytes) + reserve
    if usage.free < required:
        raise AssertionError(
            f"{usage.free} bytes free at {anchor}; need {write_bytes} + {reserve} reserve"
        )
    return {
        "target": str(path),
        "anchor": str(anchor),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "write_allowance_bytes": int(write_bytes),
        "reserve_bytes": reserve,
    }


def run(command: list[str], timeout: int = 30) -> str:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode:
        raise AssertionError(
            f"command {command!r} exited {result.returncode}: {result.stdout[-2000:]}"
        )
    return result.stdout


def sha256_file(path: Path, chunk: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    size = path.stat().st_size
    if size < 16:
        raise AssertionError(f"zero/truncated shard: {path} ({size} bytes)")
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise AssertionError(f"short safetensors length header: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        if not (2 <= header_len <= 512 << 20):
            raise AssertionError(f"implausible safetensors header length {header_len}: {path}")
        payload = handle.read(header_len)
    if len(payload) != header_len:
        raise AssertionError(f"short safetensors JSON header: {path}")
    header = json.loads(payload)
    max_end = max(
        (entry["data_offsets"][1] for name, entry in header.items() if name != "__metadata__"),
        default=0,
    )
    if size < 8 + header_len + max_end:
        raise AssertionError(f"incomplete shard {path}: {size} < {8 + header_len + max_end}")
    return header


def expert_key(layer: int, expert: int, proj: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def validate_config(config: dict) -> None:
    expected_cfg = {
        "num_hidden_layers": NUM_LAYERS,
        "first_k_dense_replace": FIRST_MOE_LAYER,
        "n_routed_experts": NUM_EXPERTS,
        "hidden_size": HIDDEN,
        "moe_intermediate_size": MOE_INTER,
        "n_group": 1,
        "topk_group": 1,
        "scoring_func": "sigmoid",
    }
    for key, expected in expected_cfg.items():
        if config.get(key) != expected:
            raise AssertionError(f"config {key}={config.get(key)!r}; expected {expected!r}")
    topk = config.get("num_experts_per_tok", config.get("top_k"))
    if topk != 8:
        raise AssertionError(f"config natural routing top-k is {topk!r}; expected 8")
    if config.get("quantization_config"):
        raise AssertionError("source has quantization_config; expected straight BF16")


def validate_source(src: Path) -> dict:
    config_path = src / "config.json"
    index_path = src / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise AssertionError(f"BF16 source lacks config/index: {src}")
    config = json.loads(config_path.read_text())
    validate_config(config)
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AssertionError("invalid or empty safetensors weight_map")
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != EXPECTED_SHARDS:
        raise AssertionError(
            f"source index references {len(shard_names)} unique shards; expected {EXPECTED_SHARDS}"
        )

    headers: dict[str, dict] = {}
    shard_bytes = 0
    for name in shard_names:
        path = src / name
        if not path.is_file():
            raise AssertionError(f"missing source shard: {path}")
        headers[name] = read_safetensors_header(path)
        shard_bytes += path.stat().st_size
    for tensor_name, shard_name in weight_map.items():
        if tensor_name not in headers[shard_name]:
            raise AssertionError(f"index maps absent tensor {tensor_name} -> {shard_name}")

    routed_count = 0
    routed_bytes = 0
    for layer in range(FIRST_MOE_LAYER, NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            for proj in PROJS:
                key = expert_key(layer, expert, proj)
                shard_name = weight_map.get(key)
                if shard_name is None:
                    raise AssertionError(f"missing BF16 routed expert tensor: {key}")
                entry = headers[shard_name][key]
                expected_shape = (
                    [MOE_INTER, HIDDEN] if proj != "down_proj" else [HIDDEN, MOE_INTER]
                )
                if entry["dtype"] != "BF16" or entry["shape"] != expected_shape:
                    raise AssertionError(
                        f"{key}: {entry['dtype']} {entry['shape']} != BF16 {expected_shape}"
                    )
                start, end = entry["data_offsets"]
                if end - start != MOE_INTER * HIDDEN * 2:
                    raise AssertionError(f"{key}: BF16 payload size mismatch")
                routed_count += 1
                routed_bytes += end - start
                for forbidden in (f"{key}_scale", f"{key}_scale_2"):
                    if forbidden in weight_map:
                        raise AssertionError(f"quantized side tensor found: {forbidden}")

    expected_routed = (NUM_LAYERS - FIRST_MOE_LAYER) * NUM_EXPERTS * len(PROJS)
    if routed_count != expected_routed:
        raise AssertionError(f"routed tensor count {routed_count} != {expected_routed}")
    keys = tuple(weight_map)
    if not any(key.startswith("model.layers.78.") for key in keys):
        raise AssertionError("MTP layer 78 tensors absent")
    for layer in range(FIRST_MOE_LAYER, NUM_LAYERS):
        if not any(key.startswith(f"model.layers.{layer}.mlp.shared_experts.") for key in keys):
            raise AssertionError(f"layer {layer}: shared expert tensors absent")
        if f"model.layers.{layer}.mlp.gate.weight" not in weight_map:
            raise AssertionError(f"layer {layer}: router/gate weight absent")
    return {
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "unique_shards": len(shard_names),
        "shard_bytes": shard_bytes,
        "indexed_tensors": len(weight_map),
        "routed_bf16_tensors": routed_count,
        "routed_bf16_bytes": routed_bytes,
    }


def inspect_source_progress(src: Path, *, smoke: bool = False) -> tuple[str, dict]:
    config_path = src / "config.json"
    index_path = src / "model.safetensors.index.json"
    present_files = list(src.glob("*.safetensors")) if src.is_dir() else []
    present_bytes = sum(path.stat().st_size for path in present_files)
    if not config_path.is_file() or not index_path.is_file():
        return "PENDING", {
            "path": str(src),
            "directory_exists": src.is_dir(),
            "config_present": config_path.is_file(),
            "index_present": index_path.is_file(),
            "present_safetensors": len(present_files),
            "present_safetensors_bytes": present_bytes,
            "strict_checks_skipped": "config/index not both available",
        }
    config = json.loads(config_path.read_text())
    validate_config(config)
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AssertionError("invalid or empty safetensors weight_map")
    names = sorted(set(weight_map.values()))
    if len(names) != EXPECTED_SHARDS:
        raise AssertionError(f"index has {len(names)} shards; expected {EXPECTED_SHARDS}")
    missing = [name for name in names if not (src / name).is_file()]
    zero = [name for name in names if (src / name).is_file() and (src / name).stat().st_size == 0]
    if missing or zero:
        complete = [name for name in names if name not in set(missing) and name not in set(zero)]
        unreadable: list[dict[str, str]] = []
        for name in complete:
            try:
                read_safetensors_header(src / name)
            except Exception as exc:
                unreadable.append({"name": name, "error": str(exc)})
        if unreadable and not (smoke and present_bytes < SOURCE_EXPECTED_BYTES):
            raise AssertionError(f"present source shard header invalid: {unreadable[:3]}")
        return "PENDING", {
            "path": str(src),
            "indexed_shards": len(names),
            "complete_shards": len(complete),
            "missing_shards": len(missing),
            "zero_shards": len(zero),
            "unreadable_present_shards": unreadable[:8],
            "complete_bytes": sum((src / name).stat().st_size for name in complete),
            "strict_checks_skipped": "download incomplete",
        }
    try:
        return "PASS", validate_source(src)
    except Exception as exc:
        if smoke and present_bytes < SOURCE_EXPECTED_BYTES:
            return "PENDING", {
                "path": str(src),
                "indexed_shards": len(names),
                "present_safetensors_bytes": present_bytes,
                "expected_source_bytes": SOURCE_EXPECTED_BYTES,
                "strict_checks_skipped": "download appears in flight",
                "deferred_validation_error": str(exc),
            }
        raise


def validate_corpus(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"owner corpus absent: {path}")
    digest = sha256_file(path)
    if digest != CORPUS_SHA256:
        raise AssertionError(f"corpus sha256 {digest} != owner-pinned {CORPUS_SHA256}")
    axes: Counter[str] = Counter()
    rows = 0
    meta_tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("text"), str):
                raise AssertionError(f"corpus line {line_no}: missing string text")
            axes[record.get("axis")] += 1
            meta_tokens += int(record.get("meta", {}).get("calib_tokens", 0))
            rows += 1
    if rows != CORPUS_ROWS:
        raise AssertionError(f"corpus has {rows} rows; expected {CORPUS_ROWS}")
    if dict(axes) != AXIS_COUNTS:
        raise AssertionError(f"corpus axes {dict(axes)} != {AXIS_COUNTS}")
    return {
        "path": str(path),
        "sha256": digest,
        "rows": rows,
        "axes": dict(sorted(axes.items())),
        "metadata_calib_tokens": meta_tokens,
    }


def validate_plan(path: Path, src: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan.get("schema") != "glm52-b300-capture-plan-v1":
        raise AssertionError(f"unexpected capture plan schema: {plan.get('schema')!r}")
    if plan.get("corpus_sha256") != CORPUS_SHA256 or plan.get("calibration_baseline") is not True:
        raise AssertionError("capture plan is not the mandatory owner calibration baseline")
    if plan.get("owner_corpus_only") is not True:
        raise AssertionError("capture plan does not declare owner-corpus-only calibration")
    if int(plan.get("corpus_rows", -1)) != CORPUS_ROWS or plan.get("axis_rows") != AXIS_COUNTS:
        raise AssertionError("capture plan corpus row/axis counts do not match the owner baseline")
    if plan.get("selection_policy") != "owner-corpus-axis-separated-luke-multipass-no-repeat-v1":
        raise AssertionError("capture plan selection policy is not the Luke-style owner baseline")
    routing = plan.get("routing")
    expected_routing = {
        "natural": True,
        "forced_expert_activation": False,
        "scoring_func": "sigmoid",
        "top_k": 8,
        "n_group": 1,
        "topk_group": 1,
    }
    if routing != expected_routing:
        raise AssertionError(f"capture routing {routing} != {expected_routing}")
    if int(plan.get("capture_tp", -1)) != 8 or int(plan.get("output_tp", -1)) != 4:
        raise AssertionError("capture/output TP must be TP8/TP4")
    if [item.get("axis") for item in plan.get("passes", [])] != list(AXES):
        raise AssertionError("plan must contain one ordered pass per owner axis")
    canonical = dict(plan)
    fingerprint = canonical.pop("capture_fingerprint", None)
    got = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != got:
        raise AssertionError(f"capture plan fingerprint {fingerprint} != {got}")
    config_path = src / "config.json"
    index_path = src / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise AssertionError("cannot bind capture plan to source: config/index absent")
    current_source = {
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
    }
    if plan.get("source") != current_source:
        raise AssertionError(
            f"capture plan source mismatch: {plan.get('source')} != {current_source}"
        )
    total_tokens = int(plan["total_tokens"])
    capture_bytes = total_tokens * 75 * (HIDDEN * 2 + 8)
    if int(plan.get("capture_bytes", -1)) != capture_bytes:
        raise AssertionError("capture plan byte arithmetic mismatch")
    return {
        "path": str(path),
        "capture_fingerprint": fingerprint,
        "passes": len(plan["passes"]),
        "samples": sum(len(item["samples"]) for item in plan["passes"]),
        "tokens_per_layer": total_tokens,
        "capture_bytes": capture_bytes,
        "capture_gib": capture_bytes / GIB,
        "source": current_source,
    }


def check_hardware() -> dict:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise AssertionError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {"index": int(fields[0]), "name": fields[1], "memory_mib": int(fields[2]), "cc": fields[3]}
        )
    if len(rows) != EXPECTED_GPUS:
        raise AssertionError(f"found {len(rows)} GPUs; expected {EXPECTED_GPUS}")
    for expected_index, row in enumerate(rows):
        if row["index"] != expected_index:
            raise AssertionError(f"GPU indices are not 0..7: {rows}")
        if row["name"] != EXPECTED_GPU_NAME or row["memory_mib"] != EXPECTED_GPU_MIB:
            raise AssertionError(f"unexpected GPU identity/capacity: {row}")
        if not row["cc"].startswith("10."):
            raise AssertionError(f"unexpected B300 compute capability: {row}")
    return {
        "gpus": rows,
        "aggregate_mib": sum(row["memory_mib"] for row in rows),
        "aggregate_gib": sum(row["memory_mib"] for row in rows) / 1024,
        "aggregate_tib": sum(row["memory_mib"] for row in rows) / 1024 / 1024,
        "owner_build_arch": OWNER_BUILD_ARCH,
    }


def check_nvlink() -> dict:
    output = run(["nvidia-smi", "nvlink", "-s"])
    lines = [line for line in output.splitlines() if re.match(r"^\s*Link \d+:.*GB/s", line)]
    if len(lines) != EXPECTED_NVLINK_LINES:
        raise AssertionError(f"active NVLink speed lines={len(lines)}; expected 144")
    return {"active_speed_lines": len(lines), "sample": lines[:2]}


def meminfo() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields = value.split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    if total < 2_000 * GIB:
        raise AssertionError(f"host RAM {total / GIB:.2f} GiB; expected at least 2000 GiB")
    return {"total_bytes": total, "total_gib": total / GIB, "available_bytes": available, "available_gib": available / GIB}


def check_shm() -> dict:
    path = Path("/dev/shm")
    usage = shutil.disk_usage(path)
    stat = os.statvfs(path)
    swap_lines = Path("/proc/swaps").read_text().splitlines()[1:]
    swap_used = sum(int(line.split()[3]) * 1024 for line in swap_lines if line.split())
    window_bytes = 8 * 1_048_576 * (HIDDEN * 2 + 8)
    if usage.free < window_bytes + 64 * GIB:
        raise AssertionError("/dev/shm cannot hold an eight-layer window plus 64 GiB reserve")
    return {
        "path": str(path),
        "capacity_bytes": usage.total,
        "capacity_gib": usage.total / GIB,
        "free_bytes": usage.free,
        "nominal_full_capture_bytes": NOMINAL_CAPTURE_BYTES,
        "nominal_full_capture_gib": NOMINAL_CAPTURE_BYTES / GIB,
        "eight_layer_window_bytes": window_bytes,
        "swap_entries": len(swap_lines),
        "swap_used_bytes": swap_used,
        "block_size": stat.f_frsize,
        "policy": "capture-data windows required; overlay forbidden",
    }


def check_python_cuda() -> tuple[dict, str]:
    if sys.version_info[:2] != (3, 12):
        raise AssertionError(f"Python {platform.python_version()}; expected 3.12.x")
    nvcc = run(["nvcc", "--version"])
    match = re.search(r"release\s+(\d+\.\d+)", nvcc)
    if not match:
        raise AssertionError("could not parse nvcc release")
    return ({"executable": sys.executable, "python": platform.python_version(), "nvcc_release": match.group(1)}, match.group(1))


def check_torch(nvcc_release: str) -> tuple[str, dict]:
    try:
        import torch
    except Exception as exc:
        return "PENDING", {"reason": f"torch not importable in {sys.executable}: {exc}"}
    evidence = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "nvcc_release": nvcc_release,
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() != EXPECTED_GPUS:
        raise AssertionError(f"Torch CUDA visibility mismatch: {evidence}")
    evidence["gpu0"] = torch.cuda.get_device_name(0)
    evidence["capability0"] = list(torch.cuda.get_device_capability(0))
    torch_major = int(str(torch.version.cuda).split(".", 1)[0])
    nvcc_major = int(nvcc_release.split(".", 1)[0])
    if torch_major != nvcc_major:
        raise AssertionError(f"Torch/nvcc CUDA major mismatch: {evidence}")
    return "PASS", evidence


def projected_disk_need(src: Path) -> int:
    source_bytes = sum(path.stat().st_size for path in src.glob("*.safetensors")) if src.is_dir() else 0
    remaining_source = max(0, SOURCE_EXPECTED_BYTES - source_bytes)
    return (
        remaining_source
        + ENCODE_WORK_ALLOWANCE_BYTES
        + ASSEMBLED_OUTPUT_ALLOWANCE_BYTES
        + DEFAULT_DISK_RESERVE_BYTES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-5.2 B300 measured-host preflight")
    parser.add_argument("--smoke", action="store_true", help="allow only expected source/plan pending state")
    parser.add_argument("--src", default="/workspace/bf16", type=Path)
    parser.add_argument(
        "--corpus", default="/workspace/tr3/calib/reap_recall_calib.jsonl", type=Path
    )
    parser.add_argument("--plan", default="/workspace/tr3/capture_plan.json", type=Path)
    parser.add_argument("--work-root", default="/workspace/tr3", type=Path)
    args = parser.parse_args()
    src = args.src.resolve()
    corpus = args.corpus.resolve()
    plan = args.plan.resolve()
    work_root = args.work_root.resolve()
    report = Report()

    checks = [
        ("disk guard", lambda: assert_disk_free(work_root)),
        ("B300 hardware", check_hardware),
        ("NVLink", check_nvlink),
        ("host RAM", meminfo),
        ("host-RAM capture", check_shm),
        ("owner calibration corpus", lambda: validate_corpus(corpus)),
    ]
    for name, function in checks:
        try:
            report.passed(name, function())
        except Exception as exc:
            report.failed(name, str(exc))

    try:
        disk = shutil.disk_usage(_existing_ancestor(work_root))
        needed = projected_disk_need(src)
        if disk.free < needed:
            raise AssertionError(f"projected free need={needed}, actual free={disk.free}")
        report.passed("projected conversion disk", {"free_bytes": disk.free, "required_bytes": needed})
    except Exception as exc:
        report.failed("projected conversion disk", str(exc))

    nvcc_release = "unknown"
    try:
        evidence, nvcc_release = check_python_cuda()
        report.passed("Python/CUDA toolkit", evidence)
    except Exception as exc:
        report.failed("Python/CUDA toolkit", str(exc))
    if nvcc_release != "unknown":
        try:
            status, evidence = check_torch(nvcc_release)
            report.add(status, "Torch CUDA runtime/build pairing", evidence)
        except Exception as exc:
            report.failed("Torch CUDA runtime/build pairing", str(exc))

    guide = Path("/etc/vast-agents-guide.md")
    base_guide = Path("/etc/vast_agents/base.md")
    if guide.is_file() and base_guide.is_file():
        report.passed("Vast supervisor guide", {"guide": str(guide), "base": str(base_guide)})
    else:
        report.failed("Vast supervisor guide", "required operating guide missing")

    try:
        status, evidence = inspect_source_progress(src, smoke=args.smoke)
        if status == "PENDING" and not args.smoke:
            report.failed("BF16 source", evidence)
        else:
            report.add(status, "BF16 source", evidence)
    except Exception as exc:
        report.failed("BF16 source", str(exc))

    if plan.is_file():
        try:
            report.passed("capture plan", validate_plan(plan, src))
        except Exception as exc:
            report.failed("capture plan", str(exc))
    elif args.smoke:
        report.pending("capture plan", {"path": str(plan), "reason": "build after tokenizer/source arrives"})
    else:
        report.failed("capture plan", f"absent: {plan}")

    counts = report.counts()
    summary = {"counts": dict(counts), "smoke": args.smoke, "checks": report.items}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if counts["FAIL"]:
        print(f"PREFLIGHT FAILED: {counts['FAIL']} contradiction(s)")
        raise SystemExit(1)
    if not args.smoke and counts["PENDING"]:
        print(f"STRICT PREFLIGHT FAILED: {counts['PENDING']} pending check(s)")
        raise SystemExit(1)
    mode = "SMOKE" if args.smoke else "STRICT"
    print(f"{mode} PREFLIGHT PASSED: {counts['PASS']} pass, {counts['PENDING']} pending")


if __name__ == "__main__":
    main()
