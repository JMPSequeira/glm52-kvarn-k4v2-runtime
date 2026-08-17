#!/usr/bin/env python3
"""Build/load exllamav3 0.0.43's six quantizer ops without importing its package."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import site
import subprocess
import sys
import sysconfig


# The owner requires an sm_100 build even though this driver's runtime reports
# the B300 devices as compute capability 10.3. Keep build target and runtime
# observation separate.
ARCH_LIST = "10.0"
REQUIRED_VERSION = "0.0.43"
REQUIRED_OPS = (
    "had_r_128",
    "pack_trellis",
    "quantize_tiles",
    "reconstruct",
    "reconstruct_slice",
    "unpack_trellis",
)
GIB = 1 << 30
DEFAULT_BUILD_DIR = Path("/workspace/tr3/torch_extensions/exllamav3_ext_sm100")
DEFAULT_DISK_RESERVE_BYTES = 256 * GIB
EXT_BUILD_ALLOWANCE_BYTES = 32 * GIB


def _existing_ancestor(path: Path) -> Path:
    current = path.expanduser().resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"no existing filesystem ancestor for {path}")
        current = current.parent
    return current


def assert_disk_free(path: Path, write_bytes: int, label: str) -> dict:
    reserve = int(os.environ.get("B300_DISK_RESERVE_BYTES", DEFAULT_DISK_RESERVE_BYTES))
    anchor = _existing_ancestor(path)
    usage = shutil.disk_usage(anchor)
    required = int(write_bytes) + reserve
    if usage.free < required:
        raise RuntimeError(
            f"DISK GUARD {label}: {usage.free} bytes free at {anchor}; "
            f"need {write_bytes} write bytes + {reserve} reserve = {required}"
        )
    result = {
        "target": str(path),
        "anchor": str(anchor),
        "free_bytes": usage.free,
        "write_allowance_bytes": int(write_bytes),
        "reserve_bytes": reserve,
    }
    print(f"DISK GUARD PASS: {label}: {result}", flush=True)
    return result


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        dist = importlib.metadata.distribution("exllamav3")
        roots.append(Path(dist.locate_file("exllamav3/exllamav3_ext")))
    except importlib.metadata.PackageNotFoundError:
        pass
    prefixes: list[str] = []
    try:
        prefixes.extend(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        prefixes.append(user_site)
    purelib = sysconfig.get_path("purelib")
    if purelib:
        prefixes.append(purelib)
    prefixes.extend(path for path in sys.path if path)
    roots.extend(Path(path) / "exllamav3" / "exllamav3_ext" for path in prefixes)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def find_source_root() -> Path:
    for root in _candidate_roots():
        if (root / "bindings.cpp").is_file() and (root / "quant" / "quantize.cu").is_file():
            return root.resolve()
    searched = "\n  ".join(str(path) for path in _candidate_roots())
    raise FileNotFoundError(
        "exllamav3 extension sources not found. Install exllamav3==0.0.43 "
        f"with its source package. Searched:\n  {searched}"
    )


def _assert_version() -> str:
    try:
        version = importlib.metadata.version("exllamav3")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("exllamav3 is not installed in this Python environment") from exc
    if version.split("+", 1)[0] != REQUIRED_VERSION:
        raise RuntimeError(f"encoder is pinned to exllamav3 {REQUIRED_VERSION}; found {version}")
    return version


def observed_ops(ext: object) -> list[str]:
    return sorted(
        name
        for name in dir(ext)
        if not name.startswith("_") and callable(getattr(ext, name, None))
    )


def _assert_ops(ext: object) -> list[str]:
    actual = observed_ops(ext)
    missing = [name for name in REQUIRED_OPS if name not in actual]
    if missing:
        raise RuntimeError(f"exllamav3_ext missing {missing}; actual callable ops={actual}")
    return actual


def _nvcc_release(cuda_home: str) -> str:
    output = subprocess.run(
        [str(Path(cuda_home) / "bin" / "nvcc"), "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if not match:
        raise RuntimeError(f"could not parse nvcc release: {output}")
    return match.group(1)


def build(*, verbose: bool | None = None):
    """Return the guarded sm_100 module and register it as ``exllamav3_ext``."""
    existing = sys.modules.get("exllamav3_ext")
    if existing is not None:
        actual = _assert_ops(existing)
        metadata = getattr(existing, "_b300_bootstrap", {})
        if metadata.get("arch_list") != ARCH_LIST:
            raise RuntimeError(
                f"refusing unverified/preloaded extension: metadata={metadata!r}, ops={actual}"
            )
        return existing

    version = _assert_version()
    source_root = find_source_root()
    sources = sorted(
        str(path.resolve())
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in {".c", ".cpp", ".cu"}
    )
    if not sources:
        raise RuntimeError(f"no extension sources under {source_root}")

    os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH_LIST
    os.environ.setdefault("MAX_JOBS", "32")
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not CUDA_HOME:
        raise RuntimeError("torch.utils.cpp_extension did not resolve CUDA_HOME")
    nvcc_release = _nvcc_release(CUDA_HOME)
    torch_cuda = str(torch.version.cuda)
    if int(nvcc_release.split(".", 1)[0]) != int(torch_cuda.split(".", 1)[0]):
        raise RuntimeError(
            f"CUDA compiler/PyTorch major mismatch: CUDA_HOME={CUDA_HOME}, "
            f"nvcc={nvcc_release}, torch CUDA={torch_cuda}"
        )

    build_path = Path(os.environ.get("EXL3_B300_BUILD_DIR", str(DEFAULT_BUILD_DIR)))
    build_path = build_path.expanduser().resolve(strict=False)
    assert_disk_free(build_path, EXT_BUILD_ALLOWANCE_BYTES, "extension build")
    build_path.mkdir(parents=True, exist_ok=True)
    assert_disk_free(build_path, EXT_BUILD_ALLOWANCE_BYTES, "extension build pre-load")

    extra_cuda_cflags = [
        "-lineinfo",
        "-O3",
        "--use_fast_math",
        "-Xcudafe",
        "--diag_suppress=177",
        "-Xcudafe",
        "--diag_suppress=20012",
    ]
    cuda_host_cxx = os.environ.get("CUDAHOSTCXX")
    if cuda_host_cxx:
        extra_cuda_cflags += ["-ccbin", cuda_host_cxx]
    if verbose is None:
        verbose = os.environ.get("EXL3_B300_EXT_VERBOSE", "0") == "1"
    ext = load(
        name="exllamav3_ext",
        sources=sources,
        extra_include_paths=[str(source_root)],
        extra_cflags=["-Ofast"],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=[],
        build_directory=str(build_path),
        verbose=verbose,
    )
    actual = _assert_ops(ext)
    sys.modules["exllamav3_ext"] = ext
    ext.__dict__.setdefault("_b300_bootstrap", {})
    ext.__dict__["_b300_bootstrap"].update(
        {
            "arch_list": ARCH_LIST,
            "exllamav3": version,
            "source_root": str(source_root),
            "build_directory": str(build_path),
            "nvcc_release": nvcc_release,
            "torch_cuda": torch_cuda,
            "observed_ops": actual,
        }
    )
    return ext


def _functional_smoke(ext: object, torch) -> dict:
    device = torch.device("cuda:0")
    input_tiles = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 256, device=device),
            torch.linspace(0.75, -0.25, 256, device=device),
        )
    ).contiguous()
    output_tiles = torch.zeros_like(input_tiles)
    output_indices = torch.zeros((2, 256), device=device, dtype=torch.int16)
    temp_costs = torch.zeros((2, 2, 8192), device=device, dtype=torch.float16)
    temp_edges = torch.zeros((2, 256, 8192), device=device, dtype=torch.int16)
    ext.quantize_tiles(
        input_tiles, output_tiles, output_indices, temp_costs, temp_edges, 3, True, False
    )

    # pack_trellis stores the low K bits of cyclic 16-bit trellis states;
    # quantize_tiles supplies a valid state path. Independent values such as
    # arange(256) % 2**K are not a valid input contract for an identity test.
    encoded = output_indices.view(1, 2, 256).contiguous()
    packed_tiles = torch.zeros((1, 2, 48), device=device, dtype=torch.int16)
    unpacked = torch.zeros_like(encoded)
    ext.pack_trellis(packed_tiles, encoded, 3)
    ext.unpack_trellis(unpacked, packed_tiles, 3)
    torch.cuda.synchronize()
    if not torch.equal(unpacked, encoded):
        raise RuntimeError("valid trellis-state pack/unpack CUDA round-trip mismatch")
    if torch.equal(packed_tiles[:, 0], packed_tiles[:, 1]):
        raise RuntimeError("distinct quantized smoke tiles produced identical trellis data")

    packed = torch.cat(
        (packed_tiles[:, 0:1].repeat(1, 8, 1), packed_tiles[:, 1:2].repeat(1, 8, 1)),
        dim=1,
    ).contiguous()
    reconstructed = torch.empty((16, 256), device=device, dtype=torch.float16)
    ext.reconstruct(reconstructed, packed, 3, True, False)
    reconstructed_slice = torch.empty((16, 128), device=device, dtype=torch.float16)
    ext.reconstruct_slice(reconstructed_slice, packed, 3, True, False, 128)

    had_out = torch.empty((1, 128), device=device, dtype=torch.float32)
    ext.had_r_128(
        torch.arange(128, device=device, dtype=torch.float32).view(1, 128),
        had_out,
        None,
        None,
        1.0,
    )
    torch.cuda.synchronize()
    if not torch.equal(reconstructed_slice, reconstructed[:, 128:]):
        raise RuntimeError("reconstruct_slice differs from full reconstruct")
    if not bool(torch.isfinite(reconstructed).all()):
        raise RuntimeError("reconstruct produced non-finite output")
    if not bool(torch.isfinite(output_tiles).all()) or not bool(torch.isfinite(had_out).all()):
        raise RuntimeError("quantize_tiles/had_r_128 produced non-finite output")
    return {
        "pack_trellis_launched": True,
        "unpack_trellis_launched": True,
        "valid_trellis_pack_unpack_identity": True,
        "reconstruct_launched": True,
        "reconstruct_slice_launched": True,
        "reconstruct_slice_matches_full": True,
        "reconstruct_finite": True,
        "quantize_tiles_launched": True,
        "quantize_tiles_finite": True,
        "had_r_128_launched": True,
        "had_r_128_finite": True,
    }


def _cubin_arches(module_path: str) -> list[str]:
    tool = shutil.which("cuobjdump")
    if not tool or not module_path:
        return []
    result = subprocess.run(
        [tool, "--list-elf", module_path],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return sorted(set(re.findall(r"sm_[0-9a-z]+", result.stdout)))


def hardware_check() -> None:
    ext = build(verbose=True)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError(
            f"expected eight CUDA devices, available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}"
        )
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if cap != (10, 3) or "B300" not in name:
        raise RuntimeError(f"expected B300-class CC 10.3 hardware, got {name!r} {cap}")
    actual = _assert_ops(ext)
    smoke = _functional_smoke(ext, torch)
    arches = _cubin_arches(str(getattr(ext, "__file__", "")))
    print(
        "EXL3 EXT OK: "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={name!r} capability={cap} build_arch={ARCH_LIST} "
        f"module={getattr(ext, '__file__', None)} cubins={arches} "
        f"source={find_source_root()}"
    )
    print(f"ACTUAL EXT OPS: {actual}")
    print(f"CUDA OP SMOKE: {smoke}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="flash-attn-free exllamav3 0.0.43 extension loader for B300"
    )
    parser.add_argument("--check", action="store_true", help="build and launch CUDA/op smoke")
    parser.add_argument("--print-source", action="store_true", help="print source root without torch")
    args = parser.parse_args()
    if args.print_source:
        _assert_version()
        print(find_source_root())
    elif args.check:
        hardware_check()
    else:
        build()


if __name__ == "__main__":
    main()
