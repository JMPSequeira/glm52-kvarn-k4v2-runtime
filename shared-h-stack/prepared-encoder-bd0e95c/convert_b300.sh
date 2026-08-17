#!/usr/bin/env bash
# Real-box helper for the owner-locked GLM-5.2 BF16 -> EXL3 conversion.
# Long capture/encode/assemble invocations must be launched in the foreground
# by a Vast supervisor program; this file never daemonizes a job.
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-/usr/bin/python3}"
BF16_SRC="${BF16_SRC:-/workspace/bf16}"
OWNER_CORPUS="${OWNER_CORPUS:-/workspace/tr3/calib/reap_recall_calib.jsonl}"
BASE_ENCODER_PY="${BASE_ENCODER_PY:-/workspace/tr3/encode_tr3_v31.py}"
WORK_ROOT="${WORK_ROOT:-/workspace/tr3}"
WORK_DIR="${WORK_DIR:-${WORK_ROOT}/encode-work}"
OUT_DIR="${OUT_DIR:-${WORK_ROOT}/output/GLM-5.2-EXL3-TR3}"
CAPTURE_PLAN="${CAPTURE_PLAN:-${WORK_ROOT}/capture_plan.json}"
CAPTURE_DIR="${CAPTURE_DIR:-/dev/shm/glm52-tr3-capture}"
LAYERS="${LAYERS:-3-10}"
MAX_WINDOW_LAYERS="${MAX_WINDOW_LAYERS:-8}"
DISK_RESERVE_BYTES="${B300_DISK_RESERVE_BYTES:-274877906944}"
RAMFS_RESERVE_BYTES="${B300_RAMFS_RESERVE_BYTES:-68719476736}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
EXL3_B300_BUILD_DIR="${EXL3_B300_BUILD_DIR:-${WORK_ROOT}/torch_extensions/exllamav3_ext_sm100}"
MAX_JOBS="${MAX_JOBS:-32}"

export CUDA_HOME EXL3_B300_BUILD_DIR MAX_JOBS
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

die() {
    echo "FATAL: $*" >&2
    exit 1
}

existing_ancestor() {
    local target="$1"
    while [[ ! -e "$target" ]]; do
        [[ "$target" != "/" ]] || break
        target="$(dirname -- "$target")"
    done
    printf '%s\n' "$target"
}

disk_guard() {
    local target="$1" write_bytes="$2" label="$3"
    local anchor free required
    anchor="$(existing_ancestor "$target")"
    free="$(df -B1 --output=avail "$anchor" | tail -n 1 | tr -d ' ')"
    required=$((write_bytes + DISK_RESERVE_BYTES))
    (( free >= required )) || \
        die "DISK GUARD ${label}: free=${free}, need write=${write_bytes} + reserve=${DISK_RESERVE_BYTES}"
    echo "DISK GUARD PASS ${label}: target=${target} free=${free} write=${write_bytes} reserve=${DISK_RESERVE_BYTES}"
}

ram_guard() {
    local target="$1" write_bytes="$2" label="$3"
    local anchor fs_type free required
    anchor="$(existing_ancestor "$target")"
    fs_type="$(findmnt -n -o FSTYPE --target "$anchor")"
    [[ "$fs_type" == "tmpfs" || "$fs_type" == "ramfs" ]] || \
        die "RAM GUARD ${label}: ${target} is backed by ${fs_type}"
    free="$(df -B1 --output=avail "$anchor" | tail -n 1 | tr -d ' ')"
    required=$((write_bytes + RAMFS_RESERVE_BYTES))
    (( free >= required )) || \
        die "RAM GUARD ${label}: free=${free}, need write=${write_bytes} + reserve=${RAMFS_RESERVE_BYTES}"
    echo "RAM GUARD PASS ${label}: target=${target} fs=${fs_type} free=${free}"
}

compile_check() {
    disk_guard "$WORK_ROOT" 1073741824 "compile metadata"
    "$PYTHON" -m py_compile \
        "$SCRIPT_DIR/bootstrap_ext_b300.py" \
        "$SCRIPT_DIR/capture_b300.py" \
        "$SCRIPT_DIR/encode_b300.py" \
        "$SCRIPT_DIR/preflight_b300.py"
}

smoke_preflight() {
    compile_check
    "$PYTHON" "$SCRIPT_DIR/preflight_b300.py" --smoke \
        --src "$BF16_SRC" --corpus "$OWNER_CORPUS" \
        --plan "$CAPTURE_PLAN" --work-root "$WORK_ROOT"
}

build_ext() {
    disk_guard "$EXL3_B300_BUILD_DIR" 34359738368 "extension build"
    "$PYTHON" "$SCRIPT_DIR/bootstrap_ext_b300.py" --check
}

build_plan() {
    disk_guard "$CAPTURE_PLAN" 1073741824 "capture plan"
    "$PYTHON" "$SCRIPT_DIR/capture_b300.py" --plan \
        --src "$BF16_SRC" --corpus "$OWNER_CORPUS" --plan-file "$CAPTURE_PLAN"
}

capture_window() {
    [[ -f "$CAPTURE_PLAN" ]] || die "capture plan absent: $CAPTURE_PLAN"
    local payload
    payload="$($PYTHON -c 'import json,sys; p=json.load(open(sys.argv[1])); parts=[x.strip() for x in sys.argv[2].split(",") if x.strip()]; layers=set(sum(([int(x)] if "-" not in x else list(range(int(x.split("-",1)[0]),int(x.split("-",1)[1])+1)) for x in parts),[])); print(int(p["total_tokens"])*len(layers)*(6144*2+8))' "$CAPTURE_PLAN" "$LAYERS")"
    ram_guard "$CAPTURE_DIR" "$payload" "capture window $LAYERS"
    "$PYTHON" "$SCRIPT_DIR/capture_b300.py" --capture \
        --src "$BF16_SRC" --corpus "$OWNER_CORPUS" --plan-file "$CAPTURE_PLAN" \
        --capture-dir "$CAPTURE_DIR" --layers "$LAYERS" \
        --max-window-layers "$MAX_WINDOW_LAYERS"
}

encode_window() {
    disk_guard "$WORK_DIR" 34359738368 "encode window $LAYERS"
    ram_guard "$CAPTURE_DIR" 0 "encoded capture source"
    "$PYTHON" "$SCRIPT_DIR/encode_b300.py" --encode --consume-capture \
        --base-encoder "$BASE_ENCODER_PY" --src "$BF16_SRC" --work "$WORK_DIR" \
        --capture-manifest "$CAPTURE_PLAN" --capture-dir "$CAPTURE_DIR" \
        --layers "$LAYERS" --workers 8 --gpus 8
}

assemble_output() {
    disk_guard "$OUT_DIR" 549755813888 "assembled checkpoint"
    "$PYTHON" "$SCRIPT_DIR/encode_b300.py" --assemble \
        --base-encoder "$BASE_ENCODER_PY" --src "$BF16_SRC" --work "$WORK_DIR" \
        --capture-manifest "$CAPTURE_PLAN" --out "$OUT_DIR" --io-workers 8
}

case "${1:-preflight}" in
    preflight) smoke_preflight ;;
    ext) build_ext ;;
    plan) build_plan ;;
    capture-window) capture_window ;;
    encode-window) encode_window ;;
    assemble) assemble_output ;;
    status)
        disk_guard "$WORK_ROOT" 1073741824 "status"
        df -h /workspace /dev/shm
        ;;
    *)
        die "usage: $0 {preflight|ext|plan|capture-window|encode-window|assemble|status}; set LAYERS to an <=8-layer comma list"
        ;;
esac
