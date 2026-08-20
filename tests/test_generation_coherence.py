"""Generation-coherence regression tests for the KVarN K4V2 TR3 stack.

Flushes the KVarN generation-path corruption bug class WITHOUT eyeballing:
greedy generation from a fixed battery (multiturn chat, tool-bearing
requests), re-scored through the verified-clean prefill prompt-logprob path.

Thresholds calibrated on the clean control (nvfp4_ds_mla, goldens/):
  clean mean continuation logprob in [-0.44, -0.04], frac(<-3) <= 0.008.
A word-salad continuation scores mean logprob around -2..-6 with
frac(<-3) >= 0.3; thresholds sit between with wide margins.

Run:  KVARN_TEST_ENDPOINT=http://localhost:8001 pytest test_generation_coherence.py
The server must already be booted (any KV format) with the GLM-5.2 chat
serving config.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvarn_coherence_probe import run_battery  # noqa: E402

ENDPOINT = os.environ.get("KVARN_TEST_ENDPOINT", "http://localhost:8001")

# Coherence thresholds (see module docstring for calibration).
MEAN_CONT_LOGPROB_MIN = -1.0
FRAC_BELOW_M3_MAX = 0.10
MIN_CONT_TOKENS = 30

CASES_REQUIRING_LONG_OUTPUT = {"sky_blue", "haiku", "multiturn", "tool_math"}


@pytest.fixture(scope="module")
def report() -> dict:
    if not os.environ.get("KVARN_TEST_ENDPOINT"):
        pytest.skip("KVARN_TEST_ENDPOINT not set; no server under test")
    return run_battery(ENDPOINT)


@pytest.mark.parametrize(
    "case", ["sky_blue", "haiku", "multiturn", "tool_math", "captured_cli"]
)
def test_generation_coherence(report: dict, case: str) -> None:
    """Generated text must be high-probability under the clean prefill path."""
    cases = report["cases"]
    assert case in cases, f"case {case} missing from battery report"
    rescore = cases[case].get("rescore") or {}
    assert "error" not in rescore, f"{case}: rescore failed: {rescore.get('error')}"

    mean_lp = rescore["mean_cont_logprob"]
    frac_m3 = rescore["frac_cont_below_m3"]
    n = rescore["n_continuation_tokens"]

    if case in CASES_REQUIRING_LONG_OUTPUT:
        assert n >= MIN_CONT_TOKENS, (
            f"{case}: continuation too short ({n} tokens) to be a meaningful "
            "corruption probe"
        )
    assert mean_lp >= MEAN_CONT_LOGPROB_MIN, (
        f"{case}: mean continuation logprob {mean_lp:.3f} < "
        f"{MEAN_CONT_LOGPROB_MIN} — generated text is word salad under the "
        "clean prefill scorer (KVarN generation-path corruption)"
    )
    assert frac_m3 <= FRAC_BELOW_M3_MAX, (
        f"{case}: {frac_m3:.1%} of continuation tokens below logprob -3 "
        f"(clean baseline <= 0.8%) — decode-path KV corruption"
    )


def test_battery_has_all_cases(report: dict) -> None:
    expected = {"sky_blue", "haiku", "multiturn", "tool_math", "captured_cli"}
    assert expected <= set(report["cases"])


def test_golden_similarity(report: dict) -> None:
    """Soft cross-format check vs the clean-control golden (not gating on
    exact tokens: KVarN quantization legitimately diverges)."""
    golden_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "goldens",
        "nvfp4_control_run1.json",
    )
    if not os.path.exists(golden_path):
        pytest.skip("golden report not present")
    from kvarn_coherence_probe import compare_reports

    with open(golden_path) as f:
        golden = json.load(f)
    cmp = compare_reports(report, golden)
    for name, m in cmp.items():
        assert m["jaccard"] >= 0.15, (
            f"{name}: token jaccard {m['jaccard']:.3f} vs clean-control "
            "golden far below the KVarN-quantization divergence floor"
        )
