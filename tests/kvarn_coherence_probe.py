#!/usr/bin/env python3
"""KVarN generation-coherence probe library.

Mathematical (no-eyeball) coherence metrics for the KVarN generation-path
corruption bug class:

  1. Generate greedily from a fixed battery (multiturn chat, tool-bearing
     captured CLI request, plain completions).
  2. Re-score each generated continuation with a FRESH prefill request
     (prompt_logprobs on /v1/completions with prompt_token_ids rendered
     through the model's real chat template).  The prefill/prompt-logprob
     path is the verified-clean evaluation path, so the mean token
     log-probability of the generated span is a corruption-sensitive
     coherence score computed by machinery that never touches the decode
     KV read path under test.
  3. Compare greedy continuations to a golden captured from the clean
     control (nvfp4_ds_mla) with a longest-common-token-prefix metric and
     a token-overlap similarity.

Writes a JSON report.  Thresholds live in test_generation_coherence.py.
"""

from __future__ import annotations

import glob
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

CAPTURED_REQUESTS = os.environ.get(
    "KVARN_CAPTURED_REQUESTS", "/tmp/captured_requests.jsonl"
)
MODEL_SNAPSHOT_GLOB = (
    "/home/js/.cache/huggingface/hub/"
    "models--jpsequeira--GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2/snapshots/*"
)

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        path = sorted(glob.glob(MODEL_SNAPSHOT_GLOB))[0]
        _TOK = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOK


def _post(endpoint: str, path: str, body: dict[str, Any], timeout: int = 900):
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_ready(endpoint: str, timeout: float = 3600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                endpoint.rstrip("/") + "/v1/models", timeout=10
            ) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"server at {endpoint} not ready within {timeout}s")


def load_captured_request(index: int = 0) -> dict[str, Any]:
    with open(CAPTURED_REQUESTS) as f:
        for i, line in enumerate(f):
            if i == index:
                body = json.loads(json.loads(line)["body"])
                body.pop("stream", None)
                body.pop("stream_options", None)
                return body
    raise IndexError(f"captured request {index} not found")


def fixed_battery() -> dict[str, list[dict[str, str]]]:
    """Deterministic conversation battery (messages lists for chat API)."""
    return {
        "sky_blue": [
            {
                "role": "user",
                "content": "Write one short paragraph about why the sky is blue.",
            }
        ],
        "haiku": [
            {"role": "user", "content": "Write a haiku about autumn leaves."}
        ],
        "multiturn": [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "can you think a bit about life"},
            {"role": "user", "content": "what's up?"},
        ],
    }


def chat_generate(
    endpoint: str,
    messages: list[dict[str, str]],
    tools: list | None = None,
    max_tokens: int = 200,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "GLM-5.2",
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
    }
    if tools:
        body["tools"] = tools
    out = _post(endpoint, "/v1/chat/completions", body)
    choice = out["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    token_ids = choice.get("token_ids") or []
    return {
        "text": text,
        "reasoning": reasoning,
        "token_ids": token_ids,
        "full_text": (reasoning + text) if reasoning else text,
    }


def render_tokens(
    messages: list[dict[str, Any]],
    tools: list | None = None,
) -> tuple[list[int], list[int]]:
    """Render messages + final assistant continuation through the real chat
    template.  Returns (full_token_ids, prefix_token_ids) where prefix is the
    rendered conversation with add_generation_prompt=True."""
    tok = _tokenizer()

    def _render(msgs, add_gen):
        ids = tok.apply_chat_template(
            msgs,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_gen,
            reasoning_effort="high",
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(t) for t in ids]

    prefix = _render(messages[:-1], True)
    full = _render(messages, False)
    return full, prefix


def rescore_continuation(
    endpoint: str,
    messages: list[dict[str, Any]],
    tools: list | None = None,
) -> dict[str, Any]:
    """Score the final assistant message of `messages` via the clean prefill
    path.  messages[-1] must be the assistant turn under test."""
    full, prefix = render_tokens(messages, tools)
    n_cont = len(full) - len(prefix)
    if n_cont <= 0:
        return {"error": "empty continuation"}
    try:
        out = _post(
            endpoint,
            "/v1/completions",
            {
                "model": "GLM-5.2",
                "prompt": full,
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 0,
            },
        )
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200]!r}"}
    choice = out["choices"][0]
    pl = choice.get("prompt_logprobs") or []
    # prompt_logprobs[0] is None (no context for the first token); align
    # positions by keeping None placeholders.
    vals = [None] + [
        list(d.values())[0]["logprob"] for d in pl[1:] if d
    ]
    if len(vals) != len(full) or any(v is None for v in vals[1:]):
        return {
            "error": f"token count mismatch: server {len(pl)} vs host {len(full)}"
        }
    cont = [v for v in vals[len(prefix) :] if v is not None]
    tok = _tokenizer()
    cont_ids = full[len(prefix) :]
    return {
        "n_prompt_tokens": len(full),
        "n_continuation_tokens": len(cont),
        "mean_cont_logprob": sum(cont) / len(cont),
        "worst_cont_logprob": min(cont),
        "frac_cont_below_m3": sum(1 for v in cont if v < -3.0) / len(cont),
        "frac_cont_below_m6": sum(1 for v in cont if v < -6.0) / len(cont),
        "cont_token_ids": cont_ids,
        "cont_sample": tok.decode(cont_ids[:40]),
    }


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def token_set_similarity(a: list[int], b: list[int]) -> float:
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0

def compare_reports(
    candidate: dict[str, Any], golden: dict[str, Any]
) -> dict[str, Any]:
    """Token-level comparison of candidate vs golden continuations."""
    out: dict[str, Any] = {}
    for name, case in candidate["cases"].items():
        gcase = golden["cases"].get(name)
        if gcase is None:
            continue
        ct = (case.get("rescore") or {}).get("cont_token_ids") or []
        gt = (gcase.get("rescore") or {}).get("cont_token_ids") or []
        out[name] = {
            "candidate_tokens": len(ct),
            "golden_tokens": len(gt),
            "lcp_tokens": longest_common_prefix_len(ct, gt),
            "jaccard": token_set_similarity(ct, gt),
        }
    return out


def run_battery(
    endpoint: str, max_tokens: int = 200, include_captured: bool = True
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "endpoint": endpoint,
        "ts": time.time(),
        "max_tokens": max_tokens,
        "cases": {},
    }
    battery = fixed_battery()

    for name in ("sky_blue", "haiku"):
        messages = battery[name]
        gen = chat_generate(endpoint, messages, max_tokens=max_tokens)
        rescore_messages = messages + [
            {
                "role": "assistant",
                "content": gen["text"],
                "reasoning_content": gen["reasoning"],
            }
        ]
        report["cases"][name] = {
            "kind": "single_turn",
            "messages": messages,
            "output_text": gen["text"],
            "output_reasoning": gen["reasoning"],
            "token_ids": gen["token_ids"],
            "rescore": rescore_continuation(endpoint, rescore_messages),
        }

    turns = battery["multiturn"]
    history: list[dict[str, Any]] = []
    outputs = []
    for turn in turns:
        gen = chat_generate(endpoint, history + [turn], max_tokens=128)
        history = history + [
            turn,
            {
                "role": "assistant",
                "content": gen["text"],
                "reasoning_content": gen["reasoning"],
            },
        ]
        outputs.append(gen)
    report["cases"]["multiturn"] = {
        "kind": "multiturn",
        "turns": turns,
        "outputs_text": [o["text"] for o in outputs],
        "outputs_reasoning": [o["reasoning"] for o in outputs],
        "token_ids": outputs[-1]["token_ids"],
        "rescore": rescore_continuation(endpoint, history),
    }

    # Tool-bearing long-thinking probe: same 12-tool context as the captured
    # CLI request, but a prompt that provokes several hundred reasoning
    # tokens, so decode-path corruption compounds over many steps.
    if include_captured:
        body = load_captured_request(0)
        tools = body.get("tools")
        tool_math_messages = [
            {
                "role": "user",
                "content": (
                    "Do NOT call any tools for this. Compute the sum of all "
                    "prime numbers below 100 entirely in your head, showing "
                    "your step-by-step reasoning, and then describe (without "
                    "calling it) how you would double-check the result with "
                    "one of the available tools."
                ),
            }
        ]
        gen = chat_generate(
            endpoint, tool_math_messages, tools=tools, max_tokens=512
        )
        rescore_messages = tool_math_messages + [
            {
                "role": "assistant",
                "content": gen["text"],
                "reasoning_content": gen["reasoning"],
            }
        ]
        report["cases"]["tool_math"] = {
            "kind": "tool_math",
            "output_text": gen["text"],
            "output_reasoning": gen["reasoning"],
            "token_ids": gen["token_ids"],
            "rescore": rescore_continuation(
                endpoint, rescore_messages, tools=tools
            ),
        }

    if include_captured:
        body = load_captured_request(0)
        messages = body["messages"]
        tools = body.get("tools")
        gen = chat_generate(
            endpoint, messages, tools=tools, max_tokens=256
        )
        rescore_messages = messages + [
            {
                "role": "assistant",
                "content": gen["text"],
                "reasoning_content": gen["reasoning"],
            }
        ]
        report["cases"]["captured_cli"] = {
            "kind": "captured_cli",
            "output_text": gen["text"],
            "output_reasoning": gen["reasoning"],
            "token_ids": gen["token_ids"],
            "rescore": rescore_continuation(
                endpoint, rescore_messages, tools=tools
            ),
        }

    return report



def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("endpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--no-captured", action="store_true")
    ap.add_argument("--wait-ready", action="store_true")
    ap.add_argument("--golden", help="golden report to compare against")
    args = ap.parse_args()
    if args.wait_ready:
        wait_ready(args.endpoint)
    report = run_battery(
        args.endpoint, max_tokens=args.max_tokens, include_captured=not args.no_captured
    )
    if args.golden and os.path.exists(args.golden):
        with open(args.golden) as f:
            golden = json.load(f)
        report["golden_comparison"] = compare_reports(report, golden)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    for name, case in report["cases"].items():
        rs = case.get("rescore", {})
        print(
            f"{name}: mean_cont_lp={rs.get('mean_cont_logprob')} "
            f"frac<-3={rs.get('frac_cont_below_m3')} "
            f"tokens={rs.get('n_continuation_tokens')}"
        )
    gc = report.get("golden_comparison")
    if gc:
        for name, m in gc.items():
            print(f"{name}: LCP={m['lcp_tokens']} jaccard={m['jaccard']:.3f}")


if __name__ == "__main__":
    main()
