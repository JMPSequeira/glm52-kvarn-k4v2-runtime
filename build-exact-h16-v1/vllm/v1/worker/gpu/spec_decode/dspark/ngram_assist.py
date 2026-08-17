# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark n-gram assist: host-side per-slot draft-token override.

Live implementation of the promoted offline rule
(``dspark-alignment/ngram_assist_sim/ngram_assist_sim.py``:
rule=override, k=5, corpus dAL +0.873):

  * Index (one per request, host-side): rightmost-occurrence k-gram ->
    continuation over the request's authoritative token-id stream
    (prompt tokens staged at admission + accepted tokens as decode
    advances). O(1) amortized work per token; the most recent
    ``max_index_tokens`` k-grams are retained (deterministic
    lazy-deletion window).
  * Assist (per draft slot j): query = last k tokens of
    (stream + tokens already chosen for slots < j). On an exact match
    the proposed token is overridden with the matched continuation;
    otherwise the drafter's token stands. This is the dynamic per-slot
    re-query the offline sweep promoted (pre-staged run continuation
    scored -0.005 dAL; the benefit lives entirely here).

Application point: HOST-SIDE, after the captured draft graph has
replayed and before the proposal tensor is returned by ``propose()``.
The FULL draft CUDA graph (parallel backbone forward + sequential
Markov sampling) is untouched, so capture and replay semantics are
bit-identical to the unassisted drafter.

Exactness (greedy verify): the target accepts a draft token iff it
equals the target's own greedy argmax at that position, and the bonus
token is the target's argmax. Changing the proposal source therefore
cannot change the served token stream: greedy completions are
token-for-token identical, and only acceptance length (throughput)
moves. This mirrors the adaptive-depth controller's contract
(see ``adaptive_depth.py``: host-side decisions, served outputs
bit-identical under greedy verify).

Documented deviations from the offline replay, both forced by FULL
graph capture and both visible in the JSONL log:

  1. The in-graph Markov chain conditions slot j+1 on the drafter's
     own slot-j token, not the overridden one. The override still
     enters slot j+1's n-gram QUERY (chosen-prefix re-query), which is
     where the offline gain lives.
  2. Overrides may land on slots that the verification capacity
     manager later drops (``limit_draft_tokens`` / per-request
     capacity); the log's slot indices are proposal slots.

Fail-closed: default-off, and at runtime an empty index, unset token
stream, batch above the guard, dummy/profile run, or any internal
error all leave the drafter's proposal exactly as the graph produced
it. Deterministic: same inputs -> same proposals (pure dict lookups,
rightmost occurrence wins, no sampling, no timing-dependent state).

Overhead: one pinned D2H of ``total_len`` [max_num_reqs] int32, one
pinned D2H of the [num_reqs, n_slots] draft block, one stream
synchronize (the graph must finish before host lookups), num_reqs x
n_slots tuple-hash lookups, and one pinned H2D write-back only when an
override fired. The synchronize is the dominant term (it collapses
async-scheduling run-ahead for this step); the Python loop is bounded
by ``VLLM_DSPARK_NGRAM_MAX_BATCH``.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# Purge assist state for request slots not observed for this many
# assisted iterations (finished or preempted-away requests). Keyed by
# req_state_idx, so the live set is bounded by max_num_reqs anyway;
# this only bounds the stale tail.
_PURGE_STALE_ITERS = 4096


class NgramAssistIndex:
    """Rightmost-occurrence k-gram -> continuation over a token stream.

    Speculator-agnostic (fed plain token ids from any authoritative
    stream) so the same core can back the MTP fallback arm. Mirrors
    ``NgramIndex`` in the offline simulation: later occurrences
    overwrite earlier ones, occurrences may overlap/self-extend, and
    insertion admits every k-gram whose continuation is a known token.
    """

    __slots__ = ("k", "max_grams", "tokens", "fed", "inserted", "table", "queue")

    def __init__(self, k: int, max_grams: int) -> None:
        self.k = k
        self.max_grams = max_grams
        # Python ints (converted once per token via ndarray.tolist()).
        self.tokens: list[int] = []
        # Number of stream tokens acknowledged (len(self.tokens)).
        self.fed = 0
        # Next gram start position to insert.
        self.inserted = 0
        # key -> (continuation token, start of the rightmost occurrence)
        self.table: dict[tuple[int, ...], tuple[int, int]] = {}
        # (start, key) in ascending start order; overwritten entries are
        # lazily deleted. Deque + dict yields the bounded most-recent
        # window deterministically.
        self.queue: deque[tuple[int, tuple[int, ...]]] = deque()

    def reset(self) -> None:
        self.tokens.clear()
        self.queue.clear()
        self.table.clear()
        self.fed = 0
        self.inserted = 0

    def advance(self, row: np.ndarray, upto: int) -> None:
        """Acknowledge that the stream now has ``upto`` tokens and insert
        every k-gram whose continuation index is < upto (continuation is
        a known token). Matches the sim's ``i + k <= upto`` rule with
        ``upto`` the anchor-inclusive known prefix length."""
        if upto < self.fed:
            # The runner re-admitted the request (e.g. preemption) and
            # rewrote the row; reseed from scratch on the next call.
            self.reset()
            return
        if upto > self.fed:
            self.tokens.extend(row[self.fed : upto].tolist())
            self.fed = upto
        k = self.k
        toks = self.tokens
        table = self.table
        queue = self.queue
        i = self.inserted
        while i + k < upto:
            key = tuple(toks[i : i + k])
            table[key] = (toks[i + k], i)
            queue.append((i, key))
            i += 1
        self.inserted = i
        # Bounded most-recent window: evict grams whose continuation
        # fell out of the last max_grams tokens, plus stale (overwritten)
        # queue entries. Queue order == start order, so pop from the left.
        floor = upto - self.max_grams
        while queue:
            start, key = queue[0]
            hit = table.get(key)
            if hit is None or hit[1] != start:
                queue.popleft()  # overwritten later; its live entry is elsewhere
                continue
            if start + k < floor:
                queue.popleft()
                del table[key]
                continue
            break

    def lookup(self, query: tuple[int, ...]) -> int | None:
        """Continuation of the rightmost occurrence of ``query``, or None."""
        hit = self.table.get(query)
        return hit[0] if hit is not None else None

    def __len__(self) -> int:
        return len(self.table)


class NgramAssist:
    """Owns the per-request indices and applies the override rule.

    Composition follows the adaptive-depth controller pattern: the
    speculator calls ``begin_step()`` before the captured draft graph
    replays (staging the async length readback) and ``apply()`` after
    it returns; everything between is the unmodified graph.
    """

    def __init__(
        self,
        *,
        k: int,
        max_batch: int,
        max_index_tokens: int,
        log_path: str,
        device: torch.device,
        max_num_reqs: int,
        num_speculative_steps: int,
        debug_iters: int = 0,
    ) -> None:
        if k < 1:
            raise ValueError(
                f"VLLM_DSPARK_NGRAM_K must be >= 1, got {k}."
            )
        self.k = k
        self.max_batch = max_batch
        self.max_index_tokens = max_index_tokens
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.num_speculative_steps = num_speculative_steps

        self._log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            # Line-buffered append: one atomic write per JSONL record.
            self._log_file = open(log_path, "a", buffering=1)

        # One-shot diagnostics: first N assisted iterations dump per-request
        # index state + slot-0 query via logger.info (VLLM_DSPARK_NGRAM_DEBUG).
        self._debug_iters = debug_iters

        # Authoritative token stream, handed over by the runner via the
        # speculator's set_token_stream hook. UVA-backed: host-visible
        # pinned memory the GPU writes directly (no extra copies).
        self._token_rows: np.ndarray | None = None
        self._total_len_gpu: torch.Tensor | None = None

        # Pinned readback buffers. _pinned_len is filled at begin_step
        # (enqueued before the draft graph); _pinned_draft at apply()
        # (enqueued after it). Both are complete once apply() syncs.
        self._pinned_len = torch.zeros(max_num_reqs, dtype=torch.int32, pin_memory=True)
        self._pinned_len_np = self._pinned_len.numpy()
        self._pinned_draft = torch.zeros(
            (max_num_reqs, num_speculative_steps),
            dtype=torch.int64,
            pin_memory=True,
        )
        self._pinned_draft_np = self._pinned_draft.numpy()

        # req_state_idx -> [req_id, index, last_seen_iteration]
        self._entries: dict[int, list] = {}
        self._iter = 0
        self._staged_num_reqs = 0
        self._staged_idx: np.ndarray | None = None
        self._warned = False

    # -- runner/speculator plumbing ---------------------------------------

    def set_token_stream(
        self,
        all_token_ids,  # StagedWriteTensor (UVA-backed), [max_num_reqs, max_model_len]
        total_len_gpu: torch.Tensor,  # [max_num_reqs], updated by post_update
    ) -> None:
        """Register the authoritative per-request token stream and its
        per-slot lengths. Called once by the runner; without it the
        assist stays inert (fail-closed)."""
        uva_buf = getattr(all_token_ids, "_uva_buf", None)
        rows = getattr(uva_buf, "np", None) if uva_buf is not None else None
        if rows is None or total_len_gpu is None:
            self._fail("token stream is not UVA-host-visible; assist disabled")
            return
        self._token_rows = rows
        self._total_len_gpu = total_len_gpu
        logger.info(
            "DSpark ngram assist: token stream hooked (rows %s, total_len %s)",
            tuple(rows.shape),
            tuple(total_len_gpu.shape),
        )

    def begin_step(self, input_batch: InputBatch) -> None:
        """Stage the per-request stream-length readback for this step.

        Must run BEFORE the captured draft graph is replayed: the copy
        is enqueued behind this step's post_update (which finalizes the
        accepted-token stream and total_len) and completes by the time
        apply() synchronizes on the graph.
        """
        if self._total_len_gpu is None:
            return
        self._staged_num_reqs = input_batch.num_reqs
        self._staged_idx = input_batch.idx_mapping_np[: input_batch.num_reqs]
        try:
            self._pinned_len.copy_(self._total_len_gpu, non_blocking=True)
        except Exception:
            self._fail("total_len readback staging failed; step unassisted")

    def apply(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        *,
        is_profile: bool = False,
        dummy_run: bool = False,
    ) -> torch.Tensor:
        """Override draft tokens host-side; returns ``draft_tokens`` (the
        same tensor, modified in place only when an override fired)."""
        if (
            self._token_rows is None
            or self._total_len_gpu is None
            or is_profile
            or dummy_run
        ):
            return draft_tokens
        num_reqs = input_batch.num_reqs
        if num_reqs <= 0 or num_reqs > self.max_batch:
            return draft_tokens
        if draft_tokens.ndim != 2 or draft_tokens.shape[1] <= 0:
            return draft_tokens
        if self._staged_idx is None or self._staged_num_reqs != num_reqs:
            return draft_tokens
        rows = self._token_rows
        idx_np = self._staged_idx
        k = self.k
        # Fast fail-closed: nothing can fire while no active request has
        # an index (early decode / fresh prompts), so skip the sync.
        usable = False
        for b in range(num_reqs):
            idx = int(idx_np[b])
            if idx < 0:
                continue
            entry = self._entries.get(idx)
            if entry is not None and len(entry[1]) > 0:
                usable = True
                break
            if int(self._pinned_len_np[idx]) > k:
                usable = True  # first observation will seed a nonempty index
                break
        if not usable:
            return draft_tokens
        try:
            t0 = time.perf_counter()
            n_slots = draft_tokens.shape[1]
            # Pinned D2H + wait for the graph: everything enqueued before
            # now (post_update's stream writes, the length readback, the
            # draft graph) is complete and host-visible afterwards.
            buf = self._pinned_draft[:num_reqs, :n_slots]
            buf.copy_(draft_tokens, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            t_sync = time.perf_counter()
            tok_np = self._pinned_draft_np[:num_reqs, :n_slots]
            lens = self._pinned_len_np
            self._iter += 1
            want_debug = self._debug_iters >= self._iter
            changed_any = False
            log_reqs = [] if self._log_file is not None else None
            fired_total = 0
            changed_total = 0
            slots_total = 0
            for b in range(num_reqs):
                idx = int(idx_np[b])
                if idx < 0:
                    continue
                req_id = input_batch.req_ids[b]
                entry = self._entries.get(idx)
                if entry is None or entry[0] != req_id:
                    index = NgramAssistIndex(k, self.max_index_tokens)
                    self._entries[idx] = [req_id, index, self._iter]
                else:
                    index = entry[1]
                    entry[2] = self._iter
                length = int(lens[idx])
                # Advance the index over newly accepted tokens. A shrink
                # means the request was re-admitted; NgramAssistIndex
                # resets and reseeds from the rewritten row.
                index.advance(rows[idx], length)
                if len(index) == 0 or length < k:
                    continue
                tokens = index.tokens
                # Query suffix at slot j: last k tokens of (stream +
                # tokens already chosen for slots < j) -- the dynamic
                # per-slot re-query of the promoted offline rule.
                query_list = tokens[length - k : length]
                slots_log = [] if log_reqs is not None else None
                chosen: list[int] = []
                for slot in range(n_slots):
                    base = int(tok_np[b, slot])
                    if base < 0:
                        # Declined-to-draft sentinel: never override.
                        break
                    slots_total += 1
                    query = tuple((query_list + chosen)[-k:])
                    cand = index.lookup(query)
                    if cand is not None:
                        fired_total += 1
                        if cand != base:
                            tok_np[b, slot] = cand
                            changed_any = True
                            changed_total += 1
                        if slots_log is not None:
                            slots_log.append(
                                {
                                    "s": slot,
                                    "fired": 1,
                                    "changed": int(cand != base),
                                }
                            )
                    chosen.append(int(tok_np[b, slot]))
                    # Keep the chosen window to the last k tokens.
                    if len(chosen) > k:
                        del chosen[0]
                if log_reqs is not None and slots_log:
                    log_reqs.append(
                        {"req": req_id, "idx": idx, "len": length, "slots": slots_log}
                    )
                if want_debug:
                    self._dump_debug(req_id, idx, length, index, query_list)
            if changed_any:
                draft_tokens.copy_(buf, non_blocking=True)
            if log_reqs is not None:
                # Top-level aggregates so naive per-line sums work;
                # sync_wait_us is the graph-completion wait (in sync
                # scheduling the host would block on the output copy
                # around here anyway), apply_us is pure host work after
                # the sync.
                self._log_file.write(
                    json.dumps(
                        {
                            "t": time.time(),
                            "iter": self._iter,
                            "k": k,
                            "num_reqs": num_reqs,
                            "n_slots": n_slots,
                            "slots_total": slots_total,
                            "fired": fired_total,
                            "changed": changed_total,
                            "sync_wait_us": round((t_sync - t0) * 1e6, 1),
                            "apply_us": round((time.perf_counter() - t_sync) * 1e6, 1),
                            "reqs": log_reqs,
                        }
                    )
                    + "\n"
                )
            if self._iter % 1024 == 0:
                self._purge_stale()
            return draft_tokens
        except Exception:
            self._fail("assist lookup failed; proposal left unassisted")
            return draft_tokens

    def _dump_debug(
        self,
        req_id: str,
        idx: int,
        length: int,
        index: NgramAssistIndex,
        query_list: list[int],
    ) -> None:
        """One-shot per-request diagnostics for the first N assisted
        iterations: index size/cursors, slot-0 query, and sample keys
        (to localize silent misses)."""
        sample_keys = list(index.table.items())[:3]
        logger.info(
            "DSpark ngram assist dbg iter=%d req=%s idx=%d len=%d grams=%d "
            "fed=%d inserted=%d q0=%s sample_keys=%s",
            self._iter,
            req_id,
            idx,
            length,
            len(index),
            index.fed,
            index.inserted,
            query_list,
            [(list(key), val) for key, val in sample_keys],
        )

    # -- internals ---------------------------------------------------------

    def _purge_stale(self) -> None:
        for idx in [i for i, e in self._entries.items() if self._iter - e[2] > _PURGE_STALE_ITERS]:
            del self._entries[idx]

    def _fail(self, message: str) -> None:
        """Fail closed: disable the assist permanently and warn once."""
        if not self._warned:
            logger.warning("DSpark ngram assist: %s", message)
            self._warned = True
        self._token_rows = None
