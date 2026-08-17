# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP table extension: n-gram table tokens beyond the MTP head depth.

Extends the MTP draft block past the head's ``num_spec_tokens`` steps with
table tokens drawn from a per-request online n-gram index (the
``NgramAssistIndex`` pattern from the DSpark n-gram assist, promoted rule:
chain-mode keying, per-slot re-query, top-1 continuation, fan-out 1):

  * The engine is configured with the FULL width ``m = head + extension``
    as ``num_speculative_tokens`` (static width: scheduler padding, verify
    rows, and CUDA graph capture sizes all key off ``m``; set
    ``CUDAGRAPH_CAPTURE_SIZES`` to ``m + 1, 1``).
  * The MTP head still runs only its ``head`` steps; slots
    ``head..m-1`` are ordinary deeper chain rows whose tokens come from
    the table instead of the head. Standard causal mask and standard
    rejection apply — the drafter merely provides a different proposal
    source for those rows.
  * Keying (chain-mode k=1 by default): the query for extension slot j is
    the last ``k`` tokens of (authoritative stream + tokens already chosen
    for slots < j). The key is therefore the chain's OWN last token;
    context-only keying is intentionally NOT implemented (measured
    worthless offline).
  * On a table miss (or any precondition failure) the remaining extension
    slots are filled with the -1 declined-to-draft sentinel: greedy verify
    rejects ``-1`` unconditionally, so the request behaves exactly as the
    unextended arm. The chain stops at the first miss.

Exactness: greedy verify accepts a draft token iff it equals the target's
own argmax, so changing/extending the proposal source cannot change the
served token stream — only acceptance length moves. Same contract as the
adaptive-depth controller and the DSpark n-gram assist.

Application point: HOST-SIDE, after the draft decode graphs have replayed
and before ``propose()`` returns the block (the NgramAssist composition
pattern: ``begin_step()`` before the graphs stage the length readback,
``extend()`` after them). The captured draft graphs are untouched.

Deterministic: pure dict lookups, rightmost occurrence wins, no sampling,
no timing-dependent state. Fail-closed: default-off; unset token stream,
probabilistic draft sampling (extension slots have no draft logits),
dummy/profile runs, or any internal error all leave the block in
declined (-1) extension slots, i.e. unextended behavior.

Overhead: one pinned D2H of ``total_len`` (staged before the draft
graphs), one pinned D2H of the ``[num_reqs, head]`` chain prefix and one
stream synchronize per step (the graphs must finish before host lookups),
plus a pinned H2D write-back when any slot fired.
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

# Declined-to-draft sentinel: greedy verify rejects it unconditionally
# (rejection sampler compares against a non-negative target argmax), so a
# -1 extension slot is exactly "no proposal" — unextended behavior.
DECLINED = -1

# Purge table state for request slots not observed for this many extended
# iterations (finished or preempted-away requests).
_PURGE_STALE_ITERS = 4096


class NgramTableIndex:
    """Rightmost-occurrence k-gram -> continuation over a token stream.

    Same core as the DSpark n-gram assist's ``NgramAssistIndex``: later
    occurrences overwrite earlier ones, occurrences may overlap or
    self-extend, and a bounded most-recent window is retained
    deterministically.
    """

    __slots__ = ("k", "max_grams", "tokens", "fed", "inserted", "table", "queue")

    def __init__(self, k: int, max_grams: int) -> None:
        if k < 1:
            raise ValueError(f"table k must be >= 1, got {k}.")
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
        every k-gram whose continuation index is < ``upto``."""
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
        # Bounded most-recent window (see NgramAssistIndex).
        floor = upto - self.max_grams
        while queue:
            start, key = queue[0]
            hit = table.get(key)
            if hit is None or hit[1] != start:
                queue.popleft()
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


class MTPTableExtension:
    """Owns the per-request tables and fills the extension slots.

    Composition follows the adaptive-depth controller / n-gram assist
    pattern: the MTP speculator calls ``begin_step()`` before the draft
    decode graphs replay and ``extend()`` after they return; everything
    between is the unmodified graph.
    """

    def __init__(
        self,
        *,
        slots: int,
        k: int,
        log_path: str,
        device: torch.device,
        max_num_reqs: int,
        num_speculative_steps: int,
        max_index_tokens: int = 65536,
    ) -> None:
        if slots < 1:
            raise ValueError(f"table extension slots must be >= 1, got {slots}.")
        if slots >= num_speculative_steps:
            raise ValueError(
                "table extension requires at least one MTP head step: "
                f"slots={slots} >= num_speculative_tokens="
                f"{num_speculative_steps}."
            )
        self.slots = slots
        self.k = k
        self.max_index_tokens = max_index_tokens
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.num_speculative_steps = num_speculative_steps
        self.head_steps = num_speculative_steps - slots

        self._log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self._log_file = open(log_path, "a", buffering=1)

        # Authoritative token stream, handed over by the runner
        # (UVA-backed all_token_ids + GPU total_len). Without it the
        # extension stays inert (fail-closed).
        self._token_rows: np.ndarray | None = None
        self._total_len_gpu: torch.Tensor | None = None

        # Pinned readback buffers: _pinned_len staged at begin_step (before
        # the draft graphs), _pinned_chain copied at extend() (after them).
        self._pinned_len = torch.zeros(max_num_reqs, dtype=torch.int32, pin_memory=True)
        self._pinned_len_np = self._pinned_len.numpy()
        self._pinned_chain = torch.zeros(
            (max_num_reqs, self.head_steps), dtype=torch.int64, pin_memory=True
        )
        self._pinned_chain_np = self._pinned_chain.numpy()
        self._pinned_ext = torch.zeros(
            (max_num_reqs, slots), dtype=torch.int64, pin_memory=True
        )

        # req_state_idx -> [req_id, index, last_seen_iteration]
        self._entries: dict[int, list] = {}
        self._iter = 0
        self._staged_idx: np.ndarray | None = None
        self._staged_num_reqs = 0
        self._warned = False

    # -- runner plumbing ---------------------------------------------------

    def set_token_stream(self, all_token_ids, total_len_gpu) -> None:
        """Register the authoritative per-request token stream and its
        per-slot lengths. Called once by the runner; without it the
        extension stays inert (fail-closed)."""
        uva_buf = getattr(all_token_ids, "_uva_buf", None)
        rows = getattr(uva_buf, "np", None) if uva_buf is not None else None
        if rows is None or total_len_gpu is None:
            self._fail("token stream is not UVA-host-visible; extension disabled")
            return
        self._token_rows = rows
        self._total_len_gpu = total_len_gpu
        logger.info(
            "MTP table extension: token stream hooked (rows %s, head_steps %d, "
            "slots %d, k %d)",
            tuple(rows.shape),
            self.head_steps,
            self.slots,
            self.k,
        )

    def begin_step(self, input_batch: "InputBatch") -> None:
        """Stage the per-request stream-length readback. Must run BEFORE
        the draft decode graphs: the copy is enqueued behind this step's
        post_update and completes by the time extend() synchronizes."""
        if self._total_len_gpu is None:
            return
        self._staged_num_reqs = input_batch.num_reqs
        self._staged_idx = input_batch.idx_mapping_np[: input_batch.num_reqs]
        try:
            self._pinned_len.copy_(self._total_len_gpu, non_blocking=True)
        except Exception:
            self._fail("total_len readback staging failed; step unextended")

    # -- extension ---------------------------------------------------------

    def extend(
        self,
        input_batch: "InputBatch",
        draft_tokens: torch.Tensor,
        *,
        dummy_run: bool = False,
        is_profile: bool = False,
    ) -> torch.Tensor:
        """Fill slots ``head_steps..num_speculative_steps-1`` of the draft
        block with table tokens; declines (-1) on any miss. Returns the
        full-width ``draft_tokens`` (modified in place)."""
        width = self.num_speculative_steps
        head = self.head_steps
        ext = draft_tokens[: input_batch.num_reqs, head:width]
        if (
            self._token_rows is None
            or self._total_len_gpu is None
            or is_profile
            or dummy_run
        ):
            ext.fill_(DECLINED)
            return draft_tokens
        num_reqs = input_batch.num_reqs
        if num_reqs <= 0:
            return draft_tokens
        if self._staged_idx is None or self._staged_num_reqs != num_reqs:
            ext.fill_(DECLINED)
            return draft_tokens
        rows = self._token_rows
        idx_np = self._staged_idx
        k = self.k
        # Fast fail-closed: nothing can fire while no active request has a
        # table (early decode / fresh prompts), so skip the sync.
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
                usable = True  # first observation will seed a nonempty table
                break
        if not usable:
            ext.fill_(DECLINED)
            return draft_tokens
        try:
            t0 = time.perf_counter()
            # Pinned D2H of the chain prefix + wait for the graphs:
            # everything enqueued before now (post_update's stream writes,
            # the length readback, the draft graphs) is host-visible after.
            chain_buf = self._pinned_chain[:num_reqs, :head]
            chain_buf.copy_(draft_tokens[:num_reqs, :head], non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            t_sync = time.perf_counter()
            chain_np = self._pinned_chain_np[:num_reqs, :head]
            ext_np = self._pinned_ext[:num_reqs].numpy()
            lens = self._pinned_len_np
            self._iter += 1
            fired_total = 0
            slots_total = 0
            log_reqs = [] if self._log_file is not None else None
            for b in range(num_reqs):
                idx = int(idx_np[b])
                ext_np[b, :] = DECLINED
                if idx < 0:
                    continue
                req_id = input_batch.req_ids[b]
                entry = self._entries.get(idx)
                if entry is None or entry[0] != req_id:
                    table = NgramTableIndex(k, self.max_index_tokens)
                    self._entries[idx] = [req_id, table, self._iter]
                else:
                    table = entry[1]
                    entry[2] = self._iter
                length = int(lens[idx])
                # Advance the table over newly accepted tokens. A shrink
                # means the request was re-admitted; the table resets and
                # reseeds from the rewritten row.
                table.advance(rows[idx], length)
                if len(table) == 0 or length < k:
                    continue
                tokens = table.tokens
                # Chain context: last k tokens of the authoritative stream
                # followed by the head's own drafted chain. The query for
                # extension slot j is the last k tokens of that context
                # plus any tokens already chosen for slots head..j-1 (the
                # dynamic per-slot re-query; chain's OWN last token is the
                # key under k=1).
                chosen = list(tokens[length - k : length]) + [
                    int(t) for t in chain_np[b]
                ]
                chosen = chosen[-k:]
                slots_log = [] if log_reqs is not None else None
                for s in range(self.slots):
                    slots_total += 1
                    cand = table.lookup(tuple(chosen))
                    if cand is None:
                        # Miss: decline this slot and stop the chain here.
                        if slots_log is not None:
                            slots_log.append({"s": s, "fired": 0})
                        break
                    fired_total += 1
                    ext_np[b, s] = cand
                    chosen.append(int(cand))
                    if len(chosen) > k:
                        del chosen[0]
                    if slots_log is not None:
                        slots_log.append({"s": s, "fired": 1, "tok": int(cand)})
                if log_reqs is not None and slots_log:
                    log_reqs.append(
                        {"req": req_id, "idx": idx, "len": length, "slots": slots_log}
                    )
            draft_tokens[:num_reqs, head:width].copy_(
                torch.from_numpy(ext_np), non_blocking=True
            )
            if log_reqs is not None:
                self._log_file.write(
                    json.dumps(
                        {
                            "t": time.time(),
                            "iter": self._iter,
                            "k": k,
                            "num_reqs": num_reqs,
                            "head": head,
                            "width": width,
                            "slots_total": slots_total,
                            "fired": fired_total,
                            "sync_wait_us": round((t_sync - t0) * 1e6, 1),
                            "apply_us": round(
                                (time.perf_counter() - t_sync) * 1e6, 1
                            ),
                            "reqs": log_reqs,
                        }
                    )
                    + "\n"
                )
            if self._iter % 1024 == 0:
                self._purge_stale()
            return draft_tokens
        except Exception:
            self._fail("table lookup failed; extension slots declined")
            try:
                ext.fill_(DECLINED)
            except Exception:
                pass
            return draft_tokens

    # -- internals ---------------------------------------------------------

    def _purge_stale(self) -> None:
        for idx in [
            i
            for i, e in self._entries.items()
            if self._iter - e[2] > _PURGE_STALE_ITERS
        ]:
            del self._entries[idx]

    def _fail(self, message: str) -> None:
        """Fail closed: disable the extension permanently and warn once."""
        if not self._warned:
            logger.warning("MTP table extension: %s", message)
            self._warned = True
        self._token_rows = None
