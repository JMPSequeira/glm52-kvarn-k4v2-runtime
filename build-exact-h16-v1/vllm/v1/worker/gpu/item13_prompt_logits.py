# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_CAPTURE_PLAN_ENV = "GLM_ITEM13_PROMPT_LOGITS_PLAN"
_CAPTURE_PLAN_SHA_ENV = "GLM_ITEM13_PROMPT_LOGITS_PLAN_SHA256"
_CAPTURE_DIR_ENV = "GLM_ITEM13_PROMPT_LOGITS_DIR"
_RANDOMIZED_REQUEST_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Item13PromptLogitsCapture:
    """Rank-0, serial, fail-closed raw prompt-logit capture for item 13."""

    @classmethod
    def from_env(cls) -> Item13PromptLogitsCapture | None:
        plan_value = os.getenv(_CAPTURE_PLAN_ENV)
        plan_sha256 = os.getenv(_CAPTURE_PLAN_SHA_ENV)
        output_value = os.getenv(_CAPTURE_DIR_ENV)
        configured = (plan_value, plan_sha256, output_value)
        if not any(configured):
            return None
        if not all(configured):
            raise RuntimeError(
                f"{_CAPTURE_PLAN_ENV}, {_CAPTURE_PLAN_SHA_ENV}, and {_CAPTURE_DIR_ENV} "
                "must be configured together"
            )

        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return None
        return cls(Path(plan_value), str(plan_sha256), Path(output_value))

    def __init__(self, plan_path: Path, expected_plan_sha256: str, output_dir: Path):
        if len(expected_plan_sha256) != 64:
            raise ValueError("item-13 capture plan SHA-256 must contain 64 hex digits")
        plan_bytes = plan_path.read_bytes()
        actual_plan_sha256 = _sha256_bytes(plan_bytes)
        if actual_plan_sha256 != expected_plan_sha256:
            raise ValueError(
                "item-13 capture plan hash mismatch: "
                f"expected {expected_plan_sha256}, got {actual_plan_sha256}"
            )
        plan = json.loads(plan_bytes)
        if plan.get("schema_version") != 1:
            raise ValueError("unsupported item-13 capture plan schema")
        if plan.get("vocab_size") != 154880:
            raise ValueError("item-13 capture plan vocabulary mismatch")
        rows = plan.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("item-13 capture plan has no rows")
        row_ids: set[str] = set()
        for index, row in enumerate(rows):
            if row.get("index") != index:
                raise ValueError("item-13 capture plan row indices are not contiguous")
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or len(row_id) != 64 or row_id in row_ids:
                raise ValueError("invalid or duplicate item-13 row ID")
            row_ids.add(row_id)
            if not isinstance(row.get("token_count"), int) or row["token_count"] < 2:
                raise ValueError("item-13 rows must contain at least two tokens")
            token_sha256 = row.get("token_ids_sha256")
            if not isinstance(token_sha256, str) or len(token_sha256) != 64:
                raise ValueError("item-13 row token hash is invalid")

        planned_request_ids = tuple(
            f"cmpl-item13-{index:04d}-{row['row_id']}-0"
            for index, row in enumerate(rows)
        )
        if len(set(planned_request_ids)) != len(rows):
            raise ValueError("item-13 planned request IDs are not unique")

        if not output_dir.is_dir():
            raise ValueError("item-13 capture output directory does not exist")
        if any(output_dir.iterdir()):
            raise ValueError("item-13 capture output directory must be empty")

        self.plan_path = plan_path.resolve()
        self.plan_sha256 = actual_plan_sha256
        self.output_dir = output_dir.resolve()
        self.rows: list[dict[str, Any]] = rows
        self.vocab_size = int(plan["vocab_size"])
        self._planned_request_ids = planned_request_ids
        self._planned_request_id_set = frozenset(planned_request_ids)
        self._next_index = 0
        self._active_req_id: str | None = None
        self._active_row: dict[str, Any] | None = None
        self._logits_handle: Any | None = None
        self._labels_handle: Any | None = None
        self._logits_digest: hashlib._Hash | None = None
        self._labels_digest: hashlib._Hash | None = None
        self._captured_rows = 0
        self._logits_partial: Path | None = None
        self._labels_partial: Path | None = None
        self._write_progress()

    def _write_progress(self) -> None:
        _atomic_json(
            self.output_dir / "progress.json",
            {
                "active_request_id": self._active_req_id,
                "completed_rows": self._next_index,
                "plan_path": str(self.plan_path),
                "plan_sha256": self.plan_sha256,
                "schema_version": 1,
                "total_rows": len(self.rows),
            },
        )

    def is_planned_request_id(self, req_id: str) -> bool:
        planned_req_id = _RANDOMIZED_REQUEST_SUFFIX_RE.sub("", req_id)
        return planned_req_id in self._planned_request_id_set

    def next_expected_request_id(self) -> str | None:
        if self._next_index >= len(self._planned_request_ids):
            return None
        return self._planned_request_ids[self._next_index]

    def begin(self, req_id: str) -> None:
        expected_req_id = self.next_expected_request_id()
        if expected_req_id is None:
            raise RuntimeError("item-13 capture plan is already complete")
        planned_req_id = _RANDOMIZED_REQUEST_SUFFIX_RE.sub("", req_id)
        if planned_req_id != expected_req_id:
            raise RuntimeError(
                "item-13 capture request is out of order: "
                f"expected {expected_req_id}, got {req_id}"
            )
        if self._active_req_id is not None:
            if req_id != self._active_req_id:
                raise RuntimeError("item-13 capture received concurrent prompt requests")
            return

        row = self.rows[self._next_index]
        stem = f"{self._next_index:04d}-{row['row_id']}"
        logits_partial = self.output_dir / f"{stem}.bf16.partial"
        labels_partial = self.output_dir / f"{stem}.labels.i32.partial"
        self._logits_handle = logits_partial.open("xb")
        try:
            self._labels_handle = labels_partial.open("xb")
        except BaseException:
            self._logits_handle.close()
            raise
        self._active_req_id = req_id
        self._active_row = row
        self._logits_digest = hashlib.sha256()
        self._labels_digest = hashlib.sha256()
        self._captured_rows = 0
        self._logits_partial = logits_partial
        self._labels_partial = labels_partial
        self._write_progress()

    def append(
        self, req_id: str, logits: torch.Tensor, label_token_ids: torch.Tensor
    ) -> None:
        if req_id != self._active_req_id or self._active_row is None:
            raise RuntimeError("item-13 capture append has no matching active request")
        if logits.ndim != 2 or logits.shape[1] != self.vocab_size:
            raise ValueError("item-13 captured logits have the wrong shape")
        if logits.dtype != torch.bfloat16:
            raise ValueError(f"item-13 captured logits must be BF16, got {logits.dtype}")
        if label_token_ids.ndim != 1 or label_token_ids.shape[0] != logits.shape[0]:
            raise ValueError("item-13 captured label/logit rows do not align")
        if self._captured_rows + logits.shape[0] > self._active_row["token_count"]:
            raise ValueError("item-13 captured more rows than the sealed plan allows")
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("item-13 captured nonfinite logits")

        logits_cpu = logits.detach().contiguous().view(torch.uint16).cpu().numpy()
        labels_cpu = (
            label_token_ids.detach().to(device="cpu", dtype=torch.int32).contiguous().numpy()
        )
        logits_payload = logits_cpu.tobytes(order="C")
        labels_payload = labels_cpu.astype("<i4", copy=False).tobytes(order="C")
        assert self._logits_handle is not None and self._labels_handle is not None
        assert self._logits_digest is not None and self._labels_digest is not None
        self._logits_handle.write(logits_payload)
        self._labels_handle.write(labels_payload)
        self._logits_digest.update(logits_payload)
        self._labels_digest.update(labels_payload)
        self._captured_rows += int(logits.shape[0])

    def finish(self, req_id: str, *, final_prompt_chunk: bool) -> None:
        if req_id != self._active_req_id or self._active_row is None:
            raise RuntimeError("item-13 capture finish has no matching active request")
        if not final_prompt_chunk:
            return
        expected_rows = int(self._active_row["token_count"])
        if self._captured_rows != expected_rows:
            raise ValueError(
                f"item-13 captured {self._captured_rows} rows; expected {expected_rows}"
            )

        assert self._logits_handle is not None and self._labels_handle is not None
        assert self._logits_digest is not None and self._labels_digest is not None
        assert self._logits_partial is not None and self._labels_partial is not None
        for handle in (self._logits_handle, self._labels_handle):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

        logits_path = self._logits_partial.with_suffix("")
        labels_path = self._labels_partial.with_suffix("")
        os.replace(self._logits_partial, logits_path)
        os.replace(self._labels_partial, labels_path)
        logits_bytes = expected_rows * self.vocab_size * 2
        labels_bytes = expected_rows * 4
        if logits_path.stat().st_size != logits_bytes:
            raise ValueError("item-13 logit file size mismatch")
        if labels_path.stat().st_size != labels_bytes:
            raise ValueError("item-13 label file size mismatch")

        _atomic_json(
            self.output_dir / f"{self._next_index:04d}-{self._active_row['row_id']}.json",
            {
                "domain": self._active_row["domain"],
                "index": self._next_index,
                "label_token_ids": {
                    "bytes": labels_bytes,
                    "dtype": "int32",
                    "path": labels_path.name,
                    "sha256": self._labels_digest.hexdigest(),
                },
                "logits": {
                    "bytes": logits_bytes,
                    "dtype": "bfloat16",
                    "path": logits_path.name,
                    "sha256": self._logits_digest.hexdigest(),
                    "shape": [expected_rows, self.vocab_size],
                },
                "plan_sha256": self.plan_sha256,
                "request_id": self._planned_request_ids[self._next_index],
                "row_id": self._active_row["row_id"],
                "schema_version": 1,
                "token_count": expected_rows,
                "token_ids_sha256": self._active_row["token_ids_sha256"],
            },
        )
        self._next_index += 1
        self._active_req_id = None
        self._active_row = None
        self._logits_handle = None
        self._labels_handle = None
        self._logits_digest = None
        self._labels_digest = None
        self._captured_rows = 0
        self._logits_partial = None
        self._labels_partial = None
        self._write_progress()
