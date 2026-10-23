# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Parallel-worker metric selection and prompt projection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
    _csv_bool,
    _csv_metric,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    LHR_STAGE_PERFORMANCE_CSV,
    infer_stage_rows_lower_is_better,
    logger,
)


def _row_is_better(
    row: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    lower_is_better: bool,
) -> bool:
    metric = _csv_metric(row.get("metric_value"))
    if metric is None:
        return False
    if previous is None:
        return True
    previous_metric = _csv_metric(previous.get("metric_value"))
    if previous_metric is None:
        return True
    return metric < previous_metric if lower_is_better else metric > previous_metric


def _row_metric_text(row: dict[str, Any]) -> str:
    metric = _csv_metric(row.get("metric_value"))
    return (
        f"{metric:.6f}"
        if metric is not None
        else str(row.get("metric_value") or "unknown")
    )


def _row_node_label(row: dict[str, Any]) -> str:
    stage = str(row.get("stage_id") or "").strip() or "unknown"
    lineage = str(row.get("lineage_id") or "").strip()
    return f"{lineage}:{stage}" if lineage else stage


def _row_elapsed_text(row: dict[str, Any]) -> str:
    try:
        elapsed = float(str(row.get("elapsed_min") or "").strip())
    except (TypeError, ValueError):
        elapsed = -1.0
    if elapsed >= 0:
        return f"{elapsed:.1f}m"
    try:
        seconds = float(str(row.get("solution_run_sec") or "").strip())
    except (TypeError, ValueError):
        seconds = -1.0
    return f"{seconds / 60.0:.1f}m" if seconds >= 0 else "unknown"


def _same_candidate(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> bool:
    if first is None or second is None:
        return False
    first_id = str(first.get("candidate_id") or "").strip()
    second_id = str(second.get("candidate_id") or "").strip()
    if first_id or second_id:
        return first_id == second_id
    return all(
        str(first.get(key) or "") == str(second.get(key) or "")
        for key in ("worker_id", "lineage_id", "stage_id")
    )


def _best_worker_rows(
    owner: Any,
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_best: dict[str, dict[str, Any]] = {}
    valid_best: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return raw_best, valid_best
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        logger.debug("[lnr] could not read worker peer summary csv", exc_info=True)
        return raw_best, valid_best

    task_hint = getattr(owner, "_task_metric_lower_is_better", None)
    lower_is_better = (
        bool(task_hint)
        if task_hint is not None
        else infer_stage_rows_lower_is_better(rows)
    )
    for row in rows:
        worker_id = str(row.get("worker_id") or "").strip() or "W00"
        if _csv_metric(row.get("metric_value")) is None:
            continue
        if _row_is_better(
            row,
            raw_best.get(worker_id),
            lower_is_better=lower_is_better,
        ):
            raw_best[worker_id] = row
        if _csv_bool(row.get("validation_ok"), default=True) is False:
            continue
        if _row_is_better(
            row,
            valid_best.get(worker_id),
            lower_is_better=lower_is_better,
        ):
            valid_best[worker_id] = row
    return raw_best, valid_best


def _worker_line(
    owner: Any,
    label: str,
    raw: dict[str, Any] | None,
    valid: dict[str, Any] | None,
) -> str:
    if raw is None:
        return f"- {label}: no metric-backed stage recorded yet."
    method = owner._compact_peer_method(str(raw.get("brief") or raw.get("why") or ""))
    if _same_candidate(raw, valid):
        return (
            f"- {label}: best_metric={_row_metric_text(raw)} (ok) at "
            f"{_row_node_label(raw)}; elapsed={_row_elapsed_text(raw)}; method={method}"
        )
    issue = owner._compact_peer_method(
        str(raw.get("validation_issue") or "validation_ok=false"),
        max_chars=100,
    )
    line = (
        f"- {label}: best_metric={_row_metric_text(raw)} at {_row_node_label(raw)}; "
        f"elapsed={_row_elapsed_text(raw)}; status=suspicious; issue={issue}; "
        f"method={method}"
    )
    if valid is None:
        return line + "; ok_best_metric=none"
    valid_method = owner._compact_peer_method(
        str(valid.get("brief") or valid.get("why") or "")
    )
    return (
        line
        + f"; ok_best_metric={_row_metric_text(valid)} at {_row_node_label(valid)}; "
        + f"ok_elapsed={_row_elapsed_text(valid)}; ok_method={valid_method}"
    )


def _parallel_worker_snapshot_for_prompt(self: Any) -> str:
    if not bool(getattr(self.lhr, "worker_peer_summary_enabled", True)):
        return ""
    worker_count = max(1, int(getattr(self, "worker_count", 1) or 1))
    if worker_count <= 1:
        return ""
    max_chars = max(
        600,
        int(getattr(self.lhr, "worker_peer_summary_max_chars", 2200) or 2200),
    )
    path = (
        Path(getattr(self, "global_log_dir", getattr(self, "log_dir", Path("."))))
        / LHR_STAGE_PERFORMANCE_CSV
    )
    raw_best, valid_best = _best_worker_rows(self, path)
    current_worker = self._worker_uid_prefix()
    lines = [
        "Parallel worker snapshot:",
        f"- There are {worker_count} workers exploring this task independently; "
        "use peer progress as high-level search context.",
    ]
    for index in range(worker_count):
        worker_id = f"W{index:02d}"
        label = (
            f"{worker_id} (this worker)"
            if worker_id == current_worker
            else worker_id
        )
        lines.append(
            _worker_line(
                self,
                label,
                raw_best.get(worker_id),
                valid_best.get(worker_id),
            )
        )
    summary = "\n".join(lines).strip()
    if len(summary) > max_chars:
        return summary[: max_chars - 38].rstrip() + "\n... [worker snapshot truncated]"
    return summary


__all__ = ["_parallel_worker_snapshot_for_prompt"]
