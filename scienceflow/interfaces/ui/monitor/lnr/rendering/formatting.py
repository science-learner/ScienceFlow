"""Read-only LNR monitor responsibility: formatting."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from scienceflow.research.solver.lnr.lifecycle.stage.metrics.score_summary import infer_stage_rows_lower_is_better
from scienceflow.interfaces.ui.monitor.lnr.rendering.time_trace import _time_trace_paths

def _latest_stage_by_worker(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        worker = (row.get("worker_id") or "?").upper()
        latest[worker] = _stage_brief(row)
    return latest

def _stage_brief(row: dict[str, str]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id") or "",
        "worker_id": row.get("worker_id") or "",
        "metric_value": _to_float(row.get("metric_value")),
        "metric_validity": row.get("metric_validity") or "",
        "candidate_ready": row.get("candidate_ready") or "",
        "selection_eligible": row.get("selection_eligible") or "",
        "evaluator_backend": row.get("evaluator_backend") or "",
        "evaluator_status": row.get("evaluator_status") or "",
        "artifact_sha": row.get("artifact_sha") or row.get("submission_sha") or "",
    }

def _summary_lower_is_better(
    rows: list[dict[str, str]],
    best_raw: dict[str, Any],
    best_valid: dict[str, Any],
) -> bool:
    for score in (best_valid, best_raw):
        if isinstance(score.get("lower_is_better"), bool):
            return bool(score["lower_is_better"])
    return infer_stage_rows_lower_is_better(rows)

def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}

def _first_int(row: dict[str, str], keys: tuple[str, ...]) -> int:
    for key in keys:
        val = _to_int(row.get(key))
        if val is not None:
            return val
    return 0

def _first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        val = _to_float(row.get(key))
        if val is not None:
            return val
    return None

def _max_optional(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)

def _to_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None

def _to_float(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None

def _newest_mtime(task_root: Path, rows: list[dict[str, str]]) -> float:
    candidates = [
        task_root / "task_logs" / "lhr_stage_performance.csv",
        task_root / "task_logs" / "lhr_events.jsonl",
        task_root / "task_logs" / "resource" / "resource_events.jsonl",
    ]
    candidates.extend(_time_trace_paths(task_root))
    mtimes = []
    for path in candidates:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes) if mtimes else _dt.datetime.now().timestamp()

def _format_timestamp(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")

__all__ = tuple(name for name in globals() if not name.startswith("__"))
