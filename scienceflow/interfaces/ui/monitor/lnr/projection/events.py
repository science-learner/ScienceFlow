"""Read-only LNR monitor responsibility: events."""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scienceflow.interfaces.ui.monitor.lnr.rendering.base import (
    _MAX_RESOURCE_EVENTS,
    _REAL_EPOCH_MIN_TS,
    _RECENT_JSONL_TAIL_BYTES,
    _STALE_ACTIVE_JOB_DISPLAY_SEC,
)
from scienceflow.interfaces.ui.monitor.lnr.rendering.formatting import _to_float, _to_int

def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out

def _iter_recent_jsonl(path: Path, max_events: int = _MAX_RESOURCE_EVENTS) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > _RECENT_JSONL_TAIL_BYTES:
                start = max(0, size - _RECENT_JSONL_TAIL_BYTES)
                f.seek(start)
                raw_lines = f.read().splitlines()
                if start > 0 and raw_lines:
                    raw_lines = raw_lines[1:]
            else:
                raw_lines = f.read().splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw_line in raw_lines[-max_events:]:
        line = raw_line.decode("utf-8", errors="ignore")
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out

def _iter_recent_jsonl_many(paths: Iterable[Path], max_events: int = _MAX_RESOURCE_EVENTS) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for obj in _iter_recent_jsonl(Path(path), max_events=max_events):
            try:
                key = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError):
                key = repr(obj)
            if key in seen:
                continue
            seen.add(key)
            out.append(obj)
    out.sort(key=lambda obj: _to_float(obj.get("timestamp") or obj.get("created_at")) or 0.0)
    return out[-max_events:]

def _heartbeat_progress_summary(
    payload: dict[str, Any],
    *,
    event_type: str,
    observed_at: float,
) -> dict[str, Any]:
    if event_type != "progress_heartbeat":
        return {}
    signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
    heartbeat = signals.get("heartbeat") if isinstance(signals.get("heartbeat"), dict) else {}
    structured = payload.get("structured_progress")
    if not isinstance(structured, dict):
        structured = {}
    progress = heartbeat.get("progress") if isinstance(heartbeat.get("progress"), dict) else {}
    progress_unit = ""
    if not progress:
        for key in ("progress", "batch", "step", "epoch", "nb", "items", "rows"):
            candidate = signals.get(key)
            if isinstance(candidate, dict) and ("current" in candidate or "total" in candidate):
                progress = candidate
                progress_unit = key
                break

    current = _to_float(structured.get("current"))
    if current is None:
        current = _to_float(progress.get("current"))
    total = _to_float(structured.get("total"))
    if total is None:
        total = _to_float(progress.get("total"))
    unit = str(
        structured.get("unit")
        or progress.get("unit")
        or heartbeat.get("unit")
        or progress_unit
        or ""
    )
    advanced: bool | None = None
    for raw in (structured.get("advanced"), progress.get("advanced"), heartbeat.get("advanced")):
        if isinstance(raw, bool):
            advanced = raw
            break

    metric_name = ""
    metric_value: float | None = None
    metrics = signals.get("metrics") if isinstance(signals.get("metrics"), dict) else {}
    for key, value in metrics.items():
        numeric = _to_float(value)
        if numeric is not None:
            metric_name = str(key)
            metric_value = numeric
            break

    phase = str(payload.get("phase") or payload.get("current_phase") or heartbeat.get("phase") or "")
    summary: dict[str, Any] = {}
    if current is not None:
        summary["progress_current"] = current
    if total is not None:
        summary["progress_total"] = total
    if unit:
        summary["progress_unit"] = unit
    if advanced is not None:
        summary["progress_advanced"] = advanced
    if metric_name:
        summary["metric_name"] = metric_name
    if metric_value is not None:
        summary["metric_value"] = metric_value
    if phase:
        summary["heartbeat_phase"] = phase
    if summary:
        summary["heartbeat_observed_at"] = observed_at
    return summary

def _job_summary(obj: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    event_type = str(obj.get("event_type") or obj.get("event") or "")
    command_id = str(obj.get("command_id") or payload.get("job_id") or "")
    observed_at = _to_float(obj.get("created_at") or obj.get("timestamp")) or 0.0
    outcome = str(payload.get("execution_outcome") or payload.get("outcome") or "").strip().upper()
    finished = event_type in {"resource_job_finished", "resource_kill_executed"} or (
        event_type == "resource_review_outcome" and outcome in {"KILL", "TERMINATED", "STOPPED"}
    )
    signal = payload.get("resource_review_signal")
    if not isinstance(signal, dict):
        signal = {}
    progress = payload.get("progress_snapshot")
    if not isinstance(progress, dict):
        progress = {}
    if not progress and isinstance(payload.get("filtered_signal"), dict):
        progress = payload.get("filtered_signal") or {}
    if not progress and command_id and ("elapsed_sec" in payload or "metric_history_line_count" in payload):
        progress = payload
    resource = payload.get("resource_snapshot")
    if not isinstance(resource, dict):
        resource = {}
    source_hint = payload.get("source_hint")
    if not isinstance(source_hint, dict):
        source_hint = resource.get("source_hint") if isinstance(resource.get("source_hint"), dict) else {}
    if not signal and command_id and progress:
        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        cpu_bucket = str(payload.get("cpu_bucket") or "")
        gpu_bucket = str(payload.get("gpu_bucket") or "")
        process_alive = bool(payload.get("process_alive"))
        monitor_active = event_type == "resource_monitor_heartbeat" and process_alive
        signal = {
            "active_work": monitor_active or not finished,
            "useful_progress": bool(signals or _to_int(progress.get("metric_history_line_count")) or _to_int(payload.get("stdout_lines"))),
            "metric_changed": bool(signals.get("metrics") if isinstance(signals, dict) else False),
            "gpu_bucket": gpu_bucket,
            "cpu_bucket": cpu_bucket,
        }
    liveness = progress.get("process_liveness") or resource.get("process_liveness")
    if not isinstance(liveness, dict):
        liveness = {}
    alive = liveness.get("status") in {"alive", "inconsistent"}
    liveness_known = bool(liveness.get("status")) or "process_alive" in payload
    if "process_alive" in payload:
        alive = bool(payload.get("process_alive"))
    if finished:
        alive = False
    summary = {
        "worker_id": obj.get("worker_id") or payload.get("worker_id") or "?",
        "command_id": command_id,
        "event_type": event_type,
        "observed_at": observed_at,
        "runtime_sec": _to_float(
            progress.get("runtime_sec")
            or progress.get("elapsed_sec")
            or payload.get("runtime_sec")
            or payload.get("elapsed_sec")
        ) or 0.0,
        "active_work": bool(signal.get("active_work")) and not finished,
        "useful_progress": bool(signal.get("useful_progress")),
        "gpu_bucket": str(signal.get("gpu_bucket") or ""),
        "cpu_bucket": str(signal.get("cpu_bucket") or ""),
        "metric_changed": bool(signal.get("metric_changed")),
        "metric_lines": _to_int(progress.get("metric_history_line_count")) or 0,
        "stdout_lines": _to_int(progress.get("stdout_lines") or payload.get("stdout_lines")) or 0,
        "alive": alive,
        "liveness_known": liveness_known,
        "finished": finished,
        "reason": str(signal.get("reason") or payload.get("reason_code") or ""),
        "command_excerpt": str(payload.get("command_excerpt") or resource.get("command_excerpt") or ""),
        "entrypoint": str(payload.get("entrypoint") or resource.get("entrypoint") or ""),
        "resource_class": str(payload.get("resource_class") or resource.get("resource_class") or ""),
        "known_stage": str(
            progress.get("known_stage")
            or progress.get("current_phase")
            or payload.get("known_stage")
            or payload.get("current_phase")
            or ""
        ),
        "progress_signal": str(progress.get("progress_signal") or ""),
        "progress_confidence": str(progress.get("progress_confidence") or ""),
        "output_pattern": str(progress.get("output_pattern") or ""),
        "recoverable_artifact_on_disk": bool(progress.get("recoverable_artifact_on_disk")),
        "stop_cost": str(progress.get("stop_cost") or ""),
        "finish_feasible": progress.get("finish_feasible"),
        "source_hint": source_hint,
    }
    summary.update(_heartbeat_progress_summary(payload, event_type=event_type, observed_at=observed_at))
    return summary

def _select_active_jobs_for_display(jobs: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    """Keep fresh per-worker jobs before older long-running historical records."""
    if not jobs:
        return []

    latest_by_worker: dict[str, dict[str, Any]] = {}
    for job in jobs:
        worker = str(job.get("worker_id") or "?").upper()
        previous = latest_by_worker.get(worker)
        if previous is None or _job_seen_at(job) >= _job_seen_at(previous):
            latest_by_worker[worker] = job

    selected: list[dict[str, Any]] = sorted(
        latest_by_worker.values(),
        key=lambda job: (_job_seen_at(job), float(job.get("runtime_sec") or 0.0)),
        reverse=True,
    )
    selected_ids = {str(job.get("command_id") or id(job)) for job in selected}
    for job in sorted(
        jobs,
        key=lambda item: (_job_seen_at(item), float(item.get("runtime_sec") or 0.0)),
        reverse=True,
    ):
        if len(selected) >= limit:
            break
        job_id = str(job.get("command_id") or id(job))
        if job_id in selected_ids:
            continue
        selected.append(job)
        selected_ids.add(job_id)
    return selected[:limit]

def _job_seen_at(job: dict[str, Any]) -> float:
    return max(
        float(job.get("heartbeat_observed_at") or 0.0),
        float(job.get("observed_at") or 0.0),
    )

def _job_is_active(job: dict[str, Any]) -> bool:
    if job.get("finished"):
        return False
    if job.get("alive"):
        return True
    if job.get("liveness_known"):
        return False
    if str(job.get("event_type") or "") == "progress_heartbeat":
        return bool(job.get("useful_progress")) and float(job.get("runtime_sec") or 0.0) > 0.0
    return float(job.get("runtime_sec") or 0.0) > 0.0

def _job_is_stale_for_display(job: dict[str, Any]) -> bool:
    seen_at = _job_seen_at(job)
    if seen_at < _REAL_EPOCH_MIN_TS:
        return False
    return _dt.datetime.now().timestamp() - seen_at > _STALE_ACTIVE_JOB_DISPLAY_SEC

__all__ = tuple(name for name in globals() if not name.startswith("__"))
