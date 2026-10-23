"""Read-only LNR monitor responsibility: projection."""

from __future__ import annotations

from typing import Any

from scienceflow.interfaces.ui.monitor.lnr.rendering.base import (
    _HEARTBEAT_CONTEXT_KEYS,
    _STALE_HEARTBEAT_CONTEXT_GAP_SEC,
)
from scienceflow.interfaces.ui.monitor.lnr.projection.events import _job_seen_at
from scienceflow.interfaces.ui.monitor.lnr.rendering.formatting import _to_float, _to_int

def _can_inherit_job_context(previous: dict[str, Any], current: dict[str, Any], key: str) -> bool:
    if key not in _HEARTBEAT_CONTEXT_KEYS:
        return True
    previous_seen = _job_seen_at(previous)
    current_seen = _job_seen_at(current)
    if previous_seen <= 0.0 or current_seen <= 0.0:
        return True
    return current_seen - previous_seen <= _STALE_HEARTBEAT_CONTEXT_GAP_SEC

def _merge_job_summary(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return current
    context_keys = {
        "command_excerpt",
        "entrypoint",
        "resource_class",
        "known_stage",
        "progress_signal",
        "progress_confidence",
        "output_pattern",
        "stop_cost",
        "source_hint",
        "heartbeat_observed_at",
        "heartbeat_phase",
        "progress_current",
        "progress_total",
        "progress_unit",
        "progress_advanced",
        "metric_name",
        "metric_value",
    }
    context_only = (
        not current.get("finished")
        and not current.get("alive")
        and float(current.get("runtime_sec") or 0.0) <= 0.0
        and current.get("event_type") == "resource_source_hint_detected"
    )
    if context_only:
        merged = dict(previous)
        for key in context_keys:
            value = current.get(key)
            if value not in (None, "", {}, []):
                merged[key] = value
        # A resume run can reuse command ids.  Context-only events are useful
        # labels, but they must not make old heartbeat/progress evidence look
        # current for a later command with the same id.
        if not any(previous.get(key) not in (None, "", {}, []) for key in _HEARTBEAT_CONTEXT_KEYS):
            merged["observed_at"] = max(
                float(previous.get("observed_at") or 0.0),
                float(current.get("observed_at") or 0.0),
            )
        return merged
    merged = dict(current)
    for key in context_keys:
        if merged.get(key) in (None, "", {}, []) and _can_inherit_job_context(previous, current, key):
            value = previous.get(key)
            if value not in (None, "", {}, []):
                merged[key] = value
    return merged

def _wait_summary(obj: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_id": obj.get("worker_id") or payload.get("worker_id") or "",
        "command_id": obj.get("command_id") or payload.get("job_id") or "",
        "event_type": obj.get("event_type") or obj.get("event") or "",
        "observed_at": _to_float(
            obj.get("created_at") or obj.get("timestamp") or payload.get("created_at")
        ) or 0.0,
        "reason": payload.get("reason_code") or payload.get("reason") or "",
        "resource_class": payload.get("resource_class") or "",
        "queue_position": _to_int(payload.get("queue_position")),
        "queue_len": _to_int(payload.get("queue_len")),
    }

def _worker_states(active_jobs: list[dict[str, Any]], latest_waits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    jobs_by_worker: dict[str, dict[str, Any]] = {}
    for job in active_jobs:
        worker = str(job.get("worker_id") or "?").upper()
        if not worker or worker == "?":
            continue
        previous = jobs_by_worker.get(worker)
        if previous is None or _worker_status_job_key(job) > _worker_status_job_key(previous):
            jobs_by_worker[worker] = job

    by_worker = {
        worker: _worker_state_from_job(job)
        for worker, job in jobs_by_worker.items()
    }
    for worker, wait in latest_waits.items():
        wait_state = _worker_state_from_wait(wait)
        previous = by_worker.get(worker)
        if previous is None or _observed_at(wait_state) >= _observed_at(previous):
            by_worker[worker] = wait_state
    return [by_worker[key] for key in sorted(by_worker)]

def _worker_status_job_key(job: dict[str, Any]) -> tuple[int, int, int, float, float]:
    """Choose stable per-worker status without letting short probes hide long work."""
    runtime = float(job.get("runtime_sec") or 0.0)
    long_bucket = 2 if runtime >= 1800.0 else (1 if runtime >= 900.0 else 0)
    useful_bucket = 1 if _job_has_useful_status_signal(job) else 0
    alive_bucket = 1 if bool(job.get("alive")) else 0
    return (alive_bucket, long_bucket, useful_bucket, _job_seen_at(job), runtime)

def _job_has_useful_status_signal(job: dict[str, Any]) -> bool:
    return bool(
        job.get("useful_progress")
        or job.get("metric_changed")
        or int(job.get("stdout_lines") or 0) > 0
        or int(job.get("metric_lines") or 0) > 0
        or str(job.get("known_stage") or "").strip()
        or str(job.get("progress_signal") or "").strip()
        or str(job.get("heartbeat_phase") or "").strip()
    )

def _worker_state_from_wait(wait: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_id": str(wait.get("worker_id") or "?").upper(),
        "command_id": wait.get("command_id") or "",
        "kind": "wait",
        "label": "WAIT",
        "runtime_sec": 0.0,
        "observed_at": float(wait.get("observed_at") or 0.0),
        "reason": wait.get("reason") or "",
        "resource_class": wait.get("resource_class") or "",
        "queue_position": wait.get("queue_position"),
        "queue_len": wait.get("queue_len"),
    }

def _worker_state_from_job(job: dict[str, Any]) -> dict[str, Any]:
    kind, label = _worker_job_kind(job)
    return {
        "worker_id": str(job.get("worker_id") or "?").upper(),
        "command_id": job.get("command_id") or "",
        "kind": kind,
        "label": label,
        "runtime_sec": float(job.get("runtime_sec") or 0.0),
        "observed_at": float(job.get("observed_at") or 0.0),
        "gpu_bucket": job.get("gpu_bucket") or "",
        "cpu_bucket": job.get("cpu_bucket") or "",
        "useful_progress": bool(job.get("useful_progress")),
        "metric_changed": bool(job.get("metric_changed")),
        "stdout_lines": int(job.get("stdout_lines") or 0),
        "metric_lines": int(job.get("metric_lines") or 0),
        "resource_class": job.get("resource_class") or "",
        "known_stage": job.get("known_stage") or "",
        "progress_signal": job.get("progress_signal") or "",
        "finish_feasible": job.get("finish_feasible"),
    }

def _worker_job_kind(job: dict[str, Any]) -> tuple[str, str]:
    text = " ".join(
        str(part or "")
        for part in (
            job.get("command_excerpt"),
            job.get("entrypoint"),
            job.get("resource_class"),
            job.get("known_stage"),
            job.get("reason"),
        )
    ).lower()
    source_hint = job.get("source_hint") if isinstance(job.get("source_hint"), dict) else {}
    runtime = float(job.get("runtime_sec") or 0.0)
    inference_tokens = ("predict", "infer", "inference", "submission", "submit", "ensemble", "test.py")
    if any(token in text for token in inference_tokens):
        return "inference", "INF"
    if any(token in text for token in ("valid", "eval", "score", "oof")):
        return "validation", "VAL"
    train_hint = (
        any(token in text for token in ("train", "fit", "trainer", "epoch", "heavy_gpu_train", "heavy_cpu_train"))
        or bool(source_hint.get("command_train_evidence"))
        or bool(source_hint.get("source_train_evidence"))
    )
    if train_hint:
        return ("long_train", "LTRN") if runtime >= 900.0 else ("train", "TRN")
    if runtime >= 1800.0:
        return "long_run", "LRUN"
    return "run", "RUN"

def _observed_at(value: dict[str, Any]) -> float:
    return float(value.get("observed_at") or 0.0)

def _recent_event(obj: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    adv = payload.get("advisory") or payload.get("parsed_response")
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    gate = decision.get("gate") if isinstance(decision.get("gate"), dict) else {}
    preference = payload.get("advisory_preference")
    advisory_status = payload.get("advisory_status")
    advisory_confidence = payload.get("advisory_confidence")
    if isinstance(adv, dict):
        preference = preference or adv.get("preference")
        advisory_status = advisory_status or adv.get("advisory_status") or adv.get("status")
        advisory_confidence = advisory_confidence or adv.get("confidence")
    boundary = payload.get("resource_review_boundary")
    boundary_kind = boundary.get("kind") if isinstance(boundary, dict) else boundary
    gate_allowed = gate.get("allowed")
    strict_gate_result = payload.get("strict_gate_result")
    if not strict_gate_result and gate_allowed is not None:
        strict_gate_result = "allow" if bool(gate_allowed) else "blocked"
    strict_gate_reason = payload.get("strict_gate_reason") or gate.get("blocked_reason") or gate.get("reason") or ""
    outcome = payload.get("execution_outcome") or payload.get("outcome") or decision.get("execution_outcome") or decision.get("canonical_outcome") or ""
    return {
        "event_type": obj.get("event_type") or obj.get("event") or "",
        "worker_id": obj.get("worker_id") or payload.get("worker_id") or "",
        "command_id": obj.get("command_id") or payload.get("job_id") or "",
        "outcome": outcome,
        "reason": payload.get("reason_code") or payload.get("reason") or decision.get("reason_code") or decision.get("reason") or "",
        "boundary": boundary_kind or "",
        "preference": preference or "",
        "advisory_status": advisory_status or "",
        "advisory_confidence": advisory_confidence or "",
        "strict_gate_result": strict_gate_result or "",
        "strict_gate_reason": strict_gate_reason or "",
        "decision_applied": payload.get("decision_applied", ""),
    }

__all__ = tuple(name for name in globals() if not name.startswith("__"))
