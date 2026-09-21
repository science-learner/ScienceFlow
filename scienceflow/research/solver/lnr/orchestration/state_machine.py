# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from scienceflow.runtime.events import legacy_event_from_runtime, runtime_event_from_legacy
from scienceflow.runtime.core.kernel.journal import JsonEventJournal

from scienceflow.research.solver.lnr.lifecycle.stage.metrics.score_summary import (
    build_score_summary,
    read_stage_performance_records,
    score_record_from_stage_payload,
)
from scienceflow.research.solver.lnr.orchestration.state_projection import (
    _append_gpu_util_rows,
    _apply_resource_event_to_views,
    _atomic_write_json,
    _atomic_write_jsonl,
    _event_dedupe_key,
    _float_or_zero,
    _gpu_util_recent_summary,
    _json_safe,
    _now_ts,
    _queue_wait_stats,
    _read_events,
    _resource_event_summary,
    _sorted_resource_pressure_states,
    _utc_iso,
    build_lhr_state_from_events,
)
from scienceflow.research.solver.lnr.orchestration.runtime.services.worker_services import LegacyRunStatus


LHR_EVENTS_JSONL = "lhr_events.jsonl"
LHR_STATE_JSON = "lhr_state.json"


def _backfill_lhr_state_from_stage_performance(state: dict[str, Any], stage_csv: Path) -> dict[str, Any]:
    records = read_stage_performance_records(stage_csv)
    if not records:
        return state
    out = dict(state)
    record_count = len(records)
    if _int_or_zero(out.get("stage_count")) < record_count:
        out["stage_count"] = record_count
        workers = out.get("workers") if isinstance(out.get("workers"), dict) else {}
        if workers:
            out["workers"] = workers
            stage_counts = Counter(
                rec.worker_id or _worker_id_from_candidate(rec.candidate_id) or "W00"
                for rec in records
            )
            for worker_id, count in stage_counts.items():
                worker = workers.get(worker_id)
                if isinstance(worker, dict) and _int_or_zero(worker.get("stage_count")) < count:
                    worker["stage_count"] = count
    best = build_score_summary(records).get("valid_best_score")
    if isinstance(best, dict) and best:
        out["global_best"] = _score_summary_best_to_lhr_global_best(best)
        as_of = _path_mtime(stage_csv)
        if as_of:
            out["global_best_as_of"] = as_of
            out["global_best_as_of_utc"] = _utc_iso(as_of)
    return out


def _score_summary_best_to_lhr_global_best(best: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": best.get("candidate_id") or "",
        "worker_id": best.get("worker_id") or _worker_id_from_candidate(best.get("candidate_id")) or "",
        "stage_id": best.get("stage_id") or "",
        "snapshot_id": "",
        "metric_value": best.get("value"),
        "metric_name": best.get("metric_name") or "",
        "lower_is_better": best.get("lower_is_better"),
        "validation_ok": best.get("validation_ok"),
        "metric_validity": best.get("metric_validity") or "",
        "score_source": "lhr_stage_performance.csv",
    }


def _worker_id_from_candidate(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("W") and len(text) >= 3 and text[1:3].isdigit():
        return text[:3]
    return ""


def _int_or_zero(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _path_mtime(path: Path) -> float:
    try:
        return float(Path(path).stat().st_mtime)
    except OSError:
        return 0.0


class LHRStateMachineStore:
    """Small state/event sidecar for the lnr solver.

    This class is intentionally independent from the LNR solver core. It mirrors LHR
    control events into one stable stream and maintains a compact monitor state.
    """

    def __init__(
        self,
        *,
        log_dir: Path,
        worker_id: str,
        worker_index: int = 0,
        worker_count: int = 1,
        ledger_filename: str = ".run_results.md",
        event_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.worker_id = str(worker_id or "W00")
        self.worker_index = int(worker_index or 0)
        self.worker_count = max(1, int(worker_count or 1))
        self.ledger_filename = str(ledger_filename or "run_results.md")
        self.event_observer = event_observer
        self.event_count = 0
        self.stage_count = 0
        self.estra_decision_count = 0
        self.estra_invalid_decision_count = 0
        self.estra_continue_count = 0
        self.estra_redirect_count = 0
        self.estra_switch_count = 0
        self.estra_current_continue_count = 0
        self.estra_current_redirect_count = 0
        self.estra_stage_continue_count = 0
        self.estra_stage_redirect_count = 0
        self.estra_switch_stage_count = 0
        self.estra_compact_count = 0
        self.resource_job_count = 0
        self.resource_guard_action_count = 0
        self.resource_cleanup_heartbeat_count = 0
        self.resource_monitor_agent_event_count = 0
        self.resource_request_count = 0
        self.resource_source_hint_count = 0
        self.resource_gpu_queue_wait_count = 0
        self.resource_gpu_queue_timeout_count = 0
        self.resource_gpu_queue_wait_durations: list[float] = []
        self._resource_gpu_queue_wait_started_at: dict[str, float] = {}
        self.resource_policy_gate_count = 0
        self.resource_planner_guard_count = 0
        self.resource_admission_deferred_count = 0
        self.resource_gpu_lease_acquired_count = 0
        self.resource_gpu_lease_released_count = 0
        self.resource_gpu_util_sample_count = 0
        self.progress_heartbeat_count = 0
        self.progress_last: dict[str, Any] = {}
        self.agent_backoff_wait_count = 0
        self.agent_backoff_wait_sec_total = 0.0
        self.agent_backoff_wait_sec_max = 0.0
        self.post_feedback_action_count = 0
        self.post_feedback_action_counts: dict[str, int] = {}
        self.post_feedback_violation_count = 0
        self.agent_backoff_wait_active: dict[str, dict[str, Any]] = {}
        self.agent_backoff_wake_reasons: dict[str, int] = {}
        self.resource_active_leases: dict[str, dict[str, Any]] = {}
        self.resource_pending_gpu_jobs: dict[str, dict[str, Any]] = {}
        self.resource_gpu_pressure_states: dict[str, dict[str, Any]] = {}
        self.resource_gpu_util_rows: dict[str, list[dict[str, Any]]] = {}
        self.resource_last_events: list[dict[str, Any]] = []
        self.global_best: dict[str, Any] | None = None
        self.global_best_as_of = 0.0
        self.run_status = "created"
        self.started_at = _now_ts()
        self.last_event_at = self.started_at

    @property
    def events_path(self) -> Path:
        return self.log_dir / LHR_EVENTS_JSONL

    @property
    def state_path(self) -> Path:
        return self.log_dir / LHR_STATE_JSON

    def _base_state(self) -> dict[str, Any]:
        now = _now_ts()
        best = dict(self.global_best or {})
        return {
            "schema_version": 1,
            "solver": "lnr",
            "run_status": self.run_status,
            "started_at": self.started_at,
            "started_at_utc": _utc_iso(self.started_at),
            "worker_count": self.worker_count,
            "workers": {
                self.worker_id: {
                    "worker_id": self.worker_id,
                    "worker_index": self.worker_index,
                    "status": self.run_status,
                    "stage_count": self.stage_count,
                    "estra_decisions": self.estra_decision_count,
                    "estra_invalid_decisions": self.estra_invalid_decision_count,
                    "estra_continue_count": self.estra_continue_count,
                    "estra_redirect_count": self.estra_redirect_count,
                    "estra_switch_count": self.estra_switch_count,
                    "estra_current_continue_count": self.estra_current_continue_count,
                    "estra_current_redirect_count": self.estra_current_redirect_count,
                    "estra_stage_continue_count": self.estra_stage_continue_count,
                    "estra_stage_redirect_count": self.estra_stage_redirect_count,
                    "estra_switch_stage_count": self.estra_switch_stage_count,
                    "estra_compact_count": self.estra_compact_count,
                    "resource_jobs": self.resource_job_count,
                    "resource_guard_actions": self.resource_guard_action_count,
                    "resource_cleanup_heartbeats": self.resource_cleanup_heartbeat_count,
                    "resource_monitor_agent_events": self.resource_monitor_agent_event_count,
                    "resource_requests": self.resource_request_count,
                    "resource_source_hints": self.resource_source_hint_count,
                    "resource_gpu_queue_waits": self.resource_gpu_queue_wait_count,
                    "resource_gpu_queue_timeouts": self.resource_gpu_queue_timeout_count,
                    **_queue_wait_stats(
                        self.resource_gpu_queue_wait_durations,
                        waits=self.resource_gpu_queue_wait_count,
                        timeouts=self.resource_gpu_queue_timeout_count,
                    ),
                    "resource_policy_gates": self.resource_policy_gate_count,
                    "resource_planner_guards": self.resource_planner_guard_count,
                    "resource_admission_deferred": self.resource_admission_deferred_count,
                    "resource_gpu_lease_acquired": self.resource_gpu_lease_acquired_count,
                    "resource_gpu_lease_released": self.resource_gpu_lease_released_count,
                    "resource_gpu_util_samples": self.resource_gpu_util_sample_count,
                    "progress_heartbeats": self.progress_heartbeat_count,
                    "agent_backoff_waits": self.agent_backoff_wait_count,
                    "post_feedback_actions": self.post_feedback_action_count,
                    "agent_backoff_wait_sec_total": round(self.agent_backoff_wait_sec_total, 3),
                    "agent_backoff_wait_sec_max": round(self.agent_backoff_wait_sec_max, 3),
                    "last_event_at": self.last_event_at,
                    "last_event_at_utc": _utc_iso(self.last_event_at),
                }
            },
            "global_best": best,
            "global_best_as_of": self.global_best_as_of,
            "global_best_as_of_utc": _utc_iso(self.global_best_as_of) if self.global_best_as_of else "",
            "stage_count": self.stage_count,
            "estra_decisions": self.estra_decision_count,
            "estra_invalid_decisions": self.estra_invalid_decision_count,
            "estra_continue_count": self.estra_continue_count,
            "estra_redirect_count": self.estra_redirect_count,
            "estra_switch_count": self.estra_switch_count,
            "estra_current_continue_count": self.estra_current_continue_count,
            "estra_current_redirect_count": self.estra_current_redirect_count,
            "estra_stage_continue_count": self.estra_stage_continue_count,
            "estra_stage_redirect_count": self.estra_stage_redirect_count,
            "estra_start_current_count": self.estra_current_continue_count + self.estra_current_redirect_count,
            "estra_start_stage_count": self.estra_stage_continue_count + self.estra_stage_redirect_count,
            "estra_intent_continue_count": self.estra_current_continue_count + self.estra_stage_continue_count,
            "estra_intent_redirect_count": self.estra_current_redirect_count + self.estra_stage_redirect_count,
            "estra_switch_stage_count": self.estra_switch_stage_count,
            "estra_compact_count": self.estra_compact_count,
            "resource_jobs": self.resource_job_count,
            "resource_guard_actions": self.resource_guard_action_count,
            "resource_cleanup_heartbeats": self.resource_cleanup_heartbeat_count,
            "resource_monitor_agent_events": self.resource_monitor_agent_event_count,
            "resource_requests": self.resource_request_count,
            "resource_source_hints": self.resource_source_hint_count,
            "resource_gpu_queue_waits": self.resource_gpu_queue_wait_count,
            "resource_gpu_queue_timeouts": self.resource_gpu_queue_timeout_count,
            **_queue_wait_stats(
                self.resource_gpu_queue_wait_durations,
                waits=self.resource_gpu_queue_wait_count,
                timeouts=self.resource_gpu_queue_timeout_count,
            ),
            "resource_policy_gates": self.resource_policy_gate_count,
            "resource_planner_guards": self.resource_planner_guard_count,
            "resource_admission_deferred": self.resource_admission_deferred_count,
            "resource_gpu_lease_acquired": self.resource_gpu_lease_acquired_count,
            "resource_gpu_lease_released": self.resource_gpu_lease_released_count,
            "resource_gpu_util_samples": self.resource_gpu_util_sample_count,
            "progress_heartbeats": self.progress_heartbeat_count,
            "progress_last": dict(self.progress_last or {}),
            "agent_backoff_waits": self.agent_backoff_wait_count,
            "post_feedback_actions": self.post_feedback_action_count,
            "post_feedback_action_counts": dict(sorted(self.post_feedback_action_counts.items())),
            "post_feedback_violations": self.post_feedback_violation_count,
            "agent_backoff_wait_sec_total": round(self.agent_backoff_wait_sec_total, 3),
            "agent_backoff_wait_sec_max": round(self.agent_backoff_wait_sec_max, 3),
            "agent_backoff_wait_active": list(self.agent_backoff_wait_active.values()),
            "agent_backoff_wake_reasons": dict(sorted(self.agent_backoff_wake_reasons.items())),
            "resource_active_leases": list(self.resource_active_leases.values()),
            "resource_pending_gpu_jobs": list(self.resource_pending_gpu_jobs.values()),
            "resource_gpu_pressure_states": _sorted_resource_pressure_states(self.resource_gpu_pressure_states),
            "resource_gpu_util_recent": _gpu_util_recent_summary(self.resource_gpu_util_rows),
            "resource_last_events": self.resource_last_events[-12:],
            "event_count": self.event_count,
            "ledger_filename": self.ledger_filename,
            "generated_at": now,
            "generated_at_utc": _utc_iso(now),
            "eventual_consistency": "single-writer state; coordinator aggregation is added in a later phase",
        }

    def write_state(self) -> None:
        _atomic_write_json(self.state_path, self._base_state())

    def mark_run_status(
        self,
        status: LegacyRunStatus | str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        typed_status = LegacyRunStatus(status)
        status_value = typed_status.value
        self.run_status = status_value
        self.append_event(
            "worker_status",
            task_type="repl_search",
            task_id=f"repl_search:{self.worker_id}",
            status=status_value,
            payload=payload or {},
        )

    def append_event(
        self,
        event_type: str,
        *,
        task_type: str = "",
        task_id: str = "",
        status: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = _now_ts()
        legacy_fields = {
            "schema_version": 1,
            "timestamp": now,
            "timestamp_utc": _utc_iso(now),
            "event": str(event_type or "event"),
            "task_type": str(task_type or ""),
            "task_id": str(task_id or ""),
            "status": str(status or ""),
            "worker_id": self.worker_id,
            "worker_index": self.worker_index,
            "payload": _json_safe(payload or {}),
        }
        runtime_event = runtime_event_from_legacy(
            legacy_fields,
            session_id=str(task_id or self.worker_id),
            run_id=self.log_dir.parent.name or "scienceflow",
            agent_id=self.worker_id,
        )
        event = legacy_event_from_runtime(runtime_event)
        JsonEventJournal(self.events_path).append(event)
        self.event_count += 1
        self.last_event_at = now
        self._update_counters(event)
        self.write_state()
        if self.event_observer is not None:
            try:
                self.event_observer(event)
            except Exception:
                # Compatibility observers must never alter the authoritative legacy path.
                pass

    @classmethod
    def aggregate_logs(
        cls,
        *,
        output_log_dir: Path,
        input_log_dirs: list[Path],
        worker_count: int = 1,
        run_status: str = "",
        ledger_filename: str = ".run_results.md",
    ) -> dict[str, Any]:
        seen: set[str] = set()
        events: list[dict[str, Any]] = []
        for log_dir in input_log_dirs:
            for event in _read_events(Path(log_dir) / LHR_EVENTS_JSONL):
                key = _event_dedupe_key(event)
                if key in seen:
                    continue
                seen.add(key)
                events.append(event)
        events.sort(key=lambda e: float(e.get("timestamp") or 0.0))
        out_dir = Path(output_log_dir)
        _atomic_write_jsonl(out_dir / LHR_EVENTS_JSONL, events)
        state = build_lhr_state_from_events(
            events,
            worker_count=worker_count,
            run_status=run_status,
            ledger_filename=ledger_filename,
        )
        state = _backfill_lhr_state_from_stage_performance(
            state,
            out_dir / "lhr_stage_performance.csv",
        )
        _atomic_write_json(out_dir / LHR_STATE_JSON, state)
        return state

    def _update_counters(self, event: dict[str, Any]) -> None:
        et = str(event.get("event") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        _apply_resource_event_to_views(
            event,
            active_leases=self.resource_active_leases,
            pending_jobs=self.resource_pending_gpu_jobs,
            last_events=self.resource_last_events,
            pressure_states=self.resource_gpu_pressure_states,
        )
        if et == "stage_captured":
            self.stage_count += 1
            self._maybe_update_best(payload)
        elif et == "estra_decision":
            self.estra_decision_count += 1
            action = str(payload.get("action") or "")
            startpoint = str(payload.get("startpoint") or ("previous_stage" if action == "switch_stage" else "current_workspace"))
            intent = str(payload.get("intent") or ("redirect" if action == "keep_but_redirect" else "continue"))
            valid_estra_action = action in {"keep_current", "keep_but_redirect", "switch_stage"}
            if action == "keep_current":
                self.estra_continue_count += 1
            elif action == "keep_but_redirect":
                self.estra_redirect_count += 1
            elif action == "switch_stage":
                self.estra_switch_count += 1
            if valid_estra_action:
                if startpoint == "previous_stage" and intent == "redirect":
                    self.estra_stage_redirect_count += 1
                elif startpoint == "previous_stage":
                    self.estra_stage_continue_count += 1
                elif intent == "redirect":
                    self.estra_current_redirect_count += 1
                else:
                    self.estra_current_continue_count += 1
        elif et == "estra_invalid":
            self.estra_invalid_decision_count += 1
        elif et == "estra_stage_switched":
            self.estra_switch_stage_count += 1
        elif et == "estra_keep_current_compacted":
            self.estra_compact_count += 1
        elif et == "resource_job_started":
            self.resource_job_count += 1
        elif et == "resource_guard_action":
            self.resource_guard_action_count += 1
        elif et == "resource_cleanup_heartbeat":
            self.resource_cleanup_heartbeat_count += 1
        elif et == "resource_monitor_agent_shadow":
            self.resource_monitor_agent_event_count += 1
        elif et == "resource_request_created":
            self.resource_request_count += 1
        elif et == "resource_source_hint_detected":
            self.resource_source_hint_count += 1
        elif et == "resource_gpu_queue_wait_started":
            self.resource_gpu_queue_wait_count += 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            if job_id:
                self._resource_gpu_queue_wait_started_at[job_id] = _float_or_zero(event.get("timestamp"))
        elif et == "resource_gpu_queue_timeout":
            self.resource_gpu_queue_timeout_count += 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            started_at = self._resource_gpu_queue_wait_started_at.pop(job_id, 0.0) if job_id else 0.0
            elapsed = _float_or_zero(payload.get("elapsed_sec"))
            if elapsed <= 0.0 and started_at > 0.0:
                elapsed = _float_or_zero(event.get("timestamp")) - started_at
            self.resource_gpu_queue_wait_durations.append(max(0.0, elapsed))
        elif et == "resource_policy_gate":
            self.resource_policy_gate_count += 1
        elif et == "resource_planner_guard":
            self.resource_planner_guard_count += 1
        elif et == "resource_admission_deferred":
            self.resource_admission_deferred_count += 1
        elif et == "resource_gpu_lease_acquired":
            self.resource_gpu_lease_acquired_count += 1
            job_id = str(payload.get("job_id") or event.get("task_id") or "")
            started_at = self._resource_gpu_queue_wait_started_at.pop(job_id, 0.0) if job_id else 0.0
            if started_at > 0.0:
                self.resource_gpu_queue_wait_durations.append(
                    max(0.0, _float_or_zero(event.get("timestamp")) - started_at)
                )
        elif et == "resource_gpu_lease_released":
            self.resource_gpu_lease_released_count += 1
        elif et == "resource_gpu_util_sampled":
            self.resource_gpu_util_sample_count += 1
            _append_gpu_util_rows(self.resource_gpu_util_rows, event)
        elif et == "progress_heartbeat":
            self.progress_heartbeat_count += 1
            self.progress_last = _resource_event_summary(event)
        elif et == "agent_backoff_wait_started":
            self.agent_backoff_wait_count += 1
            wait_id = str(payload.get("wait_id") or event.get("task_id") or "")
            if wait_id:
                self.agent_backoff_wait_active[wait_id] = _resource_event_summary(event)
        elif et == "agent_backoff_wait_finished":
            elapsed = _float_or_zero(payload.get("elapsed_sec"))
            self.agent_backoff_wait_sec_total += elapsed
            self.agent_backoff_wait_sec_max = max(self.agent_backoff_wait_sec_max, elapsed)
            wake = str(payload.get("wake_reason") or "unknown")
            self.agent_backoff_wake_reasons[wake] = self.agent_backoff_wake_reasons.get(wake, 0) + 1
            wait_id = str(payload.get("wait_id") or event.get("task_id") or "")
            if wait_id:
                self.agent_backoff_wait_active.pop(wait_id, None)
        elif et == "post_feedback_action":
            self.post_feedback_action_count += 1
            action = str(payload.get("action") or "unknown")
            self.post_feedback_action_counts[action] = self.post_feedback_action_counts.get(action, 0) + 1
            if bool(payload.get("violates_feedback")):
                self.post_feedback_violation_count += 1

    def _maybe_update_best(self, payload: dict[str, Any]) -> None:
        record = score_record_from_stage_payload(payload, worker_id=self.worker_id)
        if record is None or not record.valid_comparable:
            return
        metric = record.value
        if metric != metric:
            return
        current_metric = None
        if self.global_best:
            try:
                current_metric = float(self.global_best.get("metric_value"))
            except (TypeError, ValueError):
                current_metric = None
        lower_is_better = record.lower_is_better
        better = current_metric is None or (metric < current_metric if lower_is_better else metric > current_metric)
        if not better:
            return
        now = _now_ts()
        self.global_best = {
            "candidate_id": record.candidate_id or f"{self.worker_id}:{payload.get('stage_id') or ''}",
            "worker_id": record.worker_id or self.worker_id,
            "stage_id": record.stage_id or payload.get("stage_id") or "",
            "snapshot_id": payload.get("snapshot_id") or "",
            "metric_value": metric,
            "metric_name": record.metric_name,
            "lower_is_better": lower_is_better,
            "validation_ok": record.validation_ok,
            "metric_validity": record.metric_validity,
            "validity": record.validity,
            "evaluator_backend": record.evaluator_backend,
            "evaluator_status": record.evaluator_status,
        }
        self.global_best_as_of = now
