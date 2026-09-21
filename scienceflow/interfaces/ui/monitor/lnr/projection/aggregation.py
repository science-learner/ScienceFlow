"""Read-only LNR monitor responsibility: aggregation."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scienceflow.research.solver.lnr.lifecycle.stage.metrics.score_summary import (
    build_score_summary,
    score_record_from_stage_performance_row,
)
from scienceflow.interfaces.ui.monitor.lnr.projection.events import _iter_recent_jsonl, _iter_recent_jsonl_many, _job_is_active, _job_is_stale_for_display, _job_summary, _select_active_jobs_for_display
from scienceflow.interfaces.ui.monitor.lnr.rendering.formatting import _latest_stage_by_worker, _stage_brief, _summary_lower_is_better, _to_float, _to_int, _truthy
from scienceflow.interfaces.ui.monitor.lnr.projection.summary import _merge_job_summary, _recent_event, _wait_summary, _worker_states
from scienceflow.interfaces.ui.monitor.lnr.rendering.reader import _first_non_empty, _load_json_dict

def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []

def _summarize_stages(rows: list[dict[str, str]]) -> dict[str, Any]:
    records = [rec for row in rows if (rec := score_record_from_stage_performance_row(row)) is not None]
    score_summary = build_score_summary(records)
    best_raw = score_summary.get("best_score") if isinstance(score_summary.get("best_score"), dict) else {}
    best_valid = score_summary.get("valid_best_score") if isinstance(score_summary.get("valid_best_score"), dict) else {}
    lower = _summary_lower_is_better(rows, best_raw, best_valid)
    with_metric = [r for r in rows if _to_float(r.get("metric_value")) is not None]
    ready = [
        r for r in with_metric
        if _truthy(r.get("candidate_ready")) and _truthy(r.get("selection_eligible"))
    ]
    stage_keys = [_stage_count_key(row) for row in rows]
    stage_keys = [key for key in stage_keys if key is not None]
    unique_stage_keys = set(stage_keys)
    stages_by_worker = Counter(worker for worker, _ in unique_stage_keys)
    candidate_stages_by_worker = Counter((r.get("worker_id") or "?").upper() for r in rows)
    return {
        "stage_count": len(unique_stage_keys) if unique_stage_keys else len(rows),
        "candidate_stage_count": len(rows),
        "stages_by_worker": dict(stages_by_worker or candidate_stages_by_worker),
        "candidate_stages_by_worker": dict(candidate_stages_by_worker),
        "ready_stage_count": len([r for r in rows if _truthy(r.get("candidate_ready"))]),
        "eligible_stage_count": len(ready),
        "duplicate_stage_count": len([
            r for r in rows
            if (r.get("duplicate_submission_of_stage") or r.get("duplicate_submission_of_snapshot_id"))
        ]),
        "low_validity_stage_count": len([
            r for r in rows
            if str(r.get("metric_validity") or "").strip().lower() in {"low", "invalid"}
        ]),
        "latest_stage": _stage_brief(rows[-1]) if rows else {},
        "latest_stage_by_worker": _latest_stage_by_worker(rows),
        "best_raw_metric": best_raw.get("value"),
        "best_raw_candidate": best_raw.get("candidate_id") or "",
        "best_raw_validity": best_raw.get("metric_validity") or "",
        "best_valid_metric": best_valid.get("value"),
        "best_valid_candidate": best_valid.get("candidate_id") or "",
        "best_valid_validity": best_valid.get("metric_validity") or "",
        "lower_is_better": lower,
    }

def _stage_count_key(row: dict[str, str]) -> tuple[str, str] | None:
    worker = str(row.get("worker_id") or "?").strip().upper() or "?"
    stage_id = str(row.get("stage_id") or "").strip()
    if _looks_like_stage_id(stage_id):
        return worker, stage_id
    candidate_id = str(row.get("candidate_id") or "").strip()
    candidate_stage = candidate_id.rsplit(":", 1)[-1] if candidate_id else ""
    if _looks_like_stage_id(candidate_stage):
        return worker, candidate_stage
    return None

def _looks_like_stage_id(value: str) -> bool:
    return len(value) > 1 and value[0].upper() == "S" and value[1:].isdigit()

def _summarize_resource_events(paths: Path | Iterable[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    advisory: Counter[str] = Counter()
    boundaries: Counter[str] = Counter()
    recent: list[dict[str, Any]] = []
    latest_jobs: dict[str, dict[str, Any]] = {}
    latest_waits: dict[str, dict[str, Any]] = {}
    first_event_at: float | None = None
    last_event_at: float | None = None

    if isinstance(paths, Path):
        event_paths = [paths]
    else:
        event_paths = [Path(p) for p in paths]

    for obj in _iter_recent_jsonl_many(event_paths):
        created_at = _to_float(obj.get("created_at") or obj.get("timestamp"))
        if created_at is not None:
            first_event_at = created_at if first_event_at is None else min(first_event_at, created_at)
            last_event_at = created_at if last_event_at is None else max(last_event_at, created_at)
        event_type = str(obj.get("event_type") or obj.get("event") or "unknown")
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        counts[event_type] += 1

        outcome = payload.get("execution_outcome") or payload.get("outcome")
        if outcome:
            outcomes[str(outcome)] += 1
        boundary = payload.get("resource_review_boundary") or payload.get("review_boundary")
        if isinstance(boundary, dict) and boundary.get("kind"):
            boundaries[str(boundary["kind"])] += 1
        elif boundary:
            boundaries[str(boundary)] += 1

        adv = payload.get("advisory") or payload.get("parsed_response")
        if isinstance(adv, dict) and adv.get("preference"):
            advisory[str(adv["preference"])] += 1

        command_id = str(obj.get("command_id") or payload.get("job_id") or "")
        if command_id and event_type in {
            "snapshot",
            "kill_proposal",
            "resource_kill_executed",
            "resource_review_outcome",
            "progress_heartbeat",
            "resource_monitor_heartbeat",
            "resource_job_finished",
            "resource_source_hint_detected",
        }:
            current = _job_summary(obj, payload)
            latest_jobs[command_id] = _merge_job_summary(latest_jobs.get(command_id), current)

        if event_type in {"admission_pending", "managed_resource_wait_offered"}:
            wait = _wait_summary(obj, payload)
            worker_id = str(wait.get("worker_id") or "").upper()
            if worker_id:
                previous = latest_waits.get(worker_id)
                if previous is None or float(wait.get("observed_at") or 0.0) >= float(previous.get("observed_at") or 0.0):
                    latest_waits[worker_id] = wait

        if event_type in {
            "resource_kill_executed",
            "resource_review_outcome",
            "main_agent_advisory",
            "execution",
            "admission_pending",
            "managed_resource_wait_offered",
        }:
            recent.append(_recent_event(obj, payload))

    active_job_candidates = [
        j for j in latest_jobs.values()
        if _job_is_active(j) and not _job_is_stale_for_display(j)
    ]
    active_jobs = _select_active_jobs_for_display(active_job_candidates)
    worker_states = _worker_states(active_job_candidates, latest_waits)
    return {
        "counts": dict(counts),
        "outcomes": dict(outcomes),
        "actionable_outcomes": {
            key: value for key, value in outcomes.items() if key != "NO_ACTION"
        },
        "advisory_preferences": dict(advisory),
        "review_boundaries": dict(boundaries),
        "active_jobs": active_jobs[:6],
        "worker_states": worker_states,
        "recent": recent[-8:],
        "first_event_at": first_event_at,
        "last_event_at": last_event_at,
    }

def _summarize_estra_events(paths: Path | Iterable[Path]) -> dict[str, int]:
    actions: Counter[str] = Counter()
    axes: Counter[str] = Counter()
    requested_compact_count = 0
    actual_compact_count = 0
    invalid_count = 0
    raw_decision_count = 0
    repeat_decision_count = 0
    text_only_check_count = 0
    context_check_count = 0
    text_only_trigger_count = 0
    context_trigger_count = 0
    seen_decisions: set[tuple[str, str, str, str, str, str]] = set()
    if isinstance(paths, Path):
        events = _iter_recent_jsonl(paths)
    else:
        events = _iter_recent_jsonl_many(paths)
    for obj in events:
        event_type = str(obj.get("event") or obj.get("event_type") or "")
        if event_type == "text_only_estra_check":
            text_only_check_count += 1
            continue
        if event_type in {"context_limit_estra_check", "cache_hygiene_compact_triggered"}:
            context_check_count += 1
            continue
        if event_type in {"estra_keep_current_compacted", "estra_memory_compacted"}:
            actual_compact_count += 1
            continue
        if event_type != "estra_decision":
            continue
        raw_decision_count += 1
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        action = str(payload.get("action") or "").strip()
        startpoint = str(payload.get("startpoint") or ("previous_stage" if action == "switch_stage" else "current_workspace"))
        intent = str(payload.get("intent") or ("redirect" if action == "keep_but_redirect" else "continue"))
        decision_key = _estra_decision_key(
            obj,
            payload,
            action=action,
            startpoint=startpoint,
            intent=intent,
        )
        if decision_key in seen_decisions:
            repeat_decision_count += 1
            continue
        seen_decisions.add(decision_key)
        if action in {"switch_stage", "keep_but_redirect", "keep_current"}:
            actions[action] += 1
        else:
            invalid_count += 1
        if startpoint == "previous_stage" and intent == "redirect":
            axes["stage_redirect"] += 1
        elif startpoint == "previous_stage":
            axes["stage_continue"] += 1
        elif intent == "redirect":
            axes["current_redirect"] += 1
        else:
            axes["current_continue"] += 1
        trigger_source = str(payload.get("trigger_source") or "").strip()
        if trigger_source == "text_only":
            text_only_trigger_count += 1
        elif trigger_source in {"context_limit", "context_hygiene"}:
            context_trigger_count += 1
        if bool(payload.get("compact")):
            requested_compact_count += 1
    compact_count = actual_compact_count or requested_compact_count
    return {
        "decision_count": int(sum(actions.values()) + invalid_count),
        "raw_decision_count": int(raw_decision_count),
        "repeat_decision_count": int(repeat_decision_count),
        "switch_count": int(actions.get("switch_stage", 0)),
        "redirect_count": int(actions.get("keep_but_redirect", 0)),
        "continue_count": int(actions.get("keep_current", 0)),
        "current_continue_count": int(axes.get("current_continue", 0)),
        "current_redirect_count": int(axes.get("current_redirect", 0)),
        "stage_continue_count": int(axes.get("stage_continue", 0)),
        "stage_redirect_count": int(axes.get("stage_redirect", 0)),
        "invalid_count": int(invalid_count),
        "compact_count": int(compact_count),
        "text_only_check_count": int(text_only_check_count),
        "context_check_count": int(context_check_count),
        "text_only_trigger_count": int(text_only_trigger_count),
        "context_trigger_count": int(context_trigger_count),
    }

def _estra_decision_key(
    obj: dict[str, Any],
    payload: dict[str, Any],
    *,
    action: str,
    startpoint: str,
    intent: str,
) -> tuple[str, str, str, str, str, str]:
    worker = str(obj.get("worker_id") or payload.get("worker_id") or "?").strip().upper() or "?"
    stage = _first_non_empty(
        payload.get("latest_stage"),
        payload.get("target_stage"),
        obj.get("latest_stage"),
        payload.get("stage_id"),
        obj.get("stage_id"),
        payload.get("stage_count"),
        obj.get("stage_count"),
    )
    if not stage:
        ts = _to_float(obj.get("timestamp") or obj.get("created_at"))
        stage = f"ts:{ts:.3f}" if ts is not None else "unknown"
    trigger = str(payload.get("trigger_source") or "").strip()
    return (worker, str(stage).strip().upper(), action, startpoint, intent, trigger)

def _summarize_worker_estra_states(task_root: Path) -> dict[str, int]:
    """Return estra counters from live worker state files.

    Worker-level ``lhr_state.json`` can be fresher than the task-level event
    rollup while a run is active.  The monitor is read-only, so this is used as
    a display fallback only; it does not backfill or mutate task state.
    """

    summary = _empty_estra_summary()
    field_map = {
        "decision_count": "estra_decisions",
        "switch_count": "estra_switch_count",
        "redirect_count": "estra_redirect_count",
        "continue_count": "estra_continue_count",
        "current_continue_count": "estra_current_continue_count",
        "current_redirect_count": "estra_current_redirect_count",
        "stage_continue_count": "estra_stage_continue_count",
        "stage_redirect_count": "estra_stage_redirect_count",
        "invalid_count": "estra_invalid_decisions",
        "compact_count": "estra_compact_count",
    }
    paths = list(Path(task_root).glob("task_logs/workers/w*/lhr_state.json"))
    paths.extend(Path(task_root).glob("workers/w*/logs/lhr_state.json"))
    for path in sorted(paths):
        data = _load_json_dict(path)
        if not data:
            continue
        for out_key, state_key in field_map.items():
            summary[out_key] += _to_int(data.get(state_key)) or 0
    return summary

def _merge_estra_summaries(event_summary: dict[str, int], worker_summary: dict[str, int]) -> dict[str, int]:
    """Prefer event counters, but fill stale/missing action counts from workers."""

    merged = _empty_estra_summary()
    merged.update({key: int(event_summary.get(key, 0) or 0) for key in merged})
    action_keys = (
        "decision_count",
        "switch_count",
        "redirect_count",
        "continue_count",
        "current_continue_count",
        "current_redirect_count",
        "stage_continue_count",
        "stage_redirect_count",
        "invalid_count",
        "compact_count",
    )
    has_event_actions = any(int(event_summary.get(key, 0) or 0) for key in ("raw_decision_count", *action_keys))
    for key in action_keys:
        if has_event_actions:
            merged[key] = int(event_summary.get(key, 0) or 0)
        else:
            merged[key] = int(worker_summary.get(key, 0) or 0)
    if not has_event_actions and merged.get("decision_count"):
        merged["raw_decision_count"] = int(merged.get("decision_count") or 0)
    return merged

def _empty_estra_summary() -> dict[str, int]:
    return {
        "decision_count": 0,
        "raw_decision_count": 0,
        "repeat_decision_count": 0,
        "switch_count": 0,
        "redirect_count": 0,
        "continue_count": 0,
        "current_continue_count": 0,
        "current_redirect_count": 0,
        "stage_continue_count": 0,
        "stage_redirect_count": 0,
        "invalid_count": 0,
        "compact_count": 0,
        "text_only_check_count": 0,
        "context_check_count": 0,
        "text_only_trigger_count": 0,
        "context_trigger_count": 0,
    }

__all__ = tuple(name for name in globals() if not name.startswith("__"))
