"""Top-level read-only LNR monitor state projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scienceflow.interfaces.ui.monitor.lnr.projection.aggregation import (
    _load_csv_rows,
    _merge_estra_summaries,
    _summarize_estra_events,
    _summarize_resource_events,
    _summarize_stages,
    _summarize_worker_estra_states,
)
from scienceflow.interfaces.ui.monitor.lnr.rendering.formatting import (
    _format_timestamp,
    _newest_mtime,
)
from scienceflow.interfaces.ui.monitor.lnr.rendering.reader import (
    _filter_stale_wait_states,
    _elapsed_from_live_process_summary,
    _live_process_summary_for_root,
    _load_current_resource_snapshot,
)
from scienceflow.interfaces.ui.monitor.lnr.rendering.status import (
    _TERMINAL_DISPLAY_STATUSES,
    _display_status,
    _effective_wall_clock_budget_sec,
    _elapsed_from_final_state,
    _elapsed_from_resource_range,
    _load_llm_config_summary,
    _load_live_run_started_at,
    _load_parallel_task_state,
    _load_run_status,
    _load_wall_clock_budget_sec,
    _lnr_event_paths,
    _summarize_time_traces,
)
from scienceflow.runtime.observability.telemetry.agent.llm_config_summary import format_llm_config_summary

def load_lnr_task_state(task_root: Path) -> dict[str, Any]:
    """Return a monitor state dict for an LNR task workspace."""
    root = Path(task_root)
    if not (root / "task_logs").is_dir():
        return {}

    stages = _load_csv_rows(root / "task_logs" / "lhr_stage_performance.csv")
    stage_summary = _summarize_stages(stages)
    event_paths = _lnr_event_paths(root)
    resource_summary = _summarize_resource_events([
        root / "task_logs" / "resource" / "resource_events.jsonl",
        *event_paths,
    ])
    resource_current = _load_current_resource_snapshot(root)
    estra_summary = _merge_estra_summaries(
        _summarize_estra_events(event_paths),
        _summarize_worker_estra_states(root),
    )
    trace_summary = _summarize_time_traces(root)
    budget_sec = _load_wall_clock_budget_sec(root)
    newest_mtime = _newest_mtime(root, stages)

    active_jobs = resource_summary.get("active_jobs", [])
    worker_states = _filter_stale_wait_states(
        resource_summary.get("worker_states", []),
        resource_current,
    )
    live_process_summary = _live_process_summary_for_root(root)
    running_process_count = int(live_process_summary.get("count") or 0)
    final_state = _load_parallel_task_state(root)
    budget_sec = _effective_wall_clock_budget_sec(budget_sec, final_state)
    llm_config = _load_llm_config_summary(root, final_state=final_state, trace_summary=trace_summary)
    run_status = _load_run_status(root, final_state=final_state)
    live = running_process_count > 0 or run_status in {"run", "running"}
    run_started_at = _load_live_run_started_at(root) if live else None
    elapsed_sec = _elapsed_from_live_process_summary(live_process_summary, final_state) if live else None
    if elapsed_sec is None and not live:
        elapsed_sec = _elapsed_from_final_state(final_state)
    if elapsed_sec is None:
        elapsed_sec = _elapsed_from_resource_range(resource_summary, live=live, run_started_at=run_started_at)
    progress_ratio = (elapsed_sec / budget_sec) if budget_sec and elapsed_sec is not None else None
    status = _display_status(run_status=run_status, running_process_count=running_process_count, active_jobs=active_jobs)
    if status in _TERMINAL_DISPLAY_STATUSES:
        active_jobs = []
        worker_states = []
    return {
        "monitor_kind": "lnr_task",
        "status": status,
        "task_name": root.name,
        "exp_id": root.name,
        "task_root": str(root),
        "timestamp": _format_timestamp(newest_mtime),
        "elapsed_sec": elapsed_sec,
        "total_sec": budget_sec,
        "run_started_at": run_started_at,
        "remaining_sec": (max(0.0, budget_sec - elapsed_sec) if budget_sec and elapsed_sec is not None else None),
        "progress_ratio": (min(1.0, max(0.0, progress_ratio)) if progress_ratio is not None else None),
        "stage_count": stage_summary["stage_count"],
        "candidate_stage_count": stage_summary["candidate_stage_count"],
        "stages_by_worker": stage_summary["stages_by_worker"],
        "candidate_stages_by_worker": stage_summary["candidate_stages_by_worker"],
        "ready_stage_count": stage_summary["ready_stage_count"],
        "eligible_stage_count": stage_summary["eligible_stage_count"],
        "duplicate_stage_count": stage_summary["duplicate_stage_count"],
        "low_validity_stage_count": stage_summary["low_validity_stage_count"],
        "latest_stage": stage_summary["latest_stage"],
        "latest_stage_by_worker": stage_summary["latest_stage_by_worker"],
        "best_raw_metric": stage_summary["best_raw_metric"],
        "best_valid_metric": stage_summary["best_valid_metric"],
        "best_raw_candidate": stage_summary["best_raw_candidate"],
        "best_valid_candidate": stage_summary["best_valid_candidate"],
        "best_raw_validity": stage_summary["best_raw_validity"],
        "best_valid_validity": stage_summary["best_valid_validity"],
        "lower_is_better": stage_summary["lower_is_better"],
        "active_jobs": active_jobs,
        "active_job_count": len(active_jobs),
        "worker_states": worker_states,
        "running_process_count": running_process_count,
        "resource_counts": resource_summary["counts"],
        "resource_current": resource_current,
        "resource_recent": resource_summary["recent"],
        "resource_outcomes": resource_summary["outcomes"],
        "resource_actionable_outcomes": resource_summary["actionable_outcomes"],
        "advisory_preferences": resource_summary["advisory_preferences"],
        "review_boundaries": resource_summary["review_boundaries"],
        "estra_decision_count": estra_summary["decision_count"],
        "estra_switch_count": estra_summary["switch_count"],
        "estra_redirect_count": estra_summary["redirect_count"],
        "estra_continue_count": estra_summary["continue_count"],
        "estra_current_continue_count": estra_summary["current_continue_count"],
        "estra_current_redirect_count": estra_summary["current_redirect_count"],
        "estra_stage_continue_count": estra_summary["stage_continue_count"],
        "estra_stage_redirect_count": estra_summary["stage_redirect_count"],
        "estra_invalid_count": estra_summary["invalid_count"],
        "estra_compact_count": estra_summary["compact_count"],
        "estra_raw_decision_count": estra_summary["raw_decision_count"],
        "estra_repeat_decision_count": estra_summary["repeat_decision_count"],
        "estra_text_only_check_count": estra_summary["text_only_check_count"],
        "estra_context_check_count": estra_summary["context_check_count"],
        "estra_text_only_trigger_count": estra_summary["text_only_trigger_count"],
        "estra_context_trigger_count": estra_summary["context_trigger_count"],
        "total_tokens_in": trace_summary["tokens_in"],
        "total_tokens_out": trace_summary["tokens_out"],
        "total_tokens_cached": trace_summary["tokens_cached"],
        "total_llm_calls": trace_summary["llm_calls"],
        "total_llm_cost_usd": trace_summary["cost_usd"],
        "llm_cost_known_calls": trace_summary["cost_known_calls"],
        "llm_cache_rate": trace_summary["cache_rate"],
        "llm_config": llm_config,
        "llm_config_text": format_llm_config_summary(llm_config),
        "observed_llm_models": llm_config.get("observed_models", []) if isinstance(llm_config, dict) else [],
        "max_ttft_sec": trace_summary["max_ttft_sec"],
        "max_tpot_ms": trace_summary["max_tpot_ms"],
    }

__all__ = tuple(name for name in globals() if not name.startswith("__"))
