"""Shared imports and immutable monitor projection constants."""

from __future__ import annotations
_MAX_RESOURCE_EVENTS = 5000
_RECENT_JSONL_TAIL_BYTES = 4 * 1024 * 1024
_TERMINAL_DISPLAY_STATUSES = {
    "finished",
    "failed",
    "early_stop",
    "budget_expired",
    "budget_done",
    "steps_completed",
    "timeout",
    "skipped",
    "stopped_by_user",
}
_PARALLEL_FINAL_STATUSES = {"finished", "failed", "budget_done", "timeout", "skipped", "stopped_by_user"}
_REAL_EPOCH_MIN_TS = 1_600_000_000.0
_STALE_ACTIVE_JOB_DISPLAY_SEC = 900.0
_HEARTBEAT_CONTEXT_KEYS = {
    "heartbeat_observed_at",
    "heartbeat_phase",
    "progress_current",
    "progress_total",
    "progress_unit",
    "progress_advanced",
    "metric_name",
    "metric_value",
}
_STALE_HEARTBEAT_CONTEXT_GAP_SEC = 600.0

__all__ = tuple(name for name in globals() if not name.startswith("__"))
