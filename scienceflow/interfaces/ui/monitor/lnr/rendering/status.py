"""Read-only LNR monitor responsibility: status."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from scienceflow.interfaces.ui.monitor.lnr.projection.aggregation import _load_csv_rows
from scienceflow.interfaces.ui.monitor.lnr.projection.events import _iter_jsonl
from scienceflow.interfaces.ui.monitor.lnr.rendering.base import (
    _PARALLEL_FINAL_STATUSES,
    _TERMINAL_DISPLAY_STATUSES,
)
from scienceflow.interfaces.ui.monitor.lnr.rendering.formatting import (
    _first_float,
    _first_int,
    _max_optional,
    _to_float,
)
from scienceflow.interfaces.ui.monitor.lnr.rendering.time_trace import (
    _time_trace_paths,
)
from scienceflow.runtime.observability.telemetry.agent.llm_config_summary import (
    merge_observed_models,
    observed_models_from_log_file,
    summarize_llm_config_from_resolved_config,
)
from scienceflow.runtime.observability.telemetry.agent.llm_cost import (
    estimate_llm_cost_usd,
    extract_model_from_trace_detail,
    load_model_config_prices,
)


def _load_wall_clock_budget_sec(task_root: Path) -> float | None:
    path = task_root / "resolved_config.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    lnr = data.get("lnr") if isinstance(data, dict) else None
    if not isinstance(lnr, dict):
        return None
    value = _to_float(lnr.get("wall_clock_budget_sec"))
    return value if value and value > 0 else None

def _effective_wall_clock_budget_sec(config_budget_sec: float | None, final_state: dict[str, Any]) -> float | None:
    total_budget = _to_float(final_state.get("resume_total_budget_sec"))
    if total_budget is not None and total_budget > 0:
        if config_budget_sec is None or config_budget_sec <= 0:
            return total_budget
        return max(config_budget_sec, total_budget)
    return config_budget_sec

def _load_llm_prices(task_root: Path) -> dict[str, Any]:
    path = task_root / "resolved_config.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    agent = data.get("agent") if isinstance(data, dict) else None
    if not isinstance(agent, dict):
        return {}
    paths = []
    for role in ("code", "feedback"):
        stage = agent.get(role)
        if isinstance(stage, dict) and stage.get("model_config_path"):
            paths.append(stage["model_config_path"])
    try:
        return load_model_config_prices(*paths)
    except ValueError:
        return {}

def _load_llm_config_summary(
    task_root: Path,
    *,
    final_state: dict[str, Any],
    trace_summary: dict[str, Any],
) -> dict[str, Any]:
    state_llm = final_state.get("llm_config") if isinstance(final_state, dict) else None
    if isinstance(state_llm, dict) and state_llm:
        summary = dict(state_llm)
    else:
        summary = _load_llm_config_from_resolved_config(task_root)
    log_models = observed_models_from_log_file(final_state.get("log_file")) if isinstance(final_state, dict) else []
    observed = merge_observed_models(
        trace_summary.get("observed_models") if isinstance(trace_summary, dict) else [],
        log_models,
        summary.get("observed_models") if isinstance(summary, dict) else [],
    )
    if observed:
        summary = dict(summary) if summary else {"source": "observed"}
        summary["observed_models"] = observed
    return summary

def _load_llm_config_from_resolved_config(task_root: Path) -> dict[str, Any]:
    path = task_root / "resolved_config.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    return summarize_llm_config_from_resolved_config(data if isinstance(data, dict) else {})

def _load_parallel_task_state(task_root: Path) -> dict[str, Any]:
    path = task_root / "task_logs" / "state.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

def _load_run_status(task_root: Path, *, final_state: dict[str, Any] | None = None) -> str:
    final_status = _status_from_final_state(final_state or {})
    if final_status in _PARALLEL_FINAL_STATUSES:
        return final_status
    path = task_root / "task_logs" / "lhr_state.json"
    if not path.is_file():
        return final_status
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return final_status
    status = str(data.get("run_status") or data.get("status") or "").strip().lower()
    return status or final_status

def _load_live_run_started_at(task_root: Path) -> float | None:
    state_path = task_root / "task_logs" / "lhr_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        state = {}
    if isinstance(state, dict):
        started_at = _to_float(state.get("started_at") or state.get("run_started_at"))
        if started_at is not None and started_at > 0:
            return started_at

    multi_worker_starts: list[float] = []
    worker_starts: list[float] = []
    for path in _lnr_event_paths(task_root):
        for obj in _iter_jsonl(path):
            event = str(obj.get("event") or obj.get("event_type") or "")
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
            payload_event = str(payload.get("event") or "")
            ts = _to_float(obj.get("timestamp") or obj.get("created_at"))
            if ts is None:
                continue
            if event == "multi_worker_start" or payload_event == "multi_worker_start":
                multi_worker_starts.append(ts)
            elif event == "worker_start" or payload_event == "worker_start":
                worker_starts.append(ts)
    if multi_worker_starts:
        return max(multi_worker_starts)
    if worker_starts:
        return max(worker_starts)
    return None

def _status_from_final_state(final_state: dict[str, Any]) -> str:
    status = str(final_state.get("status") or "").strip().lower()
    error = str(final_state.get("error") or "").strip().lower()
    if status == "completed":
        return "finished"
    if status == "timeout" and error.startswith("exceeded "):
        return "budget_done"
    if status in {"budget_done", "timeout", "failed", "skipped", "stopped_by_user"}:
        return status
    return ""

def _elapsed_from_final_state(final_state: dict[str, Any]) -> float | None:
    value = _to_float(final_state.get("charged_elapsed_sec"))
    if value is None:
        value = _to_float(final_state.get("elapsed_sec"))
    return value if value is not None and value >= 0 else None

def _display_status(*, run_status: str, running_process_count: int, active_jobs: list[dict[str, Any]]) -> str:
    status = str(run_status or "").strip().lower()
    if running_process_count > 0 and status in _PARALLEL_FINAL_STATUSES:
        return "run_resume"
    if status in _TERMINAL_DISPLAY_STATUSES:
        return status
    if status in {"run", "running"} or running_process_count > 0:
        return "run"
    if active_jobs:
        return "tracked"
    return "unknown"

def _lnr_event_paths(task_root: Path) -> list[Path]:
    paths = [task_root / "task_logs" / "lhr_events.jsonl"]
    paths.extend(sorted(task_root.glob("task_logs/workers/w*/lhr_events.jsonl")))
    paths.extend(sorted(task_root.glob("workers/w*/logs/lhr_events.jsonl")))
    return paths

def _elapsed_from_resource_range(
    resource_summary: dict[str, Any],
    *,
    live: bool = False,
    run_started_at: float | None = None,
) -> float | None:
    first = _to_float(run_started_at) if live else None
    if first is None:
        first = _to_float(resource_summary.get("first_event_at"))
    last = _to_float(resource_summary.get("last_event_at"))
    if first is None:
        return None
    end = time.time() if live else last
    if end is None:
        return None
    return max(0.0, end - first)

def _summarize_time_traces(task_root: Path) -> dict[str, Any]:
    tokens_in = tokens_out = tokens_cached = calls = 0
    cost_usd = 0.0
    cost_known_calls = 0
    model_counts: Counter[str] = Counter()
    price_table = _load_llm_prices(task_root)
    max_ttft_sec: float | None = None
    max_tpot_ms: float | None = None
    paths = _time_trace_paths(task_root)
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        for row in _load_csv_rows(path):
            calls += 1
            ti = _first_int(row, ("tokens_input", "input_tokens"))
            to = _first_int(row, ("tokens_output", "output_tokens"))
            tc = _first_int(row, ("tokens_cached", "cached_tokens"))
            tokens_in += ti
            tokens_out += to
            tokens_cached += tc
            detail_s = str(row.get("detail") or "")
            model_name = extract_model_from_trace_detail(detail_s)
            if model_name:
                model_counts[model_name] += 1
            category_s = str(row.get("category") or "")
            cost_scope = category_s == "llm_api" or "token_scope=llm_api" in detail_s or not category_s
            if cost_scope:
                row_cost = estimate_llm_cost_usd(
                    model=extract_model_from_trace_detail(detail_s),
                    tokens_input=ti,
                    tokens_output=to,
                    tokens_cached=tc,
                    price_table=price_table,
                )
                if row_cost is None and "pricing=models.json" in detail_s:
                    row_cost = _first_float(row, ("llm_cost_usd", "cost_usd"))
                if row_cost is not None:
                    cost_usd += float(row_cost)
                    cost_known_calls += 1
            max_ttft_sec = _max_optional(max_ttft_sec, _first_float(row, ("ttft_sec", "ttft")))
            max_tpot_ms = _max_optional(max_tpot_ms, _first_float(row, ("tpot_ms", "tpot")))
    cache_rate = (tokens_cached / tokens_in) if tokens_in > 0 else None
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_cached": tokens_cached,
        "llm_calls": calls,
        "cost_usd": cost_usd if cost_known_calls else None,
        "cost_known_calls": cost_known_calls,
        "cache_rate": cache_rate,
        "observed_models": [name for name, _ in model_counts.most_common()],
        "max_ttft_sec": max_ttft_sec,
        "max_tpot_ms": max_tpot_ms,
    }

__all__ = tuple(name for name in globals() if not name.startswith("__"))
