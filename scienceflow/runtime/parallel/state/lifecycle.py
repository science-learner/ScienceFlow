"""ParallelRunner responsibility: state."""

from __future__ import annotations

import json
import time
from typing import Any

from scienceflow.runtime.parallel.execution.base import logger
from scienceflow.runtime.parallel.config.manifest import _BUDGET_DONE_STATUS, _load_json_file
from scienceflow.runtime.parallel.config.models import TaskResult, TaskSpec
from scienceflow.runtime.parallel.config.preparation import (
    _EXTERNAL_LLM_FAILURE_KINDS,
    _charged_elapsed_for_resume,
    _ensure_workspace_for_state_write,
    _parallel_failure_kind,
    _read_resolved_gpu_from_assignment,
    _task_state_log_dir,
    _workspace_lnr_done_payload,
)


def _write_interrupted_state(self, spec: TaskSpec, *, error: str) -> None:
    base = _ensure_workspace_for_state_write(spec.workspace)
    if base is None:
        logger.warning(
            f"[{spec.exp_id}] Skipped writing interrupted state.json; fix workspace path: {spec.workspace!r}"
        )
        return
    logs_dir = _task_state_log_dir(spec)
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"[{spec.exp_id}] cannot create logs directory: {e}")
        return
    state_file = logs_dir / "state.json"
    try:
        state = _load_json_file(state_file)
        now = time.time()
        started_at = float(state.get("run_started_at") or 0.0)
        segment_elapsed = max(0.0, now - started_at) if started_at > 0 else 0.0
        prior_charged = float(
            state.get("resume_prior_charged_elapsed_sec")
            if state.get("resume_prior_charged_elapsed_sec") is not None
            else state.get("charged_elapsed_sec")
            or 0.0
        )
        state.update(
            {
                "exp_id": spec.exp_id,
                "run_id": spec.run_id,
                "status": "stopped_by_user",
                "elapsed_sec": round(segment_elapsed, 1),
                "segment_elapsed_sec": round(segment_elapsed, 1),
                "charged_elapsed_sec": prior_charged + segment_elapsed,
                "resume_prior_charged_elapsed_sec": prior_charged,
                "resume_total_budget_sec": prior_charged + float(spec.time_limit or 0),
                "exit_code": None,
                "error": error,
                "time_limit_sec": spec.time_limit,
                "stopped_at": now,
            }
        )
        state_file.write_text(json.dumps(state, indent=2))
    except OSError as e:
        logger.warning(f"[{spec.exp_id}] Failed to write interrupted state: {e}")


def _write_running_state(
    self,
    spec: TaskSpec,
    *,
    resolved_gpu: str = "",
    llm_config: dict[str, Any] | None = None,
    log_file: str = "",
) -> None:
    base = _ensure_workspace_for_state_write(spec.workspace)
    if base is None:
        logger.warning(
            f"[{spec.exp_id}] Skipped writing running state.json; fix workspace path: {spec.workspace!r}"
        )
        return
    logs_dir = _task_state_log_dir(spec)
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"[{spec.exp_id}] cannot create logs directory: {e}")
        return
    state_file = logs_dir / "state.json"
    try:
        state = _load_json_file(state_file)
        prior_charged = float(state.get("charged_elapsed_sec") or 0.0)
        total_budget = prior_charged + float(spec.time_limit or 0)
        state.update(
            {
                "exp_id": spec.exp_id,
                "run_id": spec.run_id,
                "status": "running",
                "elapsed_sec": 0.0,
                "segment_elapsed_sec": 0.0,
                "charged_elapsed_sec": prior_charged,
                "resume_prior_charged_elapsed_sec": prior_charged,
                "resume_total_budget_sec": total_budget,
                "run_started_at": time.time(),
                "exit_code": None,
                "error": "",
                "time_limit_sec": spec.time_limit,
            }
        )
        rg = resolved_gpu or _read_resolved_gpu_from_assignment(logs_dir)
        if rg:
            state["resolved_gpu"] = rg
        if spec.gpu_auto:
            state["gpu_auto"] = True
        raw_gl = (spec.gpu_list_raw or "").strip()
        if raw_gl:
            state["raw_gpu_list"] = raw_gl
        if llm_config:
            state["llm_config"] = llm_config
        for stale_key in ("failure_kind", "resume_retriable", "stop_reason"):
            state.pop(stale_key, None)
        policy = getattr(self, "_resume_budget_policy", "remaining")
        if policy != "remaining":
            state["resume_budget_policy"] = policy
        else:
            state.pop("resume_budget_policy", None)
        if log_file:
            state["log_file"] = str(log_file)
        state_file.write_text(json.dumps(state, indent=2))
    except OSError as e:
        logger.warning(f"[{spec.exp_id}] Failed to write running state: {e}")


def _write_state(self, spec: TaskSpec, result: TaskResult, *, resolved_gpu: str = "") -> None:
    base = _ensure_workspace_for_state_write(spec.workspace)
    if base is None:
        logger.warning(
            f"[{spec.exp_id}] Skipped writing state.json; fix workspace path: {spec.workspace!r}"
        )
        return
    logs_dir = _task_state_log_dir(spec)
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"[{spec.exp_id}] cannot create logs directory: {e}")
        return
    state_file = logs_dir / "state.json"
    try:
        existing_state = _load_json_file(state_file)
        rg = resolved_gpu or _read_resolved_gpu_from_assignment(logs_dir)
        lnr_done_payload = (
            _workspace_lnr_done_payload(spec.workspace) if spec.type == "lnr" else {}
        )
        structured_failure_kind = str(
            lnr_done_payload.get("failure_kind") or ""
        ).strip().lower()
        failure_kind = (
            structured_failure_kind
            if result.status in {"failed", "error"} and structured_failure_kind
            else _parallel_failure_kind(result)
        )
        lnr_stop_reason = str(lnr_done_payload.get("stop_reason") or "").strip()
        prior_charged = float(existing_state.get("resume_prior_charged_elapsed_sec") or 0.0)
        segment_charged_elapsed_sec = _charged_elapsed_for_resume(result, failure_kind)
        charged_elapsed_sec = prior_charged + segment_charged_elapsed_sec
        state = {
            "exp_id": spec.exp_id,
            "run_id": spec.run_id,
            "status": "completed" if result.status == "success" else result.status,
            "elapsed_sec": result.elapsed_sec,
            "segment_elapsed_sec": result.elapsed_sec,
            "charged_elapsed_sec": charged_elapsed_sec,
            "resume_prior_charged_elapsed_sec": prior_charged,
            "resume_total_budget_sec": prior_charged + float(spec.time_limit or 0),
            "exit_code": result.exit_code,
            "error": result.error,
            "time_limit_sec": spec.time_limit,
        }
        llm_config = existing_state.get("llm_config")
        if isinstance(llm_config, dict) and llm_config:
            state["llm_config"] = llm_config
        log_file = result.log_file or str(existing_state.get("log_file") or "")
        if log_file:
            state["log_file"] = log_file
        policy = getattr(self, "_resume_budget_policy", "remaining")
        if policy != "remaining":
            state["resume_budget_policy"] = policy
        if result.status == "success" and lnr_stop_reason:
            state["stop_reason"] = lnr_stop_reason
        elif failure_kind == "time_budget_expired" and result.status == _BUDGET_DONE_STATUS:
            state["stop_reason"] = failure_kind
        elif failure_kind:
            state["failure_kind"] = failure_kind
            state["resume_retriable"] = result.status in {"failed", "error"}
            if failure_kind in _EXTERNAL_LLM_FAILURE_KINDS:
                state["resume_budget_policy"] = "external_llm_failure_not_charged"
        if rg:
            state["resolved_gpu"] = rg
        if spec.gpu_auto:
            state["gpu_auto"] = True
        raw_gl = (spec.gpu_list_raw or "").strip()
        if raw_gl:
            state["raw_gpu_list"] = raw_gl
        if result.status in {"failed", "error"} and lnr_done_payload:
            merge_status = str(lnr_done_payload.get("merge_status") or "").strip()
            worker_error_kinds = lnr_done_payload.get("worker_error_kinds")
            if merge_status:
                state["merge_status"] = merge_status
            if isinstance(worker_error_kinds, list) and worker_error_kinds:
                state["worker_error_kinds"] = [
                    str(kind) for kind in worker_error_kinds if str(kind).strip()
                ]
        state_file.write_text(json.dumps(state, indent=2))
    except OSError as e:
        logger.warning(f"[{spec.exp_id}] Failed to write state: {e}")
