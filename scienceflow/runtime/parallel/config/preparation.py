"""Parallel runner responsibility: preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scienceflow.runtime.parallel.execution.base import logger

if TYPE_CHECKING:
    from scienceflow.runtime.parallel.config.models import TaskResult, TaskSpec

def _task_state_log_dir(spec: "TaskSpec") -> Path:
    """Return the root-level state log directory for a parallel task."""
    root = Path(spec.workspace).expanduser()
    return root / "task_logs" if str(spec.type or "").strip().lower() == "lnr" else root / "logs"

def _parallel_state_file_candidates(task_root: str, *, task_type: str = "") -> list[Path]:
    """Return state.json candidates for the current layout policy."""
    root = Path(task_root).expanduser()
    if str(task_type or "").strip().lower() == "lnr":
        return [root / "task_logs" / "state.json"]
    return [
        root / "logs" / "state.json",
        root / "workspace" / "state.json",
        root / "state.json",
    ]

def _parallel_state_file_for_read(task_root: str, *, task_type: str = "") -> Path | None:
    for p in _parallel_state_file_candidates(task_root, task_type=task_type):
        if p.is_file():
            return p
    return None

def _ensure_workspace_for_state_write(workspace: str) -> Path | None:
    """Return directory where state.json may be written, or None if unsafe/unwritable.

    ``mkdir(parents=True)`` on paths like ``/path/to/ws`` tries to create ``/path`` first,
    which typically raises PermissionError. Only create missing segments when some
    ancestor already exists below the filesystem root.
    """
    ws = Path(workspace).expanduser()
    if ws.exists():
        return ws if ws.is_dir() else None
    target = ws.resolve(strict=False)
    p = target
    while not p.exists():
        if p.parent == p:
            return None
        if p.parent == Path("/"):
            logger.warning(
                "workspace %r is missing prefix %s under /; refusing to mkdir "
                "(use an existing directory you own)",
                workspace,
                p,
            )
            return None
        p = p.parent
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("cannot create workspace %r: %s", workspace, e)
        return None
    return target

def _write_gpu_assignment_json(spec: TaskSpec, resolved_gpu: str) -> None:
    """Persist resolved CUDA device IDs under the task state log directory."""
    base = _ensure_workspace_for_state_write(spec.workspace)
    if base is None:
        return
    logs_dir = _task_state_log_dir(spec)
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("[%s] cannot create logs for gpu_assignment.json: %s", spec.exp_id, e)
        return
    raw = (spec.gpu_list_raw or spec.gpu_list or "").strip()
    payload: dict[str, Any] = {
        "resolved_gpu": resolved_gpu if resolved_gpu else None,
        "auto": bool(spec.gpu_auto),
        "raw_gpu_list": raw if raw else None,
    }
    try:
        (logs_dir / "gpu_assignment.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("[%s] Failed to write gpu_assignment.json: %s", spec.exp_id, e)

def _read_resolved_gpu_from_assignment(logs_dir: Path) -> str:
    """Best-effort read of ``resolved_gpu`` from ``gpu_assignment.json``."""
    p = logs_dir / "gpu_assignment.json"
    if not p.is_file():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    r = data.get("resolved_gpu")
    return str(r).strip() if r is not None else ""

_EXTERNAL_LLM_FAILURE_KINDS = {"llm_quota_error", "llm_api_error"}

def _parallel_failure_kind_from_text(text: str) -> str:
    lower = str(text or "").lower()
    if not lower.strip():
        return ""
    if (
        "llm_quota_error" in lower
        or "insufficient balance" in lower
        or "pre_consume_token_quota_failed" in lower
        or "token quota is not enough" in lower
        or "permissiondeniederror" in lower
        or "error code: 402" in lower
        or "error code: 403" in lower
    ):
        return "llm_quota_error"
    if (
        "context_compact_failed" in lower
        or "compact did not fit context" in lower
        or "omitted-history main-agent request" in lower
    ):
        return "context_compact_failed"
    if (
        "llm_api_error" in lower
        or "apistatuserror" in lower
        or "authenticationerror" in lower
        or "badrequesterror" in lower
        or "reasoning_content" in lower
        or "invalid_request_error" in lower
        or "error code: 400" in lower
        or "error code: 401" in lower
    ):
        return "llm_api_error"
    if "time budget exhausted" in lower:
        return "time_budget_expired"
    if "timeout" in lower:
        return "worker_timeout"
    return ""

def _read_log_tail(path: str | Path, *, max_bytes: int = 200_000) -> str:
    try:
        p = Path(path)
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""

def _parallel_failure_kind(result: "TaskResult") -> str:
    if result.status == "success":
        return ""
    direct = _parallel_failure_kind_from_text(result.error)
    if direct:
        return direct
    return _parallel_failure_kind_from_text(_read_log_tail(result.log_file))

def _charged_elapsed_for_resume(result: "TaskResult", failure_kind: str) -> float:
    if result.status in {"failed", "error"} and failure_kind in _EXTERNAL_LLM_FAILURE_KINDS:
        return 0.0
    return float(result.elapsed_sec or 0.0)

def _workspace_lnr_event_files(workspace: str) -> list[Path]:
    task_logs = Path(workspace).expanduser() / "task_logs"
    return [
        task_logs / "lhr_events.jsonl",
        task_logs / "lhr_coordinator_events.jsonl",
    ]

def _workspace_lnr_done_payload(workspace: str) -> dict[str, Any]:
    for events_path in _workspace_lnr_event_files(workspace):
        try:
            lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines[-500:]):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "multi_worker_done":
                continue
            payload = event.get("payload")
            if isinstance(payload, dict):
                return payload
            return event if isinstance(event, dict) else {}
    return {}

def _workspace_lnr_result_status(workspace: str) -> str:
    payload = _workspace_lnr_done_payload(workspace)
    return str(payload.get("status") or "").strip().lower()

def _workspace_lnr_failure_kind(workspace: str) -> str:
    payload = _workspace_lnr_done_payload(workspace)
    if not payload:
        return ""
    # ``multi_worker_done.failure_kind`` is the coordinator's final verdict.
    # Keep it ahead of heuristic log parsing so an incidental tool timeout in a
    # worker log cannot relabel a structured ``worker_error`` as a task timeout.
    structured = str(payload.get("failure_kind") or "").strip().lower()
    if structured:
        return structured
    for key in ("worker_error_kinds", "stop_reason"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                kind = _parallel_failure_kind_from_text(str(item))
                if kind:
                    return kind
        else:
            kind = _parallel_failure_kind_from_text(str(value or ""))
            if kind:
                return kind
    for result in payload.get("worker_results") or []:
        if isinstance(result, dict):
            kind = _parallel_failure_kind_from_text(str(result.get("error") or ""))
            if kind:
                return kind
    return ""

def _state_completed_but_child_failed(workspace: str) -> bool:
    status = _workspace_lnr_result_status(workspace)
    return bool(status and status not in {"success", "completed"})

__all__ = tuple(name for name in globals() if not name.startswith("__"))
