# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""ScienceAgent responsibility: file state, valid-run tracking, policy injection, and productivity."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import PurePosixPath

from inquirycraft.memory import Message

_logger = logging.getLogger("scienceflow")


@property
def file_state_summary(self) -> str:
    s = self._memory_ctx.file_state_summary
    return s if s else "(no files tracked)"


def pin_message(self, msg: Message) -> None:
    """Pin an L1 user message so it is always included before the sliding window."""
    self._memory_ctx.pin_message(msg)


def _sha256_of_solution(self) -> str | None:
    """Hex digest of workspace ``solution.py`` if present; else None."""
    p = self._workspace_dir / "solution.py"
    if not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()
    except OSError:
        return None


def _sha256_of_workspace_file(self, rel_path: str) -> str | None:
    p = self._workspace_dir / rel_path
    if not p.is_file():
        return None
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _normalize_valid_run_source_rel_path(self, rel_path: str | None = None) -> str:
    raw = str(rel_path or "solution.py").replace("\\", "/").strip().lstrip("/")
    if not raw:
        return "solution.py"
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        return "solution.py"
    return str(path)


def _lnr_mark_valid_bare_run(
    self, *, bash_cmd: str = "", solution_rel_path: str | None = None
) -> None:
    """Bind the latest run-control-ready run to the current source/submission files."""
    source_rel = self._normalize_valid_run_source_rel_path(solution_rel_path)
    self._lnr_last_valid_solution_sha = self._sha256_of_workspace_file(source_rel) or ""
    self._lnr_last_valid_solution_rel_path = source_rel
    self._lnr_last_valid_submission_sha = (
        self._sha256_of_workspace_file("submission.csv") or ""
    )
    self._lnr_last_valid_bash_cmd = str(bash_cmd or "")
    self._lnr_snapshot_ok = True
    self._lnr_last_valid_bare_run_is_local = True


def _lnr_restore_valid_bare_run_from_snapshot(self) -> bool:
    """Restore a clone-carried run-control-ready run when file hashes still match."""
    from scienceflow.runtime.core.support.node_paths import find_node_log_path

    p = find_node_log_path(self._workspace_dir, "fullrun_tail_snapshot.json")
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("exit_code") not in (0, "0"):
        return False
    if data.get("validation_ok") is False:
        return False
    if str(data.get("submission_status") or "") == "pending_evaluator":
        return False
    try:
        metric = float(data.get("metric_value"))
    except (TypeError, ValueError):
        return False
    if metric != metric or metric in (float("inf"), float("-inf")):
        return False
    expected_solution = str(data.get("solution_sha") or "").strip()
    if not expected_solution:
        return False
    source_rel = self._normalize_valid_run_source_rel_path(
        data.get("solution_path") or "solution.py"
    )
    current_solution = self._sha256_of_workspace_file(source_rel) or ""
    if current_solution != expected_solution:
        return False
    expected_submission = str(data.get("submission_sha") or "").strip()
    current_submission = self._sha256_of_workspace_file("submission.csv") or ""
    if expected_submission and current_submission != expected_submission:
        return False
    self._lnr_last_valid_solution_sha = current_solution
    self._lnr_last_valid_solution_rel_path = source_rel
    self._lnr_last_valid_submission_sha = current_submission
    self._lnr_last_valid_bash_cmd = str(data.get("bash_cmd") or "")
    self._lnr_snapshot_ok = True
    self._lnr_snapshot_reason = "restored_gate_tail_snapshot"
    self._lnr_last_valid_bare_run_is_local = False
    return True


def _lnr_has_current_valid_bare_run(self) -> bool:
    """True only if current files still match the last run-control-ready bare run."""
    if not bool(getattr(self, "_lnr_snapshot_ok", False)) or not str(
        getattr(self, "_lnr_last_valid_solution_sha", "") or ""
    ):
        self._lnr_restore_valid_bare_run_from_snapshot()
    if not bool(getattr(self, "_lnr_snapshot_ok", False)):
        return False
    expected_solution = str(getattr(self, "_lnr_last_valid_solution_sha", "") or "")
    # Backward-compatible for tests/legacy agents constructed before this stamp existed.
    if not hasattr(self, "_lnr_last_valid_solution_sha"):
        return True
    if not expected_solution:
        return False
    source_rel = self._normalize_valid_run_source_rel_path(
        getattr(self, "_lnr_last_valid_solution_rel_path", "solution.py"),
    )
    if (self._sha256_of_workspace_file(source_rel) or "") != expected_solution:
        return False
    expected_submission = str(getattr(self, "_lnr_last_valid_submission_sha", "") or "")
    if expected_submission and (
        (self._sha256_of_workspace_file("submission.csv") or "") != expected_submission
    ):
        return False
    return True


def _lnr_invalidate_current_valid_run(self, reason: str) -> None:
    self._lnr_snapshot_ok = False
    self._lnr_snapshot_reason = reason
    self._lnr_last_valid_solution_sha = ""
    self._lnr_last_valid_solution_rel_path = "solution.py"
    self._lnr_last_valid_submission_sha = ""
    self._lnr_last_valid_bash_cmd = ""
    self._lnr_last_valid_bare_run_is_local = False
    self._lnr_stage_journal_pending = False
    self._lnr_result_md_after_success_pending = False
    result_md = self._workspace_dir / "result.md"
    if result_md.exists():
        try:
            result_md.unlink()
            _logger.info("[run-control] removed stale result.md after %s", reason)
        except OSError as exc:
            _logger.warning(
                "[run-control] failed to remove stale result.md after %s: %s",
                reason,
                exc,
            )


def _lnr_adjust_policy_injection(self, msg: str) -> str:
    if (
        bool(getattr(self, "_lnr_stop_after_bare_solution_success", False))
        and "result.md does not exist yet" in (msg or "")
        and not self._lnr_has_current_valid_bare_run()
    ):
        return (
            "result.md does not exist yet, but the current `solution.py` has no "
            "matching accepted full run in this node. First modify `solution.py` "
            "if needed, then run `python3 solution.py` with the full-training path "
            "active. Only after run-control checks pass should you write result.md. "
            "Use a tool NOW."
        )
    return msg


def _embedded_full_run_metric_token(self) -> str | None:
    """Stable token for successful embedded full-run metric (if available)."""
    from scienceflow.runtime.core.support.node_paths import find_node_log_path

    p = find_node_log_path(self._workspace_dir, "embedded_full_run_result.json")
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    mv = payload.get("metric_value")
    if mv is None:
        return None
    mn = str(payload.get("metric_name") or "").strip() or "metric"
    return f"{mn}:{mv}"


def _write_productivity_snapshot(self) -> None:
    """Persist per-run productivity counters for solver-side node triage."""
    guard_mgr = getattr(self, "_guard_manager", None)
    if guard_mgr is None:
        return
    counts = guard_mgr.productivity_counters()
    write_n = int(counts.get("write_success_count", 0))
    edit_n = int(counts.get("edit_success_count", 0))
    payload = {
        "write_count": write_n,
        "edit_count": edit_n,
        "write_success_count": write_n,
        "edit_success_count": edit_n,
        "initial_solution_sha": self._initial_solution_sha,
        "final_solution_sha": self._sha256_of_solution(),
        "had_metric": self._embedded_full_run_metric_token() is not None,
    }
    out = self._workspace_dir / ".agent_memory" / "productivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
