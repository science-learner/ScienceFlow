"""ParallelRunner responsibility: resume."""

from __future__ import annotations

import json
from typing import Any

from scienceflow.runtime.parallel.config.manifest import _positive_int_or_none
from scienceflow.runtime.parallel.config.models import TaskSpec
from scienceflow.runtime.parallel.config.preparation import (
    _EXTERNAL_LLM_FAILURE_KINDS,
    _parallel_state_file_for_read,
    _state_completed_but_child_failed,
    _workspace_lnr_failure_kind,
)


def _check_resume(self, spec: TaskSpec) -> int | None:
    """Return remaining seconds, or None if no prior state."""
    state_file = _parallel_state_file_for_read(spec.workspace, task_type=spec.type)
    if state_file is None:
        return None
    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    child_failed = _state_completed_but_child_failed(spec.workspace)
    if state.get("status") == "completed" and not child_failed:
        stop_reason = str(state.get("stop_reason") or state.get("failure_kind") or "").strip()
        if stop_reason not in {"budget_expired", "time_budget_expired"}:
            return 0

    if getattr(self, "_resume_budget_policy", "remaining") == "fresh":
        return int(spec.time_limit)

    if child_failed and _workspace_lnr_failure_kind(spec.workspace) in _EXTERNAL_LLM_FAILURE_KINDS:
        charged_elapsed = 0
    else:
        charged_elapsed = state.get("charged_elapsed_sec", state.get("elapsed_sec", 0))
    return max(0, self._logical_resume_budget_sec(spec) - int(float(charged_elapsed or 0)))


@staticmethod
def _logical_resume_budget_sec(spec: TaskSpec) -> int:
    """Return the user-facing budget that resume should spend down.

    Parallel manifests often set ``time_limit`` slightly above the LNR wall
    clock to give the child process shutdown headroom. Resume accounting
    should spend down the LNR budget, not that outer cushion.
    """
    budget = _positive_int_or_none(spec.time_limit) or 0
    if spec.phase == "run" and spec.type == "lnr" and isinstance(spec.lnr_patch, dict):
        lnr_budget = _positive_int_or_none(spec.lnr_patch.get("wall_clock_budget_sec"))
        if lnr_budget is not None:
            return min(budget, lnr_budget) if budget > 0 else lnr_budget
    return budget


@staticmethod
def _with_resume_remaining_budget(spec: TaskSpec, remaining_sec: int) -> TaskSpec:
    remaining = max(0, int(remaining_sec))
    updates: dict[str, Any] = {"time_limit": remaining}
    if spec.phase == "run" and spec.type == "lnr":
        lnr_patch = dict(spec.lnr_patch or {})
        if "wall_clock_budget_sec" in lnr_patch:
            lnr_patch["wall_clock_budget_sec"] = remaining
        updates["lnr_patch"] = lnr_patch or None
    return TaskSpec(**{**spec.__dict__, **updates})


def _write_interrupted_states(self, specs: list[TaskSpec], *, error: str) -> None:
    for spec in specs:
        self._write_interrupted_state(spec, error=error)
