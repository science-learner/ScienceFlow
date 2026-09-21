"""Shared fixtures and helpers for split resource_arbiter_control_plane contracts."""

from __future__ import annotations

import asyncio

from dataclasses import replace

import json

import time

from pathlib import Path

from scienceflow.runtime.safety.tooling.bash import _is_gpu_visibility_probe

from scienceflow.research.solver.lnr.resources.resource_advisory import build_inline_resource_advisory_prompt

from scienceflow.research.solver.lnr.resources.resource_feedback_guidance import resource_feedback_guidance_value

from scienceflow.research.solver.lnr.resources.runtime.review.decision.arbiter import (
    build_resource_arbiter_prompt,
    enforce_proposal_action_allowlist,
    enforce_repeated_stall_escalation,
    fallback_policy_decision,
    main_agent_feedback,
    normalize_arbiter_decision,
    proposal_has_severe_stalled_no_work,
)

from scienceflow.research.solver.lnr.resources.runtime.review.decision.arbiter_gate import enforce_arbiter_kill_gate

from scienceflow.research.solver.lnr.resources.runtime.review.decision.execution_facts import build_execution_facts

from scienceflow.research.solver.lnr.resources.runtime.review.evidence.progress import classify_progress_signal

from tests.lnr_resource_test_utils import make_observer

from scienceflow.research.solver.lnr.resources.resource_observer import LHRResourceObserver

def _heavy_job(observer, tmp_path: Path) -> str:
    (tmp_path / "train.py").write_text("import torch\ntorch.cuda.is_available()\nprint(\'train\')\n", encoding="utf-8")
    job_id = observer.job_created(
        command="python3 train.py --device cuda --epochs 10",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=100.0,
        workspace_dir=tmp_path,
    )
    assert job_id is not None
    return job_id

def _resource_events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "resource" / "resource_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _route_value_review_payload(no_useful_windows: int) -> tuple[dict, dict]:
    boundary = {"kind": "route_value", "reason": "no_useful_progress_windows>=2"}
    review_state = {
        "job_state_bucket": "RUNNING_NO_PROGRESS",
        "no_useful_progress_windows": no_useful_windows,
        "blocked_worker_count": 0,
        "active_waiter_pressure": False,
    }
    signal = {
        "elapsed_sec": 300.0,
        "stdout_lines": 5,
        "stdout_bytes": 200,
        "process_tree_cpu": {"total_cpu_pct": 600.0, "busy_child_count": 4},
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
        "metric_history_text": "",
        "metric_history_line_count": 0,
    }
    decision = {
        "enabled": True,
        "terminate": False,
        "would_terminate": True,
        "requires_llm_decision": True,
        "arbiter_enabled": True,
        "reason": "active_intervention:sm_route_value",
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
    }
    return decision, signal

__all__ = tuple(name for name in globals() if not name.startswith("__"))
