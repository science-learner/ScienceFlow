"""Shared fixtures and helpers for split long_horizon_repl contracts."""

from __future__ import annotations

import asyncio

import csv

import hashlib

import json

from pathlib import Path

from types import MethodType, SimpleNamespace

from unittest.mock import MagicMock

import pytest

from inquirycraft.memory import Message

from inquirycraft.tools import ToolResult

from scienceflow.research.quality.assessment import CandidateAssessmentService

from scienceflow.agent.core.runtime.agent import ScienceAgent

from scienceflow.research.state.knowledge.memory.agent.resource_feedback_memory import (
    RESOURCE_STATE_SUMMARY_MARKER,
    ResourceFeedbackMemoryDeduper,
)

from scienceflow.agent.policies import routing


class AgentRoutingTestSurface:
    _build_lnr_stage_commit_compact_messages = (
        routing._build_lnr_stage_commit_compact_messages
    )


async def _fake_ephemeral_route_prompt(
    self,
    prompt,
    *,
    trigger,
    base_messages=None,
    system_messages=None,
    timeout=None,
    llm_override=None,
    llm_role=None,
    stage_id=None,
    lineage_id=None,
    node_uid=None,
):
    del trigger, llm_role, stage_id, lineage_id, node_uid
    backend = llm_override or self.llm
    messages = [*(base_messages or []), Message.user_message(prompt)]
    if hasattr(backend, "ask_tool_stream") and (
        base_messages or not hasattr(backend, "ask")
    ):
        response = await backend.ask_tool_stream(
            messages=messages,
            system_msgs=list(system_messages or []),
            timeout=timeout,
            tools=[],
            tool_choice="none",
            parallel_tool_calls=False,
            collect_all_tool_calls=True,
        )
        return str(getattr(response, "content", None) or "") or str(
            getattr(response, "reasoning_content", None) or ""
        )
    return await backend.ask(
        messages=messages,
        system_msgs=list(system_messages or []),
        stream=False,
        timeout=timeout,
    )


from scienceflow.research.solver.lnr.orchestration import agent_hooks


class LNRHooksTestSurface:
    _lnr_maybe_periodic_inject_at_round_start = (
        agent_hooks._lnr_maybe_periodic_inject_at_round_start
    )


from scienceflow.runtime.core.process.commands import (
    looks_like_bare_solution_run,
    python_script_run_rel_path,
)

from scienceflow.runtime.safety.tooling.resource_management.resource_policy import (
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
)

from scienceflow.runtime.safety.policy.execution_policy import (
    EnsureFullRunResult,
    write_fullrun_tail_snapshot,
)

from scienceflow.research.solver.lnr.resources.resource_observer import LHRResourceObserver

from scienceflow.research.solver.lnr.support.prompt_template_store import load_prompt_template

from scienceflow.research.solver.lnr.support.prompts import (
    KEEP_CURRENT_COMPACT_TEMPLATE,
    ML_FIRST_USER_TEMPLATE,
    ESTRA_DECISION_TEMPLATE,
    ESTRA_RESUME_TEMPLATE,
    build_first_user_prompt,
    build_keep_current_compact_prompt,
    build_estra_prompt,
    build_estra_resume_prompt,
    build_stage_commit_prompt,
)

from scienceflow.research.solver.lnr.lifecycle.snapshots.snapshot_store import SnapshotStore, StageSnapshot

from scienceflow.research.solver.lnr.orchestration import solver as lnr_solver_module

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context import (
    coordination as lnr_context_coordination,
)

from scienceflow.research.solver.lnr.orchestration.coordinator.run import (
    agent_coordination as lnr_agent_coordination,
)

from scienceflow.research.solver.lnr.orchestration.state_machine import LHRStateMachineStore

from scienceflow.research.solver.lnr.orchestration.solver import (
    LHR_STAGE_PERFORMANCE_COLUMNS,
    LnrSolver,
    _append_lnr_main_agent_protocols,
    _effective_lnr_bash_timeout_sec,
    _validation_leakage_reason,
)

from scienceflow.research.solver.lnr.lifecycle.stage.records.stage_ledger import (
    append_archived_trajectory_summary,
    append_estra_summary,
    append_stage_event_summary,
    next_stage_id,
    parse_stage_cards,
    render_stage_cards,
    salvage_append_only_stage_commit,
    tail_summary_after,
    tail_summary_from_stage,
    validate_append_only_stage_commit,
    validate_stage_card,
)

from scienceflow.foundation.contracts import EstraContext

from scienceflow.research.control.estra import EstraService

from scienceflow.research.control.estra.planning.parser import parse_estra_decision

from scienceflow.research.state.knowledge.memory import MemoryCompactor


def _make_metric_snapshot_agent(ws: Path):
    agent = ScienceAgent.__new__(ScienceAgent)
    mem = MagicMock()
    mem.add_message = MagicMock()
    object.__setattr__(agent, "_workspace_dir", ws)
    object.__setattr__(agent, "_embedded_full_run_enabled", False)
    object.__setattr__(agent, "_lnr_allow_any_stage_script", False)
    object.__setattr__(agent, "_lnr_mlebench_validate_enabled", True)
    object.__setattr__(agent, "_mlebench_data_dir", None)
    object.__setattr__(agent, "_mlebench_exp_id", None)
    object.__setattr__(agent, "_scienceflow_task_profile", "mlebench")
    object.__setattr__(agent, "_scienceflow_evaluator_backend", "task_package")
    object.__setattr__(agent, "_lnr_candidate_artifact_rel", "submission.csv")
    object.__setattr__(agent, "_lnr_run_control_max_fix_rounds", 5)
    object.__setattr__(agent, "_run_control_user_injections", 0)
    object.__setattr__(agent, "_inject_run_control_user_message", MagicMock())
    object.__setattr__(agent, "_fullrun_output_tail_stdout_lines", 50)
    object.__setattr__(agent, "_fullrun_output_tail_stderr_lines", 0)
    object.__setattr__(agent, "_fullrun_output_tail_max_chars", 4000)
    object.__setattr__(agent, "_submission_history_archive_enabled", False)
    object.__setattr__(agent, "_log_info", MagicMock())
    object.__setattr__(agent, "_log_warning", MagicMock())
    object.__setattr__(agent, "memory", mem)
    return agent


def _lhr_resource_observer_with_gpu_queue(
    tmp_path: Path, worker_id: str
) -> LHRResourceObserver:
    store = LHRStateMachineStore(
        log_dir=tmp_path / worker_id / "logs", worker_id=worker_id
    )
    return LHRResourceObserver(
        state_machine=store,
        worker_id=worker_id,
        min_register_sec=600,
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_queue_max_wait_sec=2,
        gpu_queue_heartbeat_sec=1,
        gpu_max_heavy_per_gpu=1,
        gpu_lease_ttl_sec=600,
    )


def _lhr_test_ledger() -> str:
    return "\n".join(
        [
            "### S01",
            "metric: 0.07",
            "lower_is_better: true",
            "BRIEF: baseline",
            "WHY: start",
            "",
            "### S02",
            "metric: 0.08",
            "lower_is_better: true",
            "BRIEF: worse branch",
            "WHY: test route",
            "",
        ]
    )


class MinimalLhrSolverFixture(LnrSolver):
    """Typed host fixture for contracts that still exercise the façade boundary."""

    def __init__(self, tmp_path: Path) -> None:
        self.ledger_filename = "run_results.md"
        self.ledger_path = tmp_path / "run_results.md"
        self.log_dir = tmp_path / "logs"
        self.root_dir = tmp_path
        self.task_root_dir = tmp_path
        self.workspace_dir = tmp_path / "workspace"
        self.worker_id = ""
        self.worker_index = 0
        self.current_lineage_no = 1
        self.current_lineage_id = "L01"
        self.pending_stage_commit_transaction = None
        self.state_machine = LHRStateMachineStore(
            log_dir=self.log_dir,
            worker_id="W00",
        )
        self.ledger_path.write_text(_lhr_test_ledger(), encoding="utf-8")


def _minimal_lhr_solver(tmp_path: Path) -> LnrSolver:
    return MinimalLhrSolverFixture(tmp_path)


__all__ = tuple(name for name in globals() if not name.startswith("__"))
