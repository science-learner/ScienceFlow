# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Stage-capture feedback delivery and duplicate-submission governance.

Covers the LNR feedback-loop fixes:
- ``single.py``: stage-callback output is injected into conversation memory
  (the REPL discards the run() return value, so memory is the only channel);
- ``solver.py``: the duplicate-rejection counter survives ``__init__``-bypass
  construction and is restated in the estra state packet; the REPL logs an
  auditable ``repl_agent_run_returned`` event for what crossed agent.run().
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepcraft_core import Memory, Message
from deepcraft_core.tool import ToolCall, ToolResult
from deepcraft_core.tool.base import Function

from scienceflow.core.agent.tool_exec.single import SingleToolExecMixin
from scienceflow.solver.lnr.snapshot_store import StageSnapshot
from scienceflow.solver.lnr.state_machine import LHRStateMachineStore
from scienceflow.solver.lnr.solver import LnrSolver
from tests.test_long_horizon_repl import _lhr_test_ledger, _minimal_lhr_solver


def _bash_call(command: str = "python solution.py", call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=Function(name="bash", arguments=json.dumps({"command": command})),
    )


# ---------------------------------------------------------------------------
# single.py: stage-capture feedback reaches conversation memory
# ---------------------------------------------------------------------------


class _CaptureDummy(SingleToolExecMixin):
    """Minimal host for SingleToolExecMixin._run_one_tool_after_assistant_logged.

    Only the attributes the bash success path touches are provided; everything
    unrelated is stubbed to a no-op so the test exercises the real control flow
    up to (and including) the stage-capture callback branch.
    """

    def __init__(self, *, capture_out: str | None) -> None:
        self.memory = Memory(max_messages=50)
        self._workspace_dir = Path(".")
        self._embedded_full_run_enabled = False
        self._interaction_log_policy = SimpleNamespace(
            tool_call_full=False,
            llm_stream_to_file=False,
            bash_preview_max_chars=None,
            tool_result_max_lines=None,
            write_edit_tool_result_verbose=False,
            tool_result_unlimited=False,
            bash_output_dedup_enabled=False,
            bash_output_dedup_min_repeat=3,
            bash_output_dedup_summary_prefix="[log-dedup]",
        )
        self._scienceflow_stdout_max_chars = 0
        self._bash_stream_to_interaction_log = False
        self._ws_interaction_log = None
        self._ui = None
        self._guard_manager = None
        self._consecutive_write_syntax_fails = 0
        self._consecutive_infra_errors = 0
        self._debug_dynamic_steps_enabled = False
        self._debug_step_boost = 0
        self._debug_max_steps_cap = 0
        self._effective_max_steps = 10
        self._file_snapshot_latest_only = False
        self._lnr_snapshot_ok = True
        self._lnr_snapshot_reason = "gate_ok_tail_snapshot"
        self._lnr_stage_capture_on_candidate_artifact = True
        self._lnr_candidate_artifact_rel = ""
        self._lnr_force_stage_journal_after_run = False
        self._lnr_stage_commit_enabled = False
        self._lnr_stop_after_bare_solution_success = False
        self._run_policy = SimpleNamespace(
            expand_round_budget_after_tool_error=lambda **kw: kw["effective_max"]
        )
        self.classify_tool_error_for_budget = lambda *_a, **_k: "other"  # type: ignore[assignment]
        self._capture_out = capture_out
        self.capture_calls = 0
        self.synced = 0

        async def _capture(*, agent: object, args: dict, tool_result: object) -> str | None:
            self.capture_calls += 1
            return self._capture_out

        self._lnr_stage_capture_callback = _capture

        async def _execute(*, name: str, tool_input: dict, on_output: object = None) -> ToolResult:
            return ToolResult(output="ran ok")

        self.availableTools = SimpleNamespace(execute=_execute)  # type: ignore[assignment]

    def _sync_last_run_token_totals(self) -> None:
        self.synced += 1

    def _log_info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def _pop_and_log_thought(self, args: dict) -> None:
        return None

    def _maybe_normalize_tool_input_paths(self, name: str, args: dict) -> dict:
        return args

    def _maybe_rewrite_tool_result_paths(self, tool_result: ToolResult) -> ToolResult:
        return tool_result

    def _prepare_tool_feedback_for_memory(
        self, name: str, args: dict, tool_result: ToolResult, *, guard_coaching: str = ""
    ) -> str:
        return str(tool_result.output or "")

    def _next_step(self) -> int:
        return 1

    async def _maybe_embedded_full_run_after_quick_test(
        self, args: dict, tool_result: ToolResult
    ) -> str | None:
        return None

    async def _maybe_write_bare_run_tail_snapshot(
        self, args: dict, tool_result: ToolResult
    ) -> None:
        return None


async def _run_bash_tool(dummy: _CaptureDummy) -> tuple[str | None, int | None]:
    tc = _bash_call()
    assistant = Message(role="assistant", content="", tool_calls=[tc])
    return await dummy._run_one_tool_after_assistant_logged(
        tc, assistant, effective_max=10, initial_max=10,
    )


@pytest.mark.asyncio
async def test_stage_capture_feedback_injected_into_memory(
    tmp_path: Path,
) -> None:
    """Non-empty callback output must land in conversation memory as a user message.

    The REPL discards agent.run()'s return value, so without the injection the
    feedback (e.g. duplicate-candidate rejection) never reaches the agent.
    """
    dummy = _CaptureDummy(capture_out="SUBMISSION_FEEDBACK: duplicate candidate rejected")
    out, _ = await _run_bash_tool(dummy)

    assert out is not None
    assert "duplicate candidate rejected" in out
    assert dummy.capture_calls == 1
    assert dummy.synced == 1
    stored = [
        m for m in dummy.memory.chat_history_memory.retrieve(window_size=None)
        if getattr(m.memory_record.message, "role", "") == "user"
    ]
    assert any("duplicate candidate rejected" in str(m.memory_record.message.content) for m in stored)


@pytest.mark.asyncio
async def test_stage_capture_silent_path_adds_no_memory_message(
    tmp_path: Path,
) -> None:
    """A silent callback (None) must not inject anything nor early-out."""
    dummy = _CaptureDummy(capture_out=None)
    out, _ = await _run_bash_tool(dummy)

    assert out is None
    assert dummy.capture_calls == 1
    assert dummy.synced == 0
    user_msgs = [
        m for m in dummy.memory.chat_history_memory.retrieve(window_size=None)
        if getattr(m.memory_record.message, "role", "") == "user"
    ]
    assert user_msgs == []


# ---------------------------------------------------------------------------
# solver.py: duplicate-rejection counter + state-packet fact line
# ---------------------------------------------------------------------------


def _duplicate_solver(tmp_path: Path) -> LnrSolver:
    """Minimal solver with a committed S01 whose artifact hash matches."""
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True, exist_ok=True)
    solver.lhr = SimpleNamespace(
        stage_capture_enabled=True,
        stage_commit_min_seconds_between=0,
    )
    solver.last_stage_commit_ts = 0.0
    solver._metric_event_from_workspace = lambda: None
    solver._evaluator_stage_source_mode = lambda: "primary"
    solver._candidate_artifact_sha_from_workspace = lambda _event: "same-artifact"
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            "S01",
            "snapshot-one",
            tmp_path / "snapshot-one",
            0.42,
            "score",
            False,
            1,
            {"artifact_sha": "same-artifact", "gate_accepted": True},
        )
    }

    def should_not_evaluate(**_kwargs):  # pragma: no cover - must not run
        raise AssertionError("an unchanged committed artifact must not re-enter Gate")

    solver._record_evaluator_stage_events = should_not_evaluate
    return solver


def test_duplicate_rejection_counts_survive_init_bypass(tmp_path: Path) -> None:
    """The counter must increment even when __init__ was bypassed (object.__new__).

    _minimal_lhr_solver builds the solver via object.__new__, so the increment
    site cannot assume __init__ ran; a plain `+=` would raise AttributeError.
    """
    async def _run() -> None:
        solver = _duplicate_solver(tmp_path)
        assert not hasattr(solver, "_duplicate_rejection_count")

        for _ in range(3):
            out = await LnrSolver._stage_capture_callback(
                solver,
                agent=object(),
                args={},
                tool_result=ToolResult(output="ok"),
            )
            assert out is None  # silent skip toward the agent

        assert solver._duplicate_rejection_count == 3
        assert solver._last_duplicate_of_stage == "S01"

    asyncio.run(_run())


def test_state_packet_restates_duplicate_count(tmp_path: Path) -> None:
    """The estra state packet carries the deterministic duplicate-count fact line."""
    solver = _minimal_lhr_solver(tmp_path)
    solver.stage_snapshots = {}
    solver.lhr = SimpleNamespace(
        state_packet_max_chars=12000,
        state_packet_stage_card_max_chars=420,
        state_packet_archived_branch_max_chars=1500,
    )

    # Zero duplicates: no character cost.
    packet = LnrSolver._build_lhr_state_packet(
        solver, action="keep_current", target_stage="S01",
        terminal_stage="S01", reason="compact",
    )
    assert "duplicate attempts rejected" not in packet

    solver._duplicate_rejection_count = 7
    packet = LnrSolver._build_lhr_state_packet(
        solver, action="keep_current", target_stage="S01",
        terminal_stage="S01", reason="compact",
    )
    assert "Submissions: 7 duplicate attempts rejected" in packet
    assert "re-evaluation will not produce a new version" in packet


# ---------------------------------------------------------------------------
# solver.py: auditable repl_agent_run_returned event
# ---------------------------------------------------------------------------


def test_repl_logs_agent_run_return_event(tmp_path: Path) -> None:
    """_run_single must emit repl_agent_run_returned with source/early_out/preview."""
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        wall_clock_budget_sec=3600,
        max_steps=5,
        seed=0,
        num_workers=1,
        merge_enabled=False,
    )
    solver.task_desc = "task"
    solver.initial_workspace_state = ""
    solver.memory_dir = tmp_path / "memory"
    solver.deadline = float("inf")
    solver.evaluator_stop_requested = False
    solver.evaluator_stop_reason = ""
    solver.pending_estra = False
    solver._pending_stage_commit_text_active = lambda: False
    solver._parallel_worker_snapshot_for_prompt = lambda: None
    solver._resource_context_for_prompt = lambda: None
    solver._lnr_skill_hint = lambda: ""
    solver._evaluator_prompt_contract = lambda: ""
    solver._evaluator_task_profile = lambda: ""
    solver._task_runtime_prompt_contract = lambda: ""
    solver._prepare_workspace = lambda: None
    solver._load_existing_stage_snapshots = lambda: None
    solver.snapshot_store = SimpleNamespace(initialize_baseline=lambda: True)
    solver.state_machine = SimpleNamespace(
        mark_run_status=lambda status, payload=None: None
    )
    solver._result = lambda **kw: {"status": "ok", **kw}
    solver._abandon_pending_stage_commit_text = lambda agent: None
    solver._accumulate_main_run_tokens = lambda agent: None
    events: list[tuple[str, dict]] = []
    solver._jsonl = lambda name, record: events.append((name, dict(record)))

    class _FakeAgent:
        llm = object()

        async def run(self, request: str | None) -> str:
            return "duplicate candidate rejected; hash unchanged"

    solver._make_agent = lambda load_existing_memory: _FakeAgent()  # type: ignore[assignment]

    async def _run() -> dict:
        result = await LnrSolver._run_single(solver, keep_agent_open=False)
        return result

    # One loop iteration then stop: flip the deadline after the first run().
    original_run = _FakeAgent.run

    calls = {"n": 0}

    async def _run_once(self: _FakeAgent, request: str | None) -> str:  # noqa: ANN001
        calls["n"] += 1
        solver.deadline = 0.0  # end the REPL loop after this round
        return await original_run(self, request)

    _FakeAgent.run = _run_once  # type: ignore[assignment]

    result = asyncio.run(_run())
    assert result["status"] == "ok"

    repl_events = [
        (name, rec) for name, rec in events
        if name == "lhr_repl_events.jsonl"
        and rec.get("event") == "repl_agent_run_returned"
    ]
    assert len(repl_events) == 1
    _name, rec = repl_events[0]
    assert rec["source"] == "agent_run"
    assert rec["early_out"] is True
    assert "duplicate candidate rejected" in rec["preview"]
