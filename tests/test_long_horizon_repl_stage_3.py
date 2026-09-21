"""Long Horizon Repl contracts: stage."""

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.coordination import (
    _parse_stage_commit_text_block,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.stage.prompt import (
    _parse_stage_commit_json_block,
)


def test_lhr_hidden_ledger_is_omitted_from_main_tool_feedback() -> None:
    dummy = SimpleNamespace(
        _agent_hidden_workspace_filenames=(".run_results.md",),
        _agent_hidden_workspace_path_prefixes=(
            ".run_results.md",
            "logs",
            "submission_snapshots",
            ".logs",
            ".memory",
            ".scienceflow_checkpoints",
            ".git",
        ),
        _tool_output_artifacts=None,
        _exec_feedback_max_chars=4000,
    )
    for name in (
        "_get_agent_hidden_workspace_filenames",
        "_normalize_hidden_workspace_path",
        "_get_agent_hidden_workspace_path_prefixes",
        "_text_mentions_hidden_workspace_prefix",
        "_hide_agent_hidden_workspace_filename_mentions",
        "_path_mentions_hidden_workspace_file",
        "_tool_request_mentions_hidden_workspace_file",
        "_hide_agent_hidden_workspace_file_lines",
        "_prepare_tool_feedback_for_memory",
        "_dedup_resource_feedback_for_memory",
    ):
        setattr(dummy, name, MethodType(getattr(ScienceAgent, name), dummy))

    class FakeMemoryCtx:
        def record_tool_result(self, *args, **kwargs):
            raise AssertionError("hidden ledger reads must not be recorded verbatim")

    dummy._memory_ctx = FakeMemoryCtx()
    feedback = dummy._prepare_tool_feedback_for_memory(
        "read",
        {"path": ".run_results.md"},
        ToolResult(output="[.run_results.md]\nsecret stage ledger"),
    )
    assert ".run_results.md" not in feedback
    assert "secret stage ledger" not in feedback
    assert "hidden control file" in feedback

    filtered = dummy._hide_agent_hidden_workspace_file_lines(
        "FILE solution.py\nDIR logs/\nDIR submission_snapshots/\nDIR .memory/\nFILE .logs/train.log\nFILE .logs/agentic_route_response.md\nFILE .run_results.md (100 bytes)\nFILE submission.csv"
    )
    assert ".run_results.md" not in filtered
    assert "submission_snapshots" not in filtered
    assert "DIR logs" not in filtered
    assert ".memory" not in filtered
    assert "agentic_route_response" not in filtered
    assert ".logs/train.log" not in filtered
    assert "solution.py" in filtered
    assert "submission.csv" in filtered

    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "ls -la *.py artifacts/ submission_snapshots/ | head"}
    ) is True
    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "tail -30 .logs/train_classifier.log"}
    ) is True
    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "tail -30 tmp/train_classifier.log"}
    ) is False
    assert dummy._tool_request_mentions_hidden_workspace_file(
        "bash", {"command": "ls -la logs/ artifacts/ submission.csv"}
    ) is True

def test_lhr_stage_commit_text_block_parser_requires_files() -> None:
    text = """
    STAGE_COMMIT_BEGIN
    stage_id: S04
    metric: 0.12
    metric_validity: medium
    brief: valid metric but missing files.
    why: this must not be accepted because lineage would be ambiguous.
    STAGE_COMMIT_END
    """

    _parsed, _block_text, reason = _parse_stage_commit_text_block(text)

    assert reason == "missing=files"

def test_lhr_stage_commit_json_block_parser_requires_one_fenced_object() -> None:
    text = """```json
{
  "stage_id": "S04",
  "metric": -4.835054,
  "metric_validity": "high",
  "metric_source": "official evaluator",
  "lower_is_better": true,
  "run_time_sec": 12.3,
  "brief": "validated model and feature route",
  "why": "improved the held-out result",
  "files": "code=train.py weights=models/best.pt"
}
```"""

    parsed, block_text, reason = _parse_stage_commit_json_block(text)

    assert reason == ""
    assert parsed["stage_id"] == "S04"
    assert parsed["lower_is_better"] is True
    assert parsed["files"] == "code=train.py weights=models/best.pt"
    assert block_text == text

    _parsed, _block_text, reason = _parse_stage_commit_json_block(
        "extra text\n" + text,
    )
    assert reason == "missing_json_block"

def test_lhr_stage_commit_files_retry_exhaustion_uses_deterministic_fallback(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.log_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_commit_persist_agent_write_to_memory=False,
            stage_commit_persist_to_memory=False,
        )
        solver.pending_text_stage_commit = {
            "stage_id": "S03",
            "attempts": 2,
            "now": 1.0,
            "solution_sha": "solution-sha",
            "run_signature": "run-signature",
            "metric_event": {
                "metric_value": 0.42,
                "metric_name": "score",
                "metric_validity": "high",
                "lower_is_better": False,
                "submission_status": "ok",
                "artifact_path": "artifacts/best_solution.json",
                "artifact_sha": "artifact-sha",
                "gate_accepted": True,
            },
        }

        async def no_audit(**_kwargs):
            return None

        async def finalize(**_kwargs):
            return "finalized"

        solver._audit_stage_result_before_commit = no_audit
        solver._finalize_stage_capture_after_commit = finalize
        agent = SimpleNamespace()
        assistant_text = """
        STAGE_COMMIT_BEGIN
        stage_id: S03
        metric: 0.42
        metric_validity: high
        brief: accepted metric with a stale file reference.
        why: preserve the accepted evaluator evidence.
        files: code=missing.py
        STAGE_COMMIT_END
        """

        out = await solver._handle_pending_stage_commit_text(
            agent=agent,
            assistant_text=assistant_text,
        )

        assert out == "finalized"
        assert solver.pending_text_stage_commit is None
        ledger = solver.ledger_path.read_text(encoding="utf-8")
        assert "### S03" in ledger
        assert "FILES: none" in ledger
        events = (solver.log_dir / "lhr_events.jsonl").read_text(encoding="utf-8")
        assert "stage_commit_text_fallback_applied" in events
        assert "fallback_after_files_retry_exhausted" in events

    asyncio.run(_run())

def test_lhr_stage_commit_compact_context_skips_prior_stage_records() -> None:
    class Dummy(AgentRoutingTestSurface):
        pass

    dummy = Dummy()
    base_messages = [
        Message.user_message("You are solving one optimization task in a continuous REPL workspace.\nTask body"),
        Message.assistant_message(
            "STAGE_COMMIT_BEGIN\nstage_id: S01\nmetric: 1.0\nSTAGE_COMMIT_END\n\n"
            "bash/edit output:\n[stage append-only write]\nstatus: ok",
        ),
        Message.tool_message("recent solver output " + ("x" * 6000), "bash", "tool-call-1"),
        Message.assistant_message("recent design note: local repair improved the candidate"),
    ]

    compact = dummy._build_lnr_stage_commit_compact_messages(
        base_messages=base_messages,
        transient_user_prompt="[LNR_STAGE_COMMIT_REQUEST]\nFACTS:\nstage_id: S02",
    )
    text = "\n".join(str(msg.content or "") for msg in compact)

    assert "[LNR_STAGE_COMMIT_TASK_CONTEXT]" in text
    assert "[LNR_STAGE_COMMIT_RECENT_CONTEXT]" in text
    assert "recent design note" in text
    assert "[stage append-only write]" not in text
    assert "STAGE_COMMIT_BEGIN\nstage_id: S01" not in text
    assert len(text) < 12000


def test_lhr_stage_commit_missing_block_requests_retry_without_crashing(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.log_dir.mkdir(parents=True, exist_ok=True)
        solver.pending_text_stage_commit = {
            "stage_id": "S01",
            "attempts": 0,
            "metric_event": {"metric_value": 0.42, "metric_validity": "high"},
        }
        agent = SimpleNamespace()

        out = await solver._handle_pending_stage_commit_text(
            agent=agent,
            assistant_text="I will continue investigating the candidate.",
        )

        assert out is None
        assert solver.pending_text_stage_commit["attempts"] == 1
        assert agent._lnr_stage_commit_text_handled is True
        assert agent._lnr_suppress_current_text_only_memory == (
            "stage_commit_parse_failed:missing_block"
        )
        events = (solver.log_dir / "lhr_events.jsonl").read_text(encoding="utf-8")
        assert "stage_commit_text_parse_retry" in events
        assert "missing_block" in events

    asyncio.run(_run())


def test_lhr_stage_commit_transient_prompt_disables_tools(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    agent = SimpleNamespace()

    solver._set_stage_commit_transient_prompt(
        agent,
        stage_id="S01",
        metric_event={
            "metric_value": 0.123,
            "metric_name": "Final Validation Score",
            "metric_validity": "medium",
            "lower_is_better": False,
        },
    )

    assert "STAGE_COMMIT_BEGIN" in agent._lnr_transient_user_prompt
    assert agent._lnr_transient_tool_choice_none is True
    assert agent._lnr_transient_context_mode == "stage_commit_compact"
    assert agent._lnr_stage_commit_text_pending is True
    assert agent._lnr_stage_commit_text_handled is False

    solver._clear_stage_commit_transient_prompt(agent)

    assert agent._lnr_transient_user_prompt == ""
    assert agent._lnr_transient_tool_choice_none is False
    assert agent._lnr_transient_context_mode == ""
    assert agent._lnr_stage_commit_text_pending is False

def test_lhr_stage_commit_transient_prompt_can_inherit_context_and_keep_tool_prefix(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        stage_commit_context_mode="inherit",
        stage_commit_tool_choice="auto",
    )
    agent = SimpleNamespace()

    solver._set_stage_commit_transient_prompt(
        agent,
        stage_id="S01",
        metric_event={"metric_value": 0.123, "metric_validity": "medium"},
    )

    assert agent._lnr_transient_context_mode == ""
    assert agent._lnr_transient_tool_choice_none is False
    assert agent._lnr_stage_commit_text_pending is True

def test_lhr_stage_commit_transient_prompt_rejects_unknown_modes(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_tool_choice="required")
    agent = SimpleNamespace()

    with pytest.raises(ValueError, match="stage_commit_tool_choice"):
        solver._set_stage_commit_transient_prompt(
            agent,
            stage_id="S01",
            metric_event={"metric_value": 0.123},
        )

    assert not hasattr(agent, "_lnr_transient_user_prompt")
    assert not hasattr(agent, "_lnr_transient_tool_choice_none")

def test_lhr_abandon_pending_stage_commit_clears_live_agent_state(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.pending_text_stage_commit = {"stage_id": "S03"}
    agent = SimpleNamespace(
        _lnr_transient_user_prompt="stale stage prompt",
        _lnr_transient_tool_choice_none=True,
        _lnr_transient_context_mode="stage_commit_compact",
        _lnr_transient_user_prompt_active=True,
        _lnr_stage_commit_text_pending=True,
        _lnr_stage_commit_text_handled=True,
        _lnr_suppress_current_text_only_memory="stale",
    )

    solver._abandon_pending_stage_commit_text(agent)

    assert solver.pending_text_stage_commit is None
    assert agent._lnr_transient_user_prompt == ""
    assert agent._lnr_transient_tool_choice_none is False
    assert agent._lnr_transient_context_mode == ""
    assert agent._lnr_transient_user_prompt_active is False
    assert agent._lnr_stage_commit_text_pending is False
    assert agent._lnr_stage_commit_text_handled is False
    assert agent._lnr_suppress_current_text_only_memory == ""

def test_lhr_stage_commit_json_prompt_is_opt_in(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        stage_commit_context_mode="inherit",
        stage_commit_tool_choice="auto",
        stage_commit_output_format="json",
    )

    prompt = solver._build_stage_commit_text_prompt(
        stage_id="S01",
        metric_event={
            "metric_value": 0.123,
            "metric_validity": "high",
            "metric_source_note": "official evaluator",
            "lower_is_better": False,
            "run_time_sec": 12.3,
        },
    )

    assert "Return exactly one fenced JSON object and no additional text" in prompt
    assert "```json" in prompt
    assert '"stage_id": "S01"' in prompt
    assert '"lower_is_better": false' in prompt
    assert "STAGE_COMMIT_BEGIN" not in prompt

def test_lhr_stage_commit_experiment_state_uses_only_structured_query_fields(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver._task_metric_lower_is_better = True
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)
    metric_event = {
        "metric_value": 0.06,
        "metric_validity": "high",
        "lower_is_better": True,
        "selection_eligible": True,
        "metric_source_note": "untrusted text says queries_remaining=99/99",
        "extra": {
            "queries_remaining": 7,
            "queries_used": 3,
            "query_limit": 10,
        },
    }

    state = solver._build_stage_commit_experiment_state(
        stage_id="S03",
        metric_event=metric_event,
    )

    assert "global_best_stage: W00:L01:S03" in state
    assert "global_best_metric: 0.06" in state
    assert "queries_remaining: 7" in state
    assert "queries_used: 3" in state
    assert "query_limit: 10" in state
    assert "99/99" not in state

def test_lhr_stage_commit_experiment_state_does_not_readd_rejected_active_stage(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.global_log_dir = tmp_path / "task_logs"
    solver.global_log_dir.mkdir()
    solver._task_metric_lower_is_better = False
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)
    solver.stage_snapshots = {
        "S01": SimpleNamespace(
            node_uid="W00:L01:S01",
            source_event={
                "validation_ok": False,
                "selection_eligible": False,
                "metric_validity": "low",
            },
        )
    }
    history_path = solver.global_log_dir / lnr_solver_module.LHR_STAGE_PERFORMANCE_CSV
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "worker_id",
                "candidate_id",
                "stage_id",
                "metric_value",
                "validation_ok",
                "metric_validity",
                "selection_eligible",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "worker_id": "W00",
                "candidate_id": "W00:L01:S01",
                "stage_id": "S01",
                "metric_value": 0.99,
                "validation_ok": "false",
                "metric_validity": "low",
                "selection_eligible": "false",
            }
        )

    state = solver._build_stage_commit_experiment_state(
        stage_id="S03",
        metric_event={
            "metric_value": 0.06,
            "metric_validity": "high",
            "lower_is_better": False,
            "selection_eligible": True,
        },
    )

    assert "global_best_stage: S02" in state
    assert "global_best_metric: 0.08" in state
    assert "global_best_metric: 0.99" not in state

def test_lhr_stage_commit_experiment_state_does_not_repeat_committed_stage_as_pending(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver._task_metric_lower_is_better = True
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)

    state = solver._build_stage_commit_experiment_state(
        stage_id="S02",
        metric_event={
            "metric_value": 0.08,
            "metric_validity": "high",
            "lower_is_better": True,
            "selection_eligible": True,
        },
    )

    assert "current stage pending summary" not in state
    assert state.count("- S02:") == 1

def test_lhr_stage_commit_experiment_state_default_prompt_stays_compact(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=False)

    prompt = solver._build_stage_commit_text_prompt(
        stage_id="S03",
        metric_event={"metric_value": 0.06, "lower_is_better": True},
    )

    assert "EXPERIMENT_STATE" not in prompt
    assert "brief: <one compact judgment sentence>" in prompt
    assert "only your STAGE_COMMIT block and the append confirmation" in prompt
    assert "files: <code=core.py,helper.py weights=model.ckpt" in prompt
    assert "FILES guidance" not in prompt
