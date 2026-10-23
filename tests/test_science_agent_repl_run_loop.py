"""Science Agent Repl contracts: run loop."""

from __future__ import annotations

import hashlib

from inquirycraft.tools import ToolResult

from tests._science_agent_repl_support import *  # noqa: F401,F403
from tests._long_horizon_repl_support import _minimal_lhr_solver


def test_candidate_artifact_archive_is_metric_and_git_independent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    snapshots = tmp_path / "logs" / "submission_snapshots"
    ledger = tmp_path / "logs" / "checkpoints" / "artifact_archive.jsonl"
    workspace.mkdir()
    submission = workspace / "submission.csv"
    submission.write_text("id,y\n1,0.5\n", encoding="utf-8")

    first = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="submission.csv",
        artifact_kind="submission_csv",
        snapshot_dir=snapshots,
        ledger_path=ledger,
        trigger="bash",
    )
    duplicate = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="submission.csv",
        artifact_kind="submission_csv",
        snapshot_dir=snapshots,
        ledger_path=ledger,
        trigger="read",
    )
    submission.write_text("id,y\n1,0.7\n", encoding="utf-8")
    second = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="submission.csv",
        artifact_kind="submission_csv",
        snapshot_dir=snapshots,
        ledger_path=ledger,
        trigger="bash",
        tool_error=True,
    )

    assert first.archived is True
    assert first.snapshot_path.endswith(".csv")
    assert duplicate.archived is False
    assert duplicate.snapshot_path == first.snapshot_path
    assert second.archived is True
    assert second.snapshot_path != first.snapshot_path
    assert (workspace / first.snapshot_path).read_text(
        encoding="utf-8"
    ) == "id,y\n1,0.5\n"
    assert (workspace / second.snapshot_path).read_text(
        encoding="utf-8"
    ) == "id,y\n1,0.7\n"
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["metric_value"] is None
    assert rows[0]["selection_eligible"] is None
    assert rows[1]["tool_error"] is True

def test_candidate_artifact_archive_supports_nested_optimization_json(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    artifact = workspace / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"score": 2.5}\n', encoding="utf-8")

    result = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="artifacts/best_solution.json",
        artifact_kind="json_solution",
        snapshot_dir=tmp_path / "logs" / "artifact_snapshots",
        ledger_path=tmp_path / "logs" / "checkpoints" / "artifact_archive.jsonl",
        trigger="write",
    )

    assert result.archived is True
    assert result.artifact_kind == "json_solution"
    assert result.snapshot_path.endswith(".json")
    assert (workspace / result.snapshot_path).read_text(
        encoding="utf-8"
    ) == '{"score": 2.5}\n'


def test_candidate_artifact_change_remains_pending_until_stage_capture(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True)
    solver.log_dir.mkdir(parents=True)
    artifact = solver.workspace_dir / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    solver._evaluator_candidate_artifact = lambda: "artifacts/best_solution.json"
    solver._evaluator_candidate_artifact_kind = lambda: "json_solution"
    solver._jsonl = MagicMock()
    agent = SimpleNamespace(
        _lnr_candidate_artifact_known_sha="",
        _lnr_candidate_artifact_pending_sha="",
    )

    artifact.write_text('{"score": 2.5}\n', encoding="utf-8")
    solver._archive_candidate_artifact_after_tool(
        agent=agent,
        tool_name="write",
        args={},
        tool_result=ToolResult(output="ok", error=""),
    )
    pending_sha = agent._lnr_candidate_artifact_pending_sha

    solver._archive_candidate_artifact_after_tool(
        agent=agent,
        tool_name="bash",
        args={},
        tool_result=ToolResult(output="FINAL radii_sum=2.5", error=""),
    )

    assert pending_sha
    assert agent._lnr_candidate_artifact_changed_after_tool is False
    assert agent._lnr_candidate_artifact_pending_sha == pending_sha


def test_existing_candidate_artifact_is_a_baseline_not_a_new_candidate(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True)
    solver.log_dir.mkdir(parents=True)
    artifact = solver.workspace_dir / "artifacts" / "best_solution.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"score": 2.5}\n', encoding="utf-8")
    known_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    solver._evaluator_candidate_artifact = lambda: "artifacts/best_solution.json"
    solver._evaluator_candidate_artifact_kind = lambda: "json_solution"
    solver._jsonl = MagicMock()
    agent = SimpleNamespace(
        _lnr_candidate_artifact_known_sha=known_sha,
        _lnr_candidate_artifact_pending_sha="",
    )

    solver._archive_candidate_artifact_after_tool(
        agent=agent,
        tool_name="bash",
        args={},
        tool_result=ToolResult(output="FINAL radii_sum=2.5", error=""),
    )

    assert agent._lnr_candidate_artifact_changed_after_tool is False
    assert agent._lnr_candidate_artifact_pending_sha == ""

def test_repl_manifest_defaults_merge_repl_config_from_defaults_and_task() -> None:
    cfg = Config()

    apply_repl_manifest_defaults(
        cfg,
        {
            "config": "scienceflow/foundation/config/default.yaml",
            "workspace_base": "/tmp/ignored",
            "qa_max_steps": 33,
            "repl_max_steps": 120,
            "repl_tool_preset": "write_edit",
            "repl_bash_max_output_chars": 7000,
        },
        {
            "run_id": "one",
            "cpu_list": "0-3",
            "repl_max_steps": 240,
            "repl_tool_preset": "bash_write",
            "repl_pin_environment_context": False,
            "repl_code_organization_hint": "mle_multifile",
            "repl_workspace_git_enabled": False,
            "repl_workspace_git_track_globs": ["*.py", "*.md", "*.ipynb"],
            "repl_workspace_git_auto_checkpoint": False,
            "repl_workspace_git_auto_review": True,
            "repl_bash_max_stream_line_chars": 900,
            "repl_bash_observation_summary": False,
        },
    )

    assert cfg.qa_max_steps == 33
    assert cfg.repl_max_steps == 240
    assert cfg.repl_tool_preset == "bash_write"
    assert cfg.repl_pin_environment_context is False
    assert cfg.repl_code_organization_hint == "mle_multifile"
    assert cfg.repl_workspace_git_enabled is False
    assert cfg.repl_workspace_git_track_globs == ["*.py", "*.md", "*.ipynb"]
    assert cfg.repl_workspace_git_auto_checkpoint is False
    assert cfg.repl_workspace_git_auto_review is True
    assert cfg.repl_bash_max_output_chars == 7000
    assert cfg.repl_bash_max_stream_line_chars == 900
    assert cfg.repl_bash_observation_summary is False
    assert not hasattr(cfg, "workspace_base")
    assert not hasattr(cfg, "cpu_list")

def test_profile_overrides_keep_unregistered_task_defaults_neutral() -> None:
    cfg = Config()

    apply_profile_overrides(cfg)

    assert cfg.evaluator.task_profile == "auto"
    assert cfg.evaluator.backend == "auto"
    assert cfg.lnr.code_organization_hint == "beyond_mfiles"
    assert cfg.repl_code_organization_hint == ""
    assert cfg.evaluator.candidate.artifact == ""

def test_profile_overrides_isolate_opt_solver_prompt_defaults() -> None:
    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"

    apply_profile_overrides(cfg)

    assert cfg.lnr.code_organization_hint == "opt_solver"
    assert cfg.repl_code_organization_hint == "opt_solver"
    assert cfg.evaluator.candidate.artifact == ""

def test_profile_overrides_allow_configured_opt_solver_parameters() -> None:
    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"
    cfg.profile_overrides = {
        "opt_solver": {
            "lnr": {
                "code_organization_hint": "custom_opt",
                "resource_gpu_queue_enabled": False,
            },
            "repl": {"repl_code_organization_hint": "custom_repl_opt"},
            "evaluator": {
                "backend": "artifact_command",
                "candidate": {
                    "artifact": "artifacts/best_solution.json",
                    "artifact_kind": "json_solution",
                },
            },
        }
    }

    apply_profile_overrides(cfg)

    assert cfg.lnr.code_organization_hint == "custom_opt"
    assert cfg.lnr.resource_gpu_queue_enabled is False
    assert cfg.repl_code_organization_hint == "custom_repl_opt"
    assert cfg.evaluator.backend == "artifact_command"
    assert cfg.evaluator.candidate.artifact == "artifacts/best_solution.json"
    assert cfg.evaluator.candidate.artifact_kind == "json_solution"

def test_partial_profile_override_preserves_builtin_opt_solver_defaults() -> None:
    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"
    cfg.profile_overrides = {
        "opt_solver": {
            "lnr": {"resource_gpu_queue_enabled": False},
        }
    }

    apply_profile_overrides(cfg)

    assert cfg.lnr.code_organization_hint == "opt_solver"
    assert cfg.repl_code_organization_hint == "opt_solver"
    assert cfg.lnr.resource_gpu_queue_enabled is False

def test_explicit_lnr_override_can_win_after_profile_overrides() -> None:
    from scienceflow.foundation.config.schema.settings import _apply_parallel_manifest_lnr_payload

    cfg = Config()
    cfg.evaluator.task_profile = "opt_solver"
    apply_profile_overrides(cfg)

    _apply_parallel_manifest_lnr_payload(
        cfg.lnr,
        {"code_organization_hint": "mle_multifile"},
        label="lnr",
    )

    assert cfg.lnr.code_organization_hint == "mle_multifile"

def test_repl_runtime_options_default_to_lite_profile() -> None:
    cfg = Config()

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=False,
    )

    assert opts["profile"] == "lite"
    assert opts["tool_preset"] == "bash_write"
    assert opts["max_steps"] == 200
    assert opts["stable_system_prompt"] is True
    assert opts["pin_environment_context"] is True
    assert opts["code_organization_hint"] == ""
    assert opts["workspace_git_enabled"] is True
    assert opts["workspace_git_track_globs"] == ["*.py", "*.md"]
    assert opts["workspace_git_auto_review"] is False
    assert opts["workspace_git_auto_checkpoint"] is True
    assert opts["pin_task_description"] is True
    assert opts["repl_bash_write_mode"] is True
    assert opts["bash_max_output_chars"] == 6000
    assert opts["bash_max_stream_line_chars"] == 1200
    assert opts["bash_observation_summary"] is True

@pytest.mark.parametrize("profile", ["codex_like", "codex", "native", "full_native"])
def test_repl_runtime_options_rejects_removed_profile_aliases(profile: str) -> None:
    cfg = Config()
    cfg.repl_profile = profile

    with pytest.raises(ClickException, match="Invalid repl_profile"):
        _repl_resolve_runtime_options(
            cfg,
            auto_first_user_enabled=False,
        )

def test_repl_runtime_options_auto_first_user_avoids_duplicate_task_pin() -> None:
    cfg = Config()

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["pin_task_description"] is False
    cfg.repl_pin_task_description_when_auto_first_user = True
    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )
    assert opts["pin_task_description"] is True

def test_repl_runtime_options_legacy_and_fallback_steps() -> None:
    cfg = Config()
    cfg.qa_max_steps = 17
    cfg.repl_max_steps = 0
    cfg.repl_profile = "legacy"
    cfg.repl_tool_preset = "write_edit"
    cfg.repl_stable_system_prompt = False
    cfg.repl_pin_environment_context = False

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=False,
    )

    assert opts["profile"] == "legacy"
    assert opts["tool_preset"] == "write_edit"
    assert opts["max_steps"] == 17
    assert opts["stable_system_prompt"] is False
    assert opts["pin_environment_context"] is False
    assert opts["repl_bash_write_mode"] is False
    assert opts["workspace_git_enabled"] is False
    assert opts["workspace_git_auto_checkpoint"] is False

def test_repl_runtime_options_expose_code_organization_hint() -> None:
    cfg = Config()
    cfg.repl_code_organization_hint = "mle_multifile"

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["code_organization_hint"] == "mle_multifile"

def test_repl_code_organization_hint_can_request_mle_multifile_layout() -> None:
    text = _repl_code_organization_prompt("mle_multifile")

    assert "Code organization preference" in text
    assert "`train.py`" in text
    assert "`predict.py`" in text
    assert "`util.py`" in text
    assert "build_features" in text
    assert "one identical feature pipeline" in text
    assert "Do not duplicate feature-engineering code" in text
    assert "submission.csv" in text
    assert "Only add an extra submit/wrapper file" in text
    assert "thin compatible wrapper" in text
    assert "single runnable entrypoint" in text
    assert "solution.py" not in text
    assert _repl_code_organization_prompt("") == ""
    assert _repl_code_organization_prompt("off") == ""

def test_repl_code_organization_hint_can_request_reusable_predict_boundary() -> None:
    text = _repl_code_organization_prompt("beyond_mfiles")

    assert "Code organization preference" in text
    assert "reusable train/predict boundary" in text
    assert "single `solution.py` is acceptable" in text
    assert "`train.py`" in text
    assert "`predict.py`" in text
    assert "`util.py`" in text
    assert "without retraining" in text
    assert "tokenizer/vectorizer" in text
    assert "thresholds" in text
    assert "predict-only iterations" in text
    assert "test-time augmentation" in text
    assert "Do not duplicate train/test feature logic" in text

def test_repl_code_organization_hint_can_request_opt_solver_layout() -> None:
    text = _repl_code_organization_prompt("opt_solver")

    assert "Code organization preference" in text
    assert "solver-oriented workspace files" in text
    assert "`solution.py`" in text
    assert "configured candidate artifact path" in text
    assert "do not invent ML submission files" in text
    assert "resume state" in text
    assert "submission.csv" not in text
    assert "`train.py`" not in text
    assert "`predict.py`" not in text

def test_score_contract_feedback_prefers_finalize_existing_artifacts() -> None:
    from scienceflow.research.quality.embedded_fullrun import (
        _score_contract_user_message,
    )

    text = _score_contract_user_message(
        "missing required `Final Validation Score: <finite_float>` line",
        script_label="train.py",
    )

    assert "[SCORE-CONTRACT-INVALID]" in text
    assert "do not retrain from scratch" in text
    assert "`predict.py` / `score_existing.py`" in text
    assert "loads the existing artifacts" in text
    assert "Final Validation Score" in text
    assert "single-fold progress metric" in text
    assert "cheapest finalization command" in text

@pytest.mark.asyncio
async def test_agentic_text_only_route_prompt_is_ephemeral(tmp_path: Path) -> None:
    route_prompt = (
        "You returned pure text without a tool call.\n"
        "<run_results.md>\n"
        "### S01\nmetric: 0.061\nlower_is_better: true\n"
        "solution design: compact baseline\n"
        "</run_results.md>"
    )
    route_json = (
        '{"action":"rewind_to_step","target_step":"S01",'
        '"reason":"best base","next_plan":"try a distinct regularized run"}'
    )
    route_reasoning = "Compare the available stage evidence before selecting S01."
    llm = _RecordingFakeLLM(
        [
            SimpleNamespace(tool_calls=[], content="No further concrete tool step."),
            SimpleNamespace(
                tool_calls=[],
                content=route_json,
                reasoning_content=route_reasoning,
            ),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
    )
    agent._lnr_agentic_text_only_route_prompt = route_prompt

    out = await agent.run("work on the branch")

    assert "No further concrete tool step" in out
    assert "rewind_to_step" in out
    assert agent._lnr_agentic_text_only_route_prompt_injected is True
    assert agent._lnr_agentic_text_only_route_without_result_md is True
    assert llm.kwarg_snapshots[1].get("tool_choice") == "none"
    assert route_prompt in (llm.message_snapshots[1][-1].content or "")

    stored = agent.memory.chat_history_memory.retrieve(window_size=None)
    memory_text = "\n".join(str(r.memory_record.message.content or "") for r in stored)
    assert "work on the branch" in memory_text
    assert "No further concrete tool step" not in memory_text
    assert "You returned pure text without a tool call" not in memory_text
    assert "rewind_to_step" not in memory_text
    assert "run_results.md" not in memory_text

    response_path = tmp_path / ".logs" / "agentic_route_response.md"
    assert response_path.read_text(encoding="utf-8").strip() == route_json
    rows = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agentic_route_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[-1]["trigger"] == "text_only"
    assert rows[-1]["input_compact"] == route_prompt
    assert rows[-1]["raw_output"] == route_json

    journals = list((tmp_path / ".logs" / "agent_runtime_operations").glob("*.jsonl"))
    assert len(journals) == 2
    route_rows = [
        json.loads(line)
        for path in journals
        if "-text_only-" in path.name
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    intent = next(
        row
        for row in route_rows
        if row["kind"] == "llm" and row["phase"] == "intent"
    )
    settled = next(
        row
        for row in route_rows
        if row["kind"] == "llm" and row["phase"] == "settled"
    )
    assert intent["payload"]["system_messages"]
    assert intent["payload"]["provider_messages"][-1]["content"] == route_prompt
    assert intent["payload"]["call_kind"] == "ephemeral"
    assert intent["payload"]["turn_kind"] == "text_only"
    assert settled["payload"]["message"]["content"] == route_json
    assert (
        settled["payload"]["message"]["reasoning_content"] == route_reasoning
    )
    provider_rows = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_provider_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert provider_rows[-1]["route"] == "text_only"
    assert provider_rows[-1]["call_kind"] == "ephemeral"
    assert provider_rows[-1]["turn_kind"] == "text_only"


@pytest.mark.asyncio
async def test_main_agent_feedback_override_uses_same_runtime_audit(
    tmp_path: Path,
) -> None:
    code_llm = _FakeLLM([])
    feedback_llm = _RecordingFakeLLM(
        [
            SimpleNamespace(
                tool_calls=[],
                content='{"valid":true}',
                reasoning_content="validate the reported metric",
            )
        ]
    )
    feedback_llm.model = "feedback-model"
    agent = ScienceAgent(
        llm=code_llm,
        memory=Memory(max_messages=20),
        workspace_dir=tmp_path,
        max_steps=1,
    )
    agent._scienceflow_stage_id = "S02"
    agent._scienceflow_lineage_id = "L03"
    agent._scienceflow_node_uid = "W00:L03:S02"

    output = await agent.run_ephemeral_agentic_route_prompt(
        "validate metric evidence",
        trigger="metric_validity",
        base_messages=[],
        llm_override=feedback_llm,
        llm_role="feedback",
    )

    assert output == '{"valid":true}'
    provider = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_provider_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert provider[-1]["model"] == "feedback-model"
    assert provider[-1]["llm_role"] == "feedback"
    assert provider[-1]["turn_kind"] == "metric_validity"
    assert provider[-1]["route"] == "metric_validity"
    assert provider[-1]["stage_id"] == "S02"
    assert provider[-1]["lineage_id"] == "L03"
    assert provider[-1]["node_uid"] == "W00:L03:S02"
    assert provider[-1]["call_id"].endswith(":llm:1")

    journals = list(
        (tmp_path / ".logs" / "agent_runtime_operations").glob("*.jsonl")
    )
    assert len(journals) == 1
    operations = [
        json.loads(line)
        for line in journals[0].read_text(encoding="utf-8").splitlines()
    ]
    intent = next(row for row in operations if row["phase"] == "intent")
    settled = next(row for row in operations if row["phase"] == "settled")
    assert intent["payload"]["llm_role"] == "feedback"
    assert intent["payload"]["turn_kind"] == "metric_validity"
    assert intent["payload"]["correlation"] == provider[-1]["correlation"]
    assert settled["payload"]["message"]["content"] == '{"valid":true}'
    assert settled["payload"]["message"]["reasoning_content"] == (
        "validate the reported metric"
    )

def test_repl_code_agent_suffix_is_generic() -> None:
    text = _repl_code_agent_append_prompt()
    assert "continuous workspace session" in text
    assert "Inspect files, modify files, run commands, validate outcomes" in text
    assert "fixed user task" in text
    assert "targeted shell-level changes" in text
    assert "edit files" not in text
    assert "targeted edit" not in text
    assert "targeted reads with offsets or limits" in text
    assert "stage machines" in text
    assert "solution.py" not in text
    assert "submission.csv" not in text
    assert "ml_run_results.md" not in text

def test_repl_system_messages_are_layered_for_stable_code_agent(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=10,
        stable_system_prompt=True,
    )
    msgs = agent._build_system_messages()
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "system"
    assert "continuous REPL session" in (msgs[0].content or "")
    assert "workspace root" in (msgs[1].content or "")
    assert "Nomad2018" not in (msgs[0].content or "")
    assert "Nomad2018" not in (msgs[1].content or "")

def test_code_agent_core_has_cache_aware_observation_without_task_details() -> None:
    text = _code_agent_core_prompt()
    assert "structured, compact observations" in text
    assert "programmatic summaries" in text
    assert "whole-file rewrites" in text
    assert "Nomad2018" not in text
    assert "submission.csv" not in text

def test_plain_repl_disables_auto_snapshot_policy() -> None:
    assert _plain_repl_auto_continue_session(
        AutoContinuePolicy(max_text_only_retries=2),
        "off",
    )
    assert not _plain_repl_auto_continue_session(
        AutoContinuePolicy(max_text_only_retries=2),
        "full",
    )
    assert not _plain_repl_auto_continue_session(DefaultPolicy(), "off")

def test_round_budget_prompt_cap_never_exceeds_soft_total(tmp_path: Path) -> None:
    """Soft cap: LLM must not see current/total like 20/15 when actual round > cap."""
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=50,
        round_budget_prompt_cap=15,
    )
    agent._effective_max_steps = 50
    agent._current_round = 19  # 20th round (1-based current would be 20)
    text = agent._format_round_budget()
    assert "Round 15/15" in text
    assert "remaining: 0" in text
    assert "/15" in text
    assert "20/15" not in text
    assert "16/15" not in text

@pytest.mark.asyncio
async def test_no_progress_hardstop_warning_seen_before_terminate(
    tmp_path: Path,
) -> None:
    """First-stage inject must appear in LLM input before run ends on second stall."""
    llm = _RecordingFakeLLM(
        [
            _tool_msg("bash", {"command": "echo a"}),
            _tool_msg("bash", {"command": "echo b"}),
            _tool_msg("bash", {"command": "echo c"}),
            _tool_msg("bash", {"command": "echo d"}),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=200),
        workspace_dir=tmp_path,
        max_steps=12,
        no_progress_hard_stop_after=2,
        lnr_explore_streak_inject_after=0,
        lnr_single_read_streak_inject_after=0,
        no_success_run_soft_threshold=100,
        no_success_run_hard_threshold=200,
    )
    out = await agent.run("go")
    assert "no-progress hardstop" in (out or "").lower()

    assert len(llm.message_snapshots) >= 3
    joined_round_3 = "\n".join(
        str(getattr(m, "content", "") or "") for m in llm.message_snapshots[2]
    )
    joined_round_3_lower = joined_round_3.lower()
    assert "next round" in joined_round_3_lower
    assert (
        "edit" in joined_round_3_lower
        or "write" in joined_round_3_lower
        or "file-change mechanism" in joined_round_3_lower
    )
