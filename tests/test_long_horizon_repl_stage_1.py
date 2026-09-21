"""Long Horizon Repl contracts: stage."""

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403
from scienceflow.research.solver.lnr.orchestration.coordinator.run import (
    coordination,
    event_projection,
)


def test_lhr_script_run_parser_accepts_relative_variant_entrypoints() -> None:
    assert python_script_run_rel_path("python3 solution_v7.py 2>&1") == "solution_v7.py"
    assert python_script_run_rel_path("python -u train.py") == "train.py"
    assert python_script_run_rel_path("python3 ./predict.py") == "predict.py"
    assert python_script_run_rel_path("python3 ../train.py") is None
    assert python_script_run_rel_path("python3 /tmp/train.py") is None
    assert python_script_run_rel_path("QUICK_TEST_ROWS=10 python3 train.py") is None
    assert looks_like_bare_solution_run("python3 solution.py 2>&1")
    assert not looks_like_bare_solution_run("python3 solution_v7.py 2>&1")


def test_lhr_first_user_prompt_allows_reusable_train_predict_layout() -> None:
    text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)

    assert "continuous REPL workspace" in text
    assert "root-level `submission.csv`" in text
    assert "`train.py` / `predict.py` / `util.py`" in text
    assert "simple or short-run tasks" in text
    assert "same validation split or folds" in text
    assert "train/test feature logic does not drift" in text
    assert "Use allocated compute deliberately" in text
    assert "sustained idle assigned compute" in text
    assert "build or improve `solution.py`" not in text


def test_lhr_bash_timeout_respects_configured_cap_and_remaining_fuse() -> None:
    assert _effective_lnr_bash_timeout_sec(300, 6900) == 300
    assert _effective_lnr_bash_timeout_sec(14_400, 6900) == 6900
    assert _effective_lnr_bash_timeout_sec(300, 120) == 120


def test_lnr_runtime_context_appends_to_latest_existing_tool_message() -> None:
    class _History:
        def __init__(self, messages: list[Message]) -> None:
            self.messages = messages

        def retrieve(self, window_size=None):
            return [
                SimpleNamespace(memory_record=SimpleNamespace(message=message))
                for message in self.messages
            ]

    class _MemoryContext:
        def __init__(self, history: _History) -> None:
            self.history = history

        def rewrite_messages(self, messages: list[Message]) -> None:
            self.history.messages = list(messages)

    class _Harness(LNRHooksTestSurface):
        pass

    history = _History(
        [
            Message.user_message("work"),
            Message.tool_message("tool output", "bash", "call-1"),
        ],
    )
    harness = _Harness()
    harness.memory = SimpleNamespace(chat_history_memory=history)
    harness._memory_ctx = _MemoryContext(history)
    remaining = iter((100, 90))
    harness._lnr_runtime_context_provider = lambda: (
        "Runtime context (current worker limits for planning the next action):\n"
        f"wall_clock_remaining_sec: {next(remaining)}\n"
        "effective_bash_timeout_sec: 30"
    )

    harness._lnr_maybe_periodic_inject_at_round_start(1)
    harness._lnr_maybe_periodic_inject_at_round_start(1)

    assert len(history.messages) == 2
    assert str(history.messages[-1].content).count("Runtime context (") == 1
    assert "wall_clock_remaining_sec: 90" in str(history.messages[-1].content)
    assert "wall_clock_remaining_sec: 100" not in str(history.messages[-1].content)


def test_lnr_runtime_context_uses_existing_user_message_when_no_tool_result() -> None:
    class _Harness(LNRHooksTestSurface):
        pass

    messages = [Message.user_message("work")]
    history = SimpleNamespace(
        retrieve=lambda window_size=None: [
            SimpleNamespace(memory_record=SimpleNamespace(message=message))
            for message in messages
        ],
    )
    harness = _Harness()
    harness.memory = SimpleNamespace(chat_history_memory=history)
    harness._memory_ctx = SimpleNamespace(
        rewrite_messages=lambda updated: messages.__setitem__(slice(None), updated),
    )
    harness._lnr_runtime_context_provider = lambda: (
        "Runtime context (current worker limits for planning the next action):\n"
        "wall_clock_remaining_sec: 90"
    )

    harness._lnr_maybe_periodic_inject_at_round_start(0)

    assert len(messages) == 1
    assert "wall_clock_remaining_sec: 90" in str(messages[0].content)


def test_lnr_runtime_context_preserves_ordinary_text_containing_marker() -> None:
    class _Harness(LNRHooksTestSurface):
        pass

    marker = "Runtime context (current worker limits for planning the next action):"
    messages = [Message.user_message(f"quote: {marker}\nkeep this explanation")]
    history = SimpleNamespace(
        retrieve=lambda window_size=None: [
            SimpleNamespace(memory_record=SimpleNamespace(message=message))
            for message in messages
        ],
    )
    harness = _Harness()
    harness.memory = SimpleNamespace(chat_history_memory=history)
    harness._memory_ctx = SimpleNamespace(
        rewrite_messages=lambda updated: messages.__setitem__(slice(None), updated),
    )
    harness._lnr_runtime_context_provider = lambda: (
        f"{marker}\nwall_clock_remaining_sec: 90\neffective_bash_timeout_sec: 30"
    )

    harness._lnr_maybe_periodic_inject_at_round_start(0)

    content = str(messages[0].content)
    assert "keep this explanation" in content
    assert content.count(marker) == 2


def test_lhr_runtime_context_reports_effective_bash_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.deadline = 1100.0
    monkeypatch.setattr(lnr_solver_module.time, "monotonic", lambda: 1000.0)

    context = solver._runtime_context_for_agent(SimpleNamespace(_bash_timeout_sec=600))

    assert "wall_clock_remaining_sec: 100" in context
    assert "effective_bash_timeout_sec: 100" in context


def test_lhr_first_user_prompt_category_skill_hint_is_optional() -> None:
    default_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)
    hinted_text = build_first_user_prompt(
        "Task body",
        wall_clock_budget_sec=300,
        skill_hint="A category-specific skill is available. Use `skill list`.",
    )

    assert "Use `skill list`" not in default_text
    assert "A category-specific skill is available. Use `skill list`." in hinted_text


def test_lhr_first_user_prompt_seed_is_optional() -> None:
    default_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)
    seeded_text = build_first_user_prompt(
        "Task body", wall_clock_budget_sec=300, seed=2222
    )

    assert "Base experiment seed" not in default_text
    assert "Base experiment seed: 2222" in seeded_text
    assert "random_state" in seeded_text


def test_lhr_default_prompt_does_not_include_opt_solver_progress_protocol() -> None:
    text = build_first_user_prompt(
        "Train a model and write submission.csv.",
        wall_clock_budget_sec=300,
        task_profile="mlebench",
    )

    assert "Progress protocol for optimization search" not in text
    assert "tmp/progress.json" not in text


def test_lhr_prompt_template_loader_accepts_profile_subdir() -> None:
    text = load_prompt_template("task/opt_solver/first_user.md")

    assert "optimization task" in text
    assert "Progress protocol for optimization search" in text


def test_lhr_agent_does_not_fallback_from_explicit_invalid_contract(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "invalid_contract_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)
    interpretation_cb = MagicMock()
    object.__setattr__(
        agent, "_lnr_metric_output_interpretation_callback", interpretation_cb
    )

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(
                output="Final Validation Score: nan\nBest Validation AUC: 0.9\n",
                error="",
            ),
        )
    )

    assert agent._lnr_snapshot_ok is False
    assert agent._lnr_snapshot_reason == "score_contract_invalid"
    interpretation_cb.assert_not_called()
    agent._inject_run_control_user_message.assert_called_once()


def test_lhr_prompt_templates_are_markdown_backed() -> None:
    first_user_template = load_prompt_template(ML_FIRST_USER_TEMPLATE)
    estra_template = load_prompt_template(ESTRA_DECISION_TEMPLATE)

    assert "continuous REPL workspace" in first_user_template
    assert "{task}" in first_user_template
    estra_resume_template = load_prompt_template(ESTRA_RESUME_TEMPLATE)
    keep_current_template = load_prompt_template(KEEP_CURRENT_COMPACT_TEMPLATE)

    assert "Return exactly one JSON object only" in estra_template
    assert "{ledger}" in estra_template
    assert "{base_stage_memory_policy}" in estra_resume_template
    assert "{base_stage_memory_policy}" in keep_current_template
    assert "Exploration state:" in estra_resume_template
    assert "Exploration state:" in keep_current_template


def test_lnr_system_core_adds_lnr_ledger_boundary_once() -> None:
    core = "After every substantive training/validation command returns, immediately update a workspace ledger."
    text = _append_lnr_main_agent_protocols(core)
    text2 = _append_lnr_main_agent_protocols(text)

    assert "workspace ledger" in text
    assert "LNR ledger boundary" in text
    assert "Do not create extra planning systems" in text
    assert "stage-like ledgers" in text
    assert "tmp/experiment_notes.md" in text
    assert "hidden stage ledger" in text
    assert "Rules:" in text
    assert "PENDING/REPLAN/policy blocked" in text
    assert "STOP_BOUNDARY_VIOLATION" in text
    assert "Heartbeat format: SCIENCEFLOW_HB v=1" in text
    assert text2.count("LNR ledger boundary") == 1
    assert text2.count("Resource feedback is runtime context") == 1
    assert text2.count("Optional reduction payload boundary") == 1
    assert "targets three distinct defensible artifacts" in text
    assert "Do not create duplicate or cosmetically perturbed artifacts" in text
    assert "merge_payload/" in text


def test_lhr_config_defaults_enable_repl_style_git_and_code_hint() -> None:
    from scienceflow.foundation.config.schema.settings import LnrConfig

    cfg = LnrConfig()
    assert cfg.code_organization_hint == "beyond_mfiles"
    assert cfg.workspace_git_enabled is True
    assert cfg.workspace_git_auto_checkpoint is True
    assert cfg.workspace_git_track_globs == ["*.py", "*.md"]
    assert cfg.force_estra_capture_duplicate_submissions is False
    assert cfg.state_packet_max_chars == 12000
    assert cfg.resource_bash_monitor_all_enabled is True
    assert cfg.resource_bash_hard_fuse_finalization_reserve_sec == 900.0
    assert cfg.resource_research_cadence_enabled is True
    assert cfg.resource_research_cadence_observe_sec == 300.0
    assert cfg.resource_first_comparable_metric_budget_sec == 900.0
    assert cfg.resource_proven_route_metric_budget_sec == 1800.0
    assert cfg.resource_arbiter_min_progress_windows == 2
    assert cfg.resource_arbiter_kill_requires_high_confidence is True
    assert cfg.resource_gpu_util_observer_enabled is True
    assert cfg.resource_gpu_util_sample_interval_sec == 30.0
    assert cfg.resource_gpu_capacity_slots == 1.0
    assert cfg.resource_gpu_tt_max_per_gpu == 3
    assert cfg.resource_gpu_feature_max_per_gpu == 2
    assert cfg.resource_gpu_share_tt_with_train is False
    assert cfg.resource_queue_timeout_hard_gate_enabled is True
    assert cfg.resource_queue_timeout_block_train_after == 1
    assert cfg.resource_queue_timeout_tt_only_after == 2


def test_lhr_flat_dataset_adapts_legacy_split_without_exposing_shallow(
    tmp_path: Path,
) -> None:
    from scienceflow.research.solver.lnr.lifecycle.workspace.prep_fs import (
        prepare_workspace_dataset_flat,
        resolve_workspace_dataset_source,
    )

    split_root = tmp_path / "dataset_split"
    deep = split_root / "Deep"
    shallow = split_root / "Shallow"
    deep.mkdir(parents=True)
    shallow.mkdir(parents=True)
    (deep / "train.csv").write_text("x\n1\n2\n", encoding="utf-8")
    (shallow / "train.csv").write_text("x\n1\n", encoding="utf-8")

    source, layout = resolve_workspace_dataset_source(split_root)
    ws_dataset = tmp_path / "workspace" / "dataset"
    prepare_workspace_dataset_flat(source, ws_dataset)

    assert layout == "legacy_split_deep"
    assert (ws_dataset / "train.csv").resolve() == (deep / "train.csv").resolve()
    assert not (ws_dataset / "Shallow").exists()
    assert not (ws_dataset / "Deep").exists()


def test_lhr_flat_dataset_falls_back_from_self_referential_deep_symlink(
    tmp_path: Path,
) -> None:
    from scienceflow.research.solver.lnr.lifecycle.workspace.prep_fs import resolve_workspace_dataset_source

    prepared = tmp_path / "prepared"
    split_root = prepared / "dataset_split"
    deep = split_root / "Deep"
    shallow = split_root / "Shallow"
    public = prepared / "public"
    deep.mkdir(parents=True)
    shallow.mkdir(parents=True)
    public.mkdir(parents=True)
    (public / "train.csv").write_text("x\n1\n", encoding="utf-8")
    (deep / "train").symlink_to(deep / "train")

    source, layout = resolve_workspace_dataset_source(deep)
    assert source == public
    assert layout == "legacy_split_public_fallback"

    source, layout = resolve_workspace_dataset_source(split_root)
    assert source == public
    assert layout == "legacy_split_public_fallback"


def test_lhr_flat_dataset_repairs_broken_deep_media_symlink(tmp_path: Path) -> None:
    from scienceflow.research.solver.lnr.lifecycle.workspace.prep_fs import (
        prepare_workspace_dataset_flat,
        resolve_workspace_dataset_source,
    )

    prepared = tmp_path / "prepared"
    split_root = prepared / "dataset_split"
    deep = split_root / "Deep"
    shallow = split_root / "Shallow"
    public = prepared / "public"
    deep.mkdir(parents=True)
    shallow.mkdir(parents=True)
    public.mkdir(parents=True)
    (public / "train_images").mkdir()
    (public / "train_images" / "000.jpg").write_text("image", encoding="utf-8")
    (deep / "train_metadata.json").write_text("{}", encoding="utf-8")
    (deep / "train_images").symlink_to("/missing/legacy/train_images")

    source, layout = resolve_workspace_dataset_source(deep)
    assert source == deep
    assert layout == "flat"

    source, layout = resolve_workspace_dataset_source(split_root)
    assert source == deep
    assert layout == "legacy_split_deep"

    ws_dataset = tmp_path / "workspace" / "dataset"
    prepare_workspace_dataset_flat(source, ws_dataset)
    assert (ws_dataset / "train_images").resolve() == (
        public / "train_images"
    ).resolve()
    assert (ws_dataset / "train_metadata.json").resolve() == (
        deep / "train_metadata.json"
    ).resolve()


def test_config_load_ignores_removed_deep_shallow_keys(tmp_path: Path) -> None:
    from scienceflow.foundation.config.schema.settings import load_cfg

    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "enable_deep_shallow: true",
                "prep_shallow_fraction: 0.1",
                "deep_submission_dir: legacy_deep_submissions",
                "deep_solution_dir: legacy_deep_solutions",
                "agent:",
                "  deep_feedback_to_shallow: true",
                "lnr:",
                "  deep_enabled: true",
                "  ensemble_enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning):
        cfg = load_cfg(cfg_path, cli_args=False)

    assert not hasattr(cfg, "enable_deep_shallow")
    assert not hasattr(cfg, "prep_shallow_fraction")
    assert not hasattr(cfg, "deep_submission_dir")
    assert not hasattr(cfg.agent, "deep_feedback_to_shallow")
    assert not hasattr(cfg.lnr, "deep_enabled")


def test_lhr_state_packet_includes_stage_checkpoint_metadata(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        state_packet_max_chars=12000,
        state_packet_stage_card_max_chars=420,
        state_packet_archived_branch_max_chars=1500,
    )
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="snap-s01",
            snapshot_path=tmp_path / "snap-s01",
            metric_value=0.07,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=1,
            source_event={
                "source_commit_sha": "abcdef1234567890",
                "submission_snapshot": "submission_snapshots/s01.csv",
                "validation_ok": True,
                "lineage_id": "L01",
            },
            node_uid="W00:L01:S01",
            lineage_id="L01",
        ),
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="snap-s02",
            snapshot_path=tmp_path / "snap-s02",
            metric_value=0.08,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=2,
            source_event={
                "source_commit_sha": "123456abcdef9999",
                "submission_snapshot": "submission_snapshots/s02.csv",
                "validation_ok": True,
                "lineage_id": "L01",
            },
            node_uid="W00:L01:S02",
            lineage_id="L01",
        ),
    }

    packet = solver._build_lhr_state_packet(
        action="switch_stage",
        target_stage="S01",
        terminal_stage="S02",
        reason="S01 is better",
        tail_summary="S02 regressed",
    )

    assert "LHR State Packet v2" in packet
    assert "ESTRA action: switch_stage" in packet
    assert "ESTRA target stage: S01" in packet
    assert "Terminal stage before estra: S02" in packet
    assert "Best known restorable stage: S01" in packet
    assert "source_commit=abcdef123456" in packet
    assert "submission_snapshot=" not in packet
    assert "S02 regressed" in packet
    rows = [
        json.loads(line)
        for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
    ]
    built = [row for row in rows if row["event"] == "state_packet_built"][-1]
    assert built["payload"]["target_stage"] == "S01"
    assert built["payload"]["best_stage"] == "S01"


def test_stage_ledger_accepts_compact_cards_and_bold_fields() -> None:
    text = """
### S01
**metric:** 0.062
**lower_is_better:** true
**run_time_sec:** 12.3
**metric_type:** holdout
**metric_note:** valid held-out validation evidence
**metric_validity:** high
**BRIEF:** catboost baseline
**WHY:** first valid checkpoint
**FILES:** code=train.py,model.py weights=models/fold0.ckpt
route_evidence: verdict=continue; reason=valid cheap baseline; next=blend logits
"""
    cards = parse_stage_cards(text)
    assert [c.stage_id for c in cards] == ["S01"]
    assert cards[0].metric == "0.062"
    assert cards[0].run_time_sec == "12.3"
    assert cards[0].metric_type == "holdout"
    assert cards[0].metric_note == "valid held-out validation evidence"
    assert cards[0].metric_validity == "high"
    assert cards[0].files == "code=train.py,model.py weights=models/fold0.ckpt"
    assert (
        cards[0].route_evidence
        == "verdict=continue; reason=valid cheap baseline; next=blend logits"
    )
    ok, reason, card = validate_stage_card(text, "S01")
    assert ok, reason
    assert card is not None and card.brief == "catboost baseline"
    assert card.route_evidence.startswith("verdict=continue")
    rendered = render_stage_cards(cards)
    assert "FILES: code=train.py,model.py weights=models/fold0.ckpt" in rendered
    assert next_stage_id(cards) == "S02"


def test_stage_ledger_requires_files_for_stage_commit() -> None:
    text = """
### S01
metric: 0.062
lower_is_better: true
BRIEF: catboost baseline
WHY: first valid checkpoint
"""
    ok, reason, card = validate_stage_card(text, "S01")

    assert not ok
    assert card is not None
    assert "FILES" in reason


def test_stage_event_summary_does_not_create_new_stage(tmp_path: Path) -> None:
    path = tmp_path / "run_results.md"
    path.write_text(
        "### S01\nmetric: 0.060612\nlower_is_better: true\nBRIEF: Optuna LGBM route\nWHY: validation route found\nFILES: code=train.py\n",
        encoding="utf-8",
    )
    append_stage_event_summary(
        path,
        target_stage="S01",
        summary="Ready submission generated from the same route; no new research stage.",
    )

    cards = parse_stage_cards(path.read_text(encoding="utf-8"))
    assert [card.stage_id for card in cards] == ["S01"]
    assert next_stage_id(cards) == "S02"
    assert cards[0].stage_events == (
        "Ready submission generated from the same route; no new research stage.",
    )
    rendered = render_stage_cards(cards)
    assert "EVENT: Ready submission generated" in rendered


def test_append_only_stage_commit_rejects_rewriting_existing_stage() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py

### S02
metric: 0.06
lower_is_better: true
BRIEF: original branch
WHY: first improvement
FILES: code=train.py
"""
    after = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py

### S02
metric: 0.05
lower_is_better: true
BRIEF: rewritten branch
WHY: silently changed old stage
FILES: code=train.py

### S03
metric: 0.049
lower_is_better: true
BRIEF: next branch
WHY: appended after rewrite
FILES: code=train.py
"""
    ok, reason = validate_append_only_stage_commit(before, after, "S03")
    assert not ok
    assert "append-only" in reason


def test_salvage_append_only_stage_commit_extracts_new_stage_from_rewrite() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: preserved baseline
WHY: original branch
FILES: code=train.py
"""
    attempted = """### S01
metric: 0.01
lower_is_better: true
BRIEF: incorrectly rewritten baseline
WHY: should be ignored
FILES: code=train.py

### S02
metric: 0.06
lower_is_better: true
BRIEF: new branch after estra
WHY: valid new stage
FILES: code=train.py
"""
    ok, salvaged, reason = salvage_append_only_stage_commit(before, attempted, "S02")
    assert ok, reason
    assert salvaged.startswith(before)
    assert "incorrectly rewritten" not in salvaged
    assert "new branch after estra" in salvaged
    ok, reason = validate_append_only_stage_commit(before, salvaged, "S02")
    assert ok, reason


def test_append_only_stage_commit_rejects_stage_gap() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py
"""
    after = (
        before
        + "\n### S03\nmetric: 0.06\nlower_is_better: true\nBRIEF: skipped S02 after estra\nWHY: invalid gap\nFILES: code=train.py\n"
    )
    ok, reason = validate_append_only_stage_commit(before, after, "S03")
    assert not ok
    assert "next active stage" in reason


def test_append_only_stage_commit_rejects_stage_regression() -> None:
    before = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py

### S03
metric: 0.06
lower_is_better: true
BRIEF: later stage
WHY: global monotonic id
FILES: code=train.py
"""
    after = (
        before
        + "\n### S02\nmetric: 0.065\nlower_is_better: true\nBRIEF: duplicate old id\nWHY: invalid numbering\nFILES: code=train.py\n"
    )
    ok, reason = validate_append_only_stage_commit(before, after, "S02")
    assert not ok
    assert "next active stage" in reason


def test_artifact_sha_duplicate_stage_lookup() -> None:
    snap = StageSnapshot(
        stage_id="S01",
        snapshot_id="S01-abc",
        snapshot_path=Path("/tmp/snap"),
        metric_value=2.5,
        metric_name="radii_sum",
        lower_is_better=False,
        memory_cut=3,
        source_event={"artifact_sha": "same-artifact"},
    )
    assert LnrSolver._stage_with_artifact_sha({"S01": snap}, "same-artifact") is snap
    assert LnrSolver._stage_with_artifact_sha({"S01": snap}, "different") is None
    assert LnrSolver._stage_with_artifact_sha({"S01": snap}, "") is None


def test_stage_result_audit_rewrites_formal_entry_before_commit(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Kendall tau over notebooks, higher better."
        solver._task_metric_lower_is_better = False
        solver.lhr = SimpleNamespace(metric_validity_adjudicator_enabled=False)
        metric_event = {
            "metric_value": 0.99,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "declared_lower_is_better": True,
            "metric_validity": "high",
            "selection_eligible": True,
            "candidate_ready": True,
            "validation_ok": True,
            "val_score_type": "holdout",
            "submission_status": "ready",
            "run_time_sec": 12.0,
        }
        judgment = {
            "brief": "Stacking route reports a strong validation score.",
            "why": "Meta model overfits the same validation set, so this should not drive selection.",
            "metric_validity": "high",
            "lower_is_better": "true",
        }

        await solver._audit_stage_result_before_commit(
            agent=SimpleNamespace(),
            stage_id="S02",
            metric_event=metric_event,
            judgment=judgment,
            block_text="STAGE_COMMIT_BEGIN\nlower_is_better: true\nmetric_validity: high\nSTAGE_COMMIT_END",
            source="main_agent_text_block",
        )
        entry = solver._stage_commit_entry_from_metric_event(
            stage_id="S02", metric_event=metric_event, judgment=judgment
        )

        assert metric_event["lower_is_better"] is False
        assert metric_event["metric_direction_source"] == "task_description"
        assert metric_event["metric_direction_conflict"] is True
        assert metric_event["metric_validity"] == "low"
        assert metric_event["selection_eligible"] is False
        assert "lower_is_better: false" in entry
        assert "metric_validity: low" in entry
        assert "BRIEF:" in entry
        assert "WHY:" in entry

    asyncio.run(_run())


def test_stage_commit_files_filter_keeps_code_and_ml_weights_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "solve.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "repair.py").write_text("print(2)\n", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "helper.py").write_text("print(3)\n", encoding="utf-8")
    (tmp_path / "best_solution.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "fold0.ckpt").write_text("weights\n", encoding="utf-8")

    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S03",
        metric_event={
            "metric_value": 1.2,
            "lower_is_better": False,
            "solution_path": "solve.py",
        },
        judgment={
            "brief": "repair route",
            "why": "valid method code should be preserved",
            "files": "code=solve.py,repair.py,outputs/helper.py weights=best_solution.json,models/fold0.ckpt",
        },
        workspace_dir=tmp_path,
    )

    assert "FILES: code=solve.py,repair.py weights=models/fold0.ckpt" in entry
    assert "best_solution.json" not in entry
    assert "outputs/helper.py" not in entry
    cards = parse_stage_cards(entry)
    assert cards[0].files == "code=solve.py,repair.py weights=models/fold0.ckpt"


def test_stage_commit_preserves_full_brief_and_why_in_ledger() -> None:
    long_brief = " ".join(["geometry feature branch"] * 20)
    long_why = " ".join(
        ["preserve this route lesson because the validation behavior differs"] * 30
    )

    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S04",
        metric_event={
            "metric_value": 0.061,
            "lower_is_better": True,
            "run_time_sec": 12.0,
            "val_score_type": "holdout",
            "metric_validity": "high",
        },
        judgment={
            "brief": long_brief,
            "why": long_why,
            "files": "code=train.py",
        },
    )

    assert f"BRIEF: {long_brief}" in entry
    assert f"WHY: {long_why}" in entry
    assert "..." not in entry
    card = parse_stage_cards(entry)[0]
    assert card.brief == long_brief
    assert card.why == long_why


def test_stage_commit_rejects_invalid_files_instead_of_guessing_workspace_code(
    tmp_path: Path,
) -> None:
    for name in ("util.py", "train.py", "predict.py"):
        (tmp_path / name).write_text("print('stage')\n", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "helper.py").write_text(
        "print('output')\n", encoding="utf-8"
    )

    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S04",
        metric_event={"metric_value": 0.0609, "lower_is_better": True},
        judgment={
            "brief": "bowing features improved CV.",
            "why": "valid candidate should remain traceable.",
            "files": 'code name, df in [=("train", train), ("val", val)]',
        },
        workspace_dir=tmp_path,
    )

    assert "FILES:" not in entry
    ok, reason, _card = validate_stage_card(entry, "S04")
    assert not ok
    assert "FILES" in reason


def test_stage_commit_accepts_explicit_no_files_without_retry(tmp_path: Path) -> None:
    entry = LnrSolver._stage_commit_entry_from_metric_event(
        stage_id="S04",
        metric_event={"metric_value": 0.0609, "lower_is_better": True},
        judgment={
            "brief": "artifact-only evaluator candidate.",
            "why": "valid evaluator metric with no reusable code or weights.",
            "files": "none",
        },
        workspace_dir=tmp_path,
    )

    assert "FILES: none" in entry
    ok, reason, card = validate_stage_card(entry, "S04")
    assert ok, reason
    assert card is not None
    assert card.files == "none"


def test_lhr_state_machine_store_writes_unified_events_and_state(
    tmp_path: Path,
) -> None:
    store = LHRStateMachineStore(
        log_dir=tmp_path,
        worker_id="W00",
        worker_index=0,
        worker_count=1,
        ledger_filename="run_results.md",
    )
    store.mark_run_status("running")
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-abc",
            "metric_event": {
                "metric_value": 0.061,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    store.append_event(
        "estra_invalid",
        task_type="estra_decision",
        task_id="estra_decision:W00:S01",
        status="failed",
        payload={"raw": "S10"},
    )
    events = (tmp_path / "lhr_events.jsonl").read_text().splitlines()
    assert len(events) == 3
    state = __import__("json").loads((tmp_path / "lhr_state.json").read_text())
    assert state["run_status"] == "running"
    assert state["stage_count"] == 1
    assert state["estra_invalid_decisions"] == 1
    assert state["global_best"]["candidate_id"] == "W00:S01"
    assert state["workers"]["W00"]["stage_count"] == 1


def test_lhr_control_payload_relativizes_workspace_paths(tmp_path: Path) -> None:
    owner = SimpleNamespace(
        task_root_dir=tmp_path / "task",
        root_dir=tmp_path / "task",
        workspace_dir=tmp_path / "task" / "workspace",
    )
    owner._relativize_control_payload = lambda value: (
        event_projection._relativize_control_payload(owner, value)
    )
    payload = {
        "snapshot_path": str(tmp_path / "task" / ".snapshots" / "S01-abc"),
        "metric_event": {
            "snapshot_path": str(
                tmp_path / "task" / "workspace" / ".logs" / "fullrun_tail_snapshot.json"
            )
        },
    }
    clean = event_projection._relativize_control_payload(owner, payload)
    assert clean["snapshot_path"] == "./.snapshots/S01-abc"
    assert (
        clean["metric_event"]["snapshot_path"]
        == "workspace/.logs/fullrun_tail_snapshot.json"
    )


def test_lhr_jsonl_writes_unified_events_without_legacy_files(tmp_path: Path) -> None:
    owner = SimpleNamespace(
        log_dir=tmp_path / "logs",
        root_dir=tmp_path,
        task_root_dir=tmp_path,
        workspace_dir=tmp_path / "workspace",
        worker_id="",
        worker_index=0,
    )
    owner.state_machine = LHRStateMachineStore(
        log_dir=owner.log_dir,
        worker_id="W00",
        worker_index=0,
        worker_count=1,
        ledger_filename="run_results.md",
    )
    owner._jsonl_roots = lambda: event_projection._jsonl_roots(owner)
    owner._state_task_type_for_event = event_projection._state_task_type_for_event
    owner._relativize_control_payload = lambda value: (
        event_projection._relativize_control_payload(owner, value)
    )
    owner._mirror_state_event = lambda **kwargs: event_projection._mirror_state_event(
        owner, **kwargs
    )
    event_projection._jsonl(
        owner,
        "lhr_stage_events.jsonl", {"event": "stage_capture_failed", "stage_id": "S01"}
    )
    assert not (owner.log_dir / "lhr_stage_events.jsonl").exists()
    assert (owner.log_dir / "lhr_events.jsonl").is_file()
    assert (
        coordination._count_jsonl_events(
            owner, "lhr_stage_events.jsonl", "stage_capture_failed"
        )
        == 1
    )
