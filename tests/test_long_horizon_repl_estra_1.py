"""Long Horizon Repl contracts: estra."""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.run import (
    event_projection,
)
from tests._long_horizon_repl_support import *  # noqa: F401,F403


def test_lhr_estra_prompt_can_include_checkpoint_context() -> None:
    text = build_estra_prompt(
        ledger_filename=".run_results.md",
        ledger_text="### S01\nmetric: 0.06\nlower_is_better: true\nBRIEF: base\nWHY: ok\n",
        latest_stage="S02",
        switch_candidate_stages=["S01"],
        stage_checkpoint_context="- S01: metric=0.06; source_commit=abc123",
    )

    assert "Choose the next research direction for one ML search trajectory" in text
    assert "Decide two axes:" in text
    assert "`startpoint`: `current_workspace` or `previous_stage`" in text
    assert "`intent`: `continue` or `redirect`" in text
    assert "Metric is evidence, not the only selection rule" in text
    assert "route potential" in text
    assert "Stage checkpoint context" in text
    assert "source_commit=abc123" in text
    assert "Return exactly one JSON object" in text
    assert "\"startpoint\":\"current_workspace|previous_stage\"" in text

def test_append_only_stage_commit_allows_estra_summary_then_next_stage(tmp_path: Path) -> None:
    before = """# Run Results

### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py
"""
    new_file = """### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start
FILES: code=train.py
"""
    ok, reason = validate_append_only_stage_commit("", new_file, "S01")
    assert ok, reason

    path = tmp_path / "run_results.md"
    path.write_text(before)
    append_archived_trajectory_summary(path, target_stage="S01", summary="S02 regressed and was abandoned")
    with_summary = path.read_text()
    assert with_summary.startswith(before)
    assert "Archived Trajectory before estra to S01" in with_summary
    assert "HISTORICAL_EXPLORATION" in with_summary
    assert "not the active route" in with_summary

    after = with_summary + "\n### S02\nmetric: 0.06\nlower_is_better: true\nBRIEF: new branch after estra\nWHY: improves from restored stage\nFILES: code=train.py\n"
    ok, reason = validate_append_only_stage_commit(with_summary, after, "S02")
    assert ok, reason

def test_estra_tail_summary_is_non_selectable(tmp_path: Path) -> None:
    ledger = """
### S01
metric: 0.07
lower_is_better: true
BRIEF: baseline
WHY: start

### S02
metric: 0.06
lower_is_better: true
BRIEF: tuned model
WHY: better validation
"""
    summary = tail_summary_after(ledger, "S01", max_chars=500)
    assert "S02" in summary
    path = tmp_path / "run_results.md"
    path.write_text("### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: baseline\nWHY: start\n")
    append_estra_summary(path, target_stage="S01", summary=summary)
    text = path.read_text()
    assert "## ESTRA Summary after S01" in text
    assert [c.stage_id for c in parse_stage_cards(text)] == ["S01"]

def test_estra_compact_summary_always_starts_from_s02() -> None:
    ledger = """
### S01
metric: 0.070
lower_is_better: true
BRIEF: base EDA and first valid solution
WHY: preserve root understanding

### S02
metric: 0.066
lower_is_better: true
BRIEF: first branch
WHY: tries more features

### S03
metric: 0.064
lower_is_better: true
BRIEF: restored target candidate
WHY: useful code state

### S04
metric: 0.068
lower_is_better: true
BRIEF: abandoned tail
WHY: no gain
"""
    summary = tail_summary_from_stage(ledger, start_stage="S02", target_stage="S03", max_chars=1000)
    assert "S01" not in summary
    assert "S02" in summary
    assert "S03" in summary
    assert "S04" in summary
    assert "estra to S03" in summary

def test_estra_parser_recovers_deepseek_switch_text_and_pseudo_tool() -> None:
    raw = """<think>S02 is worse than S01. Let me estra to S01 and improve from there.</think>
<｜｜DSML｜｜tool_calls>cat > run_results.md ...</｜｜DSML｜｜tool_calls>"""
    parsed = parse_estra_decision(
        raw,
        switch_candidates=["S01"],
    )
    assert parsed["action"] == "switch_stage"
    assert parsed["target_stage"] == "S01"

def test_estra_parser_uses_first_json_object_not_greedy_block() -> None:
    raw = 'prefix {"action":"switch_stage","target_stage":"S02","reason":"better root"} suffix {"command":"ignored"}'
    parsed = parse_estra_decision(
        raw,
        switch_candidates=["S01", "S02"],
    )
    assert parsed["target_stage"] == "S02"

def test_estra_parser_accepts_keep_but_redirect_diagnostics() -> None:
    raw = json.dumps(
        {
            "action": "keep_but_redirect",
            "exploration_summary": "S04-S06 repeated local tweaks.",
            "bottleneck": "The route has not diagnosed validation weakness.",
            "evidence": "Metric stayed flat across recent stages.",
            "decision_reason": "Keep files but redirect the next stage.",
            "redirect_focus": "Audit validation and missing signal.",
        }
    )
    parsed = parse_estra_decision(raw, switch_candidates=["S01"])
    assert parsed["action"] == "keep_but_redirect"
    assert "validation weakness" in parsed["bottleneck"]

def test_estra_parser_accepts_two_axis_previous_stage_redirect(tmp_path: Path) -> None:
    _ = tmp_path
    raw = json.dumps(
        {
            "startpoint": "previous_stage",
            "intent": "redirect",
            "target_stage": "S01",
            "exploration_summary": "Current path repeated shallow changes.",
            "bottleneck": "The restored route needs a different validation audit.",
            "evidence": "Recent stages plateaued after parameter tweaks.",
            "decision_reason": "Restore S01 but redirect around the bottleneck.",
            "redirect_focus": "Audit CV split and missing signal.",
        }
    )
    parsed = parse_estra_decision(raw, switch_candidates=["S01"])
    decision = EstraService().normalize(
        parsed,
        context=EstraContext(
            latest_stage="S03",
            switch_candidates=("S01",),
            trigger_source="force_stage_capture",
        ),
    )
    assert decision is not None
    normalized = decision.to_dict()
    assert normalized["action"] == "switch_stage"
    assert normalized["startpoint"] == "previous_stage"
    assert normalized["intent"] == "redirect"
    assert normalized["target_stage"] == "S01"
    assert normalized["compact"] is True

def test_lhr_next_stage_id_follows_active_ledger_after_estra(tmp_path: Path) -> None:
    ledger_path = tmp_path / "run_results.md"
    ledger_path.write_text(
        "### S01\nmetric: 0.07\nlower_is_better: true\n"
        "BRIEF: restored root\nWHY: estra target\n"
    )
    owner = SimpleNamespace(ledger_path=ledger_path)
    assert event_projection._next_active_stage_id(owner) == "S02"

def test_lhr_estra_prunes_active_snapshots_to_restored_ledger(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.ledger_path.write_text(
        "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: restored root\nWHY: estra target\n",
        encoding="utf-8",
    )
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="S01-a",
            snapshot_path=tmp_path / "snapshots" / "S01-a",
            metric_value=0.07,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=2,
            source_event={"submission_sha": "keep"},
        ),
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="S02-b",
            snapshot_path=tmp_path / "snapshots" / "S02-b",
            metric_value=0.08,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=4,
            source_event={"submission_sha": "abandoned"},
        ),
    }

    solver._prune_active_stage_snapshots_to_ledger()

    assert sorted(solver.stage_snapshots) == ["S01"]
    assert solver._next_stage_id_for_logging() == "S02"
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    pruned = [row for row in rows if row["event"] == "estra_active_stage_pruned"][-1]
    assert pruned["payload"]["active_stages"] == ["S01"]
    assert pruned["payload"]["removed_stages"] == ["S02"]

def test_lhr_pending_estra_restores_exact_archived_node_uid(tmp_path: Path, monkeypatch) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = solver.workspace_dir / ".agent_memory"
        solver.memory_dir.mkdir(parents=True)
        old = StageSnapshot(
            "S01",
            "old-snapshot",
            tmp_path / "snapshots" / "old",
            0.5,
            "score",
            False,
            1,
            {},
            node_uid="W00:L01:S01",
            lineage_id="L01",
        )
        current = StageSnapshot(
            "S01",
            "current-snapshot",
            tmp_path / "snapshots" / "current",
            0.6,
            "score",
            False,
            2,
            {},
            node_uid="W00:L02:S01",
            lineage_id="L02",
        )
        restored: list[StageSnapshot] = []
        archive_dir = tmp_path / "terminal-archive"
        archive_dir.mkdir()
        solver.snapshot_store = SimpleNamespace(restore=lambda snap: restored.append(snap) or archive_dir)
        solver.stage_snapshots = {"S01": current}
        solver.archived_stage_snapshots = {
            "W00:L01:S01": old,
            "W00:L02:S01": current,
        }
        solver.pending_estra = {
            "action": "switch_stage",
            "target_stage": "S01",
            "target_node_uid": "W00:L01:S01",
            "tail_summary": "discarded tail",
            "state_packet": "state",
        }
        solver.current_lineage_no = 2
        solver.current_lineage_id = "L02"
        solver._close_lnr_interaction_loggers = lambda: None
        solver._prepare_dataset_symlink = lambda: None
        solver._prune_workspace_control_artifacts = lambda: None
        solver._prune_active_stage_snapshots_to_ledger = lambda: None
        solver._rebuild_memory_after_estra = lambda **_kwargs: (True, 3)
        solver._append_traj_summary = lambda **_kwargs: None
        solver._reset_interaction_stage_files = lambda: None
        monkeypatch.setattr(
            lnr_agent_coordination,
            "append_archived_trajectory_summary",
            lambda *_args, **_kwargs: None,
        )

        await solver._restore_pending_estra()

        assert restored == [old]
        assert solver.stage_snapshots["S01"] is old
        assert solver.current_restored_from_node_uid == "W00:L01:S01"
        stage_map = json.loads((solver.log_dir / "lhr_stage_map.json").read_text(encoding="utf-8"))
        assert stage_map["stages"]["S01"]["node_uid"] == "W00:L01:S01"
        assert set(stage_map["archive"]) == {"W00:L01:S01", "W00:L02:S01"}

    asyncio.run(_run())

def test_lhr_failed_estra_unfold_keeps_active_lineage_and_records_failure(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        snapshot = StageSnapshot(
            "S01",
            "snapshot-one",
            tmp_path / "snapshots" / "S01",
            0.5,
            "score",
            False,
            1,
            {},
            node_uid="W00:L01:S01",
            lineage_id="L01",
        )
        solver.stage_snapshots = {"S01": snapshot}
        solver.archived_stage_snapshots = {"W00:L01:S01": snapshot}
        solver.pending_estra = {
            "action": "switch_stage",
            "target_stage": "S01",
            "target_node_uid": "W00:L01:S01",
            "tail_summary": "discarded tail",
            "state_packet": "state",
        }
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.snapshot_store = SimpleNamespace(
            restore=lambda _snap: (_ for _ in ()).throw(OSError("injected unfold failure")),
        )
        solver._close_lnr_interaction_loggers = lambda: None

        await solver._restore_pending_estra()

        assert solver.current_lineage_no == 1
        assert solver.current_lineage_id == "L01"
        assert solver.stage_snapshots == {"S01": snapshot}
        assert solver.pending_estra is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        failed = [row for row in rows if row["event"] == "estra_failed"][-1]
        assert failed["payload"]["reason"] == "unfold_transaction_failed"
        assert failed["payload"]["error_type"] == "OSError"

    asyncio.run(_run())

def test_lhr_estra_memory_compact_records_context_preview(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_compact_enabled=True)
    solver.cfg = SimpleNamespace(max_messages=100)
    solver.memory_dir = tmp_path / "memory"
    snap_dir = tmp_path / "snapshots" / "S01-test"
    agent_dir = snap_dir / ".memory" / "ScienceAgent"
    agent_dir.mkdir(parents=True)
    records = [
        MemoryCompactor.user_record("FIRST USER QUERY"),
        MemoryCompactor.user_record("[LHR protected EDA facts] train shape is stable"),
    ]
    for name in ("short_term.json", "long_term.jsonl"):
        (agent_dir / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="S01-test",
            snapshot_path=snap_dir,
            metric_value=0.1,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=2,
            source_event={},
        )
    }

    ok, n_records = solver._rebuild_memory_after_estra(target_stage="S01", summary="S02 failed; avoid noisy features")

    assert ok
    assert n_records == 3
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    ctx = [row for row in rows if row["event"] == "estra_context_prepared"][-1]
    assert ctx["payload"]["target_stage"] == "S01"
    assert ctx["payload"]["base_stage"] == "S01"
    assert ctx["payload"]["memory_records"] == 3
    assert "run_results.md" not in ctx["payload"]["resume_prompt_excerpt"]
    assert ".run_results.md" not in ctx["payload"]["resume_prompt_excerpt"]
    assert "S02 failed" in ctx["payload"]["tail_summary_excerpt"]
    assert "estra resume" in " ".join(ctx["payload"]["context_shape"])

def test_lhr_estra_memory_compact_uses_first_completed_stage_as_base(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.ledger_path.write_text(
        "### S02\nmetric: 0.10\nlower_is_better: true\nBRIEF: first completed\nWHY: no S01 metric\n",
        encoding="utf-8",
    )
    solver.lhr = SimpleNamespace(estra_compact_enabled=True)
    solver.cfg = SimpleNamespace(max_messages=100)
    solver.memory_dir = tmp_path / "memory"
    snap_dir = tmp_path / "snapshots" / "S02-test"
    agent_dir = snap_dir / ".memory" / "ScienceAgent"
    agent_dir.mkdir(parents=True)
    records = [MemoryCompactor.user_record("FIRST COMPLETED STAGE MEMORY")]
    for name in ("short_term.json", "long_term.jsonl"):
        (agent_dir / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    solver.stage_snapshots = {
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="S02-test",
            snapshot_path=snap_dir,
            metric_value=0.10,
            metric_name="Final Validation Score",
            lower_is_better=True,
            memory_cut=1,
            source_event={},
        )
    }

    ok, n_records = solver._rebuild_memory_after_estra(target_stage="S02", summary="S03 failed")

    assert ok
    assert n_records == 2
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    ctx = [row for row in rows if row["event"] == "estra_context_prepared"][-1]
    assert ctx["payload"]["base_stage"] == "S02"
    assert "S02 prefix-safe memory records" in ctx["payload"]["context_shape"]
    assert "original S02 base stage" in ctx["payload"]["resume_prompt_excerpt"]
    assert "original S01 base stage" not in ctx["payload"]["resume_prompt_excerpt"]

def test_lhr_ask_estra_records_text_only_trigger_source(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {"S01": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                return '{"action":"switch_stage","target_stage":"S01","reason":"best"}'

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="text_only")
        assert decision["action"] == "switch_stage"
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["latest_stage"] == "S02"
        assert route["payload"]["target_stage"] == "S01"
        assert route["payload"]["trigger_source"] == "text_only"
        assert route["payload"]["reason"] == "best"
        assert route["payload"]["action"] == "switch_stage"
        assert route["payload"]["decision_kind"] == "switch"
        assert route["payload"]["response_chars"] > 0
        assert route["task_id"] == "estra_decision:W00:S02"

    asyncio.run(_run())

def test_lhr_ask_estra_context_limit_keep_current_means_compact_continue(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {
            "S01": StageSnapshot("S01", "snap1", tmp_path / "s1", 0.07, "Final Validation Score", True, 1, {}, node_uid="W00:L01:S01", lineage_id="L01"),
            "S02": StageSnapshot("S02", "snap2", tmp_path / "s2", 0.08, "Final Validation Score", True, 2, {}, node_uid="W00:L01:S02", lineage_id="L01"),
        }
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                return '{"action":"keep_current","reason":"continue from terminal"}'

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="context_limit")
        assert decision["action"] == "keep_current"
        assert decision["target_stage"] == "S02"
        assert decision["compact"] is True
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["target_node_uid"] == "W00:L01:S02"
        assert route["payload"]["decision_kind"] == "continue"
        assert route["payload"]["candidate_node_uids"]["S01"] == "W00:L01:S01"
        assert route["payload"]["candidate_node_uids"]["S02"] == "W00:L01:S02"

    asyncio.run(_run())

def test_lhr_ask_estra_keep_but_redirect_compacts_with_diagnostics(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {
            "S01": StageSnapshot("S01", "snap1", tmp_path / "s1", 0.07, "Final Validation Score", True, 1, {}, node_uid="W00:L01:S01", lineage_id="L01"),
            "S02": StageSnapshot("S02", "snap2", tmp_path / "s2", 0.08, "Final Validation Score", True, 2, {}, node_uid="W00:L01:S02", lineage_id="L01"),
        }
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                return json.dumps(
                    {
                        "startpoint": "current_workspace",
                        "intent": "redirect",
                        "exploration_summary": "S01-S02 repeated local parameter changes.",
                        "bottleneck": "The route has not diagnosed why validation is flat.",
                        "evidence": "Recent stage cards show no material metric gain.",
                        "missing_evidence": None,
                        "is_route_flaw": False,
                        "is_execution_flaw": True,
                        "target_stage": None,
                        "decision_reason": "Keep workspace but redirect the next stage.",
                        "redirect_focus": "Diagnose validation and missing signal.",
                    }
                )

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="force_stage_capture")
        assert decision["action"] == "keep_but_redirect"
        assert decision["target_stage"] == "S02"
        assert decision["compact"] is True
        assert "validation is flat" in decision["bottleneck"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["action"] == "keep_but_redirect"
        assert route["payload"]["decision_kind"] == "redirect"
        assert "validation is flat" in route["payload"]["bottleneck"]
        assert "Recent stage cards" in route["payload"]["evidence"]

    asyncio.run(_run())

def test_lhr_ask_estra_allows_redirect_without_historical_switch_candidate(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True)
        solver.stage_snapshots = {"S02": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 5
            _last_call_input_cached_tokens = 90

            async def ask(self, **kwargs):
                return '{"action":"keep_but_redirect","reason":"change current approach"}'

            async def ask_tool_stream(self, **kwargs):
                raise AssertionError("isolated estra should use ask")

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(solver, FakeAgent(), trigger_source="text_only")
        assert decision["action"] == "keep_but_redirect"
        assert decision["target_stage"] == "S02"
        assert decision["compact"] is True
        assert decision["reason"] == "change current approach"
        assert solver.estra_llm_calls == 1
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        payload = route["payload"]
        assert route["task_type"] == "estra_decision"
        assert payload["action"] == "keep_but_redirect"
        assert payload["switch_candidates"] == []
        assert payload["candidates"] == ["S02"]

    asyncio.run(_run())

def test_lhr_estra_trigger_allows_current_route_without_historical_candidate(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_enabled=True, estra_max_decisions=2, estra_trigger_stage_count=2)
    solver.estra_decisions = 0
    solver.last_estra_observation_key = ""
    solver.last_estra_stage_count = 0
    cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))

    assert LnrSolver._estra_trigger_allowed(solver, cards=cards, candidates=["S02"])
    assert LnrSolver._estra_trigger_allowed(solver, cards=cards, candidates=["S01", "S02"])

def test_lhr_estra_max_decisions_is_deprecated_noop(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_enabled=True, estra_max_decisions=1, estra_trigger_stage_count=2)
    solver.estra_decisions = 99
    solver.last_estra_observation_key = ""
    solver.last_estra_stage_count = 0
    cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))

    assert LnrSolver._estra_trigger_allowed(solver, cards=cards, candidates=["S01", "S02"])

def test_lhr_ask_estra_uses_main_agent_context_when_enabled(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True, estra_use_main_agent_context=True)
        solver.stage_snapshots = {"S01": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 5
            _last_call_input_cached_tokens = 95
            model = "gpt-test"
            temperature = 0.0
            max_tokens = 256
            frequency_penalty = None

            def __init__(self) -> None:
                self.seen_messages = []
                self.seen_tool_choice = None
                self.provider_request = {}

            @staticmethod
            def _add_explicit_request_seed(_kwargs) -> None:
                return None

            async def ask_tool_stream(self, **kwargs):
                from inquirycraft.llm.tool_requests import _tool_kwargs

                self.seen_messages = list(kwargs["messages"])
                self.seen_tool_choice = kwargs.get("tool_choice")
                self.provider_request = _tool_kwargs(
                    self,
                    self.seen_messages,
                    kwargs["timeout"],
                    kwargs.get("tools"),
                    self.seen_tool_choice,
                    kwargs.get("temperature"),
                    kwargs.get("parallel_tool_calls"),
                    {},
                    stream=True,
                )
                forbidden = {
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                } & self.provider_request.keys()
                if forbidden:
                    raise ValueError(
                        f"provider received tool controls without tools: {forbidden}"
                    )
                return Message.assistant_message(
                    '{"action":"switch_stage","target_stage":"S01",'
                    '"reason":"main context best"}'
                )

        class FakeMemoryCtx:
            def build_messages_for_llm(self):
                return [Message.user_message("MAIN_MEMORY_MARKER")]

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt

            def __init__(self) -> None:
                self.llm = FakeLLM()
                self._memory_ctx = FakeMemoryCtx()
                self._llm_stream_timeout_sec = 5
                self._tools_with_thought = []

            @staticmethod
            def _build_system_messages():
                return [Message.system_message("sys")]

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        agent = FakeAgent()
        decision = await LnrSolver._ask_estra(
            solver,
            agent,
            trigger_source="force_stage_capture",
        )
        assert decision["action"] == "switch_stage"
        assert agent.llm.seen_tool_choice == "none"
        assert "tool_choice" not in agent.llm.provider_request
        assert agent.llm.seen_messages[0].content == "MAIN_MEMORY_MARKER"
        assert "Stage checkpoint context" in agent.llm.seen_messages[-1].content
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["decision_mode"] == "main_agent_context"
        assert route["payload"]["trigger_source"] == "force_stage_capture"

    asyncio.run(_run())


def test_lhr_context_limit_estra_uses_isolated_context(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_decision_only=True,
            estra_use_main_agent_context=True,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 5
            _last_call_input_cached_tokens = 90

            def __init__(self) -> None:
                self.isolated_calls = 0

            async def ask(self, **kwargs):
                self.isolated_calls += 1
                return '{"action":"keep_current","reason":"compact and continue"}'

            async def ask_tool_stream(self, **kwargs):
                raise AssertionError("context-limit estra must not reuse main context")

        class FakeMemoryCtx:
            def build_messages_for_llm(self):
                raise AssertionError("context-limit estra must not read main memory")

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            def __init__(self) -> None:
                self.llm = FakeLLM()
                self._memory_ctx = FakeMemoryCtx()
                self._llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        agent = FakeAgent()
        decision = await LnrSolver._ask_estra(
            solver,
            agent,
            trigger_source="context_limit",
        )

        assert decision["action"] == "keep_current"
        assert agent.llm.isolated_calls == 1
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        route = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert route["payload"]["decision_mode"] == "isolated"
        assert route["payload"]["trigger_source"] == "context_limit"

    asyncio.run(_run())


def test_lhr_estra_error_emits_safe_current_route_fallback(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_decision_only=True,
            estra_use_main_agent_context=True,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.estra_decisions = 0
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            async def ask(self, **kwargs):
                raise TimeoutError("controller timed out")

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(
            solver,
            FakeAgent(),
            trigger_source="context_limit",
        )

        assert decision["action"] == "keep_current"
        assert decision["startpoint"] == "current_workspace"
        assert decision["intent"] == "continue"
        assert decision["target_stage"] == "S02"
        assert decision["compact"] is True
        assert solver.estra_decisions == 1
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        assert [row["event"] for row in rows] == [
            "estra_llm_error",
            "estra_decision",
        ]
        route = rows[-1]["payload"]
        assert route["decision_mode"] == "safe_fallback"
        assert route["action"] == "keep_current"

    asyncio.run(_run())


def test_lhr_ask_estra_falls_back_when_main_context_emits_tool_markup(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(estra_decision_only=True, estra_use_main_agent_context=True)
        solver.stage_snapshots = {"S01": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 5
            _last_call_input_cached_tokens = 90

            async def ask_tool_stream(self, **kwargs):
                return Message.assistant_message(
                    '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="bash"></｜｜DSML｜｜invoke>'
                )

            async def ask(self, **kwargs):
                return '{"action":"switch_stage","target_stage":"S01","reason":"fallback best"}'

        class FakeMemoryCtx:
            def build_messages_for_llm(self):
                return [Message.user_message("MAIN_MEMORY_MARKER")]

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            def __init__(self) -> None:
                self.llm = FakeLLM()
                self._memory_ctx = FakeMemoryCtx()
                self._llm_stream_timeout_sec = 5
                self._tools_with_thought = []

            @staticmethod
            def _build_system_messages():
                return [Message.system_message("sys")]

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        decision = await LnrSolver._ask_estra(
            solver,
            FakeAgent(),
            trigger_source="force_stage_capture",
        )
        assert decision["action"] == "switch_stage"
        assert decision["startpoint"] == "previous_stage"
        assert decision["intent"] == "continue"
        assert decision["target_stage"] == "S01"
        assert decision["compact"] is True
        assert decision["reason"] == "fallback best"
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows][-2:] == [
            "estra_main_context_invalid_fallback",
            "estra_decision",
        ]
        route = rows[-1]
        assert route["payload"]["decision_mode"] == "isolated_fallback"
        assert route["payload"]["trigger_source"] == "force_stage_capture"

    asyncio.run(_run())

def test_lhr_force_estra_after_s02_sets_pending_estra(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            force_estra_after_stage_count=2,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.ledger_path.write_text(
            "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: base\nWHY: ok\n\n"
            "### S02\nmetric: 0.08\nlower_is_better: true\nBRIEF: worse\nWHY: test route\n",
            encoding="utf-8",
        )

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "force_stage_capture"
            return {"action": "switch_stage", "target_stage": "S01", "reason": "forced validation"}

        solver._ask_estra = fake_ask_estra
        cards = [SimpleNamespace(stage_id="S01"), SimpleNamespace(stage_id="S02")]
        out = await LnrSolver._force_estra_after_stage_capture(
            solver,
            agent=object(),
            cards_after=cards,
        )

        assert out == "[lnr] estra switch_stage chosen from forced stage estra: S01"
        assert solver.pending_estra["target_stage"] == "S01"
        assert solver.pending_estra["reason"] == "forced validation"
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        check = [row for row in rows if row["event"] == "force_stage_estra_check"][-1]
        assert check["payload"]["latest_stage"] == "S02"
        assert check["payload"]["candidate_count"] == 2
        assert set(check["payload"]["candidate_node_uids"]) == {"S01", "S02"}
        assert rows[-1]["event"] == "state_packet_built"

    asyncio.run(_run())
