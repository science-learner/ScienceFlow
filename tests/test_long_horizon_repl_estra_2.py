"""Long Horizon Repl contracts: estra."""

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.coordination import (
    _text_only_completion_signature,
)


def test_lhr_context_hygiene_compact_uses_real_estra_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            context_hygiene_compact_enabled=True,
            context_hygiene_large_tool_output_chars=12000,
            context_hygiene_max_stages_without_compact=25,
            context_hygiene_code_churn_stage_threshold=10,
            context_hygiene_large_file_repeat_threshold=10,
            context_hygiene_low_incremental_cache_rate=0.80,
            context_hygiene_low_cache_window=5,
            context_hygiene_min_tokens_since_compact=1,
            context_hygiene_tool_output_min_interval_sec=1800.0,
            estra_enabled=True,
            estra_max_decisions=2,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.cfg = SimpleNamespace(max_messages=100)
        solver.stage_snapshots = {
            "S01": StageSnapshot("S01", "snap1", tmp_path / "s1", 0.07, "Final Validation Score", True, 1, {"source_commit_sha": "aaa111"}, node_uid="W00:L01:S01", lineage_id="L01"),
            "S02": StageSnapshot("S02", "snap2", tmp_path / "s2", 0.08, "Final Validation Score", True, 2, {"source_commit_sha": "bbb222"}, node_uid="W00:L01:S02", lineage_id="L01"),
        }
        solver.estra_decisions = 0
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0
        solver.pending_estra = None
        solver.main_tokens_in = 10
        solver.main_tokens_cached = 8
        solver.context_hygiene_last_stage_tokens_in = 0
        solver.context_hygiene_last_stage_tokens_cached = 0
        solver.context_hygiene_last_compact_tokens_in = 0
        solver.context_hygiene_last_compact_stage_count = 0
        solver.context_hygiene_last_compact_ts = 0.0
        solver.context_hygiene_cache_rates = []
        solver.started_at = 0.0

        monkeypatch.setattr(
            lnr_context_coordination,
            "evaluate_context_hygiene_compact",
            lambda **kwargs: SimpleNamespace(should_compact=True, reason="unit hygiene", facts={"unit": True}),
        )

        class FakeLLM:
            _last_call_input_tokens = 10
            _last_call_output_tokens = 2
            _last_call_input_cached_tokens = 8

            async def ask(self, **kwargs):
                assert "ESTRA trigger" not in kwargs["messages"][0].content
                return '{"action":"switch_stage","target_stage":"S01","reason":"better historical route"}'

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5
            _run_tokens_in = 0
            _run_tokens_cached = 0

            @staticmethod
            def _sanitize_agent_visible_paths(text: str) -> str:
                return text

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))
        out = await LnrSolver._context_hygiene_compact_after_stage_capture(
            solver,
            agent=FakeAgent(),
            cards_after=cards,
        )

        assert out == "[lnr] estra switch_stage chosen from context hygiene: S01"
        assert solver.estra_decisions == 1
        assert solver.pending_estra["action"] == "switch_stage"
        assert solver.pending_estra["target_stage"] == "S01"
        assert solver.pending_estra["trigger_source"] == "context_hygiene"
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        estra = [row for row in rows if row["event"] == "estra_decision"][-1]
        assert estra["payload"]["trigger_source"] == "context_hygiene"
        assert estra["payload"]["action"] == "switch_stage"
        triggered = [row for row in rows if row["event"] == "cache_hygiene_compact_triggered"][-1]
        assert triggered["payload"]["action"] == "switch_stage"
        assert triggered["payload"]["target_stage"] == "S01"

    asyncio.run(_run())

def test_lhr_force_estra_cannot_capture_duplicate_submission(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            estra_enabled=True,
            estra_max_decisions=1,
            force_estra_after_stage_count=2,
            force_estra_target_stage="S01",
            force_estra_capture_duplicate_submissions=True,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                0.07,
                "Final Validation Score",
                True,
                1,
                {"submission_sha": "same", "solution_sha": "old"},
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_estra_stage_count = 0
        solver.last_force_estra_observation_count = 0
        solver.duplicate_submission_skip_keys = set()
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver.ledger_path.write_text(
            "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: base\nWHY: ok\n",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.071,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "new",
            "submission_sha": "same",
            "solution_path": "predict.py",
            "validation_ok": True,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):  # pragma: no cover - should not run
            raise AssertionError("force estra must not capture duplicate candidate artifacts")

        class FakeSnapshotStore:
            def capture(self, **kwargs):
                return StageSnapshot(
                    kwargs["stage_id"],
                    "snap2",
                    tmp_path / "snap2",
                    kwargs.get("metric_value"),
                    kwargs.get("metric_name", ""),
                    kwargs.get("lower_is_better"),
                    kwargs.get("memory_cut", 0),
                    kwargs.get("source_event", {}),
                    node_uid=kwargs.get("node_uid", ""),
                    lineage_id=kwargs.get("lineage_id", ""),
                )

        async def fake_ask_estra(agent, *, trigger_source: str):  # pragma: no cover - should not run
            raise AssertionError("duplicate skip should happen before forced stage estra")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver.snapshot_store = FakeSnapshotStore()
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None
        solver._ask_estra = fake_ask_estra

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert solver.estra_decisions == 0
        assert solver.pending_estra is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert "stage_duplicate_submission_skipped" in events
        assert "stage_captured" not in events
        assert "force_stage_estra_check" not in events
        skipped = [row for row in rows if row["event"] == "stage_duplicate_submission_skipped"][-1]
        assert skipped["payload"]["force_capture_requested"] is True
        assert skipped["payload"]["stage_policy"] == "duplicate_candidate_same_artifact_different_source"

    asyncio.run(_run())

def test_lhr_text_only_callback_records_noop_without_stage_candidates(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.ledger_path.write_text("", encoding="utf-8")
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        async def fake_ask_estra(agent, *, trigger_source: str):
            raise AssertionError("no-op text-only estra must not call the estra LLM")

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="The task is complete.",
            round_idx=1,
            max_steps=10,
        )
        assert out is None
        assert solver.estra_decisions == 0
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["text_only_estra_noop"]
        payload = rows[0]["payload"]
        assert payload["stage_count"] == 0
        assert payload["candidate_count"] == 0
        assert payload["noop_reason"] == "no_stage_candidates"

    asyncio.run(_run())

def test_lhr_text_only_callback_suppresses_repeated_completion_context(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.ledger_path.write_text("", encoding="utf-8")
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        class _Memory:
            def __init__(self) -> None:
                self.messages = []

            def add_message(self, msg) -> None:
                self.messages.append(msg)

        agent = SimpleNamespace(memory=_Memory())
        text = "The task is complete. Final Validation Score: 0.8320054902957775"

        first = await LnrSolver._text_only_estra_callback(
            solver,
            agent=agent,
            assistant_text=text,
            round_idx=1,
            max_steps=10,
        )
        assert first is None
        assert not getattr(agent, "_lnr_suppress_current_text_only_memory", "")

        second = await LnrSolver._text_only_estra_callback(
            solver,
            agent=agent,
            assistant_text=text,
            round_idx=2,
            max_steps=10,
        )
        assert second is None
        assert getattr(agent, "_lnr_suppress_current_text_only_memory", "").startswith(
            "repeated_text_only_completion:2",
        )
        assert not hasattr(agent, "_scienceflow_worker_search_outcome")
        assert len(agent.memory.messages) == 1
        continue_prompt = str(agent.memory.messages[0].content)
        assert "[LNR_CONTINUE_SEARCH]" in continue_prompt
        assert "workers stop only when the time budget expires" in continue_prompt
        assert "Exploration state:" in continue_prompt
        assert "completed evidence, not a stop signal" in continue_prompt
        assert "not something to repeat" in continue_prompt
        assert "Do not overwrite the best submission without a validated candidate" in continue_prompt
        assert solver.estra_decisions == 0
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert events == ["text_only_estra_noop", "text_only_completion_duplicate_suppressed"]
        suppressed = rows[-1]["payload"]
        assert suppressed["repeat_count"] == 2
        assert suppressed["signature"] == "task_complete|final_validation_score:0.8320054902957775"
        assert suppressed["continue_prompt_injected"] is True
        assert "LNR_CONTINUE_SEARCH" in suppressed["continue_prompt"]
        assert "completed evidence, not a stop signal" in suppressed["continue_prompt"]

    asyncio.run(_run())

def test_lhr_text_only_completion_signature_covers_terminal_status_text() -> None:
    assert _text_only_completion_signature("Goodbye.") == "terminal_text"
    assert _text_only_completion_signature(
        "I understand. The conversation has definitively concluded. I will not respond further. Goodbye."
    ) == "terminal_text"
    assert _text_only_completion_signature(
        "Final status confirmed. All experiments concluded. No further actions from this worker."
    ) == "terminal_text"

def test_lhr_text_only_callback_sets_pending_estra_without_stage_capture(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "text_only"
            return {"action": "switch_stage", "target_stage": "S01", "reason": "best"}

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="I am done.",
            round_idx=3,
            max_steps=10,
        )
        assert out == "[lnr] estra switch_stage chosen from text-only: S01"
        assert solver.pending_estra["target_stage"] == "S01"
        assert "S02" in solver.pending_estra["tail_summary"]
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["text_only_estra_check", "state_packet_built"]
        check = [row for row in rows if row["event"] == "text_only_estra_check"][0]["payload"]
        assert check["assistant_text"] == "I am done."
        assert check["assistant_text_chars"] == len("I am done.")
        assert check["round"] == 3
        assert check["max_steps"] == 10

    asyncio.run(_run())

def test_lhr_text_only_callback_keep_current_compacts_without_switch_candidate(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "text_only"
            return {"action": "keep_current", "target_stage": "S02", "reason": "continue current route"}

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="Final answer: 0.89468 AUC",
            round_idx=5,
            max_steps=10,
        )

        assert out == "[lnr] estra keep_current chosen from text-only; compact current stage: S02"
        assert solver.pending_estra["action"] == "keep_current"
        assert solver.pending_estra["target_stage"] == "S02"
        assert solver.pending_estra["trigger_source"] == "text_only"
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["text_only_estra_check", "state_packet_built"]
        check = rows[0]["payload"]
        assert check["assistant_text"] == "Final answer: 0.89468 AUC"

    asyncio.run(_run())


def test_lhr_text_only_estra_does_not_repeat_without_a_new_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 0
        solver.last_estra_observation_key = ""
        solver.estra_decisions = 0
        solver.pending_estra = None
        solver.deadline = 1e12
        calls = []

        async def fake_ask_estra(agent, *, trigger_source: str):
            calls.append(trigger_source)
            return {
                "action": "keep_current",
                "target_stage": "S02",
                "reason": "continue current route",
            }

        solver._ask_estra = fake_ask_estra
        first = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="Finished this attempt.",
            round_idx=3,
            max_steps=10,
        )
        solver.pending_estra = None
        second = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="Finished another attempt.",
            round_idx=4,
            max_steps=10,
        )
        solver.ledger_path.write_text(
            solver.ledger_path.read_text(encoding="utf-8")
            + "\n### S03\nmetric: 0.06\nlower_is_better: true\nBRIEF: improved\nWHY: new stage\n",
            encoding="utf-8",
        )
        solver.stage_snapshots["S03"] = object()
        third = await LnrSolver._text_only_estra_callback(
            solver,
            agent=object(),
            assistant_text="Finished the new stage.",
            round_idx=5,
            max_steps=10,
        )

        assert first is not None
        assert second is None
        assert third is not None
        assert calls == ["text_only", "text_only"]
        # The mocked decision adapter bypasses the canonical event emitter;
        # callback deduplication is asserted through the observed calls.

    asyncio.run(_run())


def test_lhr_text_only_estra_honors_stage_trigger_threshold(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.ledger_path.write_text(
        "### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: baseline\nWHY: start\n",
        encoding="utf-8",
    )
    solver.lhr = SimpleNamespace(estra_enabled=True, estra_trigger_stage_count=2)
    solver.stage_snapshots = {"S01": object()}
    solver.last_estra_stage_count = 0
    solver.last_estra_observation_key = ""
    solver.pending_estra = None
    solver.deadline = 1e12
    cards, _latest, candidates = solver._estra_candidates_for_current_ledger()

    assert not solver._text_only_estra_trigger_allowed(
        cards=cards,
        candidates=candidates,
    )

def test_lhr_switch_estra_synthesizes_target_tail_archive_summary(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.ledger_path.write_text(
            """### S01
metric: 0.90
lower_is_better: false
BRIEF: stable target route
WHY: best reusable checkpoint

### S02
metric: 0.88
lower_is_better: false
BRIEF: noisy feature branch
WHY: regressed after adding noisy aggregate

### S03
metric: 0.87
lower_is_better: false
BRIEF: deeper variant
WHY: repeated shallow tuning without recovery
""",
            encoding="utf-8",
        )
        solver.lhr = SimpleNamespace(
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
            estra_archive_summary_llm_enabled=True,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object(), "S03": object()}
        solver.estra_tokens_in = 0
        solver.estra_tokens_out = 0
        solver.estra_tokens_cached = 0
        solver.estra_llm_calls = 0

        class FakeLLM:
            _last_call_input_tokens = 50
            _last_call_output_tokens = 8
            _last_call_input_cached_tokens = 45

            async def ask(self, **kwargs):
                prompt = kwargs["messages"][0].content
                assert "historical exploration" in prompt.lower()
                assert "Abandoned trajectory after S01 before estra" in prompt
                assert "S02" in prompt and "S03" in prompt
                assert "stable target route" not in prompt
                return "Avoid noisy aggregate feature branch; deeper variant repeated shallow tuning without recovery.\nRestart from S01 and test a materially different signal."

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt
            llm = FakeLLM()
            _llm_stream_timeout_sec = 5

            @staticmethod
            def _record_llm_call(*args, **kwargs) -> None:
                return None

        await LnrSolver._set_pending_estra_from_decision(
            solver,
            agent=FakeAgent(),
            target="S01",
            decision={"action": "switch_stage", "target_stage": "S01", "compact": True, "reason": "return to stable target"},
        )

        summary = solver.pending_estra["tail_summary"]
        assert "Avoid noisy aggregate" in summary
        assert "Restart from S01" in summary
        assert "Abandoned trajectory after" not in summary
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        assert "Historical Exploration Summary" in solver.pending_estra["state_packet"]
        assert "abandoned branch evidence" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "estra_archive_summary_synthesized" in [row["event"] for row in rows]

    asyncio.run(_run())

def test_lhr_context_limit_callback_sets_pending_estra_before_compact(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=1,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 0
        solver.estra_decisions = 0
        solver.pending_estra = None

        async def fake_ask_estra(agent, *, trigger_source: str):
            assert trigger_source == "context_limit"
            return {"action": "switch_stage", "target_stage": "S01", "reason": "context full"}

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=9,
            max_steps=100,
            omitted=7,
        )
        assert out == "[lnr] estra switch_stage chosen from context-limit: S01"
        assert solver.pending_estra["target_stage"] == "S01"
        assert solver.pending_estra["reason"] == "context full"
        assert "S02" in solver.pending_estra["tail_summary"]
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert [row["event"] for row in rows] == ["context_limit_estra_check", "state_packet_built"]
        check = [row for row in rows if row["event"] == "context_limit_estra_check"][0]
        assert check["payload"]["trigger_source"] == "context_limit"
        assert check["payload"]["omitted_before"] == 7

    asyncio.run(_run())

def test_lhr_context_limit_bypasses_regular_estra_gate(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=0,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 2
        solver.estra_decisions = 0
        solver.pending_estra = None

        called = []

        async def fake_ask_estra(agent, *, trigger_source: str):
            called.append(trigger_source)
            return {
                "action": "keep_current",
                "target_stage": "S02",
                "reason": "continue after isolated review",
            }

        solver._ask_estra = fake_ask_estra
        out = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=10,
            max_steps=100,
            omitted=3,
        )

        assert called == ["context_limit"]
        assert out == "[lnr] estra keep_current chosen from context-limit; compact current stage: S02"
        assert solver.pending_estra["target_stage"] == "S02"
        assert solver.pending_estra["reason"] == "continue after isolated review"
        assert "LHR State Packet v2" in solver.pending_estra["state_packet"]
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "estra_deterministic_fallback" not in [row["event"] for row in rows]
        assert "state_packet_built" in [row["event"] for row in rows]

    asyncio.run(_run())
