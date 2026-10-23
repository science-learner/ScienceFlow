"""Long Horizon Repl contracts: finalization."""

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403


def test_lhr_invalid_submission_is_not_captured_as_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {}
        solver.last_captured_solution_sha = ""
        solver.last_captured_run_signature = ""
        solver.last_stage_commit_ts = 0.0
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
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.052,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "new-source",
            "solution_path": "solution.py",
            "submission_sha": "bad-submission",
            "validation_ok": True,
            "submission_validation_ok": False,
            "submission_status": "invalid_submission",
            "selection_eligible": True,
            "selection_score": 0.052,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):
            raise AssertionError("invalid submission must not request a stage commit")

        class FakeSnapshotStore:
            def capture(self, **kwargs):
                raise AssertionError("invalid submission must not create a stage snapshot")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver.snapshot_store = FakeSnapshotStore()
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        async def fake_force_estra_after_stage_capture(**kwargs):
            raise AssertionError("invalid submission must not trigger estra")

        solver._force_estra_after_stage_capture = fake_force_estra_after_stage_capture

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S03" not in solver.stage_snapshots
        assert solver.last_captured_solution_sha == "new-source"
        assert solver.last_captured_run_signature
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        events = [row["event"] for row in rows]
        assert "stage_invalid_submission_skipped" in events
        assert "stage_invalid_submission_metric_stage_kept" not in events
        assert "stage_captured" not in events

    asyncio.run(_run())


def test_lhr_ready_submission_materializes_existing_stage_without_new_stage(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                0.060612,
                "Final Validation Score",
                True,
                1,
                {
                    "submission_sha": "stale-old",
                    "solution_sha": "same-source",
                    "candidate_ready": False,
                },
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
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
            "### S01\nmetric: 0.060612\nlower_is_better: true\nBRIEF: Optuna LGBM route\nWHY: validation route found, submission not ready\n",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.060612,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "same-source",
            "submission_sha": "ready-new",
            "solution_path": "train.py",
            "validation_ok": True,
            "selection_eligible": True,
            "selection_score": 0.060612,
            "source_changed": False,
        }

        async def fake_stage_commit(
            *, agent, stage_id, metric_event
        ):  # pragma: no cover - must not run
            raise AssertionError(
                "materializing the same route must not request a new stage commit"
            )

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("materialization event must not append a new stage row")
        )

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert sorted(solver.stage_snapshots) == ["S01"]
        assert solver.stage_snapshots["S01"].source_event["candidate_ready"] is True
        assert (
            solver.stage_snapshots["S01"].source_event["materialized_ready_submission"]
            is True
        )
        cards = parse_stage_cards(solver.ledger_path.read_text(encoding="utf-8"))
        assert [card.stage_id for card in cards] == ["S01"]
        assert cards[0].stage_events == (
            "Ready submission generated from the same route; no new research stage.",
        )
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        events = [row["event"] for row in rows]
        assert "stage_materialized_ready_submission" in events
        assert "stage_captured" not in events
        materialized = [
            row for row in rows if row["event"] == "stage_materialized_ready_submission"
        ][-1]
        assert materialized["payload"]["target_stage"] == "S01"
        assert (
            materialized["payload"]["stage_policy"]
            == "materialized_existing_stage_no_new_research_stage"
        )

    asyncio.run(_run())
