"""Long Horizon Repl contracts: finalization."""
# ruff: noqa: F405 - this split contract intentionally consumes the shared fixture API.

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403
from scienceflow.research.solver.lnr.orchestration.coordinator.run import agent_coordination, coordination
from scienceflow.research.solver.lnr.orchestration.runtime import LnrRuntime, RunMode, RunSpec, RuntimeServices


def test_run_one_worker_resolves_extracted_solver_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scienceflow.runtime import workflow as orchestrator_module
    from scienceflow.research.solver.lnr.orchestration.coordinator import solver as coordinator_solver
    from scienceflow.research.solver.lnr.orchestration.coordinator.run import coordination as run_coordination

    worker_root = tmp_path / "workers" / "w00"
    worker_root.mkdir(parents=True)
    worker_cfg = SimpleNamespace(
        task_workspace_root_dir=worker_root,
        lnr=SimpleNamespace(wall_clock_budget_sec=60),
    )

    class FakeOrchestrator:
        def __init__(self, cfg: object) -> None:
            self.cfg = cfg

    class FakeWorkerSolver:
        def __init__(self, **_kwargs: object) -> None:
            self._live_agent = None

        async def _run_single(self, *, keep_agent_open: bool) -> dict[str, object]:
            assert keep_agent_open is False
            return {"status": "completed"}

    monkeypatch.setattr(orchestrator_module, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(coordinator_solver, "LnrSolver", FakeWorkerSolver)

    cleaned: list[Path] = []
    host = SimpleNamespace(
        cfg=SimpleNamespace(),
        root_dir=tmp_path,
        lhr=SimpleNamespace(merge_enabled=False),
        task_desc="task",
        solver_name="lnr",
        _make_worker_cfg=lambda _index, _count: worker_cfg,
        _worker_extra_env=lambda **_kwargs: {},
        _jsonl=lambda _name, _event: None,
        _global_merge_reserve_sec=lambda: 0,
        _cleanup_worker_root_artifacts=lambda path: cleaned.append(path),
    )

    result = asyncio.run(run_coordination._run_one_worker(host, 0, 2))

    assert result == {
        "worker_id": "W00",
        "worker_index": 0,
        "status": "completed",
    }
    assert cleaned == [worker_root]


def test_lhr_first_user_prompt_worker_id_is_optional() -> None:
    default_text = build_first_user_prompt("Task body", wall_clock_budget_sec=300)
    worker_text = build_first_user_prompt(
        "Task body", wall_clock_budget_sec=300, worker_id="w01"
    )

    assert "Worker identity" not in default_text
    assert "Worker identity: W01" in worker_text
    assert "worker-local notes" in worker_text


def test_lhr_opt_solver_prompt_uses_profile_template_without_submission_contract() -> (
    None
):
    text = build_first_user_prompt(
        "Produce artifacts/best_solution.json.",
        wall_clock_budget_sec=300,
        task_profile="opt_solver",
        task_runtime_contract=(
            "- Candidate artifact path: `artifacts/best_solution.json`.\n"
            "- Task Python executable: `/opt/env/bin/python`."
        ),
    )

    assert "optimization task" in text
    assert "artifacts/best_solution.json" in text
    assert "Task Python executable" in text
    assert "Progress protocol for optimization search" in text
    assert "tmp/progress.json" in text
    assert "SCIENCEFLOW_HB v=1" in text
    assert "root-level `submission.csv`" not in text
    assert "Final Validation Score" not in text


def test_lhr_keep_current_compact_frames_rsna_final_summary_as_evidence() -> None:
    summary = (
        "## Final deliverable - S05 restored, all exploration complete\n"
        "The workspace is in its best confirmed state.\n"
        "Best deliverable remains submission.csv at 0.2324 holdout WLL."
    )
    prompt = build_keep_current_compact_prompt(
        terminal_stage="S14",
        next_stage="S15",
        base_stage="S05",
        compact_summary=summary,
        state_packet="ESTRA action: keep_current\nLatest stage: S14",
    )

    assert prompt.index("Exploration state:") < prompt.index(
        "Current compact state note:"
    )
    assert "Prior summaries are completed evidence, not a stop signal" in prompt
    assert "not something to repeat" in prompt
    assert "Final deliverable - S05 restored" in prompt


def test_submission_sha_duplicate_stage_lookup() -> None:
    snap = StageSnapshot(
        stage_id="S01",
        snapshot_id="S01-abc",
        snapshot_path=Path("/tmp/snap"),
        metric_value=0.06,
        metric_name="Final Validation Score",
        lower_is_better=True,
        memory_cut=3,
        source_event={"submission_sha": "same-submission"},
    )
    assert (
        LnrSolver._stage_with_submission_sha({"S01": snap}, "same-submission") is snap
    )
    assert LnrSolver._stage_with_submission_sha({"S01": snap}, "different") is None
    assert LnrSolver._stage_with_submission_sha({"S01": snap}, "") is None


def test_multi_worker_cpu_slice_is_contiguous() -> None:
    ids = list(range(128, 160))
    assert LnrSolver._slice_cpu_ids(ids, worker_index=0, worker_count=2) == list(
        range(128, 144)
    )
    assert LnrSolver._slice_cpu_ids(ids, worker_index=1, worker_count=2) == list(
        range(144, 160)
    )
    assert LnrSolver._format_cpu_ids(list(range(128, 144))) == "128-143"


def test_lhr_final_artifact_mode_is_opt_in_without_restricting_merge() -> None:
    owner = SimpleNamespace(lhr=SimpleNamespace(merge_enabled=True))
    assert coordination._final_artifact_mode(owner) == "workspace"

    owner.lhr.final_artifact_mode = "best_stage"
    assert coordination._final_artifact_mode(owner) == "best_stage"

    owner.lhr.final_artifact_mode = "unknown"
    with pytest.raises(ValueError, match="unsupported lnr.final_artifact_mode"):
        coordination._final_artifact_mode(owner)


def test_multi_worker_extra_env_exposes_worker_cpu_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCIENCEFLOW_TASK_CPU_LIST", "96-127")
    owner = SimpleNamespace(
        cfg=SimpleNamespace(exec=SimpleNamespace(cpu_list="")),
        lhr=SimpleNamespace(omp_threads_cap=8, seed=0),
    )

    env = coordination._worker_extra_env(owner, worker_index=2, worker_count=4)

    assert env["SCIENCEFLOW_TASK_CPU_LIST"] == "96-127"
    assert env["SCIENCEFLOW_WORKER_CPU_LIST"] == "112-119"
    assert env["SCIENCEFLOW_CPU_LIST"] == "112-119"
    assert env["_SCIENCEFLOW_CPU_SET"] == "112-119"
    assert env["OMP_NUM_THREADS"] == "8"


def test_multi_worker_failure_kind_uses_llm_quota_error() -> None:
    failure_kind, kinds = LnrSolver._multi_worker_failure_kind(
        worker_results=[
            {
                "status": "failed",
                "error": "APIStatusError: Error code: 402 - Insufficient Balance",
            },
            {
                "status": "failed",
                "error": "APIStatusError: Error code: 402 - Insufficient Balance",
            },
        ],
    )

    assert failure_kind == "llm_quota_error"
    assert kinds == ["llm_quota_error"]


def test_multi_worker_failure_kind_reports_mixed_worker_failures() -> None:
    failure_kind, kinds = LnrSolver._multi_worker_failure_kind(
        worker_results=[
            {
                "status": "failed",
                "error": "APIStatusError: Error code: 402 - Insufficient Balance",
            },
            {
                "status": "failed",
                "error": "RuntimeError: lnr compact did not fit context",
            },
        ],
    )

    assert failure_kind == "worker_failed_mixed"
    assert kinds == ["context_compact_failed", "llm_quota_error"]


def test_multi_worker_failure_kind_handles_failed_workers_without_error() -> None:
    failure_kind, kinds = LnrSolver._multi_worker_failure_kind(
        worker_results=[{"status": "failed"}, {"status": "failed", "error": ""}],
    )

    assert failure_kind == "all_workers_failed"
    assert kinds == []


def test_no_candidate_workers_have_explicit_failure_kind() -> None:
    failure_kind, kinds = LnrSolver._multi_worker_failure_kind(
        worker_results=[{"status": "no_candidate", "stage_count": 0}],
    )

    assert failure_kind == "no_stage_candidate"
    assert kinds == []


def test_single_worker_result_cannot_report_success_without_stage() -> None:
    owner = SimpleNamespace(
        stage_snapshots={},
        solver_name="lnr",
        estra_decisions=0,
        main_tokens_in=0,
        main_tokens_cached=0,
        main_tokens_out=0,
        main_llm_calls=0,
        stage_tokens_in=0,
        stage_tokens_cached=0,
        stage_llm_calls=0,
        estra_tokens_in=0,
        estra_tokens_cached=0,
        estra_llm_calls=0,
        workspace_dir=Path("workspace"),
        ledger_path=Path("stages.md"),
    )
    owner._estra_decision_counts = lambda: {
        key: 0
        for key in (
            "continue",
            "redirect",
            "switch",
            "current_continue",
            "current_redirect",
            "stage_continue",
            "stage_redirect",
        )
    }
    owner._count_jsonl_events = lambda *_args: 0
    owner._score_summary_for_prompt = lambda: {}

    result = agent_coordination._result(owner, stop_reason="budget_expired")

    assert result["status"] == "no_candidate"
    assert result["outcome"] == "stop_no_candidate"
    assert result["failure_kind"] == "no_stage_candidate"


def test_multi_worker_stop_reason_preserves_success_budget_semantics() -> None:
    reason, kinds = LnrSolver._multi_worker_stop_reason(
        run_succeeded=True,
        worker_results=[{"status": "success", "error": ""}],
    )

    assert reason == "budget_expired"
    assert kinds == []


def test_multi_worker_stop_reason_preserves_query_budget_exhaustion() -> None:
    reason, kinds = LnrSolver._multi_worker_stop_reason(
        run_succeeded=True,
        worker_results=[
            {
                "status": "success",
                "stop_reason": "evaluator_query_budget_exhausted",
            },
            {
                "status": "success",
                "stop_reason": "evaluator_query_budget_exhausted",
            },
        ],
    )

    assert reason == "evaluator_query_budget_exhausted"
    assert kinds == []


def test_multi_worker_stop_reason_keeps_mixed_success_reasons_generic() -> None:
    reason, kinds = LnrSolver._multi_worker_stop_reason(
        run_succeeded=True,
        worker_results=[
            {
                "status": "success",
                "stop_reason": "evaluator_query_budget_exhausted",
            },
            {"status": "success", "stop_reason": "budget_expired"},
        ],
    )

    assert reason == "budget_expired"
    assert kinds == []


def test_worker_error_kind_classifies_tool_output_and_transport_errors() -> None:
    assert (
        LnrSolver._worker_error_kind(
            "ValueError: Separator is found, but chunk is longer than limit"
        )
        == "tool_output_limit"
    )
    assert (
        LnrSolver._worker_error_kind("ReadError: stream disconnected")
        == "llm_transport_error"
    )


def test_multi_worker_does_not_require_merge_enabled() -> None:
    async def _run() -> None:
        called: list[str] = []

        async def run_multi_worker() -> dict[str, object]:
            called.append("multi")
            return {"mode": "multi", "merge_enabled": False}

        async def run_single() -> dict[str, object]:
            called.append("single")
            return {"mode": "single"}

        runtime = LnrRuntime(
            RuntimeServices(
                spec_factory=lambda: RunSpec(
                    worker_id="",
                    worker_count=3,
                    mode=RunMode.MULTI_WORKER,
                ),
                run_single=run_single,
                run_multi=run_multi_worker,
            )
        )
        result = await runtime.run()

        assert result == {"mode": "multi", "merge_enabled": False}
        assert called == ["multi"]

    asyncio.run(_run())


def test_lhr_run_uses_coordinator_layout_for_one_worker() -> None:
    async def _run() -> None:
        called: list[str] = []

        async def run_multi_worker() -> dict[str, object]:
            called.append("multi")
            return {"mode": "multi", "num_workers": 1}

        async def run_single() -> dict[str, object]:  # pragma: no cover - must not run
            called.append("single")
            return {"mode": "single"}

        runtime = LnrRuntime(
            RuntimeServices(
                spec_factory=lambda: RunSpec(
                    worker_id="",
                    worker_count=1,
                    mode=RunMode.MULTI_WORKER,
                ),
                run_single=run_single,
                run_multi=run_multi_worker,
            )
        )
        result = await runtime.run()

        assert result == {"mode": "multi", "num_workers": 1}
        assert called == ["multi"]

    asyncio.run(_run())


def test_multi_worker_budget_reserves_time_for_global_merge() -> None:
    owner = SimpleNamespace(
        lhr=SimpleNamespace(
            merge_enabled=True,
            wall_clock_budget_sec=3600,
            global_merge_wall_clock_sec=900,
        )
    )

    assert coordination._global_merge_reserve_sec(owner) == 540
    assert coordination._worker_wall_clock_budget_sec(owner) == 3060


def test_multi_worker_stage_collection_writes_candidates_without_selection(
    tmp_path: Path,
) -> None:
    owner = SimpleNamespace(_merge_dir=lambda: tmp_path / "merge")
    worker_results = [
        {
            "worker_id": "W00",
            "status": "success",
            "best_stage": "S01",
            "best_metric": 0.1,
        },
        {
            "worker_id": "W01",
            "status": "success",
            "best_stage": "S02",
            "best_metric": 0.2,
        },
    ]
    candidates = [
        {
            "candidate_id": "W00:S01",
            "metric_value": 0.1,
            "snapshot_path": "/tmp/w00/s01",
            "submission_sha": "aaa",
        },
        {
            "candidate_id": "W01:S02",
            "metric_value": 0.2,
            "snapshot_path": "/tmp/w01/s02",
            "submission_sha": "bbb",
        },
    ]

    out_dir = coordination._write_stage_collection_outputs(
        owner,
        worker_results=worker_results,
        candidates=candidates,
    )

    assert out_dir == tmp_path / "merge"
    assert (out_dir / "global_candidates.jsonl").read_text(encoding="utf-8").count(
        "candidate_id"
    ) == 2
    stage_map = json.loads(
        (out_dir / "global_stage_map.json").read_text(encoding="utf-8")
    )
    assert "selected" not in stage_map
    assert stage_map["merge"]["merge_mode"] == "not_run"
    report = (out_dir / "stage_collection_report.md").read_text(encoding="utf-8")
    assert "candidate_count: 2" in report
    assert "W00:S01" in report


def test_lhr_state_machine_aggregates_worker_event_logs(tmp_path: Path) -> None:
    out = tmp_path / "logs"
    w0 = tmp_path / "w00" / "logs"
    w1 = tmp_path / "w01" / "logs"
    s0 = LHRStateMachineStore(
        log_dir=w0, worker_id="W00", worker_index=0, worker_count=2
    )
    s1 = LHRStateMachineStore(
        log_dir=w1, worker_id="W01", worker_index=1, worker_count=2
    )
    s0.mark_run_status("finished")
    s0.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-a",
            "metric_event": {
                "metric_value": 0.07,
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    s1.mark_run_status("finished")
    s1.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W01:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-b",
            "metric_event": {
                "metric_value": 0.06,
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    state = LHRStateMachineStore.aggregate_logs(
        output_log_dir=out,
        input_log_dirs=[w0, w1],
        worker_count=2,
        run_status="finished",
        ledger_filename="run_results.md",
    )
    assert state["stage_count"] == 2
    assert state["global_best"]["candidate_id"] == "W01:S01"
    assert state["workers"]["W00"]["stage_count"] == 1
    assert state["workers"]["W01"]["stage_count"] == 1
    assert (out / "lhr_events.jsonl").is_file()
    assert (out / "lhr_state.json").is_file()


@pytest.mark.asyncio
async def test_lhr_multi_worker_live_aggregation_refreshes_until_cancelled() -> None:
    calls: list[tuple[int, str]] = []

    def fake_aggregate_worker_state(
        *, n_workers: int, run_status: str
    ) -> dict[str, object]:
        calls.append((n_workers, run_status))
        return {"stage_count": len(calls)}

    owner = SimpleNamespace(_aggregate_worker_state=fake_aggregate_worker_state)
    task = asyncio.create_task(
        coordination._aggregate_worker_state_periodically(
            owner,
            n_workers=2,
            run_status="running",
            interval_sec=60.0,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == [(2, "running")]


def test_lhr_primary_gate_accepts_complete_result_without_submission(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True, exist_ok=True)
    solver.cfg = SimpleNamespace(
        exp_id="nomad2018-predict-transparent-conductors",
        evaluator=SimpleNamespace(
            enabled=True,
            task_profile="mlebench",
            backend="task_package",
            stage_source_mode="primary",
            event_log="evaluator_events.jsonl",
            metric=SimpleNamespace(selection_requires_direction=True),
        ),
    )
    solver.deadline = float("inf")
    solver.gate_service = CandidateAssessmentService.default()

    facts = solver._record_evaluator_stage_events(
        stage_id="S03",
        metric_event={
            "metric_value": 0.8,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "val_score_type": "holdout",
            "execution_scale": "direct_full",
            "bash_cmd": "python3 solution.py",
            "solution_sha": "complete-source",
            "submission_status": "missing_submission",
        },
    )

    assert facts["gate_accepted"] is True
    assert facts["gate_reason_code"] == "eligible"
    assert facts["evaluator_backend"] == "result_signal"
    assert facts["candidate_ready"] is True
    assert facts["submission_status"] == "missing_submission"
    assert facts["artifact_sha"] == ""


def test_lhr_duplicate_submission_without_semantic_source_change_is_skipped(
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
                0.07,
                "Final Validation Score",
                True,
                1,
                {
                    "submission_sha": "same",
                    "solution_sha": "old",
                    "candidate_ready": True,
                },
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.duplicate_submission_skip_keys = set()
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
            """### S01
metric: 0.07
lower_is_better: true
BRIEF: base
WHY: ok
""",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.061,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "same-source",
            "submission_sha": "same",
            "solution_path": "train.py",
            "validation_ok": True,
            "selection_eligible": True,
            "selection_score": 0.061,
            "source_changed": False,
        }

        async def fake_stage_commit(
            *, agent, stage_id, metric_event
        ):  # pragma: no cover - should not run
            raise AssertionError(
                "duplicate capture without semantic source change should be skipped"
            )

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S02" not in solver.stage_snapshots
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        events = [row["event"] for row in rows]
        assert "stage_duplicate_submission_skipped" in events
        assert "stage_duplicate_submission_metric_stage_kept" not in events
        assert "stage_captured" not in events
        skipped = [
            row for row in rows if row["event"] == "stage_duplicate_submission_skipped"
        ][-1]
        assert (
            skipped["payload"]["stage_policy"]
            == "duplicate_candidate_no_semantic_workspace_change"
        )

    asyncio.run(_run())


def test_lhr_duplicate_submission_same_design_is_skipped_even_with_source_change(
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
            force_estra_capture_duplicate_submissions=True,
            force_estra_after_stage_count=2,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
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
                {
                    "submission_sha": "same",
                    "solution_sha": "same-source",
                    "artifact_sha": "same",
                    "candidate_ready": True,
                },
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver.last_captured_solution_sha = ""
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.duplicate_submission_skip_keys = set()
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
            "metric_value": 0.07,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "same-source",
            "submission_sha": "same",
            "artifact_sha": "same",
            "solution_path": "predict.py",
            "validation_ok": True,
            "selection_eligible": True,
            "selection_score": 0.07,
            "source_changed": True,
        }

        async def fake_stage_commit(
            *, agent, stage_id, metric_event
        ):  # pragma: no cover - should not run
            raise AssertionError("same design duplicate should not be captured")

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S02" not in solver.stage_snapshots
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        skipped = [
            row for row in rows if row["event"] == "stage_duplicate_submission_skipped"
        ][-1]
        assert skipped["payload"]["semantic_source_changed"] is True
        assert skipped["payload"]["same_design_duplicate"] is True
        assert skipped["payload"]["stage_policy"] == "duplicate_candidate_same_design"
        assert "stage_captured" not in [row["event"] for row in rows]

    asyncio.run(_run())


def test_lhr_duplicate_artifact_without_submission_is_skipped(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=False,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md", "artifacts/*.json"],
        )
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                2.617322,
                "radii_sum",
                False,
                1,
                {
                    "artifact_sha": "same-artifact",
                    "artifact_path": "artifacts/best_solution.json",
                    "candidate_ready": True,
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
            "### S01\nmetric: 2.617322\nlower_is_better: false\nBRIEF: base\nWHY: ok\n",
            encoding="utf-8",
        )
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 2.617322,
            "metric_name": "radii_sum",
            "lower_is_better": False,
            "artifact_sha": "same-artifact",
            "artifact_path": "artifacts/best_solution.json",
            "validation_ok": True,
            "selection_eligible": True,
            "source_changed": True,
        }

        async def fake_stage_commit(
            *, agent, stage_id, metric_event
        ):  # pragma: no cover - should not run
            raise AssertionError(
                "duplicate artifact without submission should not be captured"
            )

        solver._ephemeral_stage_commit = fake_stage_commit
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        skipped = [
            row for row in rows if row["event"] == "stage_duplicate_submission_skipped"
        ][-1]
        assert skipped["payload"]["artifact_sha"] == "same-artifact"
        assert (
            skipped["payload"]["stage_policy"]
            == "duplicate_candidate_same_artifact_different_source"
        )
        assert "stage_captured" not in [row["event"] for row in rows]

    asyncio.run(_run())
