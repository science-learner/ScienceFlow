"""Long Horizon Repl contracts: stage."""

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403

from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
    _stage_run_signature,
)


def test_lhr_state_machine_aggregate_global_best_ignores_unverified_stage_event(tmp_path: Path) -> None:
    out = tmp_path / "logs"
    w0 = tmp_path / "w00" / "logs"
    store = LHRStateMachineStore(log_dir=w0, worker_id="W00", worker_index=0, worker_count=1)
    store.mark_run_status("finished")
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S01",
        status="succeeded",
        payload={
            "stage_id": "S01",
            "snapshot_id": "S01-raw",
            "metric_event": {
                "metric_value": 0.01,
                "metric_name": "solver_score",
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": False,
                "selection_eligible": False,
                "metric_validity": "high",
            },
        },
    )
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S02",
        status="succeeded",
        payload={
            "stage_id": "S02",
            "snapshot_id": "S02-valid",
            "metric_event": {
                "metric_value": 0.08,
                "metric_name": "solver_score",
                "lower_is_better": True,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "medium",
                "evaluator_backend": "artifact_command",
                "evaluator_status": "ok",
                "submission_status": "ready",
            },
        },
    )

    state = LHRStateMachineStore.aggregate_logs(
        output_log_dir=out,
        input_log_dirs=[w0],
        worker_count=1,
        run_status="finished",
        ledger_filename="run_results.md",
    )

    assert state["global_best"]["stage_id"] == "S02"
    assert state["global_best"]["metric_value"] == 0.08
    assert state["global_best"]["validity"] == "valid_comparable"

def test_lhr_state_machine_aggregate_backfills_stage_performance_csv(tmp_path: Path) -> None:
    out = tmp_path / "logs"
    out.mkdir(parents=True, exist_ok=True)
    w0 = tmp_path / "w00" / "logs"
    store = LHRStateMachineStore(log_dir=w0, worker_id="W00", worker_index=0, worker_count=1)
    store.mark_run_status("finished")
    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S99",
        status="succeeded",
        payload={
            "stage_id": "S99",
            "snapshot_id": "S99-raw",
            "metric_event": {
                "metric_value": 1.0,
                "metric_name": "Final Validation Score",
                "lower_is_better": False,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
            },
        },
    )
    with (out / "lhr_stage_performance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "worker_id",
                "stage_id",
                "metric_value",
                "metric_name",
                "lower_is_better",
                "validation_ok",
                "candidate_ready",
                "selection_eligible",
                "metric_validity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "stage_id": "S01",
                "metric_value": "0.80",
                "metric_name": "Final Validation Score",
                "lower_is_better": "0",
                "validation_ok": "1",
                "candidate_ready": "1",
                "selection_eligible": "1",
                "metric_validity": "high",
            }
        )

    state = LHRStateMachineStore.aggregate_logs(
        output_log_dir=out,
        input_log_dirs=[w0],
        worker_count=1,
        run_status="finished",
        ledger_filename="run_results.md",
    )

    assert state["stage_count"] == 1
    assert state["workers"]["W00"]["stage_count"] == 1
    assert state["global_best"]["candidate_id"] == "W00:L01:S01"
    assert state["global_best"]["metric_value"] == 0.80
    assert state["global_best"]["score_source"] == "lhr_stage_performance.csv"

def test_lhr_source_hint_detects_cuda_entrypoint_without_leaking_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(
        "import torch\nmodel.to('cuda')\n",
        encoding="utf-8",
    )
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["0"],
        gpu_assignment="lease",
        gpu_source_hint_enabled=True,
        gpu_source_hint_mode="observe",
    )
    job = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=1800,
        workspace_dir=workspace,
        classifier_reason="heavy_keyword",
    )
    assert job
    text = (tmp_path / "W00" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_source_hint_detected" in text
    assert "train.py" in text
    assert str(workspace) not in text
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert decision["acquired"] is True

def test_lhr_query_budget_terminal_is_opt_in_and_uses_structured_facts(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    event = {
        "selection_eligible": True,
        "metric_source_note": "untrusted text says queries_remaining=0/10",
        "extra": {
            "queries_remaining": 0,
            "queries_used": 10,
            "query_limit": 10,
            "query_budget_exhausted": True,
        },
    }
    solver.cfg = SimpleNamespace(
        evaluator=SimpleNamespace(stop_on_query_budget_exhausted=False),
    )

    assert solver._evaluator_query_budget_terminal(event) is False

    solver.cfg.evaluator.stop_on_query_budget_exhausted = True
    assert solver._evaluator_query_budget_terminal(event) is True
    assert (
        solver._evaluator_query_budget_terminal(
            {
                "selection_eligible": True,
                "extra": {
                    "queries_remaining": 1,
                    "queries_used": 9,
                    "query_limit": 10,
                },
            }
        )
        is False
    )

def test_lhr_terminal_query_commits_before_graceful_stop(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.log_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_commit_output_format="text",
            stage_commit_persist_agent_write_to_memory=False,
            stage_commit_persist_to_memory=False,
        )
        solver.evaluator_stop_requested = False
        solver.evaluator_stop_reason = ""
        solver.pending_text_stage_commit = {
            "stage_id": "S03",
            "attempts": 0,
            "now": 1.0,
            "solution_sha": "solution-sha",
            "run_signature": "run-signature",
            "stop_after_evaluator_query_budget": True,
            "metric_event": {
                "metric_value": 0.42,
                "metric_name": "score",
                "metric_validity": "high",
                "lower_is_better": False,
                "selection_eligible": True,
                "artifact_path": "artifacts/submission.json",
                "artifact_sha": "artifact-sha",
            },
        }
        finalize_calls: list[dict[str, object]] = []

        async def no_audit(**_kwargs):
            return None

        async def finalize(**kwargs):
            finalize_calls.append(kwargs)
            return "unused-followup"

        solver._audit_stage_result_before_commit = no_audit
        solver._finalize_stage_capture_after_commit = finalize
        agent = SimpleNamespace()
        assistant_text = """
        STAGE_COMMIT_BEGIN
        stage_id: S03
        metric: 0.42
        metric_validity: high
        brief: accepted final evaluator result.
        why: preserve the last valid query before stopping.
        files: none
        STAGE_COMMIT_END
        """

        out = await solver._handle_pending_stage_commit_text(
            agent=agent,
            assistant_text=assistant_text,
        )

        assert len(finalize_calls) == 1
        assert finalize_calls[0]["allow_followup"] is False
        assert solver.pending_text_stage_commit is None
        assert solver.evaluator_stop_requested is True
        assert solver.evaluator_stop_reason == "evaluator_query_budget_exhausted"
        assert agent._lnr_graceful_stop_requested is True
        assert "EVALUATOR_QUERY_BUDGET_EXHAUSTED" in out

    asyncio.run(_run())

def test_lhr_protected_eda_prefix_stops_before_s01_stage_commit(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        preserve_prefix_and_eda=True,
        protected_eda_mode="facts",
        protected_eda_facts_max_chars=6000,
        protected_eda_warn_chars=50_000,
    )
    messages = [
        Message.user_message("FIRST USER QUERY"),
        Message.assistant_message("inspect the dataset"),
        Message.tool_message("train shape is (100, 8)", "bash", "eda-1"),
    ]

    class _MemoryContext:
        end_index = None

        def replace_protected_raw_prefix_with_summary(
            self,
            end_index: int,
            summary: str,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.end_index = end_index
            messages[:] = [Message.user_message(summary)] + messages[end_index:]
            return {
                "end_index": 1,
                "message_count": 1,
                "chars": len(summary),
                "original_chars": 100,
                "original_message_count": end_index,
            }

    ctx = _MemoryContext()
    agent = SimpleNamespace(_memory_ctx=ctx)
    solver._agent_memory_messages = lambda _agent: list(messages)
    solver._count_memory_records = lambda: len(messages)
    solver._capture_s01_eda_prefix_end(stage_id="S01")
    messages.append(
        Message.assistant_message(
            "STAGE_COMMIT_BEGIN\nstage_id: S01\nmetric: 0.2\nSTAGE_COMMIT_END"
        )
    )

    info = solver._mark_protected_eda_prefix(agent, stage_id="S01")

    assert ctx.end_index == 3
    assert info["boundary_source"] == "pre_stage_commit"
    assert info["protected_end_index"] == 1
    assert solver._s01_eda_prefix_end_index == 1
    assert len(messages) == 2
    assert "Fixed EDA facts" in str(messages[0].content)
    assert "STAGE_COMMIT_BEGIN" in str(messages[1].content)

def test_lhr_agent_eda_summary_is_opt_in_and_replaces_only_eda_prefix(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Rank scientific candidates."
        solver.lhr = SimpleNamespace(
            preserve_prefix_and_eda=True,
            protected_eda_mode="agent",
            protected_eda_facts_max_chars=6000,
            protected_eda_summary_max_chars=8000,
            protected_eda_warn_chars=50_000,
            stage_commit_llm_timeout_sec=30,
        )
        solver.stage_tokens_in = 0
        solver.stage_tokens_out = 0
        solver.stage_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver._s01_agent_eda_summary = ""
        messages = [
            Message.user_message("Rank scientific candidates."),
            Message.assistant_message("I will inspect source and candidate distributions."),
            Message.tool_message(
                "source shape is (100, 8); candidate shape is (20, 8)",
                "bash",
                "eda-1",
            ),
        ]

        class _LLM:
            _last_call_input_tokens = 100
            _last_call_output_tokens = 50
            _last_call_input_cached_tokens = 0

            async def ask(self, **kwargs):
                assert kwargs["stream"] is False
                assert "AUTHORITATIVE S01 STAGE CARD" in kwargs["messages"][0].content
                return "## Data contract\nRank 20 candidates.\n\n## Observed data facts\nSource has 100 rows."

        class _MemoryContext:
            summary = ""
            end_index = None

            def replace_protected_raw_prefix_with_summary(self, end_index, summary, **_kwargs):
                self.end_index = end_index
                self.summary = summary
                return {
                    "end_index": 1,
                    "message_count": 1,
                    "chars": len(summary),
                    "original_chars": 100,
                    "original_message_count": 2,
                }

        ctx = _MemoryContext()
        agent = SimpleNamespace(
            llm=_LLM(),
            _memory_ctx=ctx,
            _record_llm_call=lambda *_args, **_kwargs: None,
        )
        agent.run_ephemeral_agentic_route_prompt = MethodType(
            _fake_ephemeral_route_prompt, agent
        )
        solver._agent_memory_messages = lambda _agent: list(messages)
        solver._count_memory_records = lambda: len(messages)
        solver._capture_s01_eda_prefix_end(stage_id="S01")
        messages.append(
            Message.assistant_message(
                "STAGE_COMMIT_BEGIN\nstage_id: S01\nSTAGE_COMMIT_END",
            ),
        )

        await solver._prepare_protected_eda_agent_summary(agent, stage_id="S01")
        info = solver._mark_protected_eda_prefix(agent, stage_id="S01")

        assert ctx.end_index == 3
        assert "## Data contract" in ctx.summary
        assert info["mode"] == "agent"
        assert solver.stage_llm_calls == 1

    asyncio.run(_run())

def test_lhr_agent_eda_summary_falls_back_to_facts(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        preserve_prefix_and_eda=True,
        protected_eda_mode="agent",
        protected_eda_facts_max_chars=6000,
        protected_eda_summary_max_chars=8000,
        protected_eda_warn_chars=50_000,
    )
    solver._s01_agent_eda_summary = ""
    messages = [
        Message.user_message("FIRST USER QUERY"),
        Message.tool_message("train shape is (100, 8)", "bash", "eda-1"),
    ]

    class _MemoryContext:
        summary = ""

        def replace_protected_raw_prefix_with_summary(self, end_index, summary, **_kwargs):
            self.summary = summary
            return {
                "end_index": 1,
                "message_count": 1,
                "chars": len(summary),
                "original_chars": 100,
                "original_message_count": 1,
            }

    ctx = _MemoryContext()
    solver._agent_memory_messages = lambda _agent: list(messages)
    solver._count_memory_records = lambda: len(messages)
    solver._s01_eda_prefix_end_index = len(messages)

    info = solver._mark_protected_eda_prefix(
        SimpleNamespace(_memory_ctx=ctx),
        stage_id="S01",
    )

    assert info["mode"] == "facts_fallback"
    assert "train shape is (100, 8)" in ctx.summary

def test_lhr_agent_eda_summary_provider_error_uses_facts_fallback(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Rank candidates."
        solver.lhr = SimpleNamespace(
            protected_eda_mode="agent",
            protected_eda_summary_max_chars=8000,
            stage_commit_llm_timeout_sec=30,
        )
        solver._s01_eda_prefix_end_index = 2

        class _LLM:
            async def ask(self, **_kwargs):
                raise RuntimeError("provider unavailable")

        agent = SimpleNamespace(
            llm=_LLM(),
            _record_llm_call=lambda *_args, **_kwargs: None,
        )
        agent.run_ephemeral_agentic_route_prompt = MethodType(
            _fake_ephemeral_route_prompt, agent
        )
        events = []
        solver._jsonl = lambda _name, record: events.append(record)
        solver._agent_memory_messages = lambda _agent: [
            Message.user_message("task"),
            Message.tool_message("shape=(100, 8)", "bash", "eda"),
        ]

        await solver._prepare_protected_eda_agent_summary(agent, stage_id="S01")

        assert solver._s01_agent_eda_summary == ""
        assert events[-1]["fallback"] == "facts"

    asyncio.run(_run())

def test_lhr_agent_eda_summary_propagates_cancellation(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.task_desc = "Rank candidates."
        solver.lhr = SimpleNamespace(
            protected_eda_mode="agent",
            protected_eda_summary_max_chars=8000,
            stage_commit_llm_timeout_sec=30,
        )
        solver._s01_eda_prefix_end_index = 1

        class _LLM:
            async def ask(self, **_kwargs):
                raise asyncio.CancelledError()

        agent = SimpleNamespace(llm=_LLM())
        agent.run_ephemeral_agentic_route_prompt = MethodType(
            _fake_ephemeral_route_prompt, agent
        )
        solver._agent_memory_messages = lambda _agent: [Message.user_message("task")]

        with pytest.raises(asyncio.CancelledError):
            await solver._prepare_protected_eda_agent_summary(agent, stage_id="S01")

    asyncio.run(_run())

def test_lhr_keep_current_strict_context_limit_uses_minimal_prefix(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(estra_compact_enabled=True)
    solver.cfg = SimpleNamespace(max_messages=240)
    solver.memory_dir = tmp_path / "memory"
    snap_dir = tmp_path / "snapshots" / "S01-test"
    agent_dir = snap_dir / ".memory" / "ScienceAgent"
    agent_dir.mkdir(parents=True)
    records = [
        {"message": {"role": "system", "content": "SYSTEM PREFIX"}, "role": "system"},
        MemoryCompactor.user_record("FIRST TASK PROMPT"),
    ]
    for idx in range(80):
        records.append({"message": {"role": "assistant", "content": f"assistant noise {idx}"}, "role": "assistant"})
        records.append({"message": {"role": "tool", "content": f"tool noise {idx}"}, "role": "tool"})
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
            memory_cut=len(records),
            source_event={},
        )
    }

    ok, n_records = solver._rebuild_memory_after_keep_current(
        terminal_stage="S02",
        summary="S02 compact summary",
        state_packet="LHR State Packet v2",
        strict_context_limit=True,
    )

    assert ok
    assert n_records == 3
    written = (solver.memory_dir / "ScienceAgent" / "short_term.json").read_text(encoding="utf-8")
    assert "SYSTEM PREFIX" in written
    assert "FIRST TASK PROMPT" in written
    assert "assistant noise" not in written
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    ctx = [row for row in rows if row["event"] == "estra_keep_current_context_prepared"][-1]
    assert ctx["payload"]["compact_strength"] == "strict_context_limit"
    assert "minimal task prefix records" in ctx["payload"]["context_shape"][0]

def test_lhr_context_compact_events_are_unified_and_stage_scoped(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.stage_snapshots = {"S01": object(), "S03": object()}

    solver._record_context_compact_event(
        phase="started",
        mode="inband",
        reason="context_limit",
        round=4,
        omitted_before=9,
        recent_keep=0,
    )
    solver._record_context_compact_event(
        phase="finished",
        mode="inband",
        status="ok",
        round=4,
        omitted_after=0,
        summary_chars=1234,
        tokens_in=100,
        tokens_out=10,
        tokens_cached=95,
    )

    assert not (solver.log_dir / "lhr_context_events.jsonl").exists()
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    events = [row for row in rows if row["event"].startswith("context_compact_")]
    assert [row["event"] for row in events] == [
        "context_compact_started",
        "context_compact_finished",
    ]
    assert events[0]["task_id"] == "repl_search:W00:S02"
    assert events[0]["payload"]["latest_stage"] == "S02"
    assert events[0]["payload"]["next_stage"] == "S03"
    assert events[0]["status"] == "running"
    assert events[1]["status"] == "succeeded"
    assert events[1]["payload"]["summary_chars"] == 1234

def test_lhr_terminal_query_bypasses_stage_cap_then_commits_and_stops(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.log_dir.mkdir(parents=True, exist_ok=True)
        solver.deadline = 10.0
        solver.cfg = SimpleNamespace(
            evaluator=SimpleNamespace(stop_on_query_budget_exhausted=True),
        )
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_capture_max_count=2,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            stage_commit_output_format="text",
            stage_commit_llm_timeout_sec=60,
            stage_commit_persist_agent_write_to_memory=False,
            stage_commit_persist_to_memory=False,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {}
        solver.evaluator_stop_requested = False
        solver.evaluator_stop_reason = ""
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            "metric_value": 0.42,
            "metric_name": "score",
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "artifact_path": "submission.csv",
            "artifact_sha": "ready-sha",
            "evaluator_backend": "task_package",
            "evaluator_status": "ok",
            "gate_action": "accept",
            "gate_accepted": True,
            "gate_reason_code": "eligible",
            "extra": {
                "queries_used": 10,
                "queries_remaining": 0,
                "query_limit": 10,
                "query_budget_exhausted": True,
            },
            "_gate_evaluated": True,
            "_evaluator_stage_facts_applied": True,
        }
        finalize_calls: list[dict[str, object]] = []

        async def no_audit(**_kwargs):
            return None

        async def finalize(**kwargs):
            finalize_calls.append(kwargs)
            return None

        solver._audit_stage_result_before_commit = no_audit
        solver._finalize_stage_capture_after_commit = finalize
        agent = SimpleNamespace(_run_policy=SimpleNamespace(deadline_monotonic=10.0))

        callback_out = await LnrSolver._stage_capture_callback(
            solver,
            agent=agent,
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert callback_out is None
        assert solver.pending_text_stage_commit["stage_id"] == "S03"
        assert solver.pending_text_stage_commit["stop_after_evaluator_query_budget"] is True

        stop_out = await solver._handle_pending_stage_commit_text(
            agent=agent,
            assistant_text="""
            STAGE_COMMIT_BEGIN
            stage_id: S03
            metric: 0.42
            metric_validity: high
            brief: preserve the final accepted query.
            why: the evaluator query budget is now exhausted.
            files: none
            STAGE_COMMIT_END
            """,
        )

        assert len(finalize_calls) == 1
        assert finalize_calls[0]["allow_followup"] is False
        assert "### S03" in solver.ledger_path.read_text(encoding="utf-8")
        assert solver.evaluator_stop_requested is True
        assert "EVALUATOR_QUERY_BUDGET_EXHAUSTED" in stop_out
        events = (solver.log_dir / "lhr_events.jsonl").read_text(encoding="utf-8")
        assert "stage_capture_cap_reached" not in events

    asyncio.run(_run())

def test_lhr_primary_duplicate_artifact_is_skipped_before_gate(tmp_path: Path) -> None:
    async def _run() -> None:
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
        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        skipped = [row for row in rows if row["event"] == "duplicate_candidate_pre_gate_skipped"]
        assert skipped[-1]["payload"]["duplicate_of_stage"] == "S01"

    asyncio.run(_run())

def test_lhr_run_signature_includes_cli_hyperparameters() -> None:
    base = {
        "metric_value": 0.061,
        "metric_name": "Final Validation Score",
        "solution_sha": "same-source",
        "submission_sha": "same-submission",
    }
    sig_lr_007 = _stage_run_signature(
        {**base, "bash_cmd": "python3 train.py --lr 0.07 --depth 6"}
    )
    sig_lr_003 = _stage_run_signature(
        {**base, "bash_cmd": "python3 train.py --lr 0.03 --depth 6"}
    )
    sig_env_007 = _stage_run_signature(
        {**base, "bash_cmd": "LR=0.07 python3 train.py --depth 6"}
    )
    sig_env_003 = _stage_run_signature(
        {**base, "bash_cmd": "LR=0.03 python3 train.py --depth 6"}
    )
    assert sig_lr_007 != sig_lr_003
    assert sig_env_007 != sig_env_003

def test_lhr_stage_commit_persists_agent_write_without_controller_prompt(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            stage_commit_max_turns=1,
            stage_commit_llm_timeout_sec=17,
            stage_commit_persist_to_memory=False,
            stage_commit_persist_agent_write_to_memory=True,
            stage_commit_persist_prompt_to_memory=False,
        )
        solver.stage_tokens_in = 0
        solver.stage_tokens_out = 0
        solver.stage_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        (solver.workspace_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        class FakeLLM:
            _last_call_input_tokens = 50
            _last_call_output_tokens = 4
            _last_call_input_cached_tokens = 45

            def __init__(self) -> None:
                self.timeout_seen = None
                self.tool_choice_seen = None
                self.tools_seen = None

            async def ask_tool_stream(self, **kwargs):
                self.timeout_seen = kwargs.get("timeout")
                self.tool_choice_seen = kwargs.get("tool_choice")
                self.tools_seen = kwargs.get("tools")
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "brief": "main-context LightGBM metadata experiment",
                            "why": "preserves a comparable validation route judgment",
                            "route_evidence": "verdict=continue; reason=useful cheap baseline; next=blend",
                            "files": "code=train.py",
                        }
                    ),
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="tc-hallucinated",
                            function=SimpleNamespace(name="read", arguments=json.dumps({"path": "train.py"})),
                        )
                    ],
                )

        class FakeMemoryCtx:
            def build_messages_for_llm(self):
                return [Message.user_message("FIRST USER")]

        class FakeMemory:
            def __init__(self) -> None:
                self.messages = []

            def add_message(self, msg):
                self.messages.append(msg)

        class FakeTools:
            async def execute(self, *, name, tool_input):  # pragma: no cover - stage commit must not execute tools
                raise AssertionError("stage commit fork must not execute tools")

        class FakeAgent:
            run_ephemeral_agentic_route_prompt = _fake_ephemeral_route_prompt

            def __init__(self) -> None:
                self.llm = FakeLLM()
                self._memory_ctx = FakeMemoryCtx()
                self.memory = FakeMemory()
                self.availableTools = FakeTools()
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
        ok, reason = await LnrSolver._ephemeral_stage_commit(
            solver,
            agent=agent,
            stage_id="S03",
            metric_event={
                "metric_value": 0.06,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "run_time_sec": 12.0,
                "val_score_type": "cv",
                "metric_validity": "high",
            },
        )
        assert ok, reason
        assert agent.llm.timeout_seen == 17
        assert agent.llm.tool_choice_seen == "none"
        assert agent.llm.tools_seen == []
        ledger = solver.ledger_path.read_text(encoding="utf-8")
        assert "### S03" in ledger
        assert "metric: 0.06" in ledger
        assert "BRIEF: main-context LightGBM metadata experiment" in ledger
        assert "WHY: preserves a comparable validation route judgment" in ledger
        assert len(agent.memory.messages) == 1
        assert "Forked stage bookkeeping judgment" not in str(agent.memory.messages[0].content)
        assert not getattr(agent.memory.messages[0], "tool_calls", None)
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        persisted = [row for row in rows if row["event"] == "stage_commit_persisted_to_memory"][-1]
        assert persisted["payload"]["agent_write_persisted"] is True
        assert persisted["payload"]["prompt_persisted"] is False
        assert persisted["payload"]["deterministic_append"] is True

    asyncio.run(_run())
