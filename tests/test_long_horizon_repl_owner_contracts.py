"""Long Horizon REPL contracts exercised directly through component owners."""
# ruff: noqa: F405 - this split contract consumes the shared fixture API.

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403

from scienceflow.agent.factory import AgentFactory
from scienceflow.research.quality.finalization import (
    multi_worker_failure_kind,
    multi_worker_stop_reason,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.run import agent_coordination, coordination


def test_lhr_make_agent_keeps_normal_and_slow_bash_caps() -> None:
    owner = SimpleNamespace(
        deadline=lnr_solver_module.time.monotonic() + 6900,
        worker_id="",
        worker_index=0,
        worker_count=1,
        worker_extra_env={},
        memory_dir=Path("memory"),
        log_dir=Path("logs"),
        task_root_dir=Path("task"),
        ledger_filename=".run_results.md",
        resource_observer=None,
        evaluation_service=object(),
        skill_registry=None,
        skill_task_category="",
        skill_allow_names=(),
        skill_tool_mode="category_only",
        skill_allow_generic_wildcard=False,
        skill_visible_max=1,
    )
    owner.lhr = SimpleNamespace(
        max_steps=10,
        workspace_git_enabled=False,
        workspace_git_track_globs=None,
        workspace_git_auto_review=False,
        resource_bash_hard_fuse_finalization_reserve_sec=900,
        clean_repl_mode=False,
        compact_on_context_limit=True,
    )
    owner.cfg = SimpleNamespace(
        exp_id="test",
        repl_bash_max_output_chars=8000,
        repl_bash_max_stream_line_chars=2400,
        mlebench_data_root_dir="",
    )
    bash_tool = SimpleNamespace(
        bash_timeout_sec=300,
        bash_timeout_slow_sec=600,
    )
    agent = SimpleNamespace(
        availableTools=SimpleNamespace(tool_map={"bash": bash_tool}),
        _system_prompt_core="",
        systemPrompt="",
    )
    owner.orchestrator = SimpleNamespace(
        make_llm_call_tracer=lambda **_kwargs: None,
        create_science_agent=lambda **_kwargs: agent,
    )
    owner._worker_llm_stage_override = lambda: None
    owner._task_runtime_extra_env = lambda: {}
    owner._code_organization_hint = lambda: ""
    owner._attach_lnr_interaction_logger = lambda _agent: None
    owner._sanitize_agent_prompt_surfaces = lambda _agent: None
    owner._restore_protected_eda_prefix_marker = lambda _agent: None
    owner._evaluator_stage_source_mode = lambda: "primary"
    owner._evaluator_task_profile = lambda: ""
    owner._evaluator_backend_name = lambda: ""
    owner._evaluator_candidate_artifact = lambda: "submission.csv"
    owner._worker_uid_prefix = lambda: "W00"
    owner._lnr_skill_hint = lambda: ""
    owner._agent_factory_service = lambda: AgentFactory(
        owner.orchestrator.create_science_agent
    )
    for callback_name in (
        "_archive_candidate_artifact_after_tool",
        "_stage_capture_callback",
        "_metric_output_interpretation_callback",
        "_text_only_estra_callback",
        "_context_limit_estra_callback",
        "_record_context_compact_event",
    ):
        setattr(owner, callback_name, lambda *_args, **_kwargs: None)

    built = agent_coordination._make_agent(owner, load_existing_memory=False)

    assert bash_tool.bash_timeout_sec == 300
    assert bash_tool.bash_timeout_slow_sec == 600
    assert built._bash_timeout_sec == 300
    assert built._bash_timeout_slow_sec == 600


def test_multi_worker_partial_merge_does_not_mask_all_worker_failure(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        owner = SimpleNamespace(
            lhr=SimpleNamespace(num_workers=2, merge_enabled=True),
            root_dir=tmp_path / "root",
            log_dir=tmp_path / "logs",
            solver_name="lnr",
            ledger_filename="run_results.md",
            cfg=SimpleNamespace(submission_dir=None),
        )
        statuses: list[tuple[str, dict[str, object]]] = []
        events: list[dict[str, object]] = []
        worker_results = [
            {
                "worker_id": "W00",
                "worker_index": 0,
                "status": "failed",
                "error": "ValueError: Separator is found, but chunk is longer than limit",
            },
            {
                "worker_id": "W01",
                "worker_index": 1,
                "status": "failed",
                "error": "ReadError: stream disconnected",
            },
        ]
        candidate = {"candidate_id": "W01:S01", "metric_value": 0.5}

        owner.state_machine = SimpleNamespace(
            mark_run_status=lambda status, payload=None: statuses.append(
                (status, dict(payload or {}))
            )
        )
        owner._jsonl = lambda _name, record: events.append(dict(record))

        async def run_one_worker(
            worker_index: int, _worker_count: int
        ) -> dict[str, object]:
            return dict(worker_results[worker_index])

        owner._run_one_worker = run_one_worker
        owner._aggregate_worker_state = lambda **_kwargs: None
        owner._load_worker_candidates = lambda result, **_kwargs: (
            [candidate] if result.get("worker_id") == "W01" else []
        )
        owner._write_stage_collection_outputs = lambda **_kwargs: tmp_path / "merge"
        owner._write_global_time_trace = lambda _worker_results: None
        owner._cleanup_coordinator_workspace_shell = lambda: None
        owner._merge_dir = lambda: tmp_path / "merge"

        async def write_merge_outputs(**_kwargs: object) -> dict[str, object]:
            return dict(candidate)

        async def no_op_async(*_args: object, **_kwargs: object) -> None:
            return None

        owner._write_merge_outputs = write_merge_outputs
        owner._final_artifact_mode = lambda: "workspace"
        owner._aggregate_worker_state_periodically = no_op_async
        owner._close_merge_owner_agent = no_op_async
        owner._evaluator_candidate_artifact = lambda: "submission.csv"
        owner._worker_root = lambda index: tmp_path / f"worker-{index}"
        owner._refresh_submission_links = lambda **_kwargs: None
        owner._multi_worker_failure_kind = multi_worker_failure_kind

        def project_stop_reason(*, run_succeeded, worker_results):
            if run_succeeded:
                return multi_worker_stop_reason(
                    run_succeeded=True,
                    worker_results=worker_results,
                )
            return "", multi_worker_failure_kind(worker_results)[1]

        owner._multi_worker_stop_reason = project_stop_reason

        result = await coordination._run_multi_worker(owner)

        assert result["status"] == "failed"
        assert result["stop_reason"] == ""
        assert result["failure_kind"] == "worker_failed_mixed"
        assert result["worker_error_kinds"] == [
            "llm_transport_error",
            "tool_output_limit",
        ]
        assert result["partial_merge_available"] is True
        assert "selected_candidate_id" not in result
        assert result["merge_enabled"] is True
        assert statuses[-1][0] == "failed"
        assert statuses[-1][1]["partial_merge_available"] is True
        done = [event for event in events if event.get("event") == "multi_worker_done"]
        assert done and done[-1]["status"] == "failed"

    asyncio.run(_run())
