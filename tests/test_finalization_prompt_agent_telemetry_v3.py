from __future__ import annotations

import json
from pathlib import Path

import pytest

from scienceflow.agent.factory import AgentFactory
from scienceflow.foundation.architecture.dependencies import find_dependency_violations
from scienceflow.research.quality.finalization import FinalizationService
from scienceflow.research.quality.finalization.selection.ranking import ranked_candidates
from scienceflow.research.quality.finalization.worker_outcomes import (
    multi_worker_failure_kind,
    multi_worker_stop_reason,
)
from scienceflow.research.state.knowledge.prompt import PromptContextBuilder, PromptContextRequest
from scienceflow.runtime.core.kernel import (
    HookFailureMode,
    HookOutcome,
    HookPoint,
    HookTrace,
)
from scienceflow.research.solver.lnr.support.prompts import build_first_user_prompt
from scienceflow.runtime.observability.telemetry import CorrelationJournal, HookTraceCorrelationSink


def test_finalization_solver_compatibility_modules_stay_deleted() -> None:
    lnr_root = Path(__file__).parents[1] / "scienceflow" / "research" / "solver" / "lnr"
    assert not (lnr_root / "global_merge").exists()
    assert not (lnr_root / "submission_links.py").exists()
    assert not (lnr_root / "runtime" / "finalization.py").exists()


def test_finalization_service_preserves_worker_reduction_and_fallback_order() -> None:
    service = FinalizationService()
    workers = [
        {"status": "failed", "error": "TimeoutError: worker"},
        {"status": "failed", "error": "TimeoutError: worker"},
    ]
    assert service.multi_worker_failure_kind(workers) == (
        "worker_timeout",
        ["worker_timeout"],
    )
    assert multi_worker_failure_kind(workers) == service.multi_worker_failure_kind(workers)
    assert multi_worker_stop_reason(run_succeeded=False, worker_results=workers) == (
        "worker_timeout",
        ["worker_timeout"],
    )
    candidates = [
        {
            "candidate_id": "b",
            "metric_value": 0.8,
            "lower_is_better": True,
            "candidate_ready": True,
            "artifact_sha": "b" * 64,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
        },
        {
            "candidate_id": "a",
            "metric_value": 0.2,
            "lower_is_better": True,
            "candidate_ready": True,
            "artifact_sha": "a" * 64,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
        },
    ]
    assert [row["candidate_id"] for row in ranked_candidates(candidates)] == ["a", "b"]


def test_prompt_context_projection_is_deterministic_and_render_byte_compatible() -> None:
    request = PromptContextRequest(
        task_description="Solve the task.",
        worker_id="W01",
        wall_clock_budget_sec=600,
        seed=2222,
        workspace_facts="workspace-state",
        parallel_worker_facts="peer-state",
        resource_observation="resource-state",
        gate_constraints="submission.csv required",
        skill_context="use skill X",
        tool_output_policy="stdout policy",
        task_profile="mlebench",
    )
    builder = PromptContextBuilder()
    first = builder.project(request)
    second = builder.project(request)

    assert first == second
    assert len(first.projection_hash) == 64
    projected_text = build_first_user_prompt(
        first.task_description,
        wall_clock_budget_sec=first.wall_clock_budget_sec,
        seed=first.seed,
        worker_id=first.worker_id,
        parallel_worker_snapshot=first.parallel_worker_facts,
        resource_context=first.resource_observation,
        skill_hint=first.skill_context,
        evaluator_contract=first.gate_constraints,
        task_profile=first.task_profile,
        task_runtime_contract=first.tool_output_policy,
        initial_workspace_state=first.workspace_facts,
    )
    legacy_text = build_first_user_prompt(
        "Solve the task.",
        wall_clock_budget_sec=600,
        seed=2222,
        worker_id="W01",
        parallel_worker_snapshot="peer-state",
        resource_context="resource-state",
        skill_hint="use skill X",
        evaluator_contract="submission.csv required",
        task_profile="mlebench",
        task_runtime_contract="stdout policy",
        initial_workspace_state="workspace-state",
    )
    assert projected_text.encode() == legacy_text.encode()
    assert not hasattr(builder, "write")
    assert not hasattr(builder, "run_llm")


def test_agent_factory_forwards_options_and_soft_isolates_trace() -> None:
    calls = []

    def create(**options):
        calls.append(options)
        return {"agent": len(calls)}

    def broken_trace(_record):
        raise OSError("telemetry unavailable")

    factory = AgentFactory(create, trace_sink=broken_trace)
    first = factory.create(
        "main_agent",
        correlation={"worker_id": "W00"},
        max_steps_override=5,
    )
    second = factory.create(
        "main_agent",
        correlation={"worker_id": "W00"},
        max_steps_override=5,
    )

    assert first == {"agent": 1}
    assert second == {"agent": 2}
    assert calls == [{"max_steps_override": 5}, {"max_steps_override": 5}]
    assert [record.sequence for record in factory.records] == [1, 2]
    assert factory.records[0].build_id != factory.records[1].build_id
    assert all(record.outcome == "created" for record in factory.records)


def test_agent_factory_records_failed_construction() -> None:
    def fail(**_options):
        raise RuntimeError("cannot build")

    factory = AgentFactory(fail)
    with pytest.raises(RuntimeError, match="cannot build"):
        factory.create("feedback_agent")
    assert factory.records[0].outcome == "failed"
    assert factory.records[0].error_type == "RuntimeError"


def test_solver_coordinator_has_no_direct_orchestrator_agent_construction() -> None:
    source = (
        Path(__file__).parents[1]
        / "scienceflow"
        / "research"
        / "solver"
        / "lnr"
        / "orchestration"
        / "coordinator"
        / "run"
        / "construction.py"
    ).read_text(encoding="utf-8")
    assert "self.orchestrator.create_science_agent(" not in source


def test_correlation_journal_is_idempotent_and_reopenable(tmp_path: Path) -> None:
    path = tmp_path / "correlation.jsonl"
    journal = CorrelationJournal(path)
    first = journal.record(
        source="stage_lifecycle",
        event_type="transition",
        payload={"state": "committed"},
        run_id="run-1",
        worker_id="W00",
        stage_id="S01",
        event_id="stage:S01:commit",
        sequence=7,
    )
    replay = journal.record(
        source="stage_lifecycle",
        event_type="transition",
        payload={"state": "committed"},
        run_id="run-1",
        worker_id="W00",
        stage_id="S01",
        event_id="stage:S01:commit",
        sequence=7,
    )

    assert replay.correlation_id == first.correlation_id
    assert len(path.read_text().splitlines()) == 1
    reopened = CorrelationJournal(path)
    assert reopened.append(first) is False
    row = json.loads(path.read_text())
    assert row["schema_version"] == "1.0"
    assert row["payload_hash"] == first.payload_hash


def test_hook_trace_sink_preserves_all_correlation_ids(tmp_path: Path) -> None:
    journal = CorrelationJournal(tmp_path / "hook.jsonl")
    sink = HookTraceCorrelationSink(journal)
    trace = HookTrace(
        schema_version="1.0",
        sequence=3,
        run_id="run-1",
        worker_id="W01",
        process_id="proc-1",
        stage_id="S02",
        event_id="event-1",
        hook_name="monitor",
        hook_point=HookPoint.RUN_STARTED,
        owner_component="monitoring",
        priority=0,
        failure_mode=HookFailureMode.SOFT,
        started_at=1.0,
        finished_at=2.0,
        duration_sec=1.0,
        outcome=HookOutcome.SUCCEEDED,
    )
    sink(trace)
    row = json.loads((tmp_path / "hook.jsonl").read_text())
    assert row["run_id"] == "run-1"
    assert row["worker_id"] == "W01"
    assert row["process_id"] == "proc-1"
    assert row["stage_id"] == "S02"
    assert row["event_id"] == "event-1"
    assert row["sequence"] == 3


def test_v3_9_component_dependency_boundaries() -> None:
    root = Path(__file__).parents[1] / "scienceflow"
    violations = find_dependency_violations(root)
    relevant = {
        "finalization-does-not-own-runtime-or-policy",
        "prompt-context-is-pure-projection",
        "agent-factory-is-construction-only",
        "telemetry-is-observation-only",
    }
    assert [item.format(root) for item in violations if item.rule in relevant] == []
