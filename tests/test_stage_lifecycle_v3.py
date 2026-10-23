# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Phase V3-6 stage lifecycle sequence, failure, and replay fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
import ast
from pathlib import Path
from typing import Any

import pytest

from scienceflow.runtime.core.kernel import InvalidTransitionError
from scienceflow.runtime.core.stage import (
    StageAssessment,
    StageCandidateRequest,
    StageGateDecision,
    StageLifecycleCoordinator,
    StageLifecycleEvent,
    StageLifecycleMachine,
    StageLifecyclePorts,
    StageLifecycleState,
)


@dataclass
class RecordingWorkspace:
    calls: list[str]
    fail_commit: bool = False
    prepared: dict[str, Any] = field(default_factory=dict)

    def prepare(self, request, gate):
        self.calls.append("workspace.prepare")
        self.prepared = {"stage_id": request.stage_id, "gate": gate.reason_code}
        return self.prepared

    def commit(self, transaction):
        self.calls.append("workspace.commit")
        if self.fail_commit:
            raise RuntimeError("crash after prepare")
        return {"snapshot_id": "snap-1", "snapshot_path": "/snap/1"}

    def rollback(self, transaction, *, reason: str):
        self.calls.append("workspace.rollback")
        return {"rolled_back": True, "reason": reason}


def request(trigger_id: str = "trigger-1") -> StageCandidateRequest:
    return StageCandidateRequest(
        trigger_id=trigger_id,
        stage_id="S01",
        worker_id="W00",
        candidate_id="W00:L01:S01",
        artifact_path="submission.csv",
        artifact_sha="abc",
        metric_facts={"metric_validity": "high"},
    )


def ports(
    calls: list[str],
    *,
    gate_accepted: bool = True,
    assessment_accepted: bool = True,
    fail_commit: bool = False,
    telemetry_failure: bool = False,
) -> StageLifecyclePorts:
    def archive(candidate):
        calls.append("archive")
        return {"candidate_id": candidate.candidate_id}

    def assess(candidate, archived):
        calls.append("assess")
        return StageAssessment(
            accepted=assessment_accepted,
            reason_code="eligible" if assessment_accepted else "assessment_timeout",
            facts={"archive": archived},
        )

    def gate(candidate, assessment):
        calls.append("gate")
        return StageGateDecision(
            accepted=gate_accepted,
            action="accept" if gate_accepted else "reject",
            reason_code="eligible" if gate_accepted else "metric_invalid",
            metric_name="rmsle",
            metric_value=0.123,
            lower_is_better=True,
            metric_validity="high",
            facts={"assessment": assessment.reason_code},
        )

    def project_memory(committed):
        calls.append("memory.project")
        return {"stage_id": committed.stage_id, "memory_cut": 4}

    def observe_estra(committed, memory):
        calls.append("estra.observe")
        return {"observed": True, "memory": memory}

    def notify(committed, memory):
        calls.append("telemetry.notify")
        if telemetry_failure:
            raise RuntimeError("telemetry unavailable")
        return {"notified": True}

    return StageLifecyclePorts(
        archive=archive,
        assess=assess,
        gate=gate,
        workspace=RecordingWorkspace(calls, fail_commit=fail_commit),
        project_memory=project_memory,
        observe_estra=observe_estra,
        notify_telemetry=notify,
    )


@pytest.mark.asyncio
async def test_stage_lifecycle_successful_commit_has_fixed_order() -> None:
    calls: list[str] = []
    traces: list[dict[str, Any]] = []
    coordinator = StageLifecycleCoordinator(trace_sink=traces.append)

    result = await coordinator.execute(request(), ports(calls))

    assert result.status == "committed"
    assert result.state == "completed"
    assert result.gate is not None
    assert result.gate.reason_code == "eligible"
    assert result.gate.lower_is_better is True
    assert result.gate.metric_validity == "high"
    assert result.committed is not None
    assert result.committed.snapshot_id == "snap-1"
    assert len(result.projection_hash) == 64
    assert calls == [
        "archive",
        "assess",
        "gate",
        "workspace.prepare",
        "workspace.commit",
        "memory.project",
        "estra.observe",
        "telemetry.notify",
    ]
    assert [transition.event.value for transition in coordinator.machines[0].transitions] == [
        "detect",
        "archive",
        "assess",
        "accept_gate",
        "prepare",
        "commit_workspace",
        "commit_stage",
        "project_memory",
        "observe_estra",
        "notify_telemetry",
        "complete",
    ]
    assert [trace["current"] for trace in traces] == [
        transition.current.value for transition in coordinator.machines[0].transitions
    ]
    assert all(trace["machine_type"] == "stage_lifecycle" for trace in traces)


@pytest.mark.asyncio
async def test_stage_gate_reject_leaves_workspace_and_memory_untouched() -> None:
    calls: list[str] = []
    coordinator = StageLifecycleCoordinator()

    result = await coordinator.execute(request(), ports(calls, gate_accepted=False))

    assert result.status == "rejected"
    assert result.state == "rejected"
    assert calls == ["archive", "assess", "gate"]
    assert not any(call.startswith("workspace") for call in calls)
    assert "memory.project" not in calls


@pytest.mark.asyncio
async def test_stage_assessment_failure_does_not_call_gate_or_workspace() -> None:
    calls: list[str] = []
    result = await StageLifecycleCoordinator().execute(
        request(), ports(calls, assessment_accepted=False)
    )

    assert result.status == "assessment_failed"
    assert result.state == "failed"
    assert result.error == "assessment_timeout"
    assert calls == ["archive", "assess"]


@pytest.mark.asyncio
async def test_stage_crash_after_prepare_rolls_back_once() -> None:
    calls: list[str] = []
    coordinator = StageLifecycleCoordinator()

    result = await coordinator.execute(request(), ports(calls, fail_commit=True))

    assert result.status == "rolled_back"
    assert result.state == "rolled_back"
    assert "crash after prepare" in result.error
    assert calls == [
        "archive",
        "assess",
        "gate",
        "workspace.prepare",
        "workspace.commit",
        "workspace.rollback",
    ]


@pytest.mark.asyncio
async def test_stage_duplicate_trigger_replays_without_double_commit() -> None:
    calls: list[str] = []
    coordinator = StageLifecycleCoordinator()
    first = await coordinator.execute(request(), ports(calls))
    replay = await coordinator.execute(request(), ports(calls))

    assert first.duplicate is False
    assert replay.duplicate is True
    assert replay.projection_hash == first.projection_hash
    assert calls.count("workspace.commit") == 1
    assert calls.count("memory.project") == 1


@pytest.mark.asyncio
async def test_stage_new_trigger_can_retry_after_gate_rejection() -> None:
    calls: list[str] = []
    coordinator = StageLifecycleCoordinator()

    rejected = await coordinator.execute(
        request("trigger-rejected"), ports(calls, gate_accepted=False)
    )
    committed = await coordinator.execute(
        request("trigger-retry"), ports(calls, gate_accepted=True)
    )

    assert rejected.status == "rejected"
    assert committed.status == "committed"
    assert len(coordinator.machines) == 2
    assert [machine.state for machine in coordinator.machines] == [
        StageLifecycleState.REJECTED,
        StageLifecycleState.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_stage_telemetry_failure_is_soft_after_commit() -> None:
    calls: list[str] = []
    result = await StageLifecycleCoordinator().execute(
        request(), ports(calls, telemetry_failure=True)
    )

    assert result.status == "committed"
    assert result.state == "completed"
    assert result.committed is not None
    assert result.notification_errors == ("RuntimeError: telemetry unavailable",)


def test_stage_machine_rejects_out_of_order_transition() -> None:
    machine = StageLifecycleMachine(stage_id="S01")
    with pytest.raises(InvalidTransitionError):
        machine.advance(StageLifecycleEvent.PREPARE)
    assert machine.state is StageLifecycleState.CREATED


def test_legacy_projection_replays_prepare_rollback_and_commit_sequences() -> None:
    rollback_coordinator = StageLifecycleCoordinator()
    rolled_back = rollback_coordinator.advance_legacy(
        "S01", StageLifecycleEvent.ROLLBACK, event_id="tx-1"
    )
    assert rolled_back.state is StageLifecycleState.ROLLED_BACK
    assert [row.event.value for row in rolled_back.transitions] == [
        "detect",
        "archive",
        "assess",
        "accept_gate",
        "prepare",
        "rollback",
    ]

    commit_coordinator = StageLifecycleCoordinator()
    completed = commit_coordinator.advance_legacy(
        "S01", StageLifecycleEvent.COMPLETE, event_id="tx-2"
    )
    assert completed.state is StageLifecycleState.COMPLETED


def test_stage_lifecycle_has_one_way_dependencies_and_callback_is_adapter_only() -> None:
    root = Path(__file__).parents[1] / "scienceflow"
    forbidden = (
        "scienceflow.research.solver",
        "scienceflow.research.state.knowledge.memory",
        "scienceflow.research.control.estra",
        "scienceflow.research.quality.evaluator",
        "scienceflow.research.quality.gate",
        "scienceflow.research.state.workspace",
    )
    for path in (root / "runtime" / "stage").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not str(node.module or "").startswith(forbidden), path
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(forbidden) for alias in node.names
                ), path

    solver_path = root / "research/solver/lnr/orchestration/coordinator/evaluation/capture.py"
    solver_tree = ast.parse(solver_path.read_text(encoding="utf-8"))
    callback = next(
        node
        for node in ast.walk(solver_tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_stage_capture_callback"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(callback)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_metric_event_from_workspace" not in called_attributes
    assert "_record_evaluator_stage_events" not in called_attributes
    assert "_ephemeral_stage_commit" not in called_attributes
    assert "_finalize_stage_capture_after_commit" not in called_attributes
    assert {"new_legacy_trigger", "dispatch_legacy"} <= called_attributes
