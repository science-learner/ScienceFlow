# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Application service for the ordered candidate-to-stage workflow."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from scienceflow.runtime.core.stage.contracts import (
    LegacyStageTrigger,
    StageAssessment,
    StageCandidateRequest,
    StageCommittedEvent,
    StageGateDecision,
    StageLifecycleResult,
)
from scienceflow.runtime.core.stage.machine import (
    StageLifecycleEvent,
    StageLifecycleMachine,
    StageLifecycleState,
)


class StageWorkspacePort(Protocol):
    def prepare(self, request: StageCandidateRequest, gate: StageGateDecision) -> Any: ...
    def commit(self, transaction: Any) -> Mapping[str, Any]: ...
    def rollback(self, transaction: Any, *, reason: str) -> Any: ...


StageCallable = Callable[..., Any | Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class StageLifecyclePorts:
    archive: StageCallable
    assess: StageCallable
    gate: StageCallable
    workspace: StageWorkspacePort
    project_memory: StageCallable
    observe_estra: StageCallable
    notify_telemetry: StageCallable


_ORDER = (
    (StageLifecycleState.CREATED, StageLifecycleEvent.DETECT),
    (StageLifecycleState.DETECTED, StageLifecycleEvent.ARCHIVE),
    (StageLifecycleState.ARCHIVED, StageLifecycleEvent.ASSESS),
    (StageLifecycleState.ASSESSED, StageLifecycleEvent.ACCEPT_GATE),
    (StageLifecycleState.GATE_ACCEPTED, StageLifecycleEvent.PREPARE),
    (StageLifecycleState.PREPARED, StageLifecycleEvent.COMMIT_WORKSPACE),
    (StageLifecycleState.WORKSPACE_COMMITTED, StageLifecycleEvent.COMMIT_STAGE),
    (StageLifecycleState.STAGE_COMMITTED, StageLifecycleEvent.PROJECT_MEMORY),
    (StageLifecycleState.MEMORY_PROJECTED, StageLifecycleEvent.OBSERVE_ESTRA),
    (StageLifecycleState.ESTRA_OBSERVED, StageLifecycleEvent.NOTIFY_TELEMETRY),
    (StageLifecycleState.TELEMETRY_NOTIFIED, StageLifecycleEvent.COMPLETE),
)


class StageLifecycleCoordinator:
    """Own ordering/idempotency while each domain port owns its operation."""

    def __init__(self, *, trace_sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._machines: dict[str, StageLifecycleMachine] = {}
        self._machine_history: list[StageLifecycleMachine] = []
        self._results: dict[str, StageLifecycleResult] = {}
        self._legacy_trigger_sequence = 0
        self._trace_sink = trace_sink

    def set_trace_sink(
        self, trace_sink: Callable[[dict[str, Any]], None] | None
    ) -> None:
        self._trace_sink = trace_sink

    @property
    def machines(self) -> tuple[StageLifecycleMachine, ...]:
        return tuple(self._machine_history) + tuple(
            self._machines[key] for key in sorted(self._machines)
        )

    def machine_for(self, stage_id: str) -> StageLifecycleMachine:
        clean = str(stage_id or "").strip().upper()
        if not clean:
            raise ValueError("stage_id is required")
        return self._machines.setdefault(clean, StageLifecycleMachine(stage_id=clean))

    async def execute(
        self, request: StageCandidateRequest, ports: StageLifecyclePorts
    ) -> StageLifecycleResult:
        previous = self._results.get(request.trigger_id)
        if previous is not None:
            return replace(previous, duplicate=True)
        machine = self._machine_for_attempt(request.stage_id)
        transaction: Any = None
        gate: StageGateDecision | None = None
        try:
            self._advance(machine, StageLifecycleEvent.DETECT, request.trigger_id)
            archived = await _call(ports.archive, request)
            self._advance(machine, StageLifecycleEvent.ARCHIVE, request.trigger_id)
            assessment = await _call(ports.assess, request, archived)
            if not isinstance(assessment, StageAssessment):
                raise TypeError("assessment port must return StageAssessment")
            self._advance(machine, StageLifecycleEvent.ASSESS, request.trigger_id)
            if not assessment.accepted:
                self._advance(machine, StageLifecycleEvent.FAIL, request.trigger_id)
                return self._remember(
                    request,
                    status="assessment_failed",
                    machine=machine,
                    error=assessment.reason_code,
                )
            gate = await _call(ports.gate, request, assessment)
            if not isinstance(gate, StageGateDecision):
                raise TypeError("gate port must return StageGateDecision")
            if not gate.accepted:
                self._advance(machine, StageLifecycleEvent.REJECT_GATE, request.trigger_id)
                return self._remember(
                    request, status="rejected", machine=machine, gate=gate
                )
            self._advance(machine, StageLifecycleEvent.ACCEPT_GATE, request.trigger_id)
            transaction = await _call(ports.workspace.prepare, request, gate)
            self._advance(machine, StageLifecycleEvent.PREPARE, request.trigger_id)
            snapshot = dict(await _call(ports.workspace.commit, transaction) or {})
            self._advance(machine, StageLifecycleEvent.COMMIT_WORKSPACE, request.trigger_id)
            committed = StageCommittedEvent(
                stage_id=request.stage_id,
                candidate_id=request.candidate_id,
                snapshot_id=str(snapshot.get("snapshot_id") or ""),
                snapshot_path=str(snapshot.get("snapshot_path") or ""),
                gate=gate,
                metadata=snapshot,
            )
            self._advance(machine, StageLifecycleEvent.COMMIT_STAGE, request.trigger_id)
            memory = dict(await _call(ports.project_memory, committed) or {})
            self._advance(machine, StageLifecycleEvent.PROJECT_MEMORY, request.trigger_id)
            await _call(ports.observe_estra, committed, memory)
            self._advance(machine, StageLifecycleEvent.OBSERVE_ESTRA, request.trigger_id)
            notification_errors: list[str] = []
            try:
                await _call(ports.notify_telemetry, committed, memory)
            except Exception as exc:  # telemetry is deliberately soft failure
                notification_errors.append(f"{type(exc).__name__}: {exc}")
            self._advance(machine, StageLifecycleEvent.NOTIFY_TELEMETRY, request.trigger_id)
            self._advance(machine, StageLifecycleEvent.COMPLETE, request.trigger_id)
            return self._remember(
                request,
                status="committed",
                machine=machine,
                gate=gate,
                committed=committed,
                projection_hash=_projection_hash(memory),
                notification_errors=tuple(notification_errors),
            )
        except Exception as exc:
            if transaction is not None and machine.state is StageLifecycleState.PREPARED:
                try:
                    await _call(ports.workspace.rollback, transaction, reason=str(exc))
                finally:
                    self._advance(machine, StageLifecycleEvent.ROLLBACK, request.trigger_id)
                status = "rolled_back"
            else:
                if machine.state not in {
                    StageLifecycleState.REJECTED,
                    StageLifecycleState.ROLLED_BACK,
                    StageLifecycleState.FAILED,
                }:
                    self._advance(machine, StageLifecycleEvent.FAIL, request.trigger_id)
                status = "failed"
            return self._remember(
                request,
                status=status,
                machine=machine,
                gate=gate,
                error=f"{type(exc).__name__}: {exc}",
            )

    def advance_legacy(
        self,
        stage_id: str,
        target: StageLifecycleEvent,
        *,
        event_id: str = "",
    ) -> StageLifecycleMachine:
        """Project legacy checkpoints through the same canonical state graph."""

        machine = self.machine_for(stage_id)
        if target is StageLifecycleEvent.ASSESS and machine.state in {
            StageLifecycleState.REJECTED,
            StageLifecycleState.ROLLED_BACK,
            StageLifecycleState.FAILED,
        }:
            self._machine_history.append(machine)
            clean = str(stage_id or "").strip().upper()
            machine = StageLifecycleMachine(stage_id=clean)
            self._machines[clean] = machine
        if target is StageLifecycleEvent.REJECT_GATE:
            self._advance_to(machine, StageLifecycleEvent.ASSESS, event_id)
            if machine.state is StageLifecycleState.ASSESSED:
                self._advance(machine, target, event_id)
            return machine
        if target is StageLifecycleEvent.ROLLBACK:
            self._advance_to(machine, StageLifecycleEvent.PREPARE, event_id)
            if machine.state is StageLifecycleState.PREPARED:
                self._advance(machine, target, event_id)
            return machine
        self._advance_to(machine, target, event_id)
        return machine

    def _machine_for_attempt(self, stage_id: str) -> StageLifecycleMachine:
        machine = self.machine_for(stage_id)
        if machine.state in {
            StageLifecycleState.REJECTED,
            StageLifecycleState.ROLLED_BACK,
            StageLifecycleState.FAILED,
        }:
            self._machine_history.append(machine)
            clean = str(stage_id or "").strip().upper()
            machine = StageLifecycleMachine(stage_id=clean)
            self._machines[clean] = machine
        return machine

    def new_legacy_trigger(
        self, *, worker_id: str, tool_name: str = "", tool_status: str = ""
    ) -> LegacyStageTrigger:
        self._legacy_trigger_sequence += 1
        return LegacyStageTrigger(
            trigger_id=f"stage-trigger:{worker_id or 'W00'}:{self._legacy_trigger_sequence}",
            worker_id=str(worker_id or ""),
            tool_name=str(tool_name or ""),
            tool_status=str(tool_status or ""),
        )

    async def dispatch_legacy(
        self, trigger: LegacyStageTrigger, handler: Callable[[], Any]
    ) -> Any:
        """Composition adapter: parse outside, execute legacy implementation inside."""

        _ = trigger
        return await _call(handler)

    def _advance_to(
        self,
        machine: StageLifecycleMachine,
        target: StageLifecycleEvent,
        event_id: str,
    ) -> None:
        for expected, event in _ORDER:
            if machine.state is expected:
                self._advance(machine, event, event_id)
            if event is target:
                return

    def _advance(
        self,
        machine: StageLifecycleMachine, event: StageLifecycleEvent, event_id: str
    ) -> None:
        transition = machine.advance(
            event, event_id=f"{event_id}:{event.value}" if event_id else ""
        )
        if self._trace_sink is not None:
            try:
                self._trace_sink(
                    {
                        "event": "stage_lifecycle_transition",
                        "machine_type": transition.machine_type,
                        "machine_id": transition.machine_id,
                        "sequence": transition.sequence,
                        "previous": transition.previous.value,
                        "transition_event": transition.event.value,
                        "current": transition.current.value,
                        "previous_version": transition.previous_version,
                        "current_version": transition.current_version,
                        "event_id": transition.event_id,
                        "duplicate": transition.duplicate,
                    }
                )
            except Exception:
                pass

    def _remember(
        self,
        request: StageCandidateRequest,
        *,
        status: str,
        machine: StageLifecycleMachine,
        gate: StageGateDecision | None = None,
        committed: StageCommittedEvent | None = None,
        projection_hash: str = "",
        error: str = "",
        notification_errors: tuple[str, ...] = (),
    ) -> StageLifecycleResult:
        result = StageLifecycleResult(
            trigger_id=request.trigger_id,
            stage_id=request.stage_id,
            status=status,
            state=machine.state.value,
            gate=gate,
            committed=committed,
            projection_hash=projection_hash,
            error=error,
            notification_errors=notification_errors,
        )
        self._results[request.trigger_id] = result
        return result


async def _call(function: StageCallable, *args: Any, **kwargs: Any) -> Any:
    value = function(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


def _projection_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "StageLifecycleCoordinator",
    "StageLifecyclePorts",
    "StageWorkspacePort",
]
