# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Domain-owned state machine for candidate-to-stage transactions."""

from __future__ import annotations

from enum import Enum

from scienceflow.runtime.core.kernel.state_machines import (
    InvalidTransitionError,
    TransitionEngine,
)


class StageLifecycleState(str, Enum):
    CREATED = "created"
    DETECTED = "detected"
    ARCHIVED = "archived"
    ASSESSED = "assessed"
    GATE_ACCEPTED = "gate_accepted"
    PREPARED = "prepared"
    WORKSPACE_COMMITTED = "workspace_committed"
    STAGE_COMMITTED = "stage_committed"
    MEMORY_PROJECTED = "memory_projected"
    ESTRA_OBSERVED = "estra_observed"
    TELEMETRY_NOTIFIED = "telemetry_notified"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class StageLifecycleEvent(str, Enum):
    DETECT = "detect"
    ARCHIVE = "archive"
    ASSESS = "assess"
    ACCEPT_GATE = "accept_gate"
    REJECT_GATE = "reject_gate"
    PREPARE = "prepare"
    COMMIT_WORKSPACE = "commit_workspace"
    COMMIT_STAGE = "commit_stage"
    PROJECT_MEMORY = "project_memory"
    OBSERVE_ESTRA = "observe_estra"
    NOTIFY_TELEMETRY = "notify_telemetry"
    COMPLETE = "complete"
    ROLLBACK = "rollback"
    FAIL = "fail"


_TRANSITIONS = {
    (StageLifecycleState.CREATED, StageLifecycleEvent.DETECT): StageLifecycleState.DETECTED,
    (StageLifecycleState.DETECTED, StageLifecycleEvent.ARCHIVE): StageLifecycleState.ARCHIVED,
    (StageLifecycleState.ARCHIVED, StageLifecycleEvent.ASSESS): StageLifecycleState.ASSESSED,
    (StageLifecycleState.ASSESSED, StageLifecycleEvent.ACCEPT_GATE): StageLifecycleState.GATE_ACCEPTED,
    (StageLifecycleState.ASSESSED, StageLifecycleEvent.REJECT_GATE): StageLifecycleState.REJECTED,
    (StageLifecycleState.GATE_ACCEPTED, StageLifecycleEvent.PREPARE): StageLifecycleState.PREPARED,
    (StageLifecycleState.PREPARED, StageLifecycleEvent.COMMIT_WORKSPACE): StageLifecycleState.WORKSPACE_COMMITTED,
    (StageLifecycleState.PREPARED, StageLifecycleEvent.ROLLBACK): StageLifecycleState.ROLLED_BACK,
    (StageLifecycleState.WORKSPACE_COMMITTED, StageLifecycleEvent.COMMIT_STAGE): StageLifecycleState.STAGE_COMMITTED,
    (StageLifecycleState.STAGE_COMMITTED, StageLifecycleEvent.PROJECT_MEMORY): StageLifecycleState.MEMORY_PROJECTED,
    (StageLifecycleState.MEMORY_PROJECTED, StageLifecycleEvent.OBSERVE_ESTRA): StageLifecycleState.ESTRA_OBSERVED,
    (StageLifecycleState.ESTRA_OBSERVED, StageLifecycleEvent.NOTIFY_TELEMETRY): StageLifecycleState.TELEMETRY_NOTIFIED,
    (StageLifecycleState.TELEMETRY_NOTIFIED, StageLifecycleEvent.COMPLETE): StageLifecycleState.COMPLETED,
}

for _state in (
    StageLifecycleState.DETECTED,
    StageLifecycleState.ARCHIVED,
    StageLifecycleState.ASSESSED,
    StageLifecycleState.GATE_ACCEPTED,
    StageLifecycleState.WORKSPACE_COMMITTED,
    StageLifecycleState.STAGE_COMMITTED,
    StageLifecycleState.MEMORY_PROJECTED,
    StageLifecycleState.ESTRA_OBSERVED,
    StageLifecycleState.TELEMETRY_NOTIFIED,
):
    _TRANSITIONS[(_state, StageLifecycleEvent.FAIL)] = StageLifecycleState.FAILED


def _advance(
    state: StageLifecycleState, event: StageLifecycleEvent
) -> StageLifecycleState:
    try:
        return _TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise InvalidTransitionError(
            f"illegal stage lifecycle transition: {state.value} + {event.value}"
        ) from exc


class StageLifecycleMachine(
    TransitionEngine[StageLifecycleState, StageLifecycleEvent]
):
    def __init__(self, *, stage_id: str) -> None:
        super().__init__(
            initial_state=StageLifecycleState.CREATED,
            reducer=_advance,
            machine_type="stage_lifecycle",
            machine_id=str(stage_id or ""),
        )


__all__ = [
    "StageLifecycleEvent",
    "StageLifecycleMachine",
    "StageLifecycleState",
]
