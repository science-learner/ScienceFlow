# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Typed transition engine used to coordinate domain-owned state machines."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

StateT = TypeVar("StateT", bound=Hashable)
EventT = TypeVar("EventT", bound=Hashable)


class InvalidTransitionError(ValueError):
    """Raised when a state machine rejects an event in its current state."""


class TransitionVersionConflictError(InvalidTransitionError):
    """Raised when a transition targets a stale machine version."""


@dataclass(frozen=True, slots=True)
class Transition(Generic[StateT, EventT]):
    sequence: int
    previous: StateT
    event: EventT
    current: StateT
    previous_version: int
    current_version: int
    machine_type: str = ""
    machine_id: str = ""
    event_id: str = ""
    duplicate: bool = False


class TransitionEngine(Generic[StateT, EventT]):
    """Advance a pure reducer and retain a deterministic in-memory trace.

    The engine owns transition mechanics only. The reducer and its legal graph
    remain with the domain that owns the state machine.
    """

    def __init__(
        self,
        *,
        initial_state: StateT,
        reducer: Callable[[StateT, EventT], StateT],
        machine_type: str = "",
        machine_id: str = "",
    ) -> None:
        self._state = initial_state
        self._reducer = reducer
        self._machine_type = str(machine_type or "")
        self._machine_id = str(machine_id or "")
        self._transitions: list[Transition[StateT, EventT]] = []
        self._event_results: dict[str, Transition[StateT, EventT]] = {}

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def transitions(self) -> tuple[Transition[StateT, EventT], ...]:
        return tuple(self._transitions)

    @property
    def version(self) -> int:
        return len(self._transitions)

    @property
    def machine_type(self) -> str:
        return self._machine_type

    @property
    def machine_id(self) -> str:
        return self._machine_id

    def advance(
        self,
        event: EventT,
        *,
        event_id: str = "",
        expected_version: int | None = None,
    ) -> Transition[StateT, EventT]:
        stable_id = str(event_id or "")
        if stable_id and stable_id in self._event_results:
            prior = self._event_results[stable_id]
            if prior.event != event:
                raise InvalidTransitionError(
                    f"event_id {stable_id!r} was already used for {prior.event!r}"
                )
            return Transition(
                sequence=prior.sequence,
                previous=prior.previous,
                event=prior.event,
                current=prior.current,
                previous_version=prior.previous_version,
                current_version=prior.current_version,
                machine_type=prior.machine_type,
                machine_id=prior.machine_id,
                event_id=prior.event_id,
                duplicate=True,
            )

        if expected_version is not None and int(expected_version) != self.version:
            raise TransitionVersionConflictError(
                f"expected version {int(expected_version)}, current version {self.version}"
            )

        previous = self._state
        try:
            current = self._reducer(previous, event)
        except InvalidTransitionError:
            raise
        except (KeyError, ValueError) as exc:
            raise InvalidTransitionError(
                f"illegal transition: state={previous!r}, event={event!r}"
            ) from exc
        transition = Transition(
            sequence=len(self._transitions) + 1,
            previous=previous,
            event=event,
            current=current,
            previous_version=self.version,
            current_version=self.version + 1,
            machine_type=self.machine_type,
            machine_id=self.machine_id,
            event_id=stable_id,
        )
        self._state = current
        self._transitions.append(transition)
        if stable_id:
            self._event_results[stable_id] = transition
        return transition


class RunLifecycleState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunLifecycleEvent(str, Enum):
    START = "start"
    SUCCEED = "succeed"
    FAIL = "fail"
    CANCEL = "cancel"


_RUN_TRANSITIONS: dict[
    tuple[RunLifecycleState, RunLifecycleEvent], RunLifecycleState
] = {
    (RunLifecycleState.CREATED, RunLifecycleEvent.START): RunLifecycleState.RUNNING,
    (RunLifecycleState.RUNNING, RunLifecycleEvent.SUCCEED): RunLifecycleState.SUCCEEDED,
    (RunLifecycleState.RUNNING, RunLifecycleEvent.FAIL): RunLifecycleState.FAILED,
    (RunLifecycleState.RUNNING, RunLifecycleEvent.CANCEL): RunLifecycleState.CANCELLED,
}


def _advance_run(
    state: RunLifecycleState, event: RunLifecycleEvent
) -> RunLifecycleState:
    try:
        return _RUN_TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise InvalidTransitionError(
            f"illegal run transition: {state.value} + {event.value}"
        ) from exc


class RunLifecycleMachine(TransitionEngine[RunLifecycleState, RunLifecycleEvent]):
    def __init__(self, *, run_id: str = "") -> None:
        super().__init__(
            initial_state=RunLifecycleState.CREATED,
            reducer=_advance_run,
            machine_type="run",
            machine_id=run_id,
        )


class WorkerLifecycleState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    REDUCING = "reducing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class WorkerLifecycleEvent(str, Enum):
    START = "start"
    READY = "ready"
    BEGIN_REDUCTION = "begin_reduction"
    SUCCEED = "succeed"
    FAIL = "fail"
    CANCEL = "cancel"
    TIME_OUT = "time_out"


_WORKER_TRANSITIONS: dict[
    tuple[WorkerLifecycleState, WorkerLifecycleEvent], WorkerLifecycleState
] = {
    (WorkerLifecycleState.CREATED, WorkerLifecycleEvent.START): (
        WorkerLifecycleState.STARTING
    ),
    (WorkerLifecycleState.STARTING, WorkerLifecycleEvent.READY): (
        WorkerLifecycleState.RUNNING
    ),
    (WorkerLifecycleState.RUNNING, WorkerLifecycleEvent.BEGIN_REDUCTION): (
        WorkerLifecycleState.REDUCING
    ),
    (WorkerLifecycleState.RUNNING, WorkerLifecycleEvent.SUCCEED): (
        WorkerLifecycleState.SUCCEEDED
    ),
    (WorkerLifecycleState.REDUCING, WorkerLifecycleEvent.SUCCEED): (
        WorkerLifecycleState.SUCCEEDED
    ),
    (WorkerLifecycleState.STARTING, WorkerLifecycleEvent.FAIL): (
        WorkerLifecycleState.FAILED
    ),
    (WorkerLifecycleState.RUNNING, WorkerLifecycleEvent.FAIL): (
        WorkerLifecycleState.FAILED
    ),
    (WorkerLifecycleState.REDUCING, WorkerLifecycleEvent.FAIL): (
        WorkerLifecycleState.FAILED
    ),
    (WorkerLifecycleState.STARTING, WorkerLifecycleEvent.CANCEL): (
        WorkerLifecycleState.CANCELLED
    ),
    (WorkerLifecycleState.RUNNING, WorkerLifecycleEvent.CANCEL): (
        WorkerLifecycleState.CANCELLED
    ),
    (WorkerLifecycleState.REDUCING, WorkerLifecycleEvent.CANCEL): (
        WorkerLifecycleState.CANCELLED
    ),
    (WorkerLifecycleState.STARTING, WorkerLifecycleEvent.TIME_OUT): (
        WorkerLifecycleState.TIMED_OUT
    ),
    (WorkerLifecycleState.RUNNING, WorkerLifecycleEvent.TIME_OUT): (
        WorkerLifecycleState.TIMED_OUT
    ),
    (WorkerLifecycleState.REDUCING, WorkerLifecycleEvent.TIME_OUT): (
        WorkerLifecycleState.TIMED_OUT
    ),
}


def _advance_worker(
    state: WorkerLifecycleState, event: WorkerLifecycleEvent
) -> WorkerLifecycleState:
    try:
        return _WORKER_TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise InvalidTransitionError(
            f"illegal worker transition: {state.value} + {event.value}"
        ) from exc


class WorkerLifecycleMachine(
    TransitionEngine[WorkerLifecycleState, WorkerLifecycleEvent]
):
    def __init__(self, *, worker_id: str) -> None:
        super().__init__(
            initial_state=WorkerLifecycleState.CREATED,
            reducer=_advance_worker,
            machine_type="worker",
            machine_id=worker_id,
        )


__all__ = [
    "InvalidTransitionError",
    "RunLifecycleEvent",
    "RunLifecycleMachine",
    "RunLifecycleState",
    "Transition",
    "TransitionEngine",
    "TransitionVersionConflictError",
    "WorkerLifecycleEvent",
    "WorkerLifecycleMachine",
    "WorkerLifecycleState",
]
