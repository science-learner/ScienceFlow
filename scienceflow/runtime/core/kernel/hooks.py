# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Deterministic, failure-isolated runtime hook dispatch with durable traces."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HookPoint(str, Enum):
    RUN_STARTING = "run.starting"
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"


class HookFailureMode(str, Enum):
    SOFT = "soft"
    HARD = "hard"


class HookIdempotencyScope(str, Enum):
    NONE = "none"
    EVENT = "event"
    RUN = "run"
    WORKER = "worker"


class HookOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    DUPLICATE_SKIPPED = "duplicate_skipped"


@dataclass(frozen=True, slots=True)
class HookEvent:
    point: HookPoint
    run_id: str
    worker_id: str = ""
    process_id: str = ""
    stage_id: str = ""
    event_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


HookCallback = Callable[[HookEvent], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class HookRegistration:
    schema_version: str
    name: str
    hook_point: HookPoint
    priority: int
    timeout_sec: float | None
    failure_mode: HookFailureMode
    idempotency_scope: HookIdempotencyScope
    owner_component: str


@dataclass(frozen=True, slots=True)
class HookFailure:
    name: str
    error: str
    timed_out: bool = False
    failure_mode: HookFailureMode = HookFailureMode.SOFT


@dataclass(frozen=True, slots=True)
class HookTrace:
    schema_version: str
    sequence: int
    run_id: str
    worker_id: str
    process_id: str
    stage_id: str
    event_id: str
    hook_name: str
    hook_point: HookPoint
    owner_component: str
    priority: int
    failure_mode: HookFailureMode
    started_at: float
    finished_at: float
    duration_sec: float
    outcome: HookOutcome
    error: str = ""
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class HookReport:
    point: HookPoint
    called: tuple[str, ...]
    failures: tuple[HookFailure, ...]
    skipped: tuple[str, ...] = ()
    traces: tuple[HookTrace, ...] = ()


class HookDispatchError(RuntimeError):
    """Raised when an explicitly hard hook fails closed."""

    def __init__(self, report: HookReport, failure: HookFailure) -> None:
        super().__init__(f"hard hook {failure.name} failed: {failure.error}")
        self.report = report
        self.failure = failure


@dataclass(frozen=True, slots=True)
class _RegisteredHook:
    registration: HookRegistration
    callback: HookCallback
    ordinal: int


HookTraceSink = Callable[[HookTrace], Awaitable[None] | None]


class HookDispatcher:
    """Invoke hooks in priority/order sequence with explicit failure semantics."""

    def __init__(self, *, trace_sinks: tuple[HookTraceSink, ...] = ()) -> None:
        self._registrations: dict[HookPoint, list[_RegisteredHook]] = {}
        self._registration_sequence = 0
        self._trace_sequence = 0
        self._seen: set[tuple[str, ...]] = set()
        self._traces: list[HookTrace] = []
        self._trace_sinks = list(trace_sinks)
        self._trace_sink_failures: list[str] = []

    @property
    def registrations(self) -> tuple[HookRegistration, ...]:
        rows = [
            item
            for registrations in self._registrations.values()
            for item in registrations
        ]
        rows.sort(key=lambda item: item.ordinal)
        return tuple(item.registration for item in rows)

    @property
    def traces(self) -> tuple[HookTrace, ...]:
        return tuple(self._traces)

    @property
    def trace_sink_failures(self) -> tuple[str, ...]:
        return tuple(self._trace_sink_failures)

    def subscribe_trace(self, sink: HookTraceSink) -> None:
        if sink not in self._trace_sinks:
            self._trace_sinks.append(sink)

    def register(
        self,
        point: HookPoint,
        callback: HookCallback,
        *,
        name: str = "",
        priority: int = 0,
        timeout_sec: float | None = None,
        failure_mode: HookFailureMode | str = HookFailureMode.SOFT,
        idempotency_scope: HookIdempotencyScope | str = HookIdempotencyScope.NONE,
        owner_component: str = "",
    ) -> HookRegistration:
        callback_name = name or getattr(callback, "__name__", callback.__class__.__name__)
        metadata = HookRegistration(
            schema_version="1.0",
            name=str(callback_name),
            hook_point=point,
            priority=int(priority),
            timeout_sec=None if timeout_sec is None else max(0.0, float(timeout_sec)),
            failure_mode=HookFailureMode(failure_mode),
            idempotency_scope=HookIdempotencyScope(idempotency_scope),
            owner_component=str(owner_component or "unknown"),
        )
        self._registration_sequence += 1
        self._registrations.setdefault(point, []).append(
            _RegisteredHook(
                registration=metadata,
                callback=callback,
                ordinal=self._registration_sequence,
            )
        )
        return metadata

    async def dispatch(self, event: HookEvent) -> HookReport:
        called: list[str] = []
        skipped: list[str] = []
        failures: list[HookFailure] = []
        traces: list[HookTrace] = []
        registrations = sorted(
            tuple(self._registrations.get(event.point, ())),
            key=lambda item: (-item.registration.priority, item.ordinal),
        )
        for item in registrations:
            registration = item.registration
            dedupe_key = self._idempotency_key(registration, event)
            if dedupe_key is not None and dedupe_key in self._seen:
                skipped.append(registration.name)
                started_at = time.time()
                trace = self._new_trace(
                    registration,
                    event,
                    started_at=started_at,
                    started_monotonic=time.monotonic(),
                    outcome=HookOutcome.DUPLICATE_SKIPPED,
                )
                traces.append(trace)
                await self._publish_trace(trace)
                continue
            if dedupe_key is not None:
                self._seen.add(dedupe_key)

            called.append(registration.name)
            started_at = time.time()
            started_monotonic = time.monotonic()
            failure: HookFailure | None = None
            outcome = HookOutcome.SUCCEEDED
            try:
                result = item.callback(event)
                if inspect.isawaitable(result):
                    if registration.timeout_sec is None:
                        await result
                    else:
                        await asyncio.wait_for(result, timeout=registration.timeout_sec)
            except asyncio.TimeoutError:
                outcome = HookOutcome.TIMED_OUT
                failure = HookFailure(
                    name=registration.name,
                    error="hook timed out",
                    timed_out=True,
                    failure_mode=registration.failure_mode,
                )
            except Exception as exc:
                outcome = HookOutcome.FAILED
                failure = HookFailure(
                    name=registration.name,
                    error=f"{type(exc).__name__}: {exc}",
                    failure_mode=registration.failure_mode,
                )
            trace = self._new_trace(
                registration,
                event,
                started_at=started_at,
                started_monotonic=started_monotonic,
                outcome=outcome,
                error=failure.error if failure is not None else "",
                timed_out=bool(failure and failure.timed_out),
            )
            traces.append(trace)
            await self._publish_trace(trace)
            if failure is None:
                continue
            failures.append(failure)
            if registration.failure_mode is HookFailureMode.HARD:
                report = HookReport(
                    point=event.point,
                    called=tuple(called),
                    failures=tuple(failures),
                    skipped=tuple(skipped),
                    traces=tuple(traces),
                )
                raise HookDispatchError(report, failure)
        return HookReport(
            point=event.point,
            called=tuple(called),
            failures=tuple(failures),
            skipped=tuple(skipped),
            traces=tuple(traces),
        )

    @staticmethod
    def _idempotency_key(
        registration: HookRegistration,
        event: HookEvent,
    ) -> tuple[str, ...] | None:
        scope = registration.idempotency_scope
        prefix = (registration.hook_point.value, registration.name, scope.value)
        if scope is HookIdempotencyScope.NONE:
            return None
        if scope is HookIdempotencyScope.RUN:
            return (*prefix, event.run_id)
        if scope is HookIdempotencyScope.WORKER:
            return (*prefix, event.run_id, event.worker_id)
        stable_event_id = event.event_id or ":".join(
            (
                event.run_id,
                event.worker_id,
                event.process_id,
                event.stage_id,
                event.point.value,
            )
        )
        return (*prefix, stable_event_id)

    def _new_trace(
        self,
        registration: HookRegistration,
        event: HookEvent,
        *,
        started_at: float,
        started_monotonic: float,
        outcome: HookOutcome,
        error: str = "",
        timed_out: bool = False,
    ) -> HookTrace:
        self._trace_sequence += 1
        finished_at = time.time()
        return HookTrace(
            schema_version="1.0",
            sequence=self._trace_sequence,
            run_id=event.run_id,
            worker_id=event.worker_id,
            process_id=event.process_id,
            stage_id=event.stage_id,
            event_id=event.event_id,
            hook_name=registration.name,
            hook_point=registration.hook_point,
            owner_component=registration.owner_component,
            priority=registration.priority,
            failure_mode=registration.failure_mode,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=max(0.0, time.monotonic() - started_monotonic),
            outcome=outcome,
            error=error,
            timed_out=timed_out,
        )

    async def _publish_trace(self, trace: HookTrace) -> None:
        self._traces.append(trace)
        for sink in self._trace_sinks:
            try:
                result = sink(trace)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._trace_sink_failures.append(
                    f"{getattr(sink, '__name__', type(sink).__name__)}: "
                    f"{type(exc).__name__}: {exc}"
                )


__all__ = [
    "HookCallback",
    "HookDispatchError",
    "HookDispatcher",
    "HookEvent",
    "HookFailure",
    "HookFailureMode",
    "HookIdempotencyScope",
    "HookOutcome",
    "HookPoint",
    "HookRegistration",
    "HookReport",
    "HookTrace",
    "HookTraceSink",
]
