# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Observation normalization, sequencing, deduplication and publication."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from scienceflow.runtime.observability.monitoring.runtime.contracts import ObservationEnvelope

ObservationSubscriber = Callable[[ObservationEnvelope], Awaitable[None] | None]


class MonitorService:
    """Publish immutable observations without making control decisions."""

    def __init__(
        self,
        *,
        source: str = "scienceflow.monitor",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.source = str(source)
        self._clock = clock
        self._sequence = 0
        self._seen: set[str] = set()
        self._observations: list[ObservationEnvelope] = []
        self._subscribers: list[ObservationSubscriber] = []
        self._subscriber_failures: list[str] = []

    @property
    def observations(self) -> tuple[ObservationEnvelope, ...]:
        return tuple(self._observations)

    @property
    def subscriber_failures(self) -> tuple[str, ...]:
        return tuple(self._subscriber_failures)

    def subscribe(self, subscriber: ObservationSubscriber) -> None:
        self._subscribers.append(subscriber)

    async def observe(
        self,
        kind: str,
        *,
        payload: Mapping[str, Any] | None = None,
        run_id: str = "",
        worker_id: str = "",
        correlation_id: str = "",
        timestamp: float | None = None,
    ) -> ObservationEnvelope | None:
        stable_id = str(correlation_id or "")
        if stable_id and stable_id in self._seen:
            return None
        self._sequence += 1
        envelope = ObservationEnvelope(
            schema_version="1.0",
            sequence=self._sequence,
            kind=str(kind),
            source=self.source,
            timestamp=float(self._clock() if timestamp is None else timestamp),
            run_id=str(run_id or ""),
            worker_id=str(worker_id or ""),
            correlation_id=stable_id,
            payload=dict(payload or {}),
        )
        if stable_id:
            self._seen.add(stable_id)
        self._observations.append(envelope)
        for subscriber in tuple(self._subscribers):
            try:
                result = subscriber(envelope)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._subscriber_failures.append(
                    f"{type(exc).__name__}: {exc}"
                )
        return envelope


__all__ = ["MonitorService", "ObservationSubscriber"]
