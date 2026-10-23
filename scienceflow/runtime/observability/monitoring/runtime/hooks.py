# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Runtime hook adapter for lifecycle observations."""

from __future__ import annotations

from scienceflow.runtime.observability.monitoring.runtime.service import MonitorService
from scienceflow.runtime.core.kernel.hooks import HookEvent, HookTrace


class RuntimeObservationHook:
    def __init__(self, monitor: MonitorService) -> None:
        self.monitor = monitor

    async def __call__(self, event: HookEvent) -> None:
        await self.monitor.observe(
            event.point.value,
            payload=event.payload,
            run_id=event.run_id,
            worker_id=event.worker_id,
            correlation_id=f"{event.run_id}:{event.point.value}",
        )


class HookTraceObservationSink:
    """Publish immutable hook execution evidence without controlling dispatch."""

    def __init__(self, monitor: MonitorService) -> None:
        self.monitor = monitor

    async def __call__(self, trace: HookTrace) -> None:
        await self.monitor.observe(
            "hook.trace",
            payload={
                "schema_version": trace.schema_version,
                "hook_name": trace.hook_name,
                "hook_point": trace.hook_point.value,
                "process_id": trace.process_id,
                "stage_id": trace.stage_id,
                "event_id": trace.event_id,
                "owner_component": trace.owner_component,
                "priority": trace.priority,
                "failure_mode": trace.failure_mode.value,
                "started_at": trace.started_at,
                "finished_at": trace.finished_at,
                "duration_sec": trace.duration_sec,
                "outcome": trace.outcome.value,
                "error": trace.error,
                "timed_out": trace.timed_out,
            },
            run_id=trace.run_id,
            worker_id=trace.worker_id,
            correlation_id=f"hook-trace:{trace.sequence}",
            timestamp=trace.finished_at,
        )


__all__ = ["HookTraceObservationSink", "RuntimeObservationHook"]
