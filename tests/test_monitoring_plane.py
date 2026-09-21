from __future__ import annotations

import asyncio

from scienceflow.runtime.observability.monitoring.process.process_tracker import ProcessTracker
from scienceflow.runtime.observability.monitoring import (
    HookTraceObservationSink,
    MonitorService,
    ProcessTrackerProbe,
    RuntimeObservationHook,
)
from scienceflow.runtime.core.kernel import HookDispatcher, HookEvent, HookPoint


def test_monitor_sequences_dedupes_and_isolates_subscriber_failure() -> None:
    async def exercise() -> None:
        delivered: list[int] = []
        monitor = MonitorService(source="test", clock=lambda: 12.5)

        async def subscriber(observation) -> None:
            delivered.append(observation.sequence)

        def broken(_observation) -> None:
            raise RuntimeError("subscriber failed")

        monitor.subscribe(subscriber)
        monitor.subscribe(broken)
        first = await monitor.observe(
            "runtime.started",
            payload={"status": "running"},
            correlation_id="r1:start",
        )
        duplicate = await monitor.observe(
            "runtime.started",
            correlation_id="r1:start",
        )

        assert first is not None
        assert first.schema_version == "1.0"
        assert first.timestamp == 12.5
        assert first.payload == {"status": "running"}
        assert duplicate is None
        assert delivered == [1]
        assert monitor.subscriber_failures == ("RuntimeError: subscriber failed",)

    asyncio.run(exercise())


def test_hook_trace_sink_publishes_correlated_read_only_observation() -> None:
    async def exercise() -> None:
        monitor = MonitorService(source="test.hooks")
        dispatcher = HookDispatcher(
            trace_sinks=(HookTraceObservationSink(monitor),)
        )
        dispatcher.register(
            HookPoint.RUN_STARTED,
            lambda _event: None,
            name="telemetry",
            owner_component="telemetry",
        )

        await dispatcher.dispatch(
            HookEvent(
                point=HookPoint.RUN_STARTED,
                run_id="r1",
                worker_id="W00",
                process_id="p1",
                stage_id="S01",
                event_id="evt-1",
            )
        )

        row = monitor.observations[0]
        assert row.kind == "hook.trace"
        assert row.run_id == "r1"
        assert row.worker_id == "W00"
        assert row.payload["process_id"] == "p1"
        assert row.payload["stage_id"] == "S01"
        assert row.payload["event_id"] == "evt-1"
        assert row.payload["outcome"] == "succeeded"
        assert not hasattr(monitor, "retry_hook")

    asyncio.run(exercise())


def test_runtime_hook_and_process_probe_emit_observations_only() -> None:
    async def exercise() -> None:
        monitor = MonitorService(clock=lambda: 1.0)
        hook = RuntimeObservationHook(monitor)
        await hook(
            HookEvent(
                point=HookPoint.RUN_STARTED,
                run_id="r1",
                worker_id="W00",
                payload={"mode": "single_worker"},
            )
        )
        tracker = ProcessTracker()
        tracker.register(123, "node-1")
        snapshot = await ProcessTrackerProbe(tracker).sample()

        assert monitor.observations[0].kind == "run.started"
        assert monitor.observations[0].worker_id == "W00"
        assert snapshot["processes"][0]["status"] == "running"
        assert not hasattr(monitor, "admit")
        assert not hasattr(monitor, "kill")

    asyncio.run(exercise())
