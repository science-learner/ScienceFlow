from __future__ import annotations

import asyncio

import pytest

from scienceflow.runtime.core.kernel import (
    HookDispatchError,
    HookDispatcher,
    HookEvent,
    HookFailureMode,
    HookIdempotencyScope,
    HookOutcome,
    HookPoint,
    InvalidTransitionError,
    JsonEventJournal,
    JsonStateProjection,
    RunLifecycleEvent,
    RunLifecycleMachine,
    RunLifecycleState,
    TransitionVersionConflictError,
    WorkerLifecycleEvent,
    WorkerLifecycleMachine,
    WorkerLifecycleState,
)
from scienceflow.research.solver.lnr.orchestration.runtime import LnrRuntime, RunMode, RunSpec, RuntimeServices
from scienceflow.research.solver.lnr.orchestration.runtime.execution.adapters import runtime_services_from_legacy_host


def test_run_lifecycle_rejects_illegal_transitions_and_dedupes_event_ids() -> None:
    machine = RunLifecycleMachine(run_id="r1")
    first = machine.advance(
        RunLifecycleEvent.START,
        event_id="run:start",
        expected_version=0,
    )
    duplicate = machine.advance(
        RunLifecycleEvent.START,
        event_id="run:start",
        expected_version=0,
    )

    assert first.current is RunLifecycleState.RUNNING
    assert (first.previous_version, first.current_version) == (0, 1)
    assert (first.machine_type, first.machine_id) == ("run", "r1")
    assert duplicate.duplicate is True
    assert machine.version == 1
    assert len(machine.transitions) == 1
    with pytest.raises(TransitionVersionConflictError):
        machine.advance(
            RunLifecycleEvent.SUCCEED,
            event_id="run:succeed",
            expected_version=0,
        )
    with pytest.raises(InvalidTransitionError):
        machine.advance(RunLifecycleEvent.START, event_id="another-start")
    with pytest.raises(InvalidTransitionError):
        machine.advance(RunLifecycleEvent.FAIL, event_id="run:start")


def test_worker_lifecycle_has_explicit_identity_and_reduction_state() -> None:
    machine = WorkerLifecycleMachine(worker_id="W01")
    machine.advance(WorkerLifecycleEvent.START, expected_version=0)
    machine.advance(WorkerLifecycleEvent.READY, expected_version=1)
    reducing = machine.advance(
        WorkerLifecycleEvent.BEGIN_REDUCTION,
        expected_version=2,
    )
    finished = machine.advance(WorkerLifecycleEvent.SUCCEED, expected_version=3)

    assert reducing.current is WorkerLifecycleState.REDUCING
    assert finished.current is WorkerLifecycleState.SUCCEEDED
    assert finished.current_version == 4
    assert (finished.machine_type, finished.machine_id) == ("worker", "W01")


def test_hook_dispatch_is_ordered_timeout_bounded_and_failure_isolated() -> None:
    async def exercise() -> None:
        dispatcher = HookDispatcher()
        calls: list[str] = []

        def first(_event: HookEvent) -> None:
            calls.append("first")

        async def broken(_event: HookEvent) -> None:
            calls.append("broken")
            raise RuntimeError("boom")

        async def slow(_event: HookEvent) -> None:
            calls.append("slow")
            await asyncio.sleep(0.05)

        dispatcher.register(HookPoint.RUN_STARTED, first)
        dispatcher.register(HookPoint.RUN_STARTED, broken)
        dispatcher.register(HookPoint.RUN_STARTED, slow, timeout_sec=0.001)
        report = await dispatcher.dispatch(
            HookEvent(point=HookPoint.RUN_STARTED, run_id="r1")
        )

        assert calls == ["first", "broken", "slow"]
        assert report.called == ("first", "broken", "slow")
        assert [failure.name for failure in report.failures] == ["broken", "slow"]
        assert report.failures[-1].timed_out is True

    asyncio.run(exercise())


def test_hook_metadata_priority_idempotency_and_trace_are_explicit() -> None:
    async def exercise() -> None:
        calls: list[str] = []
        dispatcher = HookDispatcher()
        low = dispatcher.register(
            HookPoint.RUN_STARTED,
            lambda _event: calls.append("low"),
            name="low",
            priority=10,
            owner_component="telemetry",
        )
        high = dispatcher.register(
            HookPoint.RUN_STARTED,
            lambda _event: calls.append("high"),
            name="high",
            priority=20,
            idempotency_scope=HookIdempotencyScope.EVENT,
            owner_component="workspace",
        )
        event = HookEvent(
            point=HookPoint.RUN_STARTED,
            run_id="r1",
            worker_id="W00",
            process_id="p1",
            stage_id="S01",
            event_id="r1:started",
        )

        first = await dispatcher.dispatch(event)
        second = await dispatcher.dispatch(event)

        assert calls == ["high", "low", "low"]
        assert first.called == ("high", "low")
        assert second.called == ("low",)
        assert second.skipped == ("high",)
        assert low.schema_version == high.schema_version == "1.0"
        assert high.priority == 20
        assert high.owner_component == "workspace"
        assert [trace.sequence for trace in dispatcher.traces] == [1, 2, 3, 4]
        assert dispatcher.traces[0].outcome is HookOutcome.SUCCEEDED
        assert dispatcher.traces[0].process_id == "p1"
        assert dispatcher.traces[0].stage_id == "S01"
        assert dispatcher.traces[2].outcome is HookOutcome.DUPLICATE_SKIPPED

    asyncio.run(exercise())


def test_hard_hook_fails_closed_while_trace_sink_failure_stays_soft() -> None:
    async def exercise() -> None:
        calls: list[str] = []

        def broken_sink(_trace) -> None:
            raise RuntimeError("trace unavailable")

        dispatcher = HookDispatcher(trace_sinks=(broken_sink,))

        def hard(_event: HookEvent) -> None:
            calls.append("hard")
            raise RuntimeError("safety invariant")

        dispatcher.register(
            HookPoint.RUN_STARTING,
            hard,
            name="safety_guard",
            failure_mode=HookFailureMode.HARD,
            owner_component="safety",
        )
        dispatcher.register(
            HookPoint.RUN_STARTING,
            lambda _event: calls.append("effect"),
            name="effect",
        )

        with pytest.raises(HookDispatchError) as caught:
            await dispatcher.dispatch(
                HookEvent(point=HookPoint.RUN_STARTING, run_id="r1")
            )

        assert calls == ["hard"]
        assert caught.value.report.called == ("safety_guard",)
        assert caught.value.failure.failure_mode is HookFailureMode.HARD
        assert dispatcher.traces[-1].outcome is HookOutcome.FAILED
        assert dispatcher.trace_sink_failures == (
            "broken_sink: RuntimeError: trace unavailable",
        )

    asyncio.run(exercise())


def test_timed_out_idempotent_hook_is_not_reapplied() -> None:
    async def exercise() -> None:
        effects: list[str] = []
        dispatcher = HookDispatcher()

        async def partial_effect(_event: HookEvent) -> None:
            effects.append("lease-release-request")
            await asyncio.sleep(0.05)

        dispatcher.register(
            HookPoint.RUN_CANCELLED,
            partial_effect,
            name="lease_release",
            timeout_sec=0.001,
            idempotency_scope=HookIdempotencyScope.EVENT,
            owner_component="resource_effect_adapter",
        )
        event = HookEvent(
            point=HookPoint.RUN_CANCELLED,
            run_id="r1",
            event_id="cancel-1",
        )

        first = await dispatcher.dispatch(event)
        second = await dispatcher.dispatch(event)

        assert effects == ["lease-release-request"]
        assert first.failures[0].timed_out is True
        assert second.called == ()
        assert second.skipped == ("lease_release",)
        assert dispatcher.traces[-1].outcome is HookOutcome.DUPLICATE_SKIPPED

    asyncio.run(exercise())


def test_lnr_runtime_advances_lifecycle_and_emits_observation_hooks() -> None:
    async def exercise() -> None:
        points: list[HookPoint] = []
        hooks = HookDispatcher()
        hooks.register(HookPoint.RUN_STARTING, lambda event: points.append(event.point))
        hooks.register(HookPoint.RUN_STARTED, lambda event: points.append(event.point))
        hooks.register(HookPoint.RUN_FINISHED, lambda event: points.append(event.point))

        async def run_single() -> dict[str, str]:
            return {"status": "ok"}

        runtime = LnrRuntime(
            RuntimeServices(
                spec_factory=lambda: RunSpec(
                    worker_id="W00",
                    worker_count=1,
                    mode=RunMode.SINGLE_WORKER,
                    run_id="r1",
                ),
                run_single=run_single,
                run_multi=run_single,
                hooks=hooks,
            )
        )
        assert await runtime.run() == {"status": "ok"}
        assert runtime.last_lifecycle is not None
        assert runtime.last_lifecycle.state is RunLifecycleState.SUCCEEDED
        assert points == [
            HookPoint.RUN_STARTING,
            HookPoint.RUN_STARTED,
            HookPoint.RUN_FINISHED,
        ]

    asyncio.run(exercise())


def test_legacy_adapter_keeps_dynamic_runners_and_wires_read_only_monitor() -> None:
    async def exercise() -> None:
        class Host:
            worker_id = "W00"
            log_dir = None

            def __init__(self) -> None:
                self.lhr = type("Lhr", (), {"num_workers": 1})()
                self.calls: list[str] = []

            async def _run_single(self) -> dict[str, str]:
                self.calls.append("original")
                return {"runner": "original"}

            async def _run_multi_worker(self) -> dict[str, str]:
                return {"runner": "multi"}

        host = Host()
        runtime = LnrRuntime(runtime_services_from_legacy_host(host))

        async def replacement() -> dict[str, str]:
            host.calls.append("replacement")
            return {"runner": "replacement"}

        host._run_single = replacement  # type: ignore[method-assign]
        assert await runtime.run() == {"runner": "replacement"}
        assert host.calls == ["replacement"]
        assert [row.kind for row in host.runtime_monitor.observations] == [
            "run.starting",
            "hook.trace",
            "run.started",
            "hook.trace",
            "run.finished",
            "hook.trace",
        ]
        traces = [
            row for row in host.runtime_monitor.observations if row.kind == "hook.trace"
        ]
        assert [row.payload["outcome"] for row in traces] == [
            "succeeded",
            "succeeded",
            "succeeded",
        ]

    asyncio.run(exercise())


def test_lnr_runtime_records_failure_without_swallowing_runner_error() -> None:
    async def exercise() -> None:
        failed: list[HookEvent] = []
        hooks = HookDispatcher()
        hooks.register(HookPoint.RUN_FAILED, failed.append)

        async def broken() -> dict[str, str]:
            raise RuntimeError("runner failed")

        runtime = LnrRuntime(
            RuntimeServices(
                spec_factory=lambda: RunSpec(
                    worker_id="W00",
                    worker_count=1,
                    mode=RunMode.SINGLE_WORKER,
                ),
                run_single=broken,
                run_multi=broken,
                hooks=hooks,
            )
        )
        with pytest.raises(RuntimeError, match="runner failed"):
            await runtime.run()
        assert runtime.last_lifecycle is not None
        assert runtime.last_lifecycle.state is RunLifecycleState.FAILED
        assert failed[0].payload["error"] == "RuntimeError: runner failed"

    asyncio.run(exercise())


def test_json_journal_and_projection_keep_legacy_wire_format(tmp_path) -> None:
    event_path = tmp_path / "events.jsonl"
    state_path = tmp_path / "state.json"
    event = {"z": 1, "payload": {"path": tmp_path / "item"}, "a": "中文"}
    JsonEventJournal(event_path).append(event)
    JsonStateProjection(state_path).write({"z": 1, "a": "中文"})

    assert event_path.read_text(encoding="utf-8") == (
        '{"a": "中文", "payload": {"path": "'
        + str(tmp_path / "item")
        + '"}, "z": 1}\n'
    )
    assert state_path.read_text(encoding="utf-8") == (
        '{\n  "a": "中文",\n  "z": 1\n}\n'
    )
    assert JsonEventJournal(event_path).read() == [
        {"a": "中文", "payload": {"path": str(tmp_path / "item")}, "z": 1}
    ]
