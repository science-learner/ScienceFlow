# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Phase V3-5 resource mechanism replay and effect ownership fixtures."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from scienceflow.research.control.resources import (
    LeaseManager,
    QueueScheduler,
    ResourceEffectCommand,
    ResourceEffectExecutor,
    ResourceEffectKind,
    ResourceObservationProjector,
    ResourceRegistry,
)
from scienceflow.research.solver.lnr.resources.runtime.execution.state.unified_store import (
    UnifiedResourceStore,
    summarize_resource_event_file,
)
from scienceflow.research.solver.lnr.resources.runtime.execution.state.models import GPUQueueConfig
from scienceflow.research.solver.lnr.resources.runtime.execution.facade import ResourceRuntime


class FakeLeaseStore:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.release_calls: list[str] = []

    def snapshot_active(self) -> dict[str, Any]:
        return self.state

    def release(self, *, job_id: str) -> dict[str, Any]:
        self.release_calls.append(job_id)
        lease = self.state.setdefault("leases", {}).pop(job_id, None)
        self.state.setdefault("waiters", {}).pop(job_id, None)
        return {"released": lease is not None, "lease": lease or {}}


class FakeEffects:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def release(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("release", kwargs))
        return {"released": True, **kwargs}

    def queue_try_acquire(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("queue_try_acquire", kwargs))
        return {"acquired": True, **kwargs}

    def queue_timeout(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("queue_timeout", kwargs))
        return {"released": True, **kwargs}


def test_resource_observation_projection_is_pure_and_replay_stable() -> None:
    projector = ResourceObservationProjector()
    raw = {"available": True, "pressure": {"mode": "YELLOW", "generation": 7}}
    snapshot = {
        "leases": {"lease-b": {}, "lease-a": {}},
        "waiters": {"waiter-z": {}, "waiter-c": {}},
    }

    first = projector.project(
        worker_id="W01",
        raw=raw,
        gpu_ids=("1", "0", "1"),
        observed_at=123.0,
        lease_snapshot=snapshot,
    )
    second = projector.project(
        worker_id="W01",
        raw=raw,
        gpu_ids=("1", "0", "1"),
        observed_at=123.0,
        lease_snapshot=snapshot,
    )

    assert first == second
    assert first.active_lease_ids == ("lease-a", "lease-b")
    assert first.pending_job_ids == ("waiter-c", "waiter-z")
    assert first.gpu_ids == ("1", "0")
    assert first.pressure == "YELLOW"
    assert snapshot["leases"] == {"lease-b": {}, "lease-a": {}}


def test_resource_registry_owns_legacy_mapping_projection() -> None:
    registry: ResourceRegistry[object] = ResourceRegistry()
    projection = registry.compat_jobs
    job = object()

    registry.register("job-1", job)
    assert projection["job-1"] is job
    assert registry.snapshot_ids() == ("job-1",)
    assert registry.unregister("job-1") is job
    assert projection == {}


def test_queue_scheduler_recovers_waiters_and_heartbeat_without_policy() -> None:
    ticks = iter((100.0, 100.5, 102.0))
    scheduler = QueueScheduler(heartbeat_sec=1.0, monotonic=lambda: next(ticks))

    assert scheduler.recover({"waiters": {"job-b": {}, "job-a": {}}}) == (
        "job-a",
        "job-b",
    )
    assert scheduler.heartbeat("job-a", elapsed_sec=3.0).emit is True
    assert scheduler.heartbeat("job-a", elapsed_sec=3.5).emit is False
    assert scheduler.heartbeat("job-a", elapsed_sec=5.0).emit is True
    scheduler.complete("job-a")
    assert scheduler.heartbeat("job-a", elapsed_sec=6.0).emit is False


def test_lease_manager_recovers_identity_and_release_projection() -> None:
    store = FakeLeaseStore(
        {
            "leases": {
                "job-2": {"gpu_ids": ["2"]},
                "job-1": {"gpu_ids": ["0", "1"]},
            },
            "waiters": {},
        }
    )
    manager = LeaseManager(store)

    assert manager.has_active("job-1") is True
    assert manager.assigned_gpu_ids["job-1"] == ["0", "1"]
    assert manager.persisted("job-2")["gpu_ids"] == ["2"]
    assert manager.release("job-1")["released"] is True
    assert manager.has_active("job-1") is False


def test_effect_executor_rejects_unvalidated_and_deduplicates_action() -> None:
    target = FakeEffects()
    executor = ResourceEffectExecutor(target)
    invalid = ResourceEffectCommand(
        command_id="job-1",
        kind=ResourceEffectKind.RELEASE,
        idempotency_key="release:job-1",
        validated=False,
        parameters={"job_id": "job-1"},
    )
    with pytest.raises(PermissionError):
        executor.execute(invalid)

    command = ResourceEffectCommand(
        command_id="job-1",
        kind=ResourceEffectKind.RELEASE,
        idempotency_key="release:job-1",
        validated=True,
        parameters={"job_id": "job-1", "elapsed_sec": 3.0, "status": "finished"},
    )
    first = executor.execute(command)
    replay = executor.execute(command)

    assert first.applied is True and first.duplicate is False
    assert replay.applied is True and replay.duplicate is True
    assert target.calls == [
        (
            "release",
            {"job_id": "job-1", "elapsed_sec": 3.0, "status": "finished"},
        )
    ]


def test_observer_bridges_have_no_direct_resource_action_calls() -> None:
    root = (
        Path(__file__).parents[1]
        / "scienceflow/research/solver/lnr/resources/runtime/observer"
    )
    forbidden = {
        "queue_try_acquire",
        "release",
        "release_idle_lease",
        "queue_timeout",
        "grant_shared_gpu_lease",
        "revoke_shared_gpu_lease",
    }
    calls: list[str] = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "self"
                and owner.attr == "resource_runtime"
                and node.func.attr in forbidden
            ):
                calls.append(node.func.attr)
    assert calls == []


def test_resource_event_counter_projection_replays_stably(tmp_path: Path) -> None:
    store = UnifiedResourceStore(tmp_path)
    store.append_event(
        "admission_granted",
        command_id="job-1",
        payload={"result": {"admission_action": "RUN_NOW"}},
    )
    store.append_event(
        "resource_guard_action",
        command_id="job-1",
        payload={"action": "CONTINUE"},
    )
    first = summarize_resource_event_file(store.events_path)
    second = summarize_resource_event_file(store.events_path)

    stable_keys = (
        "event_count",
        "malformed_events",
        "event_counts",
        "admission_actions",
        "guard_actions",
        "active_lease_count",
        "waiter_count",
    )
    assert {key: first[key] for key in stable_keys} == {
        key: second[key] for key in stable_keys
    }
    assert first["event_count"] == 2
    assert first["admission_actions"] == {"RUN_NOW": 1}
    assert first["guard_actions"] == {"CONTINUE": 1}


def test_runtime_recovers_persisted_queue_and_lease_mechanisms(tmp_path: Path) -> None:
    config = GPUQueueConfig(
        enabled=True,
        assignment="env_only",
        max_wait_sec=30.0,
        heartbeat_sec=1.0,
        pressure_min_free_mem_gb=0.0,
    )
    holder = ResourceRuntime(worker_id="W01", resource_dir=tmp_path, gpu_queue=config)
    waiter = ResourceRuntime(worker_id="W02", resource_dir=tmp_path, gpu_queue=config)

    assert holder.queue_try_acquire(
        job_id="holder",
        resource_class="heavy_gpu_candidate",
        gpu_ids=["0"],
    )["acquired"] is True
    assert waiter.queue_try_acquire(
        job_id="waiter",
        resource_class="heavy_gpu_candidate",
        gpu_ids=["0"],
    )["acquired"] is False

    recovered = ResourceRuntime(worker_id="W03", resource_dir=tmp_path, gpu_queue=config)
    assert recovered.lease_manager.has_active("holder") is True
    assert "waiter" in recovered.queue_scheduler.pending_job_ids
    assert recovered.persisted_lease(job_id="holder")["gpu_ids"] == ["0"]

    assert holder.release(job_id="holder")["released"] is True
    waiter.queue_timeout(job_id="waiter", elapsed_sec=1.0, reason="test_cleanup")
