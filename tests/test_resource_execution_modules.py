# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Independent Resource Management and Execution Value contracts."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from scienceflow.foundation.contracts import (
    AdmissionDecision,
    ExecutionDecision,
    ExecutionObservation,
    ResourceObservation,
    ResourceRequest,
    ValueAssessment,
)
from scienceflow.research.control.execution_value import ExecutionValueService
from scienceflow.research.control.execution_value.adapters import LnrExecutionValueShadow
from scienceflow.research.control.resources import ResourceManagementService


class FakeResourceBackend:
    worker_id = "W01"

    def __init__(self) -> None:
        self.gpu_store = SimpleNamespace(snapshot_active=lambda: {"leases": {"job-1": {}}})
        self.releases: list[str] = []

    def queue_try_acquire(self, **kwargs):
        return {
            "acquired": True,
            "reason": "lease_acquired",
            "request": kwargs,
        }

    def env_updates(self, *, job_id: str) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": "3", "JOB": job_id}

    def sample_gpu_util(self, *, gpu_ids: list[str]):
        return {"available": True, "gpus": gpu_ids}

    def release(self, *, job_id: str, elapsed_sec: float, status: str):
        self.releases.append(job_id)
        return {
            "released": True,
            "job_id": job_id,
            "elapsed_sec": elapsed_sec,
            "status": status,
        }


def test_resource_management_owns_admission_and_lease_mechanisms_only() -> None:
    backend = FakeResourceBackend()
    service = ResourceManagementService(backend)  # type: ignore[arg-type]
    request = ResourceRequest(
        request_id="request-1",
        command_id="job-1",
        worker_id="W01",
        resource_class="heavy_gpu_train",
        gpu_ids=("3",),
        gpu_count=1,
    )

    decision = service.admit(request)
    observation = service.observe(gpu_ids=("3",))
    released = service.release_command("job-1", elapsed_sec=12.0)

    assert decision.action == "ADMIT"
    assert decision.environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert observation.active_lease_ids == ("job-1",)
    assert released["released"] is True


def test_execution_value_recommends_but_does_not_release_resources() -> None:
    service = ExecutionValueService(no_progress_timebox_sec=60)
    resource = ResourceObservation(
        worker_id="W00",
        available=True,
        observed_at=1.0,
        gpu_ids=("0",),
    )
    progressing = ExecutionObservation(
        command_id="train",
        elapsed_sec=120,
        budget_remaining_sec=300,
        metric_improved=True,
        resource=resource,
    )
    stalled = ExecutionObservation(
        command_id="train",
        elapsed_sec=120,
        budget_remaining_sec=300,
        metric_improved=False,
        resource=resource,
    )

    assert service.decide(progressing).action == "CONTINUE"
    decision = service.decide(stalled)
    assert decision.action == "TIMEBOX"
    assert decision.timebox_sec == 60
    assert not hasattr(service, "release")


def test_lnr_execution_value_adapter_is_read_only_and_preserves_unknown_budget() -> None:
    adapter = LnrExecutionValueShadow(
        ExecutionValueService(no_progress_timebox_sec=60)
    )

    decision = adapter.decide(
        command_id="train",
        elapsed_sec=120,
        signal={"metric_value_useful": False},
        artifact_progress=False,
        recoverable_artifact=True,
    )

    assert decision["action"] == "TIMEBOX"
    assert decision["observation"]["metadata"]["budget_known"] is False
    assert decision["observation"]["budget_remaining_sec"] > 0
    assert not hasattr(adapter, "release")
    assert not hasattr(adapter, "kill")


def test_resource_and_execution_modules_obey_one_way_dependency() -> None:
    root = Path(__file__).parents[1] / "scienceflow"
    forbidden = {
        root / "research" / "control" / "resources": (
            "scienceflow.research.control.execution_value",
            "scienceflow.research.quality.evaluator",
            "scienceflow.research.quality.gate",
        ),
        root / "research" / "control" / "execution_value": (
            "scienceflow.research.control.resources",
            "scienceflow.research.quality.gate",
            "scienceflow.research.solver",
        ),
    }
    for directory, prefixes in forbidden.items():
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not str(node.module or "").startswith(prefixes), path
                elif isinstance(node, ast.Import):
                    assert all(
                        not alias.name.startswith(prefixes) for alias in node.names
                    ), path


def test_resource_execution_contracts_are_versioned() -> None:
    classes = (
        ResourceRequest,
        ResourceObservation,
        AdmissionDecision,
        ExecutionObservation,
        ValueAssessment,
        ExecutionDecision,
    )
    assert {contract.contract_schema_version() for contract in classes} == {"1.0"}
