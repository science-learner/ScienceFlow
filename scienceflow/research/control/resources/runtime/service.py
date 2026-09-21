# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Stable service boundary over resource-control mechanisms."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from scienceflow.foundation.contracts import (
    AdmissionDecision,
    ResourceObservation,
    ResourceRequest,
)
from scienceflow.research.control.resources.adapters.lnr_runtime import (
    GPUQueueConfig,
    ResourceRuntime,
)
from scienceflow.research.control.resources.policy.effects import (
    ResourceEffectCommand,
    ResourceEffectEvent,
    ResourceEffectExecutor,
    ResourceEffectKind,
)
from scienceflow.research.control.resources.runtime.observation import ResourceObservationProjector


class ResourceManagementService:
    """Own resource mechanisms; never interprets metric or route value."""

    def __init__(self, backend: ResourceRuntime | None = None) -> None:
        self.backend = backend
        self.observation_projector = ResourceObservationProjector()
        self.effect_executor = ResourceEffectExecutor(backend)

    @classmethod
    def create(
        cls,
        *,
        worker_id: str,
        resource_dir: str | Path,
        gpu_queue: GPUQueueConfig,
    ) -> "ResourceManagementService":
        return cls(
            ResourceRuntime(
                worker_id=worker_id,
                resource_dir=Path(resource_dir),
                gpu_queue=gpu_queue,
            )
        )

    def __getattr__(self, name: str) -> Any:
        """Compatibility bridge for the existing observer callback surface."""

        backend = object.__getattribute__(self, "backend")
        if backend is None:
            raise AttributeError(name)
        return getattr(backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward legacy instance monkeypatches to an existing backend field."""

        if name != "backend":
            backend = self.__dict__.get("backend")
            if backend is not None and hasattr(backend, name):
                setattr(backend, name, value)
                return
        object.__setattr__(self, name, value)

    def admit(self, request: ResourceRequest) -> AdmissionDecision:
        if self.backend is None:
            return AdmissionDecision(
                action="UNAVAILABLE",
                admitted=False,
                reason_code="resource_backend_unconfigured",
            )
        effect = self.execute_effect(
            ResourceEffectCommand(
                command_id=request.command_id,
                kind=ResourceEffectKind.ACQUIRE,
                idempotency_key=f"acquire:{request.request_id}",
                validated=True,
                parameters={
                    "job_id": request.command_id,
                    "resource_class": request.resource_class,
                    "gpu_ids": list(request.gpu_ids),
                    "request_count": request.gpu_count or None,
                    "metadata": dict(request.metadata or {}),
                },
            )
        )
        raw = dict(effect.result)
        admitted = bool(raw.get("acquired"))
        reason = str(raw.get("reason") or raw.get("reason_code") or "")
        retry_after = _float(
            raw.get("retry_after_sec")
            or raw.get("observe_more_sec")
            or raw.get("eta_next_train_sec")
        )
        return AdmissionDecision(
            action="ADMIT" if admitted else "QUEUE",
            admitted=admitted,
            reason_code=reason or ("acquired" if admitted else "resource_busy"),
            retry_after_sec=retry_after,
            lease_id=request.command_id if admitted else "",
            environment=(
                self.backend.env_updates(job_id=request.command_id) if admitted else {}
            ),
            metadata=dict(raw),
        )

    def observe(self, *, gpu_ids: tuple[str, ...] = ()) -> ResourceObservation:
        if self.backend is None:
            return ResourceObservation(
                worker_id="",
                available=False,
                observed_at=time.time(),
                gpu_ids=gpu_ids,
                metadata={"reason": "resource_backend_unconfigured"},
            )
        raw = (
            self.backend.sample_gpu_util(gpu_ids=list(gpu_ids))
            if gpu_ids
            else self.backend.sync_resource_state(reason="service_observe")
        )
        active = self.backend.gpu_store.snapshot_active()
        facts = self.observation_projector.project(
            worker_id=self.backend.worker_id,
            raw=raw,
            gpu_ids=gpu_ids,
            observed_at=time.time(),
            lease_snapshot=active,
        )
        return self.observation_projector.to_contract(facts)

    def execute_effect(self, command: ResourceEffectCommand) -> ResourceEffectEvent:
        return self.effect_executor.execute(command)

    def release_command(
        self, command_id: str, *, elapsed_sec: float = 0.0, status: str = "finished"
    ) -> dict[str, Any]:
        event = self.execute_effect(
            ResourceEffectCommand(
                command_id=command_id,
                kind=ResourceEffectKind.RELEASE,
                idempotency_key=f"release:{command_id}:{status}",
                validated=True,
                parameters={
                    "job_id": command_id,
                    "elapsed_sec": elapsed_sec,
                    "status": status,
                },
            )
        )
        return dict(event.result)


def _float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


__all__ = ["GPUQueueConfig", "ResourceManagementService"]
