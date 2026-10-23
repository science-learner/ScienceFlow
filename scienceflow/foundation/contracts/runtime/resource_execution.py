# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource mechanism and execution-value contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class ResourceRequest(VersionedContract):
    request_id: str
    command_id: str
    worker_id: str
    resource_class: str
    gpu_ids: tuple[str, ...] = ()
    gpu_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceObservation(VersionedContract):
    worker_id: str
    available: bool
    observed_at: float
    gpu_ids: tuple[str, ...] = ()
    active_lease_ids: tuple[str, ...] = ()
    pressure: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdmissionDecision(VersionedContract):
    action: str
    admitted: bool
    reason_code: str
    retry_after_sec: float = 0.0
    lease_id: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionObservation(VersionedContract):
    command_id: str
    elapsed_sec: float
    budget_remaining_sec: float
    metric_improved: bool | None = None
    artifact_progress: bool = False
    recoverable_artifact: bool = False
    resource: ResourceObservation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValueAssessment(VersionedContract):
    score: float
    confidence: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionDecision(VersionedContract):
    action: str
    reason_code: str
    assessment: ValueAssessment
    timebox_sec: float = 0.0
    resource_request: str = ""


__all__ = [
    "AdmissionDecision",
    "ExecutionDecision",
    "ExecutionObservation",
    "ResourceObservation",
    "ResourceRequest",
    "ValueAssessment",
]
