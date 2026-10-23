# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Stable contracts for one candidate-to-stage lifecycle transaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StageCandidateRequest:
    trigger_id: str
    stage_id: str
    worker_id: str = ""
    candidate_id: str = ""
    artifact_path: str = ""
    artifact_sha: str = ""
    metric_facts: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class StageAssessment:
    accepted: bool
    reason_code: str = ""
    facts: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class StageGateDecision:
    accepted: bool
    action: str
    reason_code: str
    metric_name: str = ""
    metric_value: float | None = None
    lower_is_better: bool | None = None
    metric_validity: str = "unknown"
    facts: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class StageCommittedEvent:
    stage_id: str
    candidate_id: str
    snapshot_id: str
    snapshot_path: str
    gate: StageGateDecision
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class StageLifecycleResult:
    trigger_id: str
    stage_id: str
    status: str
    state: str
    gate: StageGateDecision | None = None
    committed: StageCommittedEvent | None = None
    projection_hash: str = ""
    duplicate: bool = False
    error: str = ""
    notification_errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class LegacyStageTrigger:
    trigger_id: str
    worker_id: str
    tool_name: str = ""
    tool_status: str = ""

    schema_version = "1.0"


__all__ = [
    "LegacyStageTrigger",
    "StageAssessment",
    "StageCandidateRequest",
    "StageCommittedEvent",
    "StageGateDecision",
    "StageLifecycleResult",
]
