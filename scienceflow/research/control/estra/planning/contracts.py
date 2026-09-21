# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Versioned EStra planner, archive, and Runtime-command contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from scienceflow.foundation.contracts import EstraContext, EstraDecision


@dataclass(frozen=True, slots=True)
class EstraPlanRequest:
    request_id: str
    context: EstraContext
    prompt: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class EstraPlanAttempt:
    mode: str
    outcome: str
    raw: str = ""
    error: str = ""

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class EstraPlanResult:
    request_id: str
    decision: EstraDecision
    decision_mode: str
    attempts: tuple[EstraPlanAttempt, ...] = ()
    input_hash: str = ""
    output_hash: str = ""
    used_fallback: bool = False

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class EstraDecisionEnvelope:
    envelope_id: str
    request_id: str
    context: EstraContext
    decision: EstraDecision
    decision_mode: str
    input_hash: str
    output_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


class EstraCommandKind(str, Enum):
    RESTORE_STAGE = "restore_stage"
    COMPACT_CURRENT = "compact_current"


@dataclass(frozen=True, slots=True)
class EstraRuntimeCommand:
    command_id: str
    kind: EstraCommandKind
    target_stage: str
    reason_code: str
    evidence_refs: tuple[str, ...] = ()
    expected_stage: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class EstraRecordedPlan:
    envelope: EstraDecisionEnvelope
    command: EstraRuntimeCommand
    archived: bool

    schema_version = "1.0"


__all__ = [
    "EstraCommandKind",
    "EstraDecisionEnvelope",
    "EstraPlanAttempt",
    "EstraPlanRequest",
    "EstraPlanResult",
    "EstraRuntimeCommand",
    "EstraRecordedPlan",
]
