# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Versioned candidate-gate contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scienceflow.foundation.contracts.runtime.base import VersionedContract
from scienceflow.foundation.contracts.research.evaluation import MetricEvent


@dataclass(frozen=True)
class GateDecision(VersionedContract):
    """Task-agnostic admission decision derived from evaluator facts."""

    action: str
    accepted: bool
    candidate_ready: bool
    selection_eligible: bool
    reason_code: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "accepted": self.accepted,
            "candidate_ready": self.candidate_ready,
            "selection_eligible": self.selection_eligible,
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class CandidateAssessment(VersionedContract):
    """Authoritative evaluator event paired with its gate decision."""

    event: MetricEvent
    decision: GateDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "decision": self.decision.to_dict(),
        }


# Stable public name used by the assessment service API.
EvaluationOutcome = CandidateAssessment
