# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Stable contracts shared by independently evolving ScienceFlow modules."""

from scienceflow.foundation.contracts.runtime.base import VersionedContract
from scienceflow.foundation.contracts.research.evaluation import (
    CandidateRef,
    EvalContext,
    EvaluationRequest,
    EvaluationResult,
    MetricEvent,
)
from scienceflow.foundation.contracts.research.estra import EstraContext, EstraDecision
from scienceflow.foundation.contracts.research.gate import (
    CandidateAssessment,
    EvaluationOutcome,
    GateDecision,
)
from scienceflow.foundation.contracts.runtime.resource_execution import (
    AdmissionDecision,
    ExecutionDecision,
    ExecutionObservation,
    ResourceObservation,
    ResourceRequest,
    ValueAssessment,
)
from scienceflow.foundation.contracts.runtime.workspace_memory import (
    MemoryView,
    StageCommitted,
    StageRecord,
)

__all__ = [
    "CandidateAssessment",
    "CandidateRef",
    "AdmissionDecision",
    "EvalContext",
    "EvaluationOutcome",
    "EvaluationRequest",
    "EvaluationResult",
    "ExecutionDecision",
    "ExecutionObservation",
    "EstraContext",
    "EstraDecision",
    "GateDecision",
    "MetricEvent",
    "MemoryView",
    "ResourceObservation",
    "ResourceRequest",
    "StageCommitted",
    "StageRecord",
    "VersionedContract",
    "ValueAssessment",
]
