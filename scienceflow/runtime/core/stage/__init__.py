# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Candidate-to-stage lifecycle contracts, machine, and coordinator."""

from scienceflow.runtime.core.stage.contracts import (
    LegacyStageTrigger,
    StageAssessment,
    StageCandidateRequest,
    StageCommittedEvent,
    StageGateDecision,
    StageLifecycleResult,
)
from scienceflow.runtime.core.stage.coordinator import (
    StageLifecycleCoordinator,
    StageLifecyclePorts,
    StageWorkspacePort,
)
from scienceflow.runtime.core.stage.machine import (
    StageLifecycleEvent,
    StageLifecycleMachine,
    StageLifecycleState,
)

__all__ = [
    "LegacyStageTrigger",
    "StageAssessment",
    "StageCandidateRequest",
    "StageCommittedEvent",
    "StageGateDecision",
    "StageLifecycleCoordinator",
    "StageLifecycleEvent",
    "StageLifecycleMachine",
    "StageLifecyclePorts",
    "StageLifecycleResult",
    "StageLifecycleState",
    "StageWorkspacePort",
]
