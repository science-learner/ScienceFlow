# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource inventory, admission, queue, lease, and pressure mechanisms."""
from scienceflow.research.control.resources.policy.review_machine import (
    ResourceReviewEvent,
    ResourceReviewEventKind,
    ResourceReviewMachine,
)
from scienceflow.research.control.resources.policy.effects import (
    ResourceEffectCommand,
    ResourceEffectEvent,
    ResourceEffectExecutor,
    ResourceEffectKind,
)
from scienceflow.research.control.resources.policy.mechanisms import (
    LeaseManager,
    QueueScheduler,
    ResourceRegistry,
)
from scienceflow.research.control.resources.runtime.observation import (
    ResourceFacts,
    ResourceObservationProjector,
    project_gpu_share_decision_facts,
)
from scienceflow.research.control.resources.policy.proposals import (
    AdmissionProposalProjector,
    ResourceActionProposal,
)

__all__ = [
    "GPUQueueConfig",
    "LHRResourceObserver",
    "ResourceManagementService",
    "ResourceEffectCommand",
    "ResourceEffectEvent",
    "ResourceEffectExecutor",
    "ResourceEffectKind",
    "LeaseManager",
    "QueueScheduler",
    "ResourceRegistry",
    "ResourceFacts",
    "ResourceObservationProjector",
    "project_gpu_share_decision_facts",
    "AdmissionProposalProjector",
    "ResourceActionProposal",
    "ResourceReviewEvent",
    "ResourceReviewEventKind",
    "ResourceReviewMachine",
]


def __getattr__(name: str):
    if name in {"GPUQueueConfig", "ResourceManagementService"}:
        from scienceflow.research.control.resources.runtime.service import (
            GPUQueueConfig,
            ResourceManagementService,
        )

        return {
            "GPUQueueConfig": GPUQueueConfig,
            "ResourceManagementService": ResourceManagementService,
        }[name]
    if name == "LHRResourceObserver":
        from scienceflow.research.solver.lnr.resources.runtime.observer import (
            LHRResourceObserver,
        )

        return LHRResourceObserver
    raise AttributeError(name)
