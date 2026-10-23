# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Canonical resource observer assembled from explicit responsibility owners."""

from __future__ import annotations

__scienceflow_canonical_backend__ = True

from scienceflow.research.solver.lnr.resources.runtime.observer.allocation.admission.decision import (
    AdmissionCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.callbacks.job import JobCallbacks
from scienceflow.research.solver.lnr.resources.runtime.observer.callbacks.lifecycle import (
    LifecycleCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.callbacks.preflight import (
    PreflightCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.construction import (
    ObserverConstruction,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.allocation.lease.callbacks import (
    LeaseCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.observation.classification import (
    ObservationCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.observation.feedback import (
    FeedbackCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.observation.monitoring import (
    MonitoringCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.observation.progress import (
    ProgressObservation,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.observation.sampling import (
    GpuSampling,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.decision.arbiter import (
    ArbiterCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.decision.budget import ReviewBudget
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.decision.execution_value import (
    ExecutionValueCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.state.facts import ReviewFacts
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.state.history import ReviewHistory
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.state.intervention import (
    InterventionCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.decision.priority import (
    ReviewPriorityCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.decision.safety import (
    SafetyCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.assessment.review.state.machine import (
    ReviewStateCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.sharing.decision import (
    SharingCallbacks,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.sharing.scheduling import (
    SharingScheduling,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    process_tree_gpu_placement_snapshot as process_tree_gpu_placement_snapshot,
)


class LHRResourceObserver(
    ObserverConstruction,
    ObservationCallbacks,
    FeedbackCallbacks,
    JobCallbacks,
    PreflightCallbacks,
    LeaseCallbacks,
    AdmissionCallbacks,
    MonitoringCallbacks,
    ProgressObservation,
    GpuSampling,
    ReviewStateCallbacks,
    ReviewFacts,
    ReviewHistory,
    ExecutionValueCallbacks,
    SharingCallbacks,
    SharingScheduling,
    ReviewPriorityCallbacks,
    ReviewBudget,
    ArbiterCallbacks,
    SafetyCallbacks,
    InterventionCallbacks,
    LifecycleCallbacks,
):
    """Coordinate resource callbacks without rebinding component methods."""


__all__ = ["LHRResourceObserver", "process_tree_gpu_placement_snapshot"]
