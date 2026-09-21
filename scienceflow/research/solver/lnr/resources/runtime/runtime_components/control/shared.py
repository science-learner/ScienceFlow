# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scienceflow.runtime.safety.resource.lifecycle.gpu_sublease import plan_gpu_sublease
from scienceflow.research.control.resources.policy.mechanisms import LeaseManager, QueueScheduler
from scienceflow.runtime.safety.tooling.resource_management.resource_policy import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_READONLY_CPU,
    RESOURCE_UNKNOWN_GPU_EXEC,
)
from scienceflow.research.control.resources.runtime.feedback import resource_feedback_text

from scienceflow.research.solver.lnr.resources.runtime.control.admission.admission_safety import (
    build_admission_safety_deferred_result,
    evaluate_admission_safety,
    evaluate_yellow_safety_hold,
)
from scienceflow.research.solver.lnr.resources.runtime.control.gpu.gpu_lease_store import GPULeaseStore
from scienceflow.research.solver.lnr.resources.runtime.execution.state.history_store import ResourceHistoryStore
from scienceflow.research.solver.lnr.resources.runtime.execution.state.models import GPUQueueConfig
from scienceflow.research.solver.lnr.resources.runtime.control.admission.pressure_state import GPUPressureStateStore
from scienceflow.research.solver.lnr.resources.runtime.control.admission.priority import calculate_admission_priority
from scienceflow.research.solver.lnr.resources.runtime.execution.state.unified_store import UnifiedResourceStore
from scienceflow.research.solver.lnr.resources.runtime.execution.state.utilization import (
    sample_nvidia_smi as _sample_nvidia_smi,
)


def sample_nvidia_smi(gpu_ids: list[str]) -> dict[str, Any]:
    """Preserve the public runtime monkeypatch seam after physical extraction."""

    public_module = sys.modules.get(
        "scienceflow.research.solver.lnr.resources.runtime.execution.facade"
    )
    public_sample = getattr(public_module, "sample_nvidia_smi", None)
    if public_sample is not None and public_sample is not sample_nvidia_smi:
        return public_sample(gpu_ids)
    return _sample_nvidia_smi(gpu_ids)

_CPU_SUPPORT_CLASSES = [RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_READONLY_CPU, RESOURCE_LIGHT_CPU]
_GPU_PRESSURE_SUPPORT_CLASSES = [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, *_CPU_SUPPORT_CLASSES]
_SHARE_OVERRIDE_CANDIDATE_CLASSES = {
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
}
_SHARE_OVERRIDE_PRIMARY_CLASSES = {
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
}
_TRIAL_SHARE_CANDIDATE_CLASSES = {
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
}


@dataclass(frozen=True)
class _GPUResourcePolicy:
    resource_class: str
    slot_weight: float
    max_per_gpu: int
    incompatible_classes: tuple[str, ...]

# Export private helpers for the mechanically moved runtime methods.
__all__ = (
    'Any',
    'GPULeaseStore',
    'GPUPressureStateStore',
    'GPUQueueConfig',
    'LeaseManager',
    'Path',
    'QueueScheduler',
    'RESOURCE_GPU_FEATURE_EXTRACT',
    'RESOURCE_GPU_LIGHT_TRAIN',
    'RESOURCE_GPU_TT_LIGHT',
    'RESOURCE_HEAVY_GPU_CANDIDATE',
    'RESOURCE_HEAVY_GPU_TRAIN',
    'RESOURCE_PURE_TT_CPU',
    'RESOURCE_UNKNOWN_GPU_EXEC',
    'ResourceHistoryStore',
    'UnifiedResourceStore',
    '_CPU_SUPPORT_CLASSES',
    '_GPUResourcePolicy',
    '_GPU_PRESSURE_SUPPORT_CLASSES',
    '_SHARE_OVERRIDE_CANDIDATE_CLASSES',
    '_SHARE_OVERRIDE_PRIMARY_CLASSES',
    '_TRIAL_SHARE_CANDIDATE_CLASSES',
    'asyncio',
    'build_admission_safety_deferred_result',
    'calculate_admission_priority',
    'evaluate_admission_safety',
    'evaluate_yellow_safety_hold',
    'plan_gpu_sublease',
    'resource_feedback_text',
    'sample_nvidia_smi',
    'time',
)
