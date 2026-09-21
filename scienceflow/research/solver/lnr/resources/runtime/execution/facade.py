# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Canonical resource runtime composed from explicit responsibility owners."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.admission.admission_coordination import (
    AdmissionRuntime,
)
from scienceflow.research.solver.lnr.resources.runtime.runtime_components.admission.lease_coordination import (
    LeaseRuntime,
)
from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.monitoring import (
    MonitoringRuntime,
)
from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.pressure_coordination import (
    PressureRuntime,
)
from scienceflow.research.solver.lnr.resources.runtime.runtime_components.admission.queue_coordination import (
    QueueRuntime,
)
from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.state_policy import (
    StatePolicyRuntime,
)
from scienceflow.research.solver.lnr.resources.runtime.runtime_components.admission.wait_coordination import (
    WaitRuntime,
)
from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    sample_nvidia_smi as sample_nvidia_smi,
)


class ResourceRuntime(
    StatePolicyRuntime,
    AdmissionRuntime,
    WaitRuntime,
    QueueRuntime,
    PressureRuntime,
    LeaseRuntime,
    MonitoringRuntime,
):
    """Own Resource Runtime behavior without facade descriptor rebinding."""


__all__ = ["ResourceRuntime", "sample_nvidia_smi"]
