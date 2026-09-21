# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Adapter exposing the established LNR queue/lease backend."""

from scienceflow.research.solver.lnr.resources.runtime.execution.state.models import GPUQueueConfig
from scienceflow.research.solver.lnr.resources.runtime.execution.facade import ResourceRuntime

__all__ = ["GPUQueueConfig", "ResourceRuntime"]
