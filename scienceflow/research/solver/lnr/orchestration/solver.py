# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Stable public entrypoint for the modular LNR solver."""

from __future__ import annotations

import sys

from scienceflow.research.solver.lnr.orchestration.coordinator import solver as _implementation

_implementation.LnrSolver.__module__ = __name__
sys.modules[__name__] = _implementation
