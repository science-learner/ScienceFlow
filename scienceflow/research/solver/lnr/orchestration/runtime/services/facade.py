# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Capabilities injected into the LNR runtime lifecycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from scienceflow.runtime.core.kernel.hooks import HookDispatcher
from scienceflow.research.solver.lnr.orchestration.runtime.services.models import RunSpec

RunResult = dict[str, Any]
RunCallable = Callable[[], Awaitable[RunResult]]


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    """Explicit runtime capabilities; contains no mutable solver back-reference."""

    spec_factory: Callable[[], RunSpec]
    run_single: RunCallable
    run_multi: RunCallable
    hooks: HookDispatcher = field(default_factory=HookDispatcher)


__all__ = ["RunCallable", "RunResult", "RuntimeServices"]
