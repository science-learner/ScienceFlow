# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Runtime-owned lifecycle inputs.

The legacy solver currently owns the full mutable run state.  ``RunSpec`` is a
small, immutable routing view that lets the runtime boundary evolve without
copying or serializing that state during the compatibility phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RunMode(str, Enum):
    SINGLE_WORKER = "single_worker"
    MULTI_WORKER = "multi_worker"


@dataclass(frozen=True, slots=True)
class RunSpec:
    worker_id: str
    worker_count: int
    mode: RunMode
    run_id: str = ""

    @classmethod
    def from_coordinator(cls, solver: Any) -> "RunSpec":
        worker_id = str(getattr(solver, "worker_id", "") or "")
        lhr = getattr(solver, "lhr", None)
        worker_count = max(1, int(getattr(lhr, "num_workers", 1) or 1))
        mode = RunMode.SINGLE_WORKER if worker_id else RunMode.MULTI_WORKER
        log_dir = getattr(solver, "log_dir", None)
        run_id = str(getattr(log_dir, "name", "") or "")
        return cls(
            worker_id=worker_id,
            worker_count=worker_count,
            mode=mode,
            run_id=run_id,
        )


__all__ = ["RunMode", "RunSpec"]
