# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Wall-clock budget calculations for worker and reduction phases."""

from __future__ import annotations

from typing import Any


def global_merge_reserve_sec(lhr: Any) -> float:
    if not bool(getattr(lhr, "merge_enabled", True)):
        return 0.0
    budget = float(getattr(lhr, "wall_clock_budget_sec", 0) or 0)
    configured = float(getattr(lhr, "global_merge_wall_clock_sec", 0) or 0)
    if budget <= 120.0 or configured <= 0.0:
        return 0.0
    reserve = min(configured, max(300.0, budget * 0.15))
    return min(reserve, max(0.0, budget - 60.0))


def worker_wall_clock_budget_sec(lhr: Any) -> int:
    budget = int(float(getattr(lhr, "wall_clock_budget_sec", 0) or 0))
    reserve = int(global_merge_reserve_sec(lhr))
    if budget <= 0 or reserve <= 0:
        return budget
    return max(60, budget - reserve)


__all__ = ["global_merge_reserve_sec", "worker_wall_clock_budget_sec"]
