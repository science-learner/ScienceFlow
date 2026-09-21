# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Deterministic worker CPU and seed environment construction."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from scienceflow.runtime.core.support.system_resources import parse_cpu_list


def format_cpu_ids(cpu_ids: list[int]) -> str:
    if not cpu_ids:
        return ""
    ids = sorted(set(int(value) for value in cpu_ids))
    ranges: list[str] = []
    start = prev = ids[0]
    for current in ids[1:]:
        if current == prev + 1:
            prev = current
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = current
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def slice_cpu_ids(
    cpu_ids: list[int], *, worker_index: int, worker_count: int
) -> list[int]:
    if not cpu_ids:
        return []
    count = max(1, int(worker_count or 1))
    index = max(0, min(count - 1, int(worker_index or 0)))
    base = len(cpu_ids) // count
    remainder = len(cpu_ids) % count
    start = index * base + min(index, remainder)
    size = base + (1 if index < remainder else 0)
    return cpu_ids[start : start + size]


def build_worker_environment(
    *,
    cfg: Any,
    lhr: Any,
    worker_index: int,
    worker_count: int,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    raw = str(source.get("SCIENCEFLOW_TASK_CPU_LIST", "") or "").strip()
    if not raw:
        raw = str(source.get("SCIENCEFLOW_CPU_LIST", "") or "").strip()
    if not raw:
        raw = str(getattr(getattr(cfg, "exec", None), "cpu_list", "") or "").strip()
    cpu_ids = parse_cpu_list(raw) if raw else []
    worker_cpu_ids = slice_cpu_ids(
        cpu_ids,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    env: dict[str, str] = {}
    task_cpu_set = format_cpu_ids(cpu_ids)
    if task_cpu_set:
        env["SCIENCEFLOW_TASK_CPU_LIST"] = task_cpu_set
    cpu_set = format_cpu_ids(worker_cpu_ids)
    if cpu_set:
        env["_SCIENCEFLOW_CPU_SET"] = cpu_set
        env["SCIENCEFLOW_WORKER_CPU_LIST"] = cpu_set
        env["SCIENCEFLOW_CPU_LIST"] = cpu_set
        thread_cap = int(getattr(lhr, "omp_threads_cap", 8) or 0)
        threads = max(1, len(worker_cpu_ids))
        if thread_cap > 0:
            threads = min(thread_cap, threads)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = str(threads)
    try:
        base_seed = int(getattr(lhr, "seed", 0) or 0)
    except (TypeError, ValueError):
        base_seed = 0
    if base_seed > 0:
        worker_seed = base_seed + max(0, int(worker_index or 0))
        env["SCIENCEFLOW_RANDOM_SEED"] = str(worker_seed)
        env["SEED"] = str(worker_seed)
        env["PYTHONHASHSEED"] = str(worker_seed)
    return env


__all__ = ["build_worker_environment", "format_cpu_ids", "slice_cpu_ids"]
