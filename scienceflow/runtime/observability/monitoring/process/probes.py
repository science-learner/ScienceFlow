# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Read-only adapters around the existing metric collectors."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from scienceflow.runtime.observability.monitoring.metrics.cpu_monitor import CPUMonitor
from scienceflow.runtime.observability.monitoring.metrics.gpu_monitor import GPUMonitor
from scienceflow.runtime.observability.monitoring.process.process_tracker import ProcessTracker


class CpuMonitorProbe:
    def __init__(self, monitor: CPUMonitor) -> None:
        self.monitor = monitor

    async def sample(self) -> dict[str, Any]:
        return asdict(await self.monitor.poll_once())


class GpuMonitorProbe:
    def __init__(self, monitor: GPUMonitor) -> None:
        self.monitor = monitor

    async def sample(self) -> dict[str, Any]:
        return {"gpus": [asdict(metric) for metric in await self.monitor.poll_once()]}


class ProcessTrackerProbe:
    def __init__(self, tracker: ProcessTracker) -> None:
        self.tracker = tracker

    async def sample(self) -> dict[str, Any]:
        processes: list[dict[str, Any]] = []
        for process in self.tracker.all_processes:
            row = asdict(process)
            row["status"] = process.status.value
            processes.append(row)
        return {"processes": processes}


__all__ = ["CpuMonitorProbe", "GpuMonitorProbe", "ProcessTrackerProbe"]
