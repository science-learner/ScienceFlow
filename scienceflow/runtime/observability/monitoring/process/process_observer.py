# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Read-only adapter publishing process observations to the monitoring plane."""

from __future__ import annotations

from dataclasses import dataclass

from scienceflow.runtime.observability.monitoring.runtime.service import MonitorService
from inquirycraft.runtime import ProcessObservation


@dataclass(frozen=True, slots=True)
class ProcessObservationPublisher:
    """Translate a process event without gaining process-control authority."""

    monitor: MonitorService
    run_id: str = ""
    worker_id: str = ""

    async def __call__(self, observation: ProcessObservation) -> None:
        await self.monitor.observe(
            observation.kind,
            payload={
                "session_id": observation.session_id,
                "process_group_id": observation.process_group_id,
                "stream": (
                    observation.stream.value if observation.stream is not None else ""
                ),
                "chunk_sequence": observation.chunk_sequence,
                "byte_count": observation.byte_count,
                "returncode": observation.returncode,
                "detail": observation.detail,
            },
            run_id=self.run_id,
            worker_id=self.worker_id,
            correlation_id=f"{observation.session_id}:{observation.sequence}",
        )


__all__ = ["ProcessObservationPublisher"]
