# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Read-only translation from legacy LNR observations to value contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scienceflow.research.control.execution_value.decision.observation import ExecutionObservationBuilder
from scienceflow.research.control.execution_value.evidence.replay import ExecutionValueReplayArchive
from scienceflow.research.control.execution_value.decision.router import ExecutionValueDecisionRouter
from scienceflow.research.control.execution_value.decision.service import ExecutionValueService


class LnrExecutionValueShadow:
    """Evaluate legacy facts without acquiring, releasing, or killing resources."""

    def __init__(
        self,
        service: ExecutionValueService | None = None,
        *,
        decision_source: str = "component",
        archive_path: str | Path | None = None,
    ) -> None:
        self.service = service or ExecutionValueService()
        self.builder = ExecutionObservationBuilder()
        self.archive = ExecutionValueReplayArchive(archive_path) if archive_path else None
        self.router = ExecutionValueDecisionRouter(
            source=decision_source,
            service=self.service,
            archive=self.archive,
        )
        self.last_record = None

    def decide(
        self,
        *,
        command_id: str,
        elapsed_sec: float,
        signal: Mapping[str, Any],
        artifact_progress: bool,
        recoverable_artifact: bool,
    ) -> dict[str, Any]:
        observation = self.builder.from_lnr(
            command_id=command_id,
            elapsed_sec=elapsed_sec,
            signal=signal,
            artifact_progress=artifact_progress,
            recoverable_artifact=recoverable_artifact,
        )
        selected, record = self.router.decide(observation, signal=signal)
        self.last_record = record
        return selected


__all__ = ["LnrExecutionValueShadow"]
