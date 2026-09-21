# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Contracts connecting authoritative workspace facts to memory projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class StageRecord(VersionedContract):
    stage_id: str
    metric: str = ""
    lower_is_better: str = ""
    metric_validity: str = ""
    selection_eligible: str = ""
    brief: str = ""
    why: str = ""
    files: str = ""
    body: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "metric": self.metric,
            "lower_is_better": self.lower_is_better,
            "metric_validity": self.metric_validity,
            "selection_eligible": self.selection_eligible,
            "brief": self.brief,
            "why": self.why,
            "files": self.files,
            "body": self.body,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class StageCommitted(VersionedContract):
    """Idempotent event published after an authoritative stage is visible."""

    event_id: str
    workspace_id: str
    stage: StageRecord
    sequence: int
    occurred_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.contract_schema_version(),
            "event_type": "StageCommitted",
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "stage": self.stage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MemoryView(VersionedContract):
    text: str
    source_event_ids: tuple[str, ...] = ()
    last_sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_event_ids": list(self.source_event_ids),
            "last_sequence": self.last_sequence,
        }


__all__ = ["MemoryView", "StageCommitted", "StageRecord"]
