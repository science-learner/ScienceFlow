"""Versioned correlation envelope contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class CorrelationEnvelope(VersionedContract):
    correlation_id: str
    source: str
    event_type: str
    run_id: str = ""
    worker_id: str = ""
    process_id: str = ""
    stage_id: str = ""
    event_id: str = ""
    sequence: int = 0
    payload_hash: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["CorrelationEnvelope"]
