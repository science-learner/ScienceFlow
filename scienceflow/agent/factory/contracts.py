"""Versioned Agent Factory request and trace contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class AgentBuildRequest(VersionedContract):
    role: str
    options: Mapping[str, Any] = field(default_factory=dict)
    correlation: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentBuildRecord(VersionedContract):
    build_id: str
    sequence: int
    role: str
    option_keys: tuple[str, ...]
    correlation: Mapping[str, str]
    outcome: str
    error_type: str = ""


__all__ = ["AgentBuildRecord", "AgentBuildRequest"]
