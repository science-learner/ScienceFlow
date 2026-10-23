"""Stable inputs for one ScienceFlow agent session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AgentSessionSpec:
    request: str | None
    workspace: Path
    model: str
    max_turns: int
    first_round_tool_choice: str | None = None
    run_id: str = ""
    session_id: str = ""
    worker_id: str = ""
    stage_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["AgentSessionSpec"]
