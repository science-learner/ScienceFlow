# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Versioned EStra observation and command contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class EstraContext(VersionedContract):
    latest_stage: str
    switch_candidates: tuple[str, ...]
    trigger_source: str = "manual"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EstraDecision(VersionedContract):
    """A command for Runtime; it does not mutate workspace or memory."""

    action: str
    startpoint: str
    intent: str
    target_stage: str
    compact: bool
    reason: str = ""
    diagnostics: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "startpoint": self.startpoint,
            "intent": self.intent,
            "target_stage": self.target_stage,
            "compact": self.compact,
            "reason": self.reason,
            **dict(self.diagnostics or {}),
        }


__all__ = ["EstraContext", "EstraDecision"]
