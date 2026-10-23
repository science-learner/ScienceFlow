"""Versioned, effect-free Admission contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class AdmissionPolicyObservation(VersionedContract):
    observation_id: str
    command_id: str
    mode: str
    task_card: Mapping[str, Any] = field(default_factory=dict)
    rule_result: Mapping[str, Any] = field(default_factory=dict)
    input_hash: str = ""


@dataclass(frozen=True, slots=True)
class AdmissionPolicyDiff(VersionedContract):
    equal: bool
    changed_fields: tuple[str, ...] = ()
    component_hash: str = ""
    legacy_hash: str = ""


@dataclass(frozen=True, slots=True)
class AdmissionDecisionRecord(VersionedContract):
    record_id: str
    observation_id: str
    selected_source: str
    selected: Mapping[str, Any]
    component: Mapping[str, Any]
    legacy: Mapping[str, Any]
    diff: AdmissionPolicyDiff
    safety_veto_preserved: bool = True


__all__ = [
    "AdmissionDecisionRecord",
    "AdmissionPolicyDiff",
    "AdmissionPolicyObservation",
]
