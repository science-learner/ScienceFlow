"""Execution Value replay, evidence, diff, and safety-audit contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class ExecutionEvidenceRef(VersionedContract):
    kind: str
    reference: str
    digest: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionValueDiff(VersionedContract):
    equal: bool
    changed_fields: tuple[str, ...] = ()
    component_hash: str = ""
    legacy_hash: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionSafetyAudit(VersionedContract):
    effect_free: bool
    safety_veto_required: bool
    near_deliverable: bool
    kill_intent_revalidation_required: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionValueRecord(VersionedContract):
    record_id: str
    observation_hash: str
    selected_source: str
    selected: Mapping[str, Any]
    component: Mapping[str, Any]
    legacy: Mapping[str, Any]
    diff: ExecutionValueDiff
    safety_audit: ExecutionSafetyAudit
    evidence: tuple[ExecutionEvidenceRef, ...] = field(default_factory=tuple)


__all__ = [
    "ExecutionEvidenceRef",
    "ExecutionSafetyAudit",
    "ExecutionValueDiff",
    "ExecutionValueRecord",
]
