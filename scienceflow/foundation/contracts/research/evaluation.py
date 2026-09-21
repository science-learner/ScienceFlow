# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Versioned evaluator contracts shared by runtime, evaluators, and gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True)
class EvalContext(VersionedContract):
    """Runtime context passed to an evaluator backend."""

    task_profile: str
    task_id: str
    task_root: Path
    workspace: Path
    worker_id: str
    stage_id: str = ""
    cfg: Any = None
    wall_clock_remaining_sec: float | None = None
    task_python_executable: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRef(VersionedContract):
    """A task-specific candidate artifact described in a generic way."""

    candidate_id: str
    backend: str
    workspace: Path
    artifact_path: str
    artifact_sha: str = ""
    artifact_kind: str = ""
    stage_id: str = ""
    worker_id: str = ""
    created_at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "backend": self.backend,
            "workspace": str(self.workspace),
            "artifact_path": self.artifact_path,
            "artifact_sha": self.artifact_sha,
            "artifact_kind": self.artifact_kind,
            "stage_id": self.stage_id,
            "worker_id": self.worker_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class EvaluationResult(VersionedContract):
    """Backend-native evaluation result before conversion to MetricEvent."""

    backend: str
    ok: bool
    status: str
    metric_value: float | None
    metric_name: str
    lower_is_better: bool | None
    validation_ok: bool
    candidate_ready: bool
    selection_eligible: bool
    metric_validity: str
    reason_code: str = ""
    message: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ok": self.ok,
            "status": self.status,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "validation_ok": self.validation_ok,
            "candidate_ready": self.candidate_ready,
            "selection_eligible": self.selection_eligible,
            "metric_validity": self.metric_validity,
            "reason_code": self.reason_code,
            "message": self.message,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class MetricEvent(VersionedContract):
    """Single normalized metric event consumed by gates and reporting."""

    candidate_id: str
    worker_id: str
    stage_id: str
    metric_value: float | None
    metric_name: str
    lower_is_better: bool | None
    validation_ok: bool
    candidate_ready: bool
    selection_eligible: bool
    metric_validity: str
    metric_validity_reason_code: str
    artifact_path: str
    artifact_sha: str
    evaluator_backend: str
    evaluator_status: str
    metric_type: str
    task_profile: str = ""
    metric_note: str = ""
    run_time_sec: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "worker_id": self.worker_id,
            "stage_id": self.stage_id,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "validation_ok": self.validation_ok,
            "candidate_ready": self.candidate_ready,
            "selection_eligible": self.selection_eligible,
            "metric_validity": self.metric_validity,
            "metric_validity_reason_code": self.metric_validity_reason_code,
            "artifact_path": self.artifact_path,
            "artifact_sha": self.artifact_sha,
            "evaluator_backend": self.evaluator_backend,
            "evaluator_status": self.evaluator_status,
            "metric_type": self.metric_type,
            "task_profile": self.task_profile,
            "metric_note": self.metric_note,
            "run_time_sec": self.run_time_sec,
            "extra": dict(self.extra or {}),
        }


@dataclass(frozen=True)
class EvaluationRequest(VersionedContract):
    """One framework-level request to evaluate the current candidate."""

    context: EvalContext
    trigger: str = "stage_end"
