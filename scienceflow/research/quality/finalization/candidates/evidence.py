# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

# Canonical owner: scienceflow.research.quality.finalization.

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from scienceflow.research.quality.finalization.selection.ranking import (
    ranked_finalization_candidates,
)


_BOOL_FIELDS = {
    "candidate_ready",
    "gate_accepted",
    "lower_is_better",
    "selection_eligible",
    "validation_ok",
}
_FLOAT_FIELDS = {"metric_value", "reported_val_score", "selection_score"}
_INT_FIELDS = {"memory_cut"}
_EVIDENCE_FIELDS = {
    "artifact_path",
    "artifact_sha",
    "candidate_ready",
    "duplicate_submission_of_stage",
    "execution_mode",
    "gate_accepted",
    "gate_action",
    "gate_metric_validity",
    "gate_policy",
    "gate_policy_version",
    "gate_reason_code",
    "lineage_id",
    "lower_is_better",
    "memory_cut",
    "metric_eval_data",
    "metric_name",
    "metric_protocol",
    "metric_source_note",
    "metric_validity",
    "metric_validity_note",
    "metric_validity_reason_code",
    "metric_value",
    "node_uid",
    "reported_val_score",
    "route_id",
    "selection_eligible",
    "selection_note",
    "selection_score",
    "snapshot_id",
    "snapshot_path",
    "solution_sha",
    "submission_sha",
    "submission_snapshot",
    "submission_status",
    "train_data_used",
    "val_score_type",
    "validation_issue",
    "validation_ok",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_csv_value(key: str, value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return None
    if key in _BOOL_FIELDS:
        lowered = text.lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
        return None
    if key in _FLOAT_FIELDS:
        try:
            return float(text)
        except ValueError:
            return None
    if key in _INT_FIELDS:
        try:
            return int(float(text))
        except ValueError:
            return None
    return text


def load_peer_candidate_evidence(
    path: Path,
    *,
    worker_id: str,
) -> dict[str, dict[str, Any]]:
    """Load the latest lightweight peer row for each stage of one worker."""

    if not path.is_file():
        return {}
    wanted = str(worker_id or "").strip().upper()
    evidence: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("worker_id") or "").strip().upper() != wanted:
                    continue
                stage_id = str(row.get("stage_id") or "").strip().upper()
                if not stage_id:
                    continue
                parsed = {
                    key: converted
                    for key in _EVIDENCE_FIELDS
                    if (converted := _coerce_csv_value(key, str(row.get(key) or "")))
                    is not None
                }
                evidence[stage_id] = parsed
    except (OSError, csv.Error):
        return {}
    return evidence


def apply_candidate_evidence(
    candidate: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(candidate)
    for key, value in (evidence or {}).items():
        if key in _EVIDENCE_FIELDS and value not in (None, ""):
            merged[key] = value
    return merged


def _expected_artifact_sha(candidate: dict[str, Any], *, artifact_path: str) -> str:
    artifact_sha = str(candidate.get("artifact_sha") or "").strip()
    if artifact_sha:
        return artifact_sha
    candidate_artifact = str(candidate.get("artifact_path") or artifact_path or "")
    if Path(candidate_artifact).suffix.lower() == ".csv":
        return str(candidate.get("submission_sha") or "").strip()
    return ""


def _candidate_paths(candidate: dict[str, Any], *, artifact_path: str) -> list[Path]:
    worker_text = str(candidate.get("worker_root") or "").strip()
    snapshot_text = str(candidate.get("snapshot_path") or "").strip()
    worker_root = Path(worker_text) if worker_text else None
    snapshot = Path(snapshot_text) if snapshot_text else None
    rel_artifact = str(
        candidate.get("artifact_path") or artifact_path or "submission.csv"
    )
    paths: list[Path] = []

    explicit = str(candidate.get("artifact_source") or "").strip()
    if explicit:
        paths.append(Path(explicit))
    if snapshot is not None:
        paths.append(snapshot / rel_artifact)

    archived = str(candidate.get("submission_snapshot") or "").strip()
    if archived:
        archived_path = Path(archived)
        if archived_path.is_absolute():
            paths.append(archived_path)
        if worker_root is not None:
            paths.extend(
                (
                    worker_root / "workspace" / archived_path,
                    worker_root / archived_path,
                    worker_root / "logs" / "submission_snapshots" / archived_path.name,
                    worker_root / "logs" / "artifact_snapshots" / archived_path.name,
                )
            )

    expected_sha = _expected_artifact_sha(candidate, artifact_path=artifact_path)
    if worker_root is not None and expected_sha:
        paths.append(worker_root / "workspace" / rel_artifact)
    return paths


def _stage_snapshot_fallbacks(candidate: dict[str, Any]) -> Iterable[Path]:
    worker_text = str(candidate.get("worker_root") or "").strip()
    stage_id = str(candidate.get("stage_id") or "").strip().lower()
    if not worker_text or not stage_id:
        return ()
    worker_root = Path(worker_text)
    stage_number = stage_id.removeprefix("s").lstrip("0") or "0"
    marker = re.compile(rf"(?:^|[_-])s0*{re.escape(stage_number)}(?:[_-]|$)", re.I)
    roots = (
        worker_root / "logs" / "submission_snapshots",
        worker_root / "logs" / "artifact_snapshots",
    )
    return tuple(
        path
        for root in roots
        if root.is_dir()
        for path in sorted(root.iterdir(), reverse=True)
        if path.is_file() and marker.search(path.name)
    )


def recover_candidate_artifact(
    candidate: dict[str, Any],
    *,
    artifact_path: str,
) -> dict[str, Any]:
    """Resolve a stable local artifact and verify its content identity."""

    recovered = dict(candidate)
    expected_sha = _expected_artifact_sha(recovered, artifact_path=artifact_path)
    if not expected_sha:
        recovered["candidate_ready"] = False
        return recovered
    candidates = [*_candidate_paths(recovered, artifact_path=artifact_path)]
    candidates.extend(_stage_snapshot_fallbacks(recovered))
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            actual_sha = _sha256_file(resolved)
        except OSError:
            continue
        verified_sha = re.fullmatch(r"[0-9a-fA-F]{12,64}", expected_sha)
        if verified_sha and not actual_sha.startswith(expected_sha.lower()):
            continue
        recovered.update(
            {
                "artifact_source": str(resolved),
                "artifact_sha": actual_sha,
                "submission_sha": actual_sha,
                "candidate_ready": True,
                "submission_status": str(recovered.get("submission_status") or "ok"),
            }
        )
        return recovered

    recovered["candidate_ready"] = False
    return recovered


def _jsonl_dicts(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def load_archived_artifact_candidates(
    worker_root: Path,
    *,
    worker_id: str,
    artifact_path: str,
) -> list[dict[str, Any]]:
    """Recover evaluator-validated artifacts that never became committed stages.

    The archive ledger supplies immutable artifact identity and location; the
    evaluator event supplies validation evidence.  Requiring both prevents the
    mutable current workspace artifact from being mistaken for an older result.
    """

    root = worker_root.resolve()
    archive_file = root / "logs" / "checkpoints" / "artifact_archive.jsonl"
    evaluator_file = root / "logs" / "evaluator_events.jsonl"
    if not archive_file.is_file() or not evaluator_file.is_file():
        return []

    archived: dict[str, Path] = {}
    for record in _jsonl_dicts(archive_file):
        sha = str(record.get("artifact_sha256") or "").strip().lower()
        snapshot_text = str(record.get("snapshot_path") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{12,64}", sha) or not snapshot_text:
            continue
        snapshot = Path(snapshot_text)
        if not snapshot.is_absolute():
            snapshot = root / "workspace" / snapshot
        try:
            resolved = snapshot.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            archived[sha] = resolved

    recovered_by_sha: dict[str, dict[str, Any]] = {}
    wanted_worker = str(worker_id or "").strip().upper()
    for event in _jsonl_dicts(evaluator_file):
        if event.get("event") != "evaluator_metric_event":
            continue
        event_worker = str(event.get("worker_id") or "").strip().upper()
        if event_worker and event_worker != wanted_worker:
            continue
        if (
            event.get("candidate_ready") is not True
            or event.get("validation_ok") is not True
        ):
            continue
        event_artifact_path = str(event.get("artifact_path") or "").strip()
        if artifact_path and event_artifact_path != artifact_path:
            continue
        sha = str(event.get("artifact_sha") or "").strip().lower()
        source = archived.get(sha)
        if source is None:
            continue
        original_id = str(event.get("candidate_id") or wanted_worker or "candidate")
        candidate = {
            "candidate_id": f"{original_id}:artifact:{sha[:12]}",
            "worker_id": wanted_worker,
            "worker_root": str(root),
            "stage_id": str(event.get("stage_id") or ""),
            "artifact_path": event_artifact_path,
            "artifact_source": str(source),
            "artifact_sha": sha,
            "submission_sha": sha,
            "submission_snapshot": str(source),
            "submission_status": str(
                event.get("evaluator_status") or "validated_artifact"
            ),
            "candidate_ready": True,
            "validation_ok": True,
            "selection_eligible": event.get("selection_eligible") is True,
            "metric_name": str(event.get("metric_name") or ""),
            "metric_value": event.get("metric_value"),
            "metric_validity": str(event.get("metric_validity") or ""),
            "metric_validity_reason_code": str(
                event.get("metric_validity_reason_code") or ""
            ),
            "lower_is_better": event.get("lower_is_better"),
            "evaluator_backend": str(event.get("evaluator_backend") or ""),
            "evaluator_status": str(event.get("evaluator_status") or ""),
            "timestamp": event.get("timestamp"),
            "final_artifact_only": True,
        }
        recovered = recover_candidate_artifact(candidate, artifact_path=artifact_path)
        if recovered.get("candidate_ready") is True:
            recovered_by_sha[sha] = recovered
    return list(recovered_by_sha.values())


def prepare_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    artifact_path: str,
) -> list[dict[str, Any]]:
    return [
        recover_candidate_artifact(candidate, artifact_path=artifact_path)
        for candidate in candidates
    ]


def _identity(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("submission_sha")
        or candidate.get("artifact_sha")
        or candidate.get("candidate_id")
        or ""
    )


def _group_key(candidate: dict[str, Any], field: str) -> str:
    value = str(candidate.get(field) or "").strip()
    if value:
        return value
    if field == "lineage_id":
        return str(
            candidate.get("route_id") or candidate.get("solution_sha") or ""
        ).strip()
    return ""


def select_diverse_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Select unique candidates while preserving worker and route coverage."""

    ranked = ranked_finalization_candidates(list(candidates))
    unique: list[dict[str, Any]] = []
    seen_identity: set[str] = set()
    for candidate in ranked:
        identity = _identity(candidate)
        if not identity or identity in seen_identity:
            continue
        seen_identity.add(identity)
        unique.append(candidate)

    limit = max(0, int(max_candidates))
    selected: list[dict[str, Any]] = []
    selected_identity: set[str] = set()

    def add(candidate: dict[str, Any]) -> None:
        identity = _identity(candidate)
        if len(selected) >= limit or not identity or identity in selected_identity:
            return
        selected.append(candidate)
        selected_identity.add(identity)

    for field in ("worker_id", "lineage_id"):
        seen_groups: set[str] = set()
        for candidate in unique:
            group = _group_key(candidate, field)
            if not group or group in seen_groups:
                continue
            seen_groups.add(group)
            add(candidate)
    for candidate in unique:
        add(candidate)
    return selected
