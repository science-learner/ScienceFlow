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

import hashlib
import json
from pathlib import Path

from scienceflow.research.quality.finalization.selection.coverage import (
    build_final_coverage_plan,
    order_candidates_for_coverage,
)
from scienceflow.research.quality.finalization.candidates.evidence import (
    apply_candidate_evidence,
    load_archived_artifact_candidates,
    load_peer_candidate_evidence,
    recover_candidate_artifact,
)
from scienceflow.research.quality.finalization.candidates.pack import pack_candidates
from scienceflow.research.quality.finalization.selection.ranking import ranked_candidates


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_peer_evidence_recovers_failed_worker_submission(tmp_path: Path) -> None:
    worker = tmp_path / "workers" / "w01"
    submission = (
        worker / "logs" / "submission_snapshots" / "iter_0006_s03_submission.csv"
    )
    _write(submission, "id,target\n1,0.3\n")
    sha = hashlib.sha256(submission.read_bytes()).hexdigest()
    peer_csv = tmp_path / "task_logs" / "lhr_stage_performance.csv"
    _write(
        peer_csv,
        "worker_id,stage_id,metric_value,metric_validity,lineage_id,submission_sha,"
        "submission_snapshot,candidate_ready,selection_eligible\n"
        f"W01,S03,0.3,medium,L01,{sha},"
        "../logs/submission_snapshots/iter_0006_s03_submission.csv,1,1\n",
    )

    evidence = load_peer_candidate_evidence(peer_csv, worker_id="W01")
    candidate = apply_candidate_evidence(
        {
            "candidate_id": "W01:S03",
            "worker_id": "W01",
            "stage_id": "S03",
            "worker_root": str(worker),
            "snapshot_path": str(worker / "snapshots" / "missing"),
        },
        evidence["S03"],
    )
    recovered = recover_candidate_artifact(candidate, artifact_path="submission.csv")

    assert recovered["candidate_ready"] is True
    assert recovered["submission_sha"] == sha
    assert Path(recovered["artifact_source"]) == submission.resolve()


def test_candidate_without_sha_does_not_claim_current_workspace_artifact(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "workers" / "w01"
    _write(worker / "workspace" / "submission.csv", "id,target\n1,current\n")

    recovered = recover_candidate_artifact(
        {
            "candidate_id": "W01:S03",
            "worker_id": "W01",
            "stage_id": "S03",
            "worker_root": str(worker),
            "snapshot_path": str(worker / "snapshots" / "missing"),
        },
        artifact_path="submission.csv",
    )

    assert recovered["candidate_ready"] is False
    assert "artifact_source" not in recovered


def test_archived_validated_artifact_is_recovered_without_committed_stage(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "workers" / "w00"
    (worker / "workspace").mkdir(parents=True)
    artifact = worker / "logs" / "submission_snapshots" / "iter_0002_submission.csv"
    _write(artifact, "id,target\n1,0.42\n")
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    archive_record = {
        "artifact_path": "submission.csv",
        "artifact_sha256": sha,
        "capture_type": "candidate_artifact_persisted",
        "snapshot_path": "../logs/submission_snapshots/iter_0002_submission.csv",
    }
    evaluator_event = {
        "event": "evaluator_metric_event",
        "worker_id": "W00",
        "candidate_id": "W00:S02",
        "stage_id": "S02",
        "artifact_path": "submission.csv",
        "artifact_sha": sha,
        "candidate_ready": True,
        "validation_ok": True,
        "selection_eligible": False,
        "evaluator_backend": "task_package",
        "evaluator_status": "validated_no_metric",
        "metric_validity": "medium",
        "metric_value": None,
    }
    _write(
        worker / "logs" / "checkpoints" / "artifact_archive.jsonl",
        json.dumps(archive_record) + "\n",
    )
    _write(
        worker / "logs" / "evaluator_events.jsonl",
        json.dumps(evaluator_event) + "\n",
    )

    candidates = load_archived_artifact_candidates(
        worker,
        worker_id="W00",
        artifact_path="submission.csv",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["candidate_ready"] is True
    assert candidate["final_artifact_only"] is True
    assert candidate["selection_eligible"] is False
    assert candidate["submission_sha"] == sha
    assert Path(candidate["artifact_source"]) == artifact.resolve()

    packed = pack_candidates(
        merge_dir=tmp_path / "merge",
        merge_workspace=tmp_path / "merge_workspace",
        candidates=candidates,
        artifact_path="submission.csv",
        ledger_filename=".run_results.md",
    )
    assert [item["submission_sha"] for item in packed] == [sha]


def test_json_recovery_prefers_artifact_sha_over_conflicting_submission_sha(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    artifact = snapshot / "artifacts" / "submission.json"
    _write(artifact, '{"candidate":"value"}\n')
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()

    recovered = recover_candidate_artifact(
        {
            "candidate_id": "W00:L01:S01",
            "snapshot_path": str(snapshot),
            "artifact_path": "artifacts/submission.json",
            "artifact_sha": artifact_sha,
            "submission_sha": "0" * 64,
        },
        artifact_path="artifacts/submission.json",
    )

    assert recovered["candidate_ready"] is True
    assert recovered["artifact_sha"] == artifact_sha


def test_json_recovery_does_not_use_legacy_submission_sha(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    artifact = snapshot / "submission.json"
    _write(artifact, '{"candidate":"value"}\n')

    recovered = recover_candidate_artifact(
        {
            "candidate_id": "W00:L01:S01",
            "snapshot_path": str(snapshot),
            "submission_sha": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        artifact_path="submission.json",
    )

    assert recovered["candidate_ready"] is False
    assert "artifact_source" not in recovered


def test_pack_candidates_deduplicates_and_preserves_diversity(tmp_path: Path) -> None:
    specs = [
        ("W00:S01", "W00", "L01", 0.9, "a"),
        ("W00:S02", "W00", "L01", 0.8, "a"),
        ("W00:S03", "W00", "L02", 0.7, "b"),
        ("W01:S01", "W01", "L01", 0.1, "c"),
    ]
    candidates = []
    for candidate_id, worker_id, lineage_id, metric, value in specs:
        snap = tmp_path / candidate_id.replace(":", "-")
        _write(snap / "submission.csv", f"id,target\n1,{value}\n")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "worker_id": worker_id,
                "lineage_id": lineage_id,
                "snapshot_path": str(snap),
                "metric_value": metric,
                "lower_is_better": False,
                "submission_sha": "placeholder",
                "metric_validity": "medium",
                "selection_eligible": True,
            }
        )

    packed = pack_candidates(
        merge_dir=tmp_path / "merge",
        merge_workspace=tmp_path / "merge_ws",
        candidates=candidates,
        artifact_path="submission.csv",
        ledger_filename=".run_results.md",
        max_candidates=3,
    )

    assert {candidate["worker_id"] for candidate in packed} == {"W00", "W01"}
    assert {candidate["lineage_id"] for candidate in packed} == {"L01", "L02"}
    assert len({candidate["submission_sha"] for candidate in packed}) == 3


def test_meta_fit_candidate_ranks_after_comparable_candidate() -> None:
    ranked = ranked_candidates(
        [
            {
                "candidate_id": "W00:S01",
                "candidate_ready": True,
                "submission_sha": "meta-fit",
                "selection_eligible": True,
                "metric_validity": "medium",
                "metric_validity_reason_code": "same_validation_meta_fit",
                "metric_value": 0.9,
                "lower_is_better": False,
            },
            {
                "candidate_id": "W01:S01",
                "candidate_ready": True,
                "submission_sha": "comparable",
                "selection_eligible": True,
                "metric_validity": "medium",
                "metric_value": 0.5,
                "lower_is_better": False,
            },
        ]
    )

    assert [candidate["candidate_id"] for candidate in ranked] == [
        "W01:S01",
        "W00:S01",
    ]


def test_final_coverage_reserves_safe_medium_route_without_admitting_low() -> None:
    def candidate(
        candidate_id: str,
        *,
        metric: float,
        validity: str,
        reason: str = "",
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "candidate_ready": True,
            "submission_sha": candidate_id,
            "selection_eligible": validity != "low",
            "metric_validity": validity,
            "metric_validity_reason_code": reason,
            "metric_value": metric,
            "lower_is_better": True,
        }

    ranked = ranked_candidates(
        [
            candidate("high-best", metric=0.10, validity="high"),
            candidate("high-second", metric=0.20, validity="high"),
            candidate("medium-safe", metric=0.05, validity="medium"),
            candidate(
                "medium-meta-fit",
                metric=0.01,
                validity="medium",
                reason="same_validation_meta_fit",
            ),
            candidate("low", metric=0.001, validity="low"),
        ]
    )
    plan = build_final_coverage_plan(ranked, final_slots=3)
    ordered = order_candidates_for_coverage(
        ranked,
        plan=plan,
        coverage_first=False,
    )

    assert plan.enabled is True
    assert plan.reserved_slots == 1
    assert plan.coverage_candidate_id == "medium-safe"
    assert [row["candidate_id"] for row in ordered[:3]] == [
        "high-best",
        "medium-safe",
        "high-second",
    ]
    assert "low" not in {row["candidate_id"] for row in ordered}
