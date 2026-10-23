from __future__ import annotations

from copy import deepcopy

from tests.support.contracts.workspace_contract_compare import (
    canonical_path,
    compare_manifests,
    normalize_relative_link_target,
)


def manifest(entries: list[dict]) -> dict:
    return {
        "schema_version": "1.0",
        "entry_count": len(entries),
        "entries": entries,
    }


def file(path: str, *, digest: str = "a", rows: int = 1) -> dict:
    return {
        "path": path,
        "kind": "file",
        "mode": 0o644,
        "size": 10,
        "sha256": digest,
        "shape": {
            "format": "jsonl",
            "row_count": rows,
            "invalid_rows": 0,
            "keys": ["type"],
            "key_types": {"type": ["string"]},
            "event_type_counts": {"completed": rows},
        },
    }


def test_compatible_mode_normalizes_registered_runtime_identifiers() -> None:
    baseline = manifest(
        [
            {"path": "workers/w00", "kind": "directory", "mode": 0o755},
            file("workers/w00/logs/artifact_snapshots/iter_0001_20260822T010203Z_best_sha_abcdef123456.json"),
            file("workers/w00/logs/interaction/tool_outputs/tool_000001_bash.txt"),
            file("workers/w00/snapshots/W00-L01-S01-abcdef1234/logs/events.jsonl"),
        ]
    )
    current = manifest(
        [
            {"path": "workers/w07", "kind": "directory", "mode": 0o755},
            file("workers/w07/logs/artifact_snapshots/iter_0009_20260930T111213Z_best_sha_999999999999.json", digest="b"),
            file("workers/w07/logs/interaction/tool_outputs/tool_000042_bash.txt", digest="b", rows=4),
            file("workers/w07/snapshots/W07-L01-S01-9999999999/logs/events.jsonl", digest="b", rows=4),
        ]
    )

    report = compare_manifests(baseline, current, mode="compatible")

    assert report["compatible"] is True
    assert report["missing_paths"] == []


def test_compatible_mode_detects_missing_path_shape_and_worker() -> None:
    baseline = manifest(
        [
            {"path": "workers/w00", "kind": "directory", "mode": 0o755},
            {"path": "workers/w01", "kind": "directory", "mode": 0o755},
            file("workers/w00/logs/lhr_events.jsonl"),
            file("resolved_config.yaml"),
        ]
    )
    current_entries = [
        {"path": "workers/w00", "kind": "directory", "mode": 0o755},
        file("workers/w00/logs/lhr_events.jsonl"),
    ]
    current_entries[-1]["shape"]["keys"] = ["renamed"]

    report = compare_manifests(baseline, manifest(current_entries), mode="compatible")

    assert report["compatible"] is False
    assert report["expected_worker_count"] == 2
    assert report["actual_worker_count"] == 1
    assert "resolved_config.yaml" in report["missing_paths"]
    assert report["mismatched_entries"]


def test_strict_mode_detects_content_and_additional_path_drift() -> None:
    baseline = manifest([file("events.jsonl")])
    current = deepcopy(baseline)
    current["entries"][0]["sha256"] = "changed"
    current["entries"].append({"path": "extra", "kind": "directory", "mode": 0o755})
    current["entry_count"] += 1

    report = compare_manifests(baseline, current, mode="strict")

    assert report["compatible"] is False
    assert report["additional_paths"] == ["extra"]
    assert report["mismatched_entries"][0]["path"] == "events.jsonl"


def test_content_addressed_payloads_are_ignored_but_parent_contract_is_not() -> None:
    assert canonical_path("workers/w00/workspace/.git/objects/aa/deadbeef") is None
    assert canonical_path("workers/w00/snapshots/objects/sha256/aa/bb/hash") is None
    assert canonical_path("workers/w00/workspace/.git/objects/info") is not None
    assert canonical_path("workers/w00/logs/interaction/tool_outputs/tool_000001_bash.txt") is None
    assert (
        canonical_path("workers/w00/workspace/.scienceflow_global_merge/.scienceflow_runs")
        is None
    )


def test_cross_worker_and_local_snapshot_links_share_a_compatible_target() -> None:
    local = "../../../../snapshots/W00-L01-S01-abcdef1234/.run_results.md"
    peer = "../../../../../w01/snapshots/W01-L01-S01-9999999999/.run_results.md"
    assert normalize_relative_link_target(local) == normalize_relative_link_target(peer)
