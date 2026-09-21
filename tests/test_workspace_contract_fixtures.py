from __future__ import annotations

import json
from pathlib import Path


FIXTURE_ROOT = Path(__file__).parents[1] / "docs" / "baselines" / "circle_packing"


def test_circle_packing_workspace_contracts_are_portable_and_complete() -> None:
    expected = {
        "worker1_workspace_contract_manifest.json": 1,
        "worker2_workspace_contract_manifest.json": 2,
    }
    for filename, worker_count in expected.items():
        path = FIXTURE_ROOT / filename
        raw = path.read_text(encoding="utf-8")
        manifest = json.loads(raw)

        assert manifest["schema_version"] == "1.0"
        assert manifest["entry_count"] == len(manifest["entries"])
        assert "/home/" not in raw
        paths = {entry["path"] for entry in manifest["entries"]}
        assert "resolved_config.yaml" in paths
        assert "task_logs/state.json" in paths
        assert "task_logs/lhr_state.json" in paths
        assert "task_logs/lhr_events.jsonl" in paths
        assert "merge/worker_results.json" in paths
        assert "merge/finals/final_00/artifacts/best_solution.json" in paths
        worker_roots = {path.split("/")[1] for path in paths if path.startswith("workers/w")}
        assert len(worker_roots) == worker_count


def test_resolved_config_contracts_redact_provider_values() -> None:
    for path in sorted(FIXTURE_ROOT.glob("*resolved_config_contract.json")):
        raw = path.read_text(encoding="utf-8")
        snapshot = json.loads(raw)
        assert snapshot["schema_version"] == "1.0"
        assert "/home/" not in raw
        code = snapshot["config"]["agent"]["code"]
        assert code["api_key"]["redacted_type"] == "str"
        assert len(code["api_key"]["sha256"]) == 64
        assert code["model"]["redacted_type"] == "str"


def test_post_refactor_workspace_reports_preserve_baseline_contracts() -> None:
    reports = sorted(FIXTURE_ROOT.glob("validation_run*_workspace_contract_report.json"))
    assert len(reports) >= 2
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["schema_version"] == "1.0"
        assert report["mode"] == "compatible"
        assert report["compatible"] is True
        assert report["missing_paths"] == []
        assert report["mismatched_entries"] == []
        assert report["actual_worker_count"] == report["expected_worker_count"]
