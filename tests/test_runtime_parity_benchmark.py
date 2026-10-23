from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.support.runtime_parity.comparison.compare import strict_differences
from tests.support.runtime_parity.comparison.normalize import (
    NORMALIZATION_RULES,
    normalize_value,
)
from tests.support.runtime_parity.execution.runner import REPO_ROOT, load_manifest
from tests.support.runtime_parity.cases.scenarios import SCENARIOS


def test_runtime_parity_manifest_references_registered_cases() -> None:
    manifest = load_manifest()
    quick = set(manifest["suites"]["quick"])
    full = set(manifest["suites"]["full"])
    assert quick
    assert quick <= full
    assert full <= set(SCENARIOS)


def test_strict_comparator_reports_unknown_drift() -> None:
    differences = strict_differences(
        {"messages": [{"role": "user", "content": "same"}]},
        {"messages": [{"role": "user", "content": "changed"}], "extra": True},
    )
    assert [difference["path"] for difference in differences] == [
        "$.extra",
        "$.messages[0].content",
    ]


def test_normalization_registry_is_explicit() -> None:
    assert set(NORMALIZATION_RULES) == {
        "ansi_escape",
        "duration_sec",
        "elapsed_sec",
        "event_id",
        "pgid",
        "pid",
        "runtime_audit_artifacts",
        "timestamp",
        "tpot_ms",
        "ttft_sec",
        "workspace_root",
    }
    assert all(rule.get("approved_stage") for rule in NORMALIZATION_RULES.values())
    assert all(rule.get("risk") for rule in NORMALIZATION_RULES.values())
    assert all(
        rule.get("json_path") or rule.get("workspace_path")
        for rule in NORMALIZATION_RULES.values()
    )


def test_registered_timing_fields_normalize_old_null_and_observed_values() -> None:
    expected = {"ttft_sec": None, "tpot_ms": None}
    actual = {"ttft_sec": 0.125, "tpot_ms": 4.5}
    assert normalize_value(expected) == normalize_value(actual) == {
        "tpot_ms": "<duration>",
        "ttft_sec": "<duration>",
    }


def test_quick_runtime_parity_passes_in_parallel(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests" / "support" / "run_runtime_parity.py"),
            "--suite",
            "quick",
            "--jobs",
            "4",
            "--output",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "pass"
    assert report["totals"] == {
        "context_diff_count": 0,
        "failed_cases": 0,
        "mechanism_diff_count": 0,
        "workspace_unregistered_diff_count": 0,
    }


def test_full_runtime_parity_passes_in_parallel(tmp_path: Path) -> None:
    report_path = tmp_path / "full-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tests" / "support" / "run_runtime_parity.py"),
            "--suite",
            "full",
            "--jobs",
            "4",
            "--output",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "pass"
    assert {case["case_id"] for case in report["cases"]} >= {
        "snapshot_stop_resume_recovery",
        "worker2_estra_merge",
    }
    assert report["totals"] == {
        "context_diff_count": 0,
        "failed_cases": 0,
        "mechanism_diff_count": 0,
        "workspace_unregistered_diff_count": 0,
    }
