from __future__ import annotations

import csv
from pathlib import Path

from tools.architecture.v5_2_structure_metrics import build_report, main


def test_structure_report_tracks_real_facades_and_test_seams() -> None:
    report = build_report(include_complexity=False)

    assert report["callable_descriptor_bindings"] == {
        "LHRResourceObserver": 0,
        "LnrSolver": 234,
        "ResourceRuntime": 0,
        "ScienceAgent": 68,
    }
    seams = report["test_seams"]
    assert seams["object_new_lnr_solver"] == 0
    # Current migration seam: any increase requires an ownership refactor or
    # an explicit baseline review instead of silently expanding the surface.
    assert seams["solver_private_attribute_access"] <= 190
    assert seams["observer_private_attribute_access"] <= 181
    packages = {row["path"]: row for row in report["packages"]}
    coordinator = packages["scienceflow/research/solver/lnr/orchestration/coordinator"]
    assert coordinator["direct_business_modules"] == 2
    assert "scienceflow/research/solver/lnr/orchestration/coordinator/shared.py" in coordinator["shared_modules"]
    finalization = packages["scienceflow/research/quality/finalization"]
    assert finalization["direct_business_modules"] == 3
    assert finalization["sys_modules_proxies"] == []


def test_owner_matrix_has_no_unassigned_binding(tmp_path: Path) -> None:
    matrix = tmp_path / "owner.csv"
    assert main(["--owner-matrix", str(matrix), "--output", str(tmp_path / "report.json")]) == 0

    with matrix.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 302
    assert all(row["target_owner"] for row in rows)
    assert {row["facade"] for row in rows} == {
        "LnrSolver",
        "ScienceAgent",
    }
