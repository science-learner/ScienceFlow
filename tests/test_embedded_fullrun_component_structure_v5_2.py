"""Structural contracts for componentized embedded full-run owners."""

from __future__ import annotations

from pathlib import Path

from scienceflow.research.quality import embedded_fullrun


def test_embedded_fullrun_owners_stay_bounded() -> None:
    quality_root = Path(__file__).parents[1] / "scienceflow/research/quality"
    component_root = quality_root / "fullrun"
    limits = {
        quality_root / "embedded_fullrun.py": 100,
        quality_root / "fullrun/execution.py": 800,
        quality_root / "fullrun/snapshot.py": 800,
    }

    observed = {
        str(path.relative_to(quality_root)): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in limits
    }
    assert all(
        observed[str(path.relative_to(quality_root))] <= limit
        for path, limit in limits.items()
    ), observed
    assert {
        path.name for path in component_root.glob("*.py") if path.name != "__init__.py"
    } == {"execution.py", "snapshot.py"}


def test_embedded_fullrun_compatibility_entries_have_real_owners() -> None:
    assert embedded_fullrun._maybe_write_bare_run_tail_snapshot.__module__ == (
        "scienceflow.research.quality.fullrun.snapshot"
    )
    assert embedded_fullrun._maybe_embedded_full_run_after_quick_test.__module__ == (
        "scienceflow.research.quality.fullrun.execution"
    )
