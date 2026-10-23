"""Structural contracts for coordinator projection owners."""

from __future__ import annotations

from pathlib import Path

from scienceflow.research.solver.lnr.orchestration.coordinator.resource import advisory
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.stage import commit


def test_coordinator_projection_owners_stay_bounded() -> None:
    root = Path(__file__).parents[1] / "scienceflow/research/solver/lnr/orchestration/coordinator"
    limits = {
        "resource/advisory.py": 600,
        "resource/peer_snapshot.py": 300,
        "lifecycle/stage/commit.py": 600,
        "lifecycle/stage/experiment_state.py": 300,
    }
    observed = {
        name: len((root / name).read_text(encoding="utf-8").splitlines())
        for name in limits
    }
    assert all(observed[name] <= limit for name, limit in limits.items()), observed
    assert len(list((root / "resource").glob("*.py"))) - 1 <= 8
    assert len(list((root / "lifecycle" / "stage").glob("*.py"))) - 1 <= 8


def test_coordinator_projection_compatibility_entries_have_real_owners() -> None:
    assert advisory._parallel_worker_snapshot_for_prompt.__module__ == (
        "scienceflow.research.solver.lnr.orchestration.coordinator.resource.peer_snapshot"
    )
    assert commit._build_stage_commit_experiment_state.__module__ == (
        "scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.stage.experiment_state"
    )
