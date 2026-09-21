"""Structural contracts for dataset scan owners."""

from __future__ import annotations

from pathlib import Path

from scienceflow.research.state.dataset.discovery.scan import extract_eval_signature


def test_dataset_scan_owners_stay_bounded() -> None:
    root = Path(__file__).parents[1] / "scienceflow/research/state/dataset"
    limits = {
        "discovery/scan.py": 200,
        "discovery/scan_metadata.py": 800,
        "rendering/scan_render.py": 800,
        "discovery/scan_traversal.py": 200,
        "rendering/scan_tree.py": 100,
        "rendering/scan_tree_flat.py": 250,
        "rendering/scan_tree_nested.py": 800,
        "discovery/eval_signature.py": 120,
    }

    observed = {
        name: len((root / name).read_text(encoding="utf-8").splitlines())
        for name in limits
    }
    business_modules = {
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "__init__.py"
    }
    assert business_modules == set(limits), business_modules
    assert all(observed[name] <= limit for name, limit in limits.items()), observed
    assert len(business_modules) == 8


def test_eval_signature_keeps_scan_compatibility_entry() -> None:
    assert extract_eval_signature.__module__ == (
        "scienceflow.research.state.dataset.discovery.eval_signature"
    )
