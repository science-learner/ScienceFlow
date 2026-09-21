"""Static gates for the V4 functional namespace aggregation."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "scienceflow"
EXPECTED_TOP_LEVEL_PACKAGES = {
    "agent",
    "foundation",
    "interfaces",
    "research",
    "runtime",
}
REMOVED_TOP_LEVEL_PACKAGES = {
    "admission",
    "agent_factory",
    "assessment",
    "core",
    "estra",
    "evaluator",
    "execution_value",
    "finalization",
    "gate",
    "gates",
    "memory",
    "monitoring",
    "parallel",
    "process_runtime",
    "prompt_context",
    "resource_management",
    "runtime_kernel",
    "stage_lifecycle",
    "telemetry",
    "utils",
    "workspace",
}


def test_top_level_packages_match_functional_families() -> None:
    actual = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert actual == EXPECTED_TOP_LEVEL_PACKAGES
    assert len(actual) <= 5
    assert all(not (PACKAGE_ROOT / name).exists() for name in REMOVED_TOP_LEVEL_PACKAGES)


def test_package_root_contains_only_marker_and_cli_compatibility_entry() -> None:
    assert {path.name for path in PACKAGE_ROOT.glob("*.py")} == {
        "__init__.py",
        "cli.py",
    }


def test_no_top_level_package_has_more_than_eight_direct_python_files() -> None:
    counts = {
        path.name: len(tuple(path.glob("*.py")))
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert max(counts.values(), default=0) <= 8, counts


def test_organizational_namespaces_do_not_barrel_import_components() -> None:
    for name in ("foundation", "interfaces", "research", "runtime"):
        marker = PACKAGE_ROOT / name / "__init__.py"
        tree = ast.parse(marker.read_text(encoding="utf-8"), filename=str(marker))
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
