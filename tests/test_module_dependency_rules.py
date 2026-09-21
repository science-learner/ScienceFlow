from __future__ import annotations

from pathlib import Path

from scienceflow.foundation.architecture import DependencyRule, find_dependency_violations


def test_repository_module_dependencies_are_valid() -> None:
    package_root = Path(__file__).resolve().parents[1] / "scienceflow"
    assert find_dependency_violations(package_root) == []


def test_dependency_checker_reports_exact_import(tmp_path: Path) -> None:
    source = tmp_path / "execution_value"
    source.mkdir()
    path = source / "bad.py"
    path.write_text(
        "from scienceflow.research.control.resources import ResourceManagementService\n",
        encoding="utf-8",
    )
    rules = (
        DependencyRule(
            name="value-does-not-control-resources",
            source="execution_value",
            forbidden=("scienceflow.research.control.resources",),
            exclude_parts=(),
        ),
    )

    violations = find_dependency_violations(tmp_path, rules=rules)

    assert len(violations) == 1
    assert violations[0].path == path
    assert violations[0].line == 1
    assert violations[0].imported == "scienceflow.research.control.resources"
