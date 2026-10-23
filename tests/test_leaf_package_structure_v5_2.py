"""Repository-wide fan-out boundary for focused ScienceFlow components."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "scienceflow"
MAX_CHILD_DIRECTORIES = 5
MAX_DIRECT_MODULES = 5
# Explicit product surfaces may exceed the default only through a reviewed,
# path-specific ceiling. These values prevent further silent fan-out growth.
CHILD_DIRECTORY_LIMITS = {"research": 6}
DIRECT_MODULE_LIMITS = {
    "interfaces/ui": 6,
    "research/reporting": 10,
}


def _package_directories() -> tuple[Path, ...]:
    return tuple(
        package
        for package in (PACKAGE_ROOT, *PACKAGE_ROOT.rglob("*"))
        if package.is_dir() and "__pycache__" not in package.parts
    )


def test_every_directory_stays_within_child_directory_limit() -> None:
    violations: dict[str, list[str]] = {}
    for package in _package_directories():
        children = sorted(
            child.name
            for child in package.iterdir()
            if child.is_dir() and child.name != "__pycache__"
        )
        relative = str(package.relative_to(PACKAGE_ROOT))
        limit = CHILD_DIRECTORY_LIMITS.get(relative, MAX_CHILD_DIRECTORIES)
        if len(children) > limit:
            violations[str(package.relative_to(PACKAGE_ROOT))] = children

    assert violations == {}


def test_every_directory_stays_within_direct_business_module_limit() -> None:
    violations: dict[str, list[str]] = {}
    for package in _package_directories():
        modules = sorted(
            path.name
            for path in package.glob("*.py")
            if path.name != "__init__.py"
        )
        relative = str(package.relative_to(PACKAGE_ROOT))
        limit = DIRECT_MODULE_LIMITS.get(relative, MAX_DIRECT_MODULES)
        if len(modules) > limit:
            violations[str(package.relative_to(PACKAGE_ROOT))] = modules

    assert violations == {}
