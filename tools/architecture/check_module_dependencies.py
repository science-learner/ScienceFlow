#!/usr/bin/env python3
"""Fail when modular ScienceFlow domain boundaries are crossed."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scienceflow.foundation.architecture import find_dependency_violations  # noqa: E402


def main() -> int:
    package_root = REPO_ROOT / "scienceflow"
    violations = find_dependency_violations(package_root)
    if not violations:
        print("ScienceFlow module dependency rules: OK")
        return 0
    for violation in violations:
        print(violation.format(REPO_ROOT))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
