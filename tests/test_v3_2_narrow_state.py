# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Architecture gates for the V3.2 narrow-state migration."""

from __future__ import annotations

import ast
from pathlib import Path

from scienceflow.research.control.agent_session_state import RunSessionState


PACKAGE_ROOT = Path(__file__).parents[1] / "scienceflow"
RUN_LOOP_ROOT = PACKAGE_ROOT / "agent/runtime/run_loop_components"
V3_2_WILDCARD_IMPORT_BASELINE = 174


def _wildcard_imports(root: Path) -> list[tuple[Path, int, str | None]]:
    imports: list[tuple[Path, int, str | None]] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports.extend(
            (path, node.lineno, node.module)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
        )
    return imports


def test_production_wildcard_import_debt_cannot_grow() -> None:
    assert len(_wildcard_imports(PACKAGE_ROOT)) <= V3_2_WILDCARD_IMPORT_BASELINE


def test_run_loop_has_no_wildcard_import_or_shared_dependency_bundle() -> None:
    paths = list(RUN_LOOP_ROOT.glob("*.py"))
    violations = [
        (str(path.relative_to(PACKAGE_ROOT)), lineno, module)
        for path in paths
        for imported_path, lineno, module in _wildcard_imports(path.parent)
        if imported_path == path
    ]
    assert violations == []
    assert not (RUN_LOOP_ROOT / "shared.py").exists()


def test_run_session_state_owns_only_control_flow_state() -> None:
    state = RunSessionState.start(4)
    assert state.initial_max_steps == 4
    assert state.effective_max_steps == 4
    assert state.round_index == 0
    assert not state.aborted

    state.advance()
    state.update_effective_max(7)
    state.abort("run-policy")

    assert state.round_index == 1
    assert state.effective_max_steps == 7
    assert state.aborted
    assert state.abort_reason == "run-policy"
    assert set(RunSessionState.__dataclass_fields__) == {
        "initial_max_steps",
        "effective_max_steps",
        "round_index",
        "abort_reason",
    }
