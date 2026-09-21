# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""V3.1 hard-removal contracts for the expired legacy import window."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

from scienceflow.foundation.config.schema.settings import LnrConfig


PACKAGE_ROOT = Path(__file__).parents[1] / "scienceflow"
REMOVED_MODULES = (
    "scienceflow.research.solver.lnr.legacy_solver",
    "scienceflow.research.solver.lnr.resources.runtime.observer.legacy_controller",
)


def test_legacy_configuration_surface_is_removed(monkeypatch) -> None:
    assert "legacy_backend_mode" not in LnrConfig.__dataclass_fields__
    monkeypatch.setenv("SCIENCEFLOW_LEGACY_BACKEND_MODE", "compatibility")
    assert "legacy_backend_mode" not in LnrConfig.__dataclass_fields__


def test_removed_legacy_imports_are_not_discoverable() -> None:
    for module_name in REMOVED_MODULES:
        assert importlib.util.find_spec(module_name) is None
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            assert exc.name == module_name
        else:  # pragma: no cover - explicit release contract
            raise AssertionError(f"removed module remained importable: {module_name}")


def test_replacement_imports_are_public_and_canonical() -> None:
    solver = importlib.import_module("scienceflow.research.solver.lnr.orchestration.solver")
    observer = importlib.import_module(
        "scienceflow.research.solver.lnr.resources.runtime.observer.controller"
    )
    assert solver.LnrSolver.__module__ == "scienceflow.research.solver.lnr.orchestration.solver"
    assert (
        observer.LHRResourceObserver.__module__
        == "scienceflow.research.solver.lnr.resources.runtime.observer.controller"
    )


def test_production_contains_no_legacy_compatibility_reference() -> None:
    forbidden_text = (
        "legacy_solver",
        "legacy_controller",
        "LegacyBackendMode",
        "legacy_backend_mode",
        "SCIENCEFLOW_LEGACY_BACKEND_MODE",
    )
    for path in PACKAGE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert all(token not in source for token in forbidden_text), path
        ast.parse(source, filename=str(path))


def test_unreferenced_private_methods_remain_absent() -> None:
    removed = {
        "_stage_packet_line",
        "_tool_targets_ledger",
        "_estra_action_from_axes",
        "_normalize_estra_startpoint",
        "_normalize_estra_intent",
        "_estra_decision_reason",
        "_compact_estra_reason",
        "_parse_estra_json",
        "_worker_workspace_dir",
        "_near_deliverable_for_job",
    }
    roots = (
        PACKAGE_ROOT / "research/solver/lnr/orchestration/coordinator",
        PACKAGE_ROOT / "research/solver/lnr/resources/runtime/observer",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in root.glob("*.py")
    )
    assert all(f"def {name}(" not in source for name in removed)
