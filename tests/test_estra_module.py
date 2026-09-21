# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Contracts for EStra as a pure Runtime command producer."""

from __future__ import annotations

import ast
from pathlib import Path

from scienceflow.foundation.contracts import EstraContext, EstraDecision
from scienceflow.research.control.estra import EstraService


def test_estra_normalizes_two_axis_redirect_without_runtime_mutation() -> None:
    service = EstraService()
    context = EstraContext(
        latest_stage="S03",
        switch_candidates=("S01", "S02"),
        trigger_source="context_limit",
    )
    decision = service.decide_from_text(
        '{"startpoint":"current_workspace","intent":"redirect",'
        '"bottleneck":"validation plateau","decision_reason":"change route"}',
        context=context,
    )

    assert decision == EstraDecision(
        action="keep_but_redirect",
        startpoint="current_workspace",
        intent="redirect",
        target_stage="S03",
        compact=True,
        reason="change route | bottleneck: validation plateau",
        diagnostics={
            "startpoint": "current_workspace",
            "intent": "redirect",
            "bottleneck": "validation plateau",
            "decision_reason": "change route",
        },
    )
    assert context.latest_stage == "S03"


def test_estra_rejects_restore_target_outside_runtime_candidates() -> None:
    service = EstraService()
    context = EstraContext(latest_stage="S03", switch_candidates=("S01",))
    assert (
        service.decide_from_text(
            '{"action":"switch_stage","target_stage":"S02"}',
            context=context,
        )
        is None
    )
    accepted = service.decide_from_text("restore to S01", context=context)
    assert accepted is not None
    assert accepted.action == "switch_stage"
    assert accepted.target_stage == "S01"


def test_estra_module_cannot_import_workspace_memory_or_solver_host() -> None:
    root = Path(__file__).parents[1] / "scienceflow" / "research" / "control" / "estra"
    forbidden = (
        "scienceflow.research.state.workspace",
        "scienceflow.research.state.knowledge.memory",
        "scienceflow.research.solver.lnr.orchestration.solver",
        "scienceflow.research.solver",
    )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not str(node.module or "").startswith(forbidden), path
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(forbidden) for alias in node.names
                ), path
