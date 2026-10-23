from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from scienceflow.agent.core.ports.callback_ports import (
    AgentCallbackPorts,
    callback_ports_for,
    install_callback_ports,
    resolve_agent_callback,
)


CALLBACK_ORDER_FIXTURE = (
    "candidate_archive",
    "stage_capture",
    "metric_interpretation",
    "text_only_decision",
    "context_limit_estra",
    "context_compact_observer",
)


def test_typed_callback_ports_have_frozen_inventory_and_override_legacy_reads() -> None:
    calls: list[str] = []

    def typed(**_kwargs) -> None:
        calls.append("typed")

    def legacy(**_kwargs) -> None:
        calls.append("legacy")

    agent = SimpleNamespace(
        callback_ports=AgentCallbackPorts(candidate_archive=typed),
        _lnr_candidate_archive_callback=legacy,
    )

    callback = resolve_agent_callback(agent, "candidate_archive")
    assert callback is typed
    callback()
    assert calls == ["typed"]
    assert tuple(AgentCallbackPorts.__dataclass_fields__) == CALLBACK_ORDER_FIXTURE


def test_callback_port_install_and_scoped_replacement_preserve_other_capabilities() -> None:
    def original_text(**_) -> str:
        return "original"

    def replacement_text(**_) -> str:
        return "replacement"

    def stage(**_) -> str:
        return "stage"

    agent = SimpleNamespace()
    original = AgentCallbackPorts(
        stage_capture=stage,
        text_only_decision=original_text,
    )
    install_callback_ports(agent, original)

    temporary = callback_ports_for(agent).with_overrides(
        text_only_decision=replacement_text,
    )
    install_callback_ports(agent, temporary)

    assert resolve_agent_callback(agent, "text_only_decision") is replacement_text
    assert resolve_agent_callback(agent, "stage_capture") is stage
    install_callback_ports(agent, original)
    assert resolve_agent_callback(agent, "text_only_decision") is original_text


def test_production_composition_does_not_set_dynamic_callback_attributes() -> None:
    repository = Path(__file__).resolve().parents[1]
    targets = (
        repository / "scienceflow/research/solver/lnr/orchestration/coordinator",
        repository / "scienceflow/agent",
        repository / "scienceflow/agent/tooling/execution",
    )
    forbidden = {
        "_lnr_candidate_archive_callback",
        "_lnr_stage_capture_callback",
        "_lnr_metric_output_interpretation_callback",
        "_lnr_text_only_callback",
        "_context_limit_estra_callback",
        "_context_compact_event_callback",
    }
    violations: list[str] = []
    paths = [
        *targets[0].rglob("*.py"),
        *targets[1].rglob("*.py"),
        *targets[2].rglob("*.py"),
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "setattr" or len(node.args) < 2:
                continue
            attribute = node.args[1]
            if isinstance(attribute, ast.Constant) and attribute.value in forbidden:
                violations.append(f"{path.relative_to(repository)}:{node.lineno}")

    assert violations == []
