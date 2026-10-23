"""Structural gates for the V5.2 ScienceAgent ownership split."""

from __future__ import annotations

import inspect
import ast
from pathlib import Path

from scienceflow.agent.factory import construction
from scienceflow.agent.factory.construction.builder import (
    AgentConstructionOptions,
    ScienceAgentBuilder,
)
from scienceflow.agent.core.ports.host_ports import AgentHostPorts
from scienceflow.agent.session import (
    AgentSessionSpec,
    ScienceFlowAgentHooks,
    ScienceFlowAgentSession,
    ScienceFlowAgentSessionFactory,
)


def test_construction_uses_one_typed_builder_without_locals_forwarding() -> None:
    source = inspect.getsource(construction.__init__)

    assert "locals()" not in source
    assert "ScienceAgentBuilder(" in source
    assert len(AgentConstructionOptions.__dataclass_fields__) > 100
    assert ScienceAgentBuilder.__module__.endswith(".construction.builder")
    construction_root = (
        Path(__file__).parents[1] / "scienceflow" / "agent" / "factory" / "construction"
    )
    assert len((construction_root / "builder.py").read_text().splitlines()) < 100
    assert len((construction_root / "assembly.py").read_text().splitlines()) < 800


def test_session_public_api_is_backed_by_focused_owner_modules() -> None:
    assert AgentSessionSpec.__module__.endswith(".session.coordination.contracts")
    assert ScienceFlowAgentHooks.__module__.endswith(".session.execution.tool_result")
    assert ScienceFlowAgentSession.__module__.endswith(".session.execution.runtime")
    assert ScienceFlowAgentSessionFactory.__module__.endswith(
        ".session.coordination.host"
    )


def test_legacy_flat_agent_modules_are_retired() -> None:
    agent_root = Path(__file__).parents[1] / "scienceflow" / "agent"

    assert not (agent_root / "session.py").exists()
    assert not (agent_root / "factory" / "construction_runtime.py").exists()


def test_science_agent_session_capabilities_are_typed_not_descriptor_bound() -> None:
    agent_path = (
        Path(__file__).parents[1]
        / "scienceflow"
        / "agent"
        / "core"
        / "runtime"
        / "agent.py"
    )
    tree = ast.parse(agent_path.read_text(encoding="utf-8"))
    agent = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ScienceAgent"
    )
    bindings = {
        target.id
        for node in agent.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id != "model_config"
    }

    assert len(AgentHostPorts.__dataclass_fields__) == 19
    assert len(bindings) == 68
    assert "_maybe_normalize_tool_input_paths" not in bindings
    assert "_execute_tool_maybe_edit_guard" not in bindings
    assert "_lnr_on_round_complete" not in bindings
    assert "setattr(" not in agent_path.read_text(encoding="utf-8")
