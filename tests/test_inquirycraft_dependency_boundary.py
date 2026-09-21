from __future__ import annotations

import ast
import inspect
import tomllib
from pathlib import Path

from inquirycraft.llm import OnlineLLM
from inquirycraft.llm import OnlineLLM as OwnedOnlineLLM
from inquirycraft.memory import Memory, Message
from inquirycraft.tools import (
    BaseTool,
    Function,
    ToolCall,
    ToolCollection,
    ToolResult,
    coerce_tool_result,
)
from inquirycraft.tui import TuiHostPort


def test_scienceflow_exposes_only_mcp_as_optional_inquirycraft_integration() -> None:
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    config = tomllib.loads(project_file.read_text(encoding="utf-8"))
    project = config["project"]
    inquirycraft_extras = {
        name: dependencies
        for name, dependencies in project["optional-dependencies"].items()
        if name.startswith("inquirycraft-")
    }
    assert "inquirycraft[openai,tui]==0.9.0" in project["dependencies"]
    assert inquirycraft_extras == {
        "inquirycraft-mcp": ["inquirycraft[mcp]==0.9.0"]
    }
    assert "inquirycraft" not in config["tool"]["uv"]["sources"]


def test_contract_ci_installs_project_metadata_without_dependency_override() -> None:
    repository_root = Path(__file__).parents[1]
    workflow = (
        repository_root / ".github" / "workflows" / "scienceflow-contract-ci.yml"
    ).read_text(encoding="utf-8")

    assert "python -m pip install -e ." in workflow
    assert "requirements.txt" not in workflow
    assert "inquirycraft[" not in workflow


def test_inquirycraft_tui_model_registry_host_contract() -> None:
    selector = inspect.signature(TuiHostPort.open_chat_model_selector)
    switcher = inspect.signature(TuiHostPort.switch_chat_models)

    assert selector.parameters["config_hint"].default == ""
    assert {"model_config", "aliases", "current", "selection", "on_applied"}.issubset(
        selector.parameters
    )
    assert {"model_config", "aliases", "selection"}.issubset(switcher.parameters)


def test_light_wheel_keeps_registered_task_packages_discoverable() -> None:
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    setuptools = tomllib.loads(project_file.read_text(encoding="utf-8"))["tool"][
        "setuptools"
    ]
    package_find = setuptools["packages"]["find"]

    assert "tasks*" in package_find["include"]
    assert package_find["namespaces"] is True
    assert set(setuptools["package-data"]["tasks"]) == {
        "**/*.json",
        "**/*.md",
        "**/*.py",
        "**/*.yaml",
    }


def test_canonical_models_preserve_scienceflow_agent_contract() -> None:
    assert Message.__module__ == "inquirycraft.memory.message"
    assert Memory.__module__ == "inquirycraft.memory.history"
    assert OnlineLLM is OwnedOnlineLLM
    assert OnlineLLM.__module__ == "inquirycraft.llm.online"
    assert BaseTool.__module__ == "inquirycraft.tools.base"
    assert ToolCall.__module__ == "inquirycraft.memory.message"
    assert Function.__module__ == "inquirycraft.memory.message"
    call = ToolCall(
        id="call-1", function=Function(name="read", arguments='{"path":"x"}')
    )
    assert call.model_dump() == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"path":"x"}'},
    }
    assert ToolResult.__module__ == "inquirycraft.tools.base"
    assert list(ToolResult.model_fields) == ["output", "error", "system"]
    assert ToolCollection.__module__ == "inquirycraft.tools.collection"


def test_tool_result_migration_preserves_rendering_and_old_object_fields() -> None:
    cases = (
        {"output": "ok"},
        {"error": "failed"},
        {"output": "details", "error": "failed", "system": "raw"},
    )
    for values in cases:
        migrated = ToolResult(**values)
        assert migrated.model_dump() == {
            "output": values.get("output"),
            "error": values.get("error"),
            "system": values.get("system"),
        }
        assert bool(migrated) is bool(
            values.get("output") or values.get("error") or values.get("system")
        )
    old_shape = type(
        "OldToolResult",
        (),
        {"output": "visible", "error": "failed", "system": "lossless"},
    )()
    normalized = coerce_tool_result(old_shape)
    assert type(normalized) is ToolResult
    assert normalized.model_dump() == {
        "output": "visible",
        "error": "failed",
        "system": "lossless",
    }


def test_scienceflow_source_only_imports_inquirycraft_public_namespace() -> None:
    source_root = Path(__file__).parents[1] / "scienceflow"
    offenders: list[tuple[str, str]] = []
    internal_modules = {
        "inquirycraft.llm.base",
        "inquirycraft.runtime.resume",
        "inquirycraft.runtime.subprocess",
        "inquirycraft.runtime.subprocess_sync",
        "inquirycraft.tools.shell_guards",
        "inquirycraft.tools.shell_reducer",
        "inquirycraft.tools.shell_traceback",
    }
    compatibility_adapters = {
        "runtime/core/process/utils.py",
        "runtime/safety/tooling/workspace/shell_output.py",
    }
    for path in source_root.rglob("*.py"):
        relative_path = str(path.relative_to(source_root))
        is_compatibility_adapter = relative_path in compatibility_adapters
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
                if node.module.startswith("inquirycraft"):
                    for alias in node.names:
                        if alias.name.startswith("_") and not is_compatibility_adapter:
                            offenders.append(
                                (
                                    relative_path,
                                    f"{node.module}.{alias.name}",
                                )
                            )
            for name in names:
                if name == "deepcraft_core" or name.startswith("deepcraft_core."):
                    offenders.append((relative_path, name))
                if name == "deepcraft_agent" or name.startswith("deepcraft_agent."):
                    offenders.append((relative_path, name))
                if name in internal_modules and not is_compatibility_adapter:
                    offenders.append((relative_path, name))
    assert offenders == []


def test_retired_scienceflow_tool_runtime_directories_stay_absent() -> None:
    source_root = Path(__file__).parents[1] / "scienceflow" / "core"
    retired = [
        source_root / "agent" / "tool_exec",
        source_root / "agent" / "tools",
        source_root / "executor",
        source_root / "tools",
    ]
    assert [str(path.relative_to(source_root)) for path in retired if path.exists()] == []
