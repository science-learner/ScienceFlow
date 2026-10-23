"""Shell is available by default; cancellation and process cleanup remain intact."""

import asyncio
import json

import pytest
from click.testing import CliRunner
from inquirycraft.cli import create_cli
from inquirycraft.events import NullEventSink
from inquirycraft.runtime import CancellationToken
from inquirycraft.tools import ShellTool, default_tools
from inquirycraft.tools.base import ToolContext

from scienceflow.interfaces.cli import main
from scienceflow.interfaces.cli.commands.run.tui import build_tui_command
from scienceflow.interfaces.ui.llm_cli import (
    PoolRuntimeFactory,
    TuiPoolRuntimeFactory,
    TuiShellTool,
)


def _schema_names(tools):
    return [schema["function"]["name"] for schema in tools.schemas()]


def test_tui_shell_card_shows_elapsed_time_without_a_default_deadline(monkeypatch):
    from inquirycraft.tui import tool_card

    clock = [100.0]
    monkeypatch.setattr(tool_card, "monotonic", lambda: clock[0])
    shell = TuiPoolRuntimeFactory()._tools().get("shell")
    entry = tool_card.ToolEntry(
        "shell",
        arguments={"command": "pip install package"},
        default_timeout_sec=getattr(shell, "default_timeout_sec", None),
    )
    entry.update_tool(state="running")
    clock[0] += 65
    entry.refresh_running_status()

    title = str(entry.content.renderables[0])
    assert "01:05" in title
    assert "no output 01:05" in title
    assert "01:05/" not in title


@pytest.mark.asyncio
async def test_tui_text_tools_remain_workspace_contained(tmp_path):
    tools = TuiPoolRuntimeFactory()._tools()
    context = ToolContext(
        "session", "run", "agent", tmp_path, NullEventSink(), CancellationToken()
    )
    await tools.get("write_text").execute(
        {"path": "inside.txt", "content": "safe"}, context
    )
    assert (
        await tools.get("read_text").execute({"path": "inside.txt"}, context)
    ).output == "safe"
    with pytest.raises(ValueError, match="outside|workspace"):
        await tools.get("write_text").execute(
            {"path": "../outside.txt", "content": "no"}, context
        )


def test_tui_cli_provides_shell_without_permission_flags(tmp_path, monkeypatch):
    import inquirycraft.tui

    seen = []

    class FakeRuntime:
        async def aclose(self):
            pass

    class FakeTui:
        def __init__(self, runtime, **kwargs):
            pass

        async def run_async(self):
            pass

        async def shutdown_runtime(self):
            pass

    def create_runtime(self, options, **kwargs):
        seen.append(_schema_names(self._tools()))
        return FakeRuntime()

    monkeypatch.setattr(TuiPoolRuntimeFactory, "create_runtime", create_runtime)
    monkeypatch.setattr(inquirycraft.tui, "InquiryCraftTui", FakeTui)
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "models": {
                    "fake": {"model": "fake", "endpoints": [{"url": "", "key": "test"}]}
                },
                "defaults": {"code_models": ["fake"]},
            }
        )
    )
    args = ["--workspace", str(tmp_path), "--model-config", str(config)]
    result = CliRunner().invoke(build_tui_command(), args)
    assert result.exit_code == 0, result.output
    assert seen == [["read_text", "write_text", "workspace_info", "shell"]]


@pytest.mark.asyncio
async def test_tui_shell_has_no_default_timeout_without_changing_global_tool(
    monkeypatch,
):
    from inquirycraft.tools import shell_execution
    from inquirycraft.tools.base import ToolResult

    calls = []

    async def run_shell(arguments, context):
        calls.append(dict(arguments))
        return ToolResult(output="ok")

    monkeypatch.setattr(shell_execution, "run_shell", run_shell)
    await TuiShellTool().execute({"command": "true"}, None)
    await TuiShellTool().execute({"command": "true", "timeout": 33}, None)
    await ShellTool().execute({"command": "true"}, None)
    assert calls == [
        {"command": "true"},
        {"command": "true", "timeout": 33},
        {"command": "true"},
    ]
    assert "No default timeout" in TuiShellTool.description
    assert TuiShellTool.input_schema["properties"]["timeout"]["maximum"] == 3600
    assert ShellTool.input_schema["properties"]["timeout"]["maximum"] == 3600


@pytest.mark.asyncio
async def test_tui_shell_keeps_existing_explicit_timeout_bounds():
    result = await TuiShellTool().execute({"command": "true", "timeout": 3601}, None)
    assert "between 0.1 and 3600 seconds" in result.error


def test_allow_shell_is_not_an_agent_or_inquirycraft_option():
    runner = CliRunner()
    tui_help = runner.invoke(main, ["tui", "--help"])
    assert tui_help.exit_code == 0, tui_help.output
    assert "--allow-shell" not in tui_help.output
    for command in (["agent", "run", "--help"], ["agent", "repl", "--help"]):
        assert "--allow-shell" not in runner.invoke(main, command).output
    assert "--allow-shell" not in runner.invoke(create_cli(), ["tui", "--help"]).output
    assert runner.invoke(main, ["agent", "tools", "list"]).output.splitlines() == [
        "read_text",
        "write_text",
        "shell",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("external_cancel", [False, True])
async def test_cancellation_stops_active_tui_shell_work(tmp_path, external_cancel):
    context = ToolContext(
        "session",
        "run",
        "agent",
        tmp_path,
        NullEventSink(),
        CancellationToken(),
    )
    execution = asyncio.create_task(
        TuiShellTool().execute(
            {
                "command": (
                    "printf started > shell-started; "
                    "sleep 30; "
                    "printf finished > shell-finished"
                )
            },
            context,
        )
    )
    for _ in range(100):
        if (tmp_path / "shell-started").is_file():
            break
        await asyncio.sleep(0.02)
    assert (tmp_path / "shell-started").is_file()

    if external_cancel:
        execution.cancel()
    else:
        context.cancellation.cancel()
    result = await asyncio.wait_for(execution, timeout=5)

    assert result.error == "command cancelled"
    assert result.metadata["shell_execution"]["status"] == "cancelled"
    assert not (tmp_path / "shell-finished").exists()


def test_shell_available_by_default_without_changing_agent_tools():
    assert _schema_names(TuiPoolRuntimeFactory()._tools()) == [
        "read_text",
        "write_text",
        "workspace_info",
        "shell",
    ]
    assert _schema_names(default_tools()) == ["read_text", "write_text", "shell"]
    assert _schema_names(PoolRuntimeFactory()._tools()) == _schema_names(
        default_tools()
    )


def test_tui_interceptor_has_no_permission_command(tmp_path):
    from scienceflow.interfaces.ui.long_research import LongResearchInteraction

    interaction = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    assert "/permission" not in interaction.tui_common_commands
    assert all(spec.name != "/permission" for spec in interaction.tui_commands)
    assert str(tmp_path) in interaction.chat_context()
    assert "Shell is available" in interaction.chat_context()
