from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

from click.testing import CliRunner

from scienceflow.interfaces.cli import main
from scienceflow.interfaces.ui.logo import LOGO_PIXELS


def test_scienceflow_logo_marks_empty_cells_as_transparent() -> None:
    assert LOGO_PIXELS[0][0] == "transparent"
    assert any(pixel != "transparent" for row in LOGO_PIXELS for pixel in row)


def test_scienceflow_root_version_reports_active_source() -> None:
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    source_version = tomllib.loads(project.read_text(encoding="utf-8"))["project"]["version"]
    assert result.output.strip() == f"scienceflow, version {source_version}"


def test_scienceflow_root_defers_optional_command_runtimes() -> None:
    script = """
import sys
from scienceflow.interfaces.cli import main
assert not main.commands['tui']._resolved
assert not main.commands['agent']._resolved
assert 'openai' not in sys.modules
assert 'scienceflow.interfaces.ui.long_research' not in sys.modules
assert 'scienceflow.interfaces.ui.web_monitor.server' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_scienceflow_root_help_prioritizes_primary_commands() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    visible = {
        line.strip().split()[0]
        for line in result.output.split("Commands:", 1)[1].splitlines()
        if line.strip()
    }
    for command in ("tui", "run", "status", "stop", "resume", "config", "web"):
        assert command in visible
    for command in (
        "prep",
        "repl",
        "parallel",
        "monitor",
        "monitor-trace",
        "resource-summary",
        "replay-prepare",
        "agent",
    ):
        assert command not in visible
        assert command in main.commands
        direct = CliRunner().invoke(main, [command, "--help"])
        assert direct.exit_code == 0, direct.output


def test_scienceflow_agent_tools_use_inquirycraft_without_llm_init() -> None:
    result = CliRunner().invoke(main, ["agent", "tools", "list"])

    assert result.exit_code == 0
    assert result.output.splitlines() == ["read_text", "write_text", "shell"]


def test_scienceflow_agent_group_excludes_optional_inquirycraft_tui() -> None:
    result = CliRunner().invoke(main, ["agent", "--help"])

    assert result.exit_code == 0
    assert "tui" not in result.output.split()
    assert {"run", "repl", "tools", "events"}.issubset(result.output.split())
    assert main.commands["agent"].hidden
    root_help = CliRunner().invoke(main, ["--help"])
    assert "agent" not in root_help.output.split("Commands:", 1)[1]


def test_scienceflow_root_tui_uses_existing_inquirycraft_options() -> None:
    result = CliRunner().invoke(main, ["tui", "--help"])

    assert result.exit_code == 0
    if "--workspace" not in result.output:
        blocked = CliRunner().invoke(main, ["tui"])
        assert blocked.exit_code != 0
        assert "requires InquiryCraft with tui_interceptor_factory" in blocked.output
        return
    assert "--workspace" in result.output
    assert "--model" in result.output
    assert "One chat model alias" in result.output
    assert "full-screen ScienceFlow terminal interface" in result.output
    timeout = next(
        param for param in main.commands["tui"].params if param.name == "max_tokens"
    )
    assert "SCIENCEFLOW_TUI_MAX_TOKENS" in timeout.envvar


def test_scienceflow_tui_registers_one_resume_option() -> None:
    from scienceflow.interfaces.cli.commands.run.tui import build_tui_command

    command = build_tui_command()
    assert [param.name for param in command.params].count("resume") == 1
    assert [param.opts for param in command.params if param.name == "resume"] == [["--resume"]]


def test_scienceflow_root_tui_requires_complete_090_host_contract() -> None:
    from inspect import signature

    from inquirycraft.cli import create_cli

    required = {
        "include_tui",
        "model_config_default",
        "model_default_key",
        "tui_branding",
        "tui_interceptor_factory",
    }
    assert required.issubset(signature(create_cli).parameters)


def test_optional_placeholder_keeps_older_host_branding_usable(monkeypatch) -> None:
    from inspect import signature

    import inquirycraft.tui

    from scienceflow.interfaces.cli.commands.run.tui import build_tui_command

    original = inquirycraft.tui.TuiBranding
    captured = {}

    def legacy_branding(**options):
        captured.update(options)
        assert 'prompt_placeholder' not in options
        assert 'notice_background' not in options
        assert 'fold_system_notices' not in options
        return original(**options)

    legacy_branding.__signature__ = signature(original).replace(
        parameters=[p for name, p in signature(original).parameters.items()
                    if name not in {'prompt_placeholder', 'notice_background', 'fold_system_notices'}]
    )
    monkeypatch.setattr(inquirycraft.tui, 'TuiBranding', legacy_branding)
    assert build_tui_command().name == 'tui'
    assert captured['name'] == 'ScienceFlow'


def test_scienceflow_root_tui_binds_long_research_to_selected_workspace(
    monkeypatch, tmp_path: Path
) -> None:
    from inspect import signature

    import inquirycraft.tui
    from inquirycraft.cli import create_cli

    from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory

    if "tui_interceptor_factory" not in signature(create_cli).parameters:
        blocked = CliRunner().invoke(main, ["tui"])
        assert blocked.exit_code != 0
        assert "tui_interceptor_factory" in blocked.output
        return

    captured: dict[str, object] = {}

    class FakeRuntime:
        async def aclose(self) -> None:
            captured["runtime_closed"] = True

    def fake_create_runtime(self, options, **kwargs):
        captured["runtime_options"] = options
        return FakeRuntime()

    class FakeTui:
        def __init__(self, runtime, *, model, workspace, interceptor, branding) -> None:
            captured.update(
                runtime=runtime,
                model=model,
                workspace=workspace,
                interceptor=interceptor,
                branding=branding,
            )

        async def run_async(self) -> None:
            captured["started"] = True

        async def shutdown_runtime(self) -> None:
            await captured["runtime"].aclose()  # type: ignore[union-attr]

    monkeypatch.setattr(PoolRuntimeFactory, "create_runtime", fake_create_runtime)
    monkeypatch.setitem(vars(inquirycraft.tui), "InquiryCraftTui", FakeTui)
    config = tmp_path / "models.json"
    config.write_text(
        '{"models":{'
        '"fake":{"model":"fake","endpoints":[{"url":"","key":"test-key"}]},'
        '"backup":{"model":"backup","endpoints":[{"url":"","key":"backup-key"}]}'
        '},"defaults":{"code_models":["fake","backup"]}}'
    )
    result = CliRunner().invoke(
        main,
        [
            "tui",
            "--model-config",
            str(config),
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["branding"].name == "ScienceFlow"
    assert captured["branding"].notice_label == "SCIENCEFLOW"
    assert captured["branding"].prompt_placeholder == "Type a message · /long-research · /help"
    assert captured["branding"].notice_background == "#3B4048"
    assert captured["branding"].fold_system_notices
    assert captured["branding"].welcome_text.splitlines() == [
        "Long Research · parallel workers explore, evaluate, and preserve the best result.",
        "Start with /long-research · monitor with /tasks · resume a task with /resume N",
    ]
    assert captured["branding"].welcome_emphasis == (
        "Long Research",
        "parallel workers",
        "best result",
    )
    assert captured["started"] is True
    assert captured["workspace"] == tmp_path
    assert captured["interceptor"].workspace == tmp_path  # type: ignore[union-attr]
    assert captured["runtime_options"].model == "fake"  # type: ignore[union-attr]
    assert captured["model"] == "fake"
    assert captured["interceptor"]._selected_model_aliases == ("fake",)  # type: ignore[union-attr]
    assert captured["runtime_closed"] is True


def test_scienceflow_tui_rejects_multiple_chat_aliases(tmp_path: Path) -> None:
    config = tmp_path / "models.json"
    config.write_text(
        '{"models":{'
        '"a":{"model":"a","endpoints":[{"url":"","key":"a-key"}]},'
        '"b":{"model":"b","endpoints":[{"url":"","key":"b-key"}]}'
        '},"defaults":{"code_models":["a","b"]}}'
    )

    result = CliRunner().invoke(
        main,
        [
            "tui",
            "--models",
            "a,b",
            "--model-config",
            str(config),
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "TUI chat accepts exactly one model alias" in result.output


def test_scienceflow_tui_default_identity_keeps_explicit_override():
    from scienceflow.interfaces.cli.commands.run.tui import build_tui_command

    command = build_tui_command()
    option = next(param for param in command.params if param.name == 'system_prompt')
    assert 'You are ScienceFlow' in option.default
    with command.make_context('tui', ['--system-prompt', 'Custom research assistant']) as context:
        assert context.params['system_prompt'] == 'Custom research assistant'
