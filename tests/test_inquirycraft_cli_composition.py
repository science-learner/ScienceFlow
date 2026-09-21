from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from scienceflow.interfaces.cli import main
from scienceflow.interfaces.ui.logo import LOGO_PIXELS


def test_scienceflow_logo_marks_empty_cells_as_transparent() -> None:
    assert LOGO_PIXELS[0][0] == "transparent"
    assert any(pixel != "transparent" for row in LOGO_PIXELS for pixel in row)


def test_scienceflow_keeps_existing_commands_and_adds_agent_group() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    for command in (
        "run",
        "repl",
        "parallel",
        "tui",
        "monitor",
        "monitor-trace",
        "resource-summary",
        "replay-prepare",
        "agent",
    ):
        assert command in result.output


def test_scienceflow_agent_tools_use_inquirycraft_without_llm_init() -> None:
    result = CliRunner().invoke(main, ["agent", "tools", "list"])

    assert result.exit_code == 0
    assert result.output.splitlines() == ["read_text", "write_text", "shell"]


def test_scienceflow_agent_group_excludes_optional_inquirycraft_tui() -> None:
    result = CliRunner().invoke(main, ["agent", "--help"])

    assert result.exit_code == 0
    assert "tui" not in result.output.split()
    assert {"run", "repl", "tools", "events"}.issubset(result.output.split())


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
    assert captured["branding"].welcome_text.splitlines() == [
        "Long Research · parallel workers explore, evaluate, and preserve the best result.",
        "Start with /long-research · monitor with /tasks · continue with /resume",
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
