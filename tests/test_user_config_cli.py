from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from scienceflow.interfaces.cli import main
from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in list(os.environ):
        if name.startswith(("SCIENCEFLOW_", "CODE_", "FEEDBACK_", "XDG_")):
            os.environ.pop(name)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "scienceflow.interfaces.cli._SCIENCEFLOW_REPO_ROOT", tmp_path / "installed"
    )


def test_config_init_creates_private_file_and_never_overwrites():
    runner = CliRunner()
    result = runner.invoke(main, ["config", "init"])
    assert result.exit_code == 0, result.output
    path = Path.home() / ".config/scienceflow/models.json"
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_text('{"private": "value"}\n')
    result = runner.invoke(main, ["config", "init"])
    assert result.exit_code == 0
    assert path.read_text() == '{"private": "value"}\n'
    assert "value" not in result.output
    result = runner.invoke(main, ["config", "path"])
    assert result.output.strip() == str(path)


def test_config_init_supports_explicit_path(tmp_path):
    path = tmp_path / "custom/scienceflow/models.json"
    result = CliRunner().invoke(
        main, ["config", "init"], env={"XDG_CONFIG_HOME": str(tmp_path / "custom")}
    )
    assert result.exit_code == 0, result.output
    assert path.is_file()


def test_user_file_configures_agent_without_cli_options(monkeypatch, tmp_path):
    path = Path.home() / ".config/scienceflow/models.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "models": {
            "a": {"model": "m1", "endpoints": [{"url": "https://a/v1", "key": "a"}]},
            "b": {"model": "m2", "endpoints": [{"url": "https://b/v1", "key": "b"}]},
        },
        "defaults": {"code_models": ["a", "b"], "feedback_models": ["a"]},
    }))
    captured = {}

    class Runtime:
        async def run(self, task):
            return "done"

        async def aclose(self):
            pass

    def create(self, options, **kwargs):
        captured.update(options=options, **kwargs)
        return Runtime()

    monkeypatch.setattr(PoolRuntimeFactory, "create_runtime", create)
    runner = CliRunner()
    result = runner.invoke(main, ["agent", "run", "hello"])
    assert result.exit_code == 0, result.output
    assert captured["api_key"] == "a,b"
    assert captured["options"].model == "m1,m2"
    assert captured["options"].workspace.resolve() == Path.cwd()
    assert captured["options"].max_turns == 20
    assert captured["options"].timeout == 300
    monkeypatch.setenv("SCIENCEFLOW_AGENT_RUN_MODELS", "legacy-model")
    monkeypatch.setenv("SCIENCEFLOW_MODELS", "legacy-model")
    result = runner.invoke(main, ["agent", "run", "hello"])
    assert result.exit_code == 0, result.output
    assert captured["options"].model == "m1,m2"
    monkeypatch.setenv("SCIENCEFLOW_AGENT_RUN_TIMEOUT", "60")
    result = runner.invoke(main, ["agent", "run", "hello", "--timeout", "90"])
    assert result.exit_code == 0, result.output
    assert captured["options"].timeout == 90


def test_parallel_required_manifest_can_come_from_environment(monkeypatch, tmp_path):
    captured = {}

    async def run_manifest(manifest, **kwargs):
        captured.update(manifest=manifest, **kwargs)
        return type("Summary", (), {"results": (), "text": "done"})()

    monkeypatch.setattr(
        "scienceflow.runtime.parallel.service.run_manifest", run_manifest
    )
    monkeypatch.setenv("SCIENCEFLOW_PARALLEL_MANIFEST", str(tmp_path / "task.yaml"))
    monkeypatch.setenv("SCIENCEFLOW_PARALLEL_MAX_CONCURRENT", "3")
    result = CliRunner().invoke(main, ["parallel"])
    assert result.exit_code == 0, result.output
    assert captured["manifest"] == str(tmp_path / "task.yaml")
    assert captured["max_concurrent"] == 3


def test_missing_explicit_file_reports_clear_error_without_running(tmp_path):
    result = CliRunner().invoke(
        main, ["--env-file", str(tmp_path / "missing"), "agent", "run", "hello"]
    )
    assert result.exit_code != 0
    assert "Configuration file does not exist" in result.output


def test_tui_state_defaults_live_outside_install_directory(monkeypatch, tmp_path):
    import click

    from scienceflow.interfaces.cli.user_config import set_tui_state_defaults

    @click.command()
    @click.option("--session-log")
    def command(session_log):
        click.echo(session_log)

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    set_tui_state_defaults(command)
    first = CliRunner().invoke(command)
    assert first.exit_code == 0, first.output
    first_path = Path(first.output.strip())
    assert first_path.parent.parent == tmp_path / ".scienceflow/sessions"
    first_path.write_text("")
    second = CliRunner().invoke(command)
    assert second.exit_code == 0
    assert Path(second.output.strip()) != first_path
    restored = CliRunner().invoke(command, ["--resume"])
    assert restored.exit_code == 0
    assert Path(restored.output.strip()) == first_path


def test_environment_options_keep_click_boolean_and_numeric_types(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        main.commands["repl"], "callback", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setenv("SCIENCEFLOW_REPL_PLAIN", "true")
    monkeypatch.setenv("SCIENCEFLOW_REPL_ENABLE_SANDBOX", "false")
    monkeypatch.setenv("SCIENCEFLOW_REPL_SHOW_CODE", "8")
    result = CliRunner().invoke(main, ["repl"])
    assert result.exit_code == 0, result.output
    assert captured["plain"] is True
    assert captured["enable_sandbox"] is False
    assert captured["show_code"] == 8
    result = CliRunner().invoke(main, ["repl", "--enable-sandbox"])
    assert result.exit_code == 0, result.output
    assert captured["enable_sandbox"] is True
