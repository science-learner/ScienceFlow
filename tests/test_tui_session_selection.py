"""Workspace-local TUI session selection."""

import os
from pathlib import Path

import click
from click.testing import CliRunner

from scienceflow.interfaces.cli.tui_sessions import configure_sessions


def _command():
    @click.command()
    @click.option("--workspace", type=click.Path(path_type=Path))
    @click.option("--session")
    @click.option("--resume", is_flag=True)
    def command(
        workspace,
        session_log=None,
        event_log=None,
        operation_log=None,
        resumed_session=False,
    ):
        click.echo(session_log.parent.name)

    return configure_sessions(command)


def test_resume_prefers_most_recently_updated_session_and_accepts_prefix(tmp_path):
    root = tmp_path / ".scienceflow/sessions"
    older = root / "20260101T000000000000Z-old"
    newer = root / "20260201T000000000000Z-new"
    for directory in (older, newer):
        directory.mkdir(parents=True)
        (directory / "session.jsonl").write_text("history")
    os.utime(newer / "session.jsonl", (100, 100))
    os.utime(older / "session.jsonl", (100, 100))
    (older / "events.jsonl").write_text("recent event")
    command = _command()
    runner = CliRunner()

    latest = runner.invoke(command, ["--workspace", str(tmp_path), "--resume"])
    assert latest.exit_code == 0, latest.output
    assert latest.output.strip() == older.name

    selected = runner.invoke(
        command, ["--workspace", str(tmp_path), "--resume", "--session", "20260201"]
    )
    assert selected.exit_code == 0, selected.output
    assert selected.output.strip() == newer.name


def test_resume_only_reads_sessions_in_the_workspace(monkeypatch, tmp_path):
    state = tmp_path / "old-state"
    old = state / "workspaces/old/session.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text("history")
    monkeypatch.setenv("SCIENCEFLOW_TUI_STATE_DIR", str(state))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = CliRunner().invoke(_command(), ["--workspace", str(workspace), "--resume"])
    assert result.exit_code == 1
    assert "No saved session in this workspace" in result.output
