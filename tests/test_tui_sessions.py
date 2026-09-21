from pathlib import Path

import click
from click.testing import CliRunner

from scienceflow.interfaces.cli.tui_sessions import configure_sessions


def test_sessions_isolate_workspaces_and_preserve_old_files(tmp_path):
    @click.command()
    @click.option('--workspace', type=click.Path(path_type=Path))
    @click.option('--session-log', type=click.Path(path_type=Path))
    @click.option('--operation-log', type=click.Path(path_type=Path))
    def command(workspace, session_log, operation_log):
        assert operation_log.parent == session_log.parent
        session_log.parent.mkdir(parents=True, exist_ok=True)
        if not session_log.exists():
            session_log.write_text('history')
        click.echo(session_log)

    configure_sessions(command, lambda: tmp_path / 'state')
    runner = CliRunner()
    first = runner.invoke(command, ['--workspace', str(tmp_path / 'a')])
    assert first.exit_code == 0, first.output
    second = runner.invoke(command, ['--workspace', str(tmp_path / 'b')])
    assert second.exit_code == 0, second.output
    assert Path(first.output.strip()).parent.parent != Path(second.output.strip()).parent.parent
    resumed = runner.invoke(command, ['--workspace', str(tmp_path / 'a'), '--resume'])
    assert resumed.output == first.output
    missing = runner.invoke(command, ['--workspace', str(tmp_path / 'c'), '--resume'])
    assert missing.exit_code != 0
    previous = tmp_path / 'legacy.jsonl'
    previous.write_text('original history')
    rejected = runner.invoke(command, ['--session-log', str(previous)])
    assert rejected.exit_code != 0
    assert previous.read_text() == 'original history'
    restored = runner.invoke(command, ['--session-log', str(previous), '--resume'])
    assert restored.exit_code == 0, restored.output
    assert previous.read_text() == 'original history'


def test_launch_directory_wins_over_environment_and_legacy_history_is_recoverable(monkeypatch, tmp_path):
    from hashlib import sha256

    from scienceflow.interfaces.cli.user_config import add_environment_options

    @click.command()
    @click.option('--workspace', type=click.Path(path_type=Path))
    @click.option('--session-log', type=click.Path(path_type=Path))
    def command(workspace, session_log):
        assert session_log.is_relative_to(workspace) or session_log.is_relative_to(tmp_path / 'state')
        session_log.touch()
        click.echo(session_log)

    configure_sessions(command, lambda: tmp_path / 'state')
    add_environment_options(command, ('TUI',))
    monkeypatch.setenv('SCIENCEFLOW_WORKSPACE', str(tmp_path / 'wrong'))
    monkeypatch.setenv('SCIENCEFLOW_TUI_WORKSPACE', str(tmp_path / 'wrong'))
    monkeypatch.setenv('SCIENCEFLOW_TUI_SESSION_LOG', str(tmp_path / 'wrong.jsonl'))
    launch = tmp_path / 'launch'
    launch.mkdir()
    monkeypatch.chdir(launch)
    runner = CliRunner()
    workspace_id = sha256(str(launch).encode()).hexdigest()[:20]
    legacy = tmp_path / 'state/workspaces' / workspace_id / 'old/session.jsonl'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('')
    restored = runner.invoke(command, ['--resume'])
    assert restored.exit_code == 0, restored.output
    assert Path(restored.output.strip()) == legacy
    fresh = runner.invoke(command)
    assert fresh.exit_code == 0, fresh.output
    assert Path(fresh.output.strip()).parent.parent == launch / '.scienceflow/sessions'
    local = runner.invoke(command, ['--resume'])
    assert local.output == fresh.output
    explicit = runner.invoke(command, ['--workspace', str(tmp_path / 'chosen')])
    assert explicit.exit_code == 0, explicit.output
    assert Path(explicit.output.strip()).parent.parent == tmp_path / 'chosen/.scienceflow/sessions'
    monkeypatch.chdir(tmp_path)
    missing = runner.invoke(command, ['--resume'])
    assert missing.exit_code != 0
    assert 'No saved session' in missing.output
