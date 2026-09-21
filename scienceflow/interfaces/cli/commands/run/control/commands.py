"""Run commands default to detached experiments; --tui adds a terminal view."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import click

from scienceflow.runtime.parallel.control import (
    AmbiguousRun,
    list_runs,
    resume_run,
    start_run,
    stop_run,
)
from scienceflow.runtime.parallel.control.selection import (
    run_label,
    select_workspace_run,
    workspace_runs,
)


def open_tui(workspace, command):
    from scienceflow.interfaces.cli.commands.run.tui import build_tui_command

    build_tui_command(initial_command=command).main(args=['--workspace', str(workspace)], standalone_mode=False)


def select_run(target, action="status"):
    try:
        return select_workspace_run(action, target)
    except AmbiguousRun as exc:
        if not sys.stdin.isatty():
            raise
        from datetime import datetime
        for index, row in enumerate(exc.candidates, 1):
            created = datetime.fromtimestamp(row['created_at']).strftime('%Y-%m-%d %H:%M')
            click.echo(f"{index}. {run_label(row)} · {row['status']} · {created}")
        index = click.prompt('Choose a run (Ctrl+C cancels)', type=click.IntRange(1, len(exc.candidates)))
        return exc.candidates[index - 1]


def show(row):
    click.echo(f"{run_label(row)} · {row['status']}\nLogs: {row['log']}")


@click.command()
@click.argument('run_id', required=False)
def stop(run_id):
    """Stop by run ID or task directory; omitted target uses the current directory."""
    try:
        show(stop_run(select_run(run_id, 'stop')['run_id']))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.command()
@click.argument('run_id', required=False)
@click.option('--tui', 'with_tui', is_flag=True, help='Attach a TUI after resuming (or attach if already running).')
def resume(run_id, with_tui):
    """Resume by run ID or task directory; omitted target uses the current directory."""
    try:
        row = select_run(run_id, "resume")
        run_id = row['run_id']
        if not row['alive']:
            row = resume_run(run_id)
        show(row)
        if with_tui:
            workspace = Path((row.get('draft') or {}).get('workspace_base') or row['cwd'])
            if (row.get('draft') or {}).get('workspace_base'):
                workspace = workspace.parent
            open_tui(workspace, '/attach ' + shlex.quote(run_id))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.command()
@click.argument('run_id', required=False)
@click.option("--all", "all_workspaces", is_flag=True, help="List research across all workspaces.")
def status(run_id, all_workspaces):
    """Inspect one managed experiment, or list recent experiments."""
    try:
        rows = [select_workspace_run("status", run_id)] if run_id else (list_runs() if all_workspaces else workspace_runs(Path.cwd()))
        for row in rows:
            show(row)
        if not rows:
            click.echo('No managed runs yet.')
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def managed_run(target, *, manifest, with_tui, workspace, input_data_dir, workers, duration, cpu, gpu, config):
    try:
        if manifest:
            row = start_run(manifest)
            show(row)
            if with_tui:
                open_tui(workspace, '/attach ' + shlex.quote(row['run_id']))
            return
        arguments = [shlex.quote(target)] if target else []
        for key, value in [('data', input_data_dir), ('workers', workers), ('duration', duration), ('cpu', cpu), ('gpu', gpu)]:
            if value is not None:
                arguments.append(key + '=' + shlex.quote(str(value)))
        constraints = ' '.join(arguments)
        if with_tui:
            if config:
                raise ValueError('Use --manifest to supply a custom configuration with --tui.')
            open_tui(workspace, '/long-research ' + constraints)
        else:
            from .guided import prepare_and_start
            row = asyncio.run(prepare_and_start(Path(workspace).resolve(), constraints, config))
            show(row)
            click.echo(f"Stop: scienceflow stop {row['run_id']}\nResume: scienceflow resume {row['run_id']} --tui")
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
