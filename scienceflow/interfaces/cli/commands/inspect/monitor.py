"""Live monitor commands."""

from pathlib import Path

import click

@click.command("monitor")
@click.option(
    "--log-dir",
    "-l",
    default=None,
    type=click.Path(exists=False),
    help="Log directory containing monitor_state.json (single task)",
)
@click.option(
    "--manifest",
    "-m",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parallel manifest YAML (multi-task: one row per task)",
)
@click.option(
    "--refresh",
    "-r",
    default=2.0,
    show_default=True,
    help="Refresh interval in seconds",
)
def monitor_cmd(log_dir, manifest, refresh):
    """Watch REPL-native long-horizon run(s) via a Rich Live dashboard."""
    from scienceflow.interfaces.ui.monitor import MonitorDashboard, parse_monitor_manifest

    if bool(log_dir) == bool(manifest):
        raise click.UsageError("Specify exactly one of --log-dir or --manifest.")

    if manifest:
        entries = parse_monitor_manifest(manifest)
        if not entries:
            raise click.ClickException("Manifest has no tasks with workspace paths.")
        dashboard = MonitorDashboard(state_file=None, refresh_interval=refresh, task_entries=entries)
    else:
        state_file = Path(log_dir) / "monitor_state.json"
        dashboard = MonitorDashboard(state_file, refresh)
    dashboard.run()


@click.command("monitor-trace")
@click.option(
    "--manifest",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parallel manifest YAML to visualize.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="HTML file to write.",
)
@click.option(
    "--cache",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional JSON cache path. Defaults to monitor_trace_data.json next to output.",
)
@click.option(
    "--refresh",
    "-r",
    default=60.0,
    show_default=True,
    help="Refresh interval in seconds.",
)
@click.option("--once", is_flag=True, help="Render once and exit.")
def monitor_trace_cmd(manifest, output, cache, refresh, once):
    """Write a self-refreshing HTML trend view from monitor task logs."""
    from scienceflow.interfaces.ui.monitor_trace import run_monitor_trace

    run_monitor_trace(
        manifest_path=manifest,
        output_path=output,
        cache_path=cache,
        refresh_sec=refresh,
        once=once,
    )
