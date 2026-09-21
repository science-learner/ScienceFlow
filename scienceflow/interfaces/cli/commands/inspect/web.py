"""Workspace Web monitor launcher."""

import os
from pathlib import Path

import click

from scienceflow.interfaces.ui.web_monitor.service import instructions, start, stop


@click.command("web")
@click.argument("action", type=click.Choice(["start", "stop"]), default="start")
@click.option("--workspace", type=click.Path(file_okay=False, exists=True), default=".")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option(
    "--port",
    type=click.IntRange(0, 65535),
    default=8765,
    show_default=True,
    help="Use 0 to select an available port.",
)
@click.option("--open", "open_browser", is_flag=True)
def web(action, workspace, host, port, open_browser):
    """Start or stop the Web monitor and final-report viewer."""
    try:
        if action == "stop":
            click.echo(stop(Path(workspace)))
            return
        row = start(workspace, host=host, port=port)
        click.echo(instructions(row, workspace))
        if (
            open_browser
            and not os.environ.get("SSH_CONNECTION")
            and not os.environ.get("SSH_TTY")
        ):
            import webbrowser

            if not webbrowser.open(f"http://127.0.0.1:{row['port']}/{row['token']}/"):
                click.echo("Browser could not open automatically; copy the URL above.")
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
