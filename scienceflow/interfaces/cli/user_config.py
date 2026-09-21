"""User configuration commands and environment defaults for packaged CLI use."""

from __future__ import annotations

import os
from pathlib import Path

import click
from inquirycraft.llm import write_model_registry_template

from scienceflow.foundation.config.llm.model_registry import default_model_config_path


@click.group("config")
def config_command():
    """Manage the user configuration file."""


@config_command.command("path")
def config_path():
    """Print the user configuration path without displaying credentials."""
    click.echo(default_model_config_path())


@config_command.command("init")
def config_init():
    """Create a private configuration template; never overwrite an existing file."""
    path = default_model_config_path()
    try:
        write_model_registry_template(path)
    except FileExistsError:
        click.echo(f"Configuration already exists: {path}")
        return
    except OSError as exc:
        raise click.ClickException(
            f"Cannot create configuration at {path}: {exc.strerror}"
        ) from exc
    click.echo(f"Created {path}; edit its models and endpoints.")


def add_environment_options(command: click.Command, path: tuple[str, ...] = ()) -> None:
    """Bind every existing option without changing its type or CLI parameter name."""
    for param in command.params:
        if not isinstance(param, click.Option):
            continue
        if param.name in {"workspace", "model", "model_config"} or getattr(
            param, "scienceflow_explicit_only", False
        ):
            param.envvar = None
            param.allow_from_autoenv = False
            param.show_envvar = False
            continue
        option = next((item for item in param.opts if item.startswith("--")), "")
        if not option:
            continue
        name = option[2:].replace("-", "_").upper()
        specific = "_".join(("SCIENCEFLOW", *path, name))
        common = f"SCIENCEFLOW_{name}"
        existing = (
            [param.envvar]
            if isinstance(param.envvar, str)
            else list(param.envvar or ())
        )
        param.envvar = list(dict.fromkeys([specific, common, *existing]))
        param.show_envvar = True
    if isinstance(command, click.Group):
        for name, child in command.commands.items():
            if name != "config":
                add_environment_options(child, (*path, name.replace("-", "_").upper()))


def user_state_dir() -> Path:
    explicit = os.environ.get("SCIENCEFLOW_TUI_STATE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base.expanduser() / "scienceflow" / "tui"


def set_tui_state_defaults(command: click.Command) -> click.Command:
    from scienceflow.interfaces.cli.tui_sessions import configure_sessions

    return configure_sessions(command, user_state_dir)
