"""Root-level ScienceFlow TUI composed from InquiryCraft."""

from __future__ import annotations

from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from pathlib import Path

import click
from inquirycraft.cli import create_cli

from scienceflow.foundation.config.llm.model_registry import load_model_registry
from scienceflow.foundation.config.runtime.llm_lists import values
from scienceflow.interfaces.cli.user_config import set_tui_state_defaults
from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory, plural_cli_options
from scienceflow.interfaces.ui.logo import COMPACT_LOGO_PIXELS, LOGO_PIXELS
from scienceflow.interfaces.ui.long_research import LongResearchInteraction


def _pin_tui_chat_model(command: click.Command) -> click.Command:
    """Keep ordinary TUI chat on one alias while preserving endpoint failover."""

    callback = command.callback
    if callback is None:  # pragma: no cover - Click always supplies it here
        return command

    @wraps(callback)
    def invoke(**kwargs):
        aliases = values(kwargs.get("model"))
        if not aliases:
            registry = load_model_registry(Path(kwargs["model_config"]))
            aliases = list(registry.default_aliases("code_models")[:1])
        if len(aliases) != 1:
            raise click.ClickException(
                "TUI chat accepts exactly one model alias; switch it later with /models."
            )
        kwargs["model"] = aliases[0]
        return callback(**kwargs)

    command.callback = invoke
    for param in command.params:
        if param.name == "model":
            param.help = (
                "One chat model alias from models.json; defaults to the first code model."
            )
    return command


def build_tui_command(*, initial_command: str = "") -> click.Command:
    """Reuse InquiryCraft's TUI command while injecting one ScienceFlow interceptor."""

    required_host_parameters = {
        "include_tui",
        "model_config_default",
        "model_default_key",
        "tui_branding",
        "tui_interceptor_factory",
    }
    if not required_host_parameters.issubset(signature(create_cli).parameters):

        @click.command("tui")
        def unavailable_tui():
            """Start the full-screen interface (requires InquiryCraft TUI host APIs)."""
            raise click.ClickException(
                "ScienceFlow TUI requires the InquiryCraft 0.9.0 host and model-registry APIs. "
                "Install that release before starting TUI."
            )

        return unavailable_tui

    from inquirycraft.tui import TuiBranding

    try:
        package_version = version("scienceflow")
    except PackageNotFoundError:
        package_version = "dev"

    host = create_cli(
        name="scienceflow-tui-host",
        runtime_factory=PoolRuntimeFactory(),
        model_config_default=Path("~/.config/scienceflow/models.json"),
        model_default_key="code_models",
        include_tui=True,
        tui_branding=TuiBranding(
            name="ScienceFlow",
            large_title=True,
            large_subtitle=False,
            accent_text="Flow",
            accent_color="#ef8398",
            hide_compact_subtitle=True,
            pixels=LOGO_PIXELS,
            compact_pixels=COMPACT_LOGO_PIXELS,
            subtitle="Making Long-Horizon Research Recoverable, Adaptive, and Efficient",
            logo="SF · ScienceFlow   /   Research Assistant",
            compact_logo="SF · ScienceFlow",
            notice_label="SCIENCEFLOW",
            organization_name="HUAWEI",
            lab_name="Noah Lab",
            team_name="ScienceFlow Team",
            version=package_version,
            welcome_text=(
                "Long Research · parallel workers explore, evaluate, and preserve the best result.\n"
                "Start with /long-research · monitor with /tasks · continue with /resume"
            ),
            welcome_emphasis=("Long Research", "parallel workers", "best result"),
        ),
        tui_interceptor_factory=lambda workspace: LongResearchInteraction(
            Path(workspace), managed=True, multi_task=True, initial_command=initial_command
        ),
    )
    command = host.commands.get("tui")
    if command is None:  # pragma: no cover - defensive contract guard
        raise RuntimeError("InquiryCraft did not register its TUI command")
    for param in command.params:
        if param.name == "system_prompt":
            param.default = (
                "You are ScienceFlow, a scientific research and engineering assistant. "
                "Help users prepare and manage general research tasks, including mathematical "
                "optimization, circle packing and machine learning benchmarks. "
                "For host preparation requests, inspect relevant task materials using tools "
                "and return the requested JSON contract. Do not start research workers "
                "during preparation."
            )
    return set_tui_state_defaults(_pin_tui_chat_model(plural_cli_options(command)))


tui = build_tui_command()

__all__ = ["build_tui_command", "tui"]
