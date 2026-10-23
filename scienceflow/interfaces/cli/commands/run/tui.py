"""Root-level ScienceFlow TUI composed from InquiryCraft."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from pathlib import Path

import click
from inquirycraft.cli import create_cli

from scienceflow.foundation.config.llm.model_registry import (
    default_model_config_path,
    load_model_registry,
)
from scienceflow.foundation.config.runtime.llm_lists import values
from scienceflow.interfaces.cli.user_config import set_tui_state_defaults
from scienceflow.interfaces.ui.llm_cli import TuiPoolRuntimeFactory, plural_cli_options
from scienceflow.interfaces.ui.logo import COMPACT_LOGO_PIXELS, LOGO_PIXELS
from scienceflow.interfaces.version import get_version


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
            param.help = "One chat model alias from models.json; defaults to the first code model."
    return command


def build_tui_command(*, initial_command: str = "") -> click.Command:
    """Reuse InquiryCraft's TUI command while injecting one ScienceFlow interceptor."""

    import inquirycraft.tui as tui_host
    from inquirycraft.runtime import AgentRuntime

    required_host_parameters = {
        "include_tui",
        "model_config_default",
        "model_default_key",
        "tui_branding",
        "tui_interceptor_factory",
    }
    missing = []
    if not hasattr(tui_host, "TuiInputPreprocessor"):
        missing.append("pre-history input hook (TuiInputPreprocessor)")
    missing_parameters = (
        required_host_parameters - signature(create_cli).parameters.keys()
    )
    if missing_parameters:
        missing.append("CLI host parameters: " + ", ".join(sorted(missing_parameters)))
    if not callable(getattr(AgentRuntime, "compact_context", None)):
        missing.append("manual context compaction (AgentRuntime.compact_context)")
    if missing:

        @click.command("tui")
        def unavailable_tui():
            """Start the full-screen interface (requires InquiryCraft TUI host APIs)."""
            raise click.ClickException(
                "ScienceFlow TUI cannot start: InquiryCraft is missing "
                + "; ".join(missing)
                + ". "
                + "Install inquirycraft[openai,tui]==0.9.6 before starting TUI."
            )

        return unavailable_tui

    from inquirycraft.tui import TuiBranding

    # Optional presentation fields remain signature-filtered for host extensions.
    branding_parameters = signature(TuiBranding).parameters
    presentation_options = {
        key: value
        for key, value in {
            "prompt_placeholder": "Type a message · /long-research · /help",
            "notice_background": "#3B4048",
            "fold_system_notices": True,
        }.items()
        if key in branding_parameters
    }

    package_version = get_version()

    runtime_factory = TuiPoolRuntimeFactory()

    def build_interceptor(workspace):
        from scienceflow.interfaces.ui.long_research import LongResearchInteraction

        return LongResearchInteraction(
            Path(workspace),
            managed=True,
            multi_task=True,
            initial_command=initial_command,
            preparation_runtime_factory=runtime_factory,
        )

    host = create_cli(
        name="scienceflow-tui-host",
        runtime_factory=runtime_factory,
        model_config_default=default_model_config_path(),
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
                "Start with /long-research · monitor with /tasks · resume a task with /resume N"
            ),
            welcome_emphasis=("Long Research", "parallel workers", "best result"),
            **presentation_options,
        ),
        tui_interceptor_factory=build_interceptor,
    )
    command = host.commands.get("tui")
    if command is None:  # pragma: no cover - defensive contract guard
        raise RuntimeError("InquiryCraft did not register its TUI command")
    for param in command.params:
        if param.name == "model_config":
            param.default = default_model_config_path
        if param.name == "system_prompt":
            param.default = (
                "You are ScienceFlow, a scientific research and engineering assistant. "
                "Help users prepare and manage general research tasks, including mathematical "
                "optimization, circle packing and machine learning benchmarks. "
                "For host preparation requests, inspect relevant task materials using tools "
                "and return the requested JSON contract. Do not start research workers "
                "during preparation."
            )
    return _pin_tui_chat_model(set_tui_state_defaults(plural_cli_options(command)))


tui = build_tui_command()

__all__ = ["build_tui_command", "tui"]
