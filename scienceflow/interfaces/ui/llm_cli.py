"""Compose InquiryCraft commands with ScienceFlow's plural endpoint pools."""

from __future__ import annotations

from dataclasses import replace

import click
from inquirycraft.adapters import AskLLMAdapter
from inquirycraft.cli import ChatMonitor
from inquirycraft.memory import Conversation, JsonlConversationStore
from inquirycraft.runtime import AgentRuntime
from inquirycraft.tools import default_tools

from scienceflow.foundation.config.llm.llm_factory import build_stage_llm
from scienceflow.foundation.config.runtime.llm_lists import values
from scienceflow.foundation.config.schema.settings import StageConfig


class PoolRuntimeFactory:
    def create_runtime(self, options, *, api_key, base_url, event_sink):
        models = values(options.model)
        if not models or not values(api_key):
            raise click.ClickException(
                "models and api_keys must each contain at least one value"
            )
        stage = StageConfig(
            models=models,
            api_keys=values(api_key),
            base_urls=values(base_url),
            max_tokens=options.max_tokens or 32768,
            http_timeout=options.timeout or 300,
            api_routing_mode=options.model_routing,
        )
        try:
            client = AskLLMAdapter(build_stage_llm(stage))
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        store = (
            JsonlConversationStore(options.session_log) if options.session_log else None
        )
        monitor = ChatMonitor(
            options,
            client.backend,
            values(base_url),
            models,
            len(values(api_key)),
        )
        return monitor.bind(AgentRuntime(
            llm=client,
            compaction_policy=monitor.policy(),
            provider_observer=monitor,
            options=replace(options, model=models[0]),
            tools=default_tools(),
            event_sink=event_sink,
            conversation=store.load() if store else Conversation(),
            conversation_store=store,
        ))


def plural_cli_options(command):
    """Keep the plural model alias flag on composed InquiryCraft commands."""
    aliases = {
        "model": ("--models", "--model"),
    }
    for param in command.params:
        if param.name in aliases:
            param.opts = list(aliases[param.name])
            param.help = "Comma-separated aliases from models.json; defaults to code_models."
    if isinstance(command, click.Group):
        for child in command.commands.values():
            plural_cli_options(child)
    return command
