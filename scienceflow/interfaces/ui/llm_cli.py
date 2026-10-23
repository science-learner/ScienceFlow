"""Compose InquiryCraft commands with ScienceFlow's plural endpoint pools."""

from __future__ import annotations

from dataclasses import replace

import click
from inquirycraft.tools import ShellTool

from scienceflow.foundation.config.llm.model_registry import default_model_config_path
from scienceflow.foundation.config.runtime.llm_lists import values
from scienceflow.foundation.config.schema.settings import StageConfig


def build_stage_llm(stage: StageConfig):
    """Load the provider client only when a chat runtime is created."""
    from scienceflow.foundation.config.llm.llm_factory import build_stage_llm as build

    return build(stage)


class PoolRuntimeFactory:
    def _tools(self):
        from inquirycraft.tools import default_tools

        return default_tools()

    def create_runtime(self, options, *, api_key, base_url, event_sink, tools=None):
        from inquirycraft.adapters import AskLLMAdapter
        from inquirycraft.cli import ChatMonitor
        from inquirycraft.cli.chat_profiles import ProfileStore
        from inquirycraft.llm import SameRoundRetryPolicy
        from inquirycraft.memory import Conversation, JsonlConversationStore
        from inquirycraft.runtime import AgentRuntime

        models = values(options.model)
        keys = values(api_key)
        endpoints = values(base_url)
        if not models or not keys:
            raise click.ClickException(
                "models and api_keys must each contain at least one value"
            )
        stage = StageConfig(
            models=models,
            api_keys=keys,
            base_urls=endpoints,
            use_proxies=list(getattr(options, "use_proxies", ())),
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
            endpoints,
            models,
            len(keys),
            store=ProfileStore(
                default_model_config_path().with_name("chat_profiles.json"),
                fallback_path=ProfileStore().path,
            ),
        )
        monitor.backend_builder = lambda resolved, routing, max_tokens: build_stage_llm(
            StageConfig(
                models=list(resolved.models),
                api_keys=list(resolved.api_keys),
                base_urls=list(resolved.base_urls),
                use_proxies=list(
                    getattr(resolved, "use_proxies", (None,) * len(resolved.api_keys))
                ),
                max_tokens=max_tokens,
                http_timeout=options.timeout or 300,
                api_routing_mode=routing,
            )
        )
        return monitor.bind(
            AgentRuntime(
                llm=client,
                llm_retry_policy=SameRoundRetryPolicy(
                    max_attempts=3,
                    base_delay_sec=1.0,
                    max_delay_sec=4.0,
                ),
                compaction_policy=monitor.policy(),
                provider_observer=monitor,
                options=replace(options, model=models[0]),
                tools=self._tools() if tools is None else tools,
                event_sink=event_sink,
                conversation=store.load() if store else Conversation(),
                conversation_store=store,
            )
        )


class TuiShellTool(ShellTool):
    """Shell execution for ScienceFlow TUI chat and task preparation."""

    description = (
        "Run a shell command as the current user with full host/network access "
        "and no OS sandbox. No default timeout; optionally set a timeout "
        "in seconds (maximum 3600)."
    )


class TuiPoolRuntimeFactory(PoolRuntimeFactory):
    def _tools(self):
        from inquirycraft.tools import ReadTextTool, ToolCollection, WriteTextTool

        from scienceflow.interfaces.ui.research.control.workspace_tool import (
            WorkspaceInfoTool,
        )

        return ToolCollection(
            (ReadTextTool(), WriteTextTool(), WorkspaceInfoTool(), TuiShellTool())
        )


def plural_cli_options(command):
    """Keep the plural model alias flag on composed InquiryCraft commands."""
    aliases = {
        "model": ("--models", "--model"),
    }
    for param in command.params:
        if param.name in aliases:
            param.opts = list(aliases[param.name])
            param.help = (
                "Comma-separated aliases from models.json; defaults to code_models."
            )
    if isinstance(command, click.Group):
        for child in command.commands.values():
            plural_cli_options(child)
    return command
