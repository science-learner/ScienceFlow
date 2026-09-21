"""Console preparation reuses the same TUI-independent task preparation flow."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import yaml

from scienceflow.interfaces.ui.long_research import LongResearchInteraction
from scienceflow.research.onboarding import OnboardingState, run_preflight
from scienceflow.runtime.parallel.control import start_run


class ConsoleHost:
    busy = False

    def __init__(self, workspace):
        self.workspace = workspace
        self.runtime = None

    def conversation_snapshot(self):
        return ()

    def set_host_status(self, text):
        pass

    async def notice(self, text):
        click.echo(text)

    async def run_host_task(self, prompt):
        from inquirycraft.runtime import RuntimeOptions

        from scienceflow.foundation.config.llm.model_registry import load_model_registry
        from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory

        if self.runtime is None:
            registry = load_model_registry()
            resolved = registry.resolve(
                registry.default_aliases("code_models"), role="code"
            )
            self.runtime = PoolRuntimeFactory().create_runtime(
                RuntimeOptions(
                    model=",".join(resolved.models),
                    workspace=self.workspace,
                    stream_llm=True,
                ),
                api_key=",".join(resolved.api_keys),
                base_url=",".join(resolved.base_urls),
                event_sink=None,
            )
        async for event in self.runtime.run_stream(prompt):
            if event.type == 'tool.completed':
                click.echo('Checking task files…')
        for message in reversed(self.runtime.conversation.messages):
            if message.role == 'assistant' and message.content:
                return message.content
        raise ValueError('Preparation agent returned no answer.')


async def prepare_and_start(workspace: Path, constraints: str, config: str | None):
    workspace.mkdir(parents=True, exist_ok=True)
    host = ConsoleHost(workspace)
    interaction = LongResearchInteraction(workspace)
    try:
        await interaction.try_handle('/long-research ' + constraints, host)
        while interaction.session is not None:
            task = interaction._preparation.task
            if task is not None and not task.done():
                await task
            if interaction.session.state == OnboardingState.CONFIRM:
                manifest = interaction.files.manifest_path
                if config:
                    payload = yaml.safe_load(manifest.read_text())
                    payload['defaults']['config'] = str(Path(config).resolve())
                    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
                    if not (await asyncio.to_thread(run_preflight, manifest)).ok:
                        raise ValueError('Preflight failed with the supplied config.')
                return await asyncio.to_thread(start_run, manifest, draft=interaction.session.draft)
            if not sys.stdin.isatty():
                raise ValueError('Task needs more information. Use --tui or supply complete task/resource options.')
            answer = await asyncio.to_thread(click.prompt, 'Reply (/cancel to exit)')
            await interaction.try_handle(answer, host)
        raise ValueError('Task preparation cancelled.')
    finally:
        await interaction.aclose()
        if host.runtime is not None:
            await host.runtime.aclose()
