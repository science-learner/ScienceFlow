"""Console preparation reuses the same TUI-independent task preparation flow."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import yaml

from scienceflow.interfaces.ui.long_research import LongResearchInteraction
from scienceflow.research.onboarding import OnboardingState, run_preflight
from scienceflow.research.onboarding.library.catalog import (
    resolve_task_selection,
    task_selection_text,
)
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

    async def run_host_task(self, prompt, *, show_response=True, runtime=None):
        from inquirycraft.runtime import RuntimeOptions

        from scienceflow.foundation.config.llm.model_registry import load_model_registry
        from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory

        if runtime is None and self.runtime is None:
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
        active = runtime if runtime is not None else self.runtime
        async for event in active.run_stream(prompt):
            if event.type == "tool.completed":
                click.echo("Checking task files…")
        for message in reversed(active.conversation.messages):
            if message.role == "assistant" and message.content:
                return message.content
        raise ValueError("Preparation agent returned no answer.")


def terminal_is_interactive():
    """Require both streams so pipes never block on an unexpected prompt."""
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def choose_task(workspace: Path) -> str:
    """Select a registered task or collect a free-form description."""
    if not terminal_is_interactive():
        raise click.UsageError(
            "Missing TARGET. Use `scienceflow run TASK`, "
            '`scienceflow run "new task description"`, or `scienceflow run --manifest FILE`.'
        )
    prompt, choices = task_selection_text(workspace)
    click.echo(prompt)
    while True:
        answer = click.prompt("Task number, name, path, or description", type=str)
        try:
            selected = resolve_task_selection(answer, choices)
        except ValueError as exc:
            click.echo(str(exc), err=True)
            continue
        if selected is not None:
            return selected
        description = click.prompt("Describe the new task", type=str).strip()
        if description:
            return description
        click.echo("Task description cannot be empty.", err=True)


async def prepare_and_start(workspace: Path, constraints: str, config: str | None):
    workspace.mkdir(parents=True, exist_ok=True)
    host = ConsoleHost(workspace)
    from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory

    interaction = LongResearchInteraction(
        workspace, preparation_runtime_factory=PoolRuntimeFactory()
    )
    try:
        await interaction.try_handle("/long-research " + constraints, host)
        while True:
            task = interaction._preparation.task
            if task is not None and not task.done():
                await task
                # Location resolution may have scheduled the actual preparation.
                if interaction._preparation.task is not task:
                    continue
            if interaction.session is None:
                break
            if interaction.session.state == OnboardingState.CONFIRM:
                manifest = interaction.files.manifest_path
                if config:
                    payload = yaml.safe_load(manifest.read_text())
                    payload["defaults"]["config"] = str(Path(config).resolve())
                    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
                    if not (await asyncio.to_thread(run_preflight, manifest)).ok:
                        raise ValueError("Preflight failed with the supplied config.")
                return await asyncio.to_thread(
                    start_run, manifest, draft=interaction.session.draft
                )
            if not sys.stdin.isatty():
                raise ValueError(
                    "Task needs more information. Use --tui or supply complete task/resource options."
                )
            answer = await asyncio.to_thread(click.prompt, "Reply (/cancel to exit)")
            await interaction.try_handle(answer, host)
        raise ValueError("Task preparation cancelled.")
    finally:
        await interaction.aclose()
        if host.runtime is not None:
            await host.runtime.aclose()
