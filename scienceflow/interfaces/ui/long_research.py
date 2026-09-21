"""ScienceFlow's thin ``/long-research`` adapter for the InquiryCraft TUI."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scienceflow.interfaces.ui.research.research_preparation import ResearchPreparation
from scienceflow.research.onboarding import (
    LongResearchSession,
    OnboardingFiles,
    OnboardingState,
    PreflightReport,
    run_preflight,
    write_onboarding_files,
)
from scienceflow.runtime.core.support.system_resources import query_gpu_devices
from scienceflow.runtime.parallel.service import ParallelRunSummary, run_manifest

if TYPE_CHECKING:
    from inquirycraft.tui import TuiHostPort


_COMMANDS = frozenset({"/long-research", "/long_research"})


def long_research_argument(text: str) -> str | None:
    """Return inline constraints for the exact command, or ``None`` otherwise."""

    command, separator, argument = str(text or "").strip().partition(" ")
    if command not in _COMMANDS:
        return None
    return argument.strip() if separator else ""


def _preflight_summary(report: PreflightReport) -> str:
    """Keep successful checks quiet and retain actionable failure details."""
    failed = [check for check in report.checks if not check.ok]
    if not failed:
        return f"Preflight: {report.status.value} · all checks passed."
    details = "\n".join(f"- {check.name}: {check.detail}" for check in failed)
    return f"Preflight: {report.status.value}.\nFailed checks:\n{details}"


class LongResearchInteraction:
    """Own only TUI routing; onboarding and execution stay in existing owners."""

    @property
    def tui_commands(self):
        from inquirycraft.tui import TuiCommandSpec

        return (
            TuiCommandSpec("/long-research", "Start guided long research",
                           ("task=", "data=", "gpu=", "cpu=", "workers=", "duration=",
                            "models=", "code-models=", "feedback-models=", "model-policy="),
                           path_parameters=("data",)),
            TuiCommandSpec("/resources", "Host hardware and storage"),
            TuiCommandSpec("/models", "Show or switch the chat model", ("policy=",)),
            TuiCommandSpec("/web", "Open workspace Web monitor; /web stop stops only Web", ("stop",)),
            TuiCommandSpec("/research-usage", "Worker tokens and cache usage"),
            *((TuiCommandSpec("/stop", "Stop research and preserve progress"),
            TuiCommandSpec("/resume", "Resume a saved research run"),
            TuiCommandSpec("/attach", "Attach to a running research task"),
            TuiCommandSpec("/tasks", "Show all tasks in this workspace"),
            TuiCommandSpec("/runs", "List this workspace research history"),
            TuiCommandSpec("/select", "Choose a task from the displayed list")) if self._managed else ()),
        )

    def __init__(self, workspace: str | Path, *, managed: bool = False, multi_task: bool = False, initial_command: str = "") -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=False)
        self.session: LongResearchSession | None = None
        self.files: OnboardingFiles | None = None
        self.preflight: PreflightReport | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._host: TuiHostPort | None = None
        self._progress = None
        self._resource_task = None
        self._resources = None
        self._preparation = ResearchPreparation(self)
        from scienceflow.interfaces.ui.research.control import ManagedResearch
        self._managed = ManagedResearch(self) if managed else None
        if managed and multi_task:
            from .research.control.tasks.manager import MultiResearch
            self._managed = MultiResearch(self)
        self._task_workspace = None
        self._model_config_path = Path("~/.config/scienceflow/models.json").expanduser()
        self._selected_model_aliases: tuple[str, ...] = ()
        self._selected_feedback_aliases: tuple[str, ...] = ()
        self._model_selection = "auto"
        self._research_models_confirmed = False
        self._awaiting_research_models = False
        self.initial_command = initial_command

    def configure_model_registry(self, path: Path, aliases: tuple[str, ...]) -> None:
        from scienceflow.foundation.config.llm.model_registry import load_model_registry

        self._model_config_path = path.expanduser().resolve(strict=False)
        registry = load_model_registry(self._model_config_path)
        self._selected_model_aliases = (
            tuple(aliases) or registry.default_aliases("code_models")
        )
        self._selected_feedback_aliases = registry.default_aliases("feedback_models")
        self._model_selection = str(registry.defaults.get("selection") or "auto")

    def _remember_chat_models(self, aliases: tuple[str, ...]) -> None:
        self._selected_model_aliases = tuple(aliases)

    def tui_completions(self, text):
        from scienceflow.interfaces.ui.model_config import model_alias_completions

        matches = model_alias_completions(self._model_config_path, text)
        if matches is not None:
            return matches
        if hasattr(self._managed, 'task_catalog'):
            from .research.control.tasks.completion import task_completions
            return task_completions(self._managed, text)
        return None

    async def on_mount(self, host) -> None:
        from scienceflow.interfaces.ui.model_config import (
            model_config_setup_hint,
            model_registry_brief,
        )
        from scienceflow.interfaces.ui.research.resources import ResourceMonitor

        if hasattr(host, "set_resource_status"):
            self._resources = ResourceMonitor(self.workspace)
            self._resource_task = asyncio.create_task(self._resources.watch(host))

        try:
            model_registry_brief(
                self._model_config_path,
                self._selected_model_aliases,
                self._model_selection,
            )
        except ValueError as exc:
            await host.notice(
                f"Model configuration unavailable: {exc}\n"
                f"{model_config_setup_hint()}"
            )

        if self.initial_command:
            await self.try_handle(self.initial_command, host)
        elif self._managed:
            await self._managed.on_mount(host)

    @property
    def task_workspace(self):
        return self._task_workspace or self.workspace

    @property
    def running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()

    async def try_handle(self, text: str, host: TuiHostPort) -> bool:
        if text.strip().split(maxsplit=1)[:1] == ["/models"]:
            from scienceflow.interfaces.ui.model_config import (
                model_config_hint,
                model_config_setup_hint,
                model_registry_summary,
                parse_chat_model_selection,
            )

            await _echo_user(host, text)
            try:
                if text.strip() == "/models":
                    if host.busy:
                        await host.notice(
                            "Finish or cancel the current chat request before changing models."
                        )
                        return True
                    open_selector = getattr(host, "open_chat_model_selector", None)
                    if callable(open_selector):
                        from scienceflow.foundation.config.llm.model_registry import (
                            load_model_registry,
                        )

                        registry = load_model_registry(self._model_config_path)
                        aliases = registry.default_aliases("code_models")
                        open_selector(
                            self._model_config_path,
                            aliases=aliases,
                            current=self._selected_model_aliases,
                            selection=self._model_selection,
                            config_hint=model_config_hint(self._model_config_path),
                            on_applied=self._remember_chat_models,
                        )
                        return True
                    await host.notice(
                        model_registry_summary(
                            self._model_config_path,
                            self._selected_model_aliases,
                            self._model_selection,
                        )
                    )
                    return True
                if host.busy:
                    await host.notice(
                        "Finish or cancel the current chat request before changing models."
                    )
                    return True
                aliases, policy = parse_chat_model_selection(
                    text,
                    self._model_config_path,
                    current_policy=self._model_selection,
                )
                switch = getattr(host, "switch_chat_models", None)
                if not callable(switch):
                    raise ValueError("This TUI host cannot switch chat models.")
                applied = await switch(
                    self._model_config_path,
                    aliases=aliases,
                    selection=policy,
                )
                self._selected_model_aliases = tuple(applied)
                self._model_selection = policy
                await host.notice(
                    f"Chat model applied · {', '.join(applied)} · Policy {policy}"
                )
            except (OSError, ValueError) as exc:
                await host.notice(
                    f"Model configuration unavailable: {exc}\n"
                    f"{model_config_setup_hint()}"
                )
            return True
        if text.strip().split(maxsplit=1)[:1] == ['/web']:
            from scienceflow.interfaces.ui.web_monitor.service import (
                instructions,
                start,
                stop,
            )
            await _echo_user(host, text)
            try:
                if text.strip() == '/web stop':
                    await host.notice(await asyncio.to_thread(stop, self.workspace))
                elif text.strip() == '/web':
                    row = await asyncio.to_thread(start, self.workspace)
                    message = instructions(row, self.workspace)
                    if callable(getattr(host, 'notice_rich', None)):
                        from rich.text import Text
                        url = f"http://127.0.0.1:{row['port']}/{row['token']}/"
                        content = Text(message, style='#9297A3')
                        offset = message.index(url)
                        content.stylize(f'#8FA9C4 underline link {url}', offset, offset + len(url))
                        await host.notice_rich(content)
                    else:
                        await host.notice(message)
                else:
                    await host.notice('Use /web or /web stop.')
            except (OSError, ValueError) as exc:
                await host.notice(f'Web monitor unavailable: {exc}')
            return True
        if self._managed and await self._managed.command(text, host):
            return True
        if text.strip().casefold() == "/research-usage":
            await host.notice(self._progress.usage.details() if self._progress else "No active research usage monitor.")
            return True
        if text.strip().casefold() == "/resources":
            summary = (
                self._resources.latest
                if self._resources
                else "Resources unavailable"
            )
            resource_summary = getattr(self._managed, 'resource_summary', None)
            if callable(resource_summary):
                summary = resource_summary(summary)
            await host.notice(summary)
            return True
        if text.strip().casefold() == "/help":
            await host.notice('/web opens the workspace Web monitor · /web stop stops only Web')
            await host.notice(
                "/models selects the chat model · configure with "
                "scienceflow config init · locate with scienceflow config path"
            )
            if callable(getattr(self._managed, "new_workspace", None)):
                await host.notice("/long-research prepares a new task · /tasks overview · /status N details · /stop N stops one task · /resume N continues it · /research-usage N tokens · /resources host hardware")
                return False
            await host.notice("/long-research starts research · /status progress · /resources host hardware · /research-usage worker tokens · /stop stops · /resume continues this workspace · /runs lists local history")
            return False
        constraints = long_research_argument(text)
        if constraints is not None:
            await _echo_user(host, text)
            await self._start(constraints, host)
            return True

        if self.session is None:
            return False

        normalized = str(text or "").strip().casefold()
        if normalized in {"/help", "/clear", "/quit"}:
            return False
        if self.running and host.busy and normalized in {"/cancel", "cancel", "取消"}:
            return False
        if self.running and normalized not in {"/status", "status", "/cancel", "cancel", "取消", "run", "/run", "运行"}:
            return False
        await _echo_user(host, text)
        if normalized in {"/status", "status"}:
            await host.notice(self._status_text())
            return True
        if normalized in {"/cancel", "cancel", "取消"}:
            await self._cancel(host)
            return True
        if self._awaiting_research_models and not self.running:
            await self._apply_research_model_answer(host, text)
            return True
        action_words = {"run", "/run", "运行", "confirm", "/confirm", "确认", "preflight"}
        if self.session.state == OnboardingState.CONFIRM and not self._preparation.question:
            action_words.add("yes")
        if normalized not in action_words and not self.running:
            if await self._preparation.begin(host, text):
                return True
        if self._preparation.task is not None and not self._preparation.task.done():
            await host.notice("Task preparation is still running. Use /cancel to stop it.")
            return True
        if self._preparation.question and normalized in {"run", "/run", "运行", "preflight"}:
            await host.notice(self._preparation.question)
            return True
        update = self.session.answer(text)
        if update.action == "preflight":
            await self._preflight(host)
        elif update.action == "run":
            from .research.control.notices import research_notice

            await research_notice(host, update.message)
            self._run_task = asyncio.create_task(self._managed.launch(host) if self._managed else self._run(host))
        else:
            await host.notice(
                await self._decorate_prompt(update.message, update.question)
            )
            self._set_status(host)
        return True

    async def aclose(self) -> None:
        """Cancel an owned Parallel run when its TUI host shuts down."""

        await self._preparation.close()
        if callable(getattr(self._managed, "close", None)):
            await self._managed.close()
        if self._resource_task is not None:
            self._resource_task.cancel()
            await asyncio.gather(self._resource_task, return_exceptions=True)
            self._resource_task = None
        task = self._run_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._run_task = None
        self.session = None
        self._research_models_confirmed = False
        self._awaiting_research_models = False
        if self._host is not None:
            self._host.set_host_status("")
        self._host = None

    async def _start(self, constraints: str, host: TuiHostPort) -> None:
        if host.busy:
            await host.notice(
                "Agent is running; finish or cancel it before long research."
            )
            return
        if self.session is not None:
            await host.notice(
                "Long research is already active. Use /status or /cancel."
            )
            return
        self._host = host
        try:
            from scienceflow.interfaces.ui.model_config import (
                configure_draft_models,
                research_model_overrides,
            )

            if callable(getattr(self._managed, "new_workspace", None)):
                self._task_workspace = Path(await self._managed.new_workspace(constraints, host))
            self.session = LongResearchSession.start(
                host.conversation_snapshot(),
                workspace=self.workspace,
                constraints=constraints,
                natural_language=callable(getattr(host, "run_host_task", None)),
            )
            if self._task_workspace is not None:
                self.session.draft.workspace_base = str(self.task_workspace / "runs")
            overrides = research_model_overrides(constraints)
            for name, value in overrides.items():
                setattr(self.session.draft, name, value)
            if "model_selection" not in overrides:
                self.session.draft.model_selection = "auto"
            configure_draft_models(self.session.draft, self._model_config_path)
            self._research_models_confirmed = bool(overrides)
            self._awaiting_research_models = False
            if "code_models" not in overrides:
                # Automatic defaults choose one task-wide model. All workers
                # keep that model name while its endpoint pool handles failover.
                self.session.draft.code_models = self.session.draft.code_models[:1]
        except (ValueError, OSError) as exc:
            await host.notice(f"Invalid /long-research options: {exc}")
            self.session = None
            self._research_models_confirmed = False
            self._awaiting_research_models = False
            self._host = None
            host.set_host_status("")
            return
        self.files = None
        self.preflight = None
        self._set_status(host)
        if self.session.draft.description_sources:
            from .research.control.notices import research_notice

            await research_notice(
                host,
                "Loaded task description: "
                + ", ".join(self.session.draft.description_sources),
            )
        await self._continue_onboarding(host, constraints)

    async def _continue_onboarding(
        self,
        host: TuiHostPort,
        preparation_answer: str = "",
    ) -> None:
        session = self._require_session()
        if (
            session.state != OnboardingState.PREFLIGHT
            and await self._preparation.begin(host, preparation_answer)
        ):
            return
        await host.notice(
            await self._decorate_prompt(
                session.next_prompt(), session.next_question()
            )
        )
        if session.state == OnboardingState.PREFLIGHT:
            await self._preflight(host)

    async def _preflight(self, host: TuiHostPort) -> None:
        session = self._require_session()
        if not self._research_models_confirmed:
            from scienceflow.interfaces.ui.model_config import (
                model_config_setup_hint,
                research_model_question,
            )

            self._awaiting_research_models = True
            session.state = OnboardingState.ASKING
            try:
                question = research_model_question(
                    self._model_config_path,
                    session.draft,
                )
            except (OSError, ValueError) as exc:
                question = (
                    f"Model configuration unavailable: {exc}\n"
                    f"{model_config_setup_hint()}"
                )
            from .research.control.notices import research_notice

            await research_notice(host, question)
            self._set_status(host)
            return
        host.set_host_status("long research · preflight")
        try:
            from scienceflow.interfaces.ui.model_config import configure_draft_models

            configure_draft_models(session.draft, self._model_config_path)
            self.files = await asyncio.to_thread(
                write_onboarding_files,
                session.draft,
                self.task_workspace,
                conversation=session.conversation,
            )
            self.preflight = await asyncio.to_thread(
                run_preflight, self.files.manifest_path
            )
            update = session.apply_preflight(self.preflight)
        except Exception as exc:
            await host.notice(f"Preflight failed: {type(exc).__name__}: {exc}")
            self._set_status(host)
            return

        from .research.control.notices import research_notice

        await research_notice(host,
            f"{_preflight_summary(self.preflight)}\n"
            f"Models: code={','.join(session.draft.code_models)} · "
            f"feedback={','.join(session.draft.feedback_models)} · "
            f"policy={session.draft.model_selection or 'auto'}\n"
            f"{update.message}"
        )
        self._set_status(host)

    async def _apply_research_model_answer(
        self,
        host: TuiHostPort,
        answer: str,
    ) -> None:
        from scienceflow.interfaces.ui.model_config import (
            apply_research_model_choice,
            model_config_setup_hint,
            research_model_question,
        )

        session = self._require_session()
        try:
            apply_research_model_choice(
                self._model_config_path,
                session.draft,
                answer,
            )
        except (OSError, ValueError) as exc:
            try:
                question = research_model_question(
                    self._model_config_path,
                    session.draft,
                )
            except (OSError, ValueError):
                question = model_config_setup_hint()
            from .research.control.notices import research_notice

            await research_notice(
                host,
                f"模型配置未通过：{exc}\n"
                f"{question}"
            )
            return
        self._awaiting_research_models = False
        self._research_models_confirmed = True
        session.state = OnboardingState.PREFLIGHT
        await self._preflight(host)

    async def _run(self, host: TuiHostPort) -> None:
        files = self.files
        if files is None:
            await host.notice("Long research has no validated manifest.")
            return
        from scienceflow.interfaces.ui.long_research_progress import (
            LongResearchProgress,
        )

        progress = LongResearchProgress(self._require_session().draft)
        self._progress = progress
        await host.notice(
            f"Running in worker subprocesses (no tmux). Time budget: {progress.budget // 60} min.\n"
            f"Logs: {progress.root / 'task_logs'}\n"
            "The bar tracks elapsed time, not research completion. /cancel stops the run."
        )
        monitor = asyncio.create_task(progress.watch(host))
        outcome = "Finished"
        try:
            summary = await run_manifest(files.manifest_path)
        except asyncio.CancelledError:
            outcome = "Cancelled"
            await host.notice("Long research cancelled; resumable state was preserved.")
            raise
        except Exception as exc:
            outcome = "Failed"
            await host.notice(f"Long research failed: {type(exc).__name__}: {exc}")
        else:
            outcome = "Finished" if all(r.status in {"success", "skipped"} for r in summary.results) else "Finished with failures"
            await host.notice(_format_run_summary(summary))
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            progress.active = False
            progress.final_state = 'completed' if outcome == 'Finished' else ('stopped' if outcome == 'Cancelled' else 'failed')
            progress.final_elapsed = max(0, time.monotonic() - progress.started)
            await asyncio.to_thread(progress.snapshot)
            if hasattr(host, "set_host_panel"):
                host.set_host_panel(f"{outcome} · {progress.summary}", progress.rows)
            self._progress = None
            self._run_task = None
            self.session = None
            self._research_models_confirmed = False
            self._awaiting_research_models = False
            self.files = None
            self.preflight = None
            host.set_host_status("")

    async def _cancel(self, host: TuiHostPort) -> None:
        if self._managed and (self._managed.run_id or self._managed.launch_task):
            await self._managed.stop(host)
            return
        await self._preparation.close()
        task = self._run_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            self._require_session().cancel()
            await host.notice("Long research onboarding cancelled; returning to chat.")
        self._run_task = None
        self.session = None
        self._research_models_confirmed = False
        self._awaiting_research_models = False
        self.files = None
        self.preflight = None
        if callable(getattr(self._managed, "cancel_preparation", None)):
            await self._managed.cancel_preparation()
        host.set_host_status("")

    def _status_text(self) -> str:
        if self.running:
            progress = self._progress
            return (progress.summary + "\n" + "\n".join(progress.rows) + f"\nLogs: {progress.root}" + "\n" if progress else "") + (
                "Long research is running. Use /cancel to stop and preserve resumable state."
            )
        if self._awaiting_research_models:
            return (
                "Long research onboarding: asking. "
                "Choose research models or enter defaults."
            )
        session = self._require_session()
        status = (
            f"Long research onboarding: {session.state.value}. {session.next_prompt()}"
        )
        return status

    def _set_status(self, host: TuiHostPort) -> None:
        session = self.session
        if session is None:
            host.set_host_status("")
            return
        host.set_host_status(f"long research · {session.state.value}")

    async def _decorate_prompt(self, message: str, question: object | None) -> str:
        if getattr(question, "field", "") != "gpu_list":
            return message
        devices = await asyncio.to_thread(query_gpu_devices)
        if not devices:
            return f"{message}\nNo NVIDIA GPU inventory is currently visible; CPU remains available."
        rows = ", ".join(
            f"GPU {device.index} ({device.memory_free_mib} / "
            f"{device.memory_total_mib} MiB free)"
            for device in devices
        )
        return f"{message}\nDetected: {rows}."

    def _require_session(self) -> LongResearchSession:
        if self.session is None:
            raise RuntimeError("long research session is not active")
        return self.session


async def _echo_user(host, text: str) -> None:
    echo = getattr(host, "echo_user", None)
    if callable(echo):
        await echo(text)


def _format_run_summary(summary: ParallelRunSummary) -> str:
    failed = sum(
        result.status not in {"success", "skipped"} for result in summary.results
    )
    return (
        f"Long research finished: {len(summary.results) - failed} successful, {failed} failed.\n"
        f"{summary.text}"
    )


__all__ = ["LongResearchInteraction", "long_research_argument"]
