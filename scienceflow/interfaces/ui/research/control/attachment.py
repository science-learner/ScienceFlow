"""Attach/detach a view without owning the experiment process."""

from __future__ import annotations

import asyncio
import shlex
import time
from dataclasses import fields
from pathlib import Path

from scienceflow.research.onboarding import (
    LongResearchDraft,
    LongResearchSession,
    OnboardingState,
)
from scienceflow.runtime.parallel.control import (
    get_run,
    resume_run,
    start_run,
    stop_run,
)
from scienceflow.runtime.parallel.control.selection import (
    AmbiguousRun,
    run_label,
    select_workspace_run,
    workspace_runs,
)
from scienceflow.runtime.parallel.control.service import WorkspaceBusy

from ..long_research_progress import LongResearchProgress


class ManagedResearch:
    def __init__(self, owner):
        self.owner = owner
        self.run_id = None
        self.launch_task = None
        self.pending_choice = None

    async def on_mount(self, host):
        rows = await asyncio.to_thread(workspace_runs, self.owner.workspace)
        active = [row for row in rows if row['alive']]
        if len(active) == 1:
            await self.attach(active[0], host)
        elif active:
            await self.offer_choice('attach', active, host)
        elif any(row['status'] != 'completed' for row in rows):
            await host.notice('Saved research is available in this workspace. Use /resume to continue.')

    async def offer_choice(self, command, rows, host):
        from datetime import datetime
        self.pending_choice = (command, rows)
        lines = [f"{index}. {run_label(row)} · {row['status']} · "
                 + datetime.fromtimestamp(row.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')
                 for index, row in enumerate(rows, 1)]
        await host.notice('Choose a task:\n' + '\n'.join(lines)
                          + '\nUse /select NUMBER, or /select cancel.')

    async def command(self, text, host) -> bool:
        head = text.strip().split(maxsplit=1)
        command = head[0].lower().lstrip('/') if head else ''
        if command == 'run':
            if self.run_id and self.owner.running:
                await host.notice('Research is already running in this workspace. Showing its progress; /stop stops it.')
                return True
            return False
        if command not in {'stop', 'resume', 'attach', 'runs', 'status', 'select'}:
            return False
        if command == 'status' and not self.run_id and self.owner.session is not None:
            return False
        try:
            words = shlex.split(text)
            if len(words) > 2:
                raise ValueError(f'Use /{command} [workspace path].')
            echo = getattr(host, 'echo_user', None)
            if echo:
                await echo(text)
            if command == 'select':
                if not self.pending_choice or len(words) != 2:
                    raise ValueError('No selection is pending. Use /resume or /attach first.')
                if words[1].lower() == 'cancel':
                    self.pending_choice = None
                    await host.notice('Selection cancelled.')
                    return True
                index = int(words[1]) - 1
                command, rows = self.pending_choice
                if not 0 <= index < len(rows):
                    raise ValueError('Choose a number from the displayed list.')
                row = await asyncio.to_thread(get_run, rows[index]['run_id'])
                self.pending_choice = None
            elif command == 'runs':
                rows = await asyncio.to_thread(workspace_runs, self.owner.workspace)
                await host.notice('\n'.join(f"{run_label(row)} · {row['status']} · "
                                            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(row.get('created_at', 0)))}"
                                            for row in rows) or 'No research history in this workspace.')
                return True
            elif len(words) == 1 and self.run_id:
                row = await asyncio.to_thread(get_run, self.run_id)
            elif command == 'stop' and len(words) == 1 and self.launch_task is not None:
                row = await asyncio.shield(self.launch_task)
            else:
                row = await asyncio.to_thread(select_workspace_run, command,
                                              words[1] if len(words) == 2 else None, cwd=self.owner.workspace)
            await self.perform(command, row, host)
        except AmbiguousRun as exc:
            await self.offer_choice(command, exc.candidates, host)
        except (ValueError, OSError) as exc:
            await host.notice(str(exc))
        return True

    async def perform(self, command, row, host):
        if command == 'status':
            await host.notice(f"{run_label(row)} · {row['status']}\nLogs: {row['log']}")
        elif command == 'stop':
            await host.notice(f'Stopping {run_label(row)}; preserving completed work…')
            row = await asyncio.to_thread(stop_run, row['run_id'])
            await host.notice(f"{run_label(row)} · {row['status']}. Use /resume to continue.")
            if row['run_id'] == self.run_id and self.owner._run_task:
                self.owner._run_task.cancel()
                await asyncio.gather(self.owner._run_task, return_exceptions=True)
        else:
            if self.run_id == row['run_id'] and self.owner.running:
                await host.notice('Already showing this task. /stop stops it.')
                return
            if self.owner.session is not None or host.busy:
                raise ValueError('Finish or cancel current preparation before switching tasks.')
            if command == 'resume' and not row['alive']:
                row = await asyncio.to_thread(resume_run, row['run_id'])
            await self.attach(row, host)

    async def launch(self, host):
        try:
            self.launch_task = asyncio.create_task(asyncio.to_thread(
                start_run, self.owner.files.manifest_path, draft=self.owner.session.draft))
            row = await asyncio.shield(self.launch_task)
            await self._watch(row, host)
        except WorkspaceBusy as exc:
            row = await asyncio.to_thread(get_run, exc.run_id)
            if row['alive']:
                await host.notice(
                    '当前目录已有任务运行，正在接回其进度。本次新配置未启动，也未覆盖已有任务。'
                    '\n使用 /stop 停止已有任务；独立实验请使用另一个 workspace。')
                self._restore_draft(row)
                await self._watch(row, host)
            else:
                await host.notice(f"已有任务刚刚结束：{row['status']}。可重新准备并启动新实验。")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await host.notice(f'Could not launch managed research: {type(exc).__name__}: {exc}')
        finally:
            self._clear()
            host.set_host_status('')

    async def attach(self, row, host):
        if not row['alive']:
            await host.notice(f"{run_label(row)} · {row['status']}. Use /resume to continue.")
            return
        self.run_id = row['run_id']
        self._restore_draft(row)
        self.owner._run_task = asyncio.create_task(self._attached_watch(row, host))

    def _restore_draft(self, row):
        draft = row.get('draft')
        if draft:
            allowed = {f.name for f in fields(LongResearchDraft)}
            self.owner.session = LongResearchSession(LongResearchDraft(**{k: v for k, v in draft.items() if k in allowed}))
            self.owner.session.state = OnboardingState.RUNNING

    async def _attached_watch(self, row, host):
        try:
            await self._watch(row, host)
        finally:
            self._clear()
            host.set_host_status('')

    async def _watch(self, row, host):
        self.run_id = row['run_id']
        self.launch_task = None
        progress = LongResearchProgress(self.owner.session.draft) if self.owner.session else None
        if progress:
            progress.root = Path(row['task_roots'][0])
            progress.budget = max(1, row.get('display_budget_sec', progress.budget))
            progress.started -= max(0, time.time() - row['started_at'])
        self.owner._progress = progress
        await host.notice(f"Attached: {run_label(row)}. Exiting TUI leaves research running. /stop stops it.\nLogs: {row['log']}")
        monitor = asyncio.create_task(progress.watch(host)) if progress else None
        host.set_host_status(f"long research · {run_label(row)}")
        try:
            while row['alive']:
                await asyncio.sleep(2)
                row = await asyncio.to_thread(get_run, self.run_id)
            await host.notice(f"{run_label(row)} · {row['status']}. " + ("Use /long-research for a new task." if row['status'] == 'completed' else "Use /resume to continue."))
        finally:
            if monitor:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            if progress and not row['alive']:
                progress.active = False
                progress.final_state = row['status']
                progress.final_elapsed = max(0, (row.get('finished_at') or row['started_at']) - row['started_at'])
                await asyncio.to_thread(progress.snapshot)
                if getattr(host, 'set_host_panel', None):
                    host.set_host_panel(progress.summary, progress.rows)

    async def stop(self, host):
        run_id = self.run_id
        if run_id is None and self.launch_task is not None:
            run_id = (await asyncio.shield(self.launch_task))['run_id']
        if run_id:
            row = await asyncio.to_thread(stop_run, run_id)
            await host.notice(f"{run_label(row)} · {row['status']}. Use /resume to continue.")
            task = self.owner._run_task
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def _clear(self):
        self.run_id = None
        self.owner._run_task = None
        self.owner._progress = None
        self.owner.session = None
        self.owner.files = None
        self.owner.preflight = None
