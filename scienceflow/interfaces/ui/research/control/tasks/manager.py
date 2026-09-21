"""One preparation conversation, many independently supervised research tasks."""

import asyncio
import shlex
import time
from pathlib import Path

from scienceflow.research.onboarding import OnboardingState
from scienceflow.runtime.parallel.control import (
    get_run,
    list_runs,
    resume_run,
    start_run,
    stop_run,
)
from scienceflow.runtime.parallel.control.selection import resolve_run

from ..notices import research_notice
from .projection import TaskProjection, friendly_status
from .records import TaskIndex, preferred_task_name

_ACTIVE_STATES = frozenset({'starting', 'running', 'stopping'})
_RUNNING_STATES = frozenset({'starting', 'running'})
_DONE_STATES = frozenset({'completed', 'failed', 'stopped', 'interrupted'})
_POLL_INTERVAL_SEC = 4.0


def _task_subject(numbers):
    label = ', '.join(str(number) for number in numbers)
    return f'Task {label}' if len(numbers) == 1 else f'Tasks {label}'


def _registered_summary(registrations):
    numbers = [number for number, _ in registrations]
    packages = list(dict.fromkeys(
        Path(str(path or '')).name or 'task package' for _, path in registrations
    ))
    if len(packages) == 1:
        package_summary = packages[0]
    else:
        visible = ', '.join(packages[:3])
        hidden = f' +{len(packages) - 3}' if len(packages) > 3 else ''
        package_summary = f'{len(packages)} packages · {visible}{hidden}'
    return f'Task library · {_task_subject(numbers)} registered · {package_summary}'


def _deferred_summary(registrations):
    numbers = [number for number, _ in registrations]
    reasons = list(dict.fromkeys(
        ' '.join(str(reason).split()) for _, reason in registrations if reason
    ))
    if not reasons:
        detail = ''
    elif len(reasons) == 1:
        detail = reasons[0]
    else:
        detail = f'{len(reasons)} reasons'
    if len(detail) > 96:
        detail = detail[:93].rstrip() + '…'
    suffix = f' · {detail}' if detail else ''
    return f'Task library · {_task_subject(numbers)} saved · registration deferred{suffix}'


class MultiResearch:
    def __init__(self, owner):
        self.owner = owner
        self.index = TaskIndex(owner.workspace)
        self.projection = TaskProjection()
        self.preparing = None
        self.poll_task = None
        self.launch_task = None
        self.run_id = None  # No single run owns this conversation.
        self.entries = []
        self.rows = {}
        self.items = ()
        self.task_catalog = ()
        self.catalog_at = None
        self.registrations_seen = set()
        self.current_task_numbers = set()
        self.error = ''
        self.refresh_lock = asyncio.Lock()
        self._registry_rows = {}
        self._registry_loaded = False

    async def on_mount(self, host):
        await self.refresh(host)
        self.ensure_poll(host)

    def ensure_poll(self, host):
        if self.poll_task is None or self.poll_task.done():
            self.poll_task = asyncio.create_task(self.watch(host))

    async def new_workspace(self, name, host):
        self.preparing = await asyncio.to_thread(self.index.reserve, name)
        self.current_task_numbers.add(self.preparing['number'])
        self.ensure_poll(host)
        await research_notice(
            host,
            f"Task {self.preparing['number']} · {self.preparing['workspace']}",
        )
        return self.preparing['workspace']

    def _remember_row(self, row):
        run_id = row.get('run_id')
        if run_id:
            self._registry_rows[run_id] = row

    def _load_rows(self):
        """Load history once, then refresh only records which can still change."""
        if not self._registry_loaded:
            rows = list_runs()
            self._registry_rows = {row['run_id']: row for row in rows}
            self._registry_loaded = True
            return rows
        for run_id, cached in tuple(self._registry_rows.items()):
            if cached.get('alive') or cached.get('status') in _ACTIVE_STATES:
                try:
                    self._registry_rows[run_id] = get_run(run_id)
                except (OSError, ValueError, KeyError):
                    continue
        return sorted(
            self._registry_rows.values(),
            key=lambda row: row.get('created_at', 0),
            reverse=True,
        )

    def collect(self):
        rows = self._load_rows()
        if self.catalog_at is None:
            from scienceflow.research.onboarding.library.catalog import task_choices
            self.task_catalog = task_choices(self.owner.workspace)
            self.catalog_at = time.monotonic()
        entries = self.index.sync(rows)
        by_id = {row['run_id']: row for row in rows}
        items = tuple(
            self.projection.item(
                entry,
                by_id.get(entry.get('run_id')),
                historical=entry['number'] not in self.current_task_numbers,
            )
            for entry in sorted(entries, key=lambda entry: entry['number'], reverse=True)
        )
        return entries, by_id, items

    async def refresh(self, host):
        async with self.refresh_lock:
            self.entries, self.rows, self.items = await asyncio.to_thread(self.collect)
        registered = []
        deferred = []
        for entry in self.entries:
            row = self.rows.get(entry.get('run_id'), {})
            registration = row.get('task_registration')
            if registration and row['run_id'] not in self.registrations_seen:
                self.registrations_seen.add(row['run_id'])
                if registration['status'] == 'registered':
                    # Reload completion choices after the task library changes.
                    self.catalog_at = None
                    registered.append((entry['number'], registration.get('path', '')))
                elif registration['status'] == 'deferred':
                    deferred.append((entry['number'], registration.get('reason', '')))
        if registered:
            await host.notice(_registered_summary(registered))
        if deferred:
            await host.notice(_deferred_summary(deferred))
        running = sum(item['state'] in _RUNNING_STATES for item in self.items)
        done = sum(item['state'] in _DONE_STATES for item in self.items)
        total = len(self.items)
        title = f'Tasks · {done}/{total} done · {running} running'
        costs = [item.get('cost', (0, 0, 0)) for item in self.items]
        if any(cost[1] for cost in costs):
            from scienceflow.runtime.observability.telemetry.agent.llm_cost import (
                format_usd,
            )
            title += ' · Cost ~' + format_usd(sum(cost[0] for cost in costs))
        else:
            title += ' · Cost —'
        setter = getattr(host, 'set_host_tasks', None)
        if setter:
            setter(title, self.items)
        elif getattr(host, 'set_host_panel', None):
            host.set_host_panel(title, tuple(item['text'] for item in self.items))

    async def watch(self, host):
        while True:
            try:
                active = any(
                    row.get('alive') or row.get('status') in _ACTIVE_STATES
                    for row in self._registry_rows.values()
                )
                if (
                    not self._registry_loaded
                    or active
                    or self.preparing is not None
                    or self.catalog_at is None
                ):
                    await self.refresh(host)
                self.error = ''
            except (OSError, ValueError, KeyError) as exc:
                if str(exc) != self.error:
                    await host.notice(f'Task monitor unavailable: {exc}')
                    self.error = str(exc)
            await asyncio.sleep(_POLL_INTERVAL_SEC)

    def resource_summary(self, host_summary):
        from ...resources import allocation_summary

        return allocation_summary(host_summary, self.entries, self.rows)

    def resource_preview(self, draft):
        from ...resources import allocation_preview

        return allocation_preview(self.entries, self.rows, self.preparing, draft)

    async def launch(self, host):
        session, files, entry = self.owner.session, self.owner.files, self.preparing
        if session is None or files is None or entry is None:
            await host.notice('Prepare a task before run.')
            return
        try:
            self.launch_task = asyncio.create_task(asyncio.to_thread(
                start_run, files.manifest_path, draft=session.draft, check_resources=True, library_workspace=self.owner.workspace))
            row = await asyncio.shield(self.launch_task)
            self._remember_row(row)
            name = preferred_task_name(
                entry['name'], row, self.owner.workspace
            )
            await asyncio.to_thread(
                self.index.update,
                entry['number'],
                run_id=row['run_id'],
                status=row['status'],
                name=name,
            )
            from ...resources import task_allocation_line

            allocation = task_allocation_line(
                f"Task {entry['number']}", name, row
            )
            await research_notice(
                host,
                f"{allocation} · started\nWorkspace · {entry['workspace']}\n"
                f"/stop {entry['number']} stops only this task. "
                "/long-research prepares another task."
            )
            self.owner.session = self.owner.files = self.owner.preflight = None
            self.owner._task_workspace = None
            self.preparing = None
            await self.owner._preparation.close()
            self.ensure_poll(host)
            await self.refresh(host)
        except asyncio.CancelledError:
            raise  # Detached start completes; the reserved directory lets the next TUI recover it.
        except (OSError, ValueError) as exc:
            session.state = OnboardingState.CONFIRM
            await host.notice(f'Could not start task {entry["number"]}: {exc}\n'
                              'Edit the resource configuration or type run to retry. Existing tasks continue.')
        finally:
            self.launch_task = None
            self.owner._run_task = None
            host.set_host_status('')

    async def command(self, text, host):
        head = text.strip().split(maxsplit=1)
        command = head[0].lower().lstrip('/') if head else ''
        if command not in {'stop', 'resume', 'attach', 'runs', 'tasks', 'status', 'run', 'research-usage', 'select'}:
            return False
        if command == 'run':
            if self.owner.session is not None:
                return False
            await host.notice('Use /long-research to prepare a new task. Existing tasks continue running.')
            return True
        if command == 'status' and self.owner.session is not None and len(head) == 1:
            return False
        try:
            words = shlex.split(text)
            if len(words) > 2:
                raise ValueError(f'Use /{command} [task number].')
            if callable(getattr(host, 'echo_user', None)):
                await host.echo_user(text)
            if command in {'attach', 'runs', 'tasks'}:
                self._registry_loaded = False
            await self.refresh(host)
            self.ensure_poll(host)
            if command in {'tasks', 'runs'} or (command == 'status' and len(words) == 1):
                await host.notice('\n'.join(item['text'] for item in self.items) or 'No tasks yet. Use /long-research.')
                return True
            entry = self.select(command, words[1] if len(words) == 2 else None)
            row = await asyncio.to_thread(get_run, entry['run_id'])
            if command == 'stop':
                row = await asyncio.to_thread(stop_run, row['run_id'], timeout=0)
                await host.notice(
                    f"Stop requested · Task {entry['number']} will preserve completed work."
                )
            elif command == 'resume' and not row['alive']:
                row = await asyncio.to_thread(resume_run, row['run_id'], check_resources=True)
            self._remember_row(row)
            if command == 'research-usage':
                monitor = self.projection.monitors[(row['run_id'], row.get('attempt', 0))]
                await host.notice(monitor.usage.details())
            elif command == 'status':
                item = self.projection.item(entry, row)
                resources = '\n'.join(item.get('worker_details', {}).values())
                await host.notice(item['detail'] + (f'\n\nWorker resources (GPU device readings):\n{resources}' if resources else '') + f"\nLogs: {row['log']}")
            elif command != 'stop':
                await host.notice(f"Task {entry['number']} · {friendly_status(row['status'])} · {entry['workspace']}")
            await self.refresh(host)
            if command in {'attach', 'select'} and callable(getattr(host, 'expand_host_task', None)):
                host.expand_host_task(str(entry['number']))
        except (OSError, ValueError, KeyError) as exc:
            await host.notice(str(exc))
        return True

    def select(self, command, target):
        candidates = [e for e in self.entries if e.get('run_id') in self.rows]
        if target:
            candidates = [e for e in candidates if target in {str(e['number']), e['run_id'], e['workspace']}]
            if not candidates and not target.isdigit():
                row = resolve_run(target, cwd=self.owner.workspace)
                entry = self.index.attach_existing(row)
                self.rows[row['run_id']] = row
                self._remember_row(row)
                self.entries.append(entry)
                return entry
        elif command == 'stop':
            candidates = [e for e in candidates if self.rows[e['run_id']]['alive']]
        elif command == 'resume':
            candidates = [e for e in candidates if self.rows[e['run_id']]['status'] != 'completed']
        if not candidates:
            raise ValueError('No matching task in this workspace. Use /tasks to list task numbers.')
        if len(candidates) != 1:
            raise ValueError(f'Multiple tasks match. Use /{command} NUMBER:\n' + '\n'.join(
                f"{e['number']} · {e['name']} · {self.rows[e['run_id']]['status']}" for e in candidates))
        return candidates[0]

    async def stop(self, host):
        await host.notice('Task launch is in progress. Use /stop NUMBER once it appears in /tasks.')

    async def cancel_preparation(self):
        if self.preparing is not None:
            await asyncio.to_thread(self.index.update, self.preparing['number'], status='cancelled')
            self.preparing = None
        self.owner._task_workspace = None

    async def close(self):
        if self.preparing is not None and self.launch_task is None:
            await asyncio.to_thread(self.index.update, self.preparing['number'], status='preparation interrupted')
        if self.poll_task:
            self.poll_task.cancel()
            await asyncio.gather(self.poll_task, return_exceptions=True)
            self.poll_task = None
