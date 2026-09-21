"""Persistent task numbers and isolated directories, serialized across TUI clients."""

from collections.abc import Mapping
from pathlib import Path

from scienceflow.research.onboarding import parse_constraints
from scienceflow.runtime.parallel.control.registry import (
    locked,
    read_record,
    write_record,
)
from scienceflow.runtime.parallel.control.selection import workspace_aliases


def _is_path_name(value: str) -> bool:
    """Recognize task addresses without treating ordinary names containing '/' as paths."""
    text = str(value or '').strip()
    if not text:
        return False
    path = Path(text).expanduser()
    return (
        path.is_absolute()
        or text.startswith(('./', '../', '~/', '.\\', '..\\', '~\\'))
        or '/task_library/' in text
        or '\\task_library\\' in text
    )


def _package_name(value: str, workspace: Path) -> str:
    try:
        from scienceflow.research.onboarding.library.catalog import resolve_local

        package = resolve_local(value, workspace)
    except (OSError, ValueError):
        return ''
    if package is None:
        return ''
    for field in ('name', 'title'):
        candidate = str(package.config.get(field) or '').strip()
        if candidate:
            return candidate[:80]
    return package.task_id[:80]


def _task_text_from_constraints(value: str) -> str:
    """Remove inline runtime options from a task-board display name."""
    source = str(value or '').strip()
    try:
        task_text = str(parse_constraints(source).get('task_text') or '').strip()
    except ValueError:
        return ''
    return task_text[:80] if task_text and task_text != source else ''


def _run_name(row: Mapping[str, object]) -> str:
    draft = row.get('draft')
    if not isinstance(draft, Mapping):
        draft = {}
    for field in ('display_name', 'task_name', 'exp_id'):
        candidate = str(draft.get(field) or '').strip()
        if candidate and not _is_path_name(candidate):
            return candidate[:80]
    return ''


def preferred_task_name(
    current: str,
    row: Mapping[str, object],
    workspace: Path,
) -> str:
    """Repair address-shaped names while preserving deliberate human task names."""
    current = str(current or '').strip()
    constraint_name = _task_text_from_constraints(current)
    if constraint_name:
        return _run_name(row) or constraint_name
    if current and current != 'New task' and not _is_path_name(current):
        return current[:80]
    return _run_name(row) or _package_name(current, workspace) or 'Research'


class TaskIndex:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.directory = self.workspace / '.scienceflow/task-board'

    def _read(self):
        if not (self.directory / 'control.json').exists():
            return []
        entries = read_record(self.directory)['tasks']
        if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
            raise ValueError('Invalid task index; restore it before managing tasks.')
        return entries

    def snapshot(self):
        """Read the task index without creating directories or syncing runs."""
        return self._read()

    def reserve(self, name):
        with locked(self.directory):
            entries = self._read()
            number = max((e['number'] for e in entries), default=0) + 1
            workspace = self.workspace / 'tasks' / f'task-{number:04d}'
            if not workspace.resolve().is_relative_to(self.workspace):
                raise ValueError('Task directory must remain inside this workspace.')
            workspace.mkdir(parents=True, exist_ok=False)
            supplied_name = str(name or '').strip()
            display_name = (
                _package_name(supplied_name, self.workspace)
                or _task_text_from_constraints(supplied_name)
                or supplied_name[:80]
                or 'New task'
            )
            entry = {'number': number, 'name': display_name, 'workspace': str(workspace),
                         'run_id': None, 'status': 'preparing'}
            entries.append(entry)
            write_record(self.directory, {'tasks': entries})
            return entry

    def update(self, number, **changes):
        with locked(self.directory):
            entries = self._read()
            entry = next(e for e in entries if e['number'] == number)
            entry.update(changes)
            write_record(self.directory, {'tasks': entries})

    def sync(self, rows):
        with locked(self.directory):
            entries = self._read()
            before = repr(entries)
            known = {e['run_id'] for e in entries if e.get('run_id')}
            indexed = {e['run_id']: e for e in entries if e.get('run_id')}
            for row in sorted(rows, key=lambda r: r.get('created_at', 0)):
                if row['run_id'] in known:
                    existing = indexed[row['run_id']]
                    existing['status'] = row['status']
                    existing['name'] = preferred_task_name(
                        existing.get('name', ''), row, self.workspace
                    )
                    continue
                aliases = workspace_aliases(row)
                reserved = next((e for e in entries if not e.get('run_id')
                                 and Path(e['workspace']).resolve() in aliases), None)
                if reserved is not None:
                    reserved.update(
                        run_id=row['run_id'],
                        status=row['status'],
                        name=preferred_task_name(
                            reserved.get('name', ''), row, self.workspace
                        ),
                    )
                elif self.workspace in aliases:
                    entries.append({'number': max((e['number'] for e in entries), default=0) + 1,
                                        'name': preferred_task_name('', row, self.workspace),
                                        'workspace': row.get('control_workspace') or str(self.workspace),
                                        'run_id': row['run_id'], 'status': row['status']})
                else:
                    continue
                known.add(row['run_id'])
            if repr(entries) != before:
                write_record(self.directory, {'tasks': entries})
            return entries

    def attach_existing(self, row):
        """Explicitly import an existing run without moving its files or restarting it."""
        with locked(self.directory):
            entries = self._read()
            found = next((e for e in entries if e.get('run_id') == row['run_id']), None)
            if found is not None:
                return found
            entry = {'number': max((e['number'] for e in entries), default=0) + 1,
                         'name': preferred_task_name('', row, self.workspace),
                         'workspace': row.get('control_workspace') or row.get('cwd') or str(self.workspace),
                         'run_id': row['run_id'], 'status': row['status']}
            entries.append(entry)
            write_record(self.directory, {'tasks': entries})
            return entry
