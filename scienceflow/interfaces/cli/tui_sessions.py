"""Workspace-isolated TUI sessions with explicit history restoration."""

import errno
import os
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from functools import wraps
from inspect import unwrap
from pathlib import Path
from uuid import uuid4

import click


class FileInUseError(RuntimeError):
    """Another process owns a session writer lock."""


@contextmanager
def exclusive_file_lock(path: Path):
    """Hold a non-blocking writer lock without replacing the lock-file inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, 'r+b') as handle:
        if os.name == 'nt':
            import msvcrt

            if not os.fstat(handle.fileno()).st_size:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise FileInUseError(str(path)) from exc
                raise
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise FileInUseError(str(path)) from exc
                raise
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _updated_at(session: Path) -> float:
    companions = (
        session,
        session.with_name('events.jsonl'),
        session.with_name('operations.jsonl'),
        Path(f'{session}.usage.jsonl'),
    )
    timestamps = []
    for path in companions:
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(timestamps, default=-1.0)


def configure_sessions(command):
    filenames = {'session_log': 'session.jsonl', 'event_log': 'events.jsonl',
                 'operation_log': 'operations.jsonl'}
    supported = {param.name for param in command.params}
    for param in command.params:
        if param.name in filenames or param.name == 'workspace':
            param.envvar = None
            param.allow_from_autoenv = False
            param.scienceflow_explicit_only = True
            if param.name == 'workspace':
                param.default = Path.cwd
                param.show_default = "current directory"
            else:
                param.default = None
    host_sessions = 'session' in supported
    if host_sessions:
        for param in command.params:
            if param.name == 'resume':
                param.help = 'Resume the latest saved session in this workspace.'
            elif param.name == 'session':
                param.help = 'Saved session ID or unique prefix; requires --resume.'
    if not any(param.name == 'resume' for param in command.params):
        command.params.append(click.Option(
            ['--resume'], is_flag=True, default=False,
            help='Resume the latest saved session in this workspace; use --session-log to select one.',
        ))
    original = command.callback
    if host_sessions:
        # ScienceFlow selects the latest session and owns its writer locks.
        # Skip InquiryCraft's interactive selector when it is present.
        original = unwrap(original)

    @wraps(original)
    def callback(*args, **kwargs):
        resume = kwargs.pop('resume', False)
        identifier = kwargs.pop('session', None)
        if identifier and not resume:
            raise click.ClickException('--session requires --resume.')
        workspace = Path(kwargs.get('workspace') or Path.cwd()).expanduser().resolve()
        if 'workspace' in supported:
            kwargs['workspace'] = workspace
        root = workspace / '.scienceflow' / 'sessions'
        explicit = kwargs.get('session_log')
        if explicit:
            session = Path(explicit).expanduser().resolve()
            if resume and not session.is_file():
                raise click.ClickException('Requested session does not exist.')
            if not resume and session.exists():
                raise click.ClickException('Session file already exists; use --resume or choose a new --session-log.')
            directory = session.parent
        elif resume:
            candidates = list(root.glob('*/session.jsonl'))
            if not candidates:
                raise click.ClickException('No saved session in this workspace. Start without --resume.')
            if identifier:
                matches = [path for path in candidates if path.parent.name.startswith(identifier)]
                if len(matches) != 1:
                    raise click.ClickException(
                        f'Session {identifier!r} matches {len(matches)} saved sessions in this workspace.'
                    )
                session = matches[0]
            else:
                session = max(candidates, key=lambda path: (_updated_at(path), path.parent.name))
            directory = session.parent
        else:
            session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
            directory = root / session_id
            directory.mkdir(parents=True, mode=0o700)
            session = directory / 'session.jsonl'
        for name, filename in filenames.items():
            if (host_sessions or name in supported) and not kwargs.get(name):
                kwargs[name] = session if name == 'session_log' else directory / filename
        if host_sessions:
            kwargs['resumed_session'] = resume
        paths = sorted({Path(value).expanduser().resolve()
                        for name in filenames if (value := kwargs.get(name))})
        with ExitStack() as locks:
            try:
                for path in paths:
                    locks.enter_context(exclusive_file_lock(
                        path.with_name(path.name + '.writer.lock')
                    ))
            except FileInUseError as exc:
                raise click.ClickException(
                    f'Session is already open: {session}. Close the other writer or choose another session.'
                ) from exc
            return original(*args, **kwargs)

    command.callback = callback
    return command
