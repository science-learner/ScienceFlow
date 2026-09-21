"""Workspace-isolated TUI sessions with explicit history restoration."""

from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import click


def configure_sessions(command, state_root):
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
    command.params.append(click.Option(['--resume'], is_flag=True, default=False,
                                      help='Resume the latest saved session in this workspace; use --session-log to select one.'))
    original = command.callback

    @wraps(original)
    def callback(*args, **kwargs):
        resume = kwargs.pop('resume', False)
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
            candidates = sorted(root.glob('*/session.jsonl'))
            if not candidates:
                workspace_id = sha256(str(workspace).encode()).hexdigest()[:20]
                legacy_root = state_root() / 'workspaces' / workspace_id
                candidates = sorted(legacy_root.glob('*/session.jsonl'))
            if not candidates:
                raise click.ClickException('No saved session in this workspace. Start without --resume.')
            session = candidates[-1]
            directory = session.parent
        else:
            session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]
            directory = root / session_id
            directory.mkdir(parents=True, mode=0o700)
            session = directory / 'session.jsonl'
        for name, filename in filenames.items():
            if name in supported and not kwargs.get(name):
                kwargs[name] = session if name == 'session_log' else directory / filename
        return original(*args, **kwargs)

    command.callback = callback
    return command
