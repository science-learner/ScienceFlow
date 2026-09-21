"""Start, inspect, stop and resume the same manifest independently of a terminal."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import psutil
import yaml

from .registry import alive, locked, read_record, registry_root, run_dir, write_record


class WorkspaceBusy(ValueError):
    """An active experiment already owns an overlapping workspace."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"Task workspace is already running: {run_id}")


def start_run(manifest_path, *, draft=None, check_resources=False, library_workspace=None) -> dict:
    from scienceflow.runtime.parallel.execution.runner import ParallelRunner

    source = Path(manifest_path).expanduser().resolve()
    runner = ParallelRunner(source)
    tasks = runner._tasks
    if not tasks:
        raise ValueError('Manifest contains no tasks.')
    run_id = (tasks[0].run_id or "run-" + uuid4().hex[:12]) if len(tasks) == 1 else 'batch-' + uuid4().hex[:12]
    directory = run_dir(run_id)
    roots = [str(Path(t.workspace).resolve()) for t in tasks]
    control_workspace = str(Path(draft.workspace_base).resolve().parent) if draft is not None else None
    scopes = [control_workspace] if control_workspace else roots
    from .selection import overlaps
    # A registry-wide lock prevents concurrent starts targeting the same task root.
    with locked(registry_root()):
        for record in list_runs():
            occupied = record.get('task_roots', ()) if check_resources and 'resource_claims' not in record else record.get('lock_scopes', record.get('task_roots', ()))
            if record['alive'] and overlaps(occupied, scopes):
                raise WorkspaceBusy(record['run_id'])
        with locked(directory):
            if (directory / 'control.json').exists():
                raise ValueError(f'Run ID already exists: {run_id}. Use resume or choose a new run ID.')
            if draft is None and len(tasks) == 1:
                from scienceflow.research.onboarding import LongResearchDraft
                task = tasks[0]
                lnr = task.lnr_patch or {}
                draft = LongResearchDraft(exp_id=task.exp_id, run_id=task.run_id, task_text=task.task,
                    workspace_base=str(Path(task.workspace).parent.parent), input_data_dir=str(task.input_data_dir or ''),
                    workers=int(lnr.get('num_workers', 2)), wall_clock_sec=int(lnr.get('wall_clock_budget_sec', task.time_limit)),
                    gpu_list=str(task.gpu_list_raw), cpu_list=str(task.cpu_list))
            record = {'run_id': run_id, 'source_manifest': str(source), 'task_roots': roots,
                      'control_workspace': control_workspace, 'lock_scopes': scopes,
                      'library_workspace': str(Path(library_workspace).resolve()) if library_workspace else control_workspace,
                      'cwd': str(Path.cwd()), 'created_at': time.time(), 'attempt': 0,
                      'draft': asdict(draft) if draft is not None else None}
            # Keep the exact resolved runner inputs; relative manifest paths retain their launch cwd.
            payload = yaml.safe_load(source.read_text())
            record['manifest_payload'] = payload
            if check_resources:
                from .resource_claims import apply_reservation
                apply_reservation(record, list_runs())
            return _launch(directory, record, resume=False)


def _launch(directory: Path, record: dict, *, resume: bool) -> dict:
    payload = dict(record['manifest_payload'])
    payload['resume'] = resume
    if resume:
        payload['resume_budget_policy'] = 'remaining'
    record.update(status='starting', attempt=record['attempt'] + 1, launch_token=uuid4().hex,
                  started_at=time.time(), finished_at=None, error=None)
    manifest = directory / f"manifest-{record['attempt']}.yaml"
    fd = os.open(manifest, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    record['manifest'] = str(manifest)
    record['log'] = str(directory / f"supervisor-{record['attempt']}.log")
    write_record(directory, record)
    fd = os.open(record['log'], os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'a') as output:
        process = subprocess.Popen([sys.executable, '-m', 'scienceflow.runtime.parallel.control.worker',
                                    record['run_id'], record['launch_token']],
                                   cwd=record['cwd'], stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    record.update(pid=process.pid, process_created=psutil.Process(process.pid).create_time())
    write_record(directory, record)
    return _public(record)


def _public(record: dict) -> dict:
    # Never expose manifest credentials through status commands or the TUI.
    result = {k: v for k, v in record.items() if k not in {'manifest_payload', 'launch_token'}}
    result['alive'] = alive(record)
    if not result['alive'] and result['status'] in {'starting', 'running', 'stopping'}:
        result['status'] = 'interrupted'
    return result


def get_run(run_id: str) -> dict:
    return _public(read_record(run_dir(run_id)))


def list_runs() -> list[dict]:
    result = []
    for path in registry_root().glob('*/control.json'):
        try:
            result.append(_public(read_record(path.parent)))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(result, key=lambda row: row['created_at'], reverse=True)


def stop_run(run_id: str, *, timeout: float = 20) -> dict:
    directory = run_dir(run_id)
    with locked(directory):
        record = read_record(directory)
        if record.get('status') in {'completed', 'failed', 'stopped'} or not alive(record):
            return _public(record)
        if record['status'] != 'stopping':
            record['status'] = 'stopping'
            write_record(directory, record)
            # Do not cancel cleanup twice when multiple clients request stop.
            try:
                os.kill(record['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = get_run(run_id)
        if not current['alive']:
            return get_run(run_id)
        time.sleep(.1)
    return get_run(run_id)


def resume_run(run_id: str, *, check_resources=False) -> dict:
    directory = run_dir(run_id)
    with locked(registry_root()):
        with locked(directory):
            record = read_record(directory)
            if alive(record):
                raise ValueError('Run is already active; attach its TUI instead of resuming twice.')
            if record.get('status') == 'completed':
                raise ValueError('Run is already completed. Start a new experiment explicitly instead of resuming it.')
            check_resources = check_resources or 'resource_claims' in record
            from .selection import overlaps
            roots = record.get('task_roots', ()) if check_resources and 'resource_claims' not in record else record.get('lock_scopes', record.get('task_roots', ()))
            if any(r['alive'] and overlaps(roots, r.get('task_roots', ()) if check_resources and 'resource_claims' not in r
                                          else r.get('lock_scopes', r.get('task_roots', ()))) for r in list_runs()):
                raise ValueError('Another run is using the same task workspace.')
            if check_resources or 'resource_claims' in record:
                from .resource_claims import apply_reservation
                apply_reservation(record, list_runs())
                record['lock_scopes'] = roots
            from scienceflow.runtime.parallel.execution.runner import ParallelRunner
            runner = ParallelRunner(record['manifest'])
            if len(runner._tasks) == 1:
                remaining = runner._check_resume(runner._tasks[0])
                if remaining is not None:
                    if remaining <= 0:
                        raise ValueError('No remaining budget or task already completed; resume will not start another process.')
                    record['display_budget_sec'] = remaining
            return _launch(directory, record, resume=True)
