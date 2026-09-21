"""Private run records, atomic writes and process identity checks."""

from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import psutil


def registry_root() -> Path:
    root = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state') / 'scienceflow/managed_runs'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', run_id):
        raise ValueError('Invalid run ID.')
    return registry_root() / run_id


def read_record(directory: Path) -> dict:
    path = directory / 'control.json'
    if not path.is_file():
        raise ValueError(f'Unknown managed run: {directory.name}')
    return json.loads(path.read_text())


def write_record(directory: Path, record: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with NamedTemporaryFile(mode='w', dir=directory, delete=False) as stream:
        json.dump(record, stream, indent=2)
        temp = Path(stream.name)
    try:
        temp.replace(directory / 'control.json')
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / '.lock', os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def alive(record: dict) -> bool:
    try:
        process = psutil.Process(int(record['pid']))
        return (abs(process.create_time() - float(record['process_created'])) < .01
                and process.status() != psutil.STATUS_ZOMBIE)
    except (KeyError, ValueError, psutil.Error):
        return False
