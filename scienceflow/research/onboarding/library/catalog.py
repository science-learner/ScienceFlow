"""Bounded task-folder discovery; malformed folders cannot break completion."""

import shlex
from pathlib import Path

import yaml

from scienceflow.runtime.task_package import iter_task_packages, load_task_package


def local_packages(workspace):
    library = Path(workspace) / 'task_library'
    if not library.is_dir():
        return []
    result = []
    for folder in sorted(library.iterdir())[:500]:
        if folder.name.startswith('.') or not folder.is_dir():
            continue
        try:
            result.append(load_task_package(folder))
        except (OSError, ValueError, yaml.YAMLError):
            continue
    return result


def resolve_local(text, workspace):
    text = text.strip()
    if not text:
        return None
    try:
        words = shlex.split(text)
    except ValueError:
        words = []
    for value in (text, words[0] if words else ''):
        if not value or len(value) > 2048 or '\n' in value:
            continue
        path = Path(value).expanduser()
        path = path if path.is_absolute() else Path(workspace) / path
        try:
            exists = (path / 'task.yaml').is_file()
        except OSError:
            continue
        if exists:
            return load_task_package(path)
    matches = [spec for spec in local_packages(workspace) if spec.task_id == text or spec.source_dir.name == text]
    if len(matches) > 1:
        raise ValueError('Multiple local task versions match; choose the full folder path.')
    return matches[0] if matches else None


def task_choices(workspace):
    local = [(shlex.quote(str(spec.source_dir)), f'{spec.task_id} · Local') for spec in local_packages(workspace)]
    builtin = [(spec.task_id, f'{spec.task_id} · Built-in') for spec in iter_task_packages()]
    return tuple(local + builtin)
