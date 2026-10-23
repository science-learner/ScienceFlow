"""Bounded task discovery shared by CLI and TUI entry points."""

import shlex
from dataclasses import dataclass
from pathlib import Path

import yaml

from scienceflow.runtime.task_package import iter_task_packages, load_task_package


@dataclass(frozen=True)
class TaskChoice:
    """One selectable task while preserving its source-specific launch value."""

    value: str
    task_id: str
    source: str
    name: str = ""


def local_packages(workspace):
    library = Path(workspace) / "task_library"
    if not library.is_dir():
        return []
    result = []
    for folder in sorted(library.iterdir())[:500]:
        if folder.name.startswith(".") or not folder.is_dir():
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
    for value in (text, words[0] if words else ""):
        if not value or len(value) > 2048 or "\n" in value:
            continue
        path = Path(value).expanduser()
        path = path if path.is_absolute() else Path(workspace) / path
        try:
            exists = (path / "task.yaml").is_file()
        except OSError:
            continue
        if exists:
            return load_task_package(path)
    matches = [
        spec
        for spec in local_packages(workspace)
        if spec.task_id == text or spec.source_dir.name == text
    ]
    if len(matches) > 1:
        raise ValueError(
            "Multiple local task versions match; choose the full folder path."
        )
    return matches[0] if matches else None


def task_choices(workspace):
    return tuple(
        (shlex.quote(choice.task_id), f"{choice.name or choice.task_id} · {choice.source}")
        for choice in discover_tasks(workspace)
    )


def discover_tasks(workspace):
    """Keep every local registration alongside canonical built-in task IDs."""
    local = [
        TaskChoice(str(spec.source_dir), spec.source_dir.name, "Local", spec.task_id)
        for spec in local_packages(workspace)
    ]
    builtin = [
        TaskChoice(spec.task_id, spec.task_id, "Built-in")
        for spec in iter_task_packages()
    ]
    return tuple(local + builtin)


def task_selection_text(workspace):
    """Render the common task picker without owning terminal input."""
    choices = discover_tasks(workspace)
    lines = ["Available tasks:"]
    lines.extend(
        f"  {index}. {choice.name or choice.task_id} · {choice.source}"
        for index, choice in enumerate(choices, 1)
    )
    lines.append("  0. Describe a new task")
    return "\n".join(lines), choices


def resolve_task_selection(value, choices):
    """Resolve a displayed index or exact task ID; other text is a new task."""
    answer = str(value or "").strip()
    if not answer:
        raise ValueError(
            "Enter a task number, task name, path, or new task description."
        )
    lexer = shlex.shlex(answer, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        head = next(lexer)
    except (ValueError, StopIteration):
        return answer
    suffix = answer[lexer.instream.tell():].strip()
    if head.isdecimal():
        if suffix and "=" not in suffix:
            return answer
        index = int(head)
        if index == 0:
            return suffix or None
        if 1 <= index <= len(choices):
            selected = choices[index - 1].value
            return f"{shlex.quote(selected)} {suffix}".strip() if suffix else selected
        raise ValueError(f"Task number must be between 0 and {len(choices)}.")
    selected = next(
        (choice.value for choice in choices if head in {choice.task_id, choice.value}),
        None,
    )
    if selected is not None:
        return f"{shlex.quote(selected)} {suffix}".strip() if suffix else selected
    return answer
