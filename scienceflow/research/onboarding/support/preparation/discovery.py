"""Bounded local task discovery; documents are evidence, never shell commands."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

DESCRIPTION_NAMES = ("description.md", "description_lite.md", "task_desc.txt", "README.md", "readme.md")


def discover_task(draft, workspace: Path) -> None:
    """Resolve an explicitly supplied task directory and collect local documents."""
    text = draft.task_text.strip()
    candidate = Path(text).expanduser() if text and '\n' not in text and len(text) < 2048 else None
    if candidate is not None and not candidate.is_absolute():
        candidate = workspace / candidate
    explicit_directory = candidate is not None and candidate.is_dir()
    if candidate is not None and not candidate.exists() and (text.startswith(('/', './', '../', '~'))):
        raise ValueError(f"Task path does not exist: {candidate}")
    root = candidate.resolve() if explicit_directory else workspace
    if draft.input_data_dir and draft.input_data_dir != "none":
        data = Path(draft.input_data_dir).expanduser()
        draft.input_data_dir = str((data if data.is_absolute() else workspace / data).resolve())
        if not explicit_directory:
            root = Path(draft.input_data_dir)
    draft.task_root = str(root)
    if explicit_directory:
        draft.task_text = ''
    documents = []
    if root.is_dir():
        for name in DESCRIPTION_NAMES if explicit_directory or draft.input_data_dir else ():
            path = root / name
            if path.is_file():
                # Bound reads before decoding, including symlink targets.
                with path.open('rb') as stream:
                    content = stream.read(48_001)
                documents.append((str(path), content[:48_000].decode('utf-8', errors='replace')))
                if name != 'description_lite.md':
                    break
        draft.discovered_files = tuple(p.name for p in islice(root.iterdir(), 100)
                                       if not p.name.startswith('.'))
    if documents:
        description = '\n\n'.join(f"Source: {path}\n{text}" for path, text in documents)
        draft.description_sources = tuple(path for path, _ in documents)
        draft.task_text = description + (f"\n\nUser objective:\n{draft.task_text}" if draft.task_text else '')
    if explicit_directory and not draft.task_text:
        draft.task_text = f"Research task in {root}; objective needs clarification."
    if explicit_directory and not draft.input_data_dir:
        candidates = [root / name for name in ('data', 'dataset', 'instances') if (root / name).is_dir()]
        if len(candidates) == 1:
            draft.input_data_dir = str(candidates[0])


def draft_summary(draft) -> str:
    name = draft.exp_id or Path(draft.task_root).name or 'New task'
    objective = (
        f"{'minimize' if draft.lower_is_better else 'maximize'} {draft.metric_name}"
        if isinstance(draft.lower_is_better, bool)
        else f"{draft.metric_name} · direction at runtime"
    )
    return (f"Task: {name} · {objective}\n"
            f"Artifact: {draft.artifact_path} · Data: {draft.input_data_dir}\n"
            f"{draft.workers} workers · GPU: {draft.gpu_list} · CPU: {draft.cpu_list} · "
            f"{(draft.wall_clock_sec or 0) / 60:g} min")
