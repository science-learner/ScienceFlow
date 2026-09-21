"""Read-only discovery of worker time-trace files."""

from __future__ import annotations

from pathlib import Path

from scienceflow.runtime.observability.telemetry.trace.time_trace import TRACE_FILENAME

def _time_trace_paths(task_root: Path) -> list[Path]:
    directories = [task_root / "task_logs"]
    directories.extend(task_root.glob("task_logs/workers/w*"))
    directories.extend(task_root.glob("workers/w*/logs"))
    paths: list[Path] = []
    for directory in directories:
        candidate = directory / TRACE_FILENAME
        if candidate.is_file():
            paths.append(candidate)
    return paths

__all__ = tuple(name for name in globals() if not name.startswith("__"))
