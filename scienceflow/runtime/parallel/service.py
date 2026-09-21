"""Public service boundary for running a Parallel manifest."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scienceflow.runtime.parallel.config.models import TaskResult
from scienceflow.runtime.parallel.execution.runner import ParallelRunner

ParallelRunStarted = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class ParallelRunSummary:
    """Stable result returned to CLI and interactive hosts."""

    results: tuple[TaskResult, ...]
    text: str
    task_count: int
    max_concurrent: int


async def run_manifest(
    manifest_path: str | Path,
    *,
    max_concurrent: int | None = None,
    log_dir: str | Path | None = None,
    on_started: ParallelRunStarted | None = None,
) -> ParallelRunSummary:
    """Load and execute one existing Parallel manifest.

    ``on_started`` exists so terminal hosts can retain the historical launch
    message without reading ``ParallelRunner`` internals themselves.
    """

    runner = ParallelRunner(
        manifest_path,
        max_concurrent=max_concurrent,
        log_dir=log_dir,
    )
    task_count = len(runner._tasks)
    concurrency = int(runner._max_concurrent or 1)
    if on_started is not None:
        on_started(task_count, concurrency)
    results = tuple(await runner.run_all())
    return ParallelRunSummary(
        results=results,
        text=runner.format_summary(list(results)),
        task_count=task_count,
        max_concurrent=concurrency,
    )


__all__ = ["ParallelRunStarted", "ParallelRunSummary", "run_manifest"]
