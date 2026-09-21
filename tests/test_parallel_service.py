from __future__ import annotations

from pathlib import Path

import pytest

from scienceflow.runtime.parallel.config.models import TaskResult
from scienceflow.runtime.parallel import service


@pytest.mark.asyncio
async def test_run_manifest_is_public_shared_boundary(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[object, ...]] = []

    class Runner:
        def __init__(self, manifest_path, max_concurrent=None, log_dir=None) -> None:
            calls.append((manifest_path, max_concurrent, log_dir))
            self._tasks = [object(), object()]
            self._max_concurrent = 2

        async def run_all(self):
            return [TaskResult(exp_id="task", run_id="run", status="success")]

        @staticmethod
        def format_summary(results):
            assert len(results) == 1
            return "summary"

    monkeypatch.setattr(service, "ParallelRunner", Runner)
    started: list[tuple[int, int]] = []
    result = await service.run_manifest(
        tmp_path / "run.yaml",
        max_concurrent=2,
        log_dir=tmp_path / "logs",
        on_started=lambda count, concurrency: started.append((count, concurrency)),
    )

    assert calls == [(tmp_path / "run.yaml", 2, tmp_path / "logs")]
    assert started == [(2, 2)]
    assert result.task_count == 2
    assert result.max_concurrent == 2
    assert result.text == "summary"
    assert result.results[0].status == "success"
