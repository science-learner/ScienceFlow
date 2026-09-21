"""ParallelRunner responsibility: scheduler."""

from __future__ import annotations

import asyncio
import signal
import time
from pathlib import Path

from scienceflow.runtime.core.process.utils import kill_all_live_pgids
from scienceflow.runtime.parallel.config.manifest import _BUDGET_DONE_STATUS, _safe_filename
from scienceflow.runtime.parallel.config.models import TaskResult, TaskSpec
from scienceflow.runtime.parallel.config.preparation import _task_state_log_dir


@property
def resume_enabled(self) -> bool:
    return self._manifest.get("resume", False)


async def run_all(self) -> list[TaskResult]:
    if self.log_dir is not None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(self._max_concurrent)
    coros = [self._run_task(spec, sem) for spec in self._tasks]
    try:
        results = await asyncio.gather(*coros, return_exceptions=True)
    except (asyncio.CancelledError, KeyboardInterrupt):
        self._write_interrupted_states(self._tasks, error="interrupted by user")
        # Sweep any process groups that are still alive before propagating.
        kill_all_live_pgids(sig=signal.SIGTERM)
        await asyncio.sleep(3)
        kill_all_live_pgids(sig=signal.SIGKILL)
        raise

    final: list[TaskResult] = []
    for r in results:
        if isinstance(r, Exception):
            final.append(TaskResult(exp_id="unknown", run_id="unknown", status="error", error=str(r)))
        else:
            final.append(r)
    return final


def _task_subprocess_log_file(self, spec: TaskSpec) -> Path:
    if self.log_dir is not None:
        return self.log_dir / f"{_safe_filename(spec.run_id)}.log"
    return _task_state_log_dir(spec) / "parallel_subprocess.log"


async def _run_task(self, spec: TaskSpec, sem: asyncio.Semaphore) -> TaskResult:
    log_file = self._task_subprocess_log_file(spec)
    result = TaskResult(
        exp_id=spec.exp_id,
        run_id=spec.run_id,
        log_file=str(log_file),
    )

    if self.resume_enabled:
        remaining = self._check_resume(spec)
        if remaining is not None and remaining <= 0:
            spec = self._with_resume_remaining_budget(spec, 0)
            result.status = "skipped"
            self._write_state(spec, result, resolved_gpu="")
            return result
        if remaining is not None:
            spec = self._with_resume_remaining_budget(spec, remaining)

    resolved_gpu = ""
    interrupted_exc: BaseException | None = None
    async with sem:
        result.status = "running"
        t0 = time.monotonic()
        try:
            exit_code, resolved_gpu = await self._exec_subprocess(spec, result.log_file)
            result.exit_code = exit_code
            result.status = "success" if exit_code == 0 else "failed"
        except asyncio.TimeoutError:
            result.status = _BUDGET_DONE_STATUS
            result.error = f"time budget exhausted after {spec.time_limit}s"
        except (asyncio.CancelledError, KeyboardInterrupt) as e:
            result.status = "stopped_by_user"
            result.error = type(e).__name__
            interrupted_exc = e
        except Exception as e:
            result.status = "error"
            result.error = str(e)
        finally:
            result.elapsed_sec = round(time.monotonic() - t0, 1)

    self._write_state(spec, result, resolved_gpu=resolved_gpu)
    if interrupted_exc is not None:
        raise interrupted_exc
    return result
