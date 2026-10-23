"""Lifecycle composition and finalization for one monitored Bash process."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from inquirycraft.tools import ToolResult
from scienceflow.runtime.core.process import ProcessStatus, render_combined_output
from scienceflow.runtime.safety.tooling.bash.monitor.effects import ResourceTerminationEffects
from scienceflow.runtime.safety.tooling.bash.monitor.guard import ResourceGuardMonitor
from scienceflow.runtime.safety.tooling.bash.monitor.stream import BashStreamMonitor
from scienceflow.runtime.safety.tooling.bash.policy.admission import _resource_call
from scienceflow.runtime.safety.tooling.bash.policy.output import _postprocess_bash_execution
from scienceflow.runtime.safety.tooling.resource_management.resource_timeout_feedback import (
    build_command_timeout_feedback,
)


async def run_monitored_execution(
    self,
    *,
    cmd: str,
    ws: Any,
    proc: Any,
    session: Any,
    observe_first_active: bool,
    observe_first_window_sec: float,
    on_output: Any,
    timeout: float,
    start: float,
    start_wall: float,
    gpu_ids: list[str],
    task_physical_gpu_pool: list[str],
    resource_job_id: str | None,
    run_artifact_dir: Any,
    run_state_path: Any,
    effective_resource_class: str,
    classified: Any,
    finish_backoff_wait: Any,
) -> ToolResult:
    return await BashExecutionMonitor(locals()).run()



class BashExecutionMonitor(ResourceGuardMonitor, ResourceTerminationEffects):
    """Compose stream, guard decision, termination effect, and finalization owners."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.tool = payload["self"]
        for name, value in payload.items():
            if name != "self":
                setattr(self, name, value)



    async def run(self) -> ToolResult:
        self.stream_monitor = BashStreamMonitor(self.tool, workspace_dir=self.ws, resource_job_id=self.resource_job_id, run_artifact_dir=self.run_artifact_dir, run_state_path=self.run_state_path, started_at=self.start, started_at_wall=self.start_wall, on_output=self.on_output)
        self.io_touch = self.stream_monitor.io_touch
        self.heartbeat_active = self.stream_monitor.heartbeat_active
        self.stream_stats = self.stream_monitor.stats
        self._maybe_emit_progress_heartbeat = self.stream_monitor.emit_progress
        self._maybe_emit_artifact_progress = self.stream_monitor.emit_artifact
        self.observe_first_health_emitted = [False]
        self.guard_reason: list[str | None] = [None]
        self.guard_feedback: list[str | None] = [None]
        self.guard_recommendation_emitted = [False]
        self.guard_observe_more_until = [0.0]
        self.placement_audit_last = [0.0]
        self.guard_task: asyncio.Task[None] | None = None
        self.guard_watchdog_task: asyncio.Task[None] | None = None
        self.last_monitor_heartbeat_wall = [0.0]
        self.guard_watchdog_restarts = [0]
        self.workspace_cleanup_done = [False]
        self.session.set_chunk_consumer(self.stream_monitor.consume_chunk)
        hb_task: asyncio.Task[None] | None = None
        if self.on_output is not None and self.stream_monitor.heartbeat_interval > 0:
            hb_task = asyncio.create_task(self.stream_monitor.heartbeat(self.timeout))
        self.guard_task = asyncio.create_task(self._resource_guard())
        self.guard_watchdog_task = asyncio.create_task(self._resource_guard_watchdog())
        try:
            try:
                outcome = await self.session.run()
                if outcome.status is ProcessStatus.TIMED_OUT:
                    await self._run_workspace_gpu_cleanup(reason="workspace_cleanup_after_timeout")
                    self.finish_backoff_wait(status="timeout", wake_reason="timeout", elapsed_sec=time.time() - self.start, returncode=getattr(self.proc, "returncode", None), reason=f"timeout_after_{self.timeout:.0f}s")
                    _resource_call(self.tool.resource_observer, "job_finished", self.resource_job_id, status="timeout", returncode=getattr(self.proc, "returncode", None), elapsed_sec=time.time() - self.start, reason=f"timeout_after_{self.timeout:.0f}s")
                    timeout_feedback = build_command_timeout_feedback(
                        timeout_sec=float(self.timeout),
                        resource_class=str(self.effective_resource_class or self.classified.resource_class or ""),
                        gpu_ids=self.gpu_ids,
                        saw_progress=bool(self.stream_stats.get("saw_training_progress")),
                        saw_artifact=bool(self.stream_stats.get("last_artifact") or self.stream_stats.get("last_artifact_emitted")),
                        current_phase=str(self.stream_stats.get("current_phase") or ""),
                    )
                    raw_timeout_output = render_combined_output(outcome.output)
                    return ToolResult(output=timeout_feedback, error=f"Command timed out after {self.timeout:.0f}s", system=raw_timeout_output or self.cmd)
            except asyncio.CancelledError:
                await self._run_workspace_gpu_cleanup(reason="workspace_cleanup_after_cancelled")
                self.finish_backoff_wait(status="cancelled", wake_reason="cancelled", elapsed_sec=time.time() - self.start, returncode=getattr(self.proc, "returncode", None), reason="cancelled")
                _resource_call(self.tool.resource_observer, "job_finished", self.resource_job_id, status="cancelled", returncode=getattr(self.proc, "returncode", None), elapsed_sec=time.time() - self.start, reason="cancelled")
                raise
        finally:
            if hb_task is not None and (not hb_task.done()):
                hb_task.cancel()
                try:
                    await hb_task
                except asyncio.CancelledError:
                    pass
            if self.guard_task is not None and (not self.guard_task.done()):
                self.guard_task.cancel()
                try:
                    await self.guard_task
                except asyncio.CancelledError:
                    pass
            if self.guard_watchdog_task is not None and (not self.guard_watchdog_task.done()):
                self.guard_watchdog_task.cancel()
                try:
                    await self.guard_watchdog_task
                except asyncio.CancelledError:
                    pass
            await self._run_workspace_gpu_cleanup(reason="workspace_cleanup_after_command_finally")
            if self.on_output is not None and self.heartbeat_active[0]:
                self.on_output("\r\x1b[K")
                self.heartbeat_active[0] = False
        elapsed = time.time() - self.start
        stdout_parts = [outcome.output.stdout]
        stderr_parts = [outcome.output.stderr]
        self._maybe_emit_progress_heartbeat(dict(self.stream_stats.get("last_progress_signals") or {}), elapsed_sec=elapsed, force=True, reason="final")
        self._maybe_emit_artifact_progress(elapsed, force=True)
        return _postprocess_bash_execution(
            self.tool,
            command=self.cmd,
            workspace_dir=self.ws,
            process=self.proc,
            stdout_parts=stdout_parts,
            stderr_parts=stderr_parts,
            elapsed_sec=elapsed,
            started_at_wall=self.start_wall,
            classified=self.classified,
            effective_resource_class=self.effective_resource_class,
            resource_job_id=self.resource_job_id,
            guard_reason=self.guard_reason[0],
            guard_feedback=self.guard_feedback[0],
            finish_backoff_wait=self.finish_backoff_wait,
        )



__all__ = ("run_monitored_execution",)
