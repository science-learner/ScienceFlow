"""Bash process lifecycle orchestration.

The function body is frozen while nested monitor closures are migrated to typed state.
"""

from __future__ import annotations
from scienceflow.runtime.safety.tooling.bash.core.pure import (
    Callable,
    RESOURCE_HEAVY_GPU_TRAIN,
    ToolResult,
    _resource_call,
    _start_bash_execution,
    asyncio,
    hashlib,
    time,
)
from scienceflow.runtime.safety.tooling.bash.monitor.execution import run_monitored_execution
from scienceflow.runtime.safety.tooling.bash.core.pipeline_preparation import AgentBackoffWaitFinalizer, prepare_bash_pipeline


async def _execute_process_pipeline(
    self, command: str, *, on_output: Callable[[str], None] | None = None, **kwargs
) -> ToolResult:
    """Orchestrate policy hooks around the isolated process session."""
    kwargs.pop("config", None)
    (preparation, preparation_error) = await prepare_bash_pipeline(self, command, on_output=on_output)
    if preparation_error is not None:
        return preparation_error
    assert preparation is not None
    cmd = preparation.command
    proc_env = preparation.environment
    ws = preparation.workspace_dir
    timeout = preparation.timeout_sec
    start = preparation.started_at
    start_wall = preparation.started_at_wall
    task_physical_gpu_pool = preparation.task_physical_gpu_pool
    gpu_ids = preparation.gpu_ids
    cpu_set = preparation.cpu_set
    classified = preparation.classified
    effective_resource_class = preparation.effective_resource_class
    resource_job_id = preparation.resource_job_id
    run_artifact_dir = preparation.run_artifact_dir
    run_state_path = preparation.run_state_path
    agent_backoff_sleep_sec = preparation.agent_backoff_sleep_sec
    agent_backoff_wait_id = preparation.agent_backoff_wait_id
    finish_agent_backoff_wait = AgentBackoffWaitFinalizer(self, preparation)
    if agent_backoff_sleep_sec is not None:
        agent_backoff_wait_id = (
            f"sleep:{int(start * 1000)}:{hashlib.sha1(cmd.encode(errors='replace')).hexdigest()[:12]}"
        )
        preparation.agent_backoff_wait_id = agent_backoff_wait_id
        wake_snapshot = _resource_call(self.resource_observer, "agent_backoff_wake_snapshot")
        start_pressure_generation = 0
        if isinstance(wake_snapshot, dict):
            try:
                start_pressure_generation = int(wake_snapshot.get("pressure_generation") or 0)
            except (TypeError, ValueError):
                start_pressure_generation = 0
        _resource_call(
            self.resource_observer,
            "agent_backoff_wait_started",
            wait_id=agent_backoff_wait_id,
            planned_sleep_sec=float(agent_backoff_sleep_sec),
            command=cmd,
            reason="agent_sleep_command",
            pressure_generation=start_pressure_generation,
        )
        remaining = float(agent_backoff_sleep_sec)
        try:
            poll_sec = max(0.05, float(proc_env.get("_SCIENCEFLOW_AGENT_BACKOFF_POLL_SEC", "15") or 15.0))
        except (TypeError, ValueError):
            poll_sec = 15.0
        while remaining > 0:
            chunk = min(remaining, poll_sec)
            await asyncio.sleep(max(0.0, chunk))
            remaining -= chunk
            wake = _resource_call(
                self.resource_observer,
                "agent_backoff_wake_decision",
                start_pressure_generation=start_pressure_generation,
                target_resource_class=RESOURCE_HEAVY_GPU_TRAIN,
                gpu_ids=gpu_ids,
            )
            if isinstance(wake, dict) and wake.get("wake"):
                elapsed_sleep = time.time() - start
                finish_agent_backoff_wait(
                    status="woken",
                    wake_reason=str(wake.get("wake_reason") or "resource_available"),
                    elapsed_sec=elapsed_sleep,
                    reason="resource_backoff_wake",
                )
                _resource_call(
                    self.resource_observer,
                    "job_finished",
                    resource_job_id,
                    status="success",
                    returncode=0,
                    elapsed_sec=elapsed_sleep,
                    reason="agent_backoff_woken",
                )
                return ToolResult(output="", error=None)
        elapsed_sleep = time.time() - start
        finish_agent_backoff_wait(status="success", wake_reason="timer_elapsed", elapsed_sec=elapsed_sleep, reason="")
        _resource_call(
            self.resource_observer,
            "job_finished",
            resource_job_id,
            status="success",
            returncode=0,
            elapsed_sec=elapsed_sleep,
            reason="agent_backoff_elapsed",
        )
        return ToolResult(output="", error=None)
    (execution, execution_result) = await _start_bash_execution(
        self,
        command=cmd,
        environment=proc_env,
        workspace_dir=ws,
        gpu_ids=gpu_ids,
        cpu_set=cpu_set,
        resource_job_id=resource_job_id,
        started_at=start,
        timeout_sec=timeout,
        finish_backoff_wait=finish_agent_backoff_wait,
    )
    if execution_result is not None:
        return execution_result
    assert execution is not None
    session = execution.session
    proc = session.process
    assert proc is not None
    observe_first_active = execution.observe_first_active
    observe_first_window_sec = execution.observe_first_window_sec
    return await run_monitored_execution(
        self,
        cmd=cmd,
        ws=ws,
        proc=proc,
        session=session,
        observe_first_active=observe_first_active,
        observe_first_window_sec=observe_first_window_sec,
        on_output=on_output,
        timeout=timeout,
        start=start,
        start_wall=start_wall,
        gpu_ids=gpu_ids,
        task_physical_gpu_pool=task_physical_gpu_pool,
        resource_job_id=resource_job_id,
        run_artifact_dir=run_artifact_dir,
        run_state_path=run_state_path,
        effective_resource_class=effective_resource_class,
        classified=classified,
        finish_backoff_wait=finish_agent_backoff_wait,
    )
