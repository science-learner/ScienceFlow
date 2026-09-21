"""Preflight, leading sleep, CPU affinity, and admission for Bash execution."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shlex
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scienceflow.runtime.safety.tooling.bash.policy.admission import _bash_preflight
from scienceflow.runtime.safety.tooling.bash.core.request_builder import (
    _admit_bash_command,
    _resource_call,
)
from scienceflow.runtime.safety.tooling.bash.core.shared import (
    BashPreflight,
    Callable,
    Path,
    RESOURCE_HEAVY_GPU_TRAIN,
    ToolResult,
    _infer_timeout,
    logger,
)
from scienceflow.runtime.safety.tooling.workspace.shell_guards import (
    _format_cpu_set_compact,
    _parse_cpu_set_string,
)

if TYPE_CHECKING:
    from scienceflow.runtime.safety.tooling.bash.core.tool import BashTool


@dataclass(slots=True)
class BashPipelinePreparation:
    command: str
    environment: dict[str, str]
    workspace_dir: Path
    timeout_sec: float
    started_at: float
    started_at_wall: float
    task_physical_gpu_pool: list[str]
    gpu_ids: list[str]
    cpu_set: str
    classified: Any
    effective_resource_class: str
    resource_job_id: str | None
    run_artifact_dir: Path | None
    run_state_path: Path | None
    agent_backoff_sleep_sec: float | None = None
    agent_backoff_wait_id: str = ""
    agent_backoff_finished: bool = False


@dataclass(slots=True)
class AgentBackoffWaitFinalizer:
    tool: Any
    preparation: BashPipelinePreparation

    def __call__(
        self,
        *,
        status: str,
        wake_reason: str,
        elapsed_sec: float | None = None,
        returncode: int | None = None,
        reason: str = "",
    ) -> None:
        state = self.preparation
        if not state.agent_backoff_wait_id or state.agent_backoff_finished:
            return
        state.agent_backoff_finished = True
        _resource_call(
            self.tool.resource_observer,
            "agent_backoff_wait_finished",
            wait_id=state.agent_backoff_wait_id,
            planned_sleep_sec=float(state.agent_backoff_sleep_sec or 0.0),
            elapsed_sec=float(
                time.time() - state.started_at if elapsed_sec is None else elapsed_sec
            ),
            status=status,
            wake_reason=wake_reason,
            returncode=returncode,
            reason=reason,
        )


async def _apply_leading_sleep(
    tool: "BashTool",
    preflight: BashPreflight,
) -> tuple[str, float, float, str, bool, ToolResult | None]:
    command = preflight.command
    timeout = preflight.timeout_sec
    started_at = preflight.started_at
    wait_id = ""
    if preflight.sleep_prefix is None:
        return command, timeout, started_at, wait_id, False, None
    sleep_sec, after_sleep_command = preflight.sleep_prefix
    wait_id = (
        f"sleep:{int(started_at * 1000)}:"
        f"{hashlib.sha1(preflight.sleep_prefix_command.encode(errors='replace')).hexdigest()[:12]}"
    )
    wake_snapshot = _resource_call(
        tool.resource_observer,
        "agent_backoff_wake_snapshot",
    )
    try:
        pressure_generation = int((wake_snapshot or {}).get("pressure_generation") or 0)
    except (AttributeError, TypeError, ValueError):
        pressure_generation = 0
    _resource_call(
        tool.resource_observer,
        "agent_backoff_wait_started",
        wait_id=wait_id,
        planned_sleep_sec=float(sleep_sec),
        command=preflight.sleep_prefix_command,
        reason=(
            "agent_sleep_prefix_command"
            if after_sleep_command
            else "agent_sleep_command"
        ),
        pressure_generation=pressure_generation,
    )
    try:
        poll_sec = max(
            0.05,
            float(
                preflight.environment.get(
                    "_SCIENCEFLOW_AGENT_BACKOFF_POLL_SEC",
                    "15",
                )
                or 15.0
            ),
        )
    except (TypeError, ValueError):
        poll_sec = 15.0
    wait_status, wake_reason, finish_reason = "success", "timer_elapsed", ""
    remaining = float(sleep_sec)
    try:
        while remaining > 0:
            chunk = min(remaining, poll_sec)
            await asyncio.sleep(max(0.0, chunk))
            remaining -= chunk
            wake = _resource_call(
                tool.resource_observer,
                "agent_backoff_wake_decision",
                start_pressure_generation=pressure_generation,
                target_resource_class=RESOURCE_HEAVY_GPU_TRAIN,
                gpu_ids=preflight.gpu_ids,
            )
            if isinstance(wake, dict) and wake.get("wake"):
                wait_status = "woken"
                wake_reason = str(wake.get("wake_reason") or "resource_available")
                finish_reason = "resource_backoff_wake"
                break
    except asyncio.CancelledError:
        _resource_call(
            tool.resource_observer,
            "agent_backoff_wait_finished",
            wait_id=wait_id,
            planned_sleep_sec=float(sleep_sec),
            elapsed_sec=max(0.0, time.time() - started_at),
            status="cancelled",
            wake_reason="cancelled",
            reason="cancelled",
        )
        raise
    _resource_call(
        tool.resource_observer,
        "agent_backoff_wait_finished",
        wait_id=wait_id,
        planned_sleep_sec=float(sleep_sec),
        elapsed_sec=max(0.0, time.time() - started_at),
        status=wait_status,
        wake_reason=wake_reason,
        reason=finish_reason,
    )
    if not after_sleep_command:
        return (
            command,
            timeout,
            started_at,
            wait_id,
            True,
            ToolResult(output="", error=None),
        )
    command = after_sleep_command
    timeout = tool._apply_hard_fuse_timeout(
        _infer_timeout(
            command,
            float(tool.bash_timeout_sec),
            float(tool.bash_timeout_slow_sec),
        )
    )
    return command, timeout, time.time(), wait_id, True, None


def _apply_cpu_affinity(
    command: str,
    environment: dict[str, str],
) -> tuple[str, str]:
    cpu_set = (environment.pop("_SCIENCEFLOW_CPU_SET", "") or "").strip()
    if not cpu_set:
        return command, ""
    host_cores = os.cpu_count() or 0
    if host_cores <= 0:
        return f"taskset -c {cpu_set} bash -c {shlex.quote(command)}", cpu_set
    parsed_ids = _parse_cpu_set_string(cpu_set)
    valid_ids = [cpu_id for cpu_id in parsed_ids if cpu_id < host_cores]
    out_of_range = [cpu_id for cpu_id in parsed_ids if cpu_id >= host_cores]
    if out_of_range:
        logger.warning(
            "[bash-tool] cpu_set %r has %d out-of-range ID(s) (host has %d "
            "cores); dropping them before taskset",
            cpu_set,
            len(out_of_range),
            host_cores,
        )
    if not valid_ids:
        logger.warning(
            "[bash-tool] cpu_set %r is entirely out-of-range (host has %d "
            "cores); skipping taskset affinity",
            cpu_set,
            host_cores,
        )
        return command, ""
    normalized = _format_cpu_set_compact(valid_ids)
    return f"taskset -c {normalized} bash -c {shlex.quote(command)}", normalized


async def prepare_bash_pipeline(
    tool: "BashTool",
    command: str,
    *,
    on_output: Callable[[str], None] | None,
) -> tuple[BashPipelinePreparation | None, ToolResult | None]:
    preflight, error = _bash_preflight(tool, command)
    if error is not None:
        return None, error
    assert preflight is not None
    (
        cmd,
        timeout,
        started_at,
        wait_id,
        wait_finished,
        error,
    ) = await _apply_leading_sleep(tool, preflight)
    if error is not None:
        return None, error
    environment = preflight.environment
    cmd, cpu_set = _apply_cpu_affinity(cmd, environment)
    admission, error = await _admit_bash_command(
        tool,
        command=cmd,
        environment=environment,
        gpu_ids=preflight.gpu_ids,
        task_physical_gpu_pool=preflight.task_physical_gpu_pool,
        cpu_set=cpu_set,
        timeout_sec=timeout,
        workspace_dir=preflight.workspace_dir,
        started_at=started_at,
        on_output=on_output,
    )
    if error is not None:
        return None, error
    assert admission is not None
    queue_completed_at = time.time()
    return (
        BashPipelinePreparation(
            command=admission.command,
            environment=admission.environment,
            workspace_dir=preflight.workspace_dir,
            timeout_sec=timeout,
            started_at=queue_completed_at,
            started_at_wall=queue_completed_at,
            task_physical_gpu_pool=preflight.task_physical_gpu_pool,
            gpu_ids=admission.gpu_ids,
            cpu_set=cpu_set,
            classified=admission.classified,
            effective_resource_class=admission.effective_resource_class,
            resource_job_id=admission.resource_job_id,
            run_artifact_dir=admission.run_artifact_dir,
            run_state_path=admission.run_state_path,
            agent_backoff_wait_id=wait_id,
            agent_backoff_finished=wait_finished,
        ),
        None,
    )


__all__ = (
    "AgentBackoffWaitFinalizer",
    "BashPipelinePreparation",
    "prepare_bash_pipeline",
)
