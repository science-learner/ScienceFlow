"""Bash responsibility: process start."""

from __future__ import annotations

from scienceflow.runtime.safety.tooling.bash.core.shared import (
    BashExecutionStart,
    Callable,
    InquiryCraftProcessAdapter,
    Path,
    ProcessExecutionSession,
    ProcessRequest,
    ProcessSpawnError,
    ToolResult,
    build_spawn_failure_tool_result,
    logger,
    spawn_shell,
    terminate_tree,
    terminate_tree_recoverable,
    time,
    unregister_process_group,
)
from scienceflow.runtime.safety.tooling.bash.policy.admission import (
    _resource_call,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scienceflow.runtime.safety.tooling.bash.core.tool import BashTool


async def _start_bash_execution(
    tool: "BashTool",
    *,
    command: str,
    environment: dict[str, str],
    workspace_dir: Path,
    gpu_ids: list[str],
    cpu_set: str,
    resource_job_id: str | None,
    started_at: float,
    timeout_sec: float,
    finish_backoff_wait: Callable[..., None],
) -> tuple[BashExecutionStart | None, ToolResult | None]:
    """Spawn an admitted process and register the lease before monitoring."""

    adapter = InquiryCraftProcessAdapter(
        spawn_shell_fn=spawn_shell,
        terminate_fn=terminate_tree,
        terminate_recoverable_fn=terminate_tree_recoverable,
        unregister_fn=unregister_process_group,
    )
    session = ProcessExecutionSession(
        ProcessRequest(
            command=command,
            cwd=workspace_dir,
            environment=environment,
            timeout_sec=timeout_sec,
            stream_limit_bytes=int(tool.subprocess_stream_limit),
            session_id=str(resource_job_id or f"bash-{int(started_at * 1000)}"),
        ),
        adapter=adapter,
    )
    try:
        proc = await session.start()
    except ProcessSpawnError as spawn_error:
        exc = spawn_error.cause
        elapsed = time.time() - started_at
        finish_backoff_wait(
            status="spawn_failed",
            wake_reason="spawn_failed",
            elapsed_sec=elapsed,
            reason=type(exc).__name__,
        )
        _resource_call(
            tool.resource_observer,
            "job_finished",
            resource_job_id,
            status="spawn_failed",
            elapsed_sec=elapsed,
            reason=type(exc).__name__,
        )
        logger.warning("[bash-tool] subprocess spawn failed: %s", exc)
        return None, build_spawn_failure_tool_result(exc, elapsed_sec=elapsed)

    _resource_call(
        tool.resource_observer,
        "lease_registered",
        resource_job_id,
        pid=getattr(proc, "pid", None),
        pgid=getattr(proc, "pid", None),
        gpu_ids=gpu_ids,
        cpu_set=cpu_set or None,
    )
    observe_first_state = _resource_call(
        tool.resource_observer,
        "observe_first_state",
        resource_job_id,
    )
    if not isinstance(observe_first_state, dict):
        observe_first_state = {}
    observe_first_active = bool(observe_first_state.get("active"))
    try:
        observe_first_window_sec = max(
            1.0, float(observe_first_state.get("observe_window_sec") or 0.0)
        )
    except (TypeError, ValueError):
        observe_first_window_sec = 0.0
    return (
        BashExecutionStart(
            session=session,
            observe_first_active=observe_first_active,
            observe_first_window_sec=observe_first_window_sec,
        ),
        None,
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
