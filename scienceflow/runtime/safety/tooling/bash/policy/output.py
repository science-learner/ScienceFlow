"""Bash responsibility: output policy."""

from __future__ import annotations

from scienceflow.runtime.safety.tooling.bash.core.shared import (
    Any,
    Callable,
    Path,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ToolResult,
    _dedup_repeated_blocks,
    _distill_tracebacks,
    _has_masked_python_traceback,
    _has_silent_redirect,
    _maybe_lossless_observation_summary,
    _sanitize_model_visible_output_paths,
    _trim_output,
    logger,
    os,
)
from scienceflow.runtime.safety.tooling.bash.policy.admission import (
    _OOM_RE,
    _resource_call,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scienceflow.runtime.safety.tooling.bash.core.tool import BashTool


def _resource_guard_tool_result(
    tool: "BashTool",
    *,
    output: str,
    captured_output: str,
    returncode: int,
    elapsed_sec: float,
    resource_job_id: str | None,
    guard_reason: str,
    guard_feedback: str | None,
) -> ToolResult:
    _resource_call(
        tool.resource_observer,
        "job_finished",
        resource_job_id,
        status="resource_guard_terminated",
        returncode=returncode,
        elapsed_sec=elapsed_sec,
        reason=guard_reason,
    )
    deterministic = os.environ.get(
        "SCIENCEFLOW_DETERMINISTIC_GATE",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    header = (
        f"resource_guard_terminated, exit={returncode}"
        if deterministic
        else f"resource_guard_terminated, exit={returncode}, {elapsed_sec:.1f}s"
    )
    pieces = [f"[{header}]"]
    if str(guard_feedback or "").strip():
        pieces.append(str(guard_feedback).strip())
    if output.strip():
        pieces.append(output)
    if "boundary_violation" in guard_reason:
        error = "Resource guard stopped boundary violation"
    elif guard_reason.startswith("active_intervention:"):
        error = "Resource guard terminated slow progress"
    else:
        error = "Resource guard terminated stalled heavy"
    return ToolResult(
        output="\n".join(pieces).rstrip(),
        error=error,
        system=f"[{header}]\n{captured_output}".rstrip(),
    )


def _standard_bash_tool_result(
    tool: "BashTool",
    *,
    command: str,
    output: str,
    captured_output: str,
    returncode: int,
    elapsed_sec: float,
    resource_job_id: str | None,
) -> ToolResult:
    oom_detected = bool(
        returncode != 0 and _OOM_RE.search(captured_output or output or "")
    )
    finish_status = (
        "success" if returncode == 0 else "oom" if oom_detected else "failed"
    )
    finish_reason = (
        None
        if returncode == 0
        else "oom_detected"
        if oom_detected
        else f"non_zero_exit_{returncode}"
    )
    _resource_call(
        tool.resource_observer,
        "job_finished",
        resource_job_id,
        status=finish_status,
        returncode=returncode,
        elapsed_sec=elapsed_sec,
        reason=finish_reason,
    )
    deterministic = os.environ.get(
        "SCIENCEFLOW_DETERMINISTIC_GATE",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    header = (
        f"exit={returncode}"
        if deterministic
        else f"exit={returncode}, {elapsed_sec:.1f}s"
    )
    raw_system = f"[{header}]\n{captured_output}".rstrip()
    if returncode != 0 and elapsed_sec < 0.5 and not output.strip():
        is_quiet = _has_silent_redirect(command) or returncode == 2
        if is_quiet:
            hint = (
                f"[no-output] command exited rc={returncode} with no captured output. "
                "This typically means the targets do not exist "
                "(e.g. missing file or unmatched glob), or stderr was explicitly "
                "silenced (`2>/dev/null` / `&>/dev/null`). This is **not** an "
                'infrastructure failure; do not retry purely to "verify the env".'
            )
            logger.info(
                "[bash-tool] quiet-rc detected: rc=%d elapsed=%.2fs cmd=%r",
                returncode,
                elapsed_sec,
                command,
            )
        else:
            hint = (
                "[infra-error] Process exited immediately (< 0.5 s) with no output. "
                "This is likely an infrastructure failure (e.g. taskset/affinity "
                "misconfiguration), NOT a bug in your code. Do NOT keep retrying "
                "the same command. If this persists across multiple attempts, "
                "report it as an environment issue and stop wasting budget on "
                "bash retries."
            )
            logger.warning(
                "[bash-tool] fast-fail detected: rc=%d elapsed=%.2fs cmd=%r",
                returncode,
                elapsed_sec,
                command,
            )
        return ToolResult(
            output=f"[{header}]\n{hint}",
            error=f"non-zero exit code {returncode}",
            system=raw_system,
        )
    block = f"[{header}]\n{output}"
    if returncode != 0:
        return ToolResult(
            output=block,
            error=f"non-zero exit code {returncode}",
            system=raw_system,
        )
    return ToolResult(output=block, system=raw_system)


def _postprocess_bash_execution(
    tool: "BashTool",
    *,
    command: str,
    workspace_dir: Path,
    process: Any,
    stdout_parts: list[str],
    stderr_parts: list[str],
    elapsed_sec: float,
    started_at_wall: float,
    classified: Any,
    effective_resource_class: str,
    resource_job_id: str | None,
    guard_reason: str | None,
    guard_feedback: str | None,
    finish_backoff_wait: Callable[..., None],
) -> ToolResult:
    """Distill output, finalize resource state, and build the stable ToolResult."""

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    parts = [stdout]
    if stderr.strip():
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    output = "\n".join(parts).rstrip()
    captured_output = output
    output = _dedup_repeated_blocks(output, min_repeat=tool.dedup_min_repeat)
    output = _distill_tracebacks(
        output,
        tool.workspace_dir,
        enabled=tool.distill_tracebacks,
    )
    output = _sanitize_model_visible_output_paths(
        output,
        workspace_dir,
        tool.path_guard_extra_roots or (),
        strip_symlink_targets=bool(tool.strip_symlink_targets),
    )
    returncode = process.returncode if process.returncode is not None else -1
    finish_backoff_wait(
        status="success" if returncode == 0 else "failed",
        wake_reason="timer_elapsed" if returncode == 0 else "non_zero_exit",
        elapsed_sec=elapsed_sec,
        returncode=returncode,
        reason="" if returncode == 0 else f"non_zero_exit_{returncode}",
    )
    if _has_masked_python_traceback(command, output, returncode):
        output = (
            output.rstrip()
            + "\n[masked-zero-exit] Python traceback detected despite shell exit 0; "
            "treating this as failure because a pipeline likely hid the Python exit code."
        )
        returncode = 1
    output, raw_output_for_artifact = _maybe_lossless_observation_summary(
        command=command,
        output=output,
        enabled=bool(tool.observation_summary_enabled),
        max_chars=int(tool.max_output_chars),
        returncode=returncode,
    )
    if raw_output_for_artifact is None:
        output = _trim_output(output, tool.max_output_chars)

    heavy_resource_classes = {
        RESOURCE_HEAVY_GPU_CANDIDATE,
        RESOURCE_HEAVY_GPU_TRAIN,
        RESOURCE_GPU_FEATURE_EXTRACT,
        RESOURCE_GPU_LIGHT_TRAIN,
        RESOURCE_UNKNOWN_GPU_EXEC,
    }
    if (
        str(effective_resource_class or classified.resource_class or "")
        in heavy_resource_classes
    ):
        gap = None
        try:
            from scienceflow.research.solver.lnr.resources.runtime.execution.process.sidecar import (
                checkpoint_submission_gap,
            )

            gap = checkpoint_submission_gap(
                workspace_dir=workspace_dir,
                started_at=started_at_wall,
                candidate_artifact=str(
                    (tool.extra_env or {}).get("SCIENCEFLOW_CANDIDATE_ARTIFACT") or ""
                ),
            )
        except Exception:
            logger.debug(
                "[bash-resource] checkpoint_submission_gap failed", exc_info=True
            )
        if gap:
            checkpoint_feedback = str(
                _resource_call(
                    tool.resource_observer,
                    "checkpoint_to_submission_guard",
                    resource_job_id,
                    gap=gap,
                    elapsed_sec=elapsed_sec,
                )
                or ""
            ).strip()
            if checkpoint_feedback:
                output = (output.rstrip() + "\n" + checkpoint_feedback).strip()

    if guard_reason:
        return _resource_guard_tool_result(
            tool,
            output=output,
            captured_output=captured_output,
            returncode=returncode,
            elapsed_sec=elapsed_sec,
            resource_job_id=resource_job_id,
            guard_reason=guard_reason,
            guard_feedback=guard_feedback,
        )

    return _standard_bash_tool_result(
        tool,
        command=command,
        output=output,
        captured_output=captured_output,
        returncode=returncode,
        elapsed_sec=elapsed_sec,
        resource_job_id=resource_job_id,
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
