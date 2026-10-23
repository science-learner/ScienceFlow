"""Bash responsibility: admission."""

from __future__ import annotations

from scienceflow.runtime.safety.tooling.bash.core.shared import (
    Any,
    BashPreflight,
    Callable,
    Path,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ShellGuardRule,
    ToolResult,
    _DANGEROUS_PATTERNS,
    _command_executes_under_readonly_dir,
    _infer_timeout,
    _is_gpu_visibility_probe,
    _leading_cd_abs_path_missing,
    _parse_leading_sleep_command,
    _parse_visible_gpu_ids,
    asyncio,
    background_resource_command_blocked_error,
    dangerous_delete_command_blocked_error,
    evaluate_shell_guards,
    global_filesystem_scan_blocked_error,
    hidden_workspace_path_usage_blocked_error,
    inspect,
    interactive_stdin_blocked_error,
    logger,
    make_path_env,
    mixed_file_write_execution_blocked_error,
    normalize_bash_command_for_agent,
    os,
    privilege_escalation_blocked_error,
    process_control_blocked_error,
    re,
    shared_python_env_write_blocked_error,
    time,
    truncated_resource_output_blocked_error,
    workspace_scope_path_blocked_error,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scienceflow.runtime.safety.tooling.bash.core.tool import BashTool


def _dangerous_pattern_error(command: str) -> str | None:
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return f"Blocked potentially dangerous command: {command!r}"
    return None


def _bash_guard_error(tool: "BashTool", command: str) -> str | None:
    """Compose ScienceFlow's frozen policy on InquiryCraft's ordered guard engine."""

    rules = (
        ShellGuardRule("privilege_escalation", privilege_escalation_blocked_error),
        ShellGuardRule("process_control", process_control_blocked_error),
        ShellGuardRule("interactive_stdin", interactive_stdin_blocked_error),
        ShellGuardRule(
            "workspace_scope",
            lambda value: (
                workspace_scope_path_blocked_error(
                    value,
                    workspace_dir=tool.workspace_dir,
                    allowed_roots=tool.path_guard_extra_roots,
                )
                if bool(tool.forbid_host_absolute_paths)
                else None
            ),
        ),
        ShellGuardRule("global_scan", global_filesystem_scan_blocked_error),
        ShellGuardRule(
            "mixed_write_execution", mixed_file_write_execution_blocked_error
        ),
        ShellGuardRule(
            "background_resource", background_resource_command_blocked_error
        ),
        ShellGuardRule("dangerous_delete", dangerous_delete_command_blocked_error),
        ShellGuardRule(
            "truncated_resource_output", truncated_resource_output_blocked_error
        ),
        ShellGuardRule(
            "shared_python_env_write", shared_python_env_write_blocked_error
        ),
        ShellGuardRule(
            "hidden_workspace_path",
            lambda value: hidden_workspace_path_usage_blocked_error(
                value,
                tool.path_guard_denied_prefixes,
            ),
        ),
        ShellGuardRule("dangerous_pattern", _dangerous_pattern_error),
        ShellGuardRule(
            "readonly_execution",
            lambda value: (
                (
                    f"Blocked: executing scripts under read-only directory {blocked!r} is not allowed. "
                    "Read files with read/grep/glob/ls or non-executing bash (e.g. cat, head) instead."
                )
                if (
                    blocked := _command_executes_under_readonly_dir(
                        value, list(tool.readonly_dirs or [])
                    )
                )
                else None
            ),
        ),
        ShellGuardRule(
            "leading_absolute_cd",
            lambda value: _leading_cd_abs_path_missing(
                value,
                workspace_dir=tool.workspace_dir,
                allowed_roots=tool.path_guard_extra_roots,
            ),
        ),
    )
    return evaluate_shell_guards(command, rules).error


_FINAL_VALIDATION_SCORE_RE = re.compile(
    r"Final\s+Validation\s+Score\s*:\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
)

_TRAINING_PROGRESS_RE = re.compile(
    r"\b(train|training|fit|fitting|epoch|epochs|fold|folds|iter|iteration|"
    r"batch|loss|auc|label)\b|\(\s*\d+\s*/\s*\d+\s*\)",
    re.IGNORECASE,
)

_VALIDATION_OR_INFERENCE_RE = re.compile(
    r"\b(validating|validation|evaluate|evaluating|inference|predict|predicting|"
    r"submission|saved submission)\b",
    re.IGNORECASE,
)

_OOM_RE = re.compile(
    r"\b(cuda\s+out\s+of\s+memory|out\s+of\s+memory|oom)\b", re.IGNORECASE
)


def _queue_label_for_resource_class(resource_class: str) -> tuple[str, str]:
    cls = str(resource_class or "")
    if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
        return "GPU train", "gpu_train"
    if cls == RESOURCE_GPU_LIGHT_TRAIN:
        return "GPU light train", "gpu_light_train"
    if cls == RESOURCE_GPU_TT_LIGHT:
        return "GPU TT", "gpu_tt"
    if cls == RESOURCE_GPU_FEATURE_EXTRACT:
        return "GPU feature", "gpu_feature"
    if cls == RESOURCE_UNKNOWN_GPU_EXEC:
        return "GPU unknown", "gpu_unknown"
    return "compute", "compute"


def _executed_resource_termination_feedback(
    feedback: str,
    *,
    action: str,
) -> str:
    """Make an executed termination unambiguous to the owning agent."""
    action_text = str(action or "").upper()
    marker = "The command has already been terminated and is no longer running. Do not wait for it."
    detail = str(feedback or "").strip()
    if action_text == "KILL_AND_REPLAN" and detail.startswith(
        "Resource arbiter approved kill because"
    ):
        detail = detail.replace(
            "Resource arbiter approved kill because",
            "Resource arbiter has already terminated this command because",
            1,
        )
    return marker if not detail else marker + "\n" + detail


def _resource_queue_feedback(
    elapsed: float,
    max_wait_sec: float,
    *,
    queue_label: str,
    queue_key: str,
) -> str:
    return (
        f"RESOURCE_FEEDBACK: queue_wait_aborted because the {queue_label.lower()} slot "
        f"stayed busy for {elapsed:.1f}s, exceeding the {max_wait_sec:.0f}s queue budget; "
        f"reason={queue_key}_queue_wait_exceeded.\n"
    )


def _resource_call(observer: Any | None, method: str, *args: Any, **kwargs: Any) -> Any:
    if observer is None:
        return None
    fn = getattr(observer, method, None)
    if fn is None:
        return None
    if kwargs:
        try:
            sig = inspect.signature(fn)
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_var_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            pass
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.debug("[bash-resource] observer.%s failed", method, exc_info=True)
        return None


def _bash_preflight(
    tool: "BashTool",
    command: str,
) -> tuple[BashPreflight | None, ToolResult | None]:
    """Normalize, safety-check, and freeze environment inputs before admission."""

    raw_cmd = (command or "").strip()
    cmd, pip_rewritten = normalize_bash_command_for_agent(raw_cmd)
    if pip_rewritten:
        logger.info(
            "[bash-normalize] reason=pip_rewritten_to_uv_pip before=%r after=%r",
            raw_cmd,
            cmd,
        )
    if not cmd:
        return None, ToolResult(error="Empty command")

    guard_error = _bash_guard_error(tool, cmd)
    if guard_error:
        return None, ToolResult(error=guard_error)

    sleep_prefix = (
        _parse_leading_sleep_command(cmd)
        if tool.resource_observer is not None
        else None
    )
    if sleep_prefix is not None:
        wait_instruction = _resource_call(
            tool.resource_observer,
            "resource_wait_instruction_for_bash_sleep",
            command=cmd,
            planned_sleep_sec=float(sleep_prefix[0]),
        )
        if isinstance(wait_instruction, dict) and wait_instruction.get("available"):
            feedback = str(wait_instruction.get("feedback") or "").strip()
            if not feedback:
                feedback = (
                    "RESOURCE_FEEDBACK: RESOURCE_WAIT_REQUIRED because use resource_wait tool instead of bash sleep; "
                    "wait_tool=resource_wait; bash_sleep_allowed=false; retry_allowed=false.\n"
                )
            return None, ToolResult(output=feedback, error="Use resource_wait tool")

    timeout = _infer_timeout(
        sleep_prefix[1] if sleep_prefix and sleep_prefix[1] else cmd,
        float(tool.bash_timeout_sec),
        float(tool.bash_timeout_slow_sec),
    )
    timeout = tool._apply_hard_fuse_timeout(timeout)
    workspace_dir = Path(tool.workspace_dir).resolve()
    started_at = time.time()

    # The subprocess must use the same Python environment as ScienceFlow.
    environment: dict[str, str] = {**os.environ, **make_path_env()}
    if tool.extra_env:
        environment = {**environment, **tool.extra_env}
    task_physical_gpu_pool = _parse_visible_gpu_ids(
        environment.get("SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL")
        or environment.get("SCIENCEFLOW_RESOLVED_CUDA_DEVICES")
        or environment.get("CUDA_VISIBLE_DEVICES")
    )
    gpu_ids = _parse_visible_gpu_ids(environment.get("CUDA_VISIBLE_DEVICES"))
    lease_assignment = bool(
        _resource_call(tool.resource_observer, "lease_assignment_enabled")
    )
    if (
        lease_assignment
        and "CUDA_VISIBLE_DEVICES" in environment
        and not _is_gpu_visibility_probe(cmd)
    ):
        # Worker visibility is a pool hint until admission grants a concrete lease.
        environment["CUDA_VISIBLE_DEVICES"] = ""

    return (
        BashPreflight(
            command=cmd,
            sleep_prefix_command=cmd,
            sleep_prefix=sleep_prefix,
            timeout_sec=timeout,
            workspace_dir=workspace_dir,
            environment=environment,
            task_physical_gpu_pool=task_physical_gpu_pool,
            gpu_ids=gpu_ids,
            started_at=started_at,
        ),
        None,
    )


def _kill_revalidation_allows_termination(result: Any, *, hard_safety: bool) -> bool:
    if hard_safety:
        return True
    return bool(
        isinstance(result, dict)
        and result.get("enabled") is True
        and result.get("allow_kill") is True
    )


async def _resource_async_call(
    observer: Any | None, method: str, *args: Any, **kwargs: Any
) -> Any:
    result = _resource_call(observer, method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _resource_policy_preflight_result(
    observer: Any | None,
    job_id: str | None,
    *,
    inferred_class: str,
    gpu_ids: list[str],
    elapsed_sec: float,
    resource_context_version: int | None = None,
    resource_pressure_generation: int | None = None,
) -> ToolResult | None:
    if not job_id:
        return None
    decision = _resource_call(
        observer,
        "resource_preflight_decision",
        job_id,
        inferred_class=inferred_class,
        gpu_ids=gpu_ids,
        resource_context_version=resource_context_version,
        resource_pressure_generation=resource_pressure_generation,
    )
    if not isinstance(decision, dict) or decision.get("allowed", True):
        return None
    if decision.get("requires_admission_review"):
        reviewed = await _maybe_review_resource_admission(observer, job_id, decision)
        if isinstance(reviewed, dict):
            blocked_result = _resource_admission_tool_result(
                observer,
                job_id,
                reviewed,
                elapsed_sec=float(elapsed_sec or 0.0),
            )
            if blocked_result is not None:
                return blocked_result
            status = str(reviewed.get("status") or "").upper()
            if (
                reviewed.get("acquired")
                or status not in _RESOURCE_ADMISSION_BLOCKED_STATUSES
            ):
                return None
            decision = reviewed
    feedback = str(decision.get("feedback") or "").strip()
    feedback_suppressed = bool(decision.get("feedback_suppressed"))
    if not feedback and not feedback_suppressed:
        feedback = f"RESOURCE_FEEDBACK: resource_policy_blocked because {decision.get('reason') or 'resource_policy_gate'}.\n"
    error = str(decision.get("error") or "Resource policy blocked command")
    if feedback_suppressed:
        error = ""
    _resource_call(
        observer,
        "job_finished",
        job_id,
        status="resource_policy_blocked",
        elapsed_sec=float(elapsed_sec or 0.0),
        reason=str(decision.get("reason") or "resource_policy_gate"),
    )
    return ToolResult(output=feedback, error=error or None)


_RESOURCE_ADMISSION_BLOCKED_STATUSES = {
    "PENDING",
    "REPLAN",
    "DEFERRED",
    "DENIED_REPLAN",
    "DENIED_DUPLICATE",
}


async def _maybe_review_resource_admission(
    observer: Any | None,
    job_id: str | None,
    decision: dict[str, Any],
) -> dict[str, Any]:
    if (
        str(decision.get("status") or "").upper()
        not in _RESOURCE_ADMISSION_BLOCKED_STATUSES
    ):
        return decision
    share_review = await _resource_async_call(
        observer,
        "admission_share_decide",
        job_id,
        admission_result=decision,
    )
    if (
        isinstance(share_review, dict)
        and share_review.get("enabled")
        and isinstance(share_review.get("result"), dict)
    ):
        decision = dict(share_review["result"])
        if (
            decision.get("acquired")
            or str(decision.get("status") or "").upper()
            not in _RESOURCE_ADMISSION_BLOCKED_STATUSES
        ):
            return decision
        if decision.get("admission_llm_reviewed"):
            return decision
    review = await _resource_async_call(
        observer,
        "admission_decide",
        job_id,
        admission_result=decision,
    )
    if (
        isinstance(review, dict)
        and review.get("enabled")
        and isinstance(review.get("result"), dict)
    ):
        return dict(review["result"])
    return decision


def _resource_admission_tool_result(
    observer: Any | None,
    job_id: str | None,
    decision: dict[str, Any],
    *,
    elapsed_sec: float,
) -> ToolResult | None:
    status_upper = str(decision.get("status") or "").upper()
    if (
        decision.get("acquired")
        or status_upper not in _RESOURCE_ADMISSION_BLOCKED_STATUSES
    ):
        return None
    wait_option = (
        decision.get("resource_wait_option")
        if isinstance(decision.get("resource_wait_option"), dict)
        else {}
    )
    wait_offered = bool(wait_option.get("offered"))
    if (
        status_upper == "PENDING"
        and not decision.get("admission_llm_reviewed")
        and not wait_offered
    ):
        return None
    feedback = str(decision.get("feedback") or "").strip()
    feedback_suppressed = bool(decision.get("feedback_suppressed"))
    if not feedback and feedback_suppressed:
        feedback = ""
    elif not feedback:
        feedback = f"RESOURCE_FEEDBACK: {str(decision.get('status') or 'PENDING')} because {str(decision.get('reason') or 'gpu_slot_unavailable')}.\n"
    status = str(decision.get("status") or "PENDING").lower()
    _resource_call(
        observer,
        "job_finished",
        job_id,
        status=f"resource_admission_{status}",
        elapsed_sec=max(0.0, float(elapsed_sec or 0.0)),
        reason=str(decision.get("reason") or "gpu_slot_unavailable"),
    )
    return ToolResult(
        output=feedback,
        error=(None if feedback_suppressed else "Resource admission pending"),
    )


async def _wait_for_resource_queue(
    observer: Any | None,
    job_id: str | None,
    *,
    inferred_class: str,
    gpu_ids: list[str],
    on_output: Callable[[str], None] | None,
) -> ToolResult | None:
    decision = _resource_call(
        observer,
        "queue_try_acquire",
        job_id,
        inferred_class=inferred_class,
        gpu_ids=gpu_ids,
    )
    if not isinstance(decision, dict):
        return None
    if not decision.get("enabled") or decision.get("acquired", True):
        return None
    decision = await _maybe_review_resource_admission(observer, job_id, decision)
    blocked_result = _resource_admission_tool_result(
        observer, job_id, decision, elapsed_sec=0.0
    )
    if blocked_result is not None:
        return blocked_result

    queue_label, queue_key = _queue_label_for_resource_class(inferred_class)
    max_wait_sec = max(0.0, float(decision.get("max_wait_sec") or 0.0))
    heartbeat_sec = max(1.0, float(decision.get("heartbeat_sec") or 15.0))
    wait_start = time.time()
    last_emit_m = 0.0
    heartbeat_active = False
    try:
        while True:
            elapsed = time.time() - wait_start
            if max_wait_sec > 0 and elapsed >= max_wait_sec:
                reason = f"{queue_key}_queue_wait_exceeded_{max_wait_sec:.0f}s"
                _resource_call(
                    observer,
                    "queue_timeout",
                    job_id,
                    elapsed_sec=elapsed,
                    reason=reason,
                )
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                block = (
                    f"[queue_timeout, {elapsed:.1f}s]\n"
                    f"{queue_label} queue wait exceeded after {max_wait_sec:.0f}s"
                    f"\n{_resource_queue_feedback(elapsed, max_wait_sec, queue_label=queue_label, queue_key=queue_key)}"
                )
                return ToolResult(
                    output=block, error=f"{queue_label} queue wait exceeded"
                )

            now_m = time.monotonic()
            if on_output is not None and now_m - last_emit_m >= heartbeat_sec:
                on_output(
                    f"\r\033[K[bash queued] waiting for {queue_label.lower()} slot, "
                    f"{elapsed:.0f}s elapsed]",
                )
                heartbeat_active = True
                last_emit_m = now_m
            _resource_call(
                observer,
                "queue_wait_heartbeat",
                job_id,
                elapsed_sec=elapsed,
            )
            await asyncio.sleep(
                min(0.25, max(0.05, max_wait_sec - elapsed if max_wait_sec else 0.25))
            )
            decision = _resource_call(
                observer,
                "queue_try_acquire",
                job_id,
                inferred_class=inferred_class,
                gpu_ids=gpu_ids,
            )
            if not isinstance(decision, dict):
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                return None
            decision = await _maybe_review_resource_admission(
                observer, job_id, decision
            )
            blocked_result = _resource_admission_tool_result(
                observer, job_id, decision, elapsed_sec=time.time() - wait_start
            )
            if blocked_result is not None:
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                return blocked_result
            if not decision.get("enabled") or decision.get("acquired", True):
                if on_output is not None and heartbeat_active:
                    on_output("\r\033[K")
                return None
    except asyncio.CancelledError:
        elapsed = time.time() - wait_start
        _resource_call(
            observer,
            "queue_timeout",
            job_id,
            elapsed_sec=elapsed,
            reason="cancelled_while_queued",
        )
        raise


__all__ = tuple(name for name in globals() if not name.startswith("__"))
