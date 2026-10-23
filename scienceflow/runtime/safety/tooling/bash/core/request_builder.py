"""Bash responsibility: request builder."""

from __future__ import annotations

from scienceflow.runtime.safety.tooling.bash.core.shared import (
    Any,
    BashAdmission,
    Callable,
    Path,
    ToolResult,
    _build_gpu_boundary_feedback,
    _logical_cuda_ordinals_for_assignment,
    _normalize_cuda_visible_devices_for_task_pool,
    _parse_resource_context_version,
    _parse_resource_pressure_generation,
    _parse_resource_value_hint,
    _replace_leading_cuda_visible_devices,
    classify_bash_command,
    logger,
    re,
    time,
)
from scienceflow.runtime.safety.tooling.bash.policy.admission import (
    _resource_call,
    _resource_policy_preflight_result,
    _wait_for_resource_queue,
)
from scienceflow.runtime.safety.tooling.bash.process.streaming import (
    _observe_unknown_command,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scienceflow.runtime.safety.tooling.bash.core.tool import BashTool


def _create_resource_job_scope(
    tool: "BashTool",
    *,
    command: str,
    environment: dict[str, str],
    classified: Any,
    gpu_ids: list[str],
    cpu_set: str,
    timeout_sec: float,
    workspace_dir: Path,
) -> tuple[str | None, Path | None, Path | None, Path | None]:
    job_id = _resource_call(
        tool.resource_observer,
        "job_created",
        command=command,
        inferred_class=classified.resource_class,
        gpu_ids=gpu_ids,
        cpu_set=cpu_set or None,
        timeout_sec=timeout_sec,
        workspace_dir=workspace_dir,
        classifier_reason=classified.reason,
        value_hint_override=_parse_resource_value_hint(command, environment),
        candidate_artifact=str(environment.get("SCIENCEFLOW_CANDIDATE_ARTIFACT") or ""),
    )
    if not job_id:
        return None, None, None, None
    safe_job_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(job_id)).strip("_") or "job"
    run_dir = workspace_dir / ".scienceflow_runs" / safe_job_id
    artifact_dir = run_dir / "artifacts"
    state_path = run_dir / "run_state.json"
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("[bash-resource] failed to create run artifact dir", exc_info=True)
        return str(job_id), None, None, None
    environment.update(
        {
            "SCIENCEFLOW_RUN_ID": str(job_id),
            "SCIENCEFLOW_RUN_DIR": str(run_dir),
            "SCIENCEFLOW_RUN_ARTIFACT_DIR": str(artifact_dir),
            "SCIENCEFLOW_RUN_STATE_PATH": str(state_path),
            "SCIENCEFLOW_RECOVERABLE_STOP": "1",
            "SCIENCEFLOW_STOP_SIGNAL": "SIGUSR1",
        }
    )
    _resource_call(
        tool.resource_observer,
        "recoverable_artifact_scope_registered",
        job_id,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        run_state_path=state_path,
    )
    return str(job_id), run_dir, artifact_dir, state_path


def _planning_context(
    tool: "BashTool",
    command: str,
    environment: dict[str, str],
) -> tuple[int | None, int | None]:
    context_version = _parse_resource_context_version(command, environment)
    if context_version is None:
        raw_version = _resource_call(
            tool.resource_observer,
            "planning_resource_context_version",
        )
        try:
            context_version = int(raw_version) if raw_version is not None else None
        except (TypeError, ValueError):
            context_version = None
    pressure_generation = _parse_resource_pressure_generation(command, environment)
    if pressure_generation is None:
        raw_generation = _resource_call(
            tool.resource_observer,
            "planning_resource_pressure_generation",
        )
        try:
            pressure_generation = (
                int(raw_generation) if raw_generation is not None else None
            )
        except (TypeError, ValueError):
            pressure_generation = None
    return context_version, pressure_generation


def _apply_lease_environment(
    tool: "BashTool",
    *,
    job_id: str | None,
    command: str,
    environment: dict[str, str],
    gpu_ids: list[str],
    task_physical_gpu_pool: list[str],
    started_at: float,
) -> tuple[str, list[str], ToolResult | None]:
    updates = _resource_call(tool.resource_observer, "lease_env_updates", job_id)
    if not isinstance(updates, dict):
        return command, gpu_ids, None
    clean_updates = {
        str(key): str(value)
        for key, value in updates.items()
        if str(key).strip() and value is not None
    }
    if not clean_updates:
        return command, gpu_ids, None
    if "CUDA_VISIBLE_DEVICES" not in clean_updates:
        environment.update(clean_updates)
        return command, gpu_ids, None
    leased_cuda = clean_updates["CUDA_VISIBLE_DEVICES"]
    normalized_cuda, normalized_gpu_ids, remapped, reason = (
        _normalize_cuda_visible_devices_for_task_pool(
            leased_cuda,
            task_physical_gpu_pool,
        )
    )
    clean_updates["CUDA_VISIBLE_DEVICES"] = normalized_cuda
    clean_updates.setdefault("SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL", normalized_cuda)
    clean_updates.setdefault(
        "SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL",
        _logical_cuda_ordinals_for_assignment(normalized_gpu_ids),
    )
    if task_physical_gpu_pool:
        clean_updates.setdefault(
            "SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL",
            ",".join(task_physical_gpu_pool),
        )
    if remapped:
        clean_updates["SCIENCEFLOW_CUDA_VISIBLE_NORMALIZED_FROM"] = leased_cuda
    if (
        normalized_gpu_ids
        and task_physical_gpu_pool
        and not set(normalized_gpu_ids).issubset(set(task_physical_gpu_pool))
    ):
        violation_reason = "task_gpu_boundary_preflight_violation"
        feedback = (
            _build_gpu_boundary_feedback(
                actual_gpu_ids=normalized_gpu_ids,
                allowed_gpu_ids=task_physical_gpu_pool,
                command_scope="lease",
            )
            + "\n"
        )
        _resource_call(
            tool.resource_observer,
            "resource_guard_action",
            job_id,
            action="stop_boundary_violation",
            reason=violation_reason,
            elapsed_sec=max(0.0, time.time() - started_at),
            placement={
                "allowed_gpu_ids": task_physical_gpu_pool,
                "leased_cuda_visible_devices": leased_cuda,
                "normalized_cuda_visible_devices": normalized_cuda,
                "normalize_reason": reason,
            },
        )
        _resource_call(
            tool.resource_observer,
            "job_finished",
            job_id,
            status="resource_boundary_violation",
            elapsed_sec=max(0.0, time.time() - started_at),
            reason=violation_reason,
        )
        return (
            command,
            gpu_ids,
            ToolResult(
                output=feedback,
                error="Resource boundary violation",
            ),
        )
    environment.update(clean_updates)
    return (
        _replace_leading_cuda_visible_devices(command, normalized_cuda),
        normalized_gpu_ids,
        None,
    )


async def _admit_bash_command(
    tool: "BashTool",
    *,
    command: str,
    environment: dict[str, str],
    gpu_ids: list[str],
    task_physical_gpu_pool: list[str],
    cpu_set: str,
    timeout_sec: float,
    workspace_dir: Path,
    started_at: float,
    on_output: Callable[[str], None] | None,
) -> tuple[BashAdmission | None, ToolResult | None]:
    """Classify a command, apply policy, queue it, and materialize its lease."""

    cmd = command
    classified = classify_bash_command(cmd)
    (
        resource_job_id,
        run_dir,
        run_artifact_dir,
        run_state_path,
    ) = _create_resource_job_scope(
        tool,
        command=cmd,
        environment=environment,
        classified=classified,
        gpu_ids=gpu_ids,
        cpu_set=cpu_set,
        timeout_sec=timeout_sec,
        workspace_dir=workspace_dir,
    )

    effective_resource_class = classified.resource_class
    resource_context_version, resource_pressure_generation = _planning_context(
        tool,
        cmd,
        environment,
    )
    policy_result = await _resource_policy_preflight_result(
        tool.resource_observer,
        resource_job_id,
        inferred_class=effective_resource_class,
        gpu_ids=gpu_ids,
        elapsed_sec=time.time() - started_at,
        resource_context_version=resource_context_version,
        resource_pressure_generation=resource_pressure_generation,
    )
    if policy_result is not None:
        return None, policy_result

    promoted_class, observation_result = await _observe_unknown_command(
        observer=tool.resource_observer,
        job_id=resource_job_id,
        inferred_class=classified.resource_class,
        gpu_ids=gpu_ids,
        cmd=cmd,
        workspace_dir=workspace_dir,
        proc_env=environment,
        stream_limit=int(tool.subprocess_stream_limit),
        on_output=on_output,
    )
    if observation_result is not None:
        return None, observation_result
    if promoted_class:
        effective_resource_class = promoted_class
        policy_result = await _resource_policy_preflight_result(
            tool.resource_observer,
            resource_job_id,
            inferred_class=effective_resource_class,
            gpu_ids=gpu_ids,
            elapsed_sec=time.time() - started_at,
            resource_context_version=resource_context_version,
            resource_pressure_generation=resource_pressure_generation,
        )
        if policy_result is not None:
            return None, policy_result

    queue_result = await _wait_for_resource_queue(
        tool.resource_observer,
        resource_job_id,
        inferred_class=effective_resource_class,
        gpu_ids=gpu_ids,
        on_output=on_output,
    )
    if queue_result is not None:
        return None, queue_result

    cmd, gpu_ids, lease_error = _apply_lease_environment(
        tool,
        job_id=resource_job_id,
        command=cmd,
        environment=environment,
        gpu_ids=gpu_ids,
        task_physical_gpu_pool=task_physical_gpu_pool,
        started_at=started_at,
    )
    if lease_error is not None:
        return None, lease_error

    return (
        BashAdmission(
            command=cmd,
            environment=environment,
            gpu_ids=gpu_ids,
            classified=classified,
            effective_resource_class=effective_resource_class,
            resource_job_id=(
                str(resource_job_id) if resource_job_id is not None else None
            ),
            run_dir=run_dir,
            run_artifact_dir=run_artifact_dir,
            run_state_path=run_state_path,
        ),
        None,
    )


__all__ = tuple(name for name in globals() if not name.startswith("__"))
