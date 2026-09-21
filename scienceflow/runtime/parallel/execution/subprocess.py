"""ParallelRunner responsibility: subprocess."""

from __future__ import annotations

import time
from pathlib import Path

from scienceflow.runtime.parallel.execution.base import resolve_project_python
from scienceflow.runtime.parallel.config.models import TaskSpec
from scienceflow.runtime.parallel.state.subprocess_runtime import (
    _apply_endpoint_env,
    _apply_resource_env,
    _build_child_env,
    _build_parallel_command,
    _cleanup_subprocess_resources,
    _finalize_child_env,
    _record_subprocess_start,
    _resolve_gpu_assignment,
    _run_child,
)


@staticmethod
def child_cli_argv(spec: TaskSpec) -> list[str]:
    """Build ``python -m scienceflow.interfaces.cli …`` argv (no ``taskset`` prefix); used by tests and `_exec_subprocess`."""
    child_python, _child_bin = resolve_project_python()
    argv: list[str] = [
        child_python,
        "-m",
        "scienceflow.interfaces.cli",
    ]
    if spec.phase == "prep":
        argv.extend(
            [
                "prep",
                "--task",
                spec.task,
                "--workspace",
                spec.workspace,
                "--input-data-dir",
                str(Path(spec.input_data_dir).expanduser()),
            ],
        )
        if spec.config:
            argv.extend(["--config", spec.config])
    else:
        argv.extend(
            [
                "run",
                "--task",
                spec.task,
                "--workspace",
                spec.workspace,
                "--type",
                spec.type,
            ],
        )
        if spec.config:
            argv.extend(["--config", spec.config])
        if spec.input_data_dir.strip():
            argv.extend(
                [
                    "--input-data-dir",
                    str(Path(spec.input_data_dir).expanduser()),
                ],
            )
    return argv

async def _exec_subprocess(self, spec: TaskSpec, log_file: str) -> tuple[int, str]:
    command, cpu_list = _build_parallel_command(spec, self.child_cli_argv(spec))
    env = _build_child_env(spec, cpu_list)
    resolved_gpu, assigned_gpus, gpu_disabled = await _resolve_gpu_assignment(self, spec, env)
    _apply_resource_env(
        env,
        spec,
        resolved_gpu=resolved_gpu,
        gpu_disabled=gpu_disabled,
        effective_cpu_list=cpu_list,
    )
    inherited_pool = _apply_endpoint_env(env, spec)
    _finalize_child_env(env, spec)
    _record_subprocess_start(
        self,
        spec,
        env=env,
        resolved_gpu=resolved_gpu,
        inherited_pool=inherited_pool,
        log_file=log_file,
    )
    task_started_at = time.time()
    try:
        return_code = await _run_child(command, env, spec, log_file)
    finally:
        await _cleanup_subprocess_resources(
            self,
            spec,
            resolved_gpu=resolved_gpu,
            gpu_disabled=gpu_disabled,
            assigned_auto_gpus=assigned_gpus,
            task_started_at=task_started_at,
            log_file=log_file,
        )
    return return_code, resolved_gpu
