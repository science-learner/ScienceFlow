"""Shared Bash tool imports, immutable values, and request records."""

from __future__ import annotations
import asyncio
import hashlib
import inspect
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from inquirycraft.tools import ToolResult
from scienceflow.runtime.core.process import (
    InquiryCraftProcessAdapter,
    ProcessExecutionSession,
    ProcessRequest,
    ProcessSpawnError,
)
from scienceflow.runtime.core.process.utils import (
    spawn_shell as _spawn_shell_impl,
    terminate_process_tree as terminate_tree,
    terminate_process_tree_recoverable as terminate_tree_recoverable,
    unregister_process_group,
)
from scienceflow.runtime.safety.tooling.workspace.shell_output import (
    _dedup_repeated_blocks,
    _distill_tracebacks,
    _has_masked_python_traceback,
    _maybe_lossless_observation_summary,
    _sanitize_model_visible_output_paths,
    _trim_output,
)
from inquirycraft.tools import ShellGuardRule, evaluate_shell_guards
from scienceflow.runtime.environment import make_path_env
from scienceflow.runtime.safety.tooling.resource_management.resource_policy import (
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
    classify_bash_command,
)
from scienceflow.runtime.safety.tooling.workspace.shadow_workspace import ShadowWorkspaceManager
from scienceflow.research.solver.lnr.resources.runtime.execution.state.utilization import (
    logical_cuda_ordinals_for_assignment as _logical_cuda_ordinals_for_assignment,
    normalize_cuda_visible_devices_for_task_pool as _normalize_cuda_visible_devices_for_task_pool,
    parse_visible_gpu_ids as _parse_visible_gpu_ids,
    process_tree_gpu_placement_snapshot as _process_tree_gpu_placement_snapshot_impl,
    replace_leading_cuda_visible_devices as _replace_leading_cuda_visible_devices,
)
from scienceflow.research.solver.lnr.resources.runtime.control.gpu.gpu_feedback import (
    build_gpu_boundary_feedback as _build_gpu_boundary_feedback,
)
from scienceflow.research.solver.lnr.resources.runtime.control.gpu.workspace_gpu_guard import (
    cleanup_workspace_gpu_processes as _cleanup_workspace_gpu_processes_impl,
)
from scienceflow.runtime.safety.tooling.workspace.shell_guards import (
    _command_executes_under_readonly_dir,
    _has_silent_redirect,
    _infer_timeout,
    _leading_cd_abs_path_missing,
    background_resource_command_blocked_error,
    dangerous_delete_command_blocked_error,
    global_filesystem_scan_blocked_error,
    hidden_workspace_path_usage_blocked_error,
    interactive_stdin_blocked_error,
    mixed_file_write_execution_blocked_error,
    normalize_bash_command_for_agent,
    privilege_escalation_blocked_error,
    process_control_blocked_error,
    shared_python_env_write_blocked_error,
    truncated_resource_output_blocked_error,
    workspace_scope_path_blocked_error,
)
from scienceflow.runtime.safety.tooling.resource_management.signals import (
    _is_gpu_visibility_probe,
    _parse_leading_sleep_command,
    _parse_progress_signals,
    _parse_resource_context_version,
    _parse_resource_pressure_generation,
    _parse_resource_value_hint,
)
from scienceflow.runtime.safety.tooling.resource_management.spawn_feedback import build_spawn_failure_tool_result
logger = logging.getLogger("scienceflow")


def _public_override(name: str, default: Any) -> Any:
    """Resolve a supported patch seam from the public package."""

    public_module = sys.modules.get("scienceflow.runtime.safety.tooling.bash")
    candidate = getattr(public_module, name, None)
    return candidate if candidate is not None and candidate is not default else None


async def spawn_shell(*args: Any, **kwargs: Any) -> Any:
    override = _public_override("spawn_shell", spawn_shell)
    if override is not None:
        return await override(*args, **kwargs)
    return await _spawn_shell_impl(*args, **kwargs)


def _process_tree_gpu_placement_snapshot(*args: Any, **kwargs: Any) -> Any:
    override = _public_override(
        "_process_tree_gpu_placement_snapshot",
        _process_tree_gpu_placement_snapshot,
    )
    if override is not None:
        return override(*args, **kwargs)
    return _process_tree_gpu_placement_snapshot_impl(*args, **kwargs)


def _cleanup_workspace_gpu_processes(*args: Any, **kwargs: Any) -> Any:
    override = _public_override(
        "_cleanup_workspace_gpu_processes",
        _cleanup_workspace_gpu_processes,
    )
    if override is not None:
        return override(*args, **kwargs)
    return _cleanup_workspace_gpu_processes_impl(*args, **kwargs)
@dataclass(slots=True)
class BashPreflight:
    """Validated command and immutable process inputs for later phases."""

    command: str
    sleep_prefix_command: str
    sleep_prefix: tuple[float, str] | None
    timeout_sec: float
    workspace_dir: Path
    environment: dict[str, str]
    task_physical_gpu_pool: list[str]
    gpu_ids: list[str]
    started_at: float
@dataclass(slots=True)
class BashAdmission:
    """Command state after classification, policy review, queue, and lease."""

    command: str
    environment: dict[str, str]
    gpu_ids: list[str]
    classified: Any
    effective_resource_class: str
    resource_job_id: str | None
    run_dir: Path | None
    run_artifact_dir: Path | None
    run_state_path: Path | None
@dataclass(slots=True)
class BashExecutionStart:
    """Spawned process plus the monitor policy fixed at launch time."""

    session: ProcessExecutionSession
    observe_first_active: bool
    observe_first_window_sec: float
_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+.*(-\w*f\w*|--force|--recursive)\b.*\s/\s*$"),
    re.compile(r"\brm\s+.*\s/\s*$"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*\bof=/dev/"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),  # fork bomb
]

__all__ = (
    "Any",
    "BashAdmission",
    "BashExecutionStart",
    "BashPreflight",
    "Callable",
    "InquiryCraftProcessAdapter",
    "Path",
    "ProcessExecutionSession",
    "ProcessRequest",
    "ProcessSpawnError",
    "RESOURCE_GPU_FEATURE_EXTRACT",
    "RESOURCE_GPU_LIGHT_TRAIN",
    "RESOURCE_GPU_TT_LIGHT",
    "RESOURCE_HEAVY_GPU_CANDIDATE",
    "RESOURCE_HEAVY_GPU_TRAIN",
    "RESOURCE_UNKNOWN_GPU_EXEC",
    "ShadowWorkspaceManager",
    "ShellGuardRule",
    "ToolResult",
    "_DANGEROUS_PATTERNS",
    "_build_gpu_boundary_feedback",
    "_cleanup_workspace_gpu_processes",
    "_command_executes_under_readonly_dir",
    "_dedup_repeated_blocks",
    "_distill_tracebacks",
    "_has_masked_python_traceback",
    "_has_silent_redirect",
    "_infer_timeout",
    "_is_gpu_visibility_probe",
    "_leading_cd_abs_path_missing",
    "_logical_cuda_ordinals_for_assignment",
    "_maybe_lossless_observation_summary",
    "_normalize_cuda_visible_devices_for_task_pool",
    "_parse_leading_sleep_command",
    "_parse_progress_signals",
    "_parse_resource_context_version",
    "_parse_resource_pressure_generation",
    "_parse_resource_value_hint",
    "_parse_visible_gpu_ids",
    "_process_tree_gpu_placement_snapshot",
    "_replace_leading_cuda_visible_devices",
    "_sanitize_model_visible_output_paths",
    "_trim_output",
    "asyncio",
    "background_resource_command_blocked_error",
    "build_spawn_failure_tool_result",
    "classify_bash_command",
    "dangerous_delete_command_blocked_error",
    "evaluate_shell_guards",
    "global_filesystem_scan_blocked_error",
    "hashlib",
    "hidden_workspace_path_usage_blocked_error",
    "inspect",
    "interactive_stdin_blocked_error",
    "logger",
    "make_path_env",
    "mixed_file_write_execution_blocked_error",
    "normalize_bash_command_for_agent",
    "os",
    "privilege_escalation_blocked_error",
    "process_control_blocked_error",
    "re",
    "shared_python_env_write_blocked_error",
    "spawn_shell",
    "terminate_tree",
    "terminate_tree_recoverable",
    "time",
    "truncated_resource_output_blocked_error",
    "unregister_process_group",
    "workspace_scope_path_blocked_error",
)
