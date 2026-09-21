"""Stable ParallelRunner facade over manifest, scheduler, subprocess, and state components."""

from __future__ import annotations

import time as time

from scienceflow.runtime.parallel.config.models import TaskResult as TaskResult, TaskSpec as TaskSpec
from scienceflow.runtime.parallel.execution.base import resolve_project_python as resolve_project_python
from scienceflow.runtime.parallel.config.manifest import (
    _env_sticky_primary_index as _env_sticky_primary_index,
    _manifest_input_data_dir as _manifest_input_data_dir,
    _manifest_merge_lnr as _manifest_merge_lnr,
    _manifest_run_type as _manifest_run_type,
    _manifest_scienceflow_interaction_log_full as _manifest_scienceflow_interaction_log_full,
    _manifest_scienceflow_interaction_log_level as _manifest_scienceflow_interaction_log_level,
    _manifest_scienceflow_interaction_log_llm_stream as _manifest_scienceflow_interaction_log_llm_stream,
    _manifest_task_exp_id as _manifest_task_exp_id,
    _manifest_task_run_id as _manifest_task_run_id,
    _resolve_parallel_task_text as _resolve_parallel_task_text,
    _resolve_task_workspace as _resolve_task_workspace,
    _safe_filename as _safe_filename,
    _scienceflow_repo_root as _scienceflow_repo_root,
    _validate_parallel_manifest as _validate_parallel_manifest,
    resolve_manifest_task_workspace as resolve_manifest_task_workspace,
)
from scienceflow.runtime.parallel.config.preparation import (
    _parallel_failure_kind_from_text as _parallel_failure_kind_from_text,
    _workspace_lnr_failure_kind as _workspace_lnr_failure_kind,
    _workspace_lnr_result_status as _workspace_lnr_result_status,
    _write_gpu_assignment_json as _write_gpu_assignment_json,
)
from scienceflow.runtime.parallel.execution import construction
from scienceflow.runtime.parallel.execution import scheduler
from scienceflow.runtime.parallel.execution import subprocess
from scienceflow.runtime.parallel.state import resume
from scienceflow.runtime.parallel.state import lifecycle as state
from scienceflow.runtime.parallel.state import result

class ParallelRunner:
    """Launch multiple ScienceFlow tasks as isolated subprocesses."""
    __init__ = construction.__init__
    _load_manifest = construction._load_manifest
    resume_enabled = scheduler.resume_enabled
    run_all = scheduler.run_all
    _task_subprocess_log_file = scheduler._task_subprocess_log_file
    _run_task = scheduler._run_task
    child_cli_argv = subprocess.child_cli_argv
    _exec_subprocess = subprocess._exec_subprocess
    _check_resume = resume._check_resume
    _logical_resume_budget_sec = resume._logical_resume_budget_sec
    _with_resume_remaining_budget = resume._with_resume_remaining_budget
    _write_interrupted_states = resume._write_interrupted_states
    _write_interrupted_state = state._write_interrupted_state
    _write_running_state = state._write_running_state
    _write_state = state._write_state
    format_summary = result.format_summary
