"""Explicit ordered helper facade for parallel execution."""

# Compatibility re-exports are intentionally explicit.
# ruff: noqa: F401

from scienceflow.runtime.parallel.config.manifest import (
    _format_cpu_set_compact,
    _positive_int_or_none,
    _VALID_PHASES,
    _BUDGET_DONE_STATUS,
    _scienceflow_repo_root,
    _resolve_parallel_log_dir_override,
    _resolve_parallel_task_text,
    _safe_filename,
    _load_json_file,
    _manifest_string_list,
    _manifest_string_csv,
    _env_key_pool_size,
    _env_sticky_primary_index,
    _manifest_endpoint_pairs,
    _rotate_endpoint_pairs_for_primary,
    _manifest_task_run_id,
    _resolve_task_workspace,
    resolve_manifest_task_workspace,
    _manifest_task_exp_id,
    _manifest_input_data_dir,
    _manifest_scienceflow_interaction_log_full,
    _manifest_scienceflow_interaction_log_color,
    _manifest_scienceflow_interaction_log_llm_stream,
    _manifest_scienceflow_interaction_log_level,
    _manifest_run_type,
    _manifest_merge_lnr,
    _manifest_merge_agent,
    _validate_parallel_manifest,
)

from scienceflow.runtime.parallel.config.preparation import (
    _task_state_log_dir,
    _parallel_state_file_candidates,
    _parallel_state_file_for_read,
    _ensure_workspace_for_state_write,
    _write_gpu_assignment_json,
    _read_resolved_gpu_from_assignment,
    _EXTERNAL_LLM_FAILURE_KINDS,
    _parallel_failure_kind_from_text,
    _read_log_tail,
    _parallel_failure_kind,
    _charged_elapsed_for_resume,
    _workspace_lnr_event_files,
    _workspace_lnr_done_payload,
    _workspace_lnr_result_status,
    _workspace_lnr_failure_kind,
    _state_completed_but_child_failed,
)

from scienceflow.runtime.parallel.config.models import TaskResult, TaskSpec

__all__ = tuple(name for name in globals() if not name.startswith("__"))
