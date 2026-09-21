"""Configuration responsibility: schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageConfig:
    model: str = "gpt-4o"
    # Canonical endpoint lists. Scalar fields are retained for compatibility
    # and projected to/from the first entry at configuration boundaries.
    models: list[str] = field(default_factory=list)
    # Optional aliases resolved through ~/.config/scienceflow/models.json.
    model_aliases: list[str] = field(default_factory=list)
    model_config_path: str = ""
    model_selection: str = ""
    temp: float = 0.7
    base_url: str = ""
    api_key: str = ""
    top_p: float = 0.95
    # Optional anti-repetition bias (OpenAI-compatible APIs).
    frequency_penalty: float | None = None
    max_tokens: int = 32768
    reasoning_effort: str | None = None
    api_keys: list[str] = field(default_factory=list)
    base_urls: list[str] = field(default_factory=list)
    api_routing_mode: str = "round_robin"
    api_sticky_id: str = ""
    api_sticky_primary_index: int | None = None
    api_rate_limit_cooldown_sec: float = 60.0
    api_connection_cooldown_sec: float = 15.0
    http_timeout: int = 300
    # Extra OpenAI client default_headers (e.g. gateway tokens); per code/feedback.
    headers: dict[str, str] = field(default_factory=dict)
    # When true, pass httpx.AsyncClient(proxy=..., verify=False, Proxy-Authorization from env).
    use_proxy: bool = False
    # Provider compatibility: vLLM chat templates may allow only one leading system message.
    coalesce_system_messages: bool = False
    # Provider-context replay policy: preserve | required | omit.
    # Empty inherits the selected model alias and ultimately defaults to preserve.
    reasoning_replay: str = ""

@dataclass
class AgentConfig:
    code: StageConfig = field(default_factory=StageConfig)
    feedback: StageConfig = field(default_factory=StageConfig)

@dataclass
class ExecConfig:
    fast_debug_max_samples: int = 100
    use_filtered: bool = False
    cpu_list: str = ""
    gpu_list: str = ""

@dataclass
class InitWorkspaceConfig:
    workspace_path: str = ""

@dataclass
class LnrConfig:
    """Config for the REPL-native long-horizon solver."""

    wall_clock_budget_sec: int = 3600
    max_steps: int = 200
    num_workers: int = 1
    seed: int = 0
    omp_threads_cap: int = 8
    code_organization_hint: str = "beyond_mfiles"
    init_workspace: InitWorkspaceConfig = field(default_factory=InitWorkspaceConfig)
    workspace_git_enabled: bool = True
    workspace_git_track_globs: list[str] = field(default_factory=lambda: ["*.py", "*.md"])
    workspace_git_initial_commit: bool = True
    workspace_git_auto_checkpoint: bool = True
    workspace_git_auto_review: bool = False
    ledger_filename: str = ".run_results.md"
    stage_capture_enabled: bool = True
    stage_capture_max_count: int = 0
    stage_commit_llm_timeout_sec: float = 180.0
    stage_commit_min_seconds_between: float = 20.0
    stage_commit_require_metric: bool = True
    stage_commit_text_mode: bool = True
    stage_commit_output_format: str = "text"
    stage_commit_context_mode: str = "compact"
    stage_commit_tool_choice: str = "none"
    stage_commit_experiment_state_enabled: bool = False
    stage_commit_persist_to_memory: bool = False
    stage_commit_persist_agent_write_to_memory: bool = True
    stage_commit_persist_prompt_to_memory: bool = False
    expose_runtime_context_each_round: bool = False
    metric_validity_adjudicator_enabled: bool = True
    metric_validity_adjudicator_timeout_sec: float = 60.0
    clean_repl_mode: bool = True
    metric_validation_leakage_guard_enabled: bool = True
    lnr_skill_tool_enabled: bool = False
    lnr_skill_tool_mode: str = "category_only"
    lnr_skill_category_source: str = "tasks/ml/mlebench/competition_categories.json"
    lnr_skill_category_label_field: str = "category_label"
    lnr_skill_auto_read: bool = False
    lnr_skill_auto_read_max_chars: int = 6000
    lnr_skill_visible_max: int = 1
    lnr_skill_allow_generic_wildcard: bool = False
    snapshot_dirname: str = ".snapshots"
    archive_dirname: str = "snapshots/archives"
    workspace_snapshot_enabled: bool = True
    workspace_snapshot_verify_objects: bool = True
    estra_enabled: bool = True
    estra_trigger_stage_count: int = 2
    force_estra_after_stage_count: int = 0
    force_estra_target_stage: str = ""
    force_estra_capture_duplicate_submissions: bool = False
    estra_use_main_agent_context: bool = True
    estra_peer_evidence_enabled: bool = True
    estra_peer_evidence_max_chars: int = 1400
    estra_peer_evidence_min_delta_ratio: float = 0.0
    estra_backtrack_reflection_enabled: bool = True
    estra_backtrack_reflection_max_chars: int = 1000
    estra_reflection_prompt_enabled: bool = True
    tail_summary_max_chars: int = 1200
    estra_compact_enabled: bool = True
    estra_compact_max_chars: int = 1200
    state_packet_max_chars: int = 12000
    state_packet_stage_card_max_chars: int = 420
    state_packet_archived_branch_max_chars: int = 1500
    stage_memory_folding_enabled: bool = True
    stage_memory_context_budget_chars: int = 24_000
    stage_memory_rebuild_on_stale: bool = True
    compact_on_context_limit: bool = True
    context_limit_min_messages: int = 5000
    preserve_prefix_and_eda: bool = True
    protected_eda_mode: str = "facts"
    protected_eda_facts_max_chars: int = 6000
    protected_eda_summary_max_chars: int = 8000
    protected_eda_warn_chars: int = 50_000
    worker_peer_summary_enabled: bool = True
    worker_peer_summary_max_chars: int = 2200
    merge_enabled: bool = True
    final_artifact_mode: str = "workspace"
    merge_mode: str = "worker_reduce"
    merge_owner_worker: str = "W00"
    merge_dirname: str = "merge"
    global_merge_wall_clock_sec: float = 900.0
    merge_prediction_file_max_bytes: int = 536_870_912
    merge_prediction_total_max_bytes: int = 2_147_483_648
    merge_required_finals: int = 3
    merge_max_finals: int = 3
    worker_dirname: str = "workers"
    resource_control_mode: str = "resource_smart_llm"
    resource_monitor_enabled: bool = True
    resource_control_profile: str = "normal"
    resource_startup_policy: str = "trial_first"
    resource_trial_window_sec: float = 300.0
    resource_trial_hard_review_sec: float = 900.0
    resource_bash_monitor_all_enabled: bool = True
    resource_bash_hard_fuse_finalization_reserve_sec: float = 900.0
    resource_monitor_min_register_sec: float = 600.0
    resource_monitor_check_interval_sec: float = 10.0
    resource_monitor_stalled_stdout_sec: float = 600.0
    resource_monitor_kill_enabled: bool = True
    resource_monitor_kill_mode: str = "arbiter"
    resource_monitor_agent_mode: str = "off"
    resource_monitor_agent_min_interval_sec: float = 60.0
    resource_monitor_low_progress_enabled: bool = True
    resource_monitor_low_progress_warmup_sec: float = 1800.0
    resource_monitor_low_progress_no_heartbeat_sec: float = 1800.0
    resource_monitor_low_progress_no_artifact_sec: float = 1800.0
    resource_review_state_enabled: bool = True
    resource_review_heartbeat_sec: float = 60.0
    resource_review_warmup_windows: int = 10
    resource_review_inactive_windows: int = 3
    resource_review_value_windows: int = 5
    resource_review_progress_event_min_windows: int = 5
    resource_review_timebox_windows: int = 10
    resource_review_max_proof_windows: int = 2
    resource_review_min_timebox_sec: float = 60.0
    resource_review_max_timebox_sec: float = 1800.0
    resource_review_timebox_budget_fraction: float = 0.10
    resource_progress_heartbeat_min_interval_sec: float = 30.0
    resource_artifact_heartbeat_scan_interval_sec: float = 30.0
    resource_artifact_recoverable_settle_sec: float = 5.0
    resource_recoverable_stop_enabled: bool = True
    resource_recoverable_stop_sigusr1_grace_sec: float = 60.0
    resource_recoverable_stop_marker_exit_grace_sec: float = 10.0
    resource_recoverable_stop_sigterm_grace_sec: float = 5.0
    resource_context_prompt_enabled: bool = True
    resource_runtime_enabled: bool = True
    resource_gpu_queue_enabled: bool = True
    resource_gpu_pool: list[str] = field(default_factory=list)
    resource_gpu_default_request: int = 1
    resource_gpu_max_request: int = 1
    resource_gpu_assignment: str = "lease"
    resource_gpu_queue_max_wait_sec: float = 1800.0
    resource_gpu_queue_heartbeat_sec: float = 15.0
    resource_gpu_max_heavy_per_gpu: int = 1
    resource_gpu_capacity_slots: float = 1.0
    resource_gpu_tt_max_per_gpu: int = 3
    resource_gpu_feature_max_per_gpu: int = 2
    resource_gpu_share_tt_with_train: bool = False
    resource_gpu_share_enabled: bool = True
    resource_gpu_share_phase: str = "observe"
    resource_gpu_share_policy_profile: str = "conservative"
    resource_gpu_share_memory_profile: str = "conservative"
    resource_gpu_share_cpu_policy: str = "conservative"
    resource_gpu_trial_admission_policy: str = "llm_grant"
    resource_gpu_lease_ttl_sec: float = 7200.0
    resource_gpu_duplicate_digest_cooldown_sec: float = 600.0
    resource_gpu_duplicate_digest_threshold: int = 2
    resource_gpu_admission_queue_enabled: bool = True
    resource_gpu_admission_waiter_ttl_sec: float = 900.0
    resource_admission_llm_enabled: bool = True
    resource_admission_llm_mode: str = "blocked_states"
    resource_admission_llm_timeout_sec: float = 60.0
    resource_admission_decision_source: str = "component"
    resource_execution_value_decision_source: str = "component"
    resource_stale_pressure_observe_first_enabled: bool = True
    resource_stale_pressure_observe_window_sec: float = 180.0
    resource_stale_pressure_observe_max_sec: float = 300.0
    resource_stale_pressure_observe_min_free_mem_gb: float = 8.0
    resource_stale_pressure_healthy_skip_llm: bool = True
    resource_gpu_pressure_yellow_hold_sec: float = 120.0
    resource_gpu_pressure_red_to_yellow_sec: float = 120.0
    resource_gpu_pressure_yellow_util_pct: float = 85.0
    resource_gpu_pressure_min_free_mem_gb: float = 8.0
    resource_gpu_pressure_yellow_free_mem_buffer_gb: float = 8.0
    resource_observation_enabled: bool = True
    resource_observation_window_sec: float = 30.0
    resource_observation_shadow_workspace_enabled: bool = True
    resource_gpu_source_hint_enabled: bool = True
    resource_gpu_source_hint_mode: str = "observe"
    resource_gpu_util_observer_enabled: bool = True
    resource_gpu_util_sample_interval_sec: float = 30.0
    resource_gpu_idle_lease_guard_enabled: bool = True
    resource_gpu_idle_lease_warmup_sec: float = 180.0
    resource_gpu_idle_lease_min_samples: int = 3
    resource_gpu_idle_lease_util_pct: float = 1.0
    resource_gpu_idle_lease_mem_gb: float = 1.0
    resource_gpu_idle_lease_require_pressure: bool = False
    resource_gpu_idle_lease_action_mode: str = "release"
    resource_idle_release_admission_mode: str = "strict_exclusive"
    resource_quick_probe_guard_enabled: bool = True
    resource_quick_probe_expected_runtime_sec: float = 300.0
    resource_quick_probe_hard_review_sec: float = 600.0
    resource_quick_probe_small_scope_threshold: int = 1000
    context_hygiene_compact_enabled: bool = True
    context_hygiene_max_stages_without_compact: int = 25
    context_hygiene_code_churn_stage_threshold: int = 10
    context_hygiene_large_file_repeat_threshold: int = 10
    context_hygiene_low_incremental_cache_rate: float = 0.80
    context_hygiene_low_cache_window: int = 5
    context_hygiene_min_tokens_since_compact: int = 250000
    context_hygiene_large_tool_output_chars: int = 12000
    context_hygiene_tool_output_min_interval_sec: float = 1800.0
    resource_gpu_dataloader_bottleneck_guard_enabled: bool = True
    resource_gpu_dataloader_bottleneck_warmup_sec: float = 600.0
    resource_gpu_dataloader_bottleneck_min_samples: int = 3
    resource_gpu_dataloader_bottleneck_util_pct: float = 15.0
    resource_gpu_dataloader_bottleneck_min_mem_gb: float = 2.0
    resource_gpu_dataloader_bottleneck_child_cpu_pct: float = 200.0
    resource_gpu_dataloader_bottleneck_busy_children: int = 2
    resource_deliverable_completion_guard_enabled: bool = True
    resource_deliverable_completion_warmup_sec: float = 120.0
    resource_deliverable_completion_settle_sec: float = 120.0
    resource_deliverable_completion_scan_interval_sec: float = 60.0
    resource_deliverable_completion_quiet_sec: float = 60.0
    resource_metric_health_guard_enabled: bool = True
    resource_metric_health_warmup_sec: float = 600.0
    resource_metric_health_invalid_min_events: int = 2
    resource_metric_health_zero_score_min_events: int = 2
    resource_arbiter_enabled: bool = True
    resource_arbiter_mode: str = "llm"
    resource_arbiter_timeout_sec: float = 90.0
    resource_arbiter_min_progress_windows: int = 2
    resource_arbiter_kill_requires_high_confidence: bool = True
    resource_main_agent_advisory_enabled: bool = True
    resource_main_agent_advisory_min_interval_sec: float = 600.0
    resource_main_agent_advisory_timeout_sec: float = 60.0
    resource_advisory_mode: str = "inline_memory_edit"
    resource_arbiter_contention_review_enabled: bool = False
    resource_arbiter_contention_min_runtime_sec: float = 900.0
    resource_arbiter_contention_min_waiter_age_sec: float = 300.0
    resource_arbiter_contention_min_interval_sec: float = 600.0
    resource_research_cadence_enabled: bool = True
    resource_research_cadence_observe_sec: float = 300.0
    resource_first_comparable_metric_budget_sec: float = 900.0
    resource_proven_route_metric_budget_sec: float = 1800.0
    resource_arbiter_periodic_review_enabled: bool = False
    resource_arbiter_periodic_min_runtime_sec: float = 1800.0
    resource_arbiter_periodic_min_interval_sec: float = 900.0
    resource_arbiter_proposal_coalesce_window_sec: float = 60.0
    resource_arbiter_job_llm_call_cap: int = 12
    resource_arbiter_job_advisory_call_cap: int = 3
    resource_arbiter_job_token_cap: int = 48000
    resource_sidecar_enabled: bool = False
    resource_sidecar_min_parent_runtime_sec: float = 900.0
    estra_magent_enabled: bool = False
    estra_magent_min_parent_runtime_sec: float = 300.0
    estra_magent_sidecar_enabled: bool = False
    estra_magent_sidecar_mode: str = "cpu_only"
    estra_magent_budget_sec: float = 900.0
    estra_magent_join_inject_parent: bool = True
    estra_magent_join_inject_estra: bool = True
    estra_magent_join_inject_resource_context: bool = True
    resource_checkpoint_submission_guard_enabled: bool = True
    resource_queue_timeout_hard_gate_enabled: bool = True
    resource_queue_timeout_block_train_after: int = 1
    resource_queue_timeout_tt_only_after: int = 2

@dataclass
class EvaluatorCandidateConfig:
    artifact: str = ""
    artifact_kind: str = ""
    commit_file: str = ""
    require_sha: bool = False
    scan_mode: str = "root_file"
    emit_missing_artifact_event: bool = False

@dataclass
class EvaluatorMetricConfig:
    name: str = "metric"
    lower_is_better: bool | None = None
    type: str = ""
    regex: str = ""
    json_path: str = ""
    selection_requires_direction: bool = True

@dataclass
class EvaluatorCommandConfig:
    evaluator_command: str = ""
    python_executable: str = ""
    environment_path: str = ""
    timeout_sec: float = 300.0
    cwd: str = "workspace"
    env: dict[str, str] = field(default_factory=dict)
    stdout_tail_chars: int = 4000
    stderr_tail_chars: int = 4000

@dataclass
class GateConfig:
    policy: str = "default"
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluatorConfig:
    enabled: bool = True
    expose_wall_clock_remaining_sec: bool = False
    query_budget_scope: str = "task"
    stop_on_query_budget_exhausted: bool = False
    task_profile: str = "auto"
    backend: str = "auto"
    stage_source_mode: str = "primary"
    event_log: str = "evaluator_events.jsonl"
    cache_log: str = "evaluator_cache.jsonl"
    candidate: EvaluatorCandidateConfig = field(default_factory=EvaluatorCandidateConfig)
    metric: EvaluatorMetricConfig = field(default_factory=EvaluatorMetricConfig)
    command: EvaluatorCommandConfig = field(default_factory=EvaluatorCommandConfig)
    package_source: str = ""  # Optional portable folder; append to preserve positional callers.

@dataclass
class WorkspaceConfig:
    """Workspace, task identity, and global run metadata."""

    # Single-task output root. LNR workers use indexed execution roots below it
    # (workers/w00, workers/w01, ...); non-worker REPL runs use this root directly.
    task_workspace_root_dir: Path = field(default_factory=lambda: Path("."))
    # Read-only competition data root. LNR exposes this as a single flat workspace/dataset.
    # Legacy prepared/dataset_split roots are adapted by exposing their Deep child as flat data.
    input_data_dir: Path = field(default_factory=lambda: Path("."))
    # Execution directory: dataset/, submissions/, cwd for code runs.
    # Filled by prep_cfg from task_workspace_root_dir.
    workspace_dir: Path = field(default_factory=lambda: Path("./workspace"))
    log_dir: Path = field(default_factory=lambda: Path("."))
    submission_dir: Path = field(default_factory=lambda: Path("."))
    exp_id: str = ""
    mlebench_data_root_dir: str = ""
    # Optional task metadata override when leaderboard information is unavailable.
    custom_metric_name: str = ""
    custom_is_lower_better: bool | None = None
    enable_time_trace: bool = True
    max_messages: int = 100

@dataclass
class ReplConfig:
    """Generic REPL profile, data preview, and workspace-git controls."""

    # QA / generic ScienceAgent: max LLM↔tool rounds per single user request.
    qa_max_steps: int = 20
    # Migration/rollback selector. It is orchestration-only and is never added
    # to prompts, provider requests, memory, or workspace files.
    # REPL: independent max LLM↔tool rounds per user request. 0 or negative
    # falls back to qa_max_steps for legacy manifests.
    repl_max_steps: int = 200
    # REPL profile is intentionally generic; task/domain rules should come from
    # the user query, task manifest, or skills rather than the base system prompt.
    repl_profile: str = "lite"
    # REPL tool preset. "bash_write" exposes bash/read/grep/glob/ls and performs
    # file changes through bash; "write_edit" keeps the legacy write/edit tools.
    repl_tool_preset: str = "bash_write"
    # Keep REPL system bytes stable by omitting dynamic workspace state and round
    # budget from the system prompt. Dynamic facts remain in tool results/history.
    repl_stable_system_prompt: bool = True
    # Pin a small environment-context user message at REPL session start.
    repl_pin_environment_context: bool = True
    # REPL-only optional user-side code organization preference.
    repl_code_organization_hint: str = ""
    # REPL-only workspace source control.
    repl_workspace_git_enabled: bool = True
    repl_workspace_git_track_globs: list[str] = field(default_factory=lambda: ["*.py", "*.md"])
    repl_workspace_git_initial_commit: bool = True
    repl_workspace_git_auto_checkpoint: bool = True
    repl_workspace_git_auto_review: bool = False
    repl_pin_task_description_when_auto_first_user: bool = False
    # REPL-only bash output caps.
    repl_bash_max_output_chars: int = 6000
    repl_bash_max_stream_line_chars: int = 1200
    repl_bash_observation_summary: bool = True
    # Data preview controls.
    preview_raw_files: bool = False
    data_preview_max_chars: int = 32000
    data_preview_max_items_per_dir: int = 40
    data_scan_walk_budget_dirs: int | None = 200_000
    data_scan_walk_budget_files: int | None = 500_000
    data_scan_probe_binary_dirs_budget: int = 8
    data_scan_preview_raw_dirs_budget: int = 12
    data_scan_meta_sample_max_bytes: int = 50_000
    data_scan_csv_max_rows_to_scan: int = 200_000
    data_preview_refresh_after_prep: bool = False

@dataclass
class ToolConfig:
    """ScienceAgent tool runtime, memory projection, and feedback controls."""

    # ScienceAgent: cap each execution feedback block (stdout/stderr); 0 = no cap.
    exec_feedback_max_chars: int = 5000
    # ScienceAgent: max total chars for user/assistant messages sent to LLM.
    sliding_window_budget_chars: int = 60000
    sliding_window_priority_enabled: bool = True
    mid_run_compact_enabled: bool = False
    round_budget_prompt_cap: int = 15
    # Tool runtime and sandbox.
    scienceflow_tools_sandbox: bool = True
    parallel_bash_enabled: bool = True
    parallel_llm_tool_calls: bool = True
    path_guard_extra_roots: list[Path] = field(default_factory=list)
    scienceflow_llm_stream_timeout_sec: int = 1800
    stream_repetition_detection: bool = True
    stream_repetition_window_chars: int = 4000
    stream_repetition_ngram_len: int = 150
    stream_repetition_max_repeats: int = 3
    stream_max_output_chars_soft: int = 30000
    stream_repetition_retry_max: int = 2
    llm_tool_stream_max_attempts: int = 5
    llm_tool_stream_retry_base_delay_sec: float = 1.75
    llm_tool_stream_retry_max_delay_sec: float = 20.0
    scienceflow_bash_timeout_sec: float = 14400.0
    scienceflow_bash_timeout_slow_sec: float = 14400.0
    scienceflow_stdout_max_chars: int = 32768
    # Interaction log and tool-memory projection.
    scienceflow_interaction_log_level: str = "minimal"
    scienceflow_interaction_log_full: bool = False
    scienceflow_interaction_log_color: bool = True
    scienceflow_interaction_log_llm_stream: bool = False
    scienceflow_bash_stream_to_interaction_log: bool = True
    bash_output_dedup_enabled: bool = True
    bash_output_dedup_min_repeat: int = 3
    bash_output_dedup_summary_prefix: str = "[log-dedup]"
    bash_output_dedup_apply_to_memory: bool = True
    tool_memory_compression: bool = True
    grep_max_results_lines: int = 50
    msg0_compress_body: bool = False
    bash_success_tail_lines: int = 50
    bash_success_tail_lines_solution: int = 8
    bash_success_tail_lines_test: int = 120
    bash_success_tail_lines_readonly: int = 30
    bash_success_tail_lines_install: int = 5
    read_success_max_lines: int = 200
    # Tool feedback snapshots and edit diagnostics.
    write_auto_snapshot_enabled: bool = True
    write_auto_snapshot_paths: list[str] = field(default_factory=list)
    write_auto_snapshot_code_extensions: list[str] = field(default_factory=lambda: [".py"])
    write_auto_snapshot_max_lines: int = 400
    write_auto_snapshot_max_chars: int = 8_000
    write_auto_snapshot_changed_context_lines: int = 10
    write_auto_snapshot_symbol_body_lines: int = 3
    read_overlap_guard_enabled: bool = True
    rotating_runtime_error_threshold: int = 3
    rotating_distinct_types_min: int = 2
    no_success_run_soft_threshold: int = 6
    no_success_run_hard_threshold: int = 12
    edit_short_old_str_chars: int = 40
    write_return_full_max_chars: int = 12000
    write_return_full_max_lines: int = 250
    write_return_head_tail_lines: int = 40
    edit_return_full_max_chars: int = 12000
    edit_return_full_max_lines: int = 250
    edit_return_head_tail_lines: int = 40
    edit_return_change_ctx_lines: int = 8
    edit_failure_top_k_candidates: int = 3
    edit_failure_diag_max_chars: int = 2000
    file_snapshot_latest_only: bool = True

@dataclass
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    repl: ReplConfig = field(default_factory=ReplConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    lnr: LnrConfig = field(default_factory=LnrConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    # Optional task-profile defaults applied by CLI entrypoints before explicit
    # manifest lnr/agent patches. Keys are profile names such as "mlebench" or
    # "opt_solver"; supported child blocks are "lnr", "repl", and "evaluator".
    profile_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Set by :func:`load_cfg` to the resolved path of the YAML used as ``--config`` (or default.yaml).
    config_source_path: str = ""

_WORKSPACE_CONFIG_KEYS: tuple[str, ...] = (
    "task_workspace_root_dir",
    "input_data_dir",
    "workspace_dir",
    "log_dir",
    "submission_dir",
    "exp_id",
    "mlebench_data_root_dir",
    "custom_metric_name",
    "custom_is_lower_better",
    "enable_time_trace",
    "max_messages",
)

_REPL_CONFIG_KEYS: tuple[str, ...] = (
    "qa_max_steps",
    "repl_max_steps",
    "repl_profile",
    "repl_tool_preset",
    "repl_stable_system_prompt",
    "repl_pin_environment_context",
    "repl_code_organization_hint",
    "repl_workspace_git_enabled",
    "repl_workspace_git_track_globs",
    "repl_workspace_git_initial_commit",
    "repl_workspace_git_auto_checkpoint",
    "repl_workspace_git_auto_review",
    "repl_pin_task_description_when_auto_first_user",
    "repl_bash_max_output_chars",
    "repl_bash_max_stream_line_chars",
    "repl_bash_observation_summary",
    "preview_raw_files",
    "data_preview_max_chars",
    "data_preview_max_items_per_dir",
    "data_scan_walk_budget_dirs",
    "data_scan_walk_budget_files",
    "data_scan_probe_binary_dirs_budget",
    "data_scan_preview_raw_dirs_budget",
    "data_scan_meta_sample_max_bytes",
    "data_scan_csv_max_rows_to_scan",
    "data_preview_refresh_after_prep",
)

_TOOL_CONFIG_KEYS: tuple[str, ...] = (
    "exec_feedback_max_chars",
    "sliding_window_budget_chars",
    "sliding_window_priority_enabled",
    "mid_run_compact_enabled",
    "round_budget_prompt_cap",
    "scienceflow_tools_sandbox",
    "parallel_bash_enabled",
    "parallel_llm_tool_calls",
    "path_guard_extra_roots",
    "scienceflow_llm_stream_timeout_sec",
    "stream_repetition_detection",
    "stream_repetition_window_chars",
    "stream_repetition_ngram_len",
    "stream_repetition_max_repeats",
    "stream_max_output_chars_soft",
    "stream_repetition_retry_max",
    "llm_tool_stream_max_attempts",
    "llm_tool_stream_retry_base_delay_sec",
    "llm_tool_stream_retry_max_delay_sec",
    "scienceflow_bash_timeout_sec",
    "scienceflow_bash_timeout_slow_sec",
    "scienceflow_stdout_max_chars",
    "scienceflow_interaction_log_level",
    "scienceflow_interaction_log_full",
    "scienceflow_interaction_log_color",
    "scienceflow_interaction_log_llm_stream",
    "scienceflow_bash_stream_to_interaction_log",
    "bash_output_dedup_enabled",
    "bash_output_dedup_min_repeat",
    "bash_output_dedup_summary_prefix",
    "bash_output_dedup_apply_to_memory",
    "tool_memory_compression",
    "grep_max_results_lines",
    "msg0_compress_body",
    "bash_success_tail_lines",
    "bash_success_tail_lines_solution",
    "bash_success_tail_lines_test",
    "bash_success_tail_lines_readonly",
    "bash_success_tail_lines_install",
    "read_success_max_lines",
    "write_auto_snapshot_enabled",
    "write_auto_snapshot_paths",
    "write_auto_snapshot_code_extensions",
    "write_auto_snapshot_max_lines",
    "write_auto_snapshot_max_chars",
    "write_auto_snapshot_changed_context_lines",
    "write_auto_snapshot_symbol_body_lines",
    "read_overlap_guard_enabled",
    "rotating_runtime_error_threshold",
    "rotating_distinct_types_min",
    "no_success_run_soft_threshold",
    "no_success_run_hard_threshold",
    "edit_short_old_str_chars",
    "write_return_full_max_chars",
    "write_return_full_max_lines",
    "write_return_head_tail_lines",
    "edit_return_full_max_chars",
    "edit_return_full_max_lines",
    "edit_return_head_tail_lines",
    "edit_return_change_ctx_lines",
    "edit_failure_top_k_candidates",
    "edit_failure_diag_max_chars",
    "file_snapshot_latest_only",
)

__all__ = tuple(name for name in globals() if not name.startswith("__"))
