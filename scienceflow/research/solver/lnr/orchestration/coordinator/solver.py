"""Public LNR coordinator assembled from responsibility modules.

The class owns composition and run identity only. Domain implementations remain
owned by the module graph built during construction. Method assignment is explicit
so supported private monkeypatch points retain their Python descriptor semantics.
"""

from __future__ import annotations

import time

__scienceflow_canonical_backend__ = True
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    LHR_STAGE_PERFORMANCE_COLUMNS,
    LHR_STAGE_PERFORMANCE_CSV,
    _append_lnr_main_agent_protocols,
    _deterministic_gate_enabled,
    _deterministic_gate_timestamp,
    _effective_lnr_bash_timeout_sec,
    _validation_leakage_reason,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context import (
    coordination as context_coordination,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context import memory_coordination
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context import memory_projection
from scienceflow.research.solver.lnr.orchestration.coordinator.estra import adapter as estra_adapter
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.adapter import EvaluationAdapterOwner
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.capture import EvaluationCaptureOwner
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.commit import EvaluationCommitOwner
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation import metric_projection
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.profile import EvaluationProfileOwner
from scienceflow.research.solver.lnr.orchestration.coordinator.resource import advisory as resource_advisory
from scienceflow.research.solver.lnr.orchestration.coordinator.resource import (
    construction as resource_construction,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.resource import (
    observer_construction as resource_observer_construction,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.resource import (
    prompt_projection as resource_prompt_projection,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.run import agent_coordination
from scienceflow.research.solver.lnr.orchestration.coordinator.run import construction
from scienceflow.research.solver.lnr.orchestration.coordinator.run import coordination as run_coordination
from scienceflow.research.solver.lnr.orchestration.coordinator.run import event_projection
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.stage import adapter as stage_adapter
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.stage import commit as stage_commit
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.stage import prompt as stage_prompt

__all__ = (
    "LHR_STAGE_PERFORMANCE_COLUMNS",
    "LHR_STAGE_PERFORMANCE_CSV",
    "LnrSolver",
    "_append_lnr_main_agent_protocols",
    "_deterministic_gate_enabled",
    "_deterministic_gate_timestamp",
    "_effective_lnr_bash_timeout_sec",
    "_validation_leakage_reason",
    "time",
)


class LnrSolver(
    EvaluationProfileOwner,
    EvaluationAdapterOwner,
    EvaluationCaptureOwner,
    EvaluationCommitOwner,
):
    """Thin composition facade for the modular LNR runtime."""

    solver_name = "lnr"
    __init__ = construction.__init__
    _initialize_run_state = construction._initialize_run_state
    _runtime_event_observer = construction._runtime_event_observer
    _repo_root = construction._repo_root
    _skill_library_dir = construction._skill_library_dir
    _lnr_skill_category_source_path = construction._lnr_skill_category_source_path
    _load_lnr_task_category_label = construction._load_lnr_task_category_label
    _configure_lnr_category_skill = construction._configure_lnr_category_skill
    _extract_lnr_auto_skill_hint = construction._extract_lnr_auto_skill_hint
    _lnr_compact_skill_rendered = construction._lnr_compact_skill_rendered
    _lnr_auto_read_skill_block = construction._lnr_auto_read_skill_block
    _lnr_skill_hint = construction._lnr_skill_hint
    _make_resource_arbiter_decider = (
        resource_construction._make_resource_arbiter_decider
    )
    _resource_advisory_prompt = resource_construction._resource_advisory_prompt
    _drop_dangling_tool_call_tail = resource_construction._drop_dangling_tool_call_tail
    _make_resource_main_agent_advisory_decider = (
        resource_construction._make_resource_main_agent_advisory_decider
    )
    _decide_resource_main_agent_advisory = (
        resource_construction._decide_resource_main_agent_advisory
    )
    _direct_resource_main_agent_advisory = (
        resource_construction._direct_resource_main_agent_advisory
    )
    _capture_inline_resource_advisory = (
        resource_construction._capture_inline_resource_advisory
    )
    _make_resource_admission_decider = (
        resource_construction._make_resource_admission_decider
    )
    _make_resource_observer = resource_observer_construction._make_resource_observer
    _resource_observer_options_1 = (
        resource_observer_construction._resource_observer_options_1
    )
    _resource_observer_options_2 = (
        resource_observer_construction._resource_observer_options_2
    )
    _resource_observer_options_3 = (
        resource_observer_construction._resource_observer_options_3
    )
    _resource_observer_options_4 = (
        resource_observer_construction._resource_observer_options_4
    )
    _resource_observer_options_5 = (
        resource_observer_construction._resource_observer_options_5
    )
    _resource_observer_options_6 = (
        resource_observer_construction._resource_observer_options_6
    )
    _resource_observer_options_7 = (
        resource_observer_construction._resource_observer_options_7
    )
    _safe_ledger_filename = construction._safe_ledger_filename
    _jsonl_roots = event_projection._jsonl_roots
    _jsonl = event_projection._jsonl
    _record_stage_lifecycle_trace = event_projection._record_stage_lifecycle_trace
    _record_agent_factory_trace = event_projection._record_agent_factory_trace
    _state_task_type_for_event = event_projection._state_task_type_for_event
    _relativize_control_payload = event_projection._relativize_control_payload
    _mirror_state_event = event_projection._mirror_state_event
    _write_stage_map = event_projection._write_stage_map
    _load_existing_stage_snapshots = event_projection._load_existing_stage_snapshots
    _prepare_workspace = event_projection._prepare_workspace
    _init_workspace_path = event_projection._init_workspace_path
    _prepare_init_workspace = event_projection._prepare_init_workspace
    _prepare_workspace_git = event_projection._prepare_workspace_git
    _interaction_log_node_dir = event_projection._interaction_log_node_dir
    _split_logs_dir = event_projection._split_logs_dir
    _close_lnr_interaction_loggers = event_projection._close_lnr_interaction_loggers
    _attach_lnr_interaction_logger = event_projection._attach_lnr_interaction_logger
    _worker_uid_prefix = event_projection._worker_uid_prefix
    _lineage_uid_prefix = event_projection._lineage_uid_prefix
    _stage_node_uid = event_projection._stage_node_uid
    _start_new_lineage = event_projection._start_new_lineage
    _snapshot_node_uid = event_projection._snapshot_node_uid
    _archive_snapshot_key = event_projection._archive_snapshot_key
    _archived_stage_snapshot_index = event_projection._archived_stage_snapshot_index
    _index_archived_stage_snapshot = event_projection._index_archived_stage_snapshot
    _stage_snapshot_for_restore = event_projection._stage_snapshot_for_restore
    _active_node_uid_for_stage = event_projection._active_node_uid_for_stage
    _next_active_stage_id = event_projection._next_active_stage_id
    _next_stage_id_for_logging = event_projection._next_stage_id_for_logging
    _latest_stage_id_for_logging = event_projection._latest_stage_id_for_logging
    _record_context_compact_event = event_projection._record_context_compact_event
    _ensure_split_logs = event_projection._ensure_split_logs
    _prune_workspace_control_artifacts = (
        event_projection._prune_workspace_control_artifacts
    )
    _reset_interaction_stage_files = event_projection._reset_interaction_stage_files
    _reset_agent_interaction_stage = event_projection._reset_agent_interaction_stage
    _agent_memory_messages = event_projection._agent_memory_messages
    _message_content_text = memory_projection._message_content_text
    _build_protected_eda_facts_summary = (
        memory_projection._build_protected_eda_facts_summary
    )
    _mark_protected_eda_prefix = memory_projection._mark_protected_eda_prefix
    _capture_s01_eda_prefix_end = memory_projection._capture_s01_eda_prefix_end
    _build_protected_eda_agent_prompt = (
        memory_projection._build_protected_eda_agent_prompt
    )
    _normalize_protected_eda_agent_summary = (
        memory_projection._normalize_protected_eda_agent_summary
    )
    _prepare_protected_eda_agent_summary = (
        memory_projection._prepare_protected_eda_agent_summary
    )
    _restore_protected_eda_prefix_marker = (
        memory_projection._restore_protected_eda_prefix_marker
    )
    _append_traj_summary = memory_projection._append_traj_summary
    _base_stage_id = memory_projection._base_stage_id
    _base_stage_agent_memory_dir = memory_projection._base_stage_agent_memory_dir
    _stage_card_override_text = memory_projection._stage_card_override_text
    _stage_card_effective_overrides = memory_projection._stage_card_effective_overrides
    _effective_stage_cards = memory_projection._effective_stage_cards
    _effective_stage_cards_from_ledger = (
        memory_projection._effective_stage_cards_from_ledger
    )
    _stage_cards_from_snapshots = memory_projection._stage_cards_from_snapshots
    _stage_memory_view_for_prompt = memory_projection._stage_memory_view_for_prompt
    _stage_card_one_line = memory_coordination._stage_card_one_line
    _best_candidate_stage_id = memory_coordination._best_candidate_stage_id
    _safe_estra_fallback_decision = memory_coordination._safe_estra_fallback_decision
    _build_lhr_state_packet = memory_coordination._build_lhr_state_packet
    _rebuild_memory_after_estra = memory_coordination._rebuild_memory_after_estra
    _rebuild_memory_after_keep_current = (
        memory_coordination._rebuild_memory_after_keep_current
    )
    _prepare_dataset_symlink = memory_coordination._prepare_dataset_symlink
    _count_memory_records = metric_projection._count_memory_records
    _workspace_metric_audit_source_text = (
        metric_projection._workspace_metric_audit_source_text
    )
    _metric_event_from_workspace = metric_projection._metric_event_from_workspace
    _compact_peer_method = resource_prompt_projection._compact_peer_method
    _magent_recommendations_for_resource_context = (
        resource_prompt_projection._magent_recommendations_for_resource_context
    )
    _stage_payloads_for_score_summary = (
        resource_prompt_projection._stage_payloads_for_score_summary
    )
    _score_summary_for_prompt = resource_prompt_projection._score_summary_for_prompt
    _allocated_compute_context_lines = (
        resource_prompt_projection._allocated_compute_context_lines
    )
    _render_resource_context_prompt = (
        resource_prompt_projection._render_resource_context_prompt
    )
    _merge_global_resource_state = (
        resource_prompt_projection._merge_global_resource_state
    )
    _resource_prompt_policy = resource_prompt_projection._resource_prompt_policy
    _resource_context_for_prompt = (
        resource_prompt_projection._resource_context_for_prompt
    )
    _parallel_worker_snapshot_for_prompt = (
        resource_advisory._parallel_worker_snapshot_for_prompt
    )
    _estra_peer_route_evidence = resource_advisory._estra_peer_route_evidence
    _estra_backtrack_reflection = resource_advisory._estra_backtrack_reflection
    _next_global_stage_row_order = resource_advisory._next_global_stage_row_order
    _best_stage_id_so_far = resource_advisory._best_stage_id_so_far
    _stage_with_submission_sha = resource_advisory._stage_with_submission_sha
    _stage_with_artifact_sha = resource_advisory._stage_with_artifact_sha
    _duplicate_stage_same_design = resource_advisory._duplicate_stage_same_design
    _metric_lower_is_better_decision = (
        resource_advisory._metric_lower_is_better_decision
    )
    _apply_metric_direction_audit = resource_advisory._apply_metric_direction_audit
    _metric_lower_is_better_for_event = (
        resource_advisory._metric_lower_is_better_for_event
    )
    _metric_source_note = resource_advisory._metric_source_note
    _metric_validity_card_fields = resource_advisory._metric_validity_card_fields
    _task_metric_context_excerpt = resource_advisory._task_metric_context_excerpt
    _metric_validity_snapshot_tail = resource_advisory._metric_validity_snapshot_tail
    _metric_validity_fact_card = resource_advisory._metric_validity_fact_card
    _metric_feedback_llm_client = resource_advisory._metric_feedback_llm_client
    _stage_commit_one_line = stage_commit._stage_commit_one_line
    _stage_commit_query_state = stage_commit._stage_commit_query_state
    _build_stage_commit_experiment_state = (
        stage_commit._build_stage_commit_experiment_state
    )
    _stage_commit_bool_text = stage_commit._stage_commit_bool_text
    _stage_commit_explicit_no_files = stage_commit._stage_commit_explicit_no_files
    _stage_commit_default_judgment = stage_commit._stage_commit_default_judgment
    _stage_commit_fallback_judgment = stage_commit._stage_commit_fallback_judgment
    _stage_commit_metric_note = stage_commit._stage_commit_metric_note
    _stage_commit_entry_from_metric_event = (
        stage_commit._stage_commit_entry_from_metric_event
    )
    _stage_commit_parse_judgment = stage_commit._stage_commit_parse_judgment
    _fork_stage_commit_judgment = stage_commit._fork_stage_commit_judgment
    _ephemeral_stage_commit = stage_commit._ephemeral_stage_commit
    _estra_stage_checkpoint_context = stage_commit._estra_stage_checkpoint_context
    _estra_switch_candidates = estra_adapter._estra_switch_candidates
    _is_keep_like_estra_action = estra_adapter._is_keep_like_estra_action
    _estra_axes_from_action = estra_adapter._estra_axes_from_action
    _estra_decision_kind = estra_adapter._estra_decision_kind
    _derive_estra_compact = estra_adapter._derive_estra_compact
    _estra_decision_fields = estra_adapter._estra_decision_fields
    _estra_observation_key = estra_adapter._estra_observation_key
    _emit_estra_decision = estra_adapter._emit_estra_decision
    _ask_estra = estra_adapter._ask_estra
    _early_estra_decision = estra_adapter._early_estra_decision
    _plan_estra_request = estra_adapter._plan_estra_request
    _normalize_estra_plan = estra_adapter._normalize_estra_plan
    _normalize_estra_archive_summary_text = (
        estra_adapter._normalize_estra_archive_summary_text
    )
    _synthesize_estra_archive_summary = estra_adapter._synthesize_estra_archive_summary
    _set_pending_estra_from_decision = estra_adapter._set_pending_estra_from_decision
    _estra_candidates_for_current_ledger = (
        estra_adapter._estra_candidates_for_current_ledger
    )
    _estra_trigger_allowed = estra_adapter._estra_trigger_allowed
    _text_only_estra_trigger_allowed = estra_adapter._text_only_estra_trigger_allowed
    _force_estra_trigger_allowed = estra_adapter._force_estra_trigger_allowed
    _current_main_token_totals = estra_adapter._current_main_token_totals
    _note_context_hygiene_stage_delta = (
        context_coordination._note_context_hygiene_stage_delta
    )
    _context_hygiene_active_high_risk_bash = (
        context_coordination._context_hygiene_active_high_risk_bash
    )
    _context_hygiene_snapshot_can_preserve_current_state = (
        context_coordination._context_hygiene_snapshot_can_preserve_current_state
    )
    _context_hygiene_compact_after_stage_capture = (
        context_coordination._context_hygiene_compact_after_stage_capture
    )
    _evaluate_context_hygiene_after_stage = (
        context_coordination._evaluate_context_hygiene_after_stage
    )
    _force_estra_after_stage_capture = (
        context_coordination._force_estra_after_stage_capture
    )
    _text_only_estra_callback = context_coordination._text_only_estra_callback
    _context_limit_estra_callback = context_coordination._context_limit_estra_callback
    _context_limit_estra_decision = context_coordination._context_limit_estra_decision
    _parse_estra_decision = context_coordination._parse_estra_decision
    _text_only_continue_search_prompt = (
        context_coordination._text_only_continue_search_prompt
    )
    _suppress_repeated_text_only_completion = (
        context_coordination._suppress_repeated_text_only_completion
    )
    _parse_stage_commit_block = stage_prompt._parse_stage_commit_block
    _stage_commit_judgment_from_text_block = (
        stage_prompt._stage_commit_judgment_from_text_block
    )
    _build_stage_commit_text_prompt = stage_prompt._build_stage_commit_text_prompt
    _build_compact_stage_commit_text_prompt = (
        stage_prompt._build_compact_stage_commit_text_prompt
    )
    _extend_stage_commit_text_policy_deadline = (
        stage_prompt._extend_stage_commit_text_policy_deadline
    )
    _set_stage_commit_transient_prompt = stage_prompt._set_stage_commit_transient_prompt
    _clear_stage_commit_transient_prompt = (
        stage_prompt._clear_stage_commit_transient_prompt
    )
    _pending_stage_commit_text_active = stage_prompt._pending_stage_commit_text_active
    _abandon_pending_stage_commit_text = stage_prompt._abandon_pending_stage_commit_text
    _stage_commit_text_memory_message = stage_prompt._stage_commit_text_memory_message
    _append_stage_commit_from_judgment = (
        stage_adapter._append_stage_commit_from_judgment
    )
    _stage_transaction_root = stage_adapter._stage_transaction_root
    _stage_lifecycle = stage_adapter._stage_lifecycle
    _stage_transaction_service = stage_adapter._stage_transaction_service
    _prepare_stage_commit_transaction = stage_adapter._prepare_stage_commit_transaction
    _complete_stage_commit_transaction = (
        stage_adapter._complete_stage_commit_transaction
    )
    _rollback_stage_commit_transaction = (
        stage_adapter._rollback_stage_commit_transaction
    )
    _recover_stage_commit_transactions = (
        stage_adapter._recover_stage_commit_transactions
    )
    _finalize_stage_capture_after_commit = (
        stage_adapter._finalize_stage_capture_after_commit
    )
    _finalize_stage_capture_after_commit_impl = (
        stage_adapter._finalize_stage_capture_after_commit_impl
    )
    _checkpoint_stage_workspace = stage_adapter._checkpoint_stage_workspace
    _handle_pending_stage_commit_text = stage_adapter._handle_pending_stage_commit_text
    _resolve_stage_commit_text_judgment = (
        stage_adapter._resolve_stage_commit_text_judgment
    )
    _evaluator_stage_source_mode = stage_adapter._evaluator_stage_source_mode
    _make_agent = agent_coordination._make_agent
    _create_main_agent = agent_coordination._create_main_agent
    _configure_agent_workspace_surface = (
        agent_coordination._configure_agent_workspace_surface
    )
    _configure_agent_bash = agent_coordination._configure_agent_bash
    _configure_agent_protocol = agent_coordination._configure_agent_protocol
    _configure_agent_callbacks = agent_coordination._configure_agent_callbacks
    _runtime_context_for_agent = agent_coordination._runtime_context_for_agent
    _worker_llm_stage_override = agent_coordination._worker_llm_stage_override
    _configured_model_stage_override = (
        agent_coordination._configured_model_stage_override
    )
    _restore_pending_estra = agent_coordination._restore_pending_estra
    _restore_keep_pending_estra = agent_coordination._restore_keep_pending_estra
    _accumulate_main_run_tokens = agent_coordination._accumulate_main_run_tokens
    _result = agent_coordination._result
    _count_jsonl_events = run_coordination._count_jsonl_events
    _estra_decision_counts = run_coordination._estra_decision_counts
    _format_cpu_ids = run_coordination._format_cpu_ids
    _slice_cpu_ids = run_coordination._slice_cpu_ids
    _worker_extra_env = run_coordination._worker_extra_env
    _worker_root = run_coordination._worker_root
    _worker_log_dir = run_coordination._worker_log_dir
    _merge_dir = run_coordination._merge_dir
    _global_merge_reserve_sec = run_coordination._global_merge_reserve_sec
    _final_artifact_mode = run_coordination._final_artifact_mode
    _worker_wall_clock_budget_sec = run_coordination._worker_wall_clock_budget_sec
    _refresh_submission_links = run_coordination._refresh_submission_links
    _cleanup_coordinator_workspace_shell = (
        run_coordination._cleanup_coordinator_workspace_shell
    )
    _cleanup_worker_root_artifacts = run_coordination._cleanup_worker_root_artifacts
    _aggregate_worker_state = run_coordination._aggregate_worker_state
    _aggregate_worker_state_periodically = (
        run_coordination._aggregate_worker_state_periodically
    )
    _write_global_time_trace = run_coordination._write_global_time_trace
    _worker_error_kind = run_coordination._worker_error_kind
    _multi_worker_failure_kind = run_coordination._multi_worker_failure_kind
    _multi_worker_stop_reason = run_coordination._multi_worker_stop_reason
    _make_worker_cfg = run_coordination._make_worker_cfg
    _run_one_worker = run_coordination._run_one_worker
    _read_json_file = run_coordination._read_json_file
    _candidate_from_stage = run_coordination._candidate_from_stage
    _load_worker_candidates = run_coordination._load_worker_candidates
    _metric_float = run_coordination._metric_float
    _write_stage_collection_outputs = run_coordination._write_stage_collection_outputs
    _run_live_merge_agent = run_coordination._run_live_merge_agent
    _close_merge_owner_agent = run_coordination._close_merge_owner_agent
    _write_merge_outputs = run_coordination._write_merge_outputs
    _run_multi_worker = run_coordination._run_multi_worker
    _run_single = run_coordination._run_single
    run = run_coordination.run
    _ask_agent_tool_stream_guarded = run_coordination._ask_agent_tool_stream_guarded
