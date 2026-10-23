"""Phased initialization for the resource-observer facade."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    AdmissionDecisionRouter,
    AdmissionPolicyService,
    AdmissionReplayArchive,
    Any,
    ESTRAMagentConfig,
    GPUQueueConfig,
    LnrExecutionValueShadow,
    Path,
    ResourceControlProfile,
    ResourceJob,
    ResourceManagementService,
    ResourceRegistry,
    ResourceReviewConfig,
    ResourceReviewMachine,
    ResourceReviewState,
    build_gpu_share_config,
    build_resource_control_profile,
    cap_float,
    cap_int,
    trial_windows_for_profile,
)


def _initialize_control_and_advisory(target: Any, payload: dict[str, Any]) -> None:
    target.state_machine = payload["state_machine"]
    target.worker_id = str(payload["worker_id"] or "W00")
    target.control_profile: ResourceControlProfile = build_resource_control_profile(
        payload["resource_control_profile"]
    )
    target.resource_control_profile = target.control_profile.name
    target.min_register_sec = cap_float(
        float(payload["min_register_sec"]),
        target.control_profile.min_register_sec,
        minimum=0.0,
    )
    target.check_interval_sec = max(1.0, float(payload["check_interval_sec"]))
    target.stalled_stdout_sec = cap_float(
        float(payload["stalled_stdout_sec"]),
        target.control_profile.stalled_stdout_sec,
        minimum=0.0,
    )
    target.kill_mode = (
        str(payload["kill_mode"] or ("auto" if payload["kill_enabled"] else "off"))
        .strip()
        .lower()
    )
    if target.kill_mode not in {"recommend", "auto", "arbiter", "off"}:
        target.kill_mode = "recommend"
    target.kill_enabled = bool(payload["kill_enabled"]) and target.kill_mode != "off"
    target.auto_kill_enabled = target.kill_enabled and target.kill_mode == "auto"
    target.arbiter_enabled = (
        bool(payload["arbiter_enabled"]) or target.kill_mode == "arbiter"
    )
    target.arbiter_mode = str(payload["arbiter_mode"] or "policy").strip().lower()
    if target.arbiter_mode not in {"policy", "llm", "off"}:
        target.arbiter_mode = "policy"
    target.arbiter_timeout_sec = max(1.0, float(payload["arbiter_timeout_sec"] or 90.0))
    target.arbiter_decider = payload["arbiter_decider"]
    target.main_agent_advisory_enabled = bool(payload["main_agent_advisory_enabled"])
    target.main_agent_advisory_min_interval_sec = cap_float(
        float(payload["main_agent_advisory_min_interval_sec"] or 600.0),
        target.control_profile.main_agent_advisory_min_interval_sec,
        minimum=1.0,
    )
    target.main_agent_advisory_timeout_sec = max(
        1.0, float(payload["main_agent_advisory_timeout_sec"] or 60.0)
    )
    target.main_agent_advisory_decider = payload["main_agent_advisory_decider"]
    target.arbiter_contention_review_enabled = bool(
        payload["arbiter_contention_review_enabled"]
    )
    target.arbiter_contention_min_runtime_sec = max(
        0.0, float(payload["arbiter_contention_min_runtime_sec"] or 0.0)
    )
    target.arbiter_contention_min_waiter_age_sec = max(
        0.0, float(payload["arbiter_contention_min_waiter_age_sec"] or 0.0)
    )
    target.arbiter_contention_min_interval_sec = cap_float(
        float(payload["arbiter_contention_min_interval_sec"] or 600.0),
        target.control_profile.arbiter_contention_min_interval_sec,
        minimum=1.0,
    )
    target.research_cadence_enabled = bool(payload["research_cadence_enabled"])
    target.research_cadence_observe_sec = max(
        1.0, float(payload["research_cadence_observe_sec"] or 300.0)
    )
    target.first_comparable_metric_budget_sec = max(
        target.research_cadence_observe_sec,
        float(payload["first_comparable_metric_budget_sec"] or 900.0),
    )
    target.proven_route_metric_budget_sec = max(
        target.first_comparable_metric_budget_sec,
        float(payload["proven_route_metric_budget_sec"] or 1800.0),
    )
    target._research_route_metrics: dict[str, dict[str, Any]] = {}
    target.arbiter_periodic_review_enabled = bool(
        payload["arbiter_periodic_review_enabled"]
    )
    target.arbiter_periodic_min_runtime_sec = cap_float(
        float(payload["arbiter_periodic_min_runtime_sec"] or 0.0),
        target.control_profile.arbiter_periodic_min_runtime_sec,
        minimum=0.0,
    )
    target.arbiter_periodic_min_interval_sec = cap_float(
        float(payload["arbiter_periodic_min_interval_sec"] or 900.0),
        target.control_profile.arbiter_periodic_min_interval_sec,
        minimum=1.0,
    )
    target.arbiter_proposal_coalesce_window_sec = cap_float(
        float(payload["arbiter_proposal_coalesce_window_sec"] or 0.0),
        target.control_profile.arbiter_proposal_coalesce_window_sec,
        minimum=0.0,
    )
    target.arbiter_job_llm_call_cap = max(
        0, int(payload["arbiter_job_llm_call_cap"] or 0)
    )
    target.arbiter_job_advisory_call_cap = max(
        0, int(payload["arbiter_job_advisory_call_cap"] or 0)
    )


def _initialize_admission_and_monitoring(target: Any, payload: dict[str, Any]) -> None:
    target.arbiter_job_token_cap = max(0, int(payload["arbiter_job_token_cap"] or 0))
    target.admission_llm_enabled = bool(payload["admission_llm_enabled"])
    target.admission_llm_mode = (
        str(payload["admission_llm_mode"] or "low_confidence").strip().lower()
    )
    allowed_admission_modes = {
        "off",
        "low_confidence",
        "always",
        "all",
        "blocked_states",
        "blocked",
        "soft_gates",
    }
    if target.admission_llm_mode not in allowed_admission_modes:
        target.admission_llm_mode = "low_confidence"
    target.admission_llm_timeout_sec = max(
        1.0, float(payload["admission_llm_timeout_sec"] or 60.0)
    )
    target.admission_decider = payload["admission_decider"]
    target.stale_pressure_observe_first_enabled = bool(
        payload["stale_pressure_observe_first_enabled"]
    )
    target.stale_pressure_observe_window_sec = cap_float(
        float(payload["stale_pressure_observe_window_sec"] or 180.0),
        target.control_profile.stale_pressure_observe_window_sec,
        minimum=30.0,
    )
    target.stale_pressure_observe_max_sec = cap_float(
        float(
            payload["stale_pressure_observe_max_sec"]
            or target.stale_pressure_observe_window_sec
        ),
        target.control_profile.stale_pressure_observe_max_sec,
        minimum=target.stale_pressure_observe_window_sec,
    )
    target.stale_pressure_observe_min_free_mem_gb = max(
        0.0, float(payload["stale_pressure_observe_min_free_mem_gb"] or 0.0)
    )
    target.stale_pressure_healthy_skip_llm = bool(
        payload["stale_pressure_healthy_skip_llm"]
    )
    target.estra_magent = ESTRAMagentConfig.from_values(
        enabled=bool(payload["estra_magent_enabled"]),
        sidecar_enabled=bool(payload["estra_magent_sidecar_enabled"]),
        min_parent_runtime_sec=float(
            payload["estra_magent_min_parent_runtime_sec"] or 300.0
        ),
        sidecar_mode=str(payload["estra_magent_sidecar_mode"] or "cpu_only"),
        budget_sec=float(payload["estra_magent_budget_sec"] or 900.0),
        join_inject_parent=bool(payload["estra_magent_join_inject_parent"]),
        join_inject_estra=bool(payload["estra_magent_join_inject_estra"]),
        join_inject_resource_context=bool(
            payload["estra_magent_join_inject_resource_context"]
        ),
    )
    legacy_sidecar_enabled = bool(payload["sidecar_enabled"])
    target.sidecar_enabled = (
        legacy_sidecar_enabled or target.estra_magent.sidecar_allowed()
    )
    legacy_min_runtime = max(
        1.0, float(payload["sidecar_min_parent_runtime_sec"] or 900.0)
    )
    if target.estra_magent.sidecar_allowed() and legacy_sidecar_enabled:
        target.sidecar_min_parent_runtime_sec = min(
            legacy_min_runtime, target.estra_magent.min_parent_runtime_sec
        )
    elif target.estra_magent.sidecar_allowed():
        target.sidecar_min_parent_runtime_sec = (
            target.estra_magent.min_parent_runtime_sec
        )
    else:
        target.sidecar_min_parent_runtime_sec = legacy_min_runtime
    target.checkpoint_submission_guard_enabled = bool(
        payload["checkpoint_submission_guard_enabled"]
    )
    target._sidecar_jobs_started: set[str] = set()
    target._magent_observed_jobs: set[str] = set()
    target._magent_considered_jobs: set[str] = set()
    target.monitor_agent_mode = (
        str(payload["monitor_agent_mode"] or "off").strip().lower()
    )
    if target.monitor_agent_mode not in {"off", "shadow", "advisory", "gated_kill"}:
        target.monitor_agent_mode = "off"
    target.monitor_agent_min_interval_sec = max(
        1.0, float(payload["monitor_agent_min_interval_sec"] or 60.0)
    )
    target.low_progress_enabled = bool(payload["low_progress_enabled"])
    target.low_progress_warmup_sec = cap_float(
        float(payload["low_progress_warmup_sec"] or 0.0),
        target.control_profile.low_progress_warmup_sec,
        minimum=0.0,
    )
    target.low_progress_no_heartbeat_sec = cap_float(
        float(payload["low_progress_no_heartbeat_sec"] or 0.0),
        target.control_profile.low_progress_no_heartbeat_sec,
        minimum=0.0,
    )
    target.low_progress_no_artifact_sec = cap_float(
        float(payload["low_progress_no_artifact_sec"] or 0.0),
        target.control_profile.low_progress_no_artifact_sec,
        minimum=0.0,
    )
    target.bash_monitor_all_enabled = bool(payload["bash_monitor_all_enabled"])


def _initialize_services_and_state(target: Any, payload: dict[str, Any]) -> None:
    target.arbiter_min_progress_windows = cap_int(
        int(payload["arbiter_min_progress_windows"] or 1),
        target.control_profile.arbiter_min_progress_windows,
        minimum=1,
    )
    if target.control_profile.arbiter_kill_requires_high_confidence is None:
        target.arbiter_kill_requires_high_confidence = bool(
            payload["arbiter_kill_requires_high_confidence"]
        )
    else:
        target.arbiter_kill_requires_high_confidence = bool(
            target.control_profile.arbiter_kill_requires_high_confidence
        )
    target._last_monitor_agent_emit: dict[str, float] = {}
    target._last_gpu_util_emit: dict[str, float] = {}
    target._last_kill_proposal_emit: dict[str, float] = {}
    target._last_kill_proposal_id_by_dedupe: dict[str, str] = {}
    target._last_kill_proposal_boundary_by_dedupe: dict[str, dict[str, Any]] = {}
    target._last_suppressed_kill_proposal_emit: dict[str, float] = {}
    target._last_review_proposal_emit: dict[str, float] = {}
    target._review_observe_more_until: dict[str, float] = {}
    target._review_history: dict[str, dict[str, Any]] = {}
    target._last_advisory_emit: dict[str, float] = {}
    target._last_main_agent_advisory: dict[str, dict[str, Any]] = {}
    target._job_llm_call_count: dict[str, int] = {}
    target._job_advisory_call_count: dict[str, int] = {}
    target._job_llm_token_count: dict[str, int] = {}
    target._job_last_llm_metric_history_digest: dict[str, str] = {}
    target._last_idle_lease_candidate_emit: dict[str, float] = {}
    target._last_gpu_share_event_emit: dict[str, float] = {}
    target._last_admission_share_review_emit: dict[str, float] = {}
    target.gpu_share_config = build_gpu_share_config(
        enabled=bool(payload["gpu_share_enabled"]),
        phase=str(payload["gpu_share_phase"] or "observe"),
        policy_profile=str(payload["gpu_share_policy_profile"] or "conservative"),
        memory_profile=str(payload["gpu_share_memory_profile"] or "conservative"),
        cpu_policy=str(payload["gpu_share_cpu_policy"] or "conservative"),
        trial_admission_policy=str(
            payload["gpu_trial_admission_policy"] or "llm_grant"
        ),
    )
    target.resource_startup_policy = (
        str(payload["resource_startup_policy"] or "conservative").strip().lower()
    )
    (target.resource_trial_window_sec, target.resource_trial_hard_review_sec) = (
        trial_windows_for_profile(
            profile=target.control_profile.name,
            window_sec=payload["resource_trial_window_sec"],
            hard_review_sec=payload["resource_trial_hard_review_sec"],
        )
    )
    target.kill_proposal_cooldown_sec = float(
        target.control_profile.kill_proposal_cooldown_sec or 600.0
    )
    target._seq = 0
    target.resource_registry: ResourceRegistry[ResourceJob] = ResourceRegistry()
    target._jobs = target.resource_registry.compat_jobs
    module_state_root = (
        Path(payload["task_resource_dir"]) / "module_state" / target.worker_id.lower()
        if payload["task_resource_dir"] is not None
        else None
    )
    clean_admission_source = (
        str(payload["admission_decision_source"] or "component").strip().lower()
    )
    if clean_admission_source not in {"component", "legacy", "shadow"}:
        clean_admission_source = "component"
    clean_value_source = (
        str(payload["execution_value_decision_source"] or "component").strip().lower()
    )
    if clean_value_source not in {"component", "legacy", "shadow"}:
        clean_value_source = "component"
    target.admission_decision_source = clean_admission_source
    target.execution_value_decision_source = clean_value_source
    target.admission_policy = AdmissionPolicyService()
    target.admission_replay = (
        AdmissionReplayArchive(module_state_root / "admission" / "decisions.jsonl")
        if module_state_root is not None
        else None
    )
    target.admission_router = AdmissionDecisionRouter(
        source=clean_admission_source,
        service=target.admission_policy,
        archive=target.admission_replay,
    )
    target.execution_value_shadow = LnrExecutionValueShadow(
        decision_source=clean_value_source,
        archive_path=module_state_root / "execution_value" / "decisions.jsonl"
        if module_state_root is not None
        else None,
    )
    target._agent_backoff_waits: dict[str, dict[str, Any]] = {}
    target._last_agent_backoff_wait: dict[str, Any] | None = None
    target._pending_resource_feedback: dict[str, Any] | None = None
    target._active_plan_guards: dict[str, dict[str, Any]] = {}
    target._planning_resource_context_version: int | None = None
    target._planning_resource_pressure_generation: int | None = None
    target._feedback_seq = 0
    target._reported_resource_feedback: dict[str, dict[str, Any]] = {}
    target._gpu_queue_timeout_count = 0


def _initialize_review_and_observation(target: Any, payload: dict[str, Any]) -> None:
    target.timeout_hard_gate_enabled = bool(payload["timeout_hard_gate_enabled"])
    target.timeout_block_train_after = max(
        1, int(payload["timeout_block_train_after"] or 1)
    )
    target.timeout_tt_only_after = max(
        target.timeout_block_train_after, int(payload["timeout_tt_only_after"] or 2)
    )
    target.review_state_enabled = bool(payload["review_state_enabled"])
    target.review_heartbeat_sec = max(
        1.0, float(payload["review_heartbeat_sec"] or 60.0)
    )
    target.review_config = ResourceReviewConfig(
        warmup_windows=cap_int(
            int(payload["review_warmup_windows"] or 0),
            target.control_profile.review_warmup_windows,
            minimum=0,
        ),
        inactive_windows=max(1, int(payload["review_inactive_windows"] or 1)),
        value_windows=cap_int(
            int(payload["review_value_windows"] or 1),
            target.control_profile.review_value_windows,
            minimum=1,
        ),
        progress_event_min_windows=max(
            1, int(payload["review_progress_event_min_windows"] or 1)
        ),
        timebox_windows=max(1, int(payload["review_timebox_windows"] or 1)),
        max_proof_windows=max(1, int(payload["review_max_proof_windows"] or 2)),
        min_timebox_sec=max(1.0, float(payload["review_min_timebox_sec"] or 60.0)),
        max_timebox_sec=max(1.0, float(payload["review_max_timebox_sec"] or 1800.0)),
        timebox_budget_fraction=min(
            1.0, max(0.01, float(payload["review_timebox_budget_fraction"] or 0.1))
        ),
    ).normalized()
    target._review_states: dict[str, ResourceReviewState] = {}
    target._review_machines: dict[str, ResourceReviewMachine] = {}
    target.observation_enabled = bool(payload["observation_enabled"])
    target.observation_window_sec = max(
        0.05, float(payload["observation_window_sec"] or 30.0)
    )
    target.observation_shadow_workspace_enabled = bool(
        payload["observation_shadow_workspace_enabled"]
    )
    target.gpu_util_observer_enabled = bool(payload["gpu_util_observer_enabled"])
    target.gpu_util_sample_interval_sec = max(
        1.0, float(payload["gpu_util_sample_interval_sec"] or 30.0)
    )
    target.gpu_idle_lease_guard_enabled = bool(payload["gpu_idle_lease_guard_enabled"])
    target.gpu_idle_lease_warmup_sec = cap_float(
        float(payload["gpu_idle_lease_warmup_sec"] or 0.0),
        target.control_profile.gpu_idle_lease_warmup_sec,
        minimum=0.0,
    )
    target.gpu_idle_lease_min_samples = cap_int(
        int(payload["gpu_idle_lease_min_samples"] or 1),
        target.control_profile.gpu_idle_lease_min_samples,
        minimum=1,
    )
    target.gpu_idle_lease_util_pct = max(
        0.0, float(payload["gpu_idle_lease_util_pct"] or 0.0)
    )
    target.gpu_idle_lease_mem_gb = max(
        0.0, float(payload["gpu_idle_lease_mem_gb"] or 0.0)
    )
    target.gpu_idle_lease_require_pressure = bool(
        payload["gpu_idle_lease_require_pressure"]
    )
    target.gpu_idle_lease_action_mode = (
        str(payload["gpu_idle_lease_action_mode"] or "release").strip().lower()
    )
    if target.gpu_idle_lease_action_mode not in {
        "observe",
        "recommend",
        "release",
        "terminate",
    }:
        target.gpu_idle_lease_action_mode = "release"
    target.quick_probe_guard_enabled = bool(payload["quick_probe_guard_enabled"])
    target.quick_probe_expected_runtime_sec = cap_float(
        float(payload["quick_probe_expected_runtime_sec"] or 300.0),
        target.control_profile.quick_probe_expected_runtime_sec,
        minimum=30.0,
    )
    target.quick_probe_hard_review_sec = cap_float(
        float(payload["quick_probe_hard_review_sec"] or 900.0),
        target.control_profile.quick_probe_hard_review_sec,
        minimum=target.quick_probe_expected_runtime_sec,
    )
    target.quick_probe_small_scope_threshold = max(
        1, int(payload["quick_probe_small_scope_threshold"] or 1000)
    )


def _initialize_guards_and_runtime(target: Any, payload: dict[str, Any]) -> None:
    target.gpu_dataloader_bottleneck_guard_enabled = bool(
        payload["gpu_dataloader_bottleneck_guard_enabled"]
    )
    target.gpu_dataloader_bottleneck_warmup_sec = cap_float(
        float(payload["gpu_dataloader_bottleneck_warmup_sec"] or 0.0),
        target.control_profile.gpu_dataloader_bottleneck_warmup_sec,
        minimum=0.0,
    )
    target.gpu_dataloader_bottleneck_min_samples = cap_int(
        int(payload["gpu_dataloader_bottleneck_min_samples"] or 1),
        target.control_profile.gpu_dataloader_bottleneck_min_samples,
        minimum=1,
    )
    target.gpu_dataloader_bottleneck_util_pct = max(
        0.0, float(payload["gpu_dataloader_bottleneck_util_pct"] or 0.0)
    )
    target.gpu_dataloader_bottleneck_min_mem_gb = max(
        0.0, float(payload["gpu_dataloader_bottleneck_min_mem_gb"] or 0.0)
    )
    target.gpu_dataloader_bottleneck_child_cpu_pct = max(
        0.0, float(payload["gpu_dataloader_bottleneck_child_cpu_pct"] or 0.0)
    )
    target.gpu_dataloader_bottleneck_busy_children = max(
        1, int(payload["gpu_dataloader_bottleneck_busy_children"] or 1)
    )
    target.deliverable_completion_guard_enabled = bool(
        payload["deliverable_completion_guard_enabled"]
    )
    target.deliverable_completion_warmup_sec = max(
        0.0, float(payload["deliverable_completion_warmup_sec"] or 0.0)
    )
    target.deliverable_completion_settle_sec = max(
        0.0, float(payload["deliverable_completion_settle_sec"] or 0.0)
    )
    target.deliverable_completion_scan_interval_sec = max(
        0.0, float(payload["deliverable_completion_scan_interval_sec"] or 0.0)
    )
    target.deliverable_completion_quiet_sec = max(
        0.0, float(payload["deliverable_completion_quiet_sec"] or 0.0)
    )
    target.metric_health_guard_enabled = bool(payload["metric_health_guard_enabled"])
    target.metric_health_warmup_sec = max(
        0.0, float(payload["metric_health_warmup_sec"] or 0.0)
    )
    target.metric_health_invalid_min_events = max(
        1, int(payload["metric_health_invalid_min_events"] or 1)
    )
    target.metric_health_zero_score_min_events = max(
        1, int(payload["metric_health_zero_score_min_events"] or 1)
    )
    target.gpu_source_hint_enabled = bool(payload["gpu_source_hint_enabled"])
    target.gpu_source_hint_mode = (
        str(payload["gpu_source_hint_mode"] or "observe").strip().lower()
    )
    if target.gpu_source_hint_mode not in {"observe", "request", "off"}:
        target.gpu_source_hint_mode = "observe"
    target.task_resource_dir = (
        Path(payload["task_resource_dir"])
        if payload["task_resource_dir"] is not None
        else None
    )
    target.resource_runtime: ResourceManagementService | None = None
    if payload["resource_runtime_enabled"] and target.task_resource_dir is not None:
        target.resource_runtime = ResourceManagementService.create(
            worker_id=target.worker_id,
            resource_dir=target.task_resource_dir,
            gpu_queue=GPUQueueConfig(
                enabled=bool(payload["gpu_queue_enabled"]),
                gpu_pool=[str(x) for x in payload["gpu_pool"] or [] if str(x).strip()],
                default_request=int(payload["gpu_default_request"] or 1),
                max_request=int(payload["gpu_max_request"] or 1),
                assignment=str(payload["gpu_assignment"] or "env_only"),
                max_wait_sec=float(payload["gpu_queue_max_wait_sec"] or 0.0),
                heartbeat_sec=float(payload["gpu_queue_heartbeat_sec"] or 15.0),
                max_heavy_per_gpu=int(payload["gpu_max_heavy_per_gpu"] or 1),
                capacity_slots=float(payload["gpu_capacity_slots"] or 1.0),
                gpu_tt_max_per_gpu=int(payload["gpu_tt_max_per_gpu"] or 1),
                gpu_feature_max_per_gpu=int(payload["gpu_feature_max_per_gpu"] or 1),
                share_tt_with_train=bool(payload["gpu_share_tt_with_train"]),
                lease_ttl_sec=float(payload["gpu_lease_ttl_sec"] or 7200.0),
                duplicate_digest_cooldown_sec=float(
                    payload["gpu_duplicate_digest_cooldown_sec"] or 600.0
                ),
                duplicate_digest_threshold=int(
                    payload["gpu_duplicate_digest_threshold"] or 2
                ),
                admission_queue_enabled=bool(payload["gpu_admission_queue_enabled"]),
                admission_waiter_ttl_sec=float(
                    payload["gpu_admission_waiter_ttl_sec"] or 900.0
                ),
                pressure_yellow_hold_sec=float(
                    payload["gpu_pressure_yellow_hold_sec"] or 120.0
                ),
                pressure_red_to_yellow_sec=float(
                    payload["gpu_pressure_red_to_yellow_sec"] or 120.0
                ),
                pressure_yellow_util_pct=float(
                    payload["gpu_pressure_yellow_util_pct"] or 85.0
                ),
                pressure_min_free_mem_gb=float(
                    payload["gpu_pressure_min_free_mem_gb"] or 0.0
                ),
                pressure_yellow_free_mem_buffer_gb=float(
                    payload["gpu_pressure_yellow_free_mem_buffer_gb"] or 8.0
                ),
                idle_release_admission_mode=str(
                    payload["resource_idle_release_admission_mode"]
                    or "strict_exclusive"
                ),
            ),
        )


def initialize_resource_observer(target: Any, payload: dict[str, Any]) -> None:
    _initialize_control_and_advisory(target, payload)
    _initialize_admission_and_monitoring(target, payload)
    _initialize_services_and_state(target, payload)
    _initialize_review_and_observation(target, payload)
    _initialize_guards_and_runtime(target, payload)


__all__ = ("initialize_resource_observer",)
