# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Extracted coordinator component with an explicit dependency surface."""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    LHRResourceObserver,
)


def _resource_observer_options_1(
    self,
    *,
    arbiter_enabled_cfg: bool,
    gpu_share_enabled_cfg: bool,
    gpu_share_phase_cfg: str,
    contention_enabled_cfg: bool,
    contention_min_runtime_sec_cfg: float,
    contention_min_waiter_age_sec_cfg: float,
    contention_min_interval_sec_cfg: float,
) -> dict[str, Any]:
    return {
        "state_machine": self.state_machine,
        "worker_id": self.worker_id or "W00",
        "resource_control_profile": str(
            getattr(self.lhr, "resource_control_profile", "normal") or "normal"
        ),
        "min_register_sec": float(
            getattr(self.lhr, "resource_monitor_min_register_sec", 600.0) or 0.0
        ),
        "check_interval_sec": float(
            getattr(self.lhr, "resource_monitor_check_interval_sec", 10.0) or 10.0
        ),
        "stalled_stdout_sec": float(
            getattr(self.lhr, "resource_monitor_stalled_stdout_sec", 900.0) or 0.0
        ),
        "kill_enabled": bool(getattr(self.lhr, "resource_monitor_kill_enabled", True)),
        "kill_mode": str(
            getattr(self.lhr, "resource_monitor_kill_mode", "recommend") or "recommend"
        ),
        "monitor_agent_mode": str(
            getattr(self.lhr, "resource_monitor_agent_mode", "off") or "off"
        ),
        "monitor_agent_min_interval_sec": float(
            getattr(self.lhr, "resource_monitor_agent_min_interval_sec", 60.0) or 60.0
        ),
        "low_progress_enabled": bool(
            getattr(self.lhr, "resource_monitor_low_progress_enabled", True)
        ),
        "low_progress_warmup_sec": float(
            getattr(self.lhr, "resource_monitor_low_progress_warmup_sec", 1800.0) or 0.0
        ),
        "low_progress_no_heartbeat_sec": float(
            getattr(self.lhr, "resource_monitor_low_progress_no_heartbeat_sec", 1800.0)
            or 0.0
        ),
        "low_progress_no_artifact_sec": float(
            getattr(self.lhr, "resource_monitor_low_progress_no_artifact_sec", 1800.0)
            or 0.0
        ),
        "review_state_enabled": bool(
            getattr(self.lhr, "resource_review_state_enabled", True)
        ),
        "review_heartbeat_sec": float(
            getattr(self.lhr, "resource_review_heartbeat_sec", 60.0) or 60.0
        ),
        "review_warmup_windows": int(
            getattr(self.lhr, "resource_review_warmup_windows", 10) or 0
        ),
        "review_inactive_windows": int(
            getattr(self.lhr, "resource_review_inactive_windows", 3) or 1
        ),
        "review_value_windows": int(
            getattr(self.lhr, "resource_review_value_windows", 5) or 1
        ),
        "review_progress_event_min_windows": int(
            getattr(self.lhr, "resource_review_progress_event_min_windows", 5) or 1
        ),
        "review_timebox_windows": int(
            getattr(self.lhr, "resource_review_timebox_windows", 10) or 1
        ),
        "review_max_proof_windows": int(
            getattr(self.lhr, "resource_review_max_proof_windows", 2) or 2
        ),
    }


def _resource_observer_options_2(
    self,
    *,
    arbiter_enabled_cfg: bool,
    gpu_share_enabled_cfg: bool,
    gpu_share_phase_cfg: str,
    contention_enabled_cfg: bool,
    contention_min_runtime_sec_cfg: float,
    contention_min_waiter_age_sec_cfg: float,
    contention_min_interval_sec_cfg: float,
) -> dict[str, Any]:
    return {
        "review_min_timebox_sec": float(
            getattr(self.lhr, "resource_review_min_timebox_sec", 60.0) or 60.0
        ),
        "review_max_timebox_sec": float(
            getattr(self.lhr, "resource_review_max_timebox_sec", 1800.0) or 1800.0
        ),
        "review_timebox_budget_fraction": float(
            getattr(self.lhr, "resource_review_timebox_budget_fraction", 0.10) or 0.10
        ),
        "bash_monitor_all_enabled": bool(
            getattr(self.lhr, "resource_bash_monitor_all_enabled", True)
        ),
        "arbiter_min_progress_windows": int(
            getattr(self.lhr, "resource_arbiter_min_progress_windows", 2) or 2
        ),
        "arbiter_kill_requires_high_confidence": bool(
            getattr(self.lhr, "resource_arbiter_kill_requires_high_confidence", True)
        ),
        "task_resource_dir": self.global_log_dir / "resource",
        "resource_runtime_enabled": bool(
            getattr(self.lhr, "resource_runtime_enabled", True)
        ),
        "gpu_queue_enabled": bool(
            getattr(self.lhr, "resource_gpu_queue_enabled", True)
        ),
        "gpu_pool": [
            str(x)
            for x in (getattr(self.lhr, "resource_gpu_pool", []) or [])
            if str(x).strip()
        ],
        "gpu_default_request": int(
            getattr(self.lhr, "resource_gpu_default_request", 1) or 1
        ),
        "gpu_max_request": int(getattr(self.lhr, "resource_gpu_max_request", 1) or 1),
        "gpu_assignment": str(
            getattr(self.lhr, "resource_gpu_assignment", "lease") or "lease"
        ),
        "gpu_queue_max_wait_sec": float(
            getattr(self.lhr, "resource_gpu_queue_max_wait_sec", 1800.0) or 0.0
        ),
        "gpu_queue_heartbeat_sec": float(
            getattr(self.lhr, "resource_gpu_queue_heartbeat_sec", 15.0) or 15.0
        ),
        "gpu_max_heavy_per_gpu": int(
            getattr(self.lhr, "resource_gpu_max_heavy_per_gpu", 1) or 1
        ),
        "gpu_capacity_slots": float(
            getattr(self.lhr, "resource_gpu_capacity_slots", 1.0) or 1.0
        ),
        "gpu_tt_max_per_gpu": int(
            getattr(self.lhr, "resource_gpu_tt_max_per_gpu", 3) or 3
        ),
        "gpu_feature_max_per_gpu": int(
            getattr(self.lhr, "resource_gpu_feature_max_per_gpu", 2) or 2
        ),
        "gpu_share_tt_with_train": bool(
            getattr(self.lhr, "resource_gpu_share_tt_with_train", False)
        ),
        "gpu_share_enabled": gpu_share_enabled_cfg,
        "gpu_share_phase": gpu_share_phase_cfg,
    }


def _resource_observer_options_3(
    self,
    *,
    arbiter_enabled_cfg: bool,
    gpu_share_enabled_cfg: bool,
    gpu_share_phase_cfg: str,
    contention_enabled_cfg: bool,
    contention_min_runtime_sec_cfg: float,
    contention_min_waiter_age_sec_cfg: float,
    contention_min_interval_sec_cfg: float,
) -> dict[str, Any]:
    return {
        "gpu_share_policy_profile": str(
            getattr(self.lhr, "resource_gpu_share_policy_profile", "conservative")
            or "conservative"
        ),
        "gpu_share_memory_profile": str(
            getattr(self.lhr, "resource_gpu_share_memory_profile", "conservative")
            or "conservative"
        ),
        "gpu_share_cpu_policy": str(
            getattr(self.lhr, "resource_gpu_share_cpu_policy", "conservative")
            or "conservative"
        ),
        "gpu_trial_admission_policy": str(
            getattr(self.lhr, "resource_gpu_trial_admission_policy", "llm_grant")
            or "llm_grant"
        ),
        "resource_startup_policy": str(
            getattr(self.lhr, "resource_startup_policy", "trial_first") or "trial_first"
        ),
        "resource_trial_window_sec": float(
            getattr(self.lhr, "resource_trial_window_sec", 300.0) or 300.0
        ),
        "resource_trial_hard_review_sec": float(
            getattr(self.lhr, "resource_trial_hard_review_sec", 900.0) or 900.0
        ),
        "gpu_lease_ttl_sec": float(
            getattr(self.lhr, "resource_gpu_lease_ttl_sec", 7200.0) or 7200.0
        ),
        "gpu_duplicate_digest_cooldown_sec": float(
            getattr(self.lhr, "resource_gpu_duplicate_digest_cooldown_sec", 600.0)
            or 600.0
        ),
        "gpu_duplicate_digest_threshold": int(
            getattr(self.lhr, "resource_gpu_duplicate_digest_threshold", 2) or 2
        ),
        "gpu_admission_queue_enabled": bool(
            getattr(self.lhr, "resource_gpu_admission_queue_enabled", True)
        ),
        "gpu_admission_waiter_ttl_sec": float(
            getattr(self.lhr, "resource_gpu_admission_waiter_ttl_sec", 900.0) or 900.0
        ),
        "admission_llm_enabled": bool(
            getattr(self.lhr, "resource_admission_llm_enabled", False)
        ),
        "admission_llm_mode": str(
            getattr(self.lhr, "resource_admission_llm_mode", "low_confidence")
            or "low_confidence"
        ),
        "admission_llm_timeout_sec": float(
            getattr(self.lhr, "resource_admission_llm_timeout_sec", 60.0) or 60.0
        ),
        "admission_decider": self._make_resource_admission_decider(),
        "admission_decision_source": str(
            getattr(self.lhr, "resource_admission_decision_source", "component")
            or "component"
        ),
        "execution_value_decision_source": str(
            getattr(
                self.lhr,
                "resource_execution_value_decision_source",
                "component",
            )
            or "component"
        ),
        "stale_pressure_observe_first_enabled": bool(
            getattr(self.lhr, "resource_stale_pressure_observe_first_enabled", True)
        ),
        "stale_pressure_observe_window_sec": float(
            getattr(self.lhr, "resource_stale_pressure_observe_window_sec", 180.0)
            or 180.0
        ),
        "stale_pressure_observe_max_sec": float(
            getattr(self.lhr, "resource_stale_pressure_observe_max_sec", 300.0) or 300.0
        ),
        "stale_pressure_observe_min_free_mem_gb": float(
            getattr(self.lhr, "resource_stale_pressure_observe_min_free_mem_gb", 8.0)
            or 0.0
        ),
    }


def _resource_observer_options_4(
    self,
    *,
    arbiter_enabled_cfg: bool,
    gpu_share_enabled_cfg: bool,
    gpu_share_phase_cfg: str,
    contention_enabled_cfg: bool,
    contention_min_runtime_sec_cfg: float,
    contention_min_waiter_age_sec_cfg: float,
    contention_min_interval_sec_cfg: float,
) -> dict[str, Any]:
    return {
        "stale_pressure_healthy_skip_llm": bool(
            getattr(self.lhr, "resource_stale_pressure_healthy_skip_llm", True)
        ),
        "gpu_pressure_yellow_hold_sec": float(
            getattr(self.lhr, "resource_gpu_pressure_yellow_hold_sec", 120.0) or 120.0
        ),
        "gpu_pressure_red_to_yellow_sec": float(
            getattr(self.lhr, "resource_gpu_pressure_red_to_yellow_sec", 120.0) or 120.0
        ),
        "gpu_pressure_yellow_util_pct": float(
            getattr(self.lhr, "resource_gpu_pressure_yellow_util_pct", 85.0) or 85.0
        ),
        "gpu_pressure_min_free_mem_gb": float(
            getattr(self.lhr, "resource_gpu_pressure_min_free_mem_gb", 8.0) or 0.0
        ),
        "gpu_pressure_yellow_free_mem_buffer_gb": float(
            getattr(self.lhr, "resource_gpu_pressure_yellow_free_mem_buffer_gb", 8.0)
            or 8.0
        ),
        "observation_enabled": bool(
            getattr(self.lhr, "resource_observation_enabled", True)
        ),
        "observation_window_sec": float(
            getattr(self.lhr, "resource_observation_window_sec", 30.0) or 30.0
        ),
        "observation_shadow_workspace_enabled": bool(
            getattr(self.lhr, "resource_observation_shadow_workspace_enabled", True)
        ),
        "gpu_source_hint_enabled": bool(
            getattr(self.lhr, "resource_gpu_source_hint_enabled", True)
        ),
        "gpu_source_hint_mode": str(
            getattr(self.lhr, "resource_gpu_source_hint_mode", "observe") or "observe"
        ),
        "gpu_util_observer_enabled": bool(
            getattr(self.lhr, "resource_gpu_util_observer_enabled", True)
        ),
        "gpu_util_sample_interval_sec": float(
            getattr(self.lhr, "resource_gpu_util_sample_interval_sec", 30.0) or 30.0
        ),
        "gpu_idle_lease_guard_enabled": bool(
            getattr(self.lhr, "resource_gpu_idle_lease_guard_enabled", True)
        ),
        "gpu_idle_lease_warmup_sec": float(
            getattr(self.lhr, "resource_gpu_idle_lease_warmup_sec", 180.0) or 0.0
        ),
        "gpu_idle_lease_min_samples": int(
            getattr(self.lhr, "resource_gpu_idle_lease_min_samples", 3) or 0
        ),
        "gpu_idle_lease_util_pct": float(
            getattr(self.lhr, "resource_gpu_idle_lease_util_pct", 1.0) or 0.0
        ),
        "gpu_idle_lease_mem_gb": float(
            getattr(self.lhr, "resource_gpu_idle_lease_mem_gb", 1.0) or 0.0
        ),
        "gpu_idle_lease_require_pressure": bool(
            getattr(self.lhr, "resource_gpu_idle_lease_require_pressure", False)
        ),
        "gpu_idle_lease_action_mode": str(
            getattr(self.lhr, "resource_gpu_idle_lease_action_mode", "release")
            or "release"
        ),
        "resource_idle_release_admission_mode": str(
            getattr(
                self.lhr, "resource_idle_release_admission_mode", "strict_exclusive"
            )
            or "strict_exclusive"
        ),
        "quick_probe_guard_enabled": bool(
            getattr(self.lhr, "resource_quick_probe_guard_enabled", True)
        ),
    }


def _resource_observer_options_5(
    self,
    *,
    arbiter_enabled_cfg: bool,
    gpu_share_enabled_cfg: bool,
    gpu_share_phase_cfg: str,
    contention_enabled_cfg: bool,
    contention_min_runtime_sec_cfg: float,
    contention_min_waiter_age_sec_cfg: float,
    contention_min_interval_sec_cfg: float,
) -> dict[str, Any]:
    return {
        "quick_probe_expected_runtime_sec": float(
            getattr(self.lhr, "resource_quick_probe_expected_runtime_sec", 300.0)
            or 300.0
        ),
        "quick_probe_hard_review_sec": float(
            getattr(self.lhr, "resource_quick_probe_hard_review_sec", 900.0) or 900.0
        ),
        "quick_probe_small_scope_threshold": int(
            getattr(self.lhr, "resource_quick_probe_small_scope_threshold", 1000)
            or 1000
        ),
        "gpu_dataloader_bottleneck_guard_enabled": bool(
            getattr(self.lhr, "resource_gpu_dataloader_bottleneck_guard_enabled", True)
        ),
        "gpu_dataloader_bottleneck_warmup_sec": float(
            getattr(self.lhr, "resource_gpu_dataloader_bottleneck_warmup_sec", 600.0)
            or 0.0
        ),
        "gpu_dataloader_bottleneck_min_samples": int(
            getattr(self.lhr, "resource_gpu_dataloader_bottleneck_min_samples", 3) or 1
        ),
        "gpu_dataloader_bottleneck_util_pct": float(
            getattr(self.lhr, "resource_gpu_dataloader_bottleneck_util_pct", 15.0)
            or 0.0
        ),
        "gpu_dataloader_bottleneck_min_mem_gb": float(
            getattr(self.lhr, "resource_gpu_dataloader_bottleneck_min_mem_gb", 2.0)
            or 0.0
        ),
        "gpu_dataloader_bottleneck_child_cpu_pct": float(
            getattr(self.lhr, "resource_gpu_dataloader_bottleneck_child_cpu_pct", 200.0)
            or 0.0
        ),
        "gpu_dataloader_bottleneck_busy_children": int(
            getattr(self.lhr, "resource_gpu_dataloader_bottleneck_busy_children", 2)
            or 1
        ),
        "deliverable_completion_guard_enabled": bool(
            getattr(self.lhr, "resource_deliverable_completion_guard_enabled", True)
        ),
        "deliverable_completion_warmup_sec": float(
            getattr(self.lhr, "resource_deliverable_completion_warmup_sec", 120.0)
            or 0.0
        ),
        "deliverable_completion_settle_sec": float(
            getattr(self.lhr, "resource_deliverable_completion_settle_sec", 120.0)
            or 0.0
        ),
        "deliverable_completion_scan_interval_sec": float(
            getattr(self.lhr, "resource_deliverable_completion_scan_interval_sec", 60.0)
            or 0.0
        ),
        "deliverable_completion_quiet_sec": float(
            getattr(self.lhr, "resource_deliverable_completion_quiet_sec", 60.0) or 0.0
        ),
        "metric_health_guard_enabled": bool(
            getattr(self.lhr, "resource_metric_health_guard_enabled", True)
        ),
        "metric_health_warmup_sec": float(
            getattr(self.lhr, "resource_metric_health_warmup_sec", 600.0) or 0.0
        ),
        "metric_health_invalid_min_events": int(
            getattr(self.lhr, "resource_metric_health_invalid_min_events", 2) or 1
        ),
        "metric_health_zero_score_min_events": int(
            getattr(self.lhr, "resource_metric_health_zero_score_min_events", 2) or 1
        ),
        "arbiter_enabled": arbiter_enabled_cfg,
        "arbiter_mode": str(
            getattr(self.lhr, "resource_arbiter_mode", "policy") or "policy"
        ),
        "arbiter_timeout_sec": float(
            getattr(self.lhr, "resource_arbiter_timeout_sec", 90.0) or 90.0
        ),
    }


def _resource_observer_options_6(
    self,
    *,
    arbiter_enabled_cfg: bool,
    gpu_share_enabled_cfg: bool,
    gpu_share_phase_cfg: str,
    contention_enabled_cfg: bool,
    contention_min_runtime_sec_cfg: float,
    contention_min_waiter_age_sec_cfg: float,
    contention_min_interval_sec_cfg: float,
) -> dict[str, Any]:
    return {
        "arbiter_decider": self._make_resource_arbiter_decider(),
        "main_agent_advisory_enabled": bool(
            getattr(self.lhr, "resource_main_agent_advisory_enabled", False)
        ),
        "main_agent_advisory_min_interval_sec": float(
            getattr(self.lhr, "resource_main_agent_advisory_min_interval_sec", 600.0)
            or 600.0
        ),
        "main_agent_advisory_timeout_sec": float(
            getattr(self.lhr, "resource_main_agent_advisory_timeout_sec", 60.0) or 60.0
        ),
        "main_agent_advisory_decider": self._make_resource_main_agent_advisory_decider(),
        "arbiter_contention_review_enabled": contention_enabled_cfg,
        "arbiter_contention_min_runtime_sec": contention_min_runtime_sec_cfg,
        "arbiter_contention_min_waiter_age_sec": contention_min_waiter_age_sec_cfg,
        "arbiter_contention_min_interval_sec": contention_min_interval_sec_cfg,
        "research_cadence_enabled": bool(
            getattr(self.lhr, "resource_research_cadence_enabled", True)
        ),
        "research_cadence_observe_sec": float(
            getattr(self.lhr, "resource_research_cadence_observe_sec", 300.0) or 300.0
        ),
        "first_comparable_metric_budget_sec": float(
            getattr(self.lhr, "resource_first_comparable_metric_budget_sec", 900.0)
            or 900.0
        ),
        "proven_route_metric_budget_sec": float(
            getattr(self.lhr, "resource_proven_route_metric_budget_sec", 1800.0)
            or 1800.0
        ),
        "arbiter_periodic_review_enabled": bool(
            getattr(self.lhr, "resource_arbiter_periodic_review_enabled", False)
        ),
        "arbiter_periodic_min_runtime_sec": float(
            getattr(self.lhr, "resource_arbiter_periodic_min_runtime_sec", 1800.0)
            or 0.0
        ),
        "arbiter_periodic_min_interval_sec": float(
            getattr(self.lhr, "resource_arbiter_periodic_min_interval_sec", 900.0)
            or 900.0
        ),
        "arbiter_proposal_coalesce_window_sec": float(
            getattr(self.lhr, "resource_arbiter_proposal_coalesce_window_sec", 60.0)
            or 0.0
        ),
        "arbiter_job_llm_call_cap": int(
            getattr(self.lhr, "resource_arbiter_job_llm_call_cap", 6) or 0
        ),
        "arbiter_job_advisory_call_cap": int(
            getattr(self.lhr, "resource_arbiter_job_advisory_call_cap", 3) or 0
        ),
        "arbiter_job_token_cap": int(
            getattr(self.lhr, "resource_arbiter_job_token_cap", 24000) or 0
        ),
        "sidecar_enabled": bool(getattr(self.lhr, "resource_sidecar_enabled", False)),
        "sidecar_min_parent_runtime_sec": float(
            getattr(self.lhr, "resource_sidecar_min_parent_runtime_sec", 900.0) or 900.0
        ),
    }


def _resource_observer_options_7(
    self,
    *,
    arbiter_enabled_cfg: bool,
    gpu_share_enabled_cfg: bool,
    gpu_share_phase_cfg: str,
    contention_enabled_cfg: bool,
    contention_min_runtime_sec_cfg: float,
    contention_min_waiter_age_sec_cfg: float,
    contention_min_interval_sec_cfg: float,
) -> dict[str, Any]:
    return {
        "estra_magent_enabled": bool(getattr(self.lhr, "estra_magent_enabled", False)),
        "estra_magent_sidecar_enabled": bool(
            getattr(self.lhr, "estra_magent_sidecar_enabled", False)
        ),
        "estra_magent_min_parent_runtime_sec": float(
            getattr(self.lhr, "estra_magent_min_parent_runtime_sec", 300.0) or 300.0
        ),
        "estra_magent_sidecar_mode": str(
            getattr(self.lhr, "estra_magent_sidecar_mode", "cpu_only") or "cpu_only"
        ),
        "estra_magent_budget_sec": float(
            getattr(self.lhr, "estra_magent_budget_sec", 900.0) or 900.0
        ),
        "estra_magent_join_inject_parent": bool(
            getattr(self.lhr, "estra_magent_join_inject_parent", True)
        ),
        "estra_magent_join_inject_estra": bool(
            getattr(self.lhr, "estra_magent_join_inject_estra", True)
        ),
        "estra_magent_join_inject_resource_context": bool(
            getattr(self.lhr, "estra_magent_join_inject_resource_context", True)
        ),
        "checkpoint_submission_guard_enabled": bool(
            getattr(self.lhr, "resource_checkpoint_submission_guard_enabled", True)
        ),
        "timeout_hard_gate_enabled": bool(
            getattr(self.lhr, "resource_queue_timeout_hard_gate_enabled", True)
        ),
        "timeout_block_train_after": int(
            getattr(self.lhr, "resource_queue_timeout_block_train_after", 1) or 1
        ),
        "timeout_tt_only_after": int(
            getattr(self.lhr, "resource_queue_timeout_tt_only_after", 2) or 2
        ),
    }


def _make_resource_observer(self) -> LHRResourceObserver | None:
    if not bool(getattr(self.lhr, "resource_monitor_enabled", True)):
        return None
    arbiter_enabled_cfg = bool(getattr(self.lhr, "resource_arbiter_enabled", False))
    gpu_share_enabled_cfg = bool(getattr(self.lhr, "resource_gpu_share_enabled", False))
    gpu_share_phase_cfg = str(
        getattr(self.lhr, "resource_gpu_share_phase", "observe") or "observe"
    )
    contention_enabled_cfg = bool(
        getattr(self.lhr, "resource_arbiter_contention_review_enabled", False)
    )
    if (
        arbiter_enabled_cfg
        and gpu_share_enabled_cfg
        and gpu_share_phase_cfg in {"tt_share", "feature_share", "light_train"}
    ):
        contention_enabled_cfg = True
    contention_min_runtime_sec_cfg = float(
        getattr(self.lhr, "resource_arbiter_contention_min_runtime_sec", 900.0) or 0.0
    )
    contention_min_waiter_age_sec_cfg = float(
        getattr(self.lhr, "resource_arbiter_contention_min_waiter_age_sec", 300.0)
        or 0.0
    )
    contention_min_interval_sec_cfg = float(
        getattr(self.lhr, "resource_arbiter_contention_min_interval_sec", 600.0) or 0.0
    )
    if contention_enabled_cfg and gpu_share_enabled_cfg:
        contention_min_runtime_sec_cfg = min(contention_min_runtime_sec_cfg, 300.0)
        contention_min_waiter_age_sec_cfg = min(contention_min_waiter_age_sec_cfg, 30.0)
        contention_min_interval_sec_cfg = min(contention_min_interval_sec_cfg, 120.0)
    options: dict[str, Any] = {}
    option_builders = (
        _resource_observer_options_1,
        _resource_observer_options_2,
        _resource_observer_options_3,
        _resource_observer_options_4,
        _resource_observer_options_5,
        _resource_observer_options_6,
        _resource_observer_options_7,
    )
    for build_options in option_builders:
        options.update(
            build_options(
                self,
                arbiter_enabled_cfg=arbiter_enabled_cfg,
                gpu_share_enabled_cfg=gpu_share_enabled_cfg,
                gpu_share_phase_cfg=gpu_share_phase_cfg,
                contention_enabled_cfg=contention_enabled_cfg,
                contention_min_runtime_sec_cfg=contention_min_runtime_sec_cfg,
                contention_min_waiter_age_sec_cfg=contention_min_waiter_age_sec_cfg,
                contention_min_interval_sec_cfg=contention_min_interval_sec_cfg,
            )
        )
    return LHRResourceObserver(**options)
