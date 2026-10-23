"""Resource observer responsibility: configuration, state stores, service ports, and effect execution.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    LHRStateMachineStore,
    Path,
    ResourceEffectCommand,
    ResourceEffectKind,
)
from scienceflow.research.solver.lnr.resources.runtime.observer.construction_runtime import (
    initialize_resource_observer,
)


class ObserverConstruction:
    """Own this responsibility's state transitions and callbacks."""

    def __init__(
        self,
        *,
        state_machine: LHRStateMachineStore,
        worker_id: str,
        resource_control_profile: str = "normal",
        min_register_sec: float = 600.0,
        check_interval_sec: float = 10.0,
        stalled_stdout_sec: float = 900.0,
        kill_enabled: bool = True,
        kill_mode: str = "recommend",
        monitor_agent_mode: str = "off",
        monitor_agent_min_interval_sec: float = 60.0,
        low_progress_enabled: bool = True,
        low_progress_warmup_sec: float = 900.0,
        low_progress_no_heartbeat_sec: float = 900.0,
        low_progress_no_artifact_sec: float = 900.0,
        bash_monitor_all_enabled: bool = False,
        arbiter_min_progress_windows: int = 2,
        arbiter_kill_requires_high_confidence: bool = True,
        task_resource_dir: str | Path | None = None,
        resource_runtime_enabled: bool = False,
        gpu_queue_enabled: bool = False,
        gpu_pool: list[str] | None = None,
        gpu_default_request: int = 1,
        gpu_max_request: int = 1,
        gpu_assignment: str = "env_only",
        gpu_queue_max_wait_sec: float = 1800.0,
        gpu_queue_heartbeat_sec: float = 15.0,
        gpu_max_heavy_per_gpu: int = 1,
        gpu_capacity_slots: float = 1.0,
        gpu_tt_max_per_gpu: int = 3,
        gpu_feature_max_per_gpu: int = 2,
        gpu_share_tt_with_train: bool = False,
        gpu_share_enabled: bool = False,
        gpu_share_phase: str = "observe",
        gpu_share_policy_profile: str = "conservative",
        gpu_share_memory_profile: str = "conservative",
        gpu_share_cpu_policy: str = "conservative",
        gpu_trial_admission_policy: str = "llm_grant",
        resource_startup_policy: str = "conservative",
        resource_trial_window_sec: float = 300.0,
        resource_trial_hard_review_sec: float = 900.0,
        gpu_lease_ttl_sec: float = 7200.0,
        gpu_duplicate_digest_cooldown_sec: float = 600.0,
        gpu_duplicate_digest_threshold: int = 2,
        gpu_admission_queue_enabled: bool = True,
        gpu_admission_waiter_ttl_sec: float = 900.0,
        admission_llm_enabled: bool = False,
        admission_llm_mode: str = "low_confidence",
        admission_llm_timeout_sec: float = 60.0,
        admission_decider: Any | None = None,
        admission_decision_source: str = "component",
        execution_value_decision_source: str = "component",
        stale_pressure_observe_first_enabled: bool = True,
        stale_pressure_observe_window_sec: float = 180.0,
        stale_pressure_observe_max_sec: float = 300.0,
        stale_pressure_observe_min_free_mem_gb: float = 8.0,
        stale_pressure_healthy_skip_llm: bool = True,
        gpu_pressure_yellow_hold_sec: float = 120.0,
        gpu_pressure_red_to_yellow_sec: float = 120.0,
        gpu_pressure_yellow_util_pct: float = 85.0,
        gpu_pressure_min_free_mem_gb: float = 8.0,
        gpu_pressure_yellow_free_mem_buffer_gb: float = 8.0,
        observation_enabled: bool = True,
        observation_window_sec: float = 30.0,
        observation_shadow_workspace_enabled: bool = True,
        gpu_source_hint_enabled: bool = True,
        gpu_source_hint_mode: str = "observe",
        gpu_util_observer_enabled: bool = True,
        gpu_util_sample_interval_sec: float = 30.0,
        gpu_idle_lease_guard_enabled: bool = True,
        gpu_idle_lease_warmup_sec: float = 180.0,
        gpu_idle_lease_min_samples: int = 3,
        gpu_idle_lease_util_pct: float = 1.0,
        gpu_idle_lease_mem_gb: float = 1.0,
        gpu_idle_lease_require_pressure: bool = False,
        gpu_idle_lease_action_mode: str = "release",
        resource_idle_release_admission_mode: str = "strict_exclusive",
        quick_probe_guard_enabled: bool = True,
        quick_probe_expected_runtime_sec: float = 300.0,
        quick_probe_hard_review_sec: float = 900.0,
        quick_probe_small_scope_threshold: int = 1000,
        gpu_dataloader_bottleneck_guard_enabled: bool = True,
        gpu_dataloader_bottleneck_warmup_sec: float = 600.0,
        gpu_dataloader_bottleneck_min_samples: int = 3,
        gpu_dataloader_bottleneck_util_pct: float = 15.0,
        gpu_dataloader_bottleneck_min_mem_gb: float = 2.0,
        gpu_dataloader_bottleneck_child_cpu_pct: float = 200.0,
        gpu_dataloader_bottleneck_busy_children: int = 2,
        deliverable_completion_guard_enabled: bool = True,
        deliverable_completion_warmup_sec: float = 120.0,
        deliverable_completion_settle_sec: float = 120.0,
        deliverable_completion_scan_interval_sec: float = 60.0,
        deliverable_completion_quiet_sec: float = 60.0,
        metric_health_guard_enabled: bool = True,
        metric_health_warmup_sec: float = 600.0,
        metric_health_invalid_min_events: int = 2,
        metric_health_zero_score_min_events: int = 2,
        arbiter_enabled: bool = False,
        arbiter_mode: str = "policy",
        arbiter_timeout_sec: float = 90.0,
        arbiter_decider: Any | None = None,
        main_agent_advisory_enabled: bool = False,
        main_agent_advisory_min_interval_sec: float = 600.0,
        main_agent_advisory_timeout_sec: float = 60.0,
        main_agent_advisory_decider: Any | None = None,
        arbiter_contention_review_enabled: bool = False,
        arbiter_contention_min_runtime_sec: float = 900.0,
        arbiter_contention_min_waiter_age_sec: float = 300.0,
        arbiter_contention_min_interval_sec: float = 600.0,
        research_cadence_enabled: bool = True,
        research_cadence_observe_sec: float = 300.0,
        first_comparable_metric_budget_sec: float = 900.0,
        proven_route_metric_budget_sec: float = 1800.0,
        arbiter_periodic_review_enabled: bool = False,
        arbiter_periodic_min_runtime_sec: float = 1800.0,
        arbiter_periodic_min_interval_sec: float = 900.0,
        arbiter_proposal_coalesce_window_sec: float = 60.0,
        arbiter_job_llm_call_cap: int = 6,
        arbiter_job_advisory_call_cap: int = 3,
        arbiter_job_token_cap: int = 24000,
        sidecar_enabled: bool = False,
        sidecar_min_parent_runtime_sec: float = 900.0,
        estra_magent_enabled: bool = False,
        estra_magent_sidecar_enabled: bool = False,
        estra_magent_min_parent_runtime_sec: float = 300.0,
        estra_magent_sidecar_mode: str = "cpu_only",
        estra_magent_budget_sec: float = 900.0,
        estra_magent_join_inject_parent: bool = True,
        estra_magent_join_inject_estra: bool = True,
        estra_magent_join_inject_resource_context: bool = True,
        checkpoint_submission_guard_enabled: bool = True,
        timeout_hard_gate_enabled: bool = True,
        timeout_block_train_after: int = 1,
        timeout_tt_only_after: int = 2,
        review_state_enabled: bool = True,
        review_heartbeat_sec: float = 60.0,
        review_warmup_windows: int = 10,
        review_inactive_windows: int = 3,
        review_value_windows: int = 5,
        review_progress_event_min_windows: int = 5,
        review_timebox_windows: int = 10,
        review_max_proof_windows: int = 2,
        review_min_timebox_sec: float = 60.0,
        review_max_timebox_sec: float = 1800.0,
        review_timebox_budget_fraction: float = 0.1,
    ) -> None:
        initialize_resource_observer(self, locals())


    def set_planning_resource_context_version(
        self, version: int | None, *, pressure_generation: int | None = None, **_: Any
    ) -> None:
        try:
            self._planning_resource_context_version = (
                int(version) if version is not None else None
            )
        except (TypeError, ValueError):
            self._planning_resource_context_version = None
        try:
            self._planning_resource_pressure_generation = (
                int(pressure_generation) if pressure_generation is not None else None
            )
        except (TypeError, ValueError):
            self._planning_resource_pressure_generation = None


    def planning_resource_context_version(self, **_: Any) -> int | None:
        return self._planning_resource_context_version


    def planning_resource_pressure_generation(self, **_: Any) -> int | None:
        return self._planning_resource_pressure_generation


    def _execute_resource_effect(
        self,
        kind: ResourceEffectKind,
        *,
        command_id: str,
        idempotency_key: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Compatibility facade for validated Runtime-owned effects."""
        if self.resource_runtime is None:
            return {"applied": False, "reason": "resource_runtime_unavailable"}
        event = self.resource_runtime.execute_effect(
            ResourceEffectCommand(
                command_id=str(command_id or ""),
                kind=kind,
                idempotency_key=str(idempotency_key or ""),
                validated=True,
                parameters=dict(parameters),
            )
        )
        return dict(event.result)


    def lease_assignment_enabled(self, **_: Any) -> bool:
        return bool(
            self.resource_runtime and self.resource_runtime.lease_assignment_enabled()
        )
