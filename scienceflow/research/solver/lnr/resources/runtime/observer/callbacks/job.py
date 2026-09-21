# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Resource observer responsibility: job creation, startup trials, and observe-first state.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    Path,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ResourceJob,
    ResourceReviewMachine,
    ResourceSourceHint,
    TRACKABLE_CLASSES,
    classify_preflight_block,
    classify_quick_probe_command,
    detect_resource_source_hint,
    new_review_state,
    startup_policy_trial_first,
    time,
)


class JobCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def job_created(
        self,
        *,
        command: str,
        inferred_class: str,
        gpu_ids: list[str] | None = None,
        cpu_set: str | None = None,
        timeout_sec: float | int | None = None,
        workspace_dir: str | Path | None = None,
        classifier_reason: str | None = None,
        value_hint_override: dict[str, Any] | None = None,
        candidate_artifact: str = "",
    ) -> str | None:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        source_hint = ResourceSourceHint()
        if self.gpu_source_hint_enabled and self.gpu_source_hint_mode != "off":
            source_hint = detect_resource_source_hint(
                command=command,
                workspace_dir=Path(workspace_dir) if workspace_dir is not None else None,
            )
        if bool(getattr(source_hint, "command_cpu_only", False)):
            ids = []
        resource_class = self._refine_resource_class(str(inferred_class or ""), source_hint=source_hint)
        command_digest = self._digest(command)
        post_feedback_gate = self._emit_post_feedback_action(
            command=command,
            resource_class=resource_class,
            command_digest=command_digest,
            gpu_ids=ids,
            source_hint=source_hint,
        )
        if resource_class not in TRACKABLE_CLASSES and not ids:
            if not self.bash_monitor_all_enabled:
                return None
            resource_class = RESOURCE_LIGHT_CPU
        value_hint = self._infer_value_hint(
            command=command,
            resource_class=resource_class,
            source_hint=source_hint,
            timeout_sec=float(timeout_sec or 0.0),
        )
        if isinstance(value_hint_override, dict):
            clean_hint: dict[str, Any] = {}
            for key, raw in value_hint_override.items():
                if key in {
                    "expected_value_score",
                    "lineage_diversity_score",
                    "near_submission_score",
                    "worker_starvation_score",
                    "duplicate_penalty",
                    "timeout_history_penalty",
                    "long_runtime_penalty",
                }:
                    clean_hint[str(key)] = self._clip_score(raw)
                elif key in {"stage_type", "reason", "route_id", "execution_scale"}:
                    clean_hint[str(key)] = str(raw)[:160]
                elif key == "requested_gpu_count":
                    try:
                        clean_hint[str(key)] = max(0, int(raw))
                    except (TypeError, ValueError):
                        pass
            value_hint.update(clean_hint)
        quick_probe = classify_quick_probe_command(
            command,
            expected_runtime_sec=self.quick_probe_expected_runtime_sec,
            hard_review_sec=self.quick_probe_hard_review_sec,
            small_scope_threshold=self.quick_probe_small_scope_threshold,
            workspace_dir=workspace_dir,
        ).to_json()
        gpu_queue_relevant = self._gpu_queue_relevant(resource_class, source_hint, ids)
        request_count = source_hint.requested_gpu_count
        if self.gpu_source_hint_mode == "request" and source_hint.source_gpu_request:
            request_count = max(request_count or 0, source_hint.source_gpu_request)
        self._seq += 1
        job = ResourceJob(
            job_id=f"{self.worker_id}:bash:{self._seq:05d}",
            command=str(command or ""),
            command_digest=command_digest,
            resource_class=resource_class,
            gpu_ids=ids,
            cpu_set=str(cpu_set or ""),
            timeout_sec=float(timeout_sec or 0.0),
            created_at=time.time(),
            workspace_dir=Path(workspace_dir).resolve(strict=False) if workspace_dir is not None else None,
            candidate_artifact=self._safe_candidate_artifact(candidate_artifact),
            gpu_queue_relevant=gpu_queue_relevant,
            gpu_request_count=request_count,
            source_hint=source_hint,
            value_hint=value_hint,
            post_feedback_gate=(post_feedback_gate if post_feedback_gate.get("violates_feedback") else {}),
            quick_probe=quick_probe if quick_probe.get("quick_probe_candidate") else {},
        )
        self._jobs[job.job_id] = job
        initial_review_state = new_review_state(job.job_id)
        self._review_states[job.job_id] = initial_review_state
        self._review_machines[job.job_id] = ResourceReviewMachine(
            job_id=job.job_id,
            initial_state=initial_review_state,
        )
        if source_hint.hint_labels:
            self._emit(
                "resource_source_hint_detected",
                job,
                status="observed",
                payload={
                    **source_hint.to_event_payload(),
                    "classifier_reason": str(classifier_reason or ""),
                    "source_hint_mode": self.gpu_source_hint_mode,
                },
            )
        if quick_probe.get("quick_probe_candidate"):
            self._emit(
                "resource_quick_probe_detected",
                job,
                status="observed",
                payload={
                    **dict(quick_probe),
                    "classifier_reason": str(classifier_reason or ""),
                    "resource_class": resource_class,
                },
            )
        if gpu_queue_relevant:
            self._emit(
                "resource_request_created",
                job,
                status="created",
                payload={
                    "gpu_request": int(request_count or len(ids) or 1),
                    "queue_enabled": True,
                    "source_hint_mode": self.gpu_source_hint_mode,
                    "classifier_reason": str(classifier_reason or ""),
                    "value_hint": dict(value_hint),
                },
            )
        return job.job_id


    def _startup_trial_enabled(self) -> bool:
        return startup_policy_trial_first(self.resource_startup_policy)


    def _startup_trial_decision(
        self,
        job: ResourceJob,
        *,
        reason: str,
        status: str,
        scope: str,
        resource_mode: str,
        allowed_classes: list[str] | None = None,
        gpu_ids: list[str] | None = None,
        source_reason: str = "",
        extra_facts: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._startup_trial_enabled() or not job.gpu_queue_relevant or self.resource_runtime is None:
            return None
        cls = classify_preflight_block(
            reason=reason,
            source_reason=source_reason,
            resource_class=job.resource_class,
        )
        if cls.block_class != "soft_trial":
            return None
        now = time.time()
        ids = [str(x) for x in (gpu_ids or job.gpu_ids or []) if str(x).strip()]
        state = {
            "active": True,
            "source": "monitor_first_startup_trial",
            "reason": str(reason or "soft_resource_block"),
            "source_reason": str(source_reason or ""),
            "rule_status": str(status or "DENIED_REPLAN"),
            "startup_block_class": cls.block_class,
            "started_at": now,
            "trial": True,
            "trial_window_sec": self.resource_trial_window_sec,
            "trial_hard_review_sec": self.resource_trial_hard_review_sec,
            "observe_window_sec": self.resource_trial_window_sec,
            "observe_max_sec": self.resource_trial_hard_review_sec,
            "resource_mode": str(resource_mode or "YELLOW"),
            "scope": str(scope or "worker_plan"),
            "gpu_ids": ids,
            "allowed_classes": list(allowed_classes or []),
        }
        if extra_facts:
            state["extra_facts"] = dict(extra_facts)
        job.observe_first_state = state
        payload = {
            **state,
            "blocked_class": job.resource_class,
            "resource_startup_policy": self.resource_startup_policy,
        }
        self._emit("resource_startup_trial_candidate", job, status="trial", payload=payload)
        self._emit("resource_policy_gate", job, status="trial", payload=payload)
        self.resource_runtime.record_resource_event(
            "startup_trial_candidate",
            payload={"job_id": job.job_id, "resource_type": "gpu", "result": payload},
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        return {
            "allowed": True,
            "resource_class": job.resource_class,
            "status": "OBSERVE_THEN_RUN",
            "admission_action": "OBSERVE_THEN_RUN",
            "reason": str(reason or "soft_resource_block"),
            "startup_block_class": cls.block_class,
            "resource_startup_policy": self.resource_startup_policy,
            "trial": True,
            "trial_reason": str(reason or "soft_resource_block"),
            "trial_window_sec": self.resource_trial_window_sec,
            "trial_hard_review_sec": self.resource_trial_hard_review_sec,
            "feedback_suppressed": True,
        }


    def _pressure_observe_first_eligible(
        self,
        job: ResourceJob,
        *,
        gate: str,
        pressure: dict[str, Any],
        context_is_stale: bool,
    ) -> bool:
        if not (self.stale_pressure_observe_first_enabled and job.gpu_queue_relevant and self.resource_runtime is not None):
            return False
        if str(gate or "") == "duplicate_digest_cooldown":
            return False
        try:
            timeout_count = int(pressure.get("max_queue_timeout_count") or 0)
        except (TypeError, ValueError):
            timeout_count = 0
        if timeout_count > 0:
            return False
        soft_pressure_reasons = {
            "stale_resource_context",
            "block_train_after_gpu_pressure",
            "yellow_pressure_exit_guard",
            "gpu_pressure_yellow_or_red",
        }
        if not (context_is_stale or str(gate or "") in soft_pressure_reasons):
            return False
        if job.resource_class not in {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_UNKNOWN_GPU_EXEC,
            RESOURCE_GPU_FEATURE_EXTRACT,
        }:
            return False
        return True


    def _install_pressure_observe_first(
        self,
        job: ResourceJob,
        *,
        gate: str,
        status: str,
        pressure: dict[str, Any],
        red_gpu_ids: list[str],
        allowed_classes: list[str],
        resource_context_version: int | None,
        resource_pressure_generation: int | None,
        current_pressure_generation: int | None,
        pressure_snapshot: dict[str, Any],
        digest_pressure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            pressure_generation = int(pressure_snapshot.get("generation") or pressure.get("generation") or 0)
        except (TypeError, ValueError):
            pressure_generation = 0
        state = {
            "active": True,
            "source": "stale_pressure_preflight",
            "reason": str(gate or "stale_resource_context"),
            "rule_status": str(status or "DENIED_REPLAN"),
            "started_at": time.time(),
            "observe_window_sec": self.stale_pressure_observe_window_sec,
            "observe_max_sec": self.stale_pressure_observe_max_sec,
            "healthy_skip_llm": self.stale_pressure_healthy_skip_llm,
            "min_free_mem_gb": self.stale_pressure_observe_min_free_mem_gb,
            "resource_mode": str(pressure.get("resource_mode") or "RED"),
            "gpu_ids": list(red_gpu_ids or job.gpu_ids or []),
            "allowed_classes": list(allowed_classes or []),
            "cooldown_sec": float(pressure.get("cooldown_remaining_sec") or 0.0),
            "eta_next_train_sec": float(pressure.get("eta_next_train_sec") or 0.0),
            "eta_confidence": str(pressure.get("eta_confidence") or "low"),
            "pressure_generation": pressure_generation,
            "resource_context_version": resource_context_version,
            "resource_pressure_generation": resource_pressure_generation,
            "current_pressure_generation": current_pressure_generation,
        }
        job.observe_first_state = state
        payload = {
            **state,
            "blocked_class": job.resource_class,
            "scope": "per_gpu",
            "pressure": pressure_snapshot,
            "digest_pressure": dict(digest_pressure or {}),
        }
        self._emit("resource_observe_first_candidate", job, status="observing", payload=payload)
        self._emit("resource_policy_gate", job, status="observe_first", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "observe_first_candidate",
                payload={"job_id": job.job_id, "resource_type": "gpu", "result": payload},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {
            "allowed": True,
            "resource_class": job.resource_class,
            "status": "OBSERVE_FIRST",
            "reason": str(gate or "stale_resource_context"),
            "observe_first": True,
            "observe_window_sec": self.stale_pressure_observe_window_sec,
            "observe_max_sec": self.stale_pressure_observe_max_sec,
            "feedback_suppressed": True,
        }


    def observe_first_state(self, job_id: str | None, **_: Any) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"active": False}
        state = dict(self._jobs[job_id].observe_first_state or {})
        if not state.get("active"):
            return {"active": False}
        return state


    def observe_first_healthy_continue(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        stdout_age_sec: float = 0.0,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        saw_training_progress: bool = False,
        saw_final_score: bool = False,
        current_phase: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "reason": "job_missing"}
        job = self._jobs[job_id]
        state = dict(job.observe_first_state or {})
        if not state.get("active"):
            return {"enabled": False, "reason": "observe_first_inactive"}
        if state.get("healthy_emitted"):
            return {"enabled": False, "reason": "already_emitted"}
        evidence = {
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "stdout_age_sec": float(stdout_age_sec or 0.0),
            "saw_training_progress": bool(saw_training_progress),
            "saw_final_score": bool(saw_final_score),
            "current_phase": str(current_phase or ""),
            "gpu_mem_peak_gb": float(job.gpu_mem_peak_gb or 0.0),
            "idle_gpu_lease_samples": int(job.idle_gpu_lease_samples or 0),
            "dataloader_bottleneck_samples": int(job.dataloader_bottleneck_samples or 0),
            "last_gpu_util_sample": dict(job.last_gpu_util_sample or {}),
        }
        has_progress = bool(
            evidence["stdout_lines"]
            or evidence["stdout_bytes"]
            or evidence["saw_training_progress"]
            or evidence["saw_final_score"]
            or evidence["current_phase"]
            or evidence["gpu_mem_peak_gb"] > 0
        )
        health = "healthy_continue" if has_progress else "window_continue_low_confidence"
        state.update({
            "healthy_emitted": True,
            "health": health,
            "health_elapsed_sec": float(elapsed_sec or 0.0),
            "health_confidence": "medium" if has_progress else "low",
        })
        job.observe_first_state = state
        payload = {**state, "evidence": evidence, "llm_called": False}
        self._emit("resource_observe_first_healthy_continue", job, status="continued", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "observe_first_healthy_continue",
                payload={"job_id": job.job_id, "resource_type": "gpu", "result": payload},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"enabled": True, "health": health, "llm_called": False}
