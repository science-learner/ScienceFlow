# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Resource observer responsibility: deliverable, metric health, idle lease, and stalled-work safety gates.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    Path,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_LIGHT_CPU,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_READONLY_CPU,
    RESOURCE_UNKNOWN_EXEC,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ResourceEffectKind,
    ResourceJob,
    ResourceSourceHint,
    classify_deliverable_validity,
    deliverable_completion_state,
    re,
    time,
)


class SafetyCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def _job_is_expensive_resource_hold(self, job: ResourceJob) -> bool:
        cls = str(job.resource_class or "")
        safe_after_deliverable = {
            RESOURCE_PURE_TT_CPU,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_READONLY_CPU,
            RESOURCE_LIGHT_CPU,
        }
        if cls in safe_after_deliverable or bool(getattr(job.source_hint, "command_cpu_only", False)):
            return False
        if self._job_uses_expensive_gpu(job) or bool(job.gpu_ids):
            return True
        return cls in {RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_UNKNOWN_EXEC, RESOURCE_UNKNOWN_GPU_EXEC}


    def _completed_deliverable_safe_command(self, job: ResourceJob) -> bool:
        command = str(job.command or "").strip()
        if not command:
            return True
        lowered = command.lower()
        compact = re.sub(r"\s+", " ", lowered).strip()

        scratch_target = re.search(r"(?:cat\s*>|tee(?:\s+-a)?)(?:\s+)(['\"]?)([^'\"\s<>|]+)\1", compact)
        if scratch_target:
            target = scratch_target.group(2).lstrip("./")
            suffix = Path(target).suffix.lower()
            if target.startswith(("tmp/", ".memory/")) and suffix in {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".log"}:
                return True

        if re.match(r"^(pwd|ls|head|tail|wc|du|df|stat|find|rg|grep|sed)(\s|$)", compact):
            return ">" not in compact or re.search(r">\s*(tmp/|\.memory/)", compact) is not None
        if re.match(r"^(mkdir\s+-p\s+)?(tmp|\.memory)(/|\s|$)", compact):
            return True

        if re.match(r"^(python3?|uv run python)\s+(-c|<<)", compact):
            heavy_markers = (
                " train.py",
                " solution.py",
                " predict.py",
                "submission.csv",
                "to_csv",
                ".fit(",
                "fit(",
                "predict_proba",
                "lightgbm",
                "xgboost",
                "catboost",
                "tensorflow",
                "torch",
                "keras",
                "pil import image",
            )
            return not any(marker in compact for marker in heavy_markers)

        return False


    def _completed_deliverable_preflight_gate(self, job: ResourceJob) -> dict[str, Any]:
        if not (self.deliverable_completion_guard_enabled and job.workspace_dir):
            return {"blocked": False}
        if self._completed_deliverable_safe_command(job):
            return {"blocked": False, "reason": "completed_deliverable_safe_command"}
        if not self._job_is_expensive_resource_hold(job):
            return {"blocked": False}
        try:
            state = deliverable_completion_state(
                job.workspace_dir,
                terminal_signal_seen=False,
                terminal_signal_kind="",
                settle_sec=self.deliverable_completion_settle_sec,
                candidate_artifact=job.candidate_artifact,
            )
        except Exception as exc:
            return {"blocked": False, "error": str(exc)}
        validity = classify_deliverable_validity(state)
        job.last_deliverable_completion_check = {"checked_at": time.time(), "state": dict(state)}
        job.deliverable_validity = validity
        if validity == "produced_invalid":
            reason = "invalid_deliverable_schema_preflight"
        else:
            return {"blocked": False, "deliverable_completion_state": dict(state), "deliverable_validity": validity}
        allowed_classes = [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, RESOURCE_READONLY_CPU, RESOURCE_LIGHT_CPU]
        candidate_artifact = str(job.candidate_artifact or "").strip()
        unlock_condition = (
            "candidate_artifact_valid"
            if candidate_artifact and candidate_artifact != "submission.csv"
            else "submission_schema_valid"
            if validity == "produced_invalid"
            else "deliverable_state_changed"
        )
        schema_state = "invalid" if validity == "produced_invalid" else "valid"
        raw_feedback = self._append_resource_intervention_summary(
            self._resource_feedback_text(
                status="DENIED_REPLAN",
                reason=reason,
                scope="worker_deliverable",
                resource_mode="RED",
                blocked_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                allowed_classes=allowed_classes,
                unlock_condition=unlock_condition,
                blocked_until_unlock=True,
                schema_state=schema_state,
            ),
            action="DELIVERABLE_PREFLIGHT_BLOCK",
            reason=reason,
        )
        feedback_state = self._dedupe_resource_feedback_for_agent(
            job,
            feedback=raw_feedback,
            status="DENIED_REPLAN",
            reason=reason,
            scope="worker_deliverable",
            resource_mode="RED",
            blocked_class=job.resource_class,
            gpu_ids=job.gpu_ids,
            allowed_classes=allowed_classes,
            unlock_condition=unlock_condition,
            blocked_until_unlock=True,
            schema_state=schema_state,
            deliverable_validity=validity,
        )
        feedback = str(feedback_state.get("feedback") or "")
        self._remember_resource_feedback(
            job,
            status="DENIED_REPLAN",
            reason=reason,
            resource_mode="RED",
            blocked_class=job.resource_class,
            allowed_classes=allowed_classes,
        )
        self._emit(
            "resource_completed_deliverable_preflight_block",
            job,
            status="blocked",
            payload={
                "status": "DENIED_REPLAN",
                "reason": reason,
                "scope": "worker_deliverable",
                "resource_mode": "RED",
                "blocked_class": job.resource_class,
                "allowed_classes": allowed_classes,
                "deliverable_validity": validity,
                "deliverable_completion_state": dict(state),
                "candidate_artifact": candidate_artifact,
                "feedback_state_key": feedback_state.get("feedback_state_key"),
                "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
                "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
            },
        )
        return {
            "blocked": True,
            "status": "DENIED_REPLAN",
            "reason": reason,
            "feedback": feedback,
            "feedback_suppressed": bool(feedback_state.get("feedback_suppressed")),
            "feedback_state_key": feedback_state.get("feedback_state_key"),
            "resource_feedback_repeated_count": feedback_state.get("repeated_count"),
            "allowed_classes": allowed_classes,
            "deliverable_validity": validity,
            "deliverable_completion_state": dict(state),
            "candidate_artifact": candidate_artifact,
        }


    def _deliverable_completion_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.deliverable_completion_guard_enabled and job.visible):
            return {"terminate": False}
        if elapsed < self.deliverable_completion_warmup_sec:
            return {"terminate": False}
        if not job.workspace_dir:
            return {"terminate": False}
        if not self._job_is_expensive_resource_hold(job):
            return {"terminate": False}
        now = time.time()
        last = float(job.last_deliverable_completion_check.get("checked_at") or 0.0)
        if last and self.deliverable_completion_scan_interval_sec > 0 and now - last < self.deliverable_completion_scan_interval_sec:
            return {
                "terminate": False,
                "deliverable_complete_samples": job.deliverable_complete_samples,
                "deliverable_completion_state": dict(job.last_deliverable_completion_check.get("state") or {}),
            }
        terminal_events = int(signal.get("terminal_signal_events") or 0)
        terminal_seen = terminal_events > 0 or bool(signal.get("saw_final_score"))
        state = deliverable_completion_state(
            job.workspace_dir,
            terminal_signal_seen=terminal_seen,
            terminal_signal_kind=str(signal.get("terminal_signal_kind") or ("final_score" if signal.get("saw_final_score") else "")),
            settle_sec=self.deliverable_completion_settle_sec,
            candidate_artifact=job.candidate_artifact,
        )
        job.last_deliverable_completion_check = {"checked_at": now, "state": dict(state)}
        job.deliverable_validity = classify_deliverable_validity(state)
        self._update_v7_route_profile(job, signal, elapsed_sec=elapsed)
        job.deliverable_complete_samples = job.deliverable_complete_samples + 1 if state.get("complete") else 0
        quiet_enough = float(signal.get("stdout_age_sec") or 0.0) >= self.deliverable_completion_quiet_sec
        is_submission = str(state.get("mode") or "") == "submission"
        if job.deliverable_validity == "produced_invalid":
            self._record_v7_resource_event(
                "invalid_deliverable_detected",
                job,
                {
                    "deliverable_completion_state": dict(state),
                    "route_viability": dict(job.route_viability_state or {}),
                    "feedback": "submission_schema_state=invalid",
                },
            )
            return {
                "terminate": False,
                "deliverable_complete_samples": job.deliverable_complete_samples,
                "deliverable_completion_state": dict(state),
                "deliverable_quiet_enough": quiet_enough,
                "feedback": self._append_resource_intervention_summary(
                    self._resource_feedback_text(
                        status="DENIED_REPLAN",
                        reason="invalid_deliverable_schema",
                        scope="worker_deliverable",
                        resource_mode="RED",
                        blocked_class=job.resource_class,
                        gpu_ids=job.gpu_ids,
                        unlock_condition="submission_schema_valid",
                        blocked_until_unlock=True,
                        schema_state="invalid",
                    ),
                    action="INVALID_DELIVERABLE_SCHEMA",
                    reason="produced_invalid",
                ),
            }
        if not state.get("complete") or (not is_submission and not quiet_enough):
            return {
                "terminate": False,
                "deliverable_complete_samples": job.deliverable_complete_samples,
                "deliverable_completion_state": dict(state),
                "deliverable_quiet_enough": quiet_enough,
            }
        return {
            "terminate": False,
            "deliverable_complete_samples": job.deliverable_complete_samples,
            "deliverable_completion_state": dict(state),
            "deliverable_quiet_enough": quiet_enough,
        }


    def _metric_health_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.metric_health_guard_enabled and job.visible):
            return {"terminate": False}
        if elapsed < self.metric_health_warmup_sec or bool(signal.get("saw_final_score")):
            return {"terminate": False}
        if not (self._job_uses_expensive_gpu(job) or job.gpu_ids):
            return {"terminate": False}
        invalid_events = int(signal.get("invalid_metric_events") or 0)
        zero_score_events = int(signal.get("zero_score_events") or 0)
        if invalid_events >= self.metric_health_invalid_min_events:
            return {
                "terminate": True,
                "reason": "active_intervention:invalid_training_metrics_nan_inf",
                "invalid_metric_events": invalid_events,
                "zero_score_events": zero_score_events,
                "last_invalid_metric_text": str(signal.get("last_invalid_metric_text") or ""),
                "feedback": (
                    "RESOURCE_FEEDBACK: terminated_invalid_training_metrics because validation or training outputs contained NaN/Inf, so the run is not producing a usable model.\n"
                ),
            }
        if zero_score_events >= self.metric_health_zero_score_min_events and bool(signal.get("saw_training_progress")):
            return {
                "terminate": True,
                "reason": "active_intervention:invalid_training_metrics_zero_score",
                "invalid_metric_events": invalid_events,
                "zero_score_events": zero_score_events,
                "last_zero_score_text": str(signal.get("last_zero_score_text") or ""),
                "feedback": (
                    "RESOURCE_FEEDBACK: terminated_invalid_training_metrics because the validation score stayed at zero, so the run is not producing useful model signal.\n"
                ),
            }
        return {
            "terminate": False,
            "invalid_metric_events": invalid_events,
            "zero_score_events": zero_score_events,
        }


    def _record_idle_lease_release_candidate(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        pressure_reason: str,
        pressure: dict[str, Any],
        release_safety: dict[str, Any] | None = None,
        candidate_mode: str = "observe_only",
    ) -> None:
        now = time.time()
        cooldown = max(60.0, float(self.gpu_util_sample_interval_sec or 30.0))
        last = float(self._last_idle_lease_candidate_emit.get(job.job_id) or 0.0)
        if last and now - last < cooldown:
            return
        self._last_idle_lease_candidate_emit[job.job_id] = now
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        payload = {
            "action": "RELEASE_IDLE_LEASE",
            "mode": str(candidate_mode or "observe_only"),
            "would_release": True,
            "executed": False,
            "reason": "idle_gpu_lease_release_candidate",
            "elapsed_sec": elapsed,
            "idle_samples": job.idle_gpu_lease_samples,
            "min_samples": self.gpu_idle_lease_min_samples,
            "util_threshold_pct": self.gpu_idle_lease_util_pct,
            "mem_threshold_gb": self.gpu_idle_lease_mem_gb,
            "pressure_reason": pressure_reason,
            "pressure": dict(pressure or {}),
            "release_safety": dict(release_safety or {}),
            "last_gpu_util_sample": dict(job.last_gpu_util_sample or {}),
        }
        self._emit(
            "resource_idle_lease_release_candidate",
            job,
            status="observe_only",
            payload=payload,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_idle_lease_release_candidate",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )


    def _idle_lease_release_safety(self, job: ResourceJob, signal: dict[str, Any] | None = None) -> dict[str, Any]:
        hint = job.source_hint or ResourceSourceHint()
        sig = signal if isinstance(signal, dict) else {}
        phase = str(sig.get("current_phase") or "").strip().lower()
        current_mem_gb = self._current_gpu_mem_gb_for_job(job)
        idle_enough = job.idle_gpu_lease_samples >= max(1, int(self.gpu_idle_lease_min_samples or 1))
        mem_small = current_mem_gb <= max(0.0, float(self.gpu_idle_lease_mem_gb or 0.0))
        has_gpu_evidence = bool(getattr(hint, "command_gpu_compute_evidence", False)) or bool(getattr(hint, "has_gpu_evidence", False))
        known_training_phase = bool(
            any(token in phase for token in ("train", "epoch", "optimizer", "backward"))
            or (
                str(job.resource_class or "") in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN}
                and bool(getattr(hint, "has_train_evidence", False))
            )
        )
        possible_late_gpu_use = bool(known_training_phase or has_gpu_evidence)
        base = {
            "idle_samples": int(job.idle_gpu_lease_samples or 0),
            "min_samples": int(self.gpu_idle_lease_min_samples or 0),
            "current_mem_gb": current_mem_gb,
            "peak_mem_gb": float(job.gpu_mem_peak_gb or 0.0),
            "mem_threshold_gb": float(self.gpu_idle_lease_mem_gb or 0.0),
            "known_training_phase": known_training_phase,
            "has_gpu_evidence": has_gpu_evidence,
            "possible_late_gpu_use": possible_late_gpu_use,
            "phase": phase,
        }
        if not idle_enough:
            return {"safe": False, "reason": "release_window_not_satisfied", **base}
        if not mem_small:
            return {"safe": False, "reason": "gpu_mem_above_release_threshold", **base}
        if bool(getattr(hint, "command_cpu_only", False)):
            return {"safe": True, "reason": "command_cpu_only_hint", **base}
        inspected = int(getattr(hint, "source_files_inspected", 0) or 0)
        entrypoints = [str(x) for x in (getattr(hint, "entrypoints", []) or []) if str(x).strip()]
        if possible_late_gpu_use:
            return {
                "safe": True,
                "reason": "idle_small_mem_release_with_late_gpu_risk",
                "risk": "possible_late_gpu_use",
                "old_job_effect": "continues_with_existing_cuda_visible_devices",
                "new_job_policy": "admit_or_share_then_monitor_actual_contention",
                **base,
            }
        if inspected > 0:
            return {"safe": True, "reason": "source_inspected_without_gpu_evidence", "source_files_inspected": inspected, **base}
        if entrypoints:
            return {"safe": True, "reason": "entrypoint_not_inspected_but_idle_small_mem", "entrypoints": entrypoints[:5], **base}
        return {"safe": True, "reason": "command_without_gpu_evidence", **base}


    def release_idle_gpu_lease(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        reason: str = "idle_gpu_lease_release",
        feedback: str = "",
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return {"released": False, "reason": "missing_job_or_runtime"}
        job = self._jobs[job_id]
        safety = self._idle_lease_release_safety(job, job.last_signal)
        if not bool(safety.get("safe")):
            return {"released": False, "reason": "release_safety_failed", "release_safety": safety}
        result = self._execute_resource_effect(
            ResourceEffectKind.RELEASE_IDLE,
            command_id=job.job_id,
            idempotency_key=f"release-idle:{job.job_id}",
            parameters={
                "job_id": job.job_id,
                "elapsed_sec": float(elapsed_sec or 0.0),
                "reason": str(reason or "idle_gpu_lease_release"),
                "resident_mem_gb": self._current_gpu_mem_gb_for_job(job),
            },
        )
        if result.get("released"):
            job.idle_gpu_lease_samples = 0
            payload = {
                "job_id": job.job_id,
                "action": "RELEASE_IDLE_LEASE",
                "executed": True,
                "elapsed_sec": float(elapsed_sec or 0.0),
                "reason": str(reason or "idle_gpu_lease_release"),
                "feedback": str(feedback or ""),
                "release_safety": safety,
                "result": result,
            }
            self._emit("resource_idle_lease_released", job, status="released", payload=payload)
            self.resource_runtime.record_resource_event(
                "resource_guard_action",
                payload={"action": "release_idle_lease", **payload},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return result


    def _idle_gpu_lease_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.gpu_idle_lease_guard_enabled):
            return {"terminate": False}
        if not self.gpu_util_observer_enabled or self.resource_runtime is None:
            return {"terminate": False}
        if bool(signal.get("saw_final_score")):
            return {"terminate": False}
        idle_guard_classes = {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_UNKNOWN_GPU_EXEC,
        }
        if not job.gpu_ids:
            return {"terminate": False}
        if str(job.resource_class or "") not in idle_guard_classes:
            return {"terminate": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"terminate": False}
        if elapsed < self.gpu_idle_lease_warmup_sec:
            return {"terminate": False, "idle_samples": job.idle_gpu_lease_samples}
        if job.idle_gpu_lease_samples < self.gpu_idle_lease_min_samples:
            return {"terminate": False, "idle_samples": job.idle_gpu_lease_samples}
        pressure = self._gpu_pressure_active_for_job(job)
        if not pressure.get("active"):
            return {
                "terminate": False,
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure.get("reason"),
            }
        pressure_reason = str(pressure.get("reason") or "gpu_pressure")
        release_safety = self._idle_lease_release_safety(job, signal)
        if self.gpu_idle_lease_action_mode == "observe":
            self._record_idle_lease_release_candidate(
                job,
                signal,
                pressure_reason=pressure_reason,
                pressure=pressure,
                release_safety=release_safety,
                candidate_mode="observe_only",
            )
            return {
                "terminate": False,
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure_reason,
                "release_candidate": True,
                "release_action": "RELEASE_IDLE_LEASE",
                "release_mode": "observe_only",
                "release_safety": release_safety,
            }
        feedback = (
            f"RESOURCE_FEEDBACK: release_idle_gpu_lease because the command held a GPU lease while sampled GPU util <= {self.gpu_idle_lease_util_pct:.1f}% and memory <= {self.gpu_idle_lease_mem_gb:.2f}GB for {job.idle_gpu_lease_samples} samples; pressure={pressure_reason}; elapsed_sec={elapsed:.1f}.\n"
        )
        if self.gpu_idle_lease_action_mode == "release" and bool(release_safety.get("safe")):
            return {
                "terminate": False,
                "release_idle_lease": True,
                "reason": f"active_intervention:idle_gpu_lease_under_pressure:{pressure_reason}",
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure_reason,
                "release_candidate": True,
                "release_action": "RELEASE_IDLE_LEASE",
                "release_mode": "release",
                "release_safety": release_safety,
                "feedback": feedback,
            }
        if self.gpu_idle_lease_action_mode == "release":
            self._record_idle_lease_release_candidate(
                job,
                signal,
                pressure_reason=pressure_reason,
                pressure=pressure,
                release_safety=release_safety,
                candidate_mode="blocked_by_safety",
            )
            return {
                "terminate": False,
                "idle_samples": job.idle_gpu_lease_samples,
                "pressure_reason": pressure_reason,
                "release_candidate": True,
                "release_action": "RELEASE_IDLE_LEASE",
                "release_mode": "blocked_by_safety",
                "release_safety": release_safety,
            }
        return {
            "terminate": True,
            "reason": f"active_intervention:idle_gpu_lease_under_pressure:{pressure_reason}",
            "idle_samples": job.idle_gpu_lease_samples,
            "pressure_reason": pressure_reason,
            "release_candidate": True,
            "release_action": "RELEASE_IDLE_LEASE" if self.gpu_idle_lease_action_mode == "recommend" else "KILL_AND_REPLAN",
            "release_mode": self.gpu_idle_lease_action_mode,
            "release_safety": release_safety,
            "feedback": (
                f"RESOURCE_FEEDBACK: terminated_idle_gpu_lease because the command held a GPU lease while sampled GPU util <= {self.gpu_idle_lease_util_pct:.1f}% and memory <= {self.gpu_idle_lease_mem_gb:.2f}GB for {job.idle_gpu_lease_samples} samples; pressure={pressure_reason}; elapsed_sec={elapsed:.1f}.\n"
            ),
        }


    def _low_progress_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.low_progress_enabled and job.visible):
            return {"terminate": False}
        if not self._job_uses_expensive_gpu(job):
            return {"terminate": False}
        if elapsed < self.low_progress_warmup_sec or bool(signal.get("saw_final_score")):
            return {"terminate": False}
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed)
        artifact_age = self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed)
        no_progress_limit = max(0.0, float(self.low_progress_no_heartbeat_sec or 0.0))
        no_artifact_limit = max(0.0, float(self.low_progress_no_artifact_sec or 0.0))
        progress_stale = no_progress_limit > 0 and progress_age >= no_progress_limit
        artifact_stale = no_artifact_limit <= 0 or artifact_age >= no_artifact_limit
        if not (progress_stale and artifact_stale):
            return {
                "terminate": False,
                "progress_age_sec": progress_age,
                "artifact_age_sec": artifact_age,
            }
        return {
            "terminate": True,
            "reason": "active_intervention:low_progress_heartbeat_stalled",
            "progress_age_sec": progress_age,
            "artifact_age_sec": artifact_age,
            "feedback": (
                "RESOURCE_FEEDBACK: terminated_low_progress_command because no epoch/iteration/metric/artifact heartbeat arrived within the configured progress threshold.\n"
            ),
        }


    def _dataloader_bottleneck_guard_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not (self.kill_enabled and self.gpu_dataloader_bottleneck_guard_enabled and job.visible):
            return {"terminate": False}
        if not self._job_uses_expensive_gpu(job):
            return {"terminate": False}
        if self.resource_runtime is None or not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"terminate": False}
        if elapsed < self.gpu_dataloader_bottleneck_warmup_sec or bool(signal.get("saw_final_score")):
            return {"terminate": False, "dataloader_low_compute_samples": job.dataloader_bottleneck_samples}
        if job.dataloader_bottleneck_samples < self.gpu_dataloader_bottleneck_min_samples:
            return {"terminate": False, "dataloader_low_compute_samples": job.dataloader_bottleneck_samples}
        cpu = signal.get("process_tree_cpu") if isinstance(signal.get("process_tree_cpu"), dict) else {}
        child_cpu = self._float_or_none(cpu.get("child_cpu_pct")) or 0.0
        busy_children = int(cpu.get("busy_child_count") or 0)
        if child_cpu < self.gpu_dataloader_bottleneck_child_cpu_pct:
            return {
                "terminate": False,
                "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
                "child_cpu_pct": child_cpu,
                "busy_child_count": busy_children,
            }
        if busy_children < self.gpu_dataloader_bottleneck_busy_children:
            return {
                "terminate": False,
                "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
                "child_cpu_pct": child_cpu,
                "busy_child_count": busy_children,
            }
        return {
            "terminate": True,
            "reason": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
            "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
            "child_cpu_pct": child_cpu,
            "busy_child_count": busy_children,
            "feedback": (
                "RESOURCE_FEEDBACK: terminated_dataloader_bottleneck because the command held a GPU lease and loaded model memory, but GPU utilization stayed low while DataLoader/child CPU stayed high, so the GPU was waiting on input preprocessing.\n"
            ),
        }


    def stalled_guard_decision(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float,
        stdout_age_sec: float,
        process_tree_cpu: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "terminate": False, "would_terminate": False}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        if elapsed >= self.min_register_sec:
            self._promote(job, reason="elapsed_threshold", elapsed_sec=elapsed)
        if self.review_state_enabled:
            return {
                "enabled": False,
                "terminate": False,
                "would_terminate": False,
                "reason": "state_machine_handles_stalled_stdout",
                "check_interval_sec": self.check_interval_sec,
            }
        terminate = bool(
            self.kill_enabled
            and job.visible
            and self.stalled_stdout_sec > 0
            and float(stdout_age_sec or 0.0) >= self.stalled_stdout_sec
        )
        decision = self._apply_kill_approval_gate({
            "enabled": job.visible,
            "terminate": terminate,
            "reason": "stalled_stdout" if terminate else "",
            "check_interval_sec": self.check_interval_sec,
            "feedback": "RESOURCE_FEEDBACK: terminated_stalled_command because no useful output heartbeat arrived within the configured stalled-output threshold.\n" if terminate else "",
        })
        stalled_signal = {
            "elapsed_sec": elapsed,
            "stdout_age_sec": float(stdout_age_sec or 0.0),
            "stdout_lines": 0,
            "stdout_bytes": 0,
            "current_phase": "",
            "process_tree_cpu": dict(process_tree_cpu or {}),
        }
        self._update_job_progress_signal(job, stalled_signal, elapsed_sec=elapsed)
        self._record_kill_proposal_event(
            job,
            decision,
            stalled_signal,
            source="stalled_guard",
        )
        return decision
