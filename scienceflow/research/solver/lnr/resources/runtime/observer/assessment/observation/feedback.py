# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Resource observer responsibility: feedback deduplication, pressure guards, and post-feedback actions.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ResourceJob,
    ResourceSourceHint,
    decorate_post_feedback_action_after_backoff,
    feedback_guard_is_hard_constraint,
    gpu_store_snapshot_has_lease,
    gpu_store_snapshot_has_pressure,
    normalize_gpu_ids,
    re,
    reconcile_free_gpu_pressure_blocked,
    resource_class_can_use_soft_gpu_gate,
    time,
)


class FeedbackCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    @staticmethod
    def _resource_feedback_state_key(
        *,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        blocked_class: str,
        gpu_ids: list[str] | None = None,
        allowed_classes: list[str] | None = None,
        unlock_condition: str = "",
        blocked_until_unlock: bool | None = None,
        schema_state: str = "",
        artifact_state: str = "",
        progress_state: str = "",
        deliverable_validity: str = "",
    ) -> str:
        scope_key = str(scope or "task").strip().lower()
        status_key = str(status or "DENIED_REPLAN").strip().upper()
        reason_key = str(reason or "resource_policy_gate").strip().lower()
        resource_wait_status = status_key in {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN", "DENIED_DUPLICATE"}
        if resource_wait_status and reason_key not in {
            "invalid_deliverable_schema_preflight",
            "task_gpu_boundary_preflight_violation",
            "workspace_gpu_boundary_violation",
            "boundary_violation",
            "duplicate_digest_cooldown",
        }:
            if "duplicate" in reason_key:
                reason_key = "duplicate_resource_wait"
            elif "schema" in reason_key:
                reason_key = "schema_resource_wait"
            elif "gpu" in reason_key or "slot" in reason_key or "lease" in reason_key or "yellow" in reason_key or "capacity" in reason_key or "class" in reason_key:
                reason_key = "gpu_resource_wait"
            else:
                reason_key = "resource_wait"
        collapse_blocked_class = scope_key in {"worker_deliverable", "worker_plan"} or reason_key in {
            "completed_deliverable_heavy_command_block",
            "invalid_deliverable_schema_preflight",
            "post_feedback_blocked_class",
            "active_resource_plan_guard",
        }
        blocked_key = "resource_work" if collapse_blocked_class else str(blocked_class or "unknown").strip().lower()
        gpu_key = "" if scope_key == "worker_deliverable" else ",".join(sorted({str(x).strip() for x in (gpu_ids or []) if str(x).strip()}))
        allowed_key = ",".join(sorted({str(x).strip().lower() for x in (allowed_classes or []) if str(x).strip()}))
        blocked_until = "" if blocked_until_unlock is None else str(bool(blocked_until_unlock)).lower()
        parts = [
            status_key,
            reason_key,
            scope_key,
            str(resource_mode or "UNKNOWN").strip().upper(),
            blocked_key,
            gpu_key,
            allowed_key,
            str(unlock_condition or "").strip().lower(),
            blocked_until,
            str(schema_state or "").strip().lower(),
            str(artifact_state or "").strip().lower(),
            str(progress_state or "").strip().lower(),
            str(deliverable_validity or "").strip().lower(),
        ]
        return "|".join(parts)


    def _dedupe_resource_feedback_for_agent(
        self,
        job: ResourceJob,
        *,
        feedback: str,
        status: str,
        reason: str,
        scope: str,
        resource_mode: str,
        blocked_class: str,
        gpu_ids: list[str] | None = None,
        allowed_classes: list[str] | None = None,
        unlock_condition: str = "",
        blocked_until_unlock: bool | None = None,
        schema_state: str = "",
        artifact_state: str = "",
        progress_state: str = "",
        deliverable_validity: str = "",
    ) -> dict[str, Any]:
        key = self._resource_feedback_state_key(
            status=status,
            reason=reason,
            scope=scope,
            resource_mode=resource_mode,
            blocked_class=blocked_class,
            gpu_ids=gpu_ids,
            allowed_classes=allowed_classes,
            unlock_condition=unlock_condition,
            blocked_until_unlock=blocked_until_unlock,
            schema_state=schema_state,
            artifact_state=artifact_state,
            progress_state=progress_state,
            deliverable_validity=deliverable_validity,
        )
        now = time.time()
        existing = self._reported_resource_feedback.get(key)
        if isinstance(existing, dict):
            repeated_count = int(existing.get("repeated_count") or 1) + 1
            existing.update(
                {
                    "repeated_count": repeated_count,
                    "last_seen_at": now,
                    "last_job_id": job.job_id,
                    "last_command_digest": job.command_digest,
                    "last_blocked_class": blocked_class,
                },
            )
            self._emit(
                "resource_feedback_suppressed",
                job,
                status="suppressed",
                payload={
                    "feedback_state_key": key,
                    "reason": reason,
                    "scope": scope,
                    "resource_mode": resource_mode,
                    "blocked_class": blocked_class,
                    "first_job_id": existing.get("first_job_id"),
                    "repeated_count": repeated_count,
                    "suppression_reason": "same resource feedback state already reported to main agent",
                },
            )
            return {
                "feedback": "",
                "feedback_suppressed": True,
                "feedback_state_key": key,
                "repeated_count": repeated_count,
            }
        self._reported_resource_feedback[key] = {
            "first_seen_at": now,
            "last_seen_at": now,
            "first_job_id": job.job_id,
            "last_job_id": job.job_id,
            "first_command_digest": job.command_digest,
            "last_command_digest": job.command_digest,
            "first_blocked_class": blocked_class,
            "last_blocked_class": blocked_class,
            "repeated_count": 1,
        }
        return {
            "feedback": feedback,
            "feedback_suppressed": False,
            "feedback_state_key": key,
            "repeated_count": 1,
        }


    def _current_pressure_generation(self) -> int:
        if self.resource_runtime is None:
            return 0
        try:
            return int(self.resource_runtime.pressure_generation() or 0)
        except (TypeError, ValueError):
            return 0


    @staticmethod
    def _resource_class_matches_guard(resource_class: str, blocked_class: str) -> bool:
        cls = str(resource_class or "")
        blocked = str(blocked_class or "")
        heavy = {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}
        if cls in heavy and blocked in heavy:
            return True
        return bool(cls and blocked and cls == blocked)


    def _light_train_share_deferral_enabled(self, resource_class: str, gpu_ids: list[str] | None) -> bool:
        return bool(
            str(resource_class or "") == RESOURCE_GPU_LIGHT_TRAIN
            and [str(x) for x in (gpu_ids or []) if str(x).strip()]
            and self.gpu_share_config.grant_active
            and self.gpu_share_config.phase in {"feature_share", "light_train"}
            and self.arbiter_enabled
            and self.arbiter_mode == "llm"
        )


    @staticmethod
    def _feedback_guard_is_hard_constraint(guard: dict[str, Any] | None) -> bool:
        return feedback_guard_is_hard_constraint(guard)


    def _can_defer_light_train_guard_to_arbiter(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str] | None,
        guard: dict[str, Any] | None,
    ) -> bool:
        if not self._light_train_share_deferral_enabled(resource_class, gpu_ids):
            return False
        return not self._feedback_guard_is_hard_constraint(guard)


    def _gpu_ids_have_active_store_pressure(self, gpu_ids: list[str] | None) -> bool:
        if self.resource_runtime is None:
            return True
        ids = normalize_gpu_ids(gpu_ids)
        if not ids:
            return True
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return True
        return gpu_store_snapshot_has_pressure(snapshot, ids, include_waiters=True)


    def _gpu_ids_have_active_store_lease(self, gpu_ids: list[str] | None) -> bool:
        if self.resource_runtime is None:
            return True
        ids = normalize_gpu_ids(gpu_ids)
        if not ids:
            return True
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return True
        return gpu_store_snapshot_has_lease(snapshot, ids)


    def _soft_gpu_gate_can_clear_for_available_slot(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str] | None,
        guard: dict[str, Any] | None,
    ) -> bool:
        if not self.control_profile.clear_stale_soft_gpu_gate_on_free_gpu:
            return False
        if self._feedback_guard_is_hard_constraint(guard):
            return False
        if not resource_class_can_use_soft_gpu_gate(resource_class):
            return False
        ids = normalize_gpu_ids(gpu_ids)
        if not ids or self.resource_runtime is None:
            return False
        reconcile = self.resource_runtime.reconcile_free_gpu_pressure(
            gpu_ids=ids,
            reason="observed_free_before_soft_gpu_gate",
        )
        if reconcile_free_gpu_pressure_blocked(reconcile):
            return False
        return not self._gpu_ids_have_active_store_lease(ids)


    def _can_ignore_soft_gpu_guard_for_available_slot(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str] | None,
        guard: dict[str, Any] | None,
    ) -> bool:
        if self._feedback_guard_is_hard_constraint(guard):
            return False
        if not resource_class_can_use_soft_gpu_gate(resource_class):
            return False
        if self.control_profile.ignore_waiters_for_soft_gpu_gate:
            return self._soft_gpu_gate_can_clear_for_available_slot(
                resource_class=resource_class,
                gpu_ids=gpu_ids,
                guard=guard,
            )
        return not self._gpu_ids_have_active_store_pressure(gpu_ids)


    def _install_plan_guard(
        self,
        feedback: dict[str, Any],
        *,
        eta_next_train_sec: float | None = None,
        cooldown_sec: float | None = None,
    ) -> None:
        blocked = str(feedback.get("blocked_class") or "").strip()
        status = str(feedback.get("status") or "").upper()
        if not blocked or status not in {"PENDING", "REPLAN", "DEFERRED", "DENIED_REPLAN", "DENIED_DUPLICATE"}:
            return
        try:
            eta = float(eta_next_train_sec or 0.0)
        except (TypeError, ValueError):
            eta = 0.0
        try:
            cooldown = float(cooldown_sec or 0.0)
        except (TypeError, ValueError):
            cooldown = 0.0
        duration = max(eta, cooldown)
        if duration <= 0.0:
            duration = 300.0 if status in {"PENDING", "DEFERRED"} else 120.0
        duration = max(30.0, min(duration, 900.0))
        guard_id = str(feedback.get("feedback_id") or f"guard-{self.worker_id}-{self._feedback_seq:05d}")
        self._active_plan_guards[guard_id] = {
            **feedback,
            "guard_id": guard_id,
            "expires_at": time.time() + duration,
            "guard_duration_sec": duration,
            "pressure_generation": self._current_pressure_generation(),
        }


    def _active_plan_guard_decision(self, job: ResourceJob) -> dict[str, Any]:
        if not self._active_plan_guards:
            return {"blocked": False}
        if not (job.gpu_queue_relevant or self._has_gpu_intent(job.source_hint)):
            return {"blocked": False}
        now = time.time()
        current_generation = self._current_pressure_generation()
        job_gpu_ids = {str(x) for x in (job.gpu_ids or []) if str(x).strip()}
        for guard_id, raw in list(self._active_plan_guards.items()):
            if not isinstance(raw, dict):
                self._active_plan_guards.pop(guard_id, None)
                continue
            expires_at = float(raw.get("expires_at") or 0.0)
            guard_generation = int(raw.get("pressure_generation") or 0)
            if expires_at <= now or (current_generation > guard_generation and guard_generation >= 0):
                self._active_plan_guards.pop(guard_id, None)
                continue
            if not self._resource_class_matches_guard(job.resource_class, str(raw.get("blocked_class") or "")):
                continue
            if self._can_defer_light_train_guard_to_arbiter(
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                guard=raw,
            ):
                continue
            if self._can_ignore_soft_gpu_guard_for_available_slot(
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                guard=raw,
            ):
                if self.control_profile.clear_stale_soft_gpu_gate_on_free_gpu:
                    self._active_plan_guards.pop(guard_id, None)
                continue
            guard_gpu_ids = {str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()}
            if guard_gpu_ids and job_gpu_ids and not (guard_gpu_ids & job_gpu_ids):
                continue
            return {
                "blocked": True,
                "guard": raw,
                "guard_id": guard_id,
                "current_pressure_generation": current_generation,
                "remaining_sec": max(0.0, expires_at - now),
            }
        return {"blocked": False}


    def _remember_resource_feedback(
        self,
        job: ResourceJob,
        *,
        status: str,
        reason: str,
        resource_mode: str,
        blocked_class: str,
        allowed_classes: list[str] | None = None,
        eta_next_train_sec: float | None = None,
        cooldown_sec: float | None = None,
        retry_after_sec: float | None = None,
        post_feedback_action: str = "",
    ) -> None:
        self._feedback_seq += 1
        feedback = {
            "feedback_id": f"rf-{self.worker_id}-{self._feedback_seq:05d}",
            "status": str(status or ""),
            "reason": str(reason or ""),
            "resource_mode": str(resource_mode or ""),
            "blocked_class": str(blocked_class or ""),
            "allowed_classes": [str(x) for x in (allowed_classes or []) if str(x).strip()],
            "job_id": job.job_id,
            "command_digest": job.command_digest,
            "resource_class": job.resource_class,
            "gpu_ids": list(job.gpu_ids or []),
            "created_at": time.time(),
            "pressure_generation": self._current_pressure_generation(),
        }
        if retry_after_sec is not None:
            feedback["retry_after_sec"] = max(0.0, float(retry_after_sec or 0.0))
        if post_feedback_action:
            feedback["post_feedback_action"] = str(post_feedback_action or "")
        self._pending_resource_feedback = feedback
        self._install_plan_guard(
            feedback,
            eta_next_train_sec=eta_next_train_sec,
            cooldown_sec=cooldown_sec,
        )


    @staticmethod
    def _is_simple_sleep_command(command: str) -> bool:
        return bool(re.match(r"^\s*sleep\s+\d+(?:\.\d+)?[smhd]?\s*$", str(command or ""), re.IGNORECASE))


    @staticmethod
    def _is_inline_python_command(command: str) -> bool:
        return bool(re.search(r"\b(?:python|python3(?:\.\d+)?)\s+(?:-u\s+)?-c\b", str(command or "")))


    def _post_feedback_action_kind(
        self,
        *,
        command: str,
        resource_class: str,
        source_hint: ResourceSourceHint | None = None,
    ) -> str:
        cls = str(resource_class or "")
        cmd = str(command or "").strip().lower()
        if self._is_simple_sleep_command(command):
            return "sleep_backoff"
        if bool(getattr(source_hint, "command_cpu_only", False)) and cls in {
            RESOURCE_HEAVY_GPU_CANDIDATE,
            RESOURCE_HEAVY_GPU_TRAIN,
            RESOURCE_GPU_FEATURE_EXTRACT,
            RESOURCE_GPU_LIGHT_TRAIN,
            RESOURCE_GPU_TT_LIGHT,
            RESOURCE_UNKNOWN_GPU_EXEC,
            RESOURCE_HEAVY_CPU_CANDIDATE,
        }:
            return "cpu_support"
        if (
            cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}
            and self._is_inline_python_command(command)
            and not self._has_gpu_intent(source_hint)
            and not bool((source_hint or ResourceSourceHint()).entrypoints)
        ):
            return "cpu_support"
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
            return "gpu_train"
        if cls == RESOURCE_UNKNOWN_GPU_EXEC:
            return "unknown_gpu"
        if cls == RESOURCE_GPU_FEATURE_EXTRACT:
            return "gpu_feature_extract"
        if cls in {RESOURCE_GPU_TT_LIGHT, RESOURCE_PURE_TT_CPU}:
            return "tt_or_submission"
        if re.search(r"\b(submission|submit|predict|inference|infer|tta|ensemble|blend)\b", cmd):
            return "tt_or_submission"
        if re.match(r"^(ls|find|cat|head|tail|wc|du|df|pwd|python\s+-c\s+['\"]?print\()\b", cmd):
            return "readonly_cpu"
        return "cpu_support"


    def _emit_post_feedback_action(
        self,
        *,
        command: str,
        resource_class: str,
        command_digest: str,
        gpu_ids: list[str] | None = None,
        source_hint: ResourceSourceHint | None = None,
    ) -> dict[str, Any]:
        pending = self._pending_resource_feedback
        if not isinstance(pending, dict):
            return {}
        try:
            pending_generation = int(pending.get("pressure_generation") or 0)
        except (TypeError, ValueError):
            pending_generation = 0
        if self._current_pressure_generation() > pending_generation:
            self._pending_resource_feedback = None
            return {}
        self._pending_resource_feedback = None
        action = self._post_feedback_action_kind(command=command, resource_class=resource_class, source_hint=source_hint)
        cls = str(resource_class or "")
        action_decoration = decorate_post_feedback_action_after_backoff(
            action=action,
            pending_feedback=pending,
            last_backoff=self._last_agent_backoff_wait,
        )
        action = str(action_decoration.get("action") or action)
        blocked_class = str(pending.get("blocked_class") or "")
        mode = str(pending.get("resource_mode") or "").upper()
        allowed = [str(x) for x in (pending.get("allowed_classes") or []) if str(x).strip()]
        allowed_current_class = self._resource_class_in_allowed(cls, allowed)
        same_digest = bool(command_digest and command_digest == str(pending.get("command_digest") or ""))
        violates = False
        if same_digest and str(pending.get("status") or "").upper() == "DENIED_DUPLICATE":
            violates = True
        if blocked_class and cls == blocked_class and action not in {"sleep_backoff", "readonly_cpu", "cpu_support", "tt_or_submission"}:
            if not (
                self._can_defer_light_train_guard_to_arbiter(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
                or self._can_ignore_soft_gpu_guard_for_available_slot(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
            ):
                violates = True
        if mode == "RED" and cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}:
            if not (
                self._can_defer_light_train_guard_to_arbiter(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
                or self._can_ignore_soft_gpu_guard_for_available_slot(
                    resource_class=cls,
                    gpu_ids=gpu_ids,
                    guard=pending,
                )
            ):
                violates = True
        if allowed and cls in {RESOURCE_GPU_TT_LIGHT, RESOURCE_PURE_TT_CPU, RESOURCE_GPU_FEATURE_EXTRACT} and cls not in allowed:
            violates = True
        hard_safety_reason = self._hard_safety_gate_reason(str(pending.get("reason") or ""))
        hard_safety_status = str(pending.get("status") or "").upper() == "DENIED_DUPLICATE"
        if allowed_current_class and not (hard_safety_reason or hard_safety_status):
            violates = False
        created_at = float(pending.get("created_at") or time.time())
        payload = {
            "feedback_id": str(pending.get("feedback_id") or ""),
            "source_job_id": str(pending.get("job_id") or ""),
            "source_status": str(pending.get("status") or ""),
            "source_reason": str(pending.get("reason") or ""),
            "source_resource_mode": str(pending.get("resource_mode") or ""),
            "source_gpu_ids": [str(x) for x in (pending.get("gpu_ids") or []) if str(x).strip()],
            "blocked_class": blocked_class,
            "allowed_classes": allowed,
            "action": action,
            "base_action": str(action_decoration.get("base_action") or action),
            "preceded_by_agent_backoff": bool(action_decoration.get("preceded_by_agent_backoff")),
            "agent_backoff_wait_id": str(action_decoration.get("agent_backoff_wait_id") or ""),
            "agent_backoff_elapsed_sec": float(action_decoration.get("agent_backoff_elapsed_sec") or 0.0),
            "agent_backoff_wake_reason": str(action_decoration.get("agent_backoff_wake_reason") or ""),
            "current_resource_class": cls,
            "current_command_digest": str(command_digest or ""),
            "same_command_digest": same_digest,
            "violates_feedback": violates,
            "elapsed_since_feedback_sec": max(0.0, time.time() - created_at),
            "retry_after_sec": pending.get("retry_after_sec"),
            "post_feedback_action_hint": str(pending.get("post_feedback_action") or ""),
        }
        self.state_machine.append_event(
            "post_feedback_action",
            task_type="resource_feedback",
            task_id=f"post_feedback:{pending.get('feedback_id') or self._feedback_seq}",
            status="violated" if violates else "observed",
            payload=payload,
        )
        return payload


    def _promote(self, job: ResourceJob, *, reason: str, elapsed_sec: float = 0.0) -> None:
        if job.visible:
            return
        job.visible = True
        self._emit(
            "resource_job_started",
            job,
            status="running",
            payload={"reason": reason, "elapsed_sec": float(elapsed_sec)},
        )
        if job.pid is not None:
            self._emit(
                "resource_lease_registered",
                job,
                status="running",
                payload={"pid": job.pid},
            )
