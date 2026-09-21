# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource runtime responsibility: resource wait options, sharing overrides, ETA, and backoff."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    Any,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
    _GPUResourcePolicy,
    _GPU_PRESSURE_SUPPORT_CLASSES,
    _SHARE_OVERRIDE_CANDIDATE_CLASSES,
    _SHARE_OVERRIDE_PRIMARY_CLASSES,
    _TRIAL_SHARE_CANDIDATE_CLASSES,
    resource_feedback_text,
    time,
)


class AdmissionRuntime:
    """Own this responsibility's state transitions and callbacks."""

    @staticmethod
    def _resource_wait_scope_key(*, resource_class: str, gpu_ids: list[str], holder_job_ids: list[str]) -> str:
        gpu_key = ",".join(sorted({str(x) for x in (gpu_ids or []) if str(x).strip()}))
        holder_key = ",".join(sorted({str(x) for x in (holder_job_ids or []) if str(x).strip()}))
        return f"{str(resource_class or 'unknown')}|{gpu_key}|{holder_key}"


    def create_resource_wait_option(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        holder_job_ids: list[str] | None = None,
        queue_position: int | None = None,
        queue_len: int | None = None,
        command_digest: str = "",
        reason: str = "resource_busy",
        max_wait_sec: float | None = None,
    ) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        holders = [str(x) for x in (holder_job_ids or []) if str(x).strip()]
        scope_key = self._resource_wait_scope_key(resource_class=resource_class, gpu_ids=ids, holder_job_ids=holders)
        generation = self._current_pressure_generation()
        suppressed_until = float(self._resource_wait_suppressed_until.get(scope_key) or 0.0)
        if suppressed_until > time.time():
            return {"offered": False, "reason": "wait_suppressed_after_timeout", "scope_key": scope_key, "suppressed_until": suppressed_until}
        self._resource_wait_seq += 1
        token = f"rw-{self.worker_id}-{int(time.time() * 1000)}-{self._resource_wait_seq:04d}"
        wait_max = max(30.0, min(float(max_wait_sec or 600.0), 1800.0))
        record = {
            "wait_token": token,
            "worker_id": self.worker_id,
            "job_id": str(job_id or ""),
            "resource_class": str(resource_class or ""),
            "gpu_ids": ids,
            "holder_job_ids": holders,
            "queue_position": int(queue_position or 0),
            "queue_len": int(queue_len or 0),
            "command_digest": str(command_digest or ""),
            "reason": str(reason or "resource_busy"),
            "scope_key": scope_key,
            "start_pressure_generation": generation,
            "created_at": time.time(),
            "max_wait_sec": wait_max,
        }
        self._resource_wait_tokens[token] = record
        self.record_resource_event(
            "managed_resource_wait_offered",
            payload={**record, "cpu_support_preferred": True, "wait_is_optional": True},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return {"offered": True, **record}


    @staticmethod
    def _resource_wait_extra_facts(wait_option: dict[str, Any] | None) -> dict[str, str]:
        if not isinstance(wait_option, dict) or not wait_option.get("offered"):
            return {}
        return {
            "action_options": "cpu_support,resource_wait",
            "cpu_support_preferred": "true",
            "wait_is_optional": "true",
            "bash_sleep_allowed": "false",
            "wait_tool": "resource_wait",
            "wait_token": str(wait_option.get("wait_token") or ""),
            "wait_max_sec": "%.0f" % max(0.0, float(wait_option.get("max_wait_sec") or 0.0)),
            "wait_reason": str(wait_option.get("reason") or "resource_busy"),
        }


    @staticmethod
    def _duplicate_summary(details: dict[str, Any]) -> dict[str, Any]:
        raw = details.get("duplicate_digest_summary") if isinstance(details.get("duplicate_digest_summary"), dict) else {}
        active = [str(x) for x in (raw.get("duplicate_active_job_ids") or []) if str(x).strip()]
        waiters = [str(x) for x in (raw.get("duplicate_waiter_job_ids") or []) if str(x).strip()]
        try:
            oldest_wait = float(raw.get("same_digest_oldest_wait_sec") or 0.0)
        except (TypeError, ValueError):
            oldest_wait = 0.0
        return {
            "command_digest": str(raw.get("command_digest") or ""),
            "duplicate_active_job_ids": active,
            "duplicate_waiter_job_ids": waiters,
            "duplicate_count": len(active) + len(waiters),
            "same_digest_oldest_wait_sec": max(0.0, oldest_wait),
        }


    def _admission_action_for_blocked(
        self,
        *,
        policy: _GPUResourcePolicy,
        requested_count: int,
        details: dict[str, Any],
        priority: dict[str, float],
        eta_next_train_sec: float,
    ) -> tuple[str, str]:
        duplicate = self._duplicate_summary(details)
        priority_score = float(priority.get("admission_priority_score") or 0.0)
        duplicate_count = int(duplicate.get("duplicate_count") or 0)
        oldest_same_digest_wait = float(duplicate.get("same_digest_oldest_wait_sec") or 0.0)
        is_train = policy.resource_class in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}
        if duplicate_count > 0 and is_train and priority_score < 0.75:
            return "REPLAN", "duplicate_gpu_command_under_contention"
        if requested_count > 1 and oldest_same_digest_wait >= float(self.gpu_queue.admission_waiter_ttl_sec or 900.0):
            return "REPLAN", "multi_gpu_pending_wait_exceeded"
        # ETA is informational on the first blocked request. A multi-GPU task becomes
        # REPLAN only after the same pending command has actually waited too long.
        return "PENDING", str(details.get("reason") or "gpu_slot_unavailable")


    def _admission_feedback(
        self,
        *,
        action: str,
        reason: str,
        policy_resource_class: str,
        gpu_ids: list[str],
        queue_position: int,
        queue_len: int,
        eta_next_train_sec: float,
        eta_confidence: str,
        priority_score: float,
        holder_job_id: str = "",
        retry_after_sec: float | None = None,
        post_feedback_action: str = "",
        cached_backoff: bool = False,
        wait_option: dict[str, Any] | None = None,
    ) -> str:
        extra = {"admission_priority_score": f"{priority_score:.4f}"}
        extra.update(self._resource_wait_extra_facts(wait_option))
        if retry_after_sec is not None:
            extra["retry_after_sec"] = f"{max(0.0, float(retry_after_sec or 0.0)):.0f}"
        if post_feedback_action:
            extra["post_feedback_action"] = str(post_feedback_action)
        if cached_backoff:
            extra["cached_backoff"] = "true"
        return resource_feedback_text(
            status=action,
            reason=reason,
            scope="per_gpu",
            resource_mode="YELLOW",
            blocked_class=policy_resource_class,
            gpu_ids=gpu_ids,
            holder_job_id=holder_job_id,
            queue_position=queue_position,
            queue_len=queue_len,
            eta_next_train_sec=eta_next_train_sec,
            eta_confidence=eta_confidence,
            unlock_condition="holder_released" if holder_job_id else "resource_slot_available",
            blocked_until_unlock=True,
            extra_facts=extra,
        )


    @staticmethod
    def _blocker_job_ids(details: dict[str, Any]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        def add(raw: Any) -> None:
            job_id = str(raw or "").strip()
            if job_id and job_id not in seen:
                seen.add(job_id)
                out.append(job_id)
        for owners in (details.get("blockers") or {}).values() if isinstance(details.get("blockers"), dict) else []:
            for owner in owners or []:
                add(owner)
        blocker_details = details.get("blocker_details") if isinstance(details.get("blocker_details"), dict) else {}
        for raw in blocker_details.values():
            if not isinstance(raw, dict):
                continue
            for owner in raw.get("owners") or []:
                add(owner)
        return out


    @staticmethod
    def _share_override_blocker_reasons(details: dict[str, Any]) -> list[str]:
        reasons: set[str] = set()
        blocker_details = details.get("blocker_details") if isinstance(details.get("blocker_details"), dict) else {}
        for raw in blocker_details.values():
            if not isinstance(raw, dict):
                continue
            for reason in raw.get("reasons") or []:
                clean = str(reason or "").strip()
                if clean in {"slot_capacity_exceeded", "class_limit_exceeded", "incompatible_active_class"}:
                    reasons.add(clean)
        return sorted(reasons)


    def _admission_share_override_candidate(
        self,
        *,
        job_id: str,
        policy: _GPUResourcePolicy,
        details: dict[str, Any],
        priority: dict[str, float],
        value_hint: dict[str, Any],
    ) -> dict[str, Any]:
        if policy.resource_class not in _SHARE_OVERRIDE_CANDIDATE_CLASSES:
            return {"candidate": False, "reason": "secondary_class_not_share_override_candidate"}
        trial_share = policy.resource_class in _TRIAL_SHARE_CANDIDATE_CLASSES
        if trial_share:
            duplicate = self._duplicate_summary(details)
            if int(duplicate.get("duplicate_count") or 0) > 0:
                return {"candidate": False, "reason": "duplicate_gpu_command_not_trial_share_candidate"}
        block_reasons = self._share_override_blocker_reasons(details)
        trigger_reasons = {"incompatible_active_class", "slot_capacity_exceeded", "class_limit_exceeded"} if trial_share else {"incompatible_active_class"}
        if not (set(block_reasons) & trigger_reasons):
            return {"candidate": False, "reason": "no_share_override_block_reason", "block_reasons": block_reasons}
        blocker_ids = self._blocker_job_ids(details)
        if not blocker_ids:
            return {"candidate": False, "reason": "no_active_blocker"}
        try:
            snapshot = self.gpu_store.snapshot_active()
        except Exception:
            snapshot = {}
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        primary_ids: list[str] = []
        primary_classes: dict[str, str] = {}
        for blocker_id in blocker_ids:
            raw = leases.get(str(blocker_id)) if isinstance(leases.get(str(blocker_id)), dict) else {}
            meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            primary_class = str(meta.get("policy_resource_class") or meta.get("resource_class") or "")
            primary_classes[str(blocker_id)] = primary_class
            if primary_class in _SHARE_OVERRIDE_PRIMARY_CLASSES:
                primary_ids.append(str(blocker_id))
        if not primary_ids:
            return {
                "candidate": False,
                "reason": "no_heavy_primary_blocker",
                "blocker_job_ids": blocker_ids,
                "primary_classes": primary_classes,
                "block_reasons": block_reasons,
            }
        waiter = details.get("waiter") if isinstance(details.get("waiter"), dict) else {}
        required_gates = [
            "memory_high_water_mark_headroom",
            "cpu_isolation_or_low_secondary_cpu",
            "primary_progress_not_stalled",
            "primary_kill_replan_candidate_false",
            "secondary_revocation_available",
        ]
        if any(primary_classes.get(pid) not in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN} for pid in primary_ids):
            required_gates.append("non_heavy_holder_low_util_trial")
        if policy.resource_class == RESOURCE_UNKNOWN_GPU_EXEC:
            required_gates.extend([
                "unknown_gpu_conservative_memory_estimate",
                "unknown_gpu_short_initial_window",
                "unknown_gpu_requires_early_runtime_evidence",
            ])
        if trial_share:
            required_gates.extend([
                "arbiter_llm_grant_required",
                "trial_secondary_stop_first",
                "bounded_trial_initial_window",
            ])
        trial_initial_observe_sec = 90.0 if trial_share and policy.resource_class == RESOURCE_UNKNOWN_GPU_EXEC else 180.0 if trial_share else 0.0
        return {
            "candidate": True,
            "reason": "resource_contention_trial_share_review_required" if trial_share else "incompatible_active_class_share_review_required",
            "proposal_type": "task_gpu_share_review",
            "trial_share": trial_share,
            "trial_mode": "revocable_secondary_trial" if trial_share else "",
            "trial_initial_observe_sec": trial_initial_observe_sec,
            "trial_priority": "primary_protected" if trial_share else "",
            "primary_job_ids": primary_ids,
            "blocker_job_ids": blocker_ids,
            "primary_classes": primary_classes,
            "waiter_job_id": str(job_id or ""),
            "waiter_resource_class": policy.resource_class,
            "waiter_slot_weight": float(policy.slot_weight or 0.0),
            "waiter_effective_slot_weight": 0.5 if trial_share else float(policy.slot_weight or 0.0),
            "waiter_gpu_ids": [
                str(x)
                for x in (
                    waiter.get("gpu_ids")
                    or waiter.get("candidate_gpu_ids")
                    or details.get("candidate_gpu_ids")
                    or []
                )
                if str(x).strip()
            ],
            "waiter_priority_score": float(priority.get("admission_priority_score") or 0.0),
            "waiter_value_hint": dict(value_hint or {}),
            "block_reasons": block_reasons,
            "required_gates": required_gates,
        }


    def _eta_for_blocked(
        self,
        *,
        resource_class: str,
        details: dict[str, Any],
        command_digest: str = "",
        entrypoint: str = "",
    ) -> dict[str, Any]:
        cold = self._cold_eta_sec()
        summary = self.history_store.summary(
            resource_class=str(resource_class or ""),
            command_digest=str(command_digest or ""),
            entrypoint=str(entrypoint or ""),
        )
        count = int(summary.get("count") or 0)
        if count > 0:
            expected = max(60.0, float(summary.get("p80_sec") or summary.get("mean_sec") or cold))
            source = "runtime_history"
            confidence = "high" if count >= 8 else "medium" if count >= 3 else "low"
        else:
            expected = cold
            source = "cold_start"
            confidence = "low"
        blocker_ids = self._blocker_job_ids(details)
        active = self.gpu_store.snapshot_active()
        now = float(active.get("now") or time.time())
        leases = active.get("leases") if isinstance(active.get("leases"), dict) else {}
        ages: list[float] = []
        remaining: list[float] = []
        for job_id in blocker_ids:
            raw = leases.get(job_id) if isinstance(leases.get(job_id), dict) else {}
            if not raw:
                continue
            acquired_at = float(raw.get("acquired_at") or raw.get("heartbeat_at") or now)
            age = max(0.0, now - acquired_at)
            ages.append(age)
            floor = 60.0 if source == "runtime_history" else 300.0
            remaining.append(max(floor, expected - age))
        eta = min(remaining) if remaining else expected
        return {
            "eta_next_train_sec": max(0.0, float(eta)),
            "eta_confidence": confidence,
            "eta_source": source,
            "runtime_history_count": count,
            "runtime_avg_sec": float(summary.get("mean_sec") or 0.0),
            "runtime_p80_sec": float(summary.get("p80_sec") or 0.0),
            "active_blocker_count": len(blocker_ids),
            "active_blocker_job_ids": blocker_ids,
            "active_blocker_age_sec_max": max(ages or [0.0]),
        }


    def _admission_backoff_key(self, *, policy_resource_class: str, gpu_ids: list[str]) -> str:
        gpu_key = ",".join(sorted({str(x) for x in (gpu_ids or []) if str(x).strip()}))
        return f"{self.worker_id}:{str(policy_resource_class or '')}:{gpu_key}"


    def _current_pressure_generation(self) -> int:
        try:
            snap = self.pressure_store.snapshot()
            return int(snap.get("generation") or 0)
        except Exception:
            return 0


    def _waiter_wait_age_sec(self, details: dict[str, Any]) -> float:
        waiter = details.get("waiter") if isinstance(details.get("waiter"), dict) else {}
        try:
            submitted_at = float(waiter.get("submitted_at") or 0.0)
        except (TypeError, ValueError):
            submitted_at = 0.0
        if submitted_at <= 0:
            return 0.0
        return max(0.0, time.time() - submitted_at)


    def _admission_backoff_grace_sec(self) -> float:
        heartbeat = max(0.0, float(self.gpu_queue.heartbeat_sec or 0.0))
        return max(0.25, min(10.0, heartbeat * 4.0 if heartbeat > 0 else 10.0))


    def _admission_backoff_retry_sec(self, eta_next_train_sec: float) -> float:
        eta = max(0.0, float(eta_next_train_sec or 0.0))
        if eta > 0:
            return max(30.0, min(300.0, eta))
        return 120.0


    def _blockers_still_active(self, blocker_job_ids: list[str]) -> bool:
        holders = [str(x) for x in (blocker_job_ids or []) if str(x).strip()]
        if not holders:
            return False
        try:
            snapshot = self.gpu_store.snapshot_active()
        except Exception:
            return False
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        released_idle = snapshot.get("released_idle") if isinstance(snapshot.get("released_idle"), dict) else {}
        return any(job_id in leases or job_id in released_idle for job_id in holders)


    def _cached_admission_backoff(
        self,
        *,
        job_id: str,
        policy: _GPUResourcePolicy,
        target_gpu_ids: list[str],
        priority: dict[str, float],
    ) -> dict[str, Any] | None:
        if policy.resource_class not in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_UNKNOWN_GPU_EXEC}:
            return None
        key = self._admission_backoff_key(policy_resource_class=policy.resource_class, gpu_ids=target_gpu_ids)
        cached = self._admission_backoff_cache.get(key)
        if not isinstance(cached, dict):
            return None
        now = time.time()
        retry_until = float(cached.get("retry_until") or 0.0)
        if retry_until <= now:
            self._admission_backoff_cache.pop(key, None)
            return None
        if int(cached.get("pressure_generation") or 0) != self._current_pressure_generation():
            self._admission_backoff_cache.pop(key, None)
            return None
        blocker_ids = [str(x) for x in (cached.get("holder_job_ids") or []) if str(x).strip()]
        if not self._blockers_still_active(blocker_ids):
            self._admission_backoff_cache.pop(key, None)
            return None
        retry_after = max(1.0, retry_until - now)
        target_ids = [str(x) for x in (target_gpu_ids or cached.get("gpu_ids") or []) if str(x).strip()]
        opportunity = self.admission_opportunity_facts(
            resource_class=policy.resource_class,
            gpu_ids=target_ids,
            request_count=int(cached.get("requested_gpu_count") or 1),
        )
        if opportunity.get("lease_grantable_by_llm"):
            self._admission_backoff_cache.pop(key, None)
            return None
        priority_score = float(priority.get("admission_priority_score") or cached.get("admission_priority_score") or 0.0)
        feedback = self._admission_feedback(
            action="DENIED_REPLAN",
            reason="cached_gpu_admission_backoff",
            policy_resource_class=policy.resource_class,
            gpu_ids=target_ids,
            queue_position=int(cached.get("queue_position") or 1),
            queue_len=int(cached.get("queue_len") or 0),
            eta_next_train_sec=float(cached.get("eta_next_train_sec") or retry_after),
            eta_confidence=str(cached.get("eta_confidence") or "low"),
            priority_score=priority_score,
            holder_job_id=blocker_ids[0] if blocker_ids else "",
            retry_after_sec=retry_after,
            post_feedback_action="cpu_support",
            cached_backoff=True,
            wait_option=None,
        )
        release = self.lease_manager.release(str(job_id or ""))
        result = {
            "enabled": True,
            "acquired": False,
            "status": "DENIED_REPLAN",
            "admission_action": "DENIED_REPLAN",
            "reason": "cached_gpu_admission_backoff",
            "resource_mode": "YELLOW",
            "max_wait_sec": float(self.gpu_queue.max_wait_sec),
            "heartbeat_sec": float(self.gpu_queue.heartbeat_sec),
            "gpu_ids": target_ids,
            "assigned_physical_gpus": [],
            "allowed_physical_gpus": [str(x) for x in (self._configured_gpu_pool() or target_ids) if str(x).strip()],
            "candidate_physical_gpus": target_ids,
            "requested_gpu_count": int(cached.get("requested_gpu_count") or 1),
            "resource_class": str(cached.get("resource_class") or policy.resource_class),
            "policy_resource_class": policy.resource_class,
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "details": {"cached_backoff": cached, "release": release},
            "queue_started_first": False,
            "queue_position": int(cached.get("queue_position") or 1),
            "queue_len": int(cached.get("queue_len") or 0),
            "top_waiter_job_id": str(cached.get("top_waiter_job_id") or ""),
            "eta_next_train_sec": float(cached.get("eta_next_train_sec") or retry_after),
            "eta_confidence": str(cached.get("eta_confidence") or "low"),
            "holder_job_ids": blocker_ids,
            "allowed_classes": _GPU_PRESSURE_SUPPORT_CLASSES,
            "feedback": feedback,
            "admission_cached_backoff": True,
            "retry_after_sec": retry_after,
            "blocked_until_unlock": True,
            "unlock_condition": "holder_released",
            "post_feedback_action": "cpu_support",
            **priority,
        }
        self.record_resource_event(
            "admission_cached_backoff",
            payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return result


    def _remember_admission_backoff(
        self,
        *,
        key: str,
        policy: _GPUResourcePolicy,
        target_gpu_ids: list[str],
        result: dict[str, Any],
        holder_job_ids: list[str],
        eta_next_train_sec: float,
        eta_confidence: str,
        pressure_generation: int,
    ) -> None:
        if policy.resource_class not in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_UNKNOWN_GPU_EXEC}:
            return
        retry_after = self._admission_backoff_retry_sec(eta_next_train_sec)
        self._admission_backoff_cache[key] = {
            "created_at": time.time(),
            "retry_until": time.time() + retry_after,
            "retry_after_sec": retry_after,
            "pressure_generation": int(pressure_generation or 0),
            "gpu_ids": [str(x) for x in (target_gpu_ids or []) if str(x).strip()],
            "holder_job_ids": [str(x) for x in (holder_job_ids or []) if str(x).strip()],
            "resource_class": str(result.get("resource_class") or policy.resource_class),
            "policy_resource_class": policy.resource_class,
            "queue_position": int(result.get("queue_position") or 1),
            "queue_len": int(result.get("queue_len") or 0),
            "top_waiter_job_id": str(result.get("top_waiter_job_id") or ""),
            "requested_gpu_count": int(result.get("requested_gpu_count") or 1),
            "eta_next_train_sec": float(eta_next_train_sec or 0.0),
            "eta_confidence": str(eta_confidence or "low"),
            "admission_priority_score": float(result.get("admission_priority_score") or 0.0),
        }


    def _resource_wait_wake_reason(self, record: dict[str, Any]) -> str:
        start_generation = int(record.get("start_pressure_generation") or 0)
        current_generation = self._current_pressure_generation()
        if current_generation != start_generation:
            return "pressure_generation_changed"
        holders = [str(x) for x in (record.get("holder_job_ids") or []) if str(x).strip()]
        if holders and not self._blockers_still_active(holders):
            return "holder_released"
        gpu_ids = [str(x) for x in (record.get("gpu_ids") or []) if str(x).strip()]
        if gpu_ids:
            try:
                snapshot = self.gpu_store.snapshot_active()
                leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
                busy = False
                for lease in leases.values():
                    if not isinstance(lease, dict):
                        continue
                    lease_gpu_ids = {str(x) for x in (lease.get("gpu_ids") or []) if str(x).strip()}
                    if lease_gpu_ids.intersection(set(gpu_ids)):
                        busy = True
                        break
                if not busy:
                    return "gpu_slot_released"
            except Exception:
                return ""
        return ""


    def _resource_wait_feedback(
        self,
        *,
        status: str,
        reason: str,
        record: dict[str, Any] | None = None,
        extra_facts: dict[str, Any] | None = None,
    ) -> str:
        rec = record or {}
        return resource_feedback_text(
            status=status,
            reason=reason,
            scope="per_gpu",
            resource_mode="YELLOW",
            blocked_class=str(rec.get("resource_class") or "resource_wait"),
            gpu_ids=[str(x) for x in (rec.get("gpu_ids") or []) if str(x).strip()],
            holder_job_id=([str(x) for x in (rec.get("holder_job_ids") or []) if str(x).strip()] or [""])[0],
            queue_position=(int(rec.get("queue_position") or 0) if rec.get("queue_position") is not None else None),
            queue_len=(int(rec.get("queue_len") or 0) if rec.get("queue_len") is not None else None),
            unlock_condition="resource_state_changed" if status == "RESOURCE_AVAILABLE" else "resource_state_change_required",
            blocked_until_unlock=(False if status == "RESOURCE_AVAILABLE" else True),
            extra_facts=extra_facts or {},
        )
