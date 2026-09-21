# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""GPU-sharing phase, contention, cadence, efficiency, and quick-probe scheduling."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    ResourceJob,
    _SAFETY_HEARTBEAT_SOURCE,
    build_resource_state_generation,
    classify_active_lease_suspect,
    evaluate_share_phase_a,
    gpu_memory_summary,
    hashlib,
    time,
)


class SharingScheduling:
    """Own SharingScheduling resource behavior without delegated forwarding."""

    def _gpu_share_phase_a_decision(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        allow_grant: bool = True,
        allow_handoff: bool = True,
    ) -> dict[str, Any]:
        cfg = self.gpu_share_config
        if not (cfg.observe_active and self.resource_runtime is not None):
            return {"enabled": False}
        if signal.get("saw_final_score"):
            return {"enabled": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"enabled": False}
        if not self._job_uses_expensive_gpu(job):
            return {"enabled": False}
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return {"enabled": False, "reason": "gpu_share_snapshot_failed"}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        if not isinstance(sample, dict):
            sample = {}
        primary_kill = self._primary_kill_replan_candidate(job, signal)
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        observation = evaluate_share_phase_a(
            cfg=cfg,
            primary_job_id=job.job_id,
            primary_resource_class=job.resource_class,
            primary_gpu_ids=job.gpu_ids,
            elapsed_sec=elapsed,
            progress_snapshot=progress_snapshot,
            gpu_sample=sample,
            previous_mem_peak_gb=job.gpu_mem_peak_gb,
            process_tree_cpu=dict(signal.get("process_tree_cpu") or {}),
            snapshot=snapshot,
            primary_kill_replan_candidate=primary_kill,
        )
        if not observation.get("enabled"):
            return observation
        payload = {
            "schema_version": 1,
            "phase": observation.get("phase"),
            "observe_only": bool(observation.get("observe_only")),
            "primary": {
                "job_id": job.job_id,
                "worker_id": self.worker_id,
                "resource_class": job.resource_class,
                "gpu_ids": list(job.gpu_ids or []),
                "runtime_sec": elapsed,
                "progress_signal": progress_snapshot.get("progress_signal"),
                "progress_confidence": progress_snapshot.get("progress_confidence"),
                "gpu_mem_peak_gb": job.gpu_mem_peak_gb,
            },
            "waiter": observation.get("waiter") or {},
            "waiters": observation.get("waiters") or [],
            "cpu_pressure": observation.get("cpu_pressure"),
            "cpu_isolation": observation.get("cpu_isolation") or {},
            "memory": observation.get("memory") or {},
            "hard_gates": observation.get("hard_gates") or {},
            "secondary_allowed": observation.get("secondary_allowed") or {},
            "trial_share": observation.get("trial_share") or {},
            "primary_kill_replan_candidate": primary_kill,
        }
        if primary_kill.get("candidate"):
            self._record_gpu_share_event("primary_kill_replan_candidate", job, payload)
            if primary_kill.get("share_blocking_candidate") and allow_handoff:
                review = self._periodic_efficiency_review_decision(
                    job,
                    signal,
                    force=True,
                    trigger_source="gpu_share_handoff",
                    trigger_reasons=[str(x) for x in (primary_kill.get("reasons") or [])],
                )
                if isinstance(review, dict) and review.get("enabled"):
                    return {**review, "gpu_share_handoff": True}
            if primary_kill.get("share_blocking_candidate"):
                return {
                    "enabled": True,
                    "gpu_share_handoff": bool(allow_handoff),
                    "handoff_review_enabled": False,
                    "observation": observation,
                }
        if observation.get("share_candidate"):
            self._record_gpu_share_event("gpu_share_candidate", job, payload)
        grant: dict[str, Any] = {"enabled": False}
        if observation.get("share_eligible"):
            self._record_gpu_share_event("gpu_share_eligible", job, payload)
            secondary_allowed = observation.get("secondary_allowed") if isinstance(observation.get("secondary_allowed"), dict) else {}
            requires_llm_policy_decision = bool(secondary_allowed.get("requires_llm_policy_decision"))
            if allow_grant and self.arbiter_enabled and self.arbiter_mode == "llm":
                proposal = self._task_gpu_share_review_proposal(
                    job,
                    signal,
                    elapsed_sec=elapsed,
                    progress_snapshot=progress_snapshot,
                    observation=observation,
                    payload=payload,
                )
                self._record_resource_review_proposal_event(job, proposal, signal)
                return {
                    "enabled": True,
                    "terminate": False,
                    "would_terminate": False,
                    "arbiter_review": True,
                    "requires_llm_decision": True,
                    "arbiter_enabled": True,
                    "reason": str(proposal.get("reason_code") or "task_gpu_share_review:eligible_secondary_waiter"),
                    "check_interval_sec": self.check_interval_sec,
                    "feedback": "RESOURCE_FEEDBACK: task_gpu_share_review because a queued waiter can potentially share an underused task-local GPU; arbiter review requested.\n",
                    "observation": observation,
                    "proposal": proposal,
                }
            if requires_llm_policy_decision:
                grant = {
                    "enabled": False,
                    "reason": "share_policy_requires_llm_arbiter",
                    "policy_gate": secondary_allowed.get("policy_gate") or "secondary_policy",
                }
            elif allow_grant:
                grant = self._grant_shared_gpu_if_allowed(job, observation, payload)
            else:
                grant = {"enabled": False, "reason": "post_arbiter_observe_only"}
        feedback = ""
        if grant.get("granted"):
            feedback = "RESOURCE_FEEDBACK: GRANT_SHARED_GPU_LEASE accepted; a revocable secondary may run on the underused task-local GPU.\n"
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "feedback": feedback,
            "observation": observation,
            "shared_lease_grant": grant,
        }


    def _contention_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if not (self.arbiter_enabled and self.arbiter_contention_review_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if elapsed < self.arbiter_contention_min_runtime_sec or signal.get("saw_final_score"):
            return {"enabled": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"enabled": False}
        if not self._job_uses_expensive_gpu(job):
            return {"enabled": False}
        now = time.time()
        if self._review_cooldown_active(job, "resource_contention_review", min_interval_sec=self.arbiter_contention_min_interval_sec, now=now):
            return {"enabled": False}
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return {"enabled": False}
        waiters = self._contention_waiter_cards(job, snapshot=snapshot, now=now)
        if not waiters:
            return {"enabled": False}
        resource_confidence = self._resource_confidence_for_review(job, has_active_lease=True, waiter_count=len(waiters))
        proposal_type = "resource_contention_review"
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((proposal_type + str(now)).encode('utf-8')).hexdigest()[:8]}"
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        route_viability = self._update_v7_route_profile(job, signal, elapsed_sec=elapsed)
        progress_snapshot.update({
            "route_viability": dict(route_viability or {}),
        })
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        memory = gpu_memory_summary(sample if isinstance(sample, dict) else {}, job.gpu_ids, previous_peak_gb=job.gpu_mem_peak_gb)
        suspect = classify_active_lease_suspect(
            elapsed_sec=elapsed,
            has_waiter=True,
            has_active_lease=True,
            resource_class=job.resource_class,
            progress_snapshot=progress_snapshot,
            gpu_memory=memory,
            deliverable_validity="none",
            route_viability=route_viability,
            thresholds=self._lease_suspect_thresholds(),
        )
        job.active_lease_suspect_state = dict(suspect)
        trigger_reasons = [
            "active_gpu_lease",
            "pending_waiter_for_same_gpu",
            f"waiter_age_sec>={self.arbiter_contention_min_waiter_age_sec:.0f}",
        ]
        if suspect.get("resource_suspect"):
            trigger_reasons.append("active_lease_resource_suspect")
        if suspect.get("unknown_progress_suspect"):
            trigger_reasons.append("active_lease_unknown_progress_suspect")
        if suspect.get("value_suspect"):
            trigger_reasons.append("active_lease_value_suspect")
        suggested_actions = ["CONTINUE", "OBSERVE_MORE"]
        if suspect.get("resource_suspect"):
            suggested_actions.append("KILL_AND_REPLAN")
        if suspect.get("active_lease_suspect"):
            self._record_v7_resource_event(
                "active_lease_suspect",
                job,
                {
                    "proposal_type": proposal_type,
                    "waiter_count": len(waiters),
                    "waiters": waiters,
                    "active_lease_suspect": suspect,
                    "route_viability": route_viability,
                    "memory": memory,
                },
            )
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "yellow" if not suspect.get("resource_suspect") else "orange",
            "reason_code": "resource_contention:active_gpu_job_blocking_waiter",
            "trigger_reasons": trigger_reasons,
            "suggested_actions": suggested_actions,
            "requires_llm_decision": True,
            "resource_snapshot": self._resource_snapshot_for_job(job, signal),
            "progress_snapshot": progress_snapshot,
            "blocker": self._blocker_fact_card(job, signal, elapsed_sec=elapsed, resource_confidence=resource_confidence),
            "waiters": waiters,
            "decision_preview": {
                "job_id": job.job_id,
                "reason": "resource_contention_review:active_gpu_job_blocking_waiter",
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        if suspect.get("unknown_progress_suspect") and proposal.get("structured_opportunity_cost_support"):
            if "KILL_AND_REPLAN" not in suggested_actions:
                suggested_actions.append("KILL_AND_REPLAN")
            if "active_lease_unknown_progress_escalated" not in trigger_reasons:
                trigger_reasons.append("active_lease_unknown_progress_escalated")
            proposal["suggested_actions"] = suggested_actions
            proposal["trigger_reasons"] = trigger_reasons
            proposal["severity"] = "orange"
        self._mark_review_proposal_emitted(job, proposal_type, now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": "resource_contention_review:active_gpu_job_blocking_waiter",
            "check_interval_sec": self.check_interval_sec,
            "feedback": "RESOURCE_FEEDBACK: resource_contention_review because an active GPU lease is blocking a queued waiter; arbiter review requested.\n",
            "proposal": proposal,
        }


    def _research_cadence_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if not (self.research_cadence_enabled and self.arbiter_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        if not job.visible or signal.get("saw_final_score"):
            return {"enabled": False}
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        cadence = self._research_cadence_fact_card(job, signal, progress_snapshot)
        if not cadence.get("violation"):
            return {"enabled": False}
        now = time.time()
        proposal_type = "periodic_efficiency_review"
        min_interval = max(60.0, min(self.arbiter_periodic_min_interval_sec, self.research_cadence_observe_sec))
        if self._review_cooldown_active(job, proposal_type, min_interval_sec=min_interval, now=now):
            return {"enabled": False}
        reason_code = str(cadence.get("reason_code") or "research_cadence_exceeded")
        resource_snapshot = self._resource_snapshot_for_job(job, signal)
        blocker = self._blocker_fact_card(
            job,
            signal,
            elapsed_sec=elapsed,
            resource_confidence=self._resource_confidence_for_review(
                job,
                has_active_lease=bool(self.resource_runtime.has_active_lease(job_id=job.job_id)),
                waiter_count=0,
            ),
        )
        execution_facts = self._execution_facts_for_job(
            job,
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
        )
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((reason_code + str(now)).encode('utf-8')).hexdigest()[:8]}"
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "orange",
            "reason_code": reason_code,
            "trigger_reasons": [
                "research_cadence_exceeded",
                f"metric_budget_sec={float(cadence.get('metric_budget_sec') or 0.0):.0f}",
                f"route_metric_proven={str(bool(cadence.get('route_metric_proven'))).lower()}",
            ],
            "suggested_actions": ["CONTINUE", "OBSERVE_MORE", "KILL_AND_REPLAN"],
            "requires_llm_decision": True,
            "structured_opportunity_cost_support": True,
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "execution_facts": execution_facts,
            "research_cadence": cadence,
            "blocker": blocker,
            "waiters": [],
            "decision_preview": {
                "job_id": job.job_id,
                "reason": reason_code,
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        self._mark_review_proposal_emitted(job, proposal_type, now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        feedback = (
            "RESOURCE_FEEDBACK: research_cadence_review because the observed or projected time to the next comparable metric "
            f"exceeds the configured value window; elapsed_sec={elapsed:.0f}; "
            f"eta_to_next_comparable_metric_sec={cadence.get('eta_to_next_comparable_metric_sec')}; "
            f"metric_budget_sec={float(cadence.get('metric_budget_sec') or 0.0):.0f}; "
            f"route_metric_proven={str(bool(cadence.get('route_metric_proven'))).lower()}.\n"
        )
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": reason_code,
            "check_interval_sec": self.check_interval_sec,
            "feedback": feedback,
            "proposal": proposal,
        }


    def _periodic_efficiency_review_decision(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        force: bool = False,
        trigger_source: str = "periodic_efficiency",
        trigger_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        if not (self.arbiter_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        efficiency = (
            signal.get("resource_efficiency")
            or job.resource_efficiency_state
            or {}
        )
        efficiency_trigger = bool(
            efficiency.get("review_required")
            and not efficiency.get("review_limit_reached")
        )
        if not force and not self.arbiter_periodic_review_enabled and not efficiency_trigger:
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        min_runtime = (
            0.0
            if force or efficiency_trigger
            else self.arbiter_periodic_min_runtime_sec
        )
        if elapsed < min_runtime or not job.visible or signal.get("saw_final_score"):
            return {"enabled": False}
        if not self._job_is_expensive_resource_hold(job):
            return {"enabled": False}
        progress = str(job.progress_signal or "unknown")
        merged_reasons: list[str] = [str(x) for x in (trigger_reasons or []) if str(x).strip()]
        if efficiency_trigger:
            merged_reasons.append("sustained_resource_efficiency_mismatch")
            merged_reasons.extend(
                str(reason)
                for reason in efficiency.get("reason_codes") or []
                if str(reason).strip()
            )
        if progress in {"stalled", "degraded"}:
            merged_reasons.append(f"progress_signal={progress}")
        if job.idle_gpu_lease_samples > 0:
            merged_reasons.append("idle_gpu_lease_samples_present")
        if job.dataloader_bottleneck_samples > 0:
            merged_reasons.append("dataloader_bottleneck_samples_present")
        deduped_reasons = list(dict.fromkeys(merged_reasons))
        if not deduped_reasons and not force:
            return {"enabled": False}
        if not deduped_reasons:
            deduped_reasons.append("primary_kill_replan_candidate")
        now = time.time()
        if self._review_cooldown_active(job, "periodic_efficiency_review", min_interval_sec=self.arbiter_periodic_min_interval_sec, now=now):
            return {"enabled": False}
        has_lease = bool(self.resource_runtime.has_active_lease(job_id=job.job_id))
        resource_confidence = self._resource_confidence_for_review(job, has_active_lease=has_lease, waiter_count=0)
        proposal_type = "periodic_efficiency_review"
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((proposal_type + str(now)).encode('utf-8')).hexdigest()[:8]}"
        periodic_suggested_actions = ["CONTINUE", "OBSERVE_MORE", "KILL_AND_REPLAN"]
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "yellow",
            "reason_code": (
                "gpu_share_handoff:primary_kill_replan_candidate"
                if force
                else (
                    "resource_efficiency:sustained_resource_mismatch"
                    if efficiency_trigger
                    else "periodic_efficiency:low_progress_review"
                )
            ),
            "trigger_reasons": deduped_reasons,
            "trigger_source": str(trigger_source or "periodic_efficiency"),
            "suggested_actions": periodic_suggested_actions,
            "requires_llm_decision": True,
            "resource_snapshot": self._resource_snapshot_for_job(job, signal),
            "progress_snapshot": self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed),
            "blocker": self._blocker_fact_card(job, signal, elapsed_sec=elapsed, resource_confidence=resource_confidence),
            "waiters": [],
            "decision_preview": {
                "job_id": job.job_id,
                "reason": "gpu_share_handoff:primary_kill_replan_candidate" if force else "periodic_efficiency_review:low_progress_or_low_efficiency",
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        self._mark_review_proposal_emitted(job, proposal_type, now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": "gpu_share_handoff:primary_kill_replan_candidate" if force else "periodic_efficiency_review:low_progress_or_low_efficiency",
            "check_interval_sec": self.check_interval_sec,
            "feedback": (
                "RESOURCE_FEEDBACK: PRIMARY_KILL_REPLAN_CANDIDATE because a task-local GPU share review found the current GPU holder should enter kill/replan review; arbiter review requested.\n"
                if force
                else "RESOURCE_FEEDBACK: periodic_efficiency_review because long-running resource usage has degraded signals; arbiter review requested.\n"
            ),
            "proposal": proposal,
        }


    def _quick_probe_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if not (self.quick_probe_guard_enabled and self.arbiter_enabled and self.resource_runtime is not None):
            return {"enabled": False}
        quick = dict(job.quick_probe or {})
        if not quick.get("quick_probe_candidate"):
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        hard_review = max(1.0, float(quick.get("quick_probe_hard_review_sec") or self.quick_probe_hard_review_sec))
        if elapsed < hard_review or signal.get("saw_final_score"):
            return {"enabled": False}
        if int(signal.get("terminal_signal_events") or 0) > 0 or job.terminal_signal_seen:
            return {"enabled": False}
        if not job.visible:
            return {"enabled": False}
        now = time.time()
        min_interval = min(300.0, max(60.0, hard_review / 3.0))
        if self._review_cooldown_active(job, "quick_probe_review", min_interval_sec=min_interval, now=now):
            return {"enabled": False}
        expected = max(1.0, float(quick.get("quick_probe_expected_runtime_sec") or self.quick_probe_expected_runtime_sec))
        output_pattern = str(quick.get("output_pattern") or "unknown")
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        progress_snapshot.update({
            "quick_probe": quick,
            "quick_probe_candidate": True,
            "quick_probe_expected_runtime_sec": expected,
            "quick_probe_hard_review_sec": hard_review,
            "quick_probe_runtime_ratio": elapsed / expected if expected > 0 else None,
            "output_pattern": output_pattern,
            "heartbeat_source": progress_snapshot.get("heartbeat_source") or _SAFETY_HEARTBEAT_SOURCE,
        })
        resource_snapshot = self._resource_snapshot_for_job(job, signal)
        blocker = self._blocker_fact_card(
            job,
            signal,
            elapsed_sec=elapsed,
            resource_confidence=self._resource_confidence_for_review(
                job,
                has_active_lease=bool(self.resource_runtime.has_active_lease(job_id=job.job_id)),
                waiter_count=0,
            ),
        )
        execution_facts = self._execution_facts_for_job(
            job,
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
        )
        state_generation = build_resource_state_generation({
            "proposal_type": "quick_probe_review",
            "reason_code": "quick_probe_runtime_exceeded",
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "blocker": blocker,
            "gpu_ids": list(job.gpu_ids or []),
            "execution_facts": execution_facts,
        })
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1(('quick_probe_runtime_exceeded' + state_generation.control_generation_key).encode('utf-8')).hexdigest()[:8]}"
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": "quick_probe_review",
            "severity": "red",
            "reason_code": "quick_probe_runtime_exceeded",
            "trigger_reasons": [
                "quick_probe_candidate",
                f"elapsed_sec>={hard_review:.0f}",
                "terminal_signal_missing",
            ],
            "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
            "requires_llm_decision": True,
            "cooldown_break": True,
            "control_generation_key": state_generation.control_generation_key,
            "feedback_generation_key": state_generation.feedback_generation_key,
            "state_generation": state_generation.to_json(),
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "execution_facts": execution_facts,
            "blocker": blocker,
            "waiters": [],
            "structured_opportunity_cost_support": elapsed >= max(3600.0, hard_review * 4.0),
            "decision_preview": {
                "job_id": job.job_id,
                "reason": "quick_probe_runtime_exceeded",
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        proposal.update(self._attach_review_history(proposal, now=now))
        self._mark_review_proposal_emitted(job, "quick_probe_review", now=now)
        self._record_resource_review_proposal_event(job, proposal, signal)
        payload = {
            "job_id": job.job_id,
            "elapsed_sec": elapsed,
            "expected_runtime_sec": expected,
            "hard_review_sec": hard_review,
            "output_pattern": output_pattern,
            "control_generation_key": state_generation.control_generation_key,
            "feedback_generation_key": state_generation.feedback_generation_key,
            "proposal_id": proposal_id,
        }
        self.resource_runtime.record_resource_event(
            "quick_probe_forced_review",
            payload=payload,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "arbiter_review": True,
            "requires_llm_decision": True,
            "arbiter_enabled": True,
            "reason": "quick_probe_runtime_exceeded",
            "check_interval_sec": self.check_interval_sec,
            "feedback": (
                "RESOURCE_FEEDBACK: quick_probe_runtime_exceeded because a command classified as a short quick/probe/test run exceeded its hard review window; "
                f"elapsed_sec={elapsed:.0f}; expected_runtime_sec={expected:.0f}; hard_review_sec={hard_review:.0f}; output_pattern={output_pattern}.\n"
            ),
            "proposal": proposal,
        }
