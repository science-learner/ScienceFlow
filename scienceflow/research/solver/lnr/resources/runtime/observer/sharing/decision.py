# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Validated GPU-sharing grants, revocations, proposals, and applied effects."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    ResourceEffectKind,
    ResourceJob,
    classify_cpu_pressure,
    gpu_memory_summary,
    hashlib,
    project_gpu_share_decision_facts,
    time,
)


class SharingCallbacks:
    """Own SharingCallbacks resource behavior without delegated forwarding."""

    def _grant_shared_gpu_if_allowed(self, job: ResourceJob, observation: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if self.resource_runtime is None or not self.gpu_share_config.grant_active:
            return {"enabled": False, "reason": "gpu_share_observe_only"}
        if not observation.get("share_eligible"):
            return {"enabled": False, "reason": "share_not_eligible"}
        waiter = observation.get("waiter") if isinstance(observation.get("waiter"), dict) else {}
        secondary_job_id = str(waiter.get("job_id") or "").strip()
        if not secondary_job_id:
            return {"enabled": False, "reason": "missing_secondary_job_id"}
        if not (observation.get("secondary_allowed") or {}).get("allowed"):
            return {"enabled": False, "reason": str((observation.get("secondary_allowed") or {}).get("reason") or "secondary_not_allowed")}
        secondary_allowed = observation.get("secondary_allowed") if isinstance(observation.get("secondary_allowed"), dict) else {}
        metadata = {
            "lease_mode": "shared_secondary",
            "shared_primary_job_id": job.job_id,
            "shared_primary_worker_id": self.worker_id,
            "shared_primary_resource_class": job.resource_class,
            "shared_grant_reason": "task_gpu_share_review",
            "shared_grant_phase": self.gpu_share_config.phase,
            "shared_grant_cpu_pressure": observation.get("cpu_pressure"),
            "shared_grant_memory": observation.get("memory") or {},
            "shared_grant_hard_gates": observation.get("hard_gates") or {},
            "shared_effective_resource_class": secondary_allowed.get("effective_secondary_class") or "",
            "shared_effective_slot_weight": secondary_allowed.get("effective_slot_weight"),
            "trial_share": bool((observation.get("trial_share") or {}).get("enabled")) if isinstance(observation.get("trial_share"), dict) else False,
            "trial_mode": str((observation.get("trial_share") or {}).get("mode") or "") if isinstance(observation.get("trial_share"), dict) else "",
            "trial_initial_observe_sec": float((observation.get("trial_share") or {}).get("initial_observe_sec") or 0.0) if isinstance(observation.get("trial_share"), dict) else 0.0,
            "trial_primary_protected": bool((observation.get("trial_share") or {}).get("primary_protected")) if isinstance(observation.get("trial_share"), dict) else False,
        }
        result = self._execute_resource_effect(
            ResourceEffectKind.GRANT_SHARED,
            command_id=secondary_job_id,
            idempotency_key=f"grant-shared:{job.job_id}:{secondary_job_id}",
            parameters={
                "primary_job_id": job.job_id,
                "secondary_job_id": secondary_job_id,
                "metadata": metadata,
            },
        )
        if result.get("acquired"):
            self._record_gpu_share_event(
                "shared_gpu_lease_granted",
                job,
                {**payload, "grant_result": result, "action": "GRANT_SHARED_GPU_LEASE"},
            )
            return {"enabled": True, "granted": True, "result": result}
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "gpu_share_denied_cpu_support",
                payload={**payload, "grant_result": result, "action": "DENY_SHARE_USE_CPU_SUPPORT"},
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"enabled": True, "granted": False, "result": result}


    def _shared_secondary_revoked_decision(self, job: ResourceJob) -> dict[str, Any]:
        if self.resource_runtime is None or str(job.lease_mode or "") != "shared_secondary":
            return {"enabled": False}
        if not self.resource_runtime.has_active_lease(job_id=job.job_id):
            return {"enabled": False}
        persisted = self.resource_runtime.persisted_lease(job_id=job.job_id)
        meta = persisted.get("metadata") if isinstance(persisted.get("metadata"), dict) else {}
        if persisted and str(meta.get("lease_mode") or "") == "shared_secondary":
            return {"enabled": False}
        payload = {
            "job_id": job.job_id,
            "primary_job_id": job.shared_primary_job_id,
            "lease_mode_before": "shared_secondary",
            "lease_mode_after": "revoked",
            "reason": "shared_secondary_lease_revoked",
        }
        self.resource_runtime.record_resource_event(
            "shared_gpu_secondary_stopped",
            payload=payload,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        return {
            "enabled": True,
            "terminate": True,
            "would_terminate": True,
            "reason": "active_intervention:shared_secondary_lease_revoked",
            "check_interval_sec": self.check_interval_sec,
            "feedback": (
                "RESOURCE_FEEDBACK: stopped_shared_secondary because the revocable shared GPU lease was withdrawn to protect the primary job. Replan or switch to CPU support work.\n"
            ),
        }


    def _shared_runtime_review_decision(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        if self.resource_runtime is None or not self.gpu_share_config.grant_active:
            return {"enabled": False}
        persisted = self.resource_runtime.persisted_lease(job_id=job.job_id)
        meta = persisted.get("metadata") if isinstance(persisted.get("metadata"), dict) else {}
        if str(meta.get("lease_mode") or "") != "shared_primary":
            return {"enabled": False}
        secondary_ids = [str(x) for x in (meta.get("shared_secondary_job_ids") or []) if str(x).strip()]
        if not secondary_ids:
            return {"enabled": False}
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        progress = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        memory = gpu_memory_summary(sample if isinstance(sample, dict) else {}, job.gpu_ids, previous_peak_gb=job.gpu_mem_peak_gb)
        cpu_pressure = classify_cpu_pressure(dict(signal.get("process_tree_cpu") or {}), cpu_policy=self.gpu_share_config.cpu_policy)
        progress_signal = str(progress.get("progress_signal") or "unknown").lower()
        windows = int(progress.get("progress_signal_windows") or 0)
        reason = ""
        if progress_signal in {"stalled", "degraded"} and windows >= max(1, int(self.gpu_share_config.revoke_min_windows or 1)):
            reason = f"primary_progress_{progress_signal}"
        elif int(signal.get("invalid_metric_events") or 0) >= self.metric_health_invalid_min_events:
            reason = "primary_invalid_metric"
        elif bool(memory.get("sample_available")) and float(memory.get("gpu_free_mem_gb") or 0.0) < self.gpu_share_config.safety_margin_gb:
            reason = "memory_headroom_below_safety_margin"
        proposal_payload = {
            "schema_version": 1,
            "proposal_type": "shared_runtime_review",
            "primary": {
                "job_id": job.job_id,
                "lease_mode": "shared_primary",
                "progress_signal": progress_signal,
                "progress_signal_windows": windows,
                "gpu_ids": list(job.gpu_ids or []),
                "resource_class": job.resource_class,
            },
            "secondary_job_ids": secondary_ids,
            "memory": memory,
            "cpu_pressure": cpu_pressure,
            "reason": reason or "continue_shared_observe",
        }
        self._record_gpu_share_event("shared_runtime_review_proposal", job, proposal_payload)
        if not reason:
            self._record_gpu_share_event(
                "shared_runtime_review_decision",
                job,
                {**proposal_payload, "action": "CONTINUE_SHARED_OBSERVE"},
            )
            return {"enabled": True, "terminate": False, "would_terminate": False, "action": "CONTINUE_SHARED_OBSERVE"}
        releases = []
        for secondary_id in secondary_ids:
            releases.append(
                self._execute_resource_effect(
                    ResourceEffectKind.REVOKE_SHARED,
                    command_id=secondary_id,
                    idempotency_key=f"revoke-shared:{job.job_id}:{secondary_id}:{reason}",
                    parameters={
                        "primary_job_id": job.job_id,
                        "secondary_job_id": secondary_id,
                        "reason": reason,
                    },
                )
            )
        self._record_gpu_share_event(
            "shared_runtime_review_decision",
            job,
            {**proposal_payload, "action": "STOP_SECONDARY_SHARED_JOB", "releases": releases},
        )
        return {
            "enabled": True,
            "terminate": False,
            "would_terminate": False,
            "action": "STOP_SECONDARY_SHARED_JOB",
            "reason": f"shared_runtime_review:{reason}",
            "feedback": (
                f"RESOURCE_FEEDBACK: STOP_SECONDARY_SHARED_JOB because shared GPU secondary work is harming or risking the primary; reason={reason}.\n"
            ),
            "releases": releases,
        }


    def _task_gpu_share_review_proposal(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
        progress_snapshot: dict[str, Any],
        observation: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        proposal_type = "task_gpu_share_review"
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((proposal_type + str(now)).encode('utf-8')).hexdigest()[:8]}"
        waiter = observation.get("waiter") if isinstance(observation.get("waiter"), dict) else {}
        trial_share = observation.get("trial_share") if isinstance(observation.get("trial_share"), dict) else {}
        reason_code = "task_gpu_share_review:revocable_trial_waiter" if trial_share.get("enabled") else "task_gpu_share_review:eligible_secondary_waiter"
        trigger_reasons = ["share_eligible", str((observation.get("secondary_allowed") or {}).get("reason") or "secondary_allowed")]
        if trial_share.get("enabled"):
            trigger_reasons.append("revocable_trial_share")
        share_decision_facts = project_gpu_share_decision_facts(observation, decision_source="primary_runtime")
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "severity": "yellow",
            "reason_code": reason_code,
            "trigger_reasons": trigger_reasons,
            "suggested_actions": ["GRANT_SHARED_GPU_LEASE", "DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"],
            "requires_llm_decision": True,
            "resource_snapshot": {
                "gpu_ids": list(job.gpu_ids or []),
                "cpu_pressure": observation.get("cpu_pressure"),
                "cpu_isolation": observation.get("cpu_isolation") or {},
                "memory": observation.get("memory") or {},
                "hard_gates": observation.get("hard_gates") or {},
                "secondary_allowed": observation.get("secondary_allowed") or {},
                "trial_share": trial_share,
            },
            "progress_snapshot": progress_snapshot,
            "blocker": {
                "job_id": job.job_id,
                "worker_id": self.worker_id,
                "resource_class": job.resource_class,
                "runtime_sec": elapsed_sec,
                "progress_signal": progress_snapshot.get("progress_signal"),
                "progress_confidence": progress_snapshot.get("progress_confidence"),
                "share_role": "primary",
            },
            "waiters": [waiter],
            "share_observation": observation,
            "share_decision_facts": share_decision_facts,
            "share_payload": payload,
            "decision_preview": {
                "job_id": job.job_id,
                "reason": reason_code,
                "would_terminate": False,
                "arbiter_review": True,
                "requires_llm_decision": True,
            },
            "command_id": job.job_id,
        }
        return proposal


    def apply_task_gpu_share_decision(
        self,
        job_id: str | None,
        *,
        arbiter_action: str = "",
        proposal: dict[str, Any] | None = None,
        arbiter_decision: dict[str, Any] | None = None,
        elapsed_sec: float = 0.0,
        **_: Any,
    ) -> dict[str, Any]:
        action = str(arbiter_action or "").upper()
        if not job_id or job_id not in self._jobs:
            return {"enabled": False, "reason": "job_not_found", "action": action}
        job = self._jobs[job_id]
        prop = proposal if isinstance(proposal, dict) else {}
        observation = prop.get("share_observation") if isinstance(prop.get("share_observation"), dict) else {}
        payload = prop.get("share_payload") if isinstance(prop.get("share_payload"), dict) else {}
        if action == "GRANT_SHARED_GPU_LEASE":
            grant = self._grant_shared_gpu_if_allowed(job, observation, payload)
            self._record_gpu_share_event(
                "gpu_share_decision",
                job,
                {**payload, "action": action, "arbiter_decision": arbiter_decision or {}, "grant": grant},
            )
            feedback = "RESOURCE_FEEDBACK: GRANT_SHARED_GPU_LEASE accepted; a revocable secondary may run on the underused task-local GPU.\n"
            if not grant.get("granted"):
                feedback = f"RESOURCE_FEEDBACK: GRANT_SHARED_GPU_LEASE could not be applied; reason={grant.get('reason') or (grant.get('result') or {}).get('reason') or 'share_grant_failed'}.\n"
            return {"enabled": True, "action": action, "granted": bool(grant.get("granted")), "grant": grant, "feedback": feedback}
        if action in {"DENY_SHARE_USE_CPU_SUPPORT", "CONTINUE_SHARED_OBSERVE"}:
            self._record_gpu_share_event(
                "gpu_share_decision",
                job,
                {**payload, "action": action, "arbiter_decision": arbiter_decision or {}},
            )
            feedback_action = "LOCAL_GPU_BUSY_USE_CPU_SUPPORT" if action == "DENY_SHARE_USE_CPU_SUPPORT" else "CONTINUE_SHARED_OBSERVE"
            return {
                "enabled": True,
                "action": action,
                "granted": False,
                "feedback": f"RESOURCE_FEEDBACK: {feedback_action} because shared GPU grant was not approved; use CPU support or observe until the next resource review.\n",
            }
        return {"enabled": False, "reason": "share_action_not_applicable", "action": action}
