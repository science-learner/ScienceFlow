# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource runtime responsibility: shared leases, availability, release, and environment projection."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    Any,
    RESOURCE_HEAVY_GPU_TRAIN,
)


class LeaseRuntime:
    """Own this responsibility's state transitions and callbacks."""

    def grant_shared_gpu_lease(
        self,
        *,
        primary_job_id: str,
        secondary_job_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.gpu_store.grant_shared_secondary(
            primary_job_id=str(primary_job_id or ""),
            secondary_job_id=str(secondary_job_id or ""),
            metadata=dict(metadata or {}),
            max_secondary_per_primary=1,
        )
        lease = result.get("lease") if isinstance(result.get("lease"), dict) else {}
        assigned = [str(x) for x in (lease.get("gpu_ids") or result.get("assigned_gpu_ids") or []) if str(x).strip()]
        if result.get("acquired") and assigned:
            self.pressure_store.bump_generation(reason="shared_gpu_lease_granted")
            self.record_resource_event(
                "gpu_share_decision",
                payload={
                    "proposal_type": "task_gpu_share_review",
                    "action": "GRANT_SHARED_GPU_LEASE",
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "result": result,
                },
                command_id=str(primary_job_id or ""),
                lease_id=str(primary_job_id or ""),
            )
            self.record_resource_event(
                "shared_gpu_lease_granted",
                payload={
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "gpu_ids": assigned,
                    "lease_mode": "shared_secondary",
                    "result": result,
                },
                command_id=str(secondary_job_id or ""),
                lease_id=str(secondary_job_id or ""),
            )
        elif not result.get("acquired"):
            self.record_resource_event(
                "gpu_share_decision",
                payload={
                    "proposal_type": "task_gpu_share_review",
                    "action": "DENY_SHARE_USE_CPU_SUPPORT",
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "reason": str(result.get("reason") or "shared_lease_denied"),
                    "result": result,
                },
                command_id=str(primary_job_id or ""),
                lease_id=str(primary_job_id or ""),
            )
        return {**result, "assigned_gpu_ids": assigned}


    def revoke_shared_gpu_lease(
        self,
        *,
        primary_job_id: str,
        secondary_job_id: str,
        reason: str,
    ) -> dict[str, Any]:
        release = self.lease_manager.release(str(secondary_job_id or ""))
        if release.get("released"):
            self.pressure_store.bump_generation(reason="shared_gpu_lease_revoked")
            self.record_resource_event(
                "shared_gpu_secondary_stopped",
                payload={
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "reason": str(reason or "shared_runtime_review"),
                    "release": release,
                },
                command_id=str(secondary_job_id or ""),
                lease_id=str(secondary_job_id or ""),
            )
            self.record_resource_event(
                "shared_gpu_lease_revoked",
                payload={
                    "primary_job_id": str(primary_job_id or ""),
                    "secondary_job_id": str(secondary_job_id or ""),
                    "reason": str(reason or "shared_runtime_review"),
                    "release": release,
                },
                command_id=str(secondary_job_id or ""),
                lease_id=str(secondary_job_id or ""),
            )
        return release


    def resource_available_for(self, *, resource_class: str, gpu_ids: list[str] | None = None) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        cls = str(resource_class or RESOURCE_HEAVY_GPU_TRAIN)
        policy = self._policy_for_resource_class(cls)
        if policy is None:
            return {"available": True, "reason": "unmanaged_resource_class", "gpu_ids": ids}
        candidates = ids or self._configured_gpu_pool()
        if not candidates:
            return {"available": True, "reason": "resource_changed_no_gpu_candidates", "gpu_ids": [], "generation": self.pressure_generation()}
        pressure = self.pressure_gate_decision(resource_class=policy.resource_class, gpu_ids=candidates)
        if isinstance(pressure, dict) and pressure.get("blocked"):
            return {
                "available": False,
                "reason": str(pressure.get("reason") or "resource_pressure"),
                "gpu_ids": candidates,
                "pressure": pressure.get("pressure") or {},
                "generation": self.pressure_generation(),
            }
        availability = self.gpu_store.availability(
            candidate_gpu_ids=candidates,
            request_count=self._request_count(len(candidates), requested=None),
            max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
            resource_class=policy.resource_class,
            slot_weight=policy.slot_weight,
            capacity_slots=self._capacity_slots(),
            max_per_gpu=policy.max_per_gpu,
            incompatible_classes=list(policy.incompatible_classes),
        )
        return {
            **availability,
            "reason": "resource_available" if availability.get("available") else "gpu_slot_unavailable",
            "gpu_ids": candidates,
            "generation": self.pressure_generation(),
        }


    def release_idle_lease(
        self,
        *,
        job_id: str,
        elapsed_sec: float = 0.0,
        reason: str = "idle_gpu_lease",
        resident_mem_gb: float = 0.0,
    ) -> dict[str, Any]:
        """Release a GPU lease without treating the running process as finished."""
        jid = str(job_id or "")
        self.queue_scheduler.complete(jid)
        self.lease_manager.forget(jid)
        release = self.gpu_store.release_idle(
            job_id=jid,
            elapsed_sec=float(elapsed_sec or 0.0),
            reason=str(reason or "idle_gpu_lease"),
            resident_mem_gb=max(0.0, float(resident_mem_gb or 0.0)),
        )
        result = {
            "released": bool(release.get("released")),
            "elapsed_sec": float(elapsed_sec or 0.0),
            "reason": str(reason or "idle_gpu_lease"),
            "release": release,
        }
        if release.get("released"):
            generation = self.pressure_store.bump_generation(reason="idle_lease_released")
            result["pressure_generation"] = generation
            self.record_resource_event(
                "resource_idle_lease_released",
                payload={"job_id": jid, "resource_type": "gpu", "result": result, "idle_release_admission_mode": self.gpu_store.idle_release_admission_mode},
                command_id=jid,
                lease_id=jid,
            )
            self.record_resource_event(
                "resource_release",
                payload={
                    "job_id": jid,
                    "resource_type": "gpu",
                    "status": "idle_lease_released",
                    "reason": str(reason or "idle_gpu_lease"),
                    "release": release,
                },
                command_id=jid,
                lease_id=jid,
            )
        else:
            self.sync_resource_state(reason="idle_lease_release_not_found")
        return result


    def release(self, *, job_id: str, elapsed_sec: float = 0.0, status: str = "finished") -> dict[str, Any]:
        self.queue_scheduler.complete(str(job_id or ""))
        release = self.lease_manager.release(str(job_id or ""))
        if release.get("released"):
            generation = self.pressure_store.bump_generation(reason="lease_released")
            release["pressure_generation"] = generation
            lease = release.get("lease") if isinstance(release.get("lease"), dict) else {}
            meta = lease.get("metadata") if isinstance(lease.get("metadata"), dict) else {}
            resource_class = str(meta.get("policy_resource_class") or meta.get("resource_class") or "")
            history = self.history_store.record_completion(
                job_id=str(job_id or ""),
                worker_id=self.worker_id,
                resource_class=resource_class,
                gpu_ids=[str(x) for x in (lease.get("gpu_ids") or []) if str(x).strip()],
                elapsed_sec=float(elapsed_sec or 0.0),
                status=str(status or "finished"),
                command_digest=str(meta.get("command_digest") or ""),
                entrypoint=str(meta.get("entrypoint") or ""),
            )
            release["runtime_history"] = history
            released_gpu_ids = [str(x) for x in (lease.get("gpu_ids") or []) if str(x).strip()]
            if released_gpu_ids:
                release["pressure_reconcile"] = self.reconcile_free_gpu_pressure(
                    gpu_ids=released_gpu_ids,
                    reason="lease_released_observed_free",
                )
        if release.get("released"):
            self.record_resource_event(
                "resource_release",
                payload={"job_id": str(job_id or ""), "resource_type": "gpu", "status": str(status or "finished"), "release": release},
                command_id=str(job_id or ""),
                lease_id=str(job_id or ""),
            )
        else:
            self.sync_resource_state(reason="resource_release_not_found")
        return release


    def env_updates(self, *, job_id: str) -> dict[str, str]:
        ids = self.lease_manager.assigned_gpu_ids.get(str(job_id or "")) or []
        if self._assignment_mode() != "lease" or not ids:
            return {}
        pool = self._configured_gpu_pool() or ids
        return {
            "CUDA_VISIBLE_DEVICES": ",".join(ids),
            "SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL": ",".join(ids),
            "SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL": ",".join(str(i) for i, _ in enumerate(ids)),
            "SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL": ",".join(pool),
        }


    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if out == out else None
