# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource runtime responsibility: state synchronization, policy, capacity, and admission opportunity facts."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    Any,
    GPULeaseStore,
    GPUPressureStateStore,
    GPUQueueConfig,
    LeaseManager,
    Path,
    QueueScheduler,
    RESOURCE_GPU_FEATURE_EXTRACT,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ResourceHistoryStore,
    UnifiedResourceStore,
    _GPUResourcePolicy,
    calculate_admission_priority,
    evaluate_admission_safety,
    plan_gpu_sublease,
    sample_nvidia_smi,
)


class StatePolicyRuntime:
    """Own this responsibility's state transitions and callbacks."""

    def __init__(
        self,
        *,
        worker_id: str,
        resource_dir: Path,
        gpu_queue: GPUQueueConfig,
    ) -> None:
        self.worker_id = str(worker_id or "W00")
        self.resource_dir = Path(resource_dir)
        self.gpu_queue = gpu_queue
        self.gpu_store = GPULeaseStore(
            self.resource_dir / "gpu_leases.json",
            lease_ttl_sec=self.gpu_queue.lease_ttl_sec,
            admission_queue_enabled=bool(self.gpu_queue.admission_queue_enabled),
            waiter_ttl_sec=float(self.gpu_queue.admission_waiter_ttl_sec or 900.0),
            idle_release_admission_mode=str(self.gpu_queue.idle_release_admission_mode or "strict_exclusive"),
        )
        self.lease_manager = LeaseManager(self.gpu_store)
        self.queue_scheduler = QueueScheduler(
            heartbeat_sec=float(self.gpu_queue.heartbeat_sec or 15.0)
        )
        self.queue_scheduler.recover(self.gpu_store.snapshot_active())
        self.pressure_store = GPUPressureStateStore(
            self.resource_dir / "gpu_pressure.json",
            default_cooldown_sec=max(120.0, float(self.gpu_queue.max_wait_sec or 0.0)),
            duplicate_digest_cooldown_sec=float(self.gpu_queue.duplicate_digest_cooldown_sec or 600.0),
            duplicate_digest_threshold=int(self.gpu_queue.duplicate_digest_threshold or 2),
            yellow_hold_sec=float(self.gpu_queue.pressure_yellow_hold_sec or 120.0),
            red_to_yellow_sec=float(self.gpu_queue.pressure_red_to_yellow_sec or 120.0),
        )
        self.history_store = ResourceHistoryStore(self.resource_dir / "gpu_runtime_history.json")
        self.unified_store = UnifiedResourceStore(self.resource_dir)
        # Temporary compatibility projections. The component objects above own
        # these collections; legacy call sites are migrated one operation at a time.
        self._queue_started = self.queue_scheduler.pending_job_ids
        self._last_heartbeat = self.queue_scheduler.last_heartbeat
        self._leased_gpu_ids = self.lease_manager.assigned_gpu_ids
        self._admission_backoff_cache: dict[str, dict[str, Any]] = {}
        self._resource_wait_seq = 0
        self._resource_wait_tokens: dict[str, dict[str, Any]] = {}
        self._resource_wait_suppressed_until: dict[str, float] = {}


    def sync_resource_state(self, *, reason: str = "sync") -> dict[str, Any]:
        try:
            leases = self.gpu_store.snapshot_active()
            pressure = self.pressure_store.snapshot()
            history = self.history_store.snapshot()
            return self.unified_store.sync_legacy_gpu_state(
                leases=leases,
                pressure=pressure,
                runtime_history=history,
                reason=str(reason or "sync"),
            )
        except Exception as exc:
            return {"synced": False, "reason": "resource_state_sync_failed", "error": str(exc)}


    def record_resource_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        trace_id: str = "",
        proposal_id: str = "",
        decision_id: str = "",
        task_id: str = "",
        command_id: str = "",
        lease_id: str = "",
    ) -> dict[str, Any]:
        self.sync_resource_state(reason=f"event:{event_type}")
        try:
            return self.unified_store.append_event(
                event_type,
                payload=payload,
                trace_id=trace_id,
                proposal_id=proposal_id,
                decision_id=decision_id,
                task_id=task_id,
                worker_id=self.worker_id,
                command_id=command_id,
                lease_id=lease_id,
            )
        except Exception as exc:
            return {"recorded": False, "reason": "resource_event_append_failed", "error": str(exc)}


    def update_active_resource_proposal(self, proposal_id: str, proposal: dict[str, Any] | None) -> dict[str, Any]:
        try:
            return self.unified_store.update_active_proposal(proposal_id, proposal)
        except Exception as exc:
            return {"updated": False, "reason": "active_proposal_update_failed", "error": str(exc)}


    def clear_active_resource_proposals_for_command(
        self,
        command_id: str,
        *,
        proposal_type: str = "",
        reason_code: str = "",
    ) -> dict[str, Any]:
        try:
            return self.unified_store.clear_active_proposals_for_command(
                command_id,
                proposal_type=proposal_type,
                reason_code=reason_code,
            )
        except Exception as exc:
            return {"cleared": False, "reason": "active_proposal_clear_failed", "error": str(exc)}


    def _capacity_slots(self) -> float:
        try:
            capacity = float(self.gpu_queue.capacity_slots)
        except (TypeError, ValueError):
            capacity = 1.0
        return capacity if capacity > 0 else 1.0


    def _policy_for_resource_class(self, resource_class: str) -> _GPUResourcePolicy | None:
        cls = str(resource_class or "").strip()
        if cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN}:
            return _GPUResourcePolicy(
                resource_class=RESOURCE_HEAVY_GPU_TRAIN,
                slot_weight=1.0,
                max_per_gpu=max(1, int(self.gpu_queue.max_heavy_per_gpu or 1)),
                incompatible_classes=(
                    RESOURCE_GPU_TT_LIGHT,
                    RESOURCE_GPU_FEATURE_EXTRACT,
                    RESOURCE_GPU_LIGHT_TRAIN,
                    RESOURCE_UNKNOWN_GPU_EXEC,
                ),
            )
        if cls == RESOURCE_GPU_TT_LIGHT:
            incompatible = [RESOURCE_UNKNOWN_GPU_EXEC]
            if not bool(self.gpu_queue.share_tt_with_train):
                incompatible.extend([RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE])
            return _GPUResourcePolicy(
                resource_class=RESOURCE_GPU_TT_LIGHT,
                slot_weight=0.25,
                max_per_gpu=max(1, int(self.gpu_queue.gpu_tt_max_per_gpu or 1)),
                incompatible_classes=tuple(incompatible),
            )
        if cls == RESOURCE_GPU_FEATURE_EXTRACT:
            return _GPUResourcePolicy(
                resource_class=RESOURCE_GPU_FEATURE_EXTRACT,
                slot_weight=0.5,
                max_per_gpu=max(1, int(self.gpu_queue.gpu_feature_max_per_gpu or 1)),
                incompatible_classes=(
                    RESOURCE_HEAVY_GPU_TRAIN,
                    RESOURCE_HEAVY_GPU_CANDIDATE,
                    RESOURCE_UNKNOWN_GPU_EXEC,
                ),
            )
        if cls == RESOURCE_GPU_LIGHT_TRAIN:
            incompatible = [RESOURCE_UNKNOWN_GPU_EXEC]
            if not bool(self.gpu_queue.share_tt_with_train):
                incompatible.extend([RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE])
            return _GPUResourcePolicy(
                resource_class=RESOURCE_GPU_LIGHT_TRAIN,
                slot_weight=0.5,
                max_per_gpu=1,
                incompatible_classes=tuple(incompatible),
            )
        if cls == RESOURCE_UNKNOWN_GPU_EXEC:
            return _GPUResourcePolicy(
                resource_class=RESOURCE_UNKNOWN_GPU_EXEC,
                slot_weight=1.0,
                max_per_gpu=1,
                incompatible_classes=(
                    RESOURCE_HEAVY_GPU_TRAIN,
                    RESOURCE_HEAVY_GPU_CANDIDATE,
                    RESOURCE_GPU_TT_LIGHT,
                    RESOURCE_GPU_FEATURE_EXTRACT,
                    RESOURCE_GPU_LIGHT_TRAIN,
                ),
            )
        return None


    def should_queue_gpu(self, *, resource_class: str, gpu_ids: list[str]) -> bool:
        if not self.gpu_queue.enabled:
            return False
        if self._policy_for_resource_class(resource_class) is None:
            return False
        if [x for x in (gpu_ids or []) if str(x).strip()]:
            return True
        return self._assignment_mode() == "lease" and bool(self._configured_gpu_pool())


    def _assignment_mode(self) -> str:
        mode = str(self.gpu_queue.assignment or "env_only").strip().lower()
        return mode if mode in {"env_only", "lease"} else "env_only"


    def lease_assignment_enabled(self) -> bool:
        return self._assignment_mode() == "lease"


    def _configured_gpu_pool(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in self.gpu_queue.gpu_pool or []:
            gpu_id = str(raw).strip()
            if gpu_id and gpu_id not in seen:
                seen.add(gpu_id)
                out.append(gpu_id)
        return out


    def _pressure_gpu_ids(self, gpu_ids: list[str]) -> list[str]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if ids:
            return ids
        if self._assignment_mode() == "lease":
            return self._configured_gpu_pool()
        return ids


    def _request_count(self, available_count: int, *, requested: int | None = None) -> int:
        max_request = max(1, int(self.gpu_queue.max_request or 1))
        default_request = max(1, int(self.gpu_queue.default_request or 1))
        desired = max(1, int(requested or default_request))
        return max(1, min(desired, max_request, max(1, int(available_count or 1))))


    def _gpu_sublease_plan(
        self,
        *,
        candidate_gpu_ids: list[str],
        request_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = metadata if isinstance(metadata, dict) else {}
        plan = plan_gpu_sublease(
            candidate_gpu_ids=candidate_gpu_ids,
            requested_gpu_count=request_count,
            default_request=int(self.gpu_queue.default_request or 1),
            max_request=int(self.gpu_queue.max_request or 1),
            command=str(meta.get("command") or meta.get("command_excerpt") or ""),
            metadata=meta,
        )
        return plan.to_json()


    def _estimated_gpu_footprint(self, resource_class: str) -> dict[str, Any]:
        cls = str(resource_class or "").strip()
        estimates = {
            RESOURCE_GPU_TT_LIGHT: (6.0, "medium"),
            RESOURCE_GPU_FEATURE_EXTRACT: (12.0, "medium"),
            RESOURCE_GPU_LIGHT_TRAIN: (18.0, "low"),
            RESOURCE_HEAVY_GPU_CANDIDATE: (28.0, "low"),
            RESOURCE_HEAVY_GPU_TRAIN: (28.0, "low"),
            RESOURCE_UNKNOWN_GPU_EXEC: (32.0, "low"),
        }
        peak, confidence = estimates.get(cls, (24.0, "low"))
        return {
            "estimated_peak_mem_gb": float(peak),
            "confidence": confidence,
            "source": "resource_class_default",
        }


    @staticmethod
    def _sample_rows_by_gpu(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if gpu_id:
                out[gpu_id] = row
        return out


    @staticmethod
    def _row_free_mem_gb(row: dict[str, Any]) -> float | None:
        try:
            used = float(row.get("memory_used_mb"))
            total = float(row.get("memory_total_mb"))
        except (TypeError, ValueError):
            return None
        if total <= 0:
            return None
        return max(0.0, (total - used) / 1024.0)


    def admission_opportunity_facts(
        self,
        *,
        resource_class: str,
        gpu_ids: list[str],
        request_count: int | None = None,
        selected_gpu_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        policy = self._policy_for_resource_class(resource_class)
        if policy is None or self._assignment_mode() != "lease":
            return {"enabled": False, "reason": "lease_admission_not_applicable"}
        raw_candidates = selected_gpu_ids or gpu_ids or self._configured_gpu_pool()
        candidates = [str(x) for x in (raw_candidates or []) if str(x).strip()]
        if not candidates:
            return {"enabled": False, "reason": "no_candidate_gpu_ids"}
        requested = self._request_count(len(candidates), requested=request_count)
        sample = sample_nvidia_smi(candidates)
        rows = self._sample_rows_by_gpu(sample if isinstance(sample, dict) else {})
        availability = self.gpu_store.availability(
            candidate_gpu_ids=candidates,
            request_count=requested,
            max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
            resource_class=policy.resource_class,
            slot_weight=policy.slot_weight,
            capacity_slots=self._capacity_slots(),
            max_per_gpu=policy.max_per_gpu,
            incompatible_classes=list(policy.incompatible_classes),
        )
        internally_available = {str(x) for x in (availability.get("available_gpu_ids") or []) if str(x).strip()}
        footprint = self._estimated_gpu_footprint(policy.resource_class)
        try:
            estimated_peak = float(footprint.get("estimated_peak_mem_gb") or 0.0)
        except (TypeError, ValueError):
            estimated_peak = 0.0
        reserve = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        buffer = max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0))
        hard_threshold = reserve + buffer if reserve > 0 else 0.0
        footprint_threshold = estimated_peak + max(2.0, min(8.0, buffer if buffer > 0 else 4.0)) if estimated_peak > 0 else hard_threshold
        rows_out: list[dict[str, Any]] = []
        grantable: list[str] = []
        partial: list[dict[str, Any]] = []
        for gpu_id in candidates:
            row = rows.get(gpu_id, {})
            free_gb = self._row_free_mem_gb(row) if row else None
            try:
                util = float(row.get("utilization_gpu_pct") or 0.0) if row else None
            except (TypeError, ValueError):
                util = None
            internal_ok = gpu_id in internally_available
            hard_mem_ok = free_gb is None or hard_threshold <= 0 or free_gb >= hard_threshold
            footprint_ok = free_gb is None or footprint_threshold <= 0 or free_gb >= footprint_threshold
            candidate = {
                "gpu_id": gpu_id,
                "internal_lease_available": bool(internal_ok),
                "free_mem_gb": (round(float(free_gb), 3) if free_gb is not None else None),
                "utilization_gpu_pct": (round(float(util), 3) if util is not None else None),
                "hard_mem_ok": bool(hard_mem_ok),
                "footprint_ok": bool(footprint_ok),
            }
            rows_out.append(candidate)
            if internal_ok and hard_mem_ok:
                grantable.append(gpu_id)
                if free_gb is not None and free_gb >= max(hard_threshold, min(16.0, footprint_threshold)):
                    partial.append(candidate)
        return {
            "enabled": True,
            "lease_grantable_by_llm": bool(grantable),
            "full_request_grantable": len(grantable) >= requested,
            "grantable_gpu_ids": grantable,
            "partial_gpu_candidates": partial,
            "observed_gpus": rows_out,
            "requested_gpu_count": requested,
            "footprint": footprint,
            "free_mem_hard_threshold_gb": hard_threshold,
            "estimated_footprint_threshold_gb": round(float(footprint_threshold), 3),
            "availability": {
                "available": bool(availability.get("available")),
                "available_gpu_ids": list(availability.get("available_gpu_ids") or []),
                "requested_count": availability.get("requested_count"),
            },
            "sample_available": bool(isinstance(sample, dict) and sample.get("available") is not False),
        }


    def grant_llm_admission_lease(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        request_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        observe_then_run: bool = False,
        observe_sec: float = 0.0,
    ) -> dict[str, Any]:
        policy = self._policy_for_resource_class(resource_class)
        if policy is None:
            return {"enabled": False, "acquired": False, "reason": "resource_class_not_queueable"}
        if self._assignment_mode() != "lease":
            return {"enabled": False, "acquired": False, "reason": "lease_assignment_disabled"}
        candidates = [str(x) for x in (gpu_ids or self._configured_gpu_pool()) if str(x).strip()]
        if not candidates:
            return {"enabled": True, "acquired": False, "status": "PENDING", "reason": "no_candidate_gpu_ids"}
        sublease = self._gpu_sublease_plan(
            candidate_gpu_ids=candidates,
            request_count=request_count,
            metadata=metadata,
        )
        requested = int(sublease.get("requested_gpu_count") or 1)
        min_free = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        if min_free > 0:
            sample = sample_nvidia_smi(candidates)
            safety = evaluate_admission_safety(
                sample=sample if isinstance(sample, dict) else {},
                gpu_ids=candidates,
                resource_class=policy.resource_class,
                request_count=requested,
                assignment="lease",
                min_free_mem_gb=min_free,
                free_mem_buffer_gb=max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0)),
            )
            if safety.get("blocked"):
                return {
                    "enabled": True,
                    "acquired": False,
                    "status": "PENDING",
                    "reason": str(safety.get("reason") or "gpu_memory_below_admission_reserve"),
                    "safety": safety,
                    "candidate_physical_gpus": candidates,
                    "requested_gpu_count": requested,
                    "gpu_sublease": dict(sublease),
                }
            safe_ids = [str(x) for x in (safety.get("safe_gpu_ids") or candidates) if str(x).strip()]
            if safe_ids:
                candidates = safe_ids
                requested = self._request_count(len(candidates), requested=requested)
                sublease = {**sublease, "candidate_gpu_ids": list(candidates), "requested_gpu_count": requested}
        meta = {
            **dict(metadata or {}),
            **GPULeaseStore.current_owner_metadata(),
            "assignment": "lease",
            "resource_class": str(resource_class or ""),
            "policy_resource_class": policy.resource_class,
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "max_per_gpu": policy.max_per_gpu,
            "incompatible_classes": list(policy.incompatible_classes),
            "gpu_sublease": dict(sublease),
            "llm_admission_grant": True,
            "observe_then_run": bool(observe_then_run),
            "observe_sec": max(0.0, float(observe_sec or 0.0)),
        }
        acquired, details = self.gpu_store.try_acquire_any(
            job_id=str(job_id or ""),
            worker_id=self.worker_id,
            candidate_gpu_ids=candidates,
            request_count=requested,
            max_heavy_per_gpu=self.gpu_queue.max_heavy_per_gpu,
            metadata=meta,
            resource_class=policy.resource_class,
            slot_weight=policy.slot_weight,
            capacity_slots=self._capacity_slots(),
            max_per_gpu=policy.max_per_gpu,
            incompatible_classes=list(policy.incompatible_classes),
        )
        lease = details.get("lease") if isinstance(details.get("lease"), dict) else {}
        assigned = [str(x) for x in (lease.get("gpu_ids") or details.get("assigned_gpu_ids") or []) if str(x).strip()]
        env_updates = {}
        if acquired and assigned:
            self.lease_manager.remember(str(job_id or ""), assigned)
            env_updates = {
                "CUDA_VISIBLE_DEVICES": ",".join(assigned),
                "SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL": ",".join(assigned),
                "SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL": ",".join(str(i) for i, _ in enumerate(assigned)),
            }
            pool = self._configured_gpu_pool()
            if pool:
                env_updates["SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL"] = ",".join(pool)
        result = {
            "enabled": True,
            "acquired": bool(acquired),
            "status": "GRANTED" if acquired else "PENDING",
            "admission_action": "RUN_NOW" if acquired else "PENDING",
            "reason": str(details.get("reason") or ("llm_admission_lease_granted" if acquired else "llm_admission_lease_unavailable")),
            "gpu_ids": assigned or candidates,
            "assigned_physical_gpus": assigned,
            "candidate_physical_gpus": candidates,
            "allowed_physical_gpus": self._configured_gpu_pool() or candidates,
            "requested_gpu_count": requested,
            "gpu_sublease": dict(sublease),
            "env_updates": env_updates,
            "details": details,
            "policy_resource_class": policy.resource_class,
            "resource_class": str(resource_class or ""),
            "slot_weight": policy.slot_weight,
            "capacity_slots": self._capacity_slots(),
            "admission_llm_lease_attempted": True,
            "admission_observe_then_run": bool(observe_then_run and acquired),
            "admission_observe_sec": max(0.0, float(observe_sec or 0.0)),
        }
        self.record_resource_event(
            "admission_llm_lease_grant" if acquired else "admission_llm_lease_grant_failed",
            payload={"job_id": str(job_id or ""), "resource_type": "gpu", "result": result},
            command_id=str(job_id or ""),
            lease_id=str(job_id or ""),
        )
        return result


    def _cold_eta_sec(self) -> float:
        return max(300.0, min(float(self.gpu_queue.max_wait_sec or 0.0) + 300.0, 1800.0))


    def _admission_priority(self, value_hint: dict[str, Any]) -> dict[str, float]:
        return calculate_admission_priority(value_hint)
