# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource runtime responsibility: yellow safety holds, reservation pressure, and reconciliation."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    Any,
    RESOURCE_GPU_LIGHT_TRAIN,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_UNKNOWN_GPU_EXEC,
    _GPU_PRESSURE_SUPPORT_CLASSES,
    evaluate_yellow_safety_hold,
    sample_nvidia_smi,
)


class PressureRuntime:
    """Own this responsibility's state transitions and callbacks."""

    def _yellow_safety_hold(self, *, gpu_ids: list[str]) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        if not ids:
            return {"blocked": False, "gpu_ids": []}
        sample = sample_nvidia_smi(ids)
        reserve = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        return evaluate_yellow_safety_hold(
            sample=sample if isinstance(sample, dict) else {},
            gpu_ids=ids,
            util_exit_threshold_pct=50.0,
            min_free_mem_gb=reserve,
        )


    @staticmethod
    def _lease_record_gpu_ids(raw: dict[str, Any]) -> set[str]:
        return {str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()} if isinstance(raw, dict) else set()


    def _active_gpu_reservations(self, *, gpu_ids: list[str]) -> dict[str, Any]:
        ids = {str(x) for x in (gpu_ids or []) if str(x).strip()}
        if not ids:
            return {"active": False, "gpu_ids": []}
        try:
            snapshot = self.gpu_store.snapshot_active()
        except Exception as exc:
            return {"active": True, "reason": "lease_snapshot_failed", "error": str(exc), "gpu_ids": sorted(ids)}
        active: list[str] = []
        released_idle: list[str] = []
        for job_id, raw in (snapshot.get("leases") or {}).items():
            if ids & self._lease_record_gpu_ids(raw if isinstance(raw, dict) else {}):
                active.append(str(job_id))
        for job_id, raw in (snapshot.get("released_idle") or {}).items():
            if ids & self._lease_record_gpu_ids(raw if isinstance(raw, dict) else {}):
                released_idle.append(str(job_id))
        return {
            "active": bool(active or released_idle),
            "gpu_ids": sorted(ids),
            "active_lease_job_ids": active,
            "released_idle_job_ids": released_idle,
        }


    def reconcile_free_gpu_pressure(
        self,
        *,
        gpu_ids: list[str],
        reason: str = "observed_free_no_active_lease",
        sample: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        if not ids:
            return {"cleared": False, "reason": "no_gpu_ids", "gpu_ids": []}
        reservations = self._active_gpu_reservations(gpu_ids=ids)
        if reservations.get("active"):
            return {"cleared": False, "reason": "active_gpu_reservation_present", "gpu_ids": ids, "reservations": reservations}
        observed = sample if isinstance(sample, dict) else sample_nvidia_smi(ids)
        if not isinstance(observed, dict) or observed.get("available") is False:
            return {"cleared": False, "reason": "gpu_sample_unavailable", "gpu_ids": ids, "observed": observed or {}}
        rows = observed.get("gpus") if isinstance(observed.get("gpus"), list) else []
        by_id = {str(row.get("gpu_id") or row.get("index") or ""): row for row in rows if isinstance(row, dict)}
        missing = [gpu_id for gpu_id in ids if gpu_id not in by_id]
        if missing:
            return {"cleared": False, "reason": "gpu_sample_missing_ids", "gpu_ids": ids, "missing_gpu_ids": missing, "observed": observed}
        busy: list[dict[str, Any]] = []
        for gpu_id in ids:
            row = by_id[gpu_id]
            try:
                util = float(row.get("utilization_gpu_pct") or 0.0)
            except (TypeError, ValueError):
                util = 0.0
            try:
                used_mb = float(row.get("memory_used_mb") or 0.0)
            except (TypeError, ValueError):
                used_mb = 0.0
            if util > 1.0 or used_mb > 1024.0:
                busy.append({"gpu_id": gpu_id, "utilization_gpu_pct": util, "memory_used_mb": used_mb})
        if busy:
            return {"cleared": False, "reason": "gpu_not_observed_free", "gpu_ids": ids, "busy_gpus": busy, "observed": observed}
        cleared = self.pressure_store.clear_global_pressure(
            gpu_ids=ids,
            reason=str(reason or "observed_free_no_active_lease"),
            observed={"sample": observed, "reservations": reservations},
        )
        if cleared.get("cleared"):
            self.record_resource_event(
                "gpu_pressure_reconciled_free",
                payload={"gpu_ids": ids, "reason": str(reason or "observed_free_no_active_lease"), "result": cleared},
            )
        return cleared


    def pressure_gate_decision(self, *, resource_class: str, gpu_ids: list[str], command_digest: str = "") -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        if not ids:
            return {"blocked": False, "gpu_ids": [], "pressure": {}}
        duplicate = self.pressure_store.duplicate_gate_decision(gpu_ids=ids, command_digest=str(command_digest or ""))
        if isinstance(duplicate, dict) and duplicate.get("blocked"):
            return {
                **duplicate,
                "allowed_after_gate": "changed command or plan",
                "allowed_classes": ["pure_tt_cpu", "readonly_cpu", "light_cpu"],
                "resource_mode": "DUPLICATE_COOLDOWN",
            }
        reconcile = self.reconcile_free_gpu_pressure(gpu_ids=ids, reason="observed_free_before_pressure_gate")
        snapshot = self.pressure_store.snapshot(gpu_ids=ids)
        if reconcile.get("cleared"):
            snapshot = self.pressure_store.snapshot(gpu_ids=ids)
        red = {
            gpu_id: entry
            for gpu_id, entry in (snapshot.get("gpus") or {}).items()
            if isinstance(entry, dict) and entry.get("mode") == "RED"
        }
        cls = str(resource_class or "")
        if not red:
            yellow = {
                gpu_id: entry
                for gpu_id, entry in (snapshot.get("gpus") or {}).items()
                if isinstance(entry, dict) and entry.get("mode") == "YELLOW"
            }
            if yellow and cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}:
                safety = self._yellow_safety_hold(gpu_ids=sorted(yellow))
                if safety.get("blocked"):
                    return {
                        "blocked": True,
                        "status": "PENDING",
                        "reason": "yellow_pressure_exit_guard",
                        "allowed_after_gate": "support work until GPU leaves yellow safety hold",
                        "allowed_classes": _GPU_PRESSURE_SUPPORT_CLASSES,
                        "gpu_ids": ids,
                        "red_gpu_ids": sorted(safety.get("blocked_gpu_ids") or sorted(yellow)),
                        "max_queue_timeout_count": max(int((entry or {}).get("queue_timeout_count") or 0) for entry in yellow.values()),
                        "cooldown_remaining_sec": max(float((entry or {}).get("yellow_remaining_sec") or 0.0) for entry in yellow.values()),
                        "eta_next_train_sec": max(300.0, max(float((entry or {}).get("yellow_remaining_sec") or 0.0) for entry in yellow.values())),
                        "eta_confidence": "low",
                        "resource_mode": "YELLOW",
                        "pressure": snapshot,
                        "safety": safety,
                    }
            return {"blocked": False, "gpu_ids": ids, "pressure": snapshot}
        counts = [int((entry or {}).get("queue_timeout_count") or 0) for entry in red.values()]
        max_count = max(counts or [0])
        if max_count >= 2:
            blocked = cls not in set(_GPU_PRESSURE_SUPPORT_CLASSES)
            reason = "tt_only_after_gpu_pressure"
            allowed = "pure_tt_cpu,gpu_tt_light,heavy_cpu_candidate,readonly_cpu,light_cpu"
            allowed_classes = _GPU_PRESSURE_SUPPORT_CLASSES
        else:
            blocked = cls in {RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_GPU_LIGHT_TRAIN, RESOURCE_UNKNOWN_GPU_EXEC}
            reason = "block_train_after_gpu_pressure"
            allowed = "non-training work"
            allowed_classes = _GPU_PRESSURE_SUPPORT_CLASSES
        cooldown = max(float((entry or {}).get("cooldown_remaining_sec") or 0.0) for entry in red.values())
        return {
            "blocked": bool(blocked),
            "status": "DENIED_REPLAN",
            "reason": reason,
            "allowed_after_gate": allowed,
            "allowed_classes": allowed_classes,
            "gpu_ids": ids,
            "red_gpu_ids": sorted(red),
            "max_queue_timeout_count": max_count,
            "cooldown_remaining_sec": cooldown,
            "eta_next_train_sec": cooldown + 300.0,
            "eta_confidence": "low",
            "resource_mode": "RED",
            "pressure": snapshot,
        }


    def record_runtime_pressure(
        self,
        *,
        job_id: str,
        resource_class: str,
        gpu_ids: list[str],
        elapsed_sec: float,
        reason: str,
        command_digest: str = "",
    ) -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        return self.pressure_store.record_runtime_pressure(
            gpu_ids=ids,
            worker_id=self.worker_id,
            job_id=str(job_id or ""),
            resource_class=str(resource_class or ""),
            elapsed_sec=float(elapsed_sec or 0.0),
            reason=str(reason or "runtime_pressure"),
            command_digest=str(command_digest or ""),
        )


    def pressure_generation(self) -> int:
        snapshot = self.pressure_store.snapshot()
        return int(snapshot.get("generation") or 0)


    def pressure_snapshot(self, *, gpu_ids: list[str] | None = None) -> dict[str, Any]:
        return self.pressure_store.snapshot(gpu_ids=gpu_ids)


    def has_active_lease(self, *, job_id: str) -> bool:
        return self.lease_manager.has_active(job_id)


    def persisted_lease(self, *, job_id: str) -> dict[str, Any]:
        if not job_id:
            return {}
        try:
            return self.lease_manager.persisted(job_id)
        except Exception:
            return {}
