# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""GPU process placement, utilization sampling, contention, and efficiency facts."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (Any, ResourceJob, assess_resource_efficiency, time)


class GpuSampling:
    """Own GpuSampling resource behavior without delegated forwarding."""

    @staticmethod
    def _process_gpu_placement_has_usage(placement: dict[str, Any]) -> bool:
        if not isinstance(placement, dict) or placement.get("available") is not True:
            return False
        used = placement.get("used_gpu_processes")
        return bool(isinstance(used, list) and used)


    @staticmethod
    def _process_gpu_placement_mem_mb(placement: dict[str, Any]) -> float:
        if not isinstance(placement, dict) or placement.get("available") is not True:
            return 0.0
        total = 0.0
        for row in placement.get("used_gpu_processes") or []:
            if not isinstance(row, dict):
                continue
            try:
                total += max(0.0, float(row.get("used_memory_mb") or 0.0))
            except (TypeError, ValueError):
                pass
        return total


    @staticmethod
    def _process_gpu_memory_by_gpu(placement: dict[str, Any]) -> dict[str, float]:
        if not isinstance(placement, dict) or placement.get("available") is not True:
            return {}
        out: dict[str, float] = {}
        for row in placement.get("used_gpu_processes") or []:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or "").strip()
            if not gpu_id:
                continue
            try:
                used_mb = max(0.0, float(row.get("used_memory_mb") or 0.0))
            except (TypeError, ValueError):
                used_mb = 0.0
            out[gpu_id] = out.get(gpu_id, 0.0) + used_mb
        return out


    def _job_process_gpu_placement(self, job: ResourceJob) -> dict[str, Any]:
        if not job.pid:
            return {"available": False, "reason": "missing_pid", "violations": []}
        try:
            # Resolve through the public controller so instrumentation keeps its
            # stable patch point after the observer implementation moved here.
            from scienceflow.research.solver.lnr.resources.runtime.observer import controller

            return controller.process_tree_gpu_placement_snapshot(job.pid, job.gpu_ids)
        except Exception as exc:
            return {"available": False, "reason": type(exc).__name__, "violations": []}


    def _gpu_util_sample_is_idle(self, sample: dict[str, Any], gpu_ids: list[str]) -> bool:
        if not isinstance(sample, dict) or sample.get("available") is False:
            return False
        wanted = {str(x).strip() for x in (gpu_ids or []) if str(x).strip()}
        if not wanted:
            return False
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if gpu_id not in wanted:
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            used_mb = self._float_or_none(row.get("memory_used_mb"))
            if util is None or used_mb is None:
                return False
            used_gb = max(0.0, used_mb / 1024.0)
            if util > self.gpu_idle_lease_util_pct or used_gb > self.gpu_idle_lease_mem_gb:
                return False
            seen.add(gpu_id)
        return seen == wanted


    def _gpu_util_sample_is_low_compute_with_model(self, sample: dict[str, Any], gpu_ids: list[str]) -> bool:
        if not isinstance(sample, dict) or sample.get("available") is False:
            return False
        wanted = {str(x).strip() for x in (gpu_ids or []) if str(x).strip()}
        if not wanted:
            return False
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if gpu_id not in wanted:
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            used_mb = self._float_or_none(row.get("memory_used_mb"))
            if util is None or used_mb is None:
                return False
            used_gb = max(0.0, used_mb / 1024.0)
            if util > self.gpu_dataloader_bottleneck_util_pct or used_gb < self.gpu_dataloader_bottleneck_min_mem_gb:
                return False
            seen.add(gpu_id)
        return seen == wanted


    def _gpu_pressure_active_for_job(self, job: ResourceJob) -> dict[str, Any]:
        if not self.gpu_idle_lease_require_pressure:
            return {"active": True, "reason": "pressure_not_required"}
        if self.resource_runtime is None or not job.gpu_ids:
            return {"active": False, "reason": "pressure_unavailable"}
        try:
            snapshot = self.resource_runtime.pressure_snapshot(gpu_ids=job.gpu_ids)
        except Exception:
            return {"active": False, "reason": "pressure_snapshot_failed"}
        gpus = snapshot.get("gpus") if isinstance(snapshot.get("gpus"), dict) else {}
        for gpu_id, raw in gpus.items():
            if not isinstance(raw, dict):
                continue
            mode = str(raw.get("mode") or "").upper()
            queue_len = int(raw.get("queue_len") or 0)
            yellow_remaining = float(raw.get("yellow_remaining_sec") or 0.0)
            cooldown_remaining = float(raw.get("cooldown_remaining_sec") or 0.0)
            if mode in {"YELLOW", "RED"} or queue_len > 0 or yellow_remaining > 0 or cooldown_remaining > 0:
                if mode in {"YELLOW", "RED"}:
                    reason = mode.lower()
                elif cooldown_remaining > 0:
                    reason = "red_cooldown"
                elif yellow_remaining > 0:
                    reason = "yellow_hold"
                else:
                    reason = "queue_pressure"
                return {
                    "active": True,
                    "reason": reason,
                    "gpu_id": str(gpu_id),
                    "snapshot": snapshot,
                }
        return {"active": False, "reason": "no_gpu_pressure", "snapshot": snapshot}


    def _last_gpu_sample_active(self, job: ResourceJob) -> bool:
        sample_bundle = job.last_gpu_util_sample if isinstance(job.last_gpu_util_sample, dict) else {}
        placement = sample_bundle.get("process_gpu_placement") if isinstance(sample_bundle.get("process_gpu_placement"), dict) else {}
        if placement.get("available") is True and not self._process_gpu_placement_has_usage(placement):
            return False
        sample = sample_bundle.get("sample") if isinstance(sample_bundle.get("sample"), dict) else {}
        rows = sample.get("gpus") if isinstance(sample, dict) and isinstance(sample.get("gpus"), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            if util is not None and util > max(1.0, float(self.gpu_idle_lease_util_pct or 0.0)):
                return True
        return False


    def _contention_context_for_job(self, job: ResourceJob) -> dict[str, Any]:
        fallback = max(
            self.review_heartbeat_sec,
            float(self.review_config.progress_event_min_windows or 1) * self.review_heartbeat_sec,
        )
        empty = {
            "active_waiter_pressure": False,
            "blocked_worker_count": 0,
            "oldest_waiter_age_sec": 0.0,
            "contention_cap_sec": fallback,
        }
        if self.resource_runtime is None:
            return empty
        try:
            if not self.resource_runtime.has_active_lease(job_id=job.job_id):
                return empty
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception:
            return empty
        now = time.time()
        waiters = self._contention_waiter_cards(job, snapshot=snapshot, now=now)
        if not waiters:
            return empty
        oldest = max(float(row.get("queue_age_sec") or 0.0) for row in waiters)
        return {
            "active_waiter_pressure": True,
            "blocked_worker_count": len(waiters),
            "oldest_waiter_age_sec": oldest,
            "contention_cap_sec": fallback,
            "waiters": waiters,
        }


    def _update_resource_efficiency_state(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        finish = self._progress_finish_feasibility(job, signal, elapsed_sec=elapsed_sec)
        progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured = progress_payload.get("structured_progress") if isinstance(progress_payload.get("structured_progress"), dict) else {}
        assessment = assess_resource_efficiency(
            previous_state=job.resource_efficiency_state,
            cpu_set=job.cpu_set,
            process_tree_cpu=signal.get("process_tree_cpu") if isinstance(signal.get("process_tree_cpu"), dict) else {},
            assigned_gpu_count=len(job.gpu_ids or []),
            gpu_expected=self._job_uses_expensive_gpu(job),
            gpu_active=self._last_gpu_sample_active(job),
            gpu_sample_available=bool(job.last_gpu_util_sample),
            runtime_sec=elapsed_sec,
            phase=str(signal.get("current_phase") or progress_payload.get("phase") or "unknown"),
            eta_to_deliverable_sec=self._float_or_none(finish.get("eta_to_deliverable_sec")),
            eta_confidence=str(finish.get("eta_confidence") or "low"),
            remaining_useful_budget_sec=self._float_or_none(finish.get("remaining_useful_budget_sec")),
            phase_completion_protected=bool(finish.get("phase_completion_protected")),
            metric_useful=self._metric_value_useful_from_signal(signal),
            deliverable_validity=str(job.deliverable_validity or "none"),
            progress_source=str(structured.get("source") or "unknown"),
            progress_evidence_trust=str(structured.get("evidence_trust") or "unknown"),
            progress_interval_samples=int(structured.get("interval_sample_count") or 0),
            llm_call_count=int(self._job_llm_call_count.get(job.job_id) or 0),
        )
        job.resource_efficiency_state = dict(assessment)
        signal["resource_efficiency"] = dict(assessment)
        return assessment
