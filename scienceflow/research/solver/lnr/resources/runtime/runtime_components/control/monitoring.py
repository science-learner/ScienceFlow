# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Resource runtime responsibility: utilization sampling and runtime pressure recording."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    Any,
    sample_nvidia_smi,
)


class MonitoringRuntime:
    """Own this responsibility's state transitions and callbacks."""

    def _record_yellow_from_util_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        rows = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
        if not rows:
            return {"recorded": False, "reason": "no_gpu_rows", "gpu_ids": []}
        util_threshold = max(0.0, float(self.gpu_queue.pressure_yellow_util_pct or 0.0))
        min_free = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        free_threshold = min_free + max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0))
        yellow_ids: list[str] = []
        reasons: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if not gpu_id:
                continue
            util = self._float_or_none(row.get("utilization_gpu_pct"))
            used_mb = self._float_or_none(row.get("memory_used_mb"))
            total_mb = self._float_or_none(row.get("memory_total_mb"))
            reason_parts: list[str] = []
            if util is not None and util_threshold > 0 and util >= util_threshold:
                reason_parts.append("util_near_limit")
            if used_mb is not None and total_mb is not None and total_mb > 0 and free_threshold > 0:
                free_gb = max(0.0, (total_mb - used_mb) / 1024.0)
                if free_gb < free_threshold:
                    reason_parts.append("free_mem_near_reserve")
            if reason_parts:
                yellow_ids.append(gpu_id)
                reasons[gpu_id] = "+".join(reason_parts)
        if not yellow_ids:
            return {"recorded": False, "reason": "below_yellow_thresholds", "gpu_ids": []}
        return self.pressure_store.record_yellow(
            gpu_ids=yellow_ids,
            worker_id=self.worker_id,
            job_id="gpu_util_sample",
            resource_class="gpu_util_sample",
            reason="gpu_util_or_memory_near_limit",
            metadata={
                "gpu_reasons": reasons,
                "util_threshold_pct": util_threshold,
                "free_mem_threshold_gb": free_threshold,
            },
        )


    def sample_gpu_util(self, *, gpu_ids: list[str]) -> dict[str, Any]:
        ids = [str(x) for x in (gpu_ids or []) if str(x).strip()]
        sample = sample_nvidia_smi(ids)
        if isinstance(sample, dict) and sample.get("available") is not False:
            pressure = self._record_yellow_from_util_sample(sample)
            if pressure.get("recorded"):
                sample = {**sample, "pressure": pressure}
        return sample
