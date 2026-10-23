# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Pure projection from raw resource samples to stable resource facts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts import ResourceObservation


@dataclass(frozen=True, slots=True)
class ResourceFacts:
    """Policy-free facts produced from one resource observation."""

    worker_id: str
    observed_at: float
    available: bool
    gpu_ids: tuple[str, ...] = ()
    active_lease_ids: tuple[str, ...] = ()
    pending_job_ids: tuple[str, ...] = ()
    pressure: str = "unknown"
    generation: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


class ResourceObservationProjector:
    """Normalize observations without admitting, releasing, or killing work."""

    def project(
        self,
        *,
        worker_id: str,
        raw: Mapping[str, Any] | None,
        gpu_ids: tuple[str, ...] = (),
        observed_at: float | None = None,
        lease_snapshot: Mapping[str, Any] | None = None,
    ) -> ResourceFacts:
        sample = dict(raw or {})
        snapshot = dict(lease_snapshot or {})
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        waiters = snapshot.get("waiters") if isinstance(snapshot.get("waiters"), dict) else {}
        pressure = sample.get("pressure")
        if isinstance(pressure, dict):
            pressure_name = str(pressure.get("mode") or pressure.get("status") or "unknown")
        else:
            pressure_name = str(pressure or sample.get("resource_mode") or "unknown")
        generation = _int(
            sample.get("generation")
            or (pressure.get("generation") if isinstance(pressure, dict) else 0)
            or snapshot.get("generation")
        )
        return ResourceFacts(
            worker_id=str(worker_id or ""),
            observed_at=float(time.time() if observed_at is None else observed_at),
            available=bool(sample.get("available", sample.get("synced", True))),
            gpu_ids=_ids(gpu_ids or tuple(sample.get("gpu_ids") or ())),
            active_lease_ids=tuple(sorted(str(key) for key in leases)),
            pending_job_ids=tuple(sorted(str(key) for key in waiters)),
            pressure=pressure_name,
            generation=generation,
            metadata=sample,
        )

    @staticmethod
    def to_contract(facts: ResourceFacts) -> ResourceObservation:
        return ResourceObservation(
            worker_id=facts.worker_id,
            available=facts.available,
            observed_at=facts.observed_at,
            gpu_ids=facts.gpu_ids,
            active_lease_ids=facts.active_lease_ids,
            pressure=facts.pressure,
            metadata={
                **dict(facts.metadata),
                "pending_job_ids": list(facts.pending_job_ids),
                "resource_generation": facts.generation,
            },
        )


def project_gpu_share_decision_facts(
    observation: Mapping[str, Any], *, decision_source: str
) -> dict[str, Any]:
    """Project sharing evidence for a policy without selecting an action."""

    memory = observation.get("memory") if isinstance(observation.get("memory"), dict) else {}
    gates_raw = observation.get("hard_gates") if isinstance(observation.get("hard_gates"), dict) else {}
    gates = {str(key): bool(value) for key, value in gates_raw.items()}
    secondary = observation.get("secondary_allowed") if isinstance(observation.get("secondary_allowed"), dict) else {}
    trial = observation.get("trial_share") if isinstance(observation.get("trial_share"), dict) else {}
    cpu = observation.get("cpu_isolation") if isinstance(observation.get("cpu_isolation"), dict) else {}
    primary = observation.get("primary_signal") if isinstance(observation.get("primary_signal"), dict) else {}
    return {
        "schema_version": 1,
        "decision_source": str(decision_source or ""),
        "objective": "increase throughput by running revocable parallel GPU work when observed GPU headroom is idle",
        "share_eligible": bool(observation.get("share_eligible")),
        "share_candidate": bool(observation.get("share_candidate")),
        "observe_only": bool(observation.get("observe_only")),
        "hard_gates_pass": bool(gates) and all(gates.values()),
        "hard_gates": gates,
        "primary_gpu_util_p90_pct": _float(memory.get("gpu_util_p90_pct")),
        "primary_gpu_mem_current_gb": _float(memory.get("gpu_mem_current_gb")),
        "primary_gpu_mem_peak_gb": _float(memory.get("gpu_mem_peak_gb")),
        "gpu_free_mem_gb": _float(memory.get("gpu_free_mem_gb")),
        "required_headroom_gb": _float(memory.get("required_headroom_gb")),
        "secondary_estimated_peak_gb": _float(memory.get("secondary_estimated_peak_gb")),
        "cpu_pressure": str(observation.get("cpu_pressure") or "unknown"),
        "cpu_isolation_mode": str(cpu.get("mode") or "unknown"),
        "cpu_isolated": bool(cpu.get("isolated")),
        "secondary_allowed": bool(secondary.get("allowed")),
        "secondary_allowed_reason": str(secondary.get("reason") or ""),
        "secondary_policy_gate": str(secondary.get("policy_gate") or ""),
        "secondary_effective_slot_weight": _float(secondary.get("effective_slot_weight")),
        "trial_share": bool(trial.get("enabled")),
        "trial_primary_protected": bool(trial.get("primary_protected")),
        "productive_primary_signal": bool(primary.get("productive_primary_signal")) if primary else None,
        "llm_decision_focus": "grant sharing when these observed facts make the secondary cheap and revocable; deny or observe only for concrete memory, CPU, or progress uncertainty",
    }
def _ids(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["ResourceFacts", "ResourceObservationProjector", "project_gpu_share_decision_facts"]
