# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Independent registry, queue, and lease mechanisms."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, MutableMapping, Protocol, TypeVar


class LeaseStorePort(Protocol):
    def snapshot_active(self) -> dict[str, Any]: ...
    def release(self, *, job_id: str) -> dict[str, Any]: ...


JobT = TypeVar("JobT")


class ResourceRegistry(Generic[JobT]):
    """Own runtime job identity while allowing a temporary dict projection."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobT] = {}
        self._operation_sequences: dict[tuple[str, str], int] = {}

    @property
    def compat_jobs(self) -> MutableMapping[str, JobT]:
        return self._jobs

    def register(self, job_id: str, job: JobT) -> None:
        clean = str(job_id or "").strip()
        if not clean:
            raise ValueError("job_id is required")
        self._jobs[clean] = job

    def unregister(self, job_id: str) -> JobT | None:
        return self._jobs.pop(str(job_id or ""), None)

    def get(self, job_id: str) -> JobT | None:
        return self._jobs.get(str(job_id or ""))

    def snapshot_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._jobs))

    def next_operation_sequence(self, job_id: str, operation: str) -> int:
        key = (str(job_id or ""), str(operation or ""))
        sequence = self._operation_sequences.get(key, 0) + 1
        self._operation_sequences[key] = sequence
        return sequence


@dataclass(frozen=True, slots=True)
class QueueHeartbeat:
    emit: bool
    elapsed_sec: float = 0.0


class QueueScheduler:
    """Own pending membership and heartbeat cadence; never evaluates value."""

    def __init__(
        self,
        *,
        heartbeat_sec: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.heartbeat_sec = max(1.0, float(heartbeat_sec or 0.0))
        self._monotonic = monotonic
        self.pending_job_ids: set[str] = set()
        self.last_heartbeat: dict[str, float] = {}

    def recover(self, snapshot: Mapping[str, Any] | None) -> tuple[str, ...]:
        waiters = (snapshot or {}).get("waiters") if isinstance(snapshot, Mapping) else {}
        recovered = tuple(sorted(str(key) for key in waiters or {}))
        self.pending_job_ids.update(recovered)
        return recovered

    def mark_pending(self, job_id: str) -> bool:
        clean = str(job_id or "")
        first = clean not in self.pending_job_ids
        if clean:
            self.pending_job_ids.add(clean)
        return first

    def complete(self, job_id: str) -> None:
        clean = str(job_id or "")
        self.pending_job_ids.discard(clean)
        self.last_heartbeat.pop(clean, None)

    def heartbeat(self, job_id: str, *, elapsed_sec: float) -> QueueHeartbeat:
        clean = str(job_id or "")
        if not clean or clean not in self.pending_job_ids:
            return QueueHeartbeat(emit=False)
        now = self._monotonic()
        if now - self.last_heartbeat.get(clean, 0.0) < self.heartbeat_sec:
            return QueueHeartbeat(emit=False)
        self.last_heartbeat[clean] = now
        return QueueHeartbeat(emit=True, elapsed_sec=float(elapsed_sec or 0.0))


class LeaseManager:
    """Own recovered/local lease identity and idempotent release calls."""

    def __init__(self, store: LeaseStorePort) -> None:
        self.store = store
        self.assigned_gpu_ids: dict[str, list[str]] = {}
        self.recover()

    def recover(self) -> tuple[str, ...]:
        snapshot = self.store.snapshot_active()
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        self.assigned_gpu_ids.clear()
        for job_id, raw in leases.items():
            if isinstance(raw, dict):
                ids = [str(value) for value in raw.get("gpu_ids") or [] if str(value).strip()]
                if ids:
                    self.assigned_gpu_ids[str(job_id)] = ids
        return tuple(sorted(str(key) for key in leases))

    def remember(self, job_id: str, gpu_ids: list[str]) -> None:
        clean = str(job_id or "")
        ids = [str(value) for value in gpu_ids if str(value).strip()]
        if clean and ids:
            self.assigned_gpu_ids[clean] = ids

    def forget(self, job_id: str) -> None:
        self.assigned_gpu_ids.pop(str(job_id or ""), None)

    def has_active(self, job_id: str) -> bool:
        return str(job_id or "") in self.assigned_gpu_ids

    def persisted(self, job_id: str) -> dict[str, Any]:
        snapshot = self.store.snapshot_active()
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        raw = leases.get(str(job_id or ""))
        return dict(raw) if isinstance(raw, dict) else {}

    def release(self, job_id: str) -> dict[str, Any]:
        clean = str(job_id or "")
        result = self.store.release(job_id=clean)
        self.forget(clean)
        return result


__all__ = ["LeaseManager", "LeaseStorePort", "QueueHeartbeat", "QueueScheduler", "ResourceRegistry"]
