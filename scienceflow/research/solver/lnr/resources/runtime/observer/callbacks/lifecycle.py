# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Resource observer responsibility: completion guard, backoff, resource action, cleanup, and job finish.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_REVIEW_KILL,
    RESOURCE_REVIEW_NO_ACTION,
    ResourceEffectKind,
    hashlib,
    time,
)


class LifecycleCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def checkpoint_to_submission_guard(
        self,
        job_id: str | None,
        *,
        gap: dict[str, Any] | None,
        elapsed_sec: float,
    ) -> str:
        if not self.checkpoint_submission_guard_enabled or self.resource_runtime is None or not gap:
            return ""
        command_id = str(job_id or "")
        payload = {"job_id": command_id, "elapsed_sec": float(elapsed_sec or 0.0), **dict(gap)}
        self.resource_runtime.record_resource_event(
            "checkpoint_to_submission_guard",
            payload=payload,
            command_id=command_id,
            lease_id=command_id,
        )
        artifact_path = str(payload.get("artifact_path") or payload.get("submission_path") or "submission.csv")
        reason = str(payload.get("reason") or "checkpoint_without_submission")
        if artifact_path != "submission.csv":
            return (
                f"RESOURCE_FEEDBACK: {reason} because model/checkpoint artifacts were updated "
                f"but `{artifact_path}` was not updated.\n"
            )
        return (
            "RESOURCE_FEEDBACK: checkpoint_without_submission because model/checkpoint "
            "artifacts were updated but root submission.csv was not updated.\n"
        )


    def agent_backoff_wake_snapshot(self, **_: Any) -> dict[str, Any]:
        if self.resource_runtime is None:
            return {"pressure_generation": 0, "available": False, "reason": "resource_runtime_disabled"}
        return {
            "pressure_generation": self.resource_runtime.pressure_generation(),
            "available": False,
            "reason": "snapshot",
        }


    def agent_backoff_wake_decision(
        self,
        *,
        start_pressure_generation: int = 0,
        target_resource_class: str = RESOURCE_HEAVY_GPU_TRAIN,
        gpu_ids: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if self.resource_runtime is None:
            return {"wake": False, "wake_reason": "resource_runtime_disabled"}
        current = self.resource_runtime.pressure_generation()
        if current <= int(start_pressure_generation or 0):
            return {"wake": False, "wake_reason": "no_resource_change", "pressure_generation": current}
        availability = self.resource_runtime.resource_available_for(
            resource_class=str(target_resource_class or RESOURCE_HEAVY_GPU_TRAIN),
            gpu_ids=[str(x) for x in (gpu_ids or []) if str(x).strip()],
        )
        return {
            "wake": bool(availability.get("available")),
            "wake_reason": "resource_available" if availability.get("available") else str(availability.get("reason") or "resource_changed"),
            "pressure_generation": current,
            "availability": availability,
        }


    def agent_backoff_wait_started(
        self,
        *,
        wait_id: str,
        planned_sleep_sec: float,
        command: str = "",
        reason: str = "",
        **_: Any,
    ) -> None:
        wid = str(wait_id or "").strip()
        if not wid:
            return
        now = time.time()
        command_digest = hashlib.sha1(str(command or "").encode(errors="replace")).hexdigest()[:16]
        planned = max(0.0, float(planned_sleep_sec or 0.0))
        self._agent_backoff_waits[wid] = {
            "started_at": now,
            "planned_sleep_sec": planned,
            "command_digest": command_digest,
            "reason": str(reason or "agent_sleep_command"),
        }
        self.state_machine.append_event(
            "agent_backoff_wait_started",
            task_type="agent_backoff_wait",
            task_id=f"agent_backoff_wait:{wid}",
            status="waiting",
            payload={
                "wait_id": wid,
                "planned_sleep_sec": planned,
                "command_digest": command_digest,
                "reason": str(reason or "agent_sleep_command"),
            },
        )


    def agent_backoff_wait_finished(
        self,
        *,
        wait_id: str,
        planned_sleep_sec: float = 0.0,
        elapsed_sec: float = 0.0,
        status: str = "finished",
        wake_reason: str = "timer_elapsed",
        returncode: int | None = None,
        reason: str = "",
        **_: Any,
    ) -> None:
        wid = str(wait_id or "").strip()
        if not wid:
            return
        meta = self._agent_backoff_waits.pop(wid, {})
        planned = max(0.0, float(planned_sleep_sec or meta.get("planned_sleep_sec") or 0.0))
        elapsed = float(elapsed_sec or 0.0)
        if elapsed <= 0 and meta.get("started_at"):
            elapsed = max(0.0, time.time() - float(meta.get("started_at") or 0.0))
        finished_at = time.time()
        self._last_agent_backoff_wait = {
            "wait_id": wid,
            "planned_sleep_sec": planned,
            "elapsed_sec": max(0.0, elapsed),
            "status": str(status or "finished"),
            "wake_reason": str(wake_reason or "timer_elapsed"),
            "returncode": returncode,
            "reason": str(reason or ""),
            "command_digest": str(meta.get("command_digest") or ""),
            "started_at": meta.get("started_at"),
            "finished_at": finished_at,
        }
        self.state_machine.append_event(
            "agent_backoff_wait_finished",
            task_type="agent_backoff_wait",
            task_id=f"agent_backoff_wait:{wid}",
            status=str(status or "finished"),
            payload={
                "wait_id": wid,
                "planned_sleep_sec": planned,
                "elapsed_sec": max(0.0, elapsed),
                "wake_reason": str(wake_reason or "timer_elapsed"),
                "returncode": returncode,
                "reason": str(reason or ""),
                "command_digest": str(meta.get("command_digest") or ""),
            },
        )


    def resource_guard_action(self, job_id: str | None, *, action: str, reason: str, elapsed_sec: float, **extra: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        self._promote(job, reason="guard_action", elapsed_sec=float(elapsed_sec or 0.0))
        self._remember_resource_feedback(
            job,
            status="DENIED_REPLAN",
            reason=str(reason or "resource_guard_action"),
            resource_mode="RED",
            blocked_class=job.resource_class,
            allowed_classes=[RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, "readonly_cpu", "light_cpu"],
            eta_next_train_sec=420.0,
            cooldown_sec=120.0,
        )
        action_text = str(action or "")
        executed_actions = {
            "terminate",
            "terminate_by_arbiter",
            "stop_boundary_violation",
            "gpu_orphan_cleanup",
        }
        execution_outcome = RESOURCE_REVIEW_KILL if action_text in executed_actions else RESOURCE_REVIEW_NO_ACTION
        payload = {
            "action": execution_outcome,
            "raw_action": action,
            "execution_outcome": execution_outcome,
            "reason": reason,
            "status": "DENIED_REPLAN",
            "execution_status": "executed" if action_text in executed_actions else "denied",
            "resource_mode": "RED",
            "blocked_class": job.resource_class,
            "allowed_classes": [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, "readonly_cpu", "light_cpu"],
            "elapsed_sec": float(elapsed_sec or 0.0),
            "executed": action_text in executed_actions,
            "last_progress": dict(job.last_progress or {}),
            "last_artifact_progress": dict(job.last_artifact_progress or {}),
            "last_signal": dict(job.last_signal or {}),
        }
        for key, value in extra.items():
            if key and value is not None:
                payload[str(key)] = value
        self._emit(
            "resource_guard_action",
            job,
            status=str(action or ""),
            payload=payload,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "execution",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
            if action_text in {"terminate", "terminate_by_arbiter"}:
                self.resource_runtime.record_resource_event(
                    "resource_kill_executed",
                    payload=payload,
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
                self.resource_runtime.clear_active_resource_proposals_for_command(job.job_id)


    def resource_cleanup_heartbeat(
        self,
        job_id: str | None,
        *,
        action: str,
        reason: str,
        elapsed_sec: float,
        **extra: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        payload = {
            "action": str(action or "gpu_cleanup_skipped"),
            "observation_status": "OBSERVED",
            "reason": str(reason or "workspace_gpu_cleanup"),
            "elapsed_sec": float(elapsed_sec or 0.0),
        }
        for key, value in extra.items():
            if key and value is not None:
                payload[str(key)] = value
        self._emit(
            "resource_cleanup_heartbeat",
            job,
            status="observed",
            payload=payload,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_cleanup_heartbeat",
                payload={
                    "job_id": job.job_id,
                    "command_digest": job.command_digest,
                    "resource_class": job.resource_class,
                    "gpu_ids": job.gpu_ids,
                    "cpu_set": job.cpu_set,
                    "timeout_sec": job.timeout_sec,
                    **payload,
                },
                command_id=job.job_id,
                lease_id=job.job_id,
            )


    def job_finished(
        self,
        job_id: str | None,
        *,
        status: str,
        elapsed_sec: float,
        returncode: int | None = None,
        reason: str | None = None,
        **_: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs.pop(job_id)
        self._review_states.pop(job_id, None)
        self._review_machines.pop(job_id, None)
        if self.resource_runtime is not None:
            self.resource_runtime.clear_active_resource_proposals_for_command(job_id)
        job.exit_code_seen = returncode is not None
        job.terminal_signal_seen = True
        elapsed = float(elapsed_sec or 0.0)
        release: dict[str, Any] = {}
        if self.resource_runtime is not None:
            release = self._execute_resource_effect(
                ResourceEffectKind.RELEASE,
                command_id=job_id,
                idempotency_key=f"release:{job_id}:{status or 'finished'}",
                parameters={
                    "job_id": job_id,
                    "elapsed_sec": elapsed,
                    "status": str(status or "finished"),
                },
            )
            if release.get("released"):
                self._emit(
                    "resource_gpu_lease_released",
                    job,
                    status="released",
                    payload={
                        "elapsed_sec": elapsed,
                        "release_reason": str(status or "finished"),
                        "pressure_generation": release.get("pressure_generation"),
                        "lease": release.get("lease") or {},
                    },
                )
        runtime_pressure: dict[str, Any] = {}
        status_text = str(status or "")
        reason_text = str(reason or "")
        pressure_text = f"{status_text} {reason_text}".lower()
        pressure_reason = ""
        if status_text == "resource_guard_terminated":
            pressure_reason = reason_text or "guard_terminated"
        elif "out of memory" in pressure_text or "cuda oom" in pressure_text or "oom" in pressure_text:
            pressure_reason = reason_text or status_text or "oom"
        if pressure_reason and self.resource_runtime is not None and job.gpu_ids:
            runtime_pressure = self.resource_runtime.record_runtime_pressure(
                job_id=job_id,
                resource_class=job.resource_class,
                gpu_ids=job.gpu_ids,
                elapsed_sec=elapsed,
                reason=pressure_reason,
                command_digest=job.command_digest,
            )
            self._emit(
                "resource_gpu_runtime_pressure",
                job,
                status="red",
                payload={
                    "elapsed_sec": elapsed,
                    "reason": pressure_reason,
                    "release_reason": status_text,
                    "pressure": runtime_pressure,
                },
            )
        if not job.visible and elapsed >= self.min_register_sec:
            self._promote(job, reason="elapsed_threshold_finish", elapsed_sec=elapsed)
        if not job.visible:
            return
        self._emit(
            "resource_job_finished",
            job,
            status=str(status or "finished"),
            payload={
                "elapsed_sec": elapsed,
                "returncode": returncode,
                "reason": reason or "",
                "filtered_signal": dict(job.last_signal),
            },
        )
