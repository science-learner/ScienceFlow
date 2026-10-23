# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Runtime heartbeat and structured-progress observation callbacks."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (Any, ResourceJob, compact_metric_history_text, gpu_memory_summary, metric_history_line_from_progress_signals, time, update_metric_history_lines)


class MonitoringCallbacks:
    """Own MonitoringCallbacks resource behavior without delegated forwarding."""

    def _maybe_emit_monitor_agent_shadow(self, job: ResourceJob, signal: dict[str, Any]) -> None:
        if self.monitor_agent_mode == "off" or not job.visible:
            return
        now = time.time()
        last = self._last_monitor_agent_emit.get(job.job_id, 0.0)
        if now - last < self.monitor_agent_min_interval_sec:
            return
        self._last_monitor_agent_emit[job.job_id] = now
        self._emit(
            "resource_monitor_agent_shadow",
            job,
            status="observed",
            payload={
                "mode": self.monitor_agent_mode,
                "filtered_signal": dict(signal),
                "decision_provider": "reserved_interface",
                "decision_applied": False,
            },
        )


    def _maybe_emit_gpu_util_sample(self, job: ResourceJob, *, elapsed_sec: float) -> None:
        if not self.gpu_util_observer_enabled or self.resource_runtime is None:
            return
        if not job.gpu_ids:
            return
        now = time.time()
        last = self._last_gpu_util_emit.get(job.job_id, 0.0)
        if now - last < self.gpu_util_sample_interval_sec:
            return
        self._last_gpu_util_emit[job.job_id] = now
        sample = self.resource_runtime.sample_gpu_util(gpu_ids=job.gpu_ids)
        if not isinstance(sample, dict):
            sample = {"available": False, "reason": "invalid_sample", "gpus": []}
        process_gpu_placement = self._job_process_gpu_placement(job)
        process_gpu_observed = process_gpu_placement.get("available") is True
        process_has_gpu = self._process_gpu_placement_has_usage(process_gpu_placement)
        if process_gpu_observed:
            idle_now = (not process_has_gpu) or self._gpu_util_sample_is_idle(sample, job.gpu_ids)
            low_compute_now = process_has_gpu and self._gpu_util_sample_is_low_compute_with_model(sample, job.gpu_ids)
        else:
            idle_now = self._gpu_util_sample_is_idle(sample, job.gpu_ids)
            low_compute_now = self._gpu_util_sample_is_low_compute_with_model(sample, job.gpu_ids)
        mem_summary = gpu_memory_summary(sample, job.gpu_ids, previous_peak_gb=job.gpu_mem_peak_gb)
        job.gpu_mem_peak_gb = float(mem_summary.get("gpu_mem_peak_gb") or job.gpu_mem_peak_gb or 0.0)
        job.idle_gpu_lease_samples = job.idle_gpu_lease_samples + 1 if idle_now else 0
        job.dataloader_bottleneck_samples = job.dataloader_bottleneck_samples + 1 if low_compute_now else 0
        guard_snapshot = {
            "idle_now": idle_now,
            "idle_samples": job.idle_gpu_lease_samples,
            "min_samples": self.gpu_idle_lease_min_samples,
            "util_threshold_pct": self.gpu_idle_lease_util_pct,
            "mem_threshold_gb": self.gpu_idle_lease_mem_gb,
            "dataloader_low_compute_now": low_compute_now,
            "dataloader_low_compute_samples": job.dataloader_bottleneck_samples,
            "dataloader_util_threshold_pct": self.gpu_dataloader_bottleneck_util_pct,
            "dataloader_min_mem_gb": self.gpu_dataloader_bottleneck_min_mem_gb,
            "gpu_mem_peak_gb": job.gpu_mem_peak_gb,
            "process_gpu_observed": process_gpu_observed,
            "process_has_gpu": process_has_gpu,
        }
        job.last_gpu_util_sample = {
            "elapsed_sec": float(elapsed_sec or 0.0),
            "sample": sample,
            "process_gpu_placement": process_gpu_placement,
            "idle_gpu_lease_guard": guard_snapshot,
        }
        self._emit(
            "resource_gpu_util_sampled",
            job,
            status="observed",
            payload={
                "elapsed_sec": float(elapsed_sec or 0.0),
                "sample": sample,
                "pressure": sample.get("pressure") if isinstance(sample.get("pressure"), dict) else {},
                "process_gpu_placement": process_gpu_placement,
                "idle_gpu_lease_guard": guard_snapshot,
            },
        )


    def progress_heartbeat(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        phase: str = "",
        signals: dict[str, Any] | None = None,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        metric_history_text: str = "",
        metric_history_line_count: int = 0,
        emit_reason: str = "",
        **_: Any,
    ) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        stdout_line_count = int(stdout_lines or 0)
        stdout_byte_count = int(stdout_bytes or 0)
        metric_text = str(metric_history_text or "").strip()
        metric_lines = int(metric_history_line_count or 0)
        signals_payload = dict(signals or {})
        previous_progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured_progress = self._structured_progress_snapshot(
            signals_payload,
            previous_progress_payload,
            current_elapsed_sec=elapsed,
        )
        structured_metric_line = metric_history_line_from_progress_signals(signals_payload)
        if structured_metric_line and not metric_text:
            existing_text = str(job.metric_history_text or "").strip()
            existing_lines = existing_text.splitlines() if existing_text else []
            merged_lines = update_metric_history_lines(existing_lines, structured_metric_line)
            metric_text = compact_metric_history_text(merged_lines)
            metric_lines = len(merged_lines)
        payload = {
            "elapsed_sec": elapsed,
            "phase": str(phase or ""),
            "signals": signals_payload,
            "stdout_lines": stdout_line_count,
            "stdout_bytes": stdout_byte_count,
            "metric_history_text": metric_text,
            "metric_history_line_count": metric_lines,
            "emit_reason": str(emit_reason or ""),
            "resource_efficiency": {
                "stdout_lines_per_sec": round(stdout_line_count / max(1.0, elapsed), 6),
                "stdout_bytes_per_sec": round(stdout_byte_count / max(1.0, elapsed), 6),
            },
        }
        if structured_progress:
            payload["structured_progress"] = structured_progress
        job.last_progress = payload
        if metric_text:
            job.metric_history_text = metric_text
            job.metric_history_line_count = metric_lines
        signals_payload = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        artifact_payload = signals_payload.get("artifact") if isinstance(signals_payload.get("artifact"), dict) else {}
        heartbeat = signals_payload.get("heartbeat") if isinstance(signals_payload.get("heartbeat"), dict) else {}
        if artifact_payload and heartbeat:
            for key in ("route_id", "fold", "validation_protocol", "checkpoint_kind", "safe_to_resume", "mergeable"):
                value = heartbeat.get(key)
                if value not in {None, ""} and key not in artifact_payload:
                    artifact_payload[key] = value
        if artifact_payload:
            job.last_artifact_progress = payload
            job.last_recoverability = self._recoverability_from_artifact(job, artifact_payload, elapsed_sec=elapsed)
        self._emit("progress_heartbeat", job, status="observed", payload=payload)


    def stdout_heartbeat(self, job_id: str | None) -> None:
        _ = job_id


    def _structured_progress_snapshot(
        self,
        signals: dict[str, Any],
        previous_payload: dict[str, Any] | None,
        *,
        current_elapsed_sec: float,
    ) -> dict[str, Any]:
        current = self._structured_progress_entry(signals)
        if not current:
            return {}
        previous = (previous_payload or {}).get("structured_progress")
        previous = previous if isinstance(previous, dict) else {}
        previous_current = self._float_or_none(previous.get("current"))
        previous_total = self._float_or_none(previous.get("total"))
        same_unit = str(previous.get("unit") or "") == str(current.get("unit") or "")
        same_total = bool(
            previous_total is not None
            and abs(float(previous_total) - float(current["total"])) <= max(1e-9, abs(float(current["total"])) * 1e-6)
        )
        advanced = bool(same_unit and same_total and previous_current is not None and float(current["current"]) > previous_current)
        out = dict(current)
        previous_samples = int(previous.get("interval_sample_count") or 0) if same_unit and same_total else 0
        out["advanced"] = advanced
        out["interval_sample_count"] = previous_samples + 1 if advanced else previous_samples
        if same_unit and previous_current is not None:
            out["previous_current"] = previous_current
        if same_unit and previous_total is not None:
            out["previous_total"] = previous_total
        previous_elapsed = self._float_or_none((previous_payload or {}).get("elapsed_sec"))
        current_elapsed = self._float_or_none((signals.get("heartbeat") or {}).get("elapsed_s"))
        if current_elapsed is None:
            current_elapsed = max(0.0, float(current_elapsed_sec or 0.0))
        if (
            advanced
            and previous_current is not None
            and previous_elapsed is not None
            and current_elapsed is not None
            and current_elapsed > previous_elapsed
        ):
            interval_sec = current_elapsed - previous_elapsed
            rate = (float(current["current"]) - previous_current) / interval_sec
            if rate > 0.0:
                out["interval_sec"] = interval_sec
                out["rate_units_per_sec"] = rate
                out["eta_to_phase_end_sec"] = max(0.0, (float(current["total"]) - float(current["current"])) / rate)
        return out


    @staticmethod
    def _progress_unit_scope(unit: Any) -> str:
        normalized = str(unit or "").strip().lower()
        if normalized in {"batch", "batches", "step", "steps", "iteration", "iterations"}:
            return "subphase"
        if normalized in {
            "overall",
            "run",
            "runs",
            "task",
            "tasks",
            "trial",
            "trials",
            "fold",
            "folds",
            "epoch",
            "epochs",
            "round",
            "rounds",
        }:
            return "route"
        return "phase"


    def _structured_progress_entry(self, signals: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(signals, dict):
            return {}
        evidence = signals.get("progress_evidence") if isinstance(signals.get("progress_evidence"), dict) else {}
        best: dict[str, Any] = {}
        scope_rank = {"subphase": 1, "phase": 2, "route": 3}
        for unit, raw_entry in signals.items():
            if unit in {"heartbeat", "metrics", "artifact", "phase"} or not isinstance(raw_entry, dict):
                continue
            current = self._float_or_none(raw_entry.get("current"))
            total = self._float_or_none(raw_entry.get("total"))
            if current is None or total is None or total <= 0.0:
                continue
            if current < 0.0 or current > total:
                continue
            progress_scope = self._progress_unit_scope(unit)
            candidate = {
                "unit": str(unit),
                "current": current,
                "total": total,
                "progress_scope": progress_scope,
                "source": str(evidence.get("source") or raw_entry.get("source") or "unknown"),
                "evidence_trust": str(evidence.get("trust") or "unknown"),
            }
            if not best or scope_rank[progress_scope] > scope_rank[str(best["progress_scope"])]:
                best = candidate
        return best


    @staticmethod
    def _cpu_bucket(process_tree_cpu: dict[str, Any] | None) -> str:
        cpu = process_tree_cpu if isinstance(process_tree_cpu, dict) else {}
        if not cpu or not cpu.get("available", True):
            return "unknown"
        try:
            total_cpu = float(cpu.get("total_cpu_pct") or 0.0)
        except (TypeError, ValueError):
            total_cpu = 0.0
        try:
            busy_children = int(float(cpu.get("busy_child_count") or 0.0))
        except (TypeError, ValueError):
            busy_children = 0
        if total_cpu >= 300.0 or busy_children >= 2:
            return "heavy"
        if total_cpu >= 50.0 or busy_children >= 1:
            return "active"
        return "idle"


    def monitor_heartbeat(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        stdout_age_sec: float = 0.0,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        pid: int | None = None,
        returncode: int | None = None,
        process_tree_cpu: dict[str, Any] | None = None,
        source: str = "bash_guard",
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id:
            return {"recorded": False, "reason": "missing_job_id"}
        if job_id not in self._jobs:
            return {"recorded": False, "reason": "missing_job"}
        job = self._jobs[job_id]
        elapsed = max(0.0, float(elapsed_sec or 0.0))
        interval = max(0.05, float(self.review_heartbeat_sec or 60.0))
        if job.monitor_heartbeat_count > 0 and elapsed < float(job.last_monitor_heartbeat_elapsed_sec or 0.0) + interval:
            return {"recorded": False, "reason": "heartbeat_interval", "next_after_sec": interval}
        if pid is not None:
            try:
                job.pid = int(pid)
                if job.pgid is None:
                    job.pgid = int(pid)
            except (TypeError, ValueError):
                pass
        if elapsed >= self.min_register_sec:
            self._promote(job, reason="monitor_heartbeat", elapsed_sec=elapsed)
        process_alive = returncode is None
        artifact_age = self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed)
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed)
        recoverability = self._fresh_recoverability_for_job(job, elapsed_sec=elapsed)
        monitor_signal = {
            "elapsed_sec": elapsed,
            "stdout_age_sec": max(0.0, float(stdout_age_sec or 0.0)),
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "process_tree_cpu": dict(process_tree_cpu or {}),
        }
        stdout_observation = self._stdout_observation_for_job(job, monitor_signal, elapsed_sec=elapsed)
        review_state = self._review_states.get(job.job_id)
        payload = {
            "elapsed_sec": elapsed,
            "process_alive": process_alive,
            "pid": job.pid,
            "pgid": job.pgid,
            "stdout_age_sec": max(0.0, float(stdout_age_sec or 0.0)),
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "cpu_bucket": self._cpu_bucket(process_tree_cpu),
            "gpu_bucket": "active" if self._last_gpu_sample_active(job) else "idle_or_none",
            "artifact_age_sec": artifact_age,
            "progress_age_sec": progress_age,
            "recoverability": recoverability,
            "stdout_observation": stdout_observation,
            "heartbeat_count": int(job.monitor_heartbeat_count) + 1,
            "source": str(source or "bash_guard"),
            "process_tree_cpu": dict(process_tree_cpu or {}),
            "review_state": review_state.to_json() if review_state is not None else {},
        }
        job.monitor_heartbeat_count += 1
        job.last_monitor_heartbeat_elapsed_sec = elapsed
        self._emit("resource_monitor_heartbeat", job, status="observed", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_monitor_heartbeat",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"recorded": True, "heartbeat_count": job.monitor_heartbeat_count}


    def resource_monitor_gap(
        self,
        job_id: str | None,
        *,
        elapsed_sec: float = 0.0,
        gap_sec: float = 0.0,
        stdout_age_sec: float = 0.0,
        stdout_lines: int = 0,
        stdout_bytes: int = 0,
        reason: str = "",
        source: str = "bash_guard_watchdog",
        process_tree_cpu: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not job_id:
            return {"recorded": False, "reason": "missing_job_id"}
        if job_id not in self._jobs:
            return {"recorded": False, "reason": "missing_job"}
        job = self._jobs[job_id]
        payload = {
            "elapsed_sec": max(0.0, float(elapsed_sec or 0.0)),
            "gap_sec": max(0.0, float(gap_sec or 0.0)),
            "stdout_age_sec": max(0.0, float(stdout_age_sec or 0.0)),
            "stdout_lines": int(stdout_lines or 0),
            "stdout_bytes": int(stdout_bytes or 0),
            "heartbeat_count": int(job.monitor_heartbeat_count or 0),
            "last_monitor_heartbeat_elapsed_sec": float(job.last_monitor_heartbeat_elapsed_sec or 0.0),
            "reason": str(reason or "resource_monitor_gap"),
            "source": str(source or "bash_guard_watchdog"),
            "pid": job.pid,
            "pgid": job.pgid,
            "process_tree_cpu": dict(process_tree_cpu or {}),
        }
        self._emit("resource_monitor_gap", job, status="observed", payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_monitor_gap",
                payload=payload,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        return {"recorded": True, "gap_sec": payload["gap_sec"]}


    def resume_monitor_event(
        self,
        event: str,
        *,
        tool_name: str = "",
        tool_call_id: str = "",
        command: str = "",
        status: str = "",
        payload: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        body = {
            "event": str(event or "resume_monitor_event"),
            "tool_name": str(tool_name or ""),
            "tool_call_id": str(tool_call_id or ""),
            "command": str(command or "")[:1000],
            **dict(payload or {}),
        }
        self.state_machine.append_event(
            "resource_resume_monitor_event",
            task_type="resource_resume",
            task_id=f"resource_resume:{tool_call_id or event or 'unknown'}",
            status=str(status or event or "observed"),
            payload=body,
        )
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event(
                "resource_resume_monitor_event",
                payload=body,
                command_id=str(tool_call_id or ""),
                lease_id=str(tool_call_id or ""),
            )
