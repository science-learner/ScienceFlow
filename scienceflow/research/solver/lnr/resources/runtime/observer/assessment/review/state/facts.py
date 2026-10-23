# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Process, GPU, execution, progress, and recoverability review facts."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    ResourceJob,
    _SAFETY_HEARTBEAT_SOURCE,
    build_execution_facts,
    build_process_liveness,
    build_resource_state_generation,
    classify_process_lifecycle,
    time,
)


class ReviewFacts:
    """Own ReviewFacts resource behavior without delegated forwarding."""

    def _process_liveness_for_job(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        cached = signal.get("_process_liveness") if isinstance(signal.get("_process_liveness"), dict) else None
        if cached is not None:
            return dict(cached)
        terminal_seen = bool(signal.get("terminal_signal_events") or signal.get("saw_final_score") or job.terminal_signal_seen)
        process_tree_cpu = dict(signal.get("process_tree_cpu") or {})
        liveness = build_process_liveness(
            root_pid=job.pid,
            root_pgid=job.pgid,
            expected_root_start_time_epoch=job.pid_start_time_epoch,
            exit_code_seen=bool(job.exit_code_seen),
            terminal_signal_seen=terminal_seen,
            process_tree_cpu=process_tree_cpu,
        )
        lifecycle = classify_process_lifecycle(
            process_liveness=liveness,
            process_tree_cpu=process_tree_cpu,
            tool_waiting=True,
        ).to_json()
        liveness["lifecycle_status"] = lifecycle.get("status")
        liveness["lifecycle"] = lifecycle
        if liveness.get("root_pid_alive") or liveness.get("process_group_alive") or liveness.get("live_descendant_count"):
            liveness["last_known_pid_seen_at"] = time.time()
        signal["_process_liveness"] = dict(liveness)
        return liveness


    def _current_gpu_mem_gb_for_job(self, job: ResourceJob) -> float:
        sample_bundle = job.last_gpu_util_sample if isinstance(job.last_gpu_util_sample, dict) else {}
        placement = sample_bundle.get("process_gpu_placement") if isinstance(sample_bundle.get("process_gpu_placement"), dict) else {}
        if placement.get("available") is True:
            return self._process_gpu_placement_mem_mb(placement) / 1024.0
        sample = sample_bundle.get("sample") if isinstance(sample_bundle.get("sample"), dict) else {}
        rows = sample.get("gpus") if isinstance(sample, dict) and isinstance(sample.get("gpus"), list) else []
        wanted = {str(x) for x in (job.gpu_ids or []) if str(x).strip()}
        total_mb = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "").strip()
            if wanted and gpu_id not in wanted:
                continue
            try:
                total_mb += max(0.0, float(row.get("memory_used_mb") or 0.0))
            except (TypeError, ValueError):
                pass
        return total_mb / 1024.0


    def _resource_snapshot_for_job(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        gpu_rows: list[dict[str, Any]] = []
        sample = job.last_gpu_util_sample.get("sample") if isinstance(job.last_gpu_util_sample, dict) else {}
        rows = sample.get("gpus") if isinstance(sample, dict) and isinstance(sample.get("gpus"), list) else []
        placement = job.last_gpu_util_sample.get("process_gpu_placement") if isinstance(job.last_gpu_util_sample, dict) else {}
        placement_available = isinstance(placement, dict) and placement.get("available") is True
        process_mem_by_gpu = self._process_gpu_memory_by_gpu(placement) if placement_available else {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            gpu_id = str(row.get("gpu_id") or row.get("index") or "")
            process_mem_mb = process_mem_by_gpu.get(gpu_id, 0.0)
            utilization = row.get("utilization_gpu_pct")
            memory_used = row.get("memory_used_mb")
            if placement_available:
                utilization = utilization if process_mem_mb > 0.0 else 0.0
                memory_used = process_mem_mb
            gpu_rows.append({
                "resource_type": "gpu",
                "id": gpu_id,
                "utilization_gpu_pct": utilization,
                "memory_used_mb": memory_used,
                "memory_total_mb": row.get("memory_total_mb"),
                "process_gpu_attributed": placement_available,
            })
        elapsed = max(
            0.0,
            float(signal.get("elapsed_sec") or 0.0),
            float(job.last_monitor_heartbeat_elapsed_sec or 0.0),
            float((job.last_progress or {}).get("elapsed_sec") or 0.0) if isinstance(job.last_progress, dict) else 0.0,
            float((job.last_artifact_progress or {}).get("elapsed_sec") or 0.0) if isinstance(job.last_artifact_progress, dict) else 0.0,
        )
        stdout_observation = self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed)
        resources = gpu_rows + [{
            "resource_type": "cpu",
            "id": "process",
            "stdout_lines": int(signal.get("stdout_lines") or 0),
            "stdout_bytes": int(signal.get("stdout_bytes") or 0),
            "stdout_observation": stdout_observation,
            "process_tree_cpu": dict(signal.get("process_tree_cpu") or {}),
        }]
        process_liveness = self._process_liveness_for_job(job, signal)
        return {
            "resource_class_declared": job.resource_class,
            "resource_class_observed": job.resource_class,
            "primary_resource_type": "gpu" if self._job_uses_expensive_gpu(job) else "cpu",
            "recoverability": self._fresh_recoverability_for_job(job, elapsed_sec=elapsed),
            "stdout_observation": stdout_observation,
            "lease": {
                "resource_type": "gpu",
                "resource_ids": list(job.gpu_ids),
                "active": bool(self.resource_runtime and self.resource_runtime.has_active_lease(job_id=job.job_id)),
            },
            "process_liveness": process_liveness,
            "resources": resources,
        }


    def _execution_facts_for_job(
        self,
        job: ResourceJob,
        *,
        resource_snapshot: dict[str, Any],
        progress_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        gpu_expected = self._job_uses_expensive_gpu(job)
        if self.resource_runtime is not None and self.resource_runtime.has_active_lease(job_id=job.job_id):
            gpu_expected = True
        return build_execution_facts(
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
            command=job.command,
            source_hint=job.source_hint,
            assigned_gpu_ids=list(job.gpu_ids or []),
            gpu_expected=gpu_expected,
        )


    def _ensure_resource_proposal_facts(
        self,
        job: ResourceJob,
        proposal: dict[str, Any],
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        resource_snapshot = proposal.get("resource_snapshot") if isinstance(proposal.get("resource_snapshot"), dict) else {}
        progress_snapshot = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        if not resource_snapshot:
            resource_snapshot = self._resource_snapshot_for_job(job, signal)
            proposal["resource_snapshot"] = resource_snapshot
        if not progress_snapshot:
            progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=float(signal.get("elapsed_sec") or 0.0))
            proposal["progress_snapshot"] = progress_snapshot
        if not isinstance(proposal.get("execution_facts"), dict):
            proposal["execution_facts"] = self._execution_facts_for_job(
                job,
                resource_snapshot=resource_snapshot,
                progress_snapshot=progress_snapshot,
            )
        if not isinstance(proposal.get("research_cadence"), dict):
            proposal["research_cadence"] = self._research_cadence_fact_card(
                job,
                signal,
                progress_snapshot,
            )
        state = self._review_states.get(job.job_id)
        if state is not None and not isinstance(proposal.get("clear_on_cadence"), dict):
            proposal["clear_on_cadence"] = dict((state.to_json().get("clear_on_cadence") or {}))
        if not isinstance(proposal.get("contention_context"), dict):
            proposal["contention_context"] = dict(signal.get("contention_context") or self._contention_context_for_job(job))
        if not isinstance(proposal.get("budget_context"), dict):
            remaining = float(progress_snapshot.get("deadline_remaining_sec") or signal.get("deadline_remaining_sec") or 0.0)
            budget_cap = remaining * float(self.review_config.timebox_budget_fraction or 0.10) if remaining > 0.0 else float(self.review_config.max_timebox_sec or 1800.0)
            proposal["budget_context"] = {
                "remaining_budget_sec": remaining,
                "timebox_budget_cap_sec": budget_cap,
            }
        if not isinstance(proposal.get("state_generation"), dict):
            state_generation = build_resource_state_generation({
                "proposal_type": proposal.get("proposal_type"),
                "reason_code": proposal.get("reason_code"),
                "resource_snapshot": resource_snapshot,
                "progress_snapshot": progress_snapshot,
                "blocker": proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {},
                "gpu_ids": list(job.gpu_ids or []),
                "execution_facts": proposal.get("execution_facts"),
            })
            proposal["control_generation_key"] = state_generation.control_generation_key
            proposal["feedback_generation_key"] = state_generation.feedback_generation_key
            proposal["state_generation"] = state_generation.to_json()
        return proposal


    def _progress_snapshot_for_job(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        artifact = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        progress_state = dict(job.progress_signal_state or {})
        recoverability = self._fresh_recoverability_for_job(job, elapsed_sec=elapsed_sec)
        stdout_observation = signal.get("stdout_observation") if isinstance(signal.get("stdout_observation"), dict) else self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        process_liveness = self._process_liveness_for_job(job, signal)
        finish_feasibility = self._progress_finish_feasibility(job, signal, elapsed_sec=elapsed_sec)
        return {
            "runtime_sec": float(elapsed_sec or 0.0),
            "stdout_last_line_age_sec": self._float_or_none(signal.get("stdout_age_sec")),
            "stdout_lines": int(signal.get("stdout_lines") or 0),
            "stdout_bytes": int(signal.get("stdout_bytes") or 0),
            "metric_history_text": str(signal.get("metric_history_text") or job.metric_history_text or ""),
            "metric_history_line_count": int(signal.get("metric_history_line_count") or job.metric_history_line_count or 0),
            "metric_scope_key": str((signal.get("resource_metric_value") or {}).get("metric_scope_key") or ""),
            "meaningful_stdout": bool(signal.get("saw_training_progress") and stream.get("fresh")),
            "metric_last_update_age_sec": self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed_sec),
            "artifact_last_update_age_sec": self._float_or_none(artifact.get("age_sec")) if artifact else self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec),
            "artifact_updates": [artifact] if artifact else [],
            "recoverability": recoverability,
            "stdout_observation": stdout_observation,
            "recoverable_artifact_on_disk": bool(recoverability.get("recoverable_artifact_on_disk")),
            "checkpoint_on_signal_supported_observed": bool(recoverability.get("checkpoint_on_signal_supported_observed")),
            "last_recoverable_artifact_age_sec": recoverability.get("last_recoverable_artifact_age_sec"),
            "artifact_watch_pattern": str(recoverability.get("artifact_watch_pattern") or "none"),
            "stop_cost": str(recoverability.get("stop_cost") or "high"),
            "known_stage": str(signal.get("current_phase") or ""),
            "process_tree_cpu": dict(signal.get("process_tree_cpu") or {}),
            "process_liveness": process_liveness,
            "heartbeat_source": _SAFETY_HEARTBEAT_SOURCE,
            "quick_probe": dict(job.quick_probe or {}),
            "output_pattern": str((job.quick_probe or {}).get("output_pattern") or "unknown"),
            "near_submission": bool(signal.get("saw_final_score")),
            "deliverable_validity": str(job.deliverable_validity or "none"),
            "progress_signal": progress_state.get("progress_signal") or job.progress_signal or "unknown",
            "progress_confidence": progress_state.get("progress_confidence") or self._progress_confidence(job, signal, elapsed_sec=elapsed_sec),
            "progress_signal_windows": int(progress_state.get("progress_signal_windows") or job.progress_signal_windows or 0),
            "progress_signal_reason": str(progress_state.get("progress_signal_reason") or ""),
            "multi_window_low_progress": bool(progress_state.get("multi_window_low_progress")),
            "stalled_mark_count": int(job.stalled_mark_count or 0),
            "deadline_event": bool(signal.get("deadline_event")),
            "deadline_remaining_sec": float(signal.get("deadline_remaining_sec") or 0.0),
            "finalization_reserve_sec": float(signal.get("finalization_reserve_sec") or 0.0),
            **finish_feasibility,
            "resource_efficiency": dict(
                signal.get("resource_efficiency") or job.resource_efficiency_state or {}
            ),
            "phase_completion_protected": bool(
                finish_feasibility.get("phase_completion_protected")
                and not (
                    signal.get("resource_efficiency")
                    or job.resource_efficiency_state
                    or {}
                ).get("completion_grace_expired")
            ),
        }


    def _recoverability_from_artifact(
        self,
        job: ResourceJob,
        artifact: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        recoverable = bool(artifact.get("recoverable_artifact_on_disk"))
        run_state_status = str(artifact.get("run_state_status") or "").strip().lower()
        checkpoint_supported_observed = bool(
            artifact.get("run_state_confirmed") and run_state_status in {"interrupted", "completed"}
        )
        try:
            age = float(artifact.get("age_sec") or 0.0)
        except (TypeError, ValueError):
            age = 0.0
        if recoverable:
            watch_pattern = "periodic" if str(artifact.get("artifact_scope") or "") == "current_run" else "terminal"
            stop_cost = "low"
            last_recoverable_age: float | None = age
        else:
            watch_pattern = "growing" if artifact.get("candidate_artifact") else "none"
            stop_cost = "high"
            last_recoverable_age = None
        return {
            "recoverable_artifact_on_disk": recoverable,
            "checkpoint_on_signal_supported_observed": checkpoint_supported_observed,
            "last_recoverable_artifact_age_sec": last_recoverable_age,
            "artifact_watch_pattern": watch_pattern,
            "stop_cost": stop_cost,
            "artifact_scope": str(artifact.get("artifact_scope") or ""),
            "stability": str(artifact.get("stability") or ""),
            "artifact_path": str(artifact.get("path") or ""),
            "artifact_size_bytes": int(artifact.get("size_bytes") or 0),
            "artifact_age_sec": age,
            "run_state_status": run_state_status,
            "safe_to_resume": artifact.get("safe_to_resume", "unknown"),
            "route_id": str(artifact.get("route_id") or ""),
            "fold": str(artifact.get("fold") or ""),
            "validation_protocol": str(artifact.get("validation_protocol") or ""),
            "checkpoint_kind": str(artifact.get("checkpoint_kind") or ""),
            "mergeable": artifact.get("mergeable", "unknown"),
            "observed_at_elapsed_sec": float(elapsed_sec or 0.0),
            "source": "artifact_watcher",
        }
