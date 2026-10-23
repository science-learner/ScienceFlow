# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Resource review-machine transitions and progress-signal state."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    PROGRESS_WINDOW,
    RESOURCE_PRESSURE,
    ROUTE_VALUE,
    ResourceJob,
    ResourceReviewEvent,
    ResourceReviewEventKind,
    ResourceReviewMachine,
    ResourceReviewState,
    STALL,
    TIMEBOX_EXPIRED,
    build_review_signal,
    classify_progress_signal,
    new_review_state,
    next_review_boundary,
)


class ReviewStateCallbacks:
    """Own ReviewStateCallbacks resource behavior without delegated forwarding."""

    def _review_machine_for(
        self,
        job_id: str,
        state: ResourceReviewState | None = None,
    ) -> ResourceReviewMachine:
        current = state or self._review_states.get(job_id) or new_review_state(job_id)
        machine = self._review_machines.get(job_id)
        if machine is None or machine.state != current:
            machine = ResourceReviewMachine(job_id=job_id, initial_state=current)
            self._review_machines[job_id] = machine
        return machine


    def _advance_resource_review_state(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> tuple[ResourceReviewState, Any | None]:
        state = self._review_states.get(job.job_id) or new_review_state(job.job_id)
        if state.heartbeat_index > 0 and float(elapsed_sec or 0.0) < float(state.last_advance_elapsed_sec or 0.0) + self.review_heartbeat_sec:
            signal["resource_review_state"] = state.to_json()
            signal["resource_efficiency"] = dict(job.resource_efficiency_state or {})
            return state, None
        self._update_resource_efficiency_state(job, signal, elapsed_sec=elapsed_sec)
        artifact_update = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        artifact_age = self._float_or_none(artifact_update.get("age_sec")) if artifact_update else self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec)
        artifact_recent_window = max(self.review_heartbeat_sec * 2.0, self.check_interval_sec * 2.0)
        artifact_recent = bool(artifact_update) and float(artifact_age or 0.0) <= artifact_recent_window
        recoverability = self._fresh_recoverability_for_job(job, elapsed_sec=elapsed_sec)
        recoverable_artifact_recent = bool(recoverability.get("recoverable_artifact_on_disk")) and float(artifact_age or 0.0) <= max(
            self.review_heartbeat_sec,
            self.check_interval_sec * 2.0,
        )
        value_artifact_recent = (
            bool(artifact_update)
            and artifact_recent
            and self._artifact_update_is_value_bearing(
                artifact_update,
                deliverable_validity=str(job.deliverable_validity or "none"),
            )
        )
        artifact_grew = bool(recoverable_artifact_recent or value_artifact_recent)
        progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured_progress = progress_payload.get("structured_progress") if isinstance(progress_payload.get("structured_progress"), dict) else {}
        progress_payload_elapsed = self._float_or_none(progress_payload.get("elapsed_sec"))
        structured_progress_advanced = bool(
            structured_progress.get("advanced")
            and progress_payload_elapsed is not None
            and progress_payload_elapsed > float(state.last_advance_elapsed_sec or 0.0)
        )
        signal["structured_progress"] = structured_progress
        signal["structured_progress_advanced"] = structured_progress_advanced
        stdout_observation = self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        signal["stdout_observation"] = stdout_observation
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        review_stdout_lines = int(signal.get("stdout_lines") or 0) if stream.get("fresh") else 0
        review_stdout_bytes = int(signal.get("stdout_bytes") or 0) if stream.get("fresh") else 0
        review_saw_training_progress = bool(signal.get("saw_training_progress") and stream.get("fresh"))
        metric_value_useful = self._metric_value_useful_from_signal(signal)
        if structured_progress and not structured_progress_advanced:
            review_saw_training_progress = False
        if metric_value_useful is False:
            review_saw_training_progress = False
        contention_context = self._contention_context_for_job(job)
        signal["contention_context"] = contention_context
        review_signal = build_review_signal(
            elapsed_sec=elapsed_sec,
            stdout_lines=review_stdout_lines,
            stdout_bytes=review_stdout_bytes,
            previous_stdout_bytes=int(state.last_stdout_bytes or 0),
            metric_history_text=str(signal.get("metric_history_text") or job.metric_history_text or ""),
            previous_metric_history_text=str(state.last_metric_history_text or ""),
            saw_training_progress=review_saw_training_progress,
            saw_final_score=bool(signal.get("saw_final_score")),
            terminal_signal_events=int(signal.get("terminal_signal_events") or 0),
            previous_terminal_signal_events=int(state.last_terminal_signal_events or 0),
            current_phase=str(signal.get("current_phase") or ""),
            process_tree_cpu=signal.get("process_tree_cpu") if isinstance(signal.get("process_tree_cpu"), dict) else {},
            gpu_active=self._last_gpu_sample_active(job),
            gpu_unknown=not bool(job.last_gpu_util_sample),
            artifact_recent=artifact_recent,
            artifact_grew=artifact_grew,
            structured_progress_advanced=structured_progress_advanced,
            metric_value_useful=metric_value_useful,
            metric_value_status=str((signal.get("resource_metric_value") or {}).get("status") or ""),
            metric_value_delta_to_best=self._float_or_none((signal.get("resource_metric_value") or {}).get("delta_to_best")),
            metric_scope_key=str((signal.get("resource_metric_value") or {}).get("metric_scope_key") or ""),
            active_waiter_pressure=bool(contention_context.get("active_waiter_pressure")),
            blocked_worker_count=int(contention_context.get("blocked_worker_count") or 0),
        )
        machine = self._review_machine_for(job.job_id, state)
        state = machine.advance(
            ResourceReviewEvent(
                kind=ResourceReviewEventKind.HEARTBEAT,
                signal=review_signal,
                config=self.review_config,
            ),
            event_id=f"{job.job_id}:heartbeat:{int(state.heartbeat_index) + 1}",
            expected_version=machine.version,
        ).current
        boundary = next_review_boundary(state, config=self.review_config)
        if boundary is not None:
            state = machine.advance(
                ResourceReviewEvent(
                    kind=ResourceReviewEventKind.REVIEW_EMITTED,
                    boundary=boundary,
                ),
                event_id=(
                    f"{job.job_id}:review:{state.heartbeat_index}:{boundary.kind}"
                ),
                expected_version=machine.version,
            ).current
        self._review_states[job.job_id] = state
        signal["resource_review_signal"] = review_signal.to_json()
        signal["resource_review_state"] = state.to_json()
        if boundary is not None:
            signal["resource_review_boundary"] = boundary.to_json()
            self._emit(
                "resource_review_boundary",
                job,
                status=boundary.kind,
                payload={
                    "boundary": boundary.to_json(),
                    "review_state": state.to_json(),
                    "review_signal": review_signal.to_json(),
                },
            )
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event(
                    "resource_review_boundary",
                    payload={
                        "job_id": job.job_id,
                        "boundary": boundary.to_json(),
                        "review_state": state.to_json(),
                        "review_signal": review_signal.to_json(),
                    },
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
        return state, boundary


    def _state_machine_review_decision(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        if not (self.review_state_enabled and self.kill_enabled and job.visible):
            return {"enabled": False, "reason": "review_state_disabled_or_job_not_visible"}
        state, boundary = self._advance_resource_review_state(job, signal, elapsed_sec=elapsed_sec)
        if boundary is None:
            return {"enabled": False, "reason": "no_review_boundary", "review_state": state.to_json()}
        reason_map = {
            STALL: "active_intervention:sm_stall",
            ROUTE_VALUE: "active_intervention:sm_route_value",
            PROGRESS_WINDOW: "active_intervention:sm_progress_window",
            TIMEBOX_EXPIRED: "active_intervention:sm_timebox_expired",
            RESOURCE_PRESSURE: "active_intervention:sm_resource_pressure",
        }
        feedback_map = {
            STALL: "RESOURCE_FEEDBACK: recommend_stop_command because the monitored command has no active work across the configured review window.\n",
            ROUTE_VALUE: "RESOURCE_FEEDBACK: recommend_stop_command because active long-running work has low recent quality gain; keep the best artifact and change the method, search space, schedule, validation target, or stopping condition before continuing.\n",
            PROGRESS_WINDOW: "RESOURCE_FEEDBACK: recommend_review_command because active long-running work needs a quality-gain review; keep the best artifact and change the method, search space, schedule, validation target, or stopping condition if continuing.\n",
            TIMEBOX_EXPIRED: "RESOURCE_FEEDBACK: recommend_stop_command because the previous observe/timebox window expired without satisfying the expected resource observation condition.\n",
            RESOURCE_PRESSURE: "RESOURCE_FEEDBACK: recommend_review_command because a new waiter/resource pressure event appeared during the observation window.\n",
        }
        reason = reason_map.get(boundary.kind, "active_intervention:sm_review")
        feedback = feedback_map.get(boundary.kind, "RESOURCE_FEEDBACK: recommend_review_command because the resource review state reached a decision boundary.\n")
        decision = {
            "enabled": True,
            "terminate": False,
            "would_terminate": True,
            "recommended_action": "stop_and_replan",
            "requires_llm_decision": True,
            "arbiter_enabled": bool(self.arbiter_enabled and self.arbiter_mode != "off"),
            "reason": reason,
            "check_interval_sec": self.check_interval_sec,
            "feedback": self._append_resource_intervention_summary(
                self._recommendation_feedback(reason=reason, feedback=feedback),
                action="RECOMMEND_STOP_AND_REPLAN",
                reason=reason,
            ),
            "resource_review_boundary": boundary.to_json(),
            "resource_review_state": state.to_json(),
            "resource_review_signal": signal.get("resource_review_signal") or {},
            "contention_context": signal.get("contention_context") or {},
            "metric_history_text": str(signal.get("metric_history_text") or job.metric_history_text or ""),
            "metric_history_line_count": int(signal.get("metric_history_line_count") or job.metric_history_line_count or 0),
            "resource_metric_value": dict(signal.get("resource_metric_value") or {}),
            "terminal_signal_events": signal.get("terminal_signal_events"),
            "terminal_signal_kind": signal.get("terminal_signal_kind"),
            "deadline_event": signal.get("deadline_event"),
            "deadline_remaining_sec": signal.get("deadline_remaining_sec"),
            "finalization_reserve_sec": signal.get("finalization_reserve_sec"),
        }
        self._record_kill_proposal_event(job, decision, signal, source="resource_review_state")
        return decision


    def _update_job_progress_signal(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed_sec)
        artifact_update = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        artifact_age = self._float_or_none(artifact_update.get("age_sec")) if artifact_update else self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec)
        stdout_observation = self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        signal["stdout_observation"] = stdout_observation
        classify_signal = dict(signal)
        progress_payload = job.last_progress if isinstance(job.last_progress, dict) else {}
        structured_progress = progress_payload.get("structured_progress") if isinstance(progress_payload.get("structured_progress"), dict) else {}
        structured_progress_advanced = bool(structured_progress.get("advanced"))
        classify_signal["structured_progress"] = structured_progress
        classify_signal["structured_progress_advanced"] = structured_progress_advanced
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        if not stream.get("fresh"):
            classify_signal["stdout_lines"] = 0
            classify_signal["stdout_bytes"] = 0
            classify_signal["saw_training_progress"] = False
        if structured_progress and not structured_progress_advanced:
            classify_signal["saw_training_progress"] = False
        if self._metric_value_useful_from_signal(signal) is False:
            classify_signal["saw_training_progress"] = False
        state = classify_progress_signal(
            classify_signal,
            progress_age_sec=progress_age,
            artifact_age_sec=float(artifact_age or 0.0),
            low_progress_warmup_sec=self.low_progress_warmup_sec,
            stalled_stdout_sec=self.stalled_stdout_sec,
            no_progress_sec=self.low_progress_no_heartbeat_sec,
            no_artifact_sec=self.low_progress_no_artifact_sec,
            idle_samples=job.idle_gpu_lease_samples,
            dataloader_samples=job.dataloader_bottleneck_samples,
            previous_signal=job.progress_signal,
            previous_windows=job.progress_signal_windows,
            min_confidence_windows=self.arbiter_min_progress_windows,
        )
        job.progress_signal = str(state.get("progress_signal") or "unknown")
        job.progress_signal_windows = int(state.get("progress_signal_windows") or 0)
        job.progress_signal_state = dict(state)
        signal.update({
            "progress_signal": job.progress_signal,
            "progress_confidence": state.get("progress_confidence"),
            "progress_signal_windows": job.progress_signal_windows,
            "progress_signal_reason": state.get("progress_signal_reason"),
            "multi_window_low_progress": bool(state.get("multi_window_low_progress")),
        })
        return state


    def _progress_confidence(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> str:
        recent = 0
        stdout_age = self._float_or_none(signal.get("stdout_age_sec"))
        stdout_observation = signal.get("stdout_observation") if isinstance(signal.get("stdout_observation"), dict) else self._stdout_observation_for_job(job, signal, elapsed_sec=elapsed_sec)
        stream = stdout_observation.get("stdout_stream") if isinstance(stdout_observation.get("stdout_stream"), dict) else {}
        if int(signal.get("stdout_lines") or 0) > 0 and bool(stream.get("fresh")) and (stdout_age is None or self.stalled_stdout_sec <= 0 or stdout_age < self.stalled_stdout_sec):
            recent += 1
        progress_age = self._elapsed_since_progress(job.last_progress, elapsed_sec=elapsed_sec)
        progress_limit = max(0.0, float(self.low_progress_no_heartbeat_sec or 0.0))
        if job.last_progress and (progress_limit <= 0 or progress_age < progress_limit):
            recent += 1
        artifact_age = self._elapsed_since_progress(job.last_artifact_progress, elapsed_sec=elapsed_sec)
        artifact_limit = max(0.0, float(self.low_progress_no_artifact_sec or 0.0))
        if job.last_artifact_progress and (artifact_limit <= 0 or artifact_age < artifact_limit):
            recent += 1
        if recent >= 2:
            return "high"
        if recent == 1:
            return "medium"
        return "low"
