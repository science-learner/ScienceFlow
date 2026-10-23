"""Resource observer responsibility: active intervention and EStra sidecar coordination.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""
from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    MAGENT_FORK_CONSIDERED,
    MAGENT_JOIN_PACKET_READY,
    MAGENT_SIDECAR_STARTED,
    MAGENT_TRAIN_OBSERVED,
    ResourceJob,
    fork_considered_payload,
    join_packet_ready_payload,
    run_cpu_sidecar_backfill,
    sidecar_started_payload,
    train_observed_payload,
    write_join_packet,
)


class InterventionCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    @staticmethod
    def _intervention_signal(*, elapsed_sec: float, stdout_age_sec: float, stdout_lines: int=0, stdout_bytes: int=0, metric_history_text: str='', metric_history_line_count: int=0, saw_training_progress: bool=False, saw_final_score: bool=False, current_phase: str='', process_tree_cpu: dict[str, Any] | None=None, invalid_metric_events: int=0, zero_score_events: int=0, last_invalid_metric_text: str='', last_zero_score_text: str='', terminal_signal_events: int=0, terminal_signal_kind: str='', last_terminal_signal_text: str='', deadline_event: bool=False, deadline_remaining_sec: float=0.0, finalization_reserve_sec: float=0.0) -> dict[str, Any]:
        return {'elapsed_sec': float(elapsed_sec or 0.0), 'stdout_age_sec': float(stdout_age_sec or 0.0), 'stdout_lines': int(stdout_lines or 0), 'stdout_bytes': int(stdout_bytes or 0), 'metric_history_text': str(metric_history_text or ''), 'metric_history_line_count': int(metric_history_line_count or 0), 'saw_training_progress': bool(saw_training_progress), 'saw_final_score': bool(saw_final_score), 'current_phase': str(current_phase or ''), 'process_tree_cpu': dict(process_tree_cpu or {}), 'invalid_metric_events': int(invalid_metric_events or 0), 'zero_score_events': int(zero_score_events or 0), 'last_invalid_metric_text': str(last_invalid_metric_text or ''), 'last_zero_score_text': str(last_zero_score_text or ''), 'terminal_signal_events': int(terminal_signal_events or 0), 'terminal_signal_kind': str(terminal_signal_kind or ''), 'last_terminal_signal_text': str(last_terminal_signal_text or ''), 'deadline_event': bool(deadline_event), 'deadline_remaining_sec': float(deadline_remaining_sec or 0.0), 'finalization_reserve_sec': float(finalization_reserve_sec or 0.0)}

    def active_intervention_decision(self, job_id: str | None, *, elapsed_sec: float, stdout_age_sec: float, stdout_lines: int=0, stdout_bytes: int=0, metric_history_text: str='', metric_history_line_count: int=0, saw_training_progress: bool=False, saw_final_score: bool=False, current_phase: str='', process_tree_cpu: dict[str, Any] | None=None, invalid_metric_events: int=0, zero_score_events: int=0, last_invalid_metric_text: str='', last_zero_score_text: str='', terminal_signal_events: int=0, terminal_signal_kind: str='', last_terminal_signal_text: str='', deadline_event: bool=False, deadline_remaining_sec: float=0.0, finalization_reserve_sec: float=0.0, **_: Any) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {'enabled': False}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        signal = self._intervention_signal(elapsed_sec=elapsed, stdout_age_sec=stdout_age_sec, stdout_lines=stdout_lines, stdout_bytes=stdout_bytes, metric_history_text=metric_history_text, metric_history_line_count=metric_history_line_count, saw_training_progress=saw_training_progress, saw_final_score=saw_final_score, current_phase=current_phase, process_tree_cpu=process_tree_cpu, invalid_metric_events=invalid_metric_events, zero_score_events=zero_score_events, last_invalid_metric_text=last_invalid_metric_text, last_zero_score_text=last_zero_score_text, terminal_signal_events=terminal_signal_events, terminal_signal_kind=terminal_signal_kind, last_terminal_signal_text=last_terminal_signal_text, deadline_event=deadline_event, deadline_remaining_sec=deadline_remaining_sec, finalization_reserve_sec=finalization_reserve_sec)
        self._update_intervention_observation(job, signal, elapsed=elapsed)
        shared_result = self._shared_intervention_result(job, signal)
        if shared_result is not None:
            return shared_result
        deliverable_guard = self._deliverable_completion_guard_decision(job, signal)
        initial = self._initial_intervention_result(job, signal, elapsed=elapsed)
        if initial.get('direct') is not None:
            return initial['direct']
        terminate, reason, feedback = bool(initial.get('terminate')), str(initial.get('reason') or ''), str(initial.get('feedback') or '')
        metric_guard = initial['metric_guard']
        guard_result = self._resource_guard_intervention(job, signal, elapsed=elapsed, already_terminating=terminate)
        idle_guard, dataloader_guard, progress_guard = guard_result['idle'], guard_result['dataloader'], guard_result['progress']
        if not terminate and guard_result.get('terminate'):
            terminate, reason, feedback = True, str(guard_result.get('reason') or ''), str(guard_result.get('feedback') or '')
        decision = self._apply_kill_approval_gate({'enabled': bool(job.visible or deliverable_guard.get('terminate') or metric_guard.get('terminate') or idle_guard.get('terminate') or idle_guard.get('release_idle_lease') or dataloader_guard.get('terminate') or progress_guard.get('terminate')), 'terminate': terminate, 'reason': reason, 'check_interval_sec': self.check_interval_sec, 'feedback': feedback, 'progress_age_sec': progress_guard.get('progress_age_sec'), 'artifact_age_sec': progress_guard.get('artifact_age_sec'), 'idle_gpu_lease_samples': idle_guard.get('idle_samples'), 'idle_gpu_pressure_reason': idle_guard.get('pressure_reason'), 'idle_gpu_release_candidate': idle_guard.get('release_candidate'), 'idle_gpu_release_action': idle_guard.get('release_action'), 'idle_gpu_release_mode': idle_guard.get('release_mode'), 'idle_gpu_release_safety': idle_guard.get('release_safety'), 'release_idle_lease': idle_guard.get('release_idle_lease'), 'dataloader_low_compute_samples': dataloader_guard.get('dataloader_low_compute_samples'), 'dataloader_child_cpu_pct': dataloader_guard.get('child_cpu_pct'), 'dataloader_busy_child_count': dataloader_guard.get('busy_child_count'), 'route_viability': dict(job.route_viability_state or {}), 'active_lease_suspect': dict(job.active_lease_suspect_state or {}), 'terminal_signal_events': signal.get('terminal_signal_events'), 'terminal_signal_kind': signal.get('terminal_signal_kind'), 'last_terminal_signal_text': signal.get('last_terminal_signal_text'), 'deadline_event': signal.get('deadline_event'), 'deadline_remaining_sec': signal.get('deadline_remaining_sec'), 'finalization_reserve_sec': signal.get('finalization_reserve_sec'), 'invalid_metric_events': metric_guard.get('invalid_metric_events'), 'zero_score_events': metric_guard.get('zero_score_events'), 'last_invalid_metric_text': metric_guard.get('last_invalid_metric_text'), 'last_zero_score_text': metric_guard.get('last_zero_score_text')})
        if decision.get('release_idle_lease') and decision.get('feedback'):
            decision['feedback'] = self._append_resource_intervention_summary(str(decision.get('feedback') or ''), action='RELEASE_IDLE_LEASE', reason=str(decision.get('reason') or 'idle_gpu_lease'))
        self._record_kill_proposal_event(job, decision, signal, source='active_intervention')
        decision.setdefault('terminate', False)
        decision.setdefault('would_terminate', False)
        if not decision.get('terminate') and (not decision.get('would_terminate')):
            review = self._followup_intervention_review(job, signal)
            if review is not None:
                return review
        return decision

    def _update_intervention_observation(self, job: ResourceJob, signal: dict[str, Any], *, elapsed: float) -> None:
        if elapsed >= self.min_register_sec:
            self._promote(job, reason='elapsed_threshold', elapsed_sec=elapsed)
        self._maybe_emit_gpu_util_sample(job, elapsed_sec=elapsed)
        self._attach_resource_metric_value_assessment(job, signal)
        self._update_research_route_metric(job, signal)
        self._update_job_progress_signal(job, signal, elapsed_sec=elapsed)
        self._update_execution_value_shadow(job, signal, elapsed_sec=elapsed)
        job.last_signal = signal
        self._maybe_emit_monitor_agent_shadow(job, signal)

    def _shared_intervention_result(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any] | None:
        revoked = self._shared_secondary_revoked_decision(job)
        if revoked.get('terminate'):
            return revoked
        runtime = self._shared_runtime_review_decision(job, signal)
        if isinstance(runtime, dict) and runtime.get('action') == 'STOP_SECONDARY_SHARED_JOB':
            return runtime
        return None

    def _initial_intervention_result(self, job: ResourceJob, signal: dict[str, Any], *, elapsed: float) -> dict[str, Any]:
        metric_guard = self._metric_health_guard_decision(job, signal)
        if metric_guard.get('terminate'):
            return {'direct': None, 'terminate': True, 'reason': str(metric_guard.get('reason') or 'active_intervention:invalid_training_metrics'), 'feedback': str(metric_guard.get('feedback') or ''), 'metric_guard': metric_guard}
        if self.review_state_enabled:
            state_decision = self._state_machine_review_decision(job, signal, elapsed_sec=elapsed)
            if isinstance(state_decision, dict) and state_decision.get('enabled'):
                return {'direct': state_decision, 'terminate': False, 'reason': '', 'feedback': '', 'metric_guard': metric_guard}
        terminate = bool((not self.review_state_enabled) and self.kill_enabled and job.visible and (self.stalled_stdout_sec > 0) and (signal['stdout_age_sec'] >= self.stalled_stdout_sec) and (not signal['saw_final_score']))
        return {'direct': None, 'terminate': terminate, 'reason': 'active_intervention:stalled_or_low_signal' if terminate else '', 'feedback': 'RESOURCE_FEEDBACK: terminated_low_signal_command because the long-running command stopped producing useful output.\n' if terminate else '', 'metric_guard': metric_guard}

    def _resource_guard_intervention(self, job: ResourceJob, signal: dict[str, Any], *, elapsed: float, already_terminating: bool) -> dict[str, Any]:
        idle_guard = self._idle_gpu_lease_guard_decision(job, signal)
        dataloader_guard = self._dataloader_bottleneck_guard_decision(job, signal)
        progress_guard = self._low_progress_guard_decision(job, signal)
        result = {'terminate': False, 'reason': '', 'feedback': '', 'idle': idle_guard, 'dataloader': dataloader_guard, 'progress': progress_guard}
        if already_terminating:
            return result
        for guard, fallback in ((idle_guard, 'active_intervention:idle_gpu_lease_under_pressure'), (dataloader_guard, 'active_intervention:dataloader_bottleneck_low_gpu_high_cpu'), (progress_guard, 'active_intervention:low_progress_heartbeat_stalled')):
            if guard.get('terminate'):
                return {**result, 'terminate': True, 'reason': str(guard.get('reason') or fallback), 'feedback': str(guard.get('feedback') or '')}
        if not (signal.get('deadline_event') and job.visible and self._job_is_expensive_resource_hold(job)):
            return result
        finish_feasibility = self._progress_finish_feasibility(job, signal, elapsed_sec=elapsed)
        feedback = 'RESOURCE_FEEDBACK: active_but_unaffordable because the command is still producing progress, but observed ETA exceeds remaining useful budget and no recoverable artifact/metric/submission is guaranteed; prefer smaller scope, cached artifacts, blend/finalize, or stop and replan.\n' if finish_feasibility.get('finish_feasible') is False else 'RESOURCE_FEEDBACK: recommend_stop_command because task finalization reserve has started; arbiter should prefer stopping work that cannot finish before deadline based on progress and efficiency evidence.\n'
        return {**result, 'terminate': True, 'reason': 'active_intervention:deadline_finalization_reserve', 'feedback': feedback}

    def _followup_intervention_review(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any] | None:
        for decider in (self._quick_probe_review_decision, self._contention_review_decision, self._research_cadence_review_decision, self._periodic_efficiency_review_decision):
            review = decider(job, signal)
            if isinstance(review, dict) and review.get('enabled'):
                return review
        share_review = self._gpu_share_phase_a_decision(job, signal)
        if not isinstance(share_review, dict):
            return None
        shared_grant = share_review.get('shared_lease_grant')
        if share_review.get('arbiter_review') or (isinstance(shared_grant, dict) and shared_grant.get('granted')):
            return share_review
        return None

    def post_arbiter_continue_review(self, job_id: str | None, *, arbiter_action: str='', elapsed_sec: float, stdout_age_sec: float, stdout_lines: int=0, stdout_bytes: int=0, metric_history_text: str='', metric_history_line_count: int=0, saw_training_progress: bool=False, saw_final_score: bool=False, current_phase: str='', process_tree_cpu: dict[str, Any] | None=None, invalid_metric_events: int=0, zero_score_events: int=0, last_invalid_metric_text: str='', last_zero_score_text: str='', terminal_signal_events: int=0, terminal_signal_kind: str='', last_terminal_signal_text: str='', deadline_event: bool=False, deadline_remaining_sec: float=0.0, finalization_reserve_sec: float=0.0, **_: Any) -> dict[str, Any]:
        action = str(arbiter_action or '').upper()
        if action not in {'CONTINUE', 'DENY_KILL', 'OBSERVE_MORE', 'MARK_STALLED_NO_KILL'}:
            return {'enabled': False, 'reason': 'arbiter_action_not_share_safe', 'action': action}
        if not job_id or job_id not in self._jobs:
            return {'enabled': False, 'reason': 'job_not_found', 'action': action}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        signal = self._intervention_signal(elapsed_sec=elapsed, stdout_age_sec=stdout_age_sec, stdout_lines=stdout_lines, stdout_bytes=stdout_bytes, metric_history_text=metric_history_text, metric_history_line_count=metric_history_line_count, saw_training_progress=saw_training_progress, saw_final_score=saw_final_score, current_phase=current_phase, process_tree_cpu=process_tree_cpu, invalid_metric_events=invalid_metric_events, zero_score_events=zero_score_events, last_invalid_metric_text=last_invalid_metric_text, last_zero_score_text=last_zero_score_text, terminal_signal_events=terminal_signal_events, terminal_signal_kind=terminal_signal_kind, last_terminal_signal_text=last_terminal_signal_text, deadline_event=deadline_event, deadline_remaining_sec=deadline_remaining_sec, finalization_reserve_sec=finalization_reserve_sec)
        self._maybe_emit_gpu_util_sample(job, elapsed_sec=elapsed)
        self._attach_resource_metric_value_assessment(job, signal)
        self._update_research_route_metric(job, signal)
        self._update_job_progress_signal(job, signal, elapsed_sec=elapsed)
        self._update_execution_value_shadow(job, signal, elapsed_sec=elapsed)
        job.last_signal = signal
        share_safe = action in {'CONTINUE', 'DENY_KILL'}
        out = self._gpu_share_phase_a_decision(job, signal, allow_grant=share_safe, allow_handoff=share_safe)
        if isinstance(out, dict) and out.get('enabled'):
            out['post_arbiter_action'] = action
            if not share_safe:
                out['observe_only_after_arbiter'] = True
        return out

    def _magent_resource_mode_for_job(self, job: ResourceJob) -> str:
        if self.resource_runtime is None or not job.gpu_ids:
            return 'GREEN'
        try:
            snapshot = self.resource_runtime.pressure_snapshot(gpu_ids=job.gpu_ids)
        except Exception:
            return 'UNKNOWN'
        gpus = snapshot.get('gpus') if isinstance(snapshot.get('gpus'), dict) else {}
        modes = {str(row.get('mode') or '').upper() for row in gpus.values() if isinstance(row, dict)}
        if 'RED' in modes:
            return 'RED'
        if 'YELLOW' in modes:
            return 'YELLOW'
        if modes:
            return 'GREEN'
        return 'UNKNOWN'

    def _record_magent_train_observed_once(self, job_id: str, job: ResourceJob, *, elapsed_sec: float, parent_state: str) -> None:
        if not self.estra_magent.enabled or self.resource_runtime is None:
            return
        if job_id in self._magent_observed_jobs:
            return
        if float(elapsed_sec or 0.0) < self.estra_magent.min_parent_runtime_sec:
            return
        self._magent_observed_jobs.add(job_id)
        self.resource_runtime.record_resource_event(MAGENT_TRAIN_OBSERVED, payload=train_observed_payload(worker_id=self.worker_id, parent_job_id=job_id, parent_state=parent_state, elapsed_sec=float(elapsed_sec or 0.0), resource_class=job.resource_class, resource_mode=self._magent_resource_mode_for_job(job)), command_id=job_id, lease_id=job_id)

    def _record_magent_fork_considered_once(self, job_id: str, *, decision: str, reason: str, task_type: str='submission_checker') -> None:
        if not self.estra_magent.enabled or self.resource_runtime is None:
            return
        if job_id in self._magent_considered_jobs:
            return
        self._magent_considered_jobs.add(job_id)
        self.resource_runtime.record_resource_event(MAGENT_FORK_CONSIDERED, payload=fork_considered_payload(worker_id=self.worker_id, parent_job_id=job_id, decision=decision, reason=reason, task_type=task_type, sidecar_mode=self.estra_magent.sidecar_mode), command_id=job_id, lease_id=job_id)

    def maybe_run_sidecar_backfill(self, job_id: str | None, *, elapsed_sec: float, parent_state: str='healthy_running') -> dict[str, Any]:
        if self.resource_runtime is None:
            reason = 'sidecar_disabled' if not self.sidecar_enabled and (not self.estra_magent.enabled) else 'resource_runtime_missing'
            return {'started': False, 'reason': reason}
        if not job_id or job_id not in self._jobs:
            return {'started': False, 'reason': 'job_missing'}
        job = self._jobs[job_id]
        elapsed = float(elapsed_sec or 0.0)
        self._record_magent_train_observed_once(job_id, job, elapsed_sec=elapsed, parent_state=parent_state)
        if not self.sidecar_enabled:
            if elapsed >= self.estra_magent.min_parent_runtime_sec:
                self._record_magent_fork_considered_once(job_id, decision='skip', reason='sidecar_disabled')
            return {'started': False, 'reason': 'sidecar_disabled'}
        if job_id in self._sidecar_jobs_started:
            return {'started': False, 'reason': 'already_started'}
        if elapsed < self.sidecar_min_parent_runtime_sec:
            return {'started': False, 'reason': 'parent_runtime_too_short'}
        if not job.workspace_dir:
            self._record_magent_fork_considered_once(job_id, decision='skip', reason='workspace_missing')
            return {'started': False, 'reason': 'workspace_missing'}
        if self.estra_magent.enabled and (not self.estra_magent.sidecar_allowed()) and (not self.sidecar_enabled):
            self._record_magent_fork_considered_once(job_id, decision='skip', reason='sidecar_mode_disabled')
            return {'started': False, 'reason': 'sidecar_mode_disabled'}
        self._sidecar_jobs_started.add(job_id)
        self._record_magent_fork_considered_once(job_id, decision='fork', reason='long_train_with_submission_readiness_gap')
        self.resource_runtime.record_resource_event('sidecar_fork_considered', payload={'parent_job_id': job_id, 'parent_state': parent_state, 'elapsed_sec': elapsed}, command_id=job_id, lease_id=job_id)
        try:
            report = run_cpu_sidecar_backfill(resource_dir=self.resource_runtime.resource_dir, workspace_dir=job.workspace_dir, parent_job_id=job_id, parent_state=parent_state, elapsed_sec=elapsed, parent_worker_id=self.worker_id, task_type='submission_checker', budget_sec=self.estra_magent.budget_sec)
        except Exception as exc:
            self.resource_runtime.record_resource_event('sidecar_discarded', payload={'parent_job_id': job_id, 'reason': type(exc).__name__}, command_id=job_id, lease_id=job_id)
            return {'started': False, 'reason': type(exc).__name__}
        sidecar_id = str(report.get('sidecar_id') or '')
        self.resource_runtime.record_resource_event('sidecar_forked', payload={'parent_job_id': job_id, 'sidecar_id': sidecar_id, 'report_path': report.get('report_path')}, command_id=job_id, lease_id=job_id)
        if self.estra_magent.enabled:
            self.resource_runtime.record_resource_event(MAGENT_SIDECAR_STARTED, payload=sidecar_started_payload(sidecar_id=sidecar_id, parent_worker_id=self.worker_id, parent_job_id=job_id, budget_sec=self.estra_magent.budget_sec, task_type=str(report.get('task_type') or 'submission_checker'), workspace=f'sidecars/{sidecar_id}/workspace' if sidecar_id else ''), command_id=job_id, lease_id=job_id)
        join_packet = None
        if self.estra_magent.enabled:
            join_packet = write_join_packet(self.resource_runtime.resource_dir, report, parent_worker_id=self.worker_id, inject_parent=self.estra_magent.join_inject_parent, inject_estra=self.estra_magent.join_inject_estra, inject_resource_context=self.estra_magent.join_inject_resource_context)
            if join_packet:
                inject = join_packet.get('inject') if isinstance(join_packet.get('inject'), dict) else {}
                inject_targets = [name for (name, enabled) in inject.items() if enabled]
                self.resource_runtime.record_resource_event(MAGENT_JOIN_PACKET_READY, payload=join_packet_ready_payload(sidecar_id=sidecar_id, parent_worker_id=self.worker_id, quality_gate=str(join_packet.get('quality_gate') or ''), inject_targets=inject_targets, artifact_count=len(join_packet.get('artifact_refs') or [])), command_id=job_id, lease_id=job_id)
        self.resource_runtime.record_resource_event('sidecar_joined', payload={'parent_job_id': job_id, 'sidecar_id': sidecar_id, 'quality_gate': report.get('quality_gate'), 'observations': report.get('observations'), 'recommended_next_action': report.get('recommended_next_action'), 'useful_artifacts': report.get('useful_artifacts'), 'join_packet_path': (join_packet or {}).get('packet_path') if isinstance(join_packet, dict) else ''}, command_id=job_id, lease_id=job_id)
        return {'started': True, 'report': report, 'join_packet': join_packet}
