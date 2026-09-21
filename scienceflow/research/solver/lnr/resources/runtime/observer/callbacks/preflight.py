"""Resource observer responsibility: resource preflight and observation callback projection.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""
from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_HEAVY_CPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_LIGHT_CPU,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_READONLY_CPU,
    RESOURCE_UNKNOWN_EXEC,
    RESOURCE_UNKNOWN_GPU_EXEC,
    ResourceSourceHint,
    TRAIN_CLASSES,
    TT_ALLOWED_AFTER_TIMEOUT,
    replace,
)


class PreflightCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def resource_preflight_decision(self, job_id: str | None, *, inferred_class: str, gpu_ids: list[str] | None=None, resource_context_version: int | None=None, resource_pressure_generation: int | None=None, **_: Any) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs:
            return {'allowed': True}
        job = self._jobs[job_id]
        incoming = str(inferred_class or job.resource_class)
        resource_class = self._merge_resource_class(job.resource_class, incoming, source_hint=job.source_hint)
        if bool(getattr(job.source_hint, 'command_cpu_only', False)):
            ids = []
        else:
            ids = [str(x) for x in gpu_ids or job.gpu_ids or [] if str(x).strip()]
        job.resource_class = resource_class
        if ids:
            job.gpu_ids = ids
        elif bool(getattr(job.source_hint, 'command_cpu_only', False)):
            job.gpu_ids = []
        job.gpu_queue_relevant = self._gpu_queue_relevant(job.resource_class, job.source_hint, job.gpu_ids)
        deliverable_gate = self._completed_deliverable_preflight_gate(job)
        if deliverable_gate.get('blocked'):
            return {'allowed': False, 'resource_class': job.resource_class, 'status': str(deliverable_gate.get('status') or 'DENIED_REPLAN'), 'reason': str(deliverable_gate.get('reason') or 'invalid_deliverable_schema_preflight'), 'feedback': str(deliverable_gate.get('feedback') or ''), 'feedback_suppressed': bool(deliverable_gate.get('feedback_suppressed')), 'feedback_state_key': deliverable_gate.get('feedback_state_key'), 'resource_feedback_repeated_count': deliverable_gate.get('resource_feedback_repeated_count'), 'error': 'Resource policy blocked command because deliverable schema is invalid', 'allowed_classes': deliverable_gate.get('allowed_classes') or [], 'deliverable_validity': deliverable_gate.get('deliverable_validity'), 'deliverable_completion_state': deliverable_gate.get('deliverable_completion_state') or {}}
        context_is_stale, current_pressure_generation = self._resource_context_facts(
            resource_context_version=resource_context_version,
            resource_pressure_generation=resource_pressure_generation,
        )
        result = self._pressure_preflight_result(
            job,
            context_is_stale=context_is_stale,
            resource_context_version=resource_context_version,
            resource_pressure_generation=resource_pressure_generation,
            current_pressure_generation=current_pressure_generation,
        )
        if result is not None:
            return result
        result = self._post_feedback_preflight_result(job)
        if result is not None:
            return result
        result = self._active_guard_preflight_result(job)
        if result is not None:
            return result
        return self._timeout_preflight_result(job)

    def _resource_context_facts(self, *, resource_context_version: int | None, resource_pressure_generation: int | None) -> tuple[bool, int | None]:
        context_is_stale = False
        if resource_context_version is not None:
            try:
                context_is_stale = int(resource_context_version) < int(getattr(self.state_machine, 'event_count', 0) or 0)
            except (TypeError, ValueError):
                context_is_stale = False
        current_pressure_generation: int | None = None
        if self.resource_runtime is not None:
            try:
                current_pressure_generation = int(self.resource_runtime.pressure_generation() or 0)
            except (TypeError, ValueError):
                current_pressure_generation = None
        if resource_pressure_generation is not None and current_pressure_generation is not None:
            try:
                context_is_stale = context_is_stale or int(resource_pressure_generation) < current_pressure_generation
            except (TypeError, ValueError):
                pass
        return context_is_stale, current_pressure_generation

    def _pressure_preflight_result(self, job: Any, *, context_is_stale: bool, resource_context_version: int | None, resource_pressure_generation: int | None, current_pressure_generation: int | None) -> dict[str, Any] | None:
        if not (self.timeout_hard_gate_enabled and job.gpu_queue_relevant and self.resource_runtime is not None):
            return None
        pressure = self.resource_runtime.pressure_gate_decision(resource_class=job.resource_class, gpu_ids=job.gpu_ids, command_digest=job.command_digest)
        if not isinstance(pressure, dict) or not pressure.get('blocked'):
            return None
        gate = 'stale_resource_context' if context_is_stale else str(pressure.get('reason') or 'block_train_after_gpu_pressure')
        status = str(pressure.get('status') or 'DENIED_REPLAN')
        red_gpu_ids = [str(x) for x in pressure.get('red_gpu_ids') or pressure.get('gpu_ids') or [] if str(x).strip()]
        allowed_classes = [str(x) for x in pressure.get('allowed_classes') or [] if str(x).strip()]
        bypass = self._allowed_class_bypass_result(job, reason=gate, scope='per_gpu', resource_mode=str(pressure.get('resource_mode') or 'RED'), allowed_classes=allowed_classes)
        if bypass is not None:
            return bypass
        pressure_snapshot = pressure.get('pressure') if isinstance(pressure.get('pressure'), dict) else {}
        digest_pressure: dict[str, Any] = {}
        if gate != 'duplicate_digest_cooldown':
            digest_pressure = self.resource_runtime.record_policy_gate_digest(job_id=job.job_id, resource_class=job.resource_class, gpu_ids=job.gpu_ids, command_digest=job.command_digest, reason=gate)
        facts = {'pressure': pressure_snapshot, 'digest_pressure': digest_pressure, 'resource_context_version': resource_context_version, 'resource_pressure_generation': resource_pressure_generation, 'current_pressure_generation': current_pressure_generation}
        trial = self._startup_trial_decision(job, reason=gate, status=status, scope='per_gpu', resource_mode=str(pressure.get('resource_mode') or 'RED'), allowed_classes=allowed_classes, gpu_ids=red_gpu_ids, extra_facts=facts)
        if trial is not None:
            return trial
        if self._pressure_observe_first_eligible(job, gate=gate, pressure=pressure, context_is_stale=context_is_stale):
            return self._install_pressure_observe_first(job, gate=gate, status=status, pressure=pressure, red_gpu_ids=red_gpu_ids, allowed_classes=allowed_classes, resource_context_version=resource_context_version, resource_pressure_generation=resource_pressure_generation, current_pressure_generation=current_pressure_generation, pressure_snapshot=pressure_snapshot, digest_pressure=digest_pressure)
        feedback = self._resource_feedback_text(status=status, reason=gate, scope='per_gpu', resource_mode=str(pressure.get('resource_mode') or 'RED'), blocked_class=job.resource_class, gpu_ids=red_gpu_ids, allowed_classes=allowed_classes, cooldown_sec=float(pressure.get('cooldown_remaining_sec') or 0.0), eta_next_train_sec=float(pressure.get('eta_next_train_sec') or 0.0), eta_confidence=str(pressure.get('eta_confidence') or 'low'), unlock_condition='resource_pressure_cleared', blocked_until_unlock=True, pressure_generation=int(pressure_snapshot.get('generation') or pressure.get('generation') or 0), duplicate_digest_count=int(pressure.get('duplicate_digest_count') or 0) if pressure.get('duplicate_digest_count') is not None else None)
        if self._soft_gate_admission_enabled(reason=gate):
            self._emit('resource_policy_gate', job, status='review', payload={'status': status, 'reason': gate, 'scope': 'per_gpu', 'resource_mode': str(pressure.get('resource_mode') or 'RED'), 'blocked_class': job.resource_class, 'allowed_classes': allowed_classes, 'requires_admission_review': True, 'cooldown_sec': float(pressure.get('cooldown_remaining_sec') or 0.0), 'eta_next_train_sec': float(pressure.get('eta_next_train_sec') or 0.0), 'eta_confidence': str(pressure.get('eta_confidence') or ''), **facts, 'red_gpu_ids': red_gpu_ids})
            return self._soft_gate_admission_review_result(job, status=status, reason=gate, scope='per_gpu', resource_mode=str(pressure.get('resource_mode') or 'RED'), allowed_classes=allowed_classes, feedback=feedback, eta_next_train_sec=float(pressure.get('eta_next_train_sec') or 0.0), eta_confidence=str(pressure.get('eta_confidence') or 'low'), cooldown_sec=float(pressure.get('cooldown_remaining_sec') or 0.0), unlock_condition='resource_pressure_cleared', blocked_until_unlock=True, extra_facts={**facts, 'red_gpu_ids': red_gpu_ids, 'duplicate_digest_count': pressure.get('duplicate_digest_count')})
        self._remember_resource_feedback(job, status=status, reason=gate, resource_mode=str(pressure.get('resource_mode') or 'RED'), blocked_class=job.resource_class, allowed_classes=allowed_classes, eta_next_train_sec=float(pressure.get('eta_next_train_sec') or 0.0), cooldown_sec=float(pressure.get('cooldown_remaining_sec') or 0.0))
        self._emit('resource_policy_gate', job, status='blocked', payload={'status': status, 'reason': gate, 'scope': 'per_gpu', 'resource_mode': str(pressure.get('resource_mode') or 'RED'), 'gpu_queue_timeout_count': int(pressure.get('max_queue_timeout_count') or 0), 'blocked_class': job.resource_class, 'allowed_after_gate': str(pressure.get('allowed_after_gate') or ''), 'allowed_classes': allowed_classes, 'cooldown_sec': float(pressure.get('cooldown_remaining_sec') or 0.0), 'eta_next_train_sec': float(pressure.get('eta_next_train_sec') or 0.0), 'eta_confidence': str(pressure.get('eta_confidence') or ''), 'duplicate_digest_count': pressure.get('duplicate_digest_count'), **facts, 'red_gpu_ids': red_gpu_ids})
        return {'allowed': False, 'resource_class': job.resource_class, 'status': status, 'reason': gate, 'feedback': feedback, 'error': 'Resource policy blocked command after GPU pressure update'}

    def _post_feedback_preflight_result(self, job: Any) -> dict[str, Any] | None:
        if not job.post_feedback_gate:
            return None
        gate = dict(job.post_feedback_gate or {})
        source_gpu_ids = {str(x) for x in gate.get('source_gpu_ids') or [] if str(x).strip()}
        current_gpu_ids = {str(x) for x in job.gpu_ids or [] if str(x).strip()}
        same_feedback_scope = not source_gpu_ids or not current_gpu_ids or bool(source_gpu_ids & current_gpu_ids)
        if not same_feedback_scope:
            return {'allowed': True, 'resource_class': job.resource_class}
        if self._soft_gpu_gate_can_clear_for_available_slot(resource_class=job.resource_class, gpu_ids=job.gpu_ids, guard=gate):
            job.post_feedback_gate = {}
            self._emit('resource_soft_gate_cleared', job, status='allowed', payload={'reason': 'observed_free_gpu_cleared_post_feedback_gate', 'scope': 'worker_plan', 'resource_class': job.resource_class, 'gpu_ids': job.gpu_ids, 'post_feedback': gate, 'resource_control_profile': self.resource_control_profile})
            return {'allowed': True, 'resource_class': job.resource_class, 'reason': 'observed_free_gpu_cleared_post_feedback_gate'}
        allowed_classes = [str(x) for x in gate.get('allowed_classes') or [] if str(x).strip()]
        resource_mode = str(gate.get('source_resource_mode') or 'YELLOW')
        status = 'DENIED_REPLAN'
        reason = 'post_feedback_blocked_class'
        bypass = self._allowed_class_bypass_result(job, reason=reason, scope='worker_plan', resource_mode=resource_mode, allowed_classes=allowed_classes)
        if bypass is not None:
            return bypass
        source_reason = str(gate.get('source_reason') or '')
        trial = self._startup_trial_decision(job, reason=reason, status=status, scope='worker_plan', resource_mode=resource_mode, allowed_classes=allowed_classes, gpu_ids=job.gpu_ids, source_reason=source_reason, extra_facts={'post_feedback': gate})
        if trial is not None:
            return trial
        raw_feedback = self._resource_feedback_text(status=status, reason=reason, scope='worker_plan', resource_mode=resource_mode, blocked_class=job.resource_class, gpu_ids=job.gpu_ids, allowed_classes=allowed_classes)
        feedback_state = self._dedupe_resource_feedback_for_agent(job, feedback=raw_feedback, status=status, reason=reason, scope='worker_plan', resource_mode=resource_mode, blocked_class=job.resource_class, gpu_ids=job.gpu_ids, allowed_classes=allowed_classes)
        feedback = str(feedback_state.get('feedback') or '')
        if self._soft_gate_admission_enabled(reason=reason, source_reason=source_reason):
            self._emit('resource_planner_guard', job, status='review', payload={'status': status, 'reason': reason, 'scope': 'worker_plan', 'resource_mode': resource_mode, 'blocked_class': job.resource_class, 'allowed_classes': allowed_classes, 'post_feedback': gate, 'requires_admission_review': True, 'feedback_state_key': feedback_state.get('feedback_state_key'), 'feedback_suppressed': bool(feedback_state.get('feedback_suppressed')), 'resource_feedback_repeated_count': feedback_state.get('repeated_count')})
            return self._soft_gate_admission_review_result(job, status=status, reason=reason, scope='worker_plan', resource_mode=resource_mode, allowed_classes=allowed_classes, feedback=feedback, feedback_state=feedback_state, unlock_condition='resource_context_changed', blocked_until_unlock=True, extra_facts={'post_feedback': gate})
        self._remember_resource_feedback(job, status=status, reason=reason, resource_mode=resource_mode, blocked_class=job.resource_class, allowed_classes=allowed_classes)
        self._emit('resource_planner_guard', job, status='blocked', payload={'status': status, 'reason': reason, 'scope': 'worker_plan', 'resource_mode': resource_mode, 'blocked_class': job.resource_class, 'allowed_classes': allowed_classes, 'post_feedback': gate, 'feedback_state_key': feedback_state.get('feedback_state_key'), 'feedback_suppressed': bool(feedback_state.get('feedback_suppressed')), 'resource_feedback_repeated_count': feedback_state.get('repeated_count')})
        return {'allowed': False, 'resource_class': job.resource_class, 'status': status, 'reason': reason, 'feedback': feedback, 'feedback_suppressed': bool(feedback_state.get('feedback_suppressed')), 'feedback_state_key': feedback_state.get('feedback_state_key'), 'resource_feedback_repeated_count': feedback_state.get('repeated_count'), 'error': 'Resource planner guard blocked command after RESOURCE_FEEDBACK'}

    def _active_guard_preflight_result(self, job: Any) -> dict[str, Any] | None:
        active_guard = self._active_plan_guard_decision(job)
        if not active_guard.get('blocked'):
            return None
        guard = active_guard.get('guard') if isinstance(active_guard.get('guard'), dict) else {}
        allowed_classes = [str(x) for x in guard.get('allowed_classes') or [] if str(x).strip()]
        resource_mode = str(guard.get('resource_mode') or 'YELLOW')
        status = 'DENIED_REPLAN'
        reason = 'active_resource_plan_guard'
        bypass = self._allowed_class_bypass_result(job, reason=reason, scope='worker_plan', resource_mode=resource_mode, allowed_classes=allowed_classes)
        if bypass is not None:
            return bypass
        source_reason = str(guard.get('source_reason') or '')
        trial = self._startup_trial_decision(job, reason=reason, status=status, scope='worker_plan', resource_mode=resource_mode, allowed_classes=allowed_classes, gpu_ids=job.gpu_ids, source_reason=source_reason, extra_facts={'guard_id': str(active_guard.get('guard_id') or ''), 'current_pressure_generation': active_guard.get('current_pressure_generation'), 'source_feedback': guard})
        if trial is not None:
            return trial
        raw_feedback = self._resource_feedback_text(status=status, reason=reason, scope='worker_plan', resource_mode=resource_mode, blocked_class=job.resource_class, gpu_ids=job.gpu_ids, allowed_classes=allowed_classes, eta_next_train_sec=float(active_guard.get('remaining_sec') or 0.0), eta_confidence='low', unlock_condition='resource_context_changed', blocked_until_unlock=True, pressure_generation=int(active_guard.get('current_pressure_generation') or 0))
        feedback_state = self._dedupe_resource_feedback_for_agent(job, feedback=raw_feedback, status=status, reason=reason, scope='worker_plan', resource_mode=resource_mode, blocked_class=job.resource_class, gpu_ids=job.gpu_ids, allowed_classes=allowed_classes, unlock_condition='resource_context_changed', blocked_until_unlock=True)
        feedback = str(feedback_state.get('feedback') or '')
        if self._soft_gate_admission_enabled(reason=reason, source_reason=source_reason):
            self._emit('resource_planner_guard', job, status='review', payload={'status': status, 'reason': reason, 'scope': 'worker_plan', 'resource_mode': resource_mode, 'blocked_class': job.resource_class, 'allowed_classes': allowed_classes, 'guard_id': str(active_guard.get('guard_id') or ''), 'remaining_sec': float(active_guard.get('remaining_sec') or 0.0), 'current_pressure_generation': active_guard.get('current_pressure_generation'), 'source_feedback': guard, 'requires_admission_review': True, 'feedback_state_key': feedback_state.get('feedback_state_key'), 'feedback_suppressed': bool(feedback_state.get('feedback_suppressed')), 'resource_feedback_repeated_count': feedback_state.get('repeated_count')})
            return self._soft_gate_admission_review_result(job, status=status, reason=reason, scope='worker_plan', resource_mode=resource_mode, allowed_classes=allowed_classes, feedback=feedback, feedback_state=feedback_state, eta_next_train_sec=float(active_guard.get('remaining_sec') or 0.0), eta_confidence='low', unlock_condition='resource_context_changed', blocked_until_unlock=True, extra_facts={'guard_id': str(active_guard.get('guard_id') or ''), 'current_pressure_generation': active_guard.get('current_pressure_generation'), 'source_feedback': guard})
        self._remember_resource_feedback(job, status=status, reason=reason, resource_mode=resource_mode, blocked_class=job.resource_class, allowed_classes=allowed_classes, eta_next_train_sec=float(active_guard.get('remaining_sec') or 0.0))
        self._emit('resource_planner_guard', job, status='blocked', payload={'status': status, 'reason': reason, 'scope': 'worker_plan', 'resource_mode': resource_mode, 'blocked_class': job.resource_class, 'allowed_classes': allowed_classes, 'guard_id': str(active_guard.get('guard_id') or ''), 'remaining_sec': float(active_guard.get('remaining_sec') or 0.0), 'current_pressure_generation': active_guard.get('current_pressure_generation'), 'source_feedback': guard, 'feedback_state_key': feedback_state.get('feedback_state_key'), 'feedback_suppressed': bool(feedback_state.get('feedback_suppressed')), 'resource_feedback_repeated_count': feedback_state.get('repeated_count')})
        return {'allowed': False, 'resource_class': job.resource_class, 'status': status, 'reason': reason, 'feedback': feedback, 'feedback_suppressed': bool(feedback_state.get('feedback_suppressed')), 'feedback_state_key': feedback_state.get('feedback_state_key'), 'resource_feedback_repeated_count': feedback_state.get('repeated_count'), 'error': 'Resource planner guard blocked command after active RESOURCE_FEEDBACK'}

    def _timeout_preflight_result(self, job: Any) -> dict[str, Any]:
        if not self.timeout_hard_gate_enabled or self._gpu_queue_timeout_count <= 0:
            return {'allowed': True, 'resource_class': job.resource_class}
        blocked = False
        gate = ''
        if self._gpu_queue_timeout_count >= self.timeout_tt_only_after:
            blocked = job.resource_class not in TT_ALLOWED_AFTER_TIMEOUT
            gate = 'tt_only_after_queue_timeouts'
        elif self._gpu_queue_timeout_count >= self.timeout_block_train_after:
            blocked = job.resource_class in TRAIN_CLASSES or job.resource_class == RESOURCE_UNKNOWN_GPU_EXEC
            gate = 'block_train_after_queue_timeout'
        if not blocked:
            return {'allowed': True, 'resource_class': job.resource_class, 'gate': gate}
        if gate == 'tt_only_after_queue_timeouts':
            allowed = 'pure_tt_cpu,gpu_tt_light,heavy_cpu_candidate,readonly_cpu,light_cpu'
            allowed_classes = [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, RESOURCE_HEAVY_CPU_CANDIDATE, RESOURCE_READONLY_CPU, RESOURCE_LIGHT_CPU]
        else:
            allowed = 'non-training work'
            allowed_classes = [RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, 'readonly_cpu', 'light_cpu']
        bypass = self._allowed_class_bypass_result(job, reason=gate, scope='worker_local', resource_mode='RED', allowed_classes=allowed_classes)
        if bypass is not None:
            return {**bypass, 'gate': gate}
        trial = self._startup_trial_decision(job, reason=gate, status='DENIED_REPLAN', scope='worker_local', resource_mode='RED', allowed_classes=allowed_classes, gpu_ids=job.gpu_ids, extra_facts={'gpu_queue_timeout_count': self._gpu_queue_timeout_count, 'allowed_after_gate': allowed})
        if trial is not None:
            return {**trial, 'gate': gate}
        feedback = self._resource_feedback_text(status='DENIED_REPLAN', reason=gate, scope='worker_local', resource_mode='RED', blocked_class=job.resource_class, gpu_ids=job.gpu_ids, allowed_classes=allowed_classes, cooldown_sec=120.0, eta_next_train_sec=420.0, eta_confidence='low', unlock_condition='resource_pressure_cleared', blocked_until_unlock=True)
        if self._soft_gate_admission_enabled(reason=gate):
            self._emit('resource_policy_gate', job, status='review', payload={'status': 'DENIED_REPLAN', 'reason': gate, 'scope': 'worker_local', 'resource_mode': 'RED', 'gpu_queue_timeout_count': self._gpu_queue_timeout_count, 'blocked_class': job.resource_class, 'allowed_after_gate': allowed, 'allowed_classes': allowed_classes, 'cooldown_sec': 120.0, 'eta_next_train_sec': 420.0, 'eta_confidence': 'low', 'requires_admission_review': True})
            return self._soft_gate_admission_review_result(job, status='DENIED_REPLAN', reason=gate, scope='worker_local', resource_mode='RED', allowed_classes=allowed_classes, feedback=feedback, eta_next_train_sec=420.0, eta_confidence='low', cooldown_sec=120.0, unlock_condition='resource_pressure_cleared', blocked_until_unlock=True, extra_facts={'gpu_queue_timeout_count': self._gpu_queue_timeout_count, 'allowed_after_gate': allowed})
        self._remember_resource_feedback(job, status='DENIED_REPLAN', reason=gate, resource_mode='RED', blocked_class=job.resource_class, allowed_classes=allowed_classes, eta_next_train_sec=420.0, cooldown_sec=120.0)
        self._emit('resource_policy_gate', job, status='blocked', payload={'status': 'DENIED_REPLAN', 'reason': gate, 'scope': 'worker_local', 'resource_mode': 'RED', 'gpu_queue_timeout_count': self._gpu_queue_timeout_count, 'blocked_class': job.resource_class, 'allowed_after_gate': allowed, 'allowed_classes': allowed_classes, 'cooldown_sec': 120.0, 'eta_next_train_sec': 420.0, 'eta_confidence': 'low'})
        return {'allowed': False, 'resource_class': job.resource_class, 'reason': gate, 'feedback': feedback, 'error': 'Resource policy blocked command after GPU queue timeout'}

    def observation_policy(self, job_id: str | None, *, inferred_class: str, gpu_ids: list[str] | None=None, **_: Any) -> dict[str, Any]:
        if not (self.observation_enabled and self.observation_shadow_workspace_enabled):
            return {'enabled': False, 'reason': 'observation_disabled'}
        if not job_id or job_id not in self._jobs:
            return {'enabled': False, 'reason': 'job_not_found'}
        job = self._jobs[job_id]
        incoming = str(inferred_class or job.resource_class)
        resource_class = self._merge_resource_class(job.resource_class, incoming, source_hint=job.source_hint)
        ids = [str(x) for x in gpu_ids or job.gpu_ids or [] if str(x).strip()]
        if bool(getattr(job.source_hint, 'command_cpu_only', False)):
            ids = []
        if resource_class not in {RESOURCE_UNKNOWN_EXEC, RESOURCE_UNKNOWN_GPU_EXEC}:
            return {'enabled': False, 'reason': 'not_unknown_exec'}
        if not ids:
            return {'enabled': False, 'reason': 'no_gpu_scope'}
        if job.gpu_queue_relevant:
            return {'enabled': False, 'reason': 'already_queue_relevant'}
        return {'enabled': True, 'window_sec': self.observation_window_sec, 'reason': 'unknown_gpu_exec_shadow_observe', 'gpu_ids': ids}

    def observation_event(self, job_id: str | None, *, state: str, reason: str='', elapsed_sec: float=0.0, shadow_workspace: str='', **_: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        self._emit('resource_observation', self._jobs[job_id], status=str(state or 'observed'), payload={'state': str(state or ''), 'reason': str(reason or ''), 'elapsed_sec': float(elapsed_sec or 0.0), 'shadow_workspace': str(shadow_workspace or '')})

    def observation_promoted(self, job_id: str | None, *, reason: str, elapsed_sec: float=0.0, **_: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        hint = job.source_hint or ResourceSourceHint()
        labels = list(hint.hint_labels or [])
        if 'shadow_promoted' not in labels:
            labels.append('shadow_promoted')
        job.source_hint = replace(hint, command_gpu_evidence=True, command_train_evidence=True, hint_labels=labels)
        job.resource_class = RESOURCE_HEAVY_GPU_CANDIDATE
        job.gpu_queue_relevant = self._gpu_queue_relevant(job.resource_class, job.source_hint, job.gpu_ids)
        self._emit('resource_observation_promoted', job, status='promoted', payload={'reason': str(reason or ''), 'elapsed_sec': float(elapsed_sec or 0.0), 'promoted_class': job.resource_class, 'queue_relevant': bool(job.gpu_queue_relevant), 'source_hint': job.source_hint.to_event_payload()})
