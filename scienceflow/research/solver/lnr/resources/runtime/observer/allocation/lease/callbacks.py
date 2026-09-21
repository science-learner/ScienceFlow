"""Resource observer responsibility: lease registration, managed waits, and queue acquisition.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""
from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    Path,
    ResourceEffectKind,
    read_proc_info,
    time,
)


class LeaseCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def lease_registered(self, job_id: str | None, *, pid: int | None=None, pgid: int | None=None, **_: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.pid = int(pid) if pid is not None else None
        job.pgid = int(pgid) if pgid is not None else job.pid
        job.pid_start_time_epoch = None
        if job.pid is not None:
            info = read_proc_info(job.pid)
            try:
                job.pid_start_time_epoch = float(info.get('start_time_epoch')) if info.get('start_time_epoch') is not None else None
            except (TypeError, ValueError):
                job.pid_start_time_epoch = None
        if job.visible:
            self._emit('resource_lease_registered', job, status='running', payload={'pid': pid, 'pgid': pgid})

    def recoverable_artifact_scope_registered(self, job_id: str | None, *, run_dir: str | Path | None=None, artifact_dir: str | Path | None=None, run_state_path: str | Path | None=None, **_: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs[job_id]
        job.run_dir = Path(run_dir).resolve(strict=False) if run_dir is not None else None
        job.run_artifact_dir = Path(artifact_dir).resolve(strict=False) if artifact_dir is not None else None
        job.run_state_path = Path(run_state_path).resolve(strict=False) if run_state_path is not None else None
        payload = {'run_dir': str(job.run_dir or ''), 'artifact_dir': str(job.run_artifact_dir or ''), 'run_state_path': str(job.run_state_path or '')}
        self._emit('resource_recoverable_artifact_scope_registered', job, status='registered', payload=payload)
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event('resource_recoverable_artifact_scope_registered', payload={'job_id': job.job_id, 'command_digest': job.command_digest, 'resource_class': job.resource_class, **payload}, command_id=job.job_id, lease_id=job.job_id)

    def resource_wait_instruction_for_bash_sleep(self, *, command: str='', planned_sleep_sec: float | None=None) -> dict[str, Any]:
        if self.resource_runtime is None:
            return {'available': False, 'reason': 'resource_runtime_unavailable'}
        return self.resource_runtime.resource_wait_instruction_for_bash_sleep(command=command, planned_sleep_sec=planned_sleep_sec)

    async def managed_resource_wait(self, *, wait_token: str, max_wait_sec: float | None=None, reason: str='resource_busy') -> dict[str, Any]:
        if self.resource_runtime is None:
            return {'status': 'RESOURCE_WAIT_NOT_AVAILABLE', 'reason': 'resource_runtime_unavailable', 'feedback': 'RESOURCE_FEEDBACK: RESOURCE_WAIT_NOT_AVAILABLE because resource runtime is unavailable; retry_allowed=false.\n'}
        return await self.resource_runtime.managed_resource_wait(wait_token=wait_token, max_wait_sec=max_wait_sec, reason=reason)

    def queue_try_acquire(self, job_id: str | None, **kwargs: Any) -> dict[str, Any]:
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return {'enabled': False, 'acquired': True}
        job = self._jobs[job_id]
        incoming = str(kwargs.get('inferred_class') or job.resource_class)
        resource_class = self._merge_resource_class(job.resource_class, incoming, source_hint=job.source_hint)
        if bool(getattr(job.source_hint, 'command_cpu_only', False)):
            gpu_ids = []
        else:
            gpu_ids = [str(x) for x in kwargs.get('gpu_ids') or job.gpu_ids if str(x).strip()]
        job.resource_class = resource_class
        if gpu_ids:
            job.gpu_ids = gpu_ids
        elif bool(getattr(job.source_hint, 'command_cpu_only', False)):
            job.gpu_ids = []
        job.gpu_queue_relevant = self._gpu_queue_relevant(job.resource_class, job.source_hint, job.gpu_ids)
        if not job.gpu_queue_relevant:
            return {'enabled': False, 'acquired': True}
        metadata = {'command_digest': job.command_digest, 'command_excerpt': str(job.command or '')[:2000], 'entrypoint': self._command_entrypoint(job.command), 'resource_class': job.resource_class, 'cpu_set': job.cpu_set, 'gpu_request_count': job.gpu_request_count, 'source_hint': job.source_hint.to_event_payload() if job.source_hint is not None else {}, 'value_hint': dict(job.value_hint or {})}
        observe_first = dict(job.observe_first_state or {})
        if observe_first.get('active'):
            metadata.update({'observe_first': observe_first, 'observe_then_run': True, 'observe_source': str(observe_first.get('source') or 'stale_pressure_preflight'), 'observe_sec': float(observe_first.get('observe_window_sec') or self.stale_pressure_observe_window_sec)})
        attempt = self.resource_registry.next_operation_sequence(job.job_id, 'acquire')
        result = self._execute_resource_effect(ResourceEffectKind.ACQUIRE, command_id=job.job_id, idempotency_key=f'acquire:{job.job_id}:{attempt}', parameters={'job_id': job.job_id, 'resource_class': job.resource_class, 'gpu_ids': job.gpu_ids, 'request_count': job.gpu_request_count, 'metadata': metadata})
        if not result.get('enabled'):
            return result
        details = result.get('details') if isinstance(result.get('details'), dict) else {}
        for raw in details.get('reaped') or []:
            if isinstance(raw, dict):
                self._emit('resource_gpu_stale_lease_reaped', job, status='reaped', payload={'reaped_job_id': raw.get('job_id'), 'gpu_ids': raw.get('gpu_ids') or []})
        if result.get('acquired'):
            assigned_gpu_ids = [str(x) for x in result.get('gpu_ids') or [] if str(x).strip()]
            if assigned_gpu_ids:
                job.gpu_ids = assigned_gpu_ids
            lease = details.get('lease') if isinstance(details.get('lease'), dict) else {}
            lease_meta = lease.get('metadata') if isinstance(lease.get('metadata'), dict) else {}
            job.lease_mode = str(lease_meta.get('lease_mode') or 'exclusive')
            job.shared_primary_job_id = str(lease_meta.get('shared_primary_job_id') or '')
            if job.observe_first_state.get('active'):
                job.observe_first_state.update({'lease_acquired': True, 'assigned_physical_gpus': assigned_gpu_ids, 'lease_started_at': time.time()})
                result['admission_observe_then_run'] = True
                result['admission_observe_sec'] = float(job.observe_first_state.get('observe_window_sec') or self.stale_pressure_observe_window_sec)
                result['resource_trial'] = bool(job.observe_first_state.get('trial'))
                result['resource_trial_reason'] = str(job.observe_first_state.get('reason') or '')
                result['resource_trial_window_sec'] = float(job.observe_first_state.get('trial_window_sec') or job.observe_first_state.get('observe_window_sec') or result['admission_observe_sec'])
                result['resource_trial_hard_review_sec'] = float(job.observe_first_state.get('trial_hard_review_sec') or job.observe_first_state.get('observe_max_sec') or result['resource_trial_window_sec'])
                observe_payload = {**dict(job.observe_first_state), 'gpu_ids': result.get('gpu_ids') or job.gpu_ids, 'assigned_physical_gpus': assigned_gpu_ids, 'llm_called': False}
                self._emit('resource_observe_first_started', job, status='running', payload=observe_payload)
                if result.get('resource_trial'):
                    self._emit('resource_trial_started', job, status='running', payload=observe_payload)
                if self.resource_runtime is not None:
                    self.resource_runtime.record_resource_event('observe_first_started', payload={'job_id': job.job_id, 'resource_type': 'gpu', 'result': observe_payload}, command_id=job.job_id, lease_id=job.job_id)
                    if result.get('resource_trial'):
                        self.resource_runtime.record_resource_event('resource_trial_started', payload={'job_id': job.job_id, 'resource_type': 'gpu', 'result': observe_payload}, command_id=job.job_id, lease_id=job.job_id)
            self._emit('resource_gpu_lease_acquired', job, status='acquired', payload={'gpu_ids': result.get('gpu_ids') or job.gpu_ids, 'assigned_physical_gpus': result.get('assigned_physical_gpus') or result.get('gpu_ids') or job.gpu_ids, 'allowed_physical_gpus': result.get('allowed_physical_gpus') or [], 'candidate_physical_gpus': result.get('candidate_physical_gpus') or [], 'requested_gpu_count': result.get('requested_gpu_count'), 'reason': result.get('reason') or '', 'admission_status': result.get('status') or 'GRANTED', 'queue_position': result.get('queue_position'), 'queue_len': result.get('queue_len'), 'policy_resource_class': result.get('policy_resource_class') or '', 'slot_weight': result.get('slot_weight'), 'capacity_slots': result.get('capacity_slots'), 'admission_priority_score': result.get('admission_priority_score'), 'expected_value_score': result.get('expected_value_score'), 'near_submission_score': result.get('near_submission_score'), 'long_runtime_penalty': result.get('long_runtime_penalty'), 'value_hint': result.get('value_hint') or {}, 'pressure': result.get('pressure') or {}})
        elif result.get('queue_started_first'):
            self._emit('resource_gpu_queue_wait_started', job, status='pending', payload={'gpu_ids': result.get('gpu_ids') or job.gpu_ids, 'assigned_physical_gpus': result.get('assigned_physical_gpus') or [], 'allowed_physical_gpus': result.get('allowed_physical_gpus') or [], 'candidate_physical_gpus': result.get('candidate_physical_gpus') or [], 'requested_gpu_count': result.get('requested_gpu_count'), 'reason': result.get('reason') or 'gpu_slot_unavailable', 'max_wait_sec': result.get('max_wait_sec'), 'heartbeat_sec': result.get('heartbeat_sec'), 'policy_resource_class': result.get('policy_resource_class') or '', 'slot_weight': result.get('slot_weight'), 'capacity_slots': result.get('capacity_slots'), 'admission_priority_score': result.get('admission_priority_score'), 'expected_value_score': result.get('expected_value_score'), 'near_submission_score': result.get('near_submission_score'), 'long_runtime_penalty': result.get('long_runtime_penalty'), 'value_hint': result.get('value_hint') or {}, 'share_override_candidate': bool(result.get('share_override_candidate')), 'share_override': result.get('share_override') or {}})
        if str(result.get('status') or '').upper() in {'PENDING', 'REPLAN', 'DEFERRED', 'DENIED_REPLAN'}:
            self._remember_resource_feedback(job, status=str(result.get('status') or 'PENDING'), reason=str(result.get('reason') or 'gpu_slot_unavailable'), resource_mode=str(result.get('resource_mode') or 'YELLOW'), blocked_class=job.resource_class, allowed_classes=[str(x) for x in result.get('allowed_classes') or [] if str(x).strip()], eta_next_train_sec=float(result.get('eta_next_train_sec') or 0.0), retry_after_sec=float(result.get('retry_after_sec') or 0.0) if result.get('retry_after_sec') is not None else None, post_feedback_action=str(result.get('post_feedback_action') or ''))
            self._emit('resource_admission_deferred', job, status=str(result.get('status') or 'pending').lower(), payload={'status': result.get('status') or result.get('admission_action') or 'PENDING', 'admission_action': result.get('admission_action') or result.get('status') or 'PENDING', 'reason': result.get('reason') or 'gpu_slot_unavailable', 'resource_mode': result.get('resource_mode') or 'YELLOW', 'gpu_ids': result.get('gpu_ids') or job.gpu_ids, 'assigned_physical_gpus': result.get('assigned_physical_gpus') or [], 'allowed_physical_gpus': result.get('allowed_physical_gpus') or [], 'candidate_physical_gpus': result.get('candidate_physical_gpus') or [], 'requested_gpu_count': result.get('requested_gpu_count'), 'queue_position': result.get('queue_position'), 'queue_len': result.get('queue_len'), 'top_waiter_job_id': result.get('top_waiter_job_id'), 'eta_next_train_sec': result.get('eta_next_train_sec'), 'eta_confidence': result.get('eta_confidence') or 'low', 'retry_after_sec': result.get('retry_after_sec'), 'blocked_until_unlock': bool(result.get('blocked_until_unlock')), 'unlock_condition': result.get('unlock_condition') or '', 'post_feedback_action': result.get('post_feedback_action') or '', 'admission_cached_backoff': bool(result.get('admission_cached_backoff')), 'allowed_classes': result.get('allowed_classes') or [], 'policy_resource_class': result.get('policy_resource_class') or '', 'slot_weight': result.get('slot_weight'), 'capacity_slots': result.get('capacity_slots'), 'eta_source': result.get('eta_source'), 'runtime_history_count': result.get('runtime_history_count'), 'runtime_avg_sec': result.get('runtime_avg_sec'), 'runtime_p80_sec': result.get('runtime_p80_sec'), 'active_blocker_count': result.get('active_blocker_count'), 'active_blocker_age_sec_max': result.get('active_blocker_age_sec_max'), 'admission_priority_score': result.get('admission_priority_score'), 'expected_value_score': result.get('expected_value_score'), 'near_submission_score': result.get('near_submission_score'), 'long_runtime_penalty': result.get('long_runtime_penalty'), 'value_hint': result.get('value_hint') or {}, 'pressure': result.get('pressure') or {}, 'share_override_candidate': bool(result.get('share_override_candidate')), 'share_override': result.get('share_override') or {}})
        return result
