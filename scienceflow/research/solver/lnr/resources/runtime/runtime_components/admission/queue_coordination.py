"""Resource runtime responsibility: queue acquisition, timeout, and policy-gate records."""
from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.runtime_components.control.shared import (
    Any,
    GPULeaseStore,
    RESOURCE_HEAVY_GPU_CANDIDATE,
    RESOURCE_HEAVY_GPU_TRAIN,
    RESOURCE_PURE_TT_CPU,
    RESOURCE_UNKNOWN_GPU_EXEC,
    _CPU_SUPPORT_CLASSES,
    _GPU_PRESSURE_SUPPORT_CLASSES,
    build_admission_safety_deferred_result,
    evaluate_admission_safety,
    sample_nvidia_smi,
)


class QueueRuntime:
    """Own this responsibility's state transitions and callbacks."""

    def queue_try_acquire(self, *, job_id: str, resource_class: str, gpu_ids: list[str], request_count: int | None=None, metadata: dict[str, Any] | None=None) -> dict[str, Any]:
        ids = [str(x) for x in gpu_ids or [] if str(x).strip()]
        policy = self._policy_for_resource_class(resource_class)
        if policy is None or not self.should_queue_gpu(resource_class=resource_class, gpu_ids=ids):
            return {'enabled': False, 'acquired': True}
        return self._queue_try_acquire_enabled(job_id=job_id, resource_class=resource_class, gpu_ids=gpu_ids, ids=ids, request_count=request_count, metadata=metadata, policy=policy)

    def _queue_try_acquire_enabled(self, *, job_id: str, resource_class: str, gpu_ids: list[str], ids: list[str], request_count: int | None, metadata: dict[str, Any] | None, policy: Any) -> dict[str, Any]:
        assignment = self._assignment_mode()
        meta = {**dict(metadata or {}), **GPULeaseStore.current_owner_metadata(), 'assignment': assignment, 'resource_class': str(resource_class or ''), 'policy_resource_class': policy.resource_class, 'slot_weight': policy.slot_weight, 'capacity_slots': self._capacity_slots(), 'max_per_gpu': policy.max_per_gpu, 'incompatible_classes': list(policy.incompatible_classes)}
        value_hint = meta.get('value_hint') if isinstance(meta.get('value_hint'), dict) else {}
        priority = self._admission_priority(value_hint)
        meta['admission_priority'] = priority
        candidate_ids, requested_count, sublease = self._queue_assignment_plan(assignment=assignment, ids=ids, request_count=request_count, metadata=meta)
        cached_backoff = self._cached_admission_backoff(job_id=str(job_id or ''), policy=policy, target_gpu_ids=candidate_ids or ids, priority=priority)
        if cached_backoff is not None:
            return cached_backoff
        safety_result, candidate_ids, requested_count, sublease = self._queue_admission_safety(job_id=job_id, resource_class=resource_class, ids=ids, assignment=assignment, policy=policy, candidate_ids=candidate_ids, requested_count=requested_count, sublease=sublease, metadata=meta, value_hint=value_hint, priority=priority)
        if safety_result is not None:
            return safety_result
        acquired, details = self._queue_acquire_lease(job_id=job_id, ids=ids, assignment=assignment, policy=policy, candidate_ids=candidate_ids, requested_count=requested_count, request_count=request_count, metadata=meta)
        lease = details.get('lease') if isinstance(details.get('lease'), dict) else {}
        assigned_gpu_ids = [str(x) for x in lease.get('gpu_ids') or details.get('assigned_gpu_ids') or ids if str(x).strip()]
        allowed_physical_gpu_ids = [str(x) for x in (self._configured_gpu_pool() or ids or candidate_ids if assignment == 'lease' else ids) if str(x).strip()]
        boundary_result = self._queue_boundary_result(job_id=job_id, acquired=bool(acquired), assignment=assignment, assigned_gpu_ids=assigned_gpu_ids, allowed_gpu_ids=allowed_physical_gpu_ids, candidate_ids=candidate_ids, resource_class=resource_class, policy=policy, details=details, value_hint=value_hint, priority=priority)
        if boundary_result is not None:
            return boundary_result
        self._remember_acquired_lease(job_id=job_id, acquired=bool(acquired), assigned_gpu_ids=assigned_gpu_ids)
        env_updates = self._lease_environment(acquired=bool(acquired), assignment=assignment, assigned_gpu_ids=assigned_gpu_ids, allowed_gpu_ids=allowed_physical_gpu_ids)
        result = {'enabled': True, 'acquired': bool(acquired), 'reason': str(details.get('reason') or ''), 'max_wait_sec': float(self.gpu_queue.max_wait_sec), 'heartbeat_sec': float(self.gpu_queue.heartbeat_sec), 'gpu_ids': assigned_gpu_ids or ids, 'assigned_physical_gpus': assigned_gpu_ids, 'allowed_physical_gpus': allowed_physical_gpu_ids, 'candidate_physical_gpus': candidate_ids, 'requested_gpu_count': requested_count, 'gpu_sublease': dict(sublease), 'env_updates': env_updates, 'resource_class': str(resource_class or ''), 'policy_resource_class': policy.resource_class, 'slot_weight': policy.slot_weight, 'capacity_slots': self._capacity_slots(), 'details': details, 'queue_started_first': False, 'queue_position': details.get('queue_position'), 'queue_len': details.get('queue_len'), 'top_waiter_job_id': details.get('top_waiter_job_id'), 'value_hint': value_hint, **priority}
        granted = self._granted_admission_result(job_id=job_id, acquired=bool(acquired), result=result, details=details, lease=lease, assigned_gpu_ids=assigned_gpu_ids)
        if granted is not None:
            return granted
        result['queue_started_first'] = bool(self.queue_scheduler.mark_pending(job_id)) or bool(result.get('queue_started_first'))
        eta_info = self._eta_for_blocked(resource_class=policy.resource_class, details=details, command_digest=str(meta.get('command_digest') or ''), entrypoint=str(meta.get('entrypoint') or ''))
        share_override_candidate = self._admission_share_override_candidate(job_id=str(job_id or ''), policy=policy, details=details, priority=priority, value_hint=value_hint)
        share_override_record = self._record_share_override(job_id=job_id, candidate=share_override_candidate)
        eta = float(eta_info.get('eta_next_train_sec') or self._cold_eta_sec())
        eta_confidence = str(eta_info.get('eta_confidence') or 'low')
        pressure_ids = assigned_gpu_ids or ids or [str(x) for x in details.get('candidate_gpu_ids') or details.get('blocked_gpu_ids') or [] if str(x).strip()]
        yellow_pressure = self.pressure_store.record_yellow(gpu_ids=self._pressure_gpu_ids(pressure_ids), worker_id=self.worker_id, job_id=str(job_id), resource_class=policy.resource_class, reason=str(details.get('reason') or 'resource_admission_deferred'), queue_len=int(result.get('queue_len') or 0), queue_position=int(result.get('queue_position') or 1), metadata={'top_waiter_job_id': str(result.get('top_waiter_job_id') or '')})
        (admission_action, admission_reason) = self._admission_action_for_blocked(policy=policy, requested_count=requested_count, details=details, priority=priority, eta_next_train_sec=eta)
        queue_position = int(result.get('queue_position') or 1)
        queue_len = int(result.get('queue_len') or 0)
        target_gpu_ids = assigned_gpu_ids or ids or self._pressure_gpu_ids(gpu_ids)
        blocker_job_ids = [str(x) for x in eta_info.get('active_blocker_job_ids') or [] if str(x).strip()]
        backoff_key = self._admission_backoff_key(policy_resource_class=policy.resource_class, gpu_ids=target_gpu_ids)
        backoff_due = admission_action == 'PENDING' and (not bool(share_override_candidate.get('candidate'))) and (self._waiter_wait_age_sec(details) >= self._admission_backoff_grace_sec()) and (policy.resource_class in {RESOURCE_HEAVY_GPU_TRAIN, RESOURCE_HEAVY_GPU_CANDIDATE, RESOURCE_UNKNOWN_GPU_EXEC})
        retry_after_sec: float | None = None
        if admission_action == 'REPLAN' or backoff_due:
            self.lease_manager.release(str(job_id or ''))
            self.queue_scheduler.complete(str(job_id or ''))
            result['queue_started_first'] = False
        if backoff_due:
            admission_action = 'DENIED_REPLAN'
            admission_reason = 'cached_gpu_admission_backoff'
            retry_after_sec = self._admission_backoff_retry_sec(eta)
        wait_option: dict[str, Any] = {}
        if admission_action == 'PENDING' and (not backoff_due):
            wait_option = self.create_resource_wait_option(job_id=str(job_id or ''), resource_class=policy.resource_class, gpu_ids=target_gpu_ids, holder_job_ids=blocker_job_ids, queue_position=queue_position, queue_len=queue_len, command_digest=str(meta.get('command_digest') or ''), reason=admission_reason, max_wait_sec=min(600.0, max(60.0, eta if eta > 0 else 600.0)))
        result.update({'status': admission_action, 'admission_action': admission_action, 'reason': admission_reason, 'resource_mode': 'YELLOW', 'queue_position': queue_position, 'queue_len': queue_len, 'top_waiter_job_id': str(result.get('top_waiter_job_id') or ''), 'eta_next_train_sec': eta, 'eta_confidence': eta_confidence, 'holder_job_ids': blocker_job_ids, 'allowed_classes': _GPU_PRESSURE_SUPPORT_CLASSES, 'pressure': yellow_pressure, 'duplicate_digest_summary': self._duplicate_summary(details), 'share_override_candidate': bool(share_override_candidate.get('candidate')), 'share_override': share_override_candidate, 'share_override_record': share_override_record, 'admission_cached_backoff': bool(backoff_due), 'retry_after_sec': retry_after_sec, 'blocked_until_unlock': bool(backoff_due), 'unlock_condition': 'holder_released' if backoff_due else '', 'post_feedback_action': 'cpu_support' if backoff_due else '', 'resource_wait_option': wait_option, **eta_info, 'feedback': self._admission_feedback(action=admission_action, reason=admission_reason, policy_resource_class=policy.resource_class, gpu_ids=target_gpu_ids, queue_position=queue_position, queue_len=queue_len, eta_next_train_sec=eta, eta_confidence=eta_confidence, priority_score=float(priority.get('admission_priority_score') or 0.0), holder_job_id=blocker_job_ids[0] if blocker_job_ids else '', retry_after_sec=retry_after_sec, post_feedback_action='cpu_support' if backoff_due else '', cached_backoff=bool(backoff_due), wait_option=wait_option)})
        opportunity = self.admission_opportunity_facts(resource_class=policy.resource_class, gpu_ids=target_gpu_ids or candidate_ids, request_count=requested_count)
        result['admission_opportunity'] = opportunity
        result['lease_grantable_by_llm'] = bool(opportunity.get('lease_grantable_by_llm'))
        if backoff_due:
            self._remember_admission_backoff(key=backoff_key, policy=policy, target_gpu_ids=target_gpu_ids, result=result, holder_job_ids=blocker_job_ids, eta_next_train_sec=eta, eta_confidence=eta_confidence, pressure_generation=int(yellow_pressure.get('generation') or self._current_pressure_generation()))
            self.record_resource_event('admission_cached_backoff', payload={'job_id': str(job_id or ''), 'resource_type': 'gpu', 'result': result}, command_id=str(job_id or ''), lease_id=str(job_id or ''))
        if share_override_candidate.get('candidate'):
            self.record_resource_event('admission_share_override_candidate', payload={'job_id': str(job_id or ''), 'resource_type': 'gpu', 'result': result, 'share_override': share_override_candidate, 'share_override_record': share_override_record}, command_id=str(job_id or ''), lease_id=str(job_id or ''))
        self.record_resource_event('admission_pending' if admission_action == 'PENDING' else 'admission_replan', payload={'job_id': str(job_id or ''), 'resource_type': 'gpu', 'result': result}, command_id=str(job_id or ''), lease_id=str(job_id or ''))
        return result

    @staticmethod
    def _lease_environment(*, acquired: bool, assignment: str, assigned_gpu_ids: list[str], allowed_gpu_ids: list[str]) -> dict[str, str]:
        if not (acquired and assignment == 'lease' and assigned_gpu_ids):
            return {}
        env = {'CUDA_VISIBLE_DEVICES': ','.join(assigned_gpu_ids), 'SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL': ','.join(assigned_gpu_ids), 'SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL': ','.join((str(i) for (i, _) in enumerate(assigned_gpu_ids)))}
        if allowed_gpu_ids:
            env['SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL'] = ','.join(allowed_gpu_ids)
        return env

    def _queue_assignment_plan(self, *, assignment: str, ids: list[str], request_count: int | None, metadata: dict[str, Any]) -> tuple[list[str], int, dict[str, Any]]:
        if assignment == 'lease':
            candidate_ids = self._configured_gpu_pool() or ids
            sublease = self._gpu_sublease_plan(candidate_gpu_ids=candidate_ids, request_count=request_count, metadata=metadata)
            requested_count = int(sublease.get('requested_gpu_count') or 1)
            metadata['gpu_sublease'] = dict(sublease)
            return candidate_ids, requested_count, sublease
        requested_count = max(1, int(request_count or len(ids) or 1))
        return ids, requested_count, {'requested_gpu_count': requested_count, 'candidate_gpu_ids': list(ids), 'reason': 'direct_assignment'}

    def _queue_admission_safety(self, *, job_id: str, resource_class: str, ids: list[str], assignment: str, policy: Any, candidate_ids: list[str], requested_count: int, sublease: dict[str, Any], metadata: dict[str, Any], value_hint: dict[str, Any], priority: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], int, dict[str, Any]]:
        min_free = max(0.0, float(self.gpu_queue.pressure_min_free_mem_gb or 0.0))
        if min_free <= 0 or not candidate_ids:
            return None, candidate_ids, requested_count, sublease
        safety_sample = sample_nvidia_smi(candidate_ids)
        safety = evaluate_admission_safety(sample=safety_sample if isinstance(safety_sample, dict) else {}, gpu_ids=candidate_ids, resource_class=policy.resource_class, request_count=requested_count, assignment=assignment, min_free_mem_gb=min_free, free_mem_buffer_gb=max(0.0, float(self.gpu_queue.pressure_yellow_free_mem_buffer_gb or 0.0)))
        if safety.get('blocked'):
            queue_started_first = self.queue_scheduler.mark_pending(str(job_id))
            pressure_ids = [str(x) for x in safety.get('blocked_gpu_ids') or candidate_ids if str(x).strip()]
            pressure = self.pressure_store.record_yellow(gpu_ids=self._pressure_gpu_ids(pressure_ids), worker_id=self.worker_id, job_id=str(job_id), resource_class=policy.resource_class, reason=str(safety.get('reason') or 'gpu_memory_below_admission_reserve'), queue_len=0, queue_position=1, metadata={'free_mem_threshold_gb': safety.get('free_mem_threshold_gb'), 'blocked_gpu_ids': safety.get('blocked_gpu_ids') or []})
            deferred = build_admission_safety_deferred_result(resource_class=str(resource_class or ''), policy_resource_class=policy.resource_class, gpu_ids=pressure_ids or candidate_ids, safety=safety, max_wait_sec=float(self.gpu_queue.max_wait_sec), heartbeat_sec=float(self.gpu_queue.heartbeat_sec), slot_weight=policy.slot_weight, capacity_slots=self._capacity_slots(), details={'reason': str(safety.get('reason') or 'gpu_memory_below_admission_reserve'), 'candidate_gpu_ids': candidate_ids, 'blocked_gpu_ids': safety.get('blocked_gpu_ids') or [], 'safety': safety, 'gpu_sublease': dict(sublease)}, value_hint=value_hint, priority=priority, queue_started_first=queue_started_first, pressure=pressure, eta_next_train_sec=self._cold_eta_sec())
            allowed_pool_ids = [str(x) for x in (self._configured_gpu_pool() or ids or candidate_ids if assignment == 'lease' else ids) if str(x).strip()]
            opportunity = self.admission_opportunity_facts(resource_class=policy.resource_class, gpu_ids=candidate_ids or allowed_pool_ids, request_count=requested_count)
            deferred.update({'assigned_physical_gpus': [], 'allowed_physical_gpus': allowed_pool_ids, 'candidate_physical_gpus': candidate_ids, 'requested_gpu_count': requested_count, 'gpu_sublease': dict(sublease), 'admission_opportunity': opportunity, 'lease_grantable_by_llm': bool(opportunity.get('lease_grantable_by_llm'))})
            return deferred, candidate_ids, requested_count, sublease
        if assignment != 'lease':
            return None, candidate_ids, requested_count, sublease
        safe_candidate_ids = [str(x) for x in safety.get('safe_gpu_ids') or candidate_ids if str(x).strip()]
        if not safe_candidate_ids:
            return None, candidate_ids, requested_count, sublease
        requested_count = self._request_count(len(safe_candidate_ids), requested=requested_count)
        sublease = {**sublease, 'candidate_gpu_ids': list(safe_candidate_ids), 'requested_gpu_count': requested_count}
        metadata['gpu_sublease'] = dict(sublease)
        return None, safe_candidate_ids, requested_count, sublease

    def _queue_acquire_lease(self, *, job_id: str, ids: list[str], assignment: str, policy: Any, candidate_ids: list[str], requested_count: int, request_count: int | None, metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        common = {'job_id': str(job_id), 'worker_id': self.worker_id, 'max_heavy_per_gpu': self.gpu_queue.max_heavy_per_gpu, 'resource_class': policy.resource_class, 'slot_weight': policy.slot_weight, 'capacity_slots': self._capacity_slots(), 'max_per_gpu': policy.max_per_gpu, 'incompatible_classes': list(policy.incompatible_classes)}
        if assignment == 'lease':
            return self.gpu_store.try_acquire_any(candidate_gpu_ids=candidate_ids, request_count=requested_count, metadata=metadata, **common)
        return self.gpu_store.try_acquire(gpu_ids=ids, metadata={**metadata, 'request_count': request_count}, **common)

    def _queue_boundary_result(self, *, job_id: str, acquired: bool, assignment: str, assigned_gpu_ids: list[str], allowed_gpu_ids: list[str], candidate_ids: list[str], resource_class: str, policy: Any, details: dict[str, Any], value_hint: dict[str, Any], priority: dict[str, Any]) -> dict[str, Any] | None:
        outside_pool = acquired and assignment == 'lease' and allowed_gpu_ids and assigned_gpu_ids and not set(assigned_gpu_ids).issubset(set(allowed_gpu_ids))
        if not outside_pool:
            return None
        release = self.lease_manager.release(str(job_id or ''))
        self.queue_scheduler.complete(str(job_id or ''))
        result = {'enabled': True, 'acquired': False, 'status': 'REPLAN', 'admission_action': 'REPLAN', 'reason': 'assigned_gpu_outside_task_pool', 'resource_mode': 'BOUNDARY_VIOLATION', 'max_wait_sec': float(self.gpu_queue.max_wait_sec), 'heartbeat_sec': float(self.gpu_queue.heartbeat_sec), 'gpu_ids': assigned_gpu_ids, 'assigned_physical_gpus': assigned_gpu_ids, 'allowed_physical_gpus': allowed_gpu_ids, 'candidate_physical_gpus': candidate_ids, 'resource_class': str(resource_class or ''), 'policy_resource_class': policy.resource_class, 'slot_weight': policy.slot_weight, 'capacity_slots': self._capacity_slots(), 'details': {**details, 'release': release}, 'queue_started_first': False, 'queue_position': 0, 'queue_len': 0, 'allowed_classes': [RESOURCE_PURE_TT_CPU, *_CPU_SUPPORT_CLASSES], 'feedback': f"RESOURCE_FEEDBACK: REPLAN because assigned_gpu_outside_task_pool; assigned_gpu={','.join(assigned_gpu_ids)}; allowed_gpu={','.join(allowed_gpu_ids)}.\n", 'value_hint': value_hint, **priority}
        self.record_resource_event('admission_boundary_violation', payload={'job_id': str(job_id or ''), 'resource_type': 'gpu', 'result': result}, command_id=str(job_id or ''), lease_id=str(job_id or ''))
        return result

    def _remember_acquired_lease(self, *, job_id: str, acquired: bool, assigned_gpu_ids: list[str]) -> None:
        if acquired and assigned_gpu_ids:
            self.lease_manager.remember(str(job_id), assigned_gpu_ids)

    def _granted_admission_result(self, *, job_id: str, acquired: bool, result: dict[str, Any], details: dict[str, Any], lease: dict[str, Any], assigned_gpu_ids: list[str]) -> dict[str, Any] | None:
        if not acquired:
            return None
        result['status'] = 'GRANTED'
        result['admission_action'] = 'RUN_NOW'
        result['queue_position'] = int(details.get('queue_position') or 1)
        result['queue_len'] = int(details.get('queue_len') or 0)
        lease_meta = lease.get('metadata') if isinstance(lease.get('metadata'), dict) else {}
        if str(lease_meta.get('lease_mode') or '') == 'shared_secondary':
            self.record_resource_event('shared_gpu_lease_consumed', payload={'job_id': str(job_id or ''), 'primary_job_id': str(lease_meta.get('shared_primary_job_id') or ''), 'gpu_ids': assigned_gpu_ids, 'reason': str(details.get('reason') or 'already_acquired'), 'lease': lease}, command_id=str(job_id or ''), lease_id=str(job_id or ''))
        self.record_resource_event('admission_granted', payload={'job_id': str(job_id or ''), 'resource_type': 'gpu', 'result': result}, command_id=str(job_id or ''), lease_id=str(job_id or ''))
        return result

    def _record_share_override(self, *, job_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
        if not candidate.get('candidate'):
            return {'updated': False}
        try:
            return self.gpu_store.annotate_waiter_share_override(job_id=str(job_id or ''), share_override=candidate)
        except Exception as exc:
            return {'updated': False, 'reason': f'share_override_record_failed:{type(exc).__name__}'}

    def queue_wait_heartbeat(self, *, job_id: str, elapsed_sec: float) -> dict[str, Any]:
        heartbeat = self.queue_scheduler.heartbeat(job_id, elapsed_sec=elapsed_sec)
        return {'emit': heartbeat.emit, 'elapsed_sec': heartbeat.elapsed_sec}

    def queue_timeout(self, *, job_id: str, elapsed_sec: float, reason: str) -> dict[str, Any]:
        self.queue_scheduler.complete(str(job_id or ''))
        release = self.lease_manager.release(str(job_id or ''))
        result = {'released': bool(release.get('released')), 'elapsed_sec': float(elapsed_sec or 0.0), 'reason': str(reason or 'queue_timeout'), 'release': release}
        self.record_resource_event('queue_timeout', payload={'job_id': str(job_id or ''), 'resource_type': 'gpu', 'result': result}, command_id=str(job_id or ''), lease_id=str(job_id or ''))
        return result

    def record_queue_timeout_pressure(self, *, job_id: str, resource_class: str, gpu_ids: list[str], elapsed_sec: float, reason: str, command_digest: str='') -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        return self.pressure_store.record_queue_timeout(gpu_ids=ids, worker_id=self.worker_id, job_id=str(job_id or ''), resource_class=str(resource_class or ''), elapsed_sec=float(elapsed_sec or 0.0), reason=str(reason or 'queue_timeout'), command_digest=str(command_digest or ''))

    def record_policy_gate_digest(self, *, job_id: str, resource_class: str, gpu_ids: list[str], command_digest: str, reason: str) -> dict[str, Any]:
        ids = self._pressure_gpu_ids(gpu_ids)
        return self.pressure_store.record_digest_failure(gpu_ids=ids, worker_id=self.worker_id, job_id=str(job_id or ''), resource_class=str(resource_class or ''), command_digest=str(command_digest or ''), reason=str(reason or 'resource_policy_gate'))
