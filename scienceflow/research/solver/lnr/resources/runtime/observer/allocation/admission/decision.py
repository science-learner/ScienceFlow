"""Resource observer responsibility: admission facts, sharing decisions, and queue timeout.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""
from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    RESOURCE_GPU_TT_LIGHT,
    RESOURCE_PURE_TT_CPU,
    ResourceEffectKind,
    ResourceJob,
    apply_admission_decision,
    asyncio,
    evaluate_admission_share_trial,
    hashlib,
    inspect,
    parse_admission_decision_text,
    project_gpu_share_decision_facts,
    resource_feedback_text,
    time,
)


class AdmissionCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def _admission_task_card(self, job: ResourceJob, result: dict[str, Any]) -> dict[str, Any]:
        workspace = job.workspace_dir.resolve(strict=False) if job.workspace_dir is not None else None
        command = str(job.command or '')
        if workspace is not None:
            command = command.replace(str(workspace), '.')
        details = result.get('details') if isinstance(result.get('details'), dict) else {}
        return {'job_id': job.job_id, 'worker_id': self.worker_id, 'command_preview': command[:600], 'command_digest': job.command_digest, 'entrypoint': self._command_entrypoint(job.command), 'resource_class': job.resource_class, 'policy_resource_class': result.get('policy_resource_class') or '', 'gpu_ids': result.get('gpu_ids') or job.gpu_ids, 'assigned_physical_gpus': result.get('assigned_physical_gpus') or [], 'allowed_physical_gpus': result.get('allowed_physical_gpus') or [], 'candidate_physical_gpus': result.get('candidate_physical_gpus') or [], 'requested_gpu_count': result.get('requested_gpu_count') or job.gpu_request_count, 'gpu_request_count': job.gpu_request_count, 'timeout_sec': job.timeout_sec, 'queue_position': result.get('queue_position'), 'queue_len': result.get('queue_len'), 'top_waiter_job_id': result.get('top_waiter_job_id'), 'eta_next_train_sec': result.get('eta_next_train_sec'), 'eta_confidence': result.get('eta_confidence'), 'admission_priority_score': result.get('admission_priority_score'), 'value_hint': result.get('value_hint') or dict(job.value_hint or {}), 'duplicate_digest_summary': result.get('duplicate_digest_summary') or {}, 'safety': result.get('safety') or {}, 'admission_opportunity': result.get('admission_opportunity') or {}, 'observe_first': dict(job.observe_first_state or {}), 'requires_admission_review': bool(result.get('requires_admission_review')), 'soft_gate_reason': result.get('soft_gate_reason') or '', 'soft_gate_scope': result.get('soft_gate_scope') or '', 'soft_gate': result.get('soft_gate') or {}, 'lease_grantable_by_llm': bool(result.get('lease_grantable_by_llm')), 'blocked_until_unlock': bool(result.get('blocked_until_unlock')), 'unlock_condition': result.get('unlock_condition') or '', 'blocker_details': details.get('blocker_details') or {}, 'resource_mode': result.get('resource_mode') or '', 'remaining_timeout_sec': job.timeout_sec}

    def _admission_share_review_proposal(self, job: ResourceJob, admission_result: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        proposal_type = 'task_gpu_share_review'
        primary_id = str(observation.get('primary_job_id') or '')
        seed = f'{proposal_type}:{job.job_id}:{primary_id}:{now}'
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:8]}"
        trial_share = observation.get('trial_share') if isinstance(observation.get('trial_share'), dict) else {}
        reason_code = 'task_gpu_share_review:admission_shared_trial'
        trigger_reasons = ['admission_blocked_waiter', 'share_override_candidate', str((observation.get('secondary_allowed') or {}).get('reason') or 'secondary_allowed')]
        if trial_share.get('enabled'):
            trigger_reasons.append('revocable_trial_share')
        share_decision_facts = project_gpu_share_decision_facts(observation, decision_source='admission_waiter')
        return {'proposal_id': proposal_id, 'proposal_type': proposal_type, 'severity': 'yellow', 'reason_code': reason_code, 'trigger_reasons': trigger_reasons, 'suggested_actions': ['GRANT_SHARED_GPU_LEASE', 'DENY_SHARE_USE_CPU_SUPPORT', 'CONTINUE_SHARED_OBSERVE'], 'requires_llm_decision': True, 'resource_snapshot': {'gpu_ids': observation.get('primary_gpu_ids') or admission_result.get('gpu_ids') or job.gpu_ids, 'cpu_pressure': observation.get('cpu_pressure') or 'unknown', 'cpu_isolation': observation.get('cpu_isolation') or {}, 'memory': observation.get('memory') or {}, 'hard_gates': observation.get('hard_gates') or {}, 'secondary_allowed': observation.get('secondary_allowed') or {}, 'trial_share': trial_share, 'admission_immediate_review': True}, 'progress_snapshot': {'progress_signal': 'unknown', 'progress_confidence': 'low', 'heartbeat_source': 'admission_waiter', 'evidence_limitations': ['primary_progress_owned_by_holder_monitor']}, 'blocker': {'job_id': primary_id, 'worker_id': str(observation.get('primary_worker_id') or ''), 'resource_class': str(observation.get('primary_resource_class') or ''), 'gpu_ids': observation.get('primary_gpu_ids') or [], 'share_role': 'primary', 'progress_signal': 'unknown'}, 'waiters': [observation.get('waiter') or {'job_id': job.job_id, 'resource_class': job.resource_class}], 'share_observation': observation, 'share_decision_facts': share_decision_facts, 'share_payload': {'schema_version': 1, 'phase': observation.get('phase'), 'primary': {'job_id': primary_id, 'worker_id': str(observation.get('primary_worker_id') or ''), 'resource_class': str(observation.get('primary_resource_class') or ''), 'gpu_ids': observation.get('primary_gpu_ids') or [], 'progress_signal': 'unknown', 'progress_confidence': 'low'}, 'waiter': observation.get('waiter') or {}, 'waiters': observation.get('waiters') or [], 'cpu_pressure': observation.get('cpu_pressure') or 'unknown', 'cpu_isolation': observation.get('cpu_isolation') or {}, 'memory': observation.get('memory') or {}, 'hard_gates': observation.get('hard_gates') or {}, 'secondary_allowed': observation.get('secondary_allowed') or {}, 'trial_share': trial_share, 'admission_immediate_review': True}, 'admission_result': {'status': str(admission_result.get('status') or ''), 'reason': str(admission_result.get('reason') or ''), 'queue_position': admission_result.get('queue_position'), 'queue_len': admission_result.get('queue_len')}, 'decision_preview': {'job_id': job.job_id, 'reason': reason_code, 'would_terminate': False, 'arbiter_review': True, 'requires_llm_decision': True}, 'command_id': job.job_id}

    def _admission_share_cpu_support_result(self, job: ResourceJob, admission_result: dict[str, Any], *, reason: str, decision: dict[str, Any] | None=None) -> dict[str, Any]:
        out = dict(admission_result or {})
        out.update({'enabled': True, 'acquired': False, 'status': 'PENDING', 'admission_action': 'PENDING', 'reason': reason or 'local_gpu_busy_use_cpu_support', 'admission_llm_reviewed': True, 'admission_share_reviewed': True, 'post_feedback_action': 'cpu_support', 'retry_after_sec': out.get('retry_after_sec') or 120.0})
        holders = [str(x) for x in out.get('holder_job_ids') or [] if str(x).strip()]
        wait_option: dict[str, Any] = {}
        if self.resource_runtime is not None:
            wait_option = self.resource_runtime.create_resource_wait_option(job_id=job.job_id, resource_class=str(out.get('policy_resource_class') or out.get('resource_class') or job.resource_class), gpu_ids=[str(x) for x in out.get('gpu_ids') or job.gpu_ids if str(x).strip()], holder_job_ids=holders, queue_position=int(out.get('queue_position') or 0) if out.get('queue_position') is not None else None, queue_len=int(out.get('queue_len') or 0) if out.get('queue_len') is not None else None, command_digest=job.command_digest, reason=reason or 'shared GPU trial was not approved', max_wait_sec=min(600.0, max(60.0, float(out.get('eta_next_train_sec') or 600.0))))
        out['resource_wait_option'] = wait_option
        extra_facts = {'post_feedback_action': 'cpu_support', 'arbiter_action': str((decision or {}).get('action') or 'DENY_SHARE_USE_CPU_SUPPORT')}
        if self.resource_runtime is not None:
            extra_facts.update(self.resource_runtime._resource_wait_extra_facts(wait_option))
        out['feedback'] = resource_feedback_text(status='LOCAL_GPU_BUSY_USE_CPU_SUPPORT', reason=reason or 'shared GPU trial was not approved', scope='per_gpu', resource_mode=str(out.get('resource_mode') or 'YELLOW'), blocked_class=str(out.get('policy_resource_class') or out.get('resource_class') or job.resource_class), gpu_ids=[str(x) for x in out.get('gpu_ids') or job.gpu_ids if str(x).strip()], allowed_classes=[str(x) for x in out.get('allowed_classes') or [] if str(x).strip()], holder_job_id=holders[0] if holders else '', queue_position=int(out.get('queue_position') or 0) if out.get('queue_position') is not None else None, queue_len=int(out.get('queue_len') or 0) if out.get('queue_len') is not None else None, eta_next_train_sec=max(0.0, float(out.get('eta_next_train_sec') or 0.0)) if out.get('eta_next_train_sec') is not None else None, eta_confidence=str(out.get('eta_confidence') or ''), unlock_condition='holder_released_or_shared_trial_granted', blocked_until_unlock=True, extra_facts=extra_facts)
        return out

    async def admission_share_decide(self, job_id: str | None, *, admission_result: dict[str, Any] | None=None) -> dict[str, Any]:
        result = dict(admission_result or {})
        deterministic_trial_admission = self.gpu_share_config.trial_admission_policy == 'deterministic_grant_when_hard_gates_pass'
        if not self.gpu_share_config.grant_active:
            return {'enabled': False, 'reason': 'admission_share_review_disabled'}
        if not deterministic_trial_admission and (not (self.arbiter_enabled and self.arbiter_mode == 'llm')):
            return {'enabled': False, 'reason': 'admission_share_review_disabled'}
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return {'enabled': False, 'reason': 'job_missing'}
        if not bool(result.get('share_override_candidate')):
            return {'enabled': False, 'reason': 'no_share_override_candidate'}
        job = self._jobs[job_id]
        now = time.time()
        cooldown_key = f'{job.job_id}:admission_share_review'
        cooldown = max(30.0, float(self.arbiter_proposal_coalesce_window_sec or 0.0))
        last = float(self._last_admission_share_review_emit.get(cooldown_key) or 0.0)
        if last and now - last < cooldown:
            return {'enabled': True, 'reason': 'admission_share_review_cooldown', 'result': result}
        try:
            snapshot = self.resource_runtime.gpu_store.snapshot_active()
        except Exception as exc:
            return {'enabled': False, 'reason': f'gpu_snapshot_failed:{type(exc).__name__}'}
        gpu_ids = [str(x) for x in result.get('gpu_ids') or job.gpu_ids if str(x).strip()]
        sample = self.resource_runtime.sample_gpu_util(gpu_ids=gpu_ids) if gpu_ids else {'available': False, 'reason': 'missing_gpu_ids', 'gpus': []}
        observation = evaluate_admission_share_trial(cfg=self.gpu_share_config, admission_result=result, snapshot=snapshot, gpu_sample=sample if isinstance(sample, dict) else {})
        if not observation.get('enabled'):
            return {'enabled': False, 'reason': str(observation.get('reason') or 'admission_share_not_enabled'), 'observation': observation}
        self._last_admission_share_review_emit[cooldown_key] = now
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event('admission_share_review_observed', payload={'job_id': job.job_id, 'observation': observation, 'admission_result': result}, command_id=job.job_id, lease_id=job.job_id)
        if not observation.get('share_eligible'):
            return {'enabled': True, 'reason': 'admission_share_not_eligible', 'result': result, 'observation': observation}
        proposal = self._admission_share_review_proposal(job, result, observation)
        self._record_resource_review_proposal_event(job, proposal, {'elapsed_sec': 0.0})
        trial_share = observation.get('trial_share') if isinstance(observation.get('trial_share'), dict) else {}
        if deterministic_trial_admission and bool(trial_share.get('enabled')):
            decision = {'action': 'GRANT_SHARED_GPU_LEASE', 'reason': 'hard resource gates passed for deterministic revocable admission trial', 'confidence': 'high', 'source': 'deterministic_trial_admission'}
            arbiter = {'enabled': True, 'action': 'GRANT_SHARED_GPU_LEASE', 'decision': decision, 'feedback': 'RESOURCE_FEEDBACK: SHARED_TRIAL_STARTED because hard_resource_gates_passed; primary_protected=true; secondary_revocable=true.\n'}
        else:
            arbiter = await self.arbiter_decide(job.job_id, proposal=proposal, decision_preview=proposal.get('decision_preview'))
            if not isinstance(arbiter, dict) or not arbiter.get('enabled'):
                return {'enabled': True, 'reason': 'admission_share_arbiter_disabled', 'result': result, 'proposal': proposal}
            decision = arbiter.get('decision') if isinstance(arbiter.get('decision'), dict) else {}
        action = str(arbiter.get('action') or '').upper()
        if action == 'GRANT_SHARED_GPU_LEASE':
            primary_id = str(observation.get('primary_job_id') or '')
            secondary_allowed = observation.get('secondary_allowed') if isinstance(observation.get('secondary_allowed'), dict) else {}
            trial_share = observation.get('trial_share') if isinstance(observation.get('trial_share'), dict) else {}
            grant = self._execute_resource_effect(ResourceEffectKind.GRANT_SHARED, command_id=job.job_id, idempotency_key=f'grant-shared:{primary_id}:{job.job_id}', parameters={'primary_job_id': primary_id, 'secondary_job_id': job.job_id, 'metadata': {'shared_grant_reason': 'admission_task_gpu_share_review', 'shared_effective_slot_weight': float(secondary_allowed.get('effective_slot_weight') or 0.5), 'shared_policy_gate': str(secondary_allowed.get('policy_gate') or 'admission_shared_trial'), 'trial_share': bool(trial_share.get('enabled')), 'trial_mode': str(trial_share.get('mode') or ''), 'trial_initial_observe_sec': float(trial_share.get('initial_observe_sec') or 0.0), 'trial_primary_protected': bool(trial_share.get('primary_protected')), 'admission_immediate_review': True}})
            final = dict(result)
            final.update({'enabled': True, 'acquired': bool(grant.get('acquired')), 'status': 'GRANTED' if grant.get('acquired') else 'PENDING', 'admission_action': 'RUN_NOW' if grant.get('acquired') else 'PENDING', 'reason': str(grant.get('reason') or 'shared_gpu_lease_granted'), 'admission_share_reviewed': True, 'admission_shared_trial': bool(grant.get('acquired')), 'admission_share_action': action, 'shared_lease_grant': grant})
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event('admission_share_decision', payload={'job_id': job.job_id, 'action': action, 'decision': decision, 'grant': grant, 'proposal_id': proposal.get('proposal_id')}, proposal_id=str(proposal.get('proposal_id') or ''), command_id=job.job_id, lease_id=job.job_id)
            return {'enabled': True, 'action': action, 'result': final, 'proposal': proposal, 'arbiter': arbiter}
        if action == 'DENY_SHARE_USE_CPU_SUPPORT':
            final = self._admission_share_cpu_support_result(job, result, reason=str(decision.get('reason') or 'shared GPU trial was not approved'), decision=decision)
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event('admission_share_decision', payload={'job_id': job.job_id, 'action': action, 'decision': decision, 'proposal_id': proposal.get('proposal_id')}, proposal_id=str(proposal.get('proposal_id') or ''), command_id=job.job_id, lease_id=job.job_id)
            return {'enabled': True, 'action': action, 'result': final, 'proposal': proposal, 'arbiter': arbiter}
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event('admission_share_decision', payload={'job_id': job.job_id, 'action': action or 'CONTINUE_SHARED_OBSERVE', 'decision': decision, 'proposal_id': proposal.get('proposal_id')}, proposal_id=str(proposal.get('proposal_id') or ''), command_id=job.job_id, lease_id=job.job_id)
        final = dict(result)
        final['admission_share_reviewed'] = True
        final['admission_share_action'] = action or 'CONTINUE_SHARED_OBSERVE'
        return {'enabled': True, 'action': action or 'CONTINUE_SHARED_OBSERVE', 'result': final, 'proposal': proposal, 'arbiter': arbiter}

    async def admission_decide(self, job_id: str | None, *, admission_result: dict[str, Any] | None=None) -> dict[str, Any]:
        if not self.admission_llm_enabled or self.admission_llm_mode == 'off' or self.admission_decider is None:
            return {'enabled': False, 'reason': 'admission_llm_disabled'}
        if not job_id or job_id not in self._jobs:
            return {'enabled': False, 'reason': 'job_missing'}
        result = dict(admission_result or {})
        if not self.admission_router.should_review(result, mode=self.admission_llm_mode):
            return {'enabled': False, 'reason': 'rule_confident'}
        job = self._jobs[job_id]
        card = self._admission_task_card(job, result)
        raw: dict[str, Any] = {}
        llm_correlation: dict[str, Any] = {}
        source = 'resource_admission_llm'
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event('admission_llm_input', payload={'actor_id': 'resource_admission_arbiter', 'cache_namespace': 'resource_admission', 'task_card': card, 'rule_result': result}, command_id=job.job_id, lease_id=job.job_id)
        try:
            maybe = self.admission_decider(card, result)
            if inspect.isawaitable(maybe):
                maybe = await asyncio.wait_for(maybe, timeout=self.admission_llm_timeout_sec)
            if isinstance(maybe, str):
                llm_correlation = dict(getattr(maybe, 'correlation', {}) or {})
                raw = parse_admission_decision_text(maybe)
            elif isinstance(maybe, dict):
                raw = dict(maybe)
        except Exception as exc:
            source = 'admission_llm_error'
            raw = {'action': result.get('admission_action') or result.get('status') or 'PENDING', 'reason': f'admission_llm_failed:{type(exc).__name__}', 'confidence': 'low'}
        decision_record = self.admission_router.decide(raw, task_card=card, rule_result=result, mode=self.admission_llm_mode, source=source)
        decision = dict(decision_record.selected)
        if llm_correlation:
            decision['llm_correlation'] = llm_correlation
        if not decision_record.safety_veto_preserved:
            decision = {**decision, 'action': 'PENDING', 'status': 'PENDING', 'admission_action': 'PENDING', 'reason': 'admission source attempted to bypass atomic lease safety veto'}
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event('admission_policy_diff', payload={'record_id': decision_record.record_id, 'selected_source': decision_record.selected_source, 'equal': decision_record.diff.equal, 'changed_fields': list(decision_record.diff.changed_fields), 'safety_veto_preserved': decision_record.safety_veto_preserved}, command_id=job.job_id, lease_id=job.job_id)
        result_for_apply = result
        action = str(decision.get('admission_action') or decision.get('action') or '').upper()
        if action in {'RUN_NOW', 'OBSERVE_THEN_RUN'} and (not result.get('acquired')) and (self.resource_runtime is not None):
            requested_gpu_ids = [str(x) for x in decision.get('gpu_ids') or [] if str(x).strip()]
            opportunity = card.get('admission_opportunity') if isinstance(card.get('admission_opportunity'), dict) else {}
            if not requested_gpu_ids:
                requested_gpu_ids = [str(x) for x in opportunity.get('grantable_gpu_ids') or result.get('candidate_physical_gpus') or result.get('allowed_physical_gpus') or job.gpu_ids if str(x).strip()]
            request_count = len(requested_gpu_ids) if decision.get('gpu_ids') else int(result.get('requested_gpu_count') or job.gpu_request_count or len(requested_gpu_ids) or 1)
            grant = self.resource_runtime.grant_llm_admission_lease(job_id=job.job_id, resource_class=str(result.get('policy_resource_class') or result.get('resource_class') or job.resource_class), gpu_ids=requested_gpu_ids, request_count=request_count, metadata={'command_digest': job.command_digest, 'command_excerpt': str(job.command or '')[:2000], 'entrypoint': self._command_entrypoint(job.command), 'value_hint': dict(job.value_hint or {}), 'admission_llm_decision_id': str(decision.get('decision_id') or '')}, observe_then_run=action == 'OBSERVE_THEN_RUN', observe_sec=float(decision.get('observe_sec') or 0.0))
            if grant.get('acquired'):
                result_for_apply = {**result, **grant, 'acquired': True}
            else:
                result_for_apply = {**result, 'admission_llm_lease_attempt': grant, 'acquired': False}
                decision = {**decision, 'action': 'PENDING', 'status': 'PENDING', 'admission_action': 'PENDING', 'reason': 'LLM approved GPU start, but atomic lease acquire failed: ' + str(grant.get('reason') or 'gpu_slot_unavailable')}
        final_result = apply_admission_decision(result_for_apply, decision)
        if str(final_result.get('status') or '').upper() in {'PENDING', 'REPLAN', 'DEFERRED', 'DENIED_REPLAN'}:
            allowed_classes = [str(x) for x in final_result.get('allowed_classes') or [] if str(x).strip()]
            raw_feedback = str(final_result.get('feedback') or '')
            if not raw_feedback:
                raw_feedback = self._resource_feedback_text(status=str(final_result.get('status') or 'PENDING'), reason=str(final_result.get('reason') or 'resource_admission_review'), scope=str(final_result.get('scope') or 'per_gpu'), resource_mode=str(final_result.get('resource_mode') or 'YELLOW'), blocked_class=job.resource_class, gpu_ids=job.gpu_ids, allowed_classes=allowed_classes, eta_next_train_sec=float(final_result.get('eta_next_train_sec') or 0.0), eta_confidence=str(final_result.get('eta_confidence') or 'low'), unlock_condition=str(final_result.get('unlock_condition') or 'resource_context_changed'), blocked_until_unlock=bool(final_result.get('blocked_until_unlock') if final_result.get('blocked_until_unlock') is not None else True))
            feedback_state = self._dedupe_resource_feedback_for_agent(job, feedback=raw_feedback, status=str(final_result.get('status') or 'PENDING'), reason=str(final_result.get('reason') or 'resource_admission_review'), scope=str(final_result.get('scope') or 'per_gpu'), resource_mode=str(final_result.get('resource_mode') or 'YELLOW'), blocked_class=job.resource_class, gpu_ids=job.gpu_ids, allowed_classes=allowed_classes, unlock_condition=str(final_result.get('unlock_condition') or 'resource_context_changed'), blocked_until_unlock=bool(final_result.get('blocked_until_unlock') if final_result.get('blocked_until_unlock') is not None else True))
            final_result['feedback'] = str(feedback_state.get('feedback') or '')
            final_result['feedback_suppressed'] = bool(feedback_state.get('feedback_suppressed'))
            final_result['feedback_state_key'] = feedback_state.get('feedback_state_key')
            final_result['resource_feedback_repeated_count'] = feedback_state.get('repeated_count')
            if not final_result['feedback_suppressed']:
                self._remember_resource_feedback(job, status=str(final_result.get('status') or 'PENDING'), reason=str(final_result.get('reason') or 'resource_admission_review'), resource_mode=str(final_result.get('resource_mode') or 'YELLOW'), blocked_class=job.resource_class, allowed_classes=allowed_classes, eta_next_train_sec=float(final_result.get('eta_next_train_sec') or 0.0), retry_after_sec=float(final_result.get('retry_after_sec') or 0.0) if final_result.get('retry_after_sec') is not None else None, post_feedback_action=str(final_result.get('post_feedback_action') or ''))
        payload = {'actor_id': 'resource_admission_arbiter', 'cache_namespace': 'resource_admission', 'task_card': card, 'rule_result': result, 'decision': decision, 'final_result': final_result}
        if self.resource_runtime is not None:
            self.resource_runtime.record_resource_event('admission_llm_decision', payload=payload, command_id=job.job_id, lease_id=job.job_id)
        self._emit('resource_admission_llm_decision', job, status=str(final_result.get('status') or decision.get('status') or 'reviewed').lower(), payload=payload)
        return {'enabled': True, 'decision': decision, 'result': final_result}

    def queue_wait_heartbeat(self, job_id: str | None, *, elapsed_sec: float, **_: Any) -> None:
        if not job_id or job_id not in self._jobs or self.resource_runtime is None:
            return
        heartbeat = self.resource_runtime.queue_wait_heartbeat(job_id=job_id, elapsed_sec=float(elapsed_sec or 0.0))
        if not heartbeat.get('emit'):
            return
        self._emit('resource_gpu_queue_heartbeat', self._jobs[job_id], status='pending', payload={'elapsed_sec': float(elapsed_sec or 0.0)})

    def lease_env_updates(self, job_id: str | None, **_: Any) -> dict[str, str]:
        if not job_id or self.resource_runtime is None:
            return {}
        return self.resource_runtime.env_updates(job_id=job_id)

    def queue_timeout(self, job_id: str | None, *, elapsed_sec: float, reason: str, **_: Any) -> None:
        if not job_id or job_id not in self._jobs:
            return
        job = self._jobs.pop(job_id)
        self._gpu_queue_timeout_count += 1
        release: dict[str, Any] = {}
        pressure: dict[str, Any] = {}
        if self.resource_runtime is not None:
            release = self._execute_resource_effect(ResourceEffectKind.QUEUE_TIMEOUT, command_id=job_id, idempotency_key=f'queue-timeout:{job_id}', parameters={'job_id': job_id, 'elapsed_sec': float(elapsed_sec or 0.0), 'reason': str(reason or 'queue_timeout')})
            pressure = self.resource_runtime.record_queue_timeout_pressure(job_id=job_id, resource_class=job.resource_class, gpu_ids=job.gpu_ids, elapsed_sec=float(elapsed_sec or 0.0), reason=str(reason or 'queue_timeout'), command_digest=job.command_digest)
        self._remember_resource_feedback(job, status='DENIED_REPLAN', reason=str(reason or 'queue_timeout'), resource_mode='RED', blocked_class=job.resource_class, allowed_classes=[RESOURCE_PURE_TT_CPU, RESOURCE_GPU_TT_LIGHT, 'readonly_cpu', 'light_cpu'], eta_next_train_sec=420.0, cooldown_sec=120.0)
        self._emit('resource_gpu_queue_timeout', job, status='timeout', payload={'elapsed_sec': float(elapsed_sec or 0.0), 'reason': str(reason or 'queue_timeout'), 'released': bool(release.get('released')), 'gpu_queue_timeout_count': self._gpu_queue_timeout_count, 'pressure': pressure})
