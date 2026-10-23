"""Resource observer responsibility: advisory reuse, arbiter decisions, kill revalidation, and event projection.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""
from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    RESOURCE_REVIEW_KILL,
    RESOURCE_REVIEW_NO_ACTION,
    RESOURCE_REVIEW_TIMEBOX,
    ROUTE_VALUE,
    ResourceReviewEvent,
    ResourceReviewEventKind,
    STALL,
    TERMINATING_ACTIONS,
    TIMEBOX_EXPIRED,
    apply_control_profile_to_decision_delay,
    asyncio,
    build_kill_intent_snapshot,
    compute_timebox_sec,
    enforce_arbiter_kill_gate,
    enforce_proposal_action_allowlist,
    enforce_repeated_stall_escalation,
    evaluate_kill_intent_revalidation,
    fallback_policy_decision,
    inspect,
    json,
    main_agent_feedback,
    new_review_state,
    normalize_arbiter_decision,
    normalize_clear_on,
    normalize_value_review_outcome,
    parse_arbiter_decision_text,
    proposal_advisory_fact_conflicts,
    time,
)


class ArbiterCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def _advisory_scope_key(self, proposal: dict[str, Any]) -> str:
        progress = proposal.get('progress_snapshot') if isinstance(proposal.get('progress_snapshot'), dict) else {}
        metric = proposal.get('resource_metric_value') if isinstance(proposal.get('resource_metric_value'), dict) else {}
        return str(metric.get('metric_scope_key') or progress.get('metric_scope_key') or '')

    def _reusable_main_agent_advisory(self, proposal: dict[str, Any], *, job_key: str, now: float) -> dict[str, Any]:
        cached = self._last_main_agent_advisory.get(job_key)
        if not isinstance(cached, dict):
            return {}
        advisory = cached.get('advisory') if isinstance(cached.get('advisory'), dict) else {}
        preference = str(advisory.get('preference') or '').strip().lower()
        confidence = str(advisory.get('confidence') or '').strip().lower()
        stop_preferences = {'safe_to_stop', 'kill_and_replan', 'stop', 'stop_and_replan', 'replan'}
        if preference not in stop_preferences or confidence != 'high':
            return {}
        ttl_sec = max(0.0, self._nested_float(advisory, 'ttl_sec', 600.0))
        if ttl_sec <= 0.0 or now - float(cached.get('captured_at') or 0.0) > ttl_sec:
            return {}
        cached_scope = str(cached.get('metric_scope_key') or '')
        current_scope = self._advisory_scope_key(proposal)
        if cached_scope and current_scope and (cached_scope != current_scope):
            return {}
        cached_metric_digest = str(cached.get('metric_history_digest') or '')
        current_metric_digest = self._proposal_metric_history_digest(proposal)
        if cached_metric_digest and current_metric_digest and (cached_metric_digest != current_metric_digest):
            return {}
        progress = proposal.get('progress_snapshot') if isinstance(proposal.get('progress_snapshot'), dict) else {}
        metric = proposal.get('resource_metric_value') if isinstance(proposal.get('resource_metric_value'), dict) else {}
        if progress.get('near_submission') or metric.get('useful') is True:
            return {}
        if proposal_advisory_fact_conflicts(proposal, advisory):
            return {}
        reused = dict(advisory)
        reused['source'] = 'main_agent_advisory_cache'
        reused['reused'] = True
        reused['reuse_age_sec'] = max(0.0, now - float(cached.get('captured_at') or now))
        return reused

    async def _maybe_attach_main_agent_advisory(self, proposal: dict[str, Any], *, job_key: str) -> dict[str, Any]:
        if not (self.main_agent_advisory_enabled and self.main_agent_advisory_decider is not None):
            return proposal
        proposal = self._attach_budget_priority(proposal)
        if proposal.get('main_agent_advisory'):
            return proposal
        now = time.time()
        reused_advisory = self._reusable_main_agent_advisory(proposal, job_key=job_key, now=now)
        if reused_advisory:
            out = dict(proposal)
            out['main_agent_advisory'] = reused_advisory
            proposal_id = str(out.get('proposal_id') or '')
            if self.resource_runtime is not None:
                self.resource_runtime.update_active_resource_proposal(proposal_id, out)
                self.resource_runtime.record_resource_event('main_agent_advisory_reused', payload={'proposal_id': proposal_id, 'job_id': job_key, 'advisory': reused_advisory}, trace_id=f'trace_{proposal_id}' if proposal_id else '', proposal_id=proposal_id, command_id=job_key, lease_id=job_key)
            return out
        defer_reason = self._llm_budget_defer_reason(proposal, job_key, advisory=True)
        if defer_reason:
            self._record_llm_budget_deferred_event(proposal, defer_reason, advisory=True)
            return proposal
        last = float(self._last_advisory_emit.get(job_key) or 0.0)
        if job_key and last and (now - last < self.main_agent_advisory_min_interval_sec):
            return proposal
        estimate = self._estimate_llm_tokens(proposal)
        block = self._llm_budget_blocked(job_key, estimated_tokens=estimate, advisory=True)
        if block.get('blocked'):
            self._record_llm_budget_event(proposal, block, advisory=True)
            return proposal
        self._last_advisory_emit[job_key] = now
        self._note_llm_call(job_key, estimated_tokens=estimate, advisory=True, proposal=proposal)
        advisory: dict[str, Any]
        try:
            maybe = self.main_agent_advisory_decider(dict(proposal))
            if inspect.isawaitable(maybe):
                maybe = await asyncio.wait_for(maybe, timeout=self.main_agent_advisory_timeout_sec)
            if isinstance(maybe, str):
                try:
                    parsed = json.loads(maybe)
                except json.JSONDecodeError:
                    parsed = {'preference': 'unknown', 'reason': maybe[:500], 'confidence': 'low'}
                advisory = parsed if isinstance(parsed, dict) else {'preference': 'unknown', 'reason': str(maybe)[:500], 'confidence': 'low'}
            elif isinstance(maybe, dict):
                advisory = dict(maybe)
            else:
                advisory = {'preference': 'unknown', 'reason': 'empty advisory response', 'confidence': 'low'}
        except Exception as exc:
            advisory = {'preference': 'unknown', 'reason': f'main_agent_advisory_failed:{type(exc).__name__}', 'confidence': 'low'}
        audit_payload = advisory.get('_audit') if isinstance(advisory.get('_audit'), dict) else None
        audit_status = str((audit_payload or {}).get('status') or '').strip()
        clean = {'preference': str(advisory.get('preference') or 'unknown')[:80], 'confidence': str(advisory.get('confidence') or 'low')[:40], 'reason': str(advisory.get('reason') or '')[:600], 'ttl_sec': self._nested_float(advisory, 'ttl_sec', 600.0), 'source': 'main_agent_advisory', 'advisory_status': str(advisory.get('advisory_status') or audit_status or 'captured')[:80]}
        for optional_key in ('advisory_mode', 'blocked_state', 'memory_edit_applied', 'memory_edit_removed_tokens', 'advisory_tool_call_rejected', 'commitment', 'expected_next_artifact', 'observed_phase', 'observed_checkpoint', 'observed_resource_state', 'observed_metric_status'):
            if optional_key in advisory:
                clean[optional_key] = advisory.get(optional_key)
        self._last_main_agent_advisory[job_key] = {'advisory': dict(clean), 'captured_at': now, 'metric_scope_key': self._advisory_scope_key(proposal), 'metric_history_digest': self._proposal_metric_history_digest(proposal)}
        out = dict(proposal)
        out['main_agent_advisory'] = clean
        proposal_id = str(out.get('proposal_id') or '')
        if self.resource_runtime is not None:
            self.resource_runtime.update_active_resource_proposal(proposal_id, out)
            if audit_payload:
                self.resource_runtime.record_resource_event('resource_advisory_audit', payload={**audit_payload, 'proposal_id': proposal_id, 'job_id': job_key}, trace_id=f'trace_{proposal_id}' if proposal_id else '', proposal_id=proposal_id, command_id=job_key, lease_id=job_key)
            self.resource_runtime.record_resource_event('main_agent_advisory', payload={'proposal_id': proposal_id, 'job_id': job_key, 'advisory': clean, 'decision_applied': False}, trace_id=f'trace_{proposal_id}' if proposal_id else '', proposal_id=proposal_id, command_id=job_key, lease_id=job_key)
        return out

    def _computed_review_timebox_sec(self, job_id: str, proposal: dict[str, Any], *, clear_on: str) -> float:
        state = self._review_states.get(job_id) or new_review_state(job_id)
        progress = proposal.get('progress_snapshot') if isinstance(proposal.get('progress_snapshot'), dict) else {}
        budget = proposal.get('budget_context') if isinstance(proposal.get('budget_context'), dict) else {}
        contention = proposal.get('contention_context') if isinstance(proposal.get('contention_context'), dict) else {}
        remaining = self._float_or_none(budget.get('remaining_budget_sec'))
        if remaining is None:
            remaining = self._float_or_none(progress.get('deadline_remaining_sec')) or 0.0
        active_waiter_pressure = bool(contention.get('active_waiter_pressure') or int(contention.get('blocked_worker_count') or 0) > 0 or bool(proposal.get('waiters')))
        return compute_timebox_sec(state, clear_on=clear_on, heartbeat_sec=self.review_heartbeat_sec, fallback_next_review_sec=max(self.review_heartbeat_sec, float(self.review_config.progress_event_min_windows or 1) * self.review_heartbeat_sec), min_timebox_sec=float(self.review_config.min_timebox_sec or 60.0), max_timebox_sec=float(self.review_config.max_timebox_sec or 1800.0), remaining_budget_sec=float(remaining or 0.0), timebox_budget_fraction=float(self.review_config.timebox_budget_fraction or 0.1), active_waiter_pressure=active_waiter_pressure)

    async def arbiter_decide(self, job_id: str | None, *, proposal: dict[str, Any] | None=None, decision_preview: dict[str, Any] | None=None) -> dict[str, Any]:
        if not self.arbiter_enabled or self.arbiter_mode == 'off':
            return {'enabled': False, 'action': 'DENY_KILL', 'terminate': False}
        prop = dict(proposal or {})
        if not prop:
            preview = decision_preview if isinstance(decision_preview, dict) else {}
            prop = self._proposal_from_job(job_id, expected_proposal_type=str(preview.get('proposal_type') or ''), expected_reason=str(preview.get('reason_code') or preview.get('reason') or ''))
        if not prop:
            return {'enabled': False, 'action': 'OBSERVE_MORE', 'terminate': False, 'reason': 'proposal_missing'}
        job_key = self._proposal_job_id(prop, fallback=job_id)
        prop = self._attach_budget_priority(prop)
        final_escalation = self._final_resource_review_escalation(prop, job_key)
        if final_escalation:
            prop = self._with_final_resource_review_escalation(prop, final_escalation)
        raw: dict[str, Any] = {} if final_escalation else self._proof_window_override_raw_decision(prop, job_key) or {}
        source = 'proof_window_override' if raw else 'policy_fallback'
        if not raw and (not final_escalation):
            prop = await self._maybe_attach_main_agent_advisory(prop, job_key=job_key)
            prop = self._attach_review_history(prop, now=time.time())
        elif final_escalation:
            prop = self._attach_review_history(prop, now=time.time())
        if not raw and self.arbiter_mode == 'llm' and (self.arbiter_decider is not None):
            raw, source = await self._arbiter_llm_raw_decision(prop, job_key=job_key, final_escalation=bool(final_escalation))
        if not raw:
            raw = fallback_policy_decision(prop)
            source = 'policy_fallback'
        normalized, action = self._normalize_arbiter_result(raw, proposal=prop, source=source, job_id=job_id)
        outcome = normalize_value_review_outcome(action, prop, canonical_outcome=str(normalized.get('canonical_outcome') or ''), clear_on=str(normalized.get('clear_on') or ''))
        execution_outcome = outcome.outcome
        normalized['raw_action'] = action
        normalized['execution_outcome'] = execution_outcome
        normalized['value_review_outcome'] = outcome.to_json()
        review_job_key = job_key or str(job_id or '')
        self._advance_arbiter_review_state(review_job_key, proposal=prop, normalized=normalized, outcome=outcome, action=action)
        if self.resource_runtime is not None:
            proposal_id_for_event = str(normalized.get('proposal_id') or prop.get('proposal_id') or '')
            self.resource_runtime.record_resource_event('resource_review_outcome', payload={'job_id': review_job_key, 'proposal_id': proposal_id_for_event, 'action': execution_outcome, 'execution_outcome': execution_outcome, 'raw_action': action, 'canonical_outcome': str(normalized.get('canonical_outcome') or ''), 'reason_code': str(normalized.get('reason_code') or ''), 'reason': str(normalized.get('reason') or ''), 'computed_timebox_sec': normalized.get('computed_timebox_sec'), 'outcome': outcome.to_json(), 'feedback_suppressed': bool(outcome.suppress_main_agent_feedback)}, trace_id=f'trace_{proposal_id_for_event}' if proposal_id_for_event else '', proposal_id=proposal_id_for_event, decision_id=str(normalized.get('decision_id') or ''), command_id=review_job_key, lease_id=review_job_key)
        self._record_arbiter_decision_events(prop, normalized, decision_preview=decision_preview or {})
        feedback = '' if outcome.suppress_main_agent_feedback else main_agent_feedback(normalized, proposal=prop)
        return {'enabled': True, 'action': action, 'raw_action': action, 'execution_outcome': execution_outcome, 'terminate': execution_outcome == RESOURCE_REVIEW_KILL, 'observe_more': False, 'feedback': feedback, 'suppress_main_agent_feedback': bool(outcome.suppress_main_agent_feedback), 'value_review_outcome': outcome.outcome, 'computed_timebox_sec': normalized.get('computed_timebox_sec'), 'decision': normalized, 'proposal_id': normalized.get('proposal_id'), 'decision_id': normalized.get('decision_id')}

    async def _arbiter_llm_raw_decision(self, proposal: dict[str, Any], *, job_key: str, final_escalation: bool) -> tuple[dict[str, Any], str]:
        estimate = self._estimate_llm_tokens(proposal)
        budget_block = {'blocked': False} if final_escalation else self._llm_budget_blocked(job_key, estimated_tokens=estimate, advisory=False)
        if budget_block.get('blocked'):
            self._record_llm_budget_event(proposal, budget_block, advisory=False)
            return {'action': 'OBSERVE_MORE', 'reason': f"llm_budget_exhausted:{budget_block.get('reason') or 'unknown'}", 'confidence': 'low', 'observe_more_sec': 600.0}, 'llm_budget_guard'
        defer_reason = '' if final_escalation else self._llm_budget_defer_reason(proposal, job_key, advisory=False)
        if defer_reason:
            self._record_llm_budget_deferred_event(proposal, defer_reason, advisory=False)
            return {'action': 'OBSERVE_MORE', 'reason': f'llm_budget_deferred:{defer_reason}', 'confidence': 'low', 'observe_more_sec': 600.0}, 'llm_budget_deferred'
        self._note_llm_call(job_key, estimated_tokens=estimate, advisory=False, proposal=proposal)
        try:
            return await self._invoke_arbiter_decider(proposal), 'resource_arbiter_llm'
        except asyncio.TimeoutError:
            if self.resource_runtime is not None:
                proposal_id = str(proposal.get('proposal_id') or '')
                self.resource_runtime.record_resource_event('resource_arbiter_timeout', payload={'proposal_id': proposal_id, 'job_id': job_key, 'timeout_sec': self.arbiter_timeout_sec, 'fallback_action': 'OBSERVE_MORE'}, trace_id=f'trace_{proposal_id}' if proposal_id else '', proposal_id=proposal_id, command_id=job_key, lease_id=job_key)
            return {'action': 'OBSERVE_MORE', 'reason': 'arbiter_llm_timeout', 'confidence': 'low', 'observe_more_sec': 300.0}, 'llm_timeout_fallback'
        except Exception as exc:
            return {'action': 'OBSERVE_MORE', 'reason': f'arbiter_llm_failed:{type(exc).__name__}', 'confidence': 'low'}, 'llm_error_fallback'

    async def _invoke_arbiter_decider(self, proposal: dict[str, Any]) -> dict[str, Any]:
        if inspect.iscoroutinefunction(self.arbiter_decider):
            maybe = await asyncio.wait_for(self.arbiter_decider(proposal), timeout=self.arbiter_timeout_sec)
        else:
            maybe = await asyncio.wait_for(asyncio.to_thread(self.arbiter_decider, proposal), timeout=self.arbiter_timeout_sec)
            if inspect.isawaitable(maybe):
                maybe = await asyncio.wait_for(maybe, timeout=self.arbiter_timeout_sec)
        if isinstance(maybe, str):
            return parse_arbiter_decision_text(maybe)
        return dict(maybe) if isinstance(maybe, dict) else {}

    def _normalize_arbiter_result(self, raw: dict[str, Any], *, proposal: dict[str, Any], source: str, job_id: str | None) -> tuple[dict[str, Any], str]:
        normalized = normalize_arbiter_decision(raw, proposal=proposal, source=source)
        for key in ('proof_window_source', 'proof_window_retry', 'proof_window_override', 'llm_correlation'):
            if key in raw and key not in normalized:
                normalized[key] = raw.get(key)
        normalized = enforce_repeated_stall_escalation(normalized, proposal)
        normalized = enforce_arbiter_kill_gate(normalized, proposal, min_windows=self.arbiter_min_progress_windows, require_high_confidence=self.arbiter_kill_requires_high_confidence)
        normalized = enforce_proposal_action_allowlist(normalized, proposal)
        normalized = apply_control_profile_to_decision_delay(normalized, profile=self.control_profile)
        normalized = self._advisory_commitment_timebox_decision(proposal, normalized)
        action = str(normalized.get('action') or 'OBSERVE_MORE').upper()
        if action == 'KILL_AND_REPLAN':
            gate_facts = normalized.get('gate') if isinstance(normalized.get('gate'), dict) else {}
            cadence = proposal.get('research_cadence') if isinstance(proposal.get('research_cadence'), dict) else {}
            value_review_support = bool(gate_facts.get('advisory_support') or gate_facts.get('structured_opportunity_cost_support') or proposal.get('structured_opportunity_cost_support') or cadence.get('violation'))
            normalized['kill_intent_snapshot'] = build_kill_intent_snapshot(proposal.get('progress_snapshot') if isinstance(proposal.get('progress_snapshot'), dict) else {}, proposal.get('resource_metric_value') if isinstance(proposal.get('resource_metric_value'), dict) else {}, kill_basis='value_stagnation' if value_review_support else 'liveness_stall')
        if action == 'MARK_STALLED_NO_KILL' and job_id and job_id in self._jobs:
            job = self._jobs[job_id]
            job.stalled_mark_count += 1
            normalized['stalled_mark_count'] = job.stalled_mark_count
        return normalized, action

    def _advance_arbiter_review_state(self, review_job_key: str, *, proposal: dict[str, Any], normalized: dict[str, Any], outcome: Any, action: str) -> None:
        if not review_job_key or review_job_key not in self._review_states:
            return
        current_state = self._review_states[review_job_key]
        review_machine = self._review_machine_for(review_job_key, current_state)
        review_event_id = str(normalized.get('proposal_id') or proposal.get('proposal_id') or f'outcome-{review_machine.version + 1}')
        if outcome.outcome == RESOURCE_REVIEW_KILL:
            self._review_states[review_job_key] = review_machine.advance(ResourceReviewEvent(kind=ResourceReviewEventKind.APPLY_KILL), event_id=f'{review_job_key}:{review_event_id}:kill', expected_version=review_machine.version).current
            return
        if outcome.outcome == RESOURCE_REVIEW_TIMEBOX:
            clear_on = normalize_clear_on(outcome.clear_on)
            computed_timebox_sec = self._computed_review_timebox_sec(review_job_key, proposal, clear_on=clear_on)
            normalized['computed_timebox_sec'] = computed_timebox_sec
            contention = proposal.get('contention_context') if isinstance(proposal.get('contention_context'), dict) else {}
            active_waiter_pressure = bool(contention.get('active_waiter_pressure') or int(contention.get('blocked_worker_count') or 0) > 0 or bool(proposal.get('waiters')))
            proof_source = str(normalized.get('proof_window_source') or '').strip() or ('negotiated' if normalized.get('advisory_commitment_timebox') else 'system_fixed')
            self._review_states[review_job_key] = review_machine.advance(ResourceReviewEvent(kind=ResourceReviewEventKind.APPLY_TIMEBOX, parameters={'timebox_sec': computed_timebox_sec, 'heartbeat_sec': self.review_heartbeat_sec, 'clear_on': clear_on, 'active_waiter_pressure': active_waiter_pressure, 'proof_window_source': proof_source, 'counts_as_proof_failure': not active_waiter_pressure}), event_id=f'{review_job_key}:{review_event_id}:timebox', expected_version=review_machine.version).current
            return
        if not outcome.suppress_main_agent_feedback:
            return
        if current_state.in_timebox:
            normalized['active_timebox_preserved'] = True
            normalized.setdefault('active_timebox_id', current_state.timebox_id)
            normalized.setdefault('active_timebox_windows', int(current_state.timebox_windows or 0))
            normalized.setdefault('active_timebox_deadline_windows', int(current_state.timebox_deadline_windows or 0))
            self._review_states[review_job_key] = current_state
            return
        delay = normalized.get('observe_more_sec') if action in {'OBSERVE_MORE', 'MARK_STALLED_NO_KILL'} else normalized.get('ttl_sec')
        contention = proposal.get('contention_context') if isinstance(proposal.get('contention_context'), dict) else {}
        active_waiter_pressure = bool(contention.get('active_waiter_pressure') or int(contention.get('blocked_worker_count') or 0) > 0 or bool(proposal.get('waiters')) or current_state.active_waiter_pressure)
        boundary = proposal.get('resource_review_boundary') if isinstance(proposal.get('resource_review_boundary'), dict) else {}
        boundary_kind = str(boundary.get('kind') or '').strip().lower()
        reason_code = str(proposal.get('reason_code') or normalized.get('reason_code') or '').strip().lower()
        proof_window_reason = bool(boundary_kind in {STALL, ROUTE_VALUE, TIMEBOX_EXPIRED} or any(token in reason_code for token in ('low_progress', 'stalled', 'timebox_expired', 'sm_route_value', 'sm_timebox_expired', 'sm_stall')))
        self._review_states[review_job_key] = review_machine.advance(ResourceReviewEvent(kind=ResourceReviewEventKind.APPLY_NO_ACTION, parameters={'observe_more_sec': float(delay or 600.0), 'heartbeat_sec': self.review_heartbeat_sec, 'success_condition': 'useful_progress', 'counts_as_proof_failure': bool(proof_window_reason and (not active_waiter_pressure))}), event_id=f'{review_job_key}:{review_event_id}:no_action', expected_version=review_machine.version).current

    def revalidate_kill_intent(self, job_id: str | None, *, arbiter_decision: dict[str, Any] | None=None, elapsed_sec: float, stdout_age_sec: float, stdout_lines: int=0, stdout_bytes: int=0, metric_history_text: str='', metric_history_line_count: int=0, saw_training_progress: bool=False, saw_final_score: bool=False, current_phase: str='', process_tree_cpu: dict[str, Any] | None=None, invalid_metric_events: int=0, zero_score_events: int=0, last_invalid_metric_text: str='', last_zero_score_text: str='', terminal_signal_events: int=0, terminal_signal_kind: str='', last_terminal_signal_text: str='', deadline_event: bool=False, deadline_remaining_sec: float=0.0, finalization_reserve_sec: float=0.0, **_: Any) -> dict[str, Any]:
        decision = dict(arbiter_decision or {})
        original = decision.get('kill_intent_snapshot')
        if not job_id or job_id not in self._jobs:
            return {'enabled': False, 'allow_kill': True, 'reason': 'job_missing'}
        job = self._jobs[job_id]
        signal = self._intervention_signal(elapsed_sec=float(elapsed_sec or 0.0), stdout_age_sec=stdout_age_sec, stdout_lines=stdout_lines, stdout_bytes=stdout_bytes, metric_history_text=metric_history_text, metric_history_line_count=metric_history_line_count, saw_training_progress=saw_training_progress, saw_final_score=saw_final_score, current_phase=current_phase, process_tree_cpu=process_tree_cpu, invalid_metric_events=invalid_metric_events, zero_score_events=zero_score_events, last_invalid_metric_text=last_invalid_metric_text, last_zero_score_text=last_zero_score_text, terminal_signal_events=terminal_signal_events, terminal_signal_kind=terminal_signal_kind, last_terminal_signal_text=last_terminal_signal_text, deadline_event=deadline_event, deadline_remaining_sec=deadline_remaining_sec, finalization_reserve_sec=finalization_reserve_sec)
        self._attach_resource_metric_value_assessment(job, signal)
        self._update_job_progress_signal(job, signal, elapsed_sec=float(elapsed_sec or 0.0))
        fresh_progress = self._progress_snapshot_for_job(job, signal, elapsed_sec=float(elapsed_sec or 0.0))
        fresh_metric = signal.get('resource_metric_value') if isinstance(signal.get('resource_metric_value'), dict) else {}
        original_facts = original if isinstance(original, dict) else {}
        fresh = build_kill_intent_snapshot(fresh_progress, fresh_metric, kill_basis=str(original_facts.get('kill_basis') or ''))
        gate = decision.get('gate') if isinstance(decision.get('gate'), dict) else {}
        result = evaluate_kill_intent_revalidation(original if isinstance(original, dict) else {}, fresh, hard_safety=str(gate.get('kill_class') or '') == 'hard_safety')
        result.update({'enabled': True, 'fresh_snapshot': fresh})
        if not result.get('allow_kill'):
            state = self._review_states.get(job_id)
            if state is not None:
                machine = self._review_machine_for(job_id, state)
                proposal_id = str(decision.get('proposal_id') or f'stale-{machine.version + 1}')
                self._review_states[job_id] = machine.advance(ResourceReviewEvent(kind=ResourceReviewEventKind.RESET_STALE_KILL), event_id=f'{job_id}:{proposal_id}:stale_kill_reset', expected_version=machine.version).current
            self._emit('resource_kill_intent_stale', job, status='denied', payload=result)
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event('resource_kill_intent_stale', payload={'job_id': job_id, **result}, command_id=job_id, lease_id=job_id)
        return result

    def _proposal_from_job(self, job_id: str | None, *, expected_proposal_type: str='', expected_reason: str='') -> dict[str, Any] | None:
        if not job_id or self.resource_runtime is None:
            return None
        try:
            state = self.resource_runtime.unified_store._read_state_unlocked()
        except Exception:
            return None
        active = state.get('active_proposals') if isinstance(state, dict) else {}
        if not isinstance(active, dict):
            return None
        job_key = str(job_id)
        expected_type = str(expected_proposal_type or '').strip()
        expected_reason_text = str(expected_reason or '').strip()
        for proposal in active.values():
            if not isinstance(proposal, dict):
                continue
            preview = proposal.get('decision_preview') if isinstance(proposal.get('decision_preview'), dict) else {}
            command_id = str(proposal.get('command_id') or proposal.get('job_id') or preview.get('job_id') or '')
            if command_id != job_key and job_key not in str(proposal.get('proposal_id') or ''):
                continue
            if expected_type and str(proposal.get('proposal_type') or '') != expected_type:
                continue
            if expected_reason_text:
                proposal_reason = str(proposal.get('reason_code') or '').strip()
                preview_reason = str(preview.get('reason') or preview.get('reason_code') or '').strip()
                if expected_reason_text not in {proposal_reason, preview_reason}:
                    continue
            return dict(proposal)
        return None

    def _record_arbiter_decision_events(self, proposal: dict[str, Any], decision: dict[str, Any], *, decision_preview: dict[str, Any]) -> None:
        if self.resource_runtime is None:
            return
        proposal_id = str(proposal.get('proposal_id') or decision.get('proposal_id') or '')
        decision_id = str(decision.get('decision_id') or '')
        trace_id = f'trace_{proposal_id}' if proposal_id else ''
        command_id = str(decision_preview.get('job_id') or proposal.get('command_id') or '')
        proposal_type = str(proposal.get('proposal_type') or '')
        action = str(decision.get('action') or '').upper()
        self._note_review_decision_history(proposal, decision)
        gate = decision.get('gate') if isinstance(decision.get('gate'), dict) else {}
        if action == 'OBSERVE_MORE' and str(gate.get('blocked_reason') or '').startswith('stop_after_evidence_'):
            self.resource_runtime.record_resource_event('stop_after_gate_downgrade', payload={'proposal_id': proposal_id, 'proposal_type': proposal_type, 'blocked_reason': gate.get('blocked_reason'), 'stop_after_evidence_state': gate.get('stop_after_evidence_state'), 'observed_fields': gate.get('observed_fields') or []}, trace_id=trace_id, proposal_id=proposal_id, decision_id=decision_id, command_id=command_id, lease_id=command_id)
        observe_more_key = f'{command_id}:{proposal_type}' if command_id and proposal_type else ''
        wait_actions = {'OBSERVE_MORE': ('observe_more_sec', 300.0), 'MARK_STALLED_NO_KILL': ('observe_more_sec', 300.0), 'CONTINUE': ('ttl_sec', 600.0), 'DENY_KILL': ('ttl_sec', 600.0)}
        if observe_more_key and action in wait_actions and (str(decision.get('canonical_outcome') or '').upper() != RESOURCE_REVIEW_TIMEBOX):
            (delay_field, default_delay) = wait_actions[action]
            try:
                delay_sec = float(decision.get(delay_field) or default_delay)
            except (TypeError, ValueError):
                delay_sec = default_delay
            delay_sec = max(1.0, delay_sec)
            next_review_at = time.time() + delay_sec
            self._review_observe_more_until[observe_more_key] = next_review_at
            self.resource_runtime.record_resource_event('resource_review_observe_more_scheduled', payload={'proposal_id': proposal_id, 'proposal_type': proposal_type, 'action': RESOURCE_REVIEW_NO_ACTION, 'raw_action': action, 'delay_field': delay_field, 'delay_sec': delay_sec, 'observe_more_sec': delay_sec if delay_field == 'observe_more_sec' else None, 'ttl_sec': delay_sec if delay_field == 'ttl_sec' else None, 'next_review_at': next_review_at}, trace_id=trace_id, proposal_id=proposal_id, decision_id=decision_id, command_id=command_id, lease_id=command_id)
        elif observe_more_key:
            self._review_observe_more_until.pop(observe_more_key, None)
        payload = {'actor_id': 'resource_arbiter', 'cache_namespace': 'resource_arbiter', 'decision': dict(decision), 'proposal_id': proposal_id, 'decision_preview': dict(decision_preview or {}), 'main_agent_feedback': main_agent_feedback(decision, proposal=proposal)}
        self.resource_runtime.record_resource_event('decision', payload=payload, trace_id=trace_id, proposal_id=proposal_id, decision_id=decision_id, command_id=command_id, lease_id=command_id)
        execution_outcome = str(decision.get('execution_outcome') or (RESOURCE_REVIEW_KILL if action in TERMINATING_ACTIONS else RESOURCE_REVIEW_NO_ACTION))
        execution_status = 'pending_terminate' if execution_outcome == RESOURCE_REVIEW_KILL else 'no_action'
        advisory = proposal.get('main_agent_advisory') if isinstance(proposal.get('main_agent_advisory'), dict) else {}
        self.resource_runtime.record_resource_event('execution', payload={'action': execution_outcome, 'execution_outcome': execution_outcome, 'raw_action': action, 'status': execution_status, 'execution_status': 'pending' if execution_outcome == RESOURCE_REVIEW_KILL else 'no_action', 'reason': decision.get('reason'), 'kill_class': gate.get('kill_class'), 'strict_gate_result': 'allow' if bool(gate.get('allowed')) else 'blocked', 'strict_gate_reason': gate.get('blocked_reason'), 'advisory_status': advisory.get('advisory_status'), 'advisory_preference': advisory.get('preference')}, trace_id=trace_id, proposal_id=proposal_id, decision_id=decision_id, command_id=command_id, lease_id=command_id)
        self.resource_runtime.record_resource_event('feedback', payload={'message': main_agent_feedback(decision, proposal=proposal)}, trace_id=trace_id, proposal_id=proposal_id, decision_id=decision_id, command_id=command_id, lease_id=command_id)
        updated = dict(proposal)
        updated['last_decision'] = dict(decision)
        if action in TERMINATING_ACTIONS:
            updated['execution_status'] = 'pending'
            self.resource_runtime.update_active_resource_proposal(proposal_id, updated)
        elif action == 'DENY_KILL':
            self.resource_runtime.clear_active_resource_proposals_for_command(command_id, proposal_type=proposal_type, reason_code=str(proposal.get('reason_code') or ''))
        else:
            self.resource_runtime.update_active_resource_proposal(proposal_id, updated)

    def _apply_kill_approval_gate(self, decision: dict[str, Any]) -> dict[str, Any]:
        if not decision.get('terminate'):
            return decision
        decision['would_terminate'] = True
        reason = str(decision.get('reason') or 'resource_guard')
        if self.auto_kill_enabled:
            decision['recommended_action'] = 'terminate'
            decision['requires_llm_decision'] = False
            decision['arbiter_enabled'] = False
            decision['feedback'] = self._append_resource_intervention_summary(str(decision.get('feedback') or ''), action='KILL_AND_REPLAN', reason=reason)
            return decision
        decision['terminate'] = False
        decision['recommended_action'] = 'stop_and_replan'
        decision['requires_llm_decision'] = True
        decision['arbiter_enabled'] = bool(self.arbiter_enabled and self.arbiter_mode != 'off')
        decision['feedback'] = self._append_resource_intervention_summary(self._recommendation_feedback(reason=reason, feedback=str(decision.get('feedback') or '')), action='RECOMMEND_STOP_AND_REPLAN', reason=reason)
        return decision

    @staticmethod
    def _recommendation_feedback(*, reason: str, feedback: str) -> str:
        text = str(feedback or '').strip()
        replacements = {'RESOURCE_FEEDBACK: terminated_idle_gpu_lease because': 'RESOURCE_FEEDBACK: recommend_release_gpu_lease because', 'RESOURCE_FEEDBACK: terminated_low_progress_command because': 'RESOURCE_FEEDBACK: recommend_stop_command because', 'RESOURCE_FEEDBACK: terminated_stalled_command because': 'RESOURCE_FEEDBACK: recommend_stop_command because', 'RESOURCE_FEEDBACK: terminated_low_signal_command because': 'RESOURCE_FEEDBACK: recommend_stop_command because', 'RESOURCE_FEEDBACK: terminated_dataloader_bottleneck because': 'RESOURCE_FEEDBACK: recommend_stop_command because', 'RESOURCE_FEEDBACK: terminated_invalid_training_metrics because': 'RESOURCE_FEEDBACK: recommend_stop_command because'}
        for (old, new) in replacements.items():
            if text.startswith(old):
                text = new + text[len(old):]
                break
        if not text:
            text = f'RESOURCE_FEEDBACK: recommend_stop_command because {reason}.'
        if text.endswith('.'):
            text = text[:-1]
        return text + '; requires_llm_decision=true.\n'
