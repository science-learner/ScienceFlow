"""Resource Arbiter Control Plane contracts: effect."""

from __future__ import annotations

from tests._resource_arbiter_control_plane_support import *  # noqa: F401,F403


def test_arbiter_normalizes_legacy_kill_alias_to_canonical() -> None:
    proposal = {"proposal_id": "p1", "proposal_type": "kill_proposal", "reason_code": "stalled_stdout"}

    decision = normalize_arbiter_decision(
        {"action": "APPROVE_KILL_STALLED", "confidence": "high", "reason": "legacy"},
        proposal=proposal,
    )

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["original_action"] == "APPROVE_KILL_STALLED"
    assert decision["action_normalized"] is True

def test_kill_proposal_allows_idle_lease_release_action() -> None:
    proposal = {
        "proposal_id": "p-idle-release",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
    }

    gated = enforce_proposal_action_allowlist(
        {"action": "RELEASE_IDLE_LEASE", "confidence": "high", "reason": "idle gpu lease should be released"},
        proposal,
    )

    assert gated["action"] == "RELEASE_IDLE_LEASE"
    assert gated.get("gate", {}).get("blocked_reason") != "proposal_action_not_allowed"

def test_state_machine_bad_route_kill_requires_main_agent_advisory(tmp_path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "safe_to_stop",
            "confidence": "high",
            "reason": "training is CPU busy but not producing useful metric or artifact progress",
            "advisory_mode": "inline_memory_edit",
            "memory_edit_applied": False,
        }

    async def arbiter(_proposal):
        return {"action": "KILL_AND_REPLAN", "reason": "low-value route after review", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        kill_mode="auto",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
        review_state_enabled=True,
        review_heartbeat_sec=60.0,
        review_warmup_windows=0,
        review_inactive_windows=5,
        review_value_windows=2,
    )
    job_id = _heavy_job(observer, tmp_path)

    first = observer.active_intervention_decision(
        job_id,
        elapsed_sec=60.0,
        stdout_age_sec=60.0,
        stdout_lines=0,
        stdout_bytes=0,
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert not first.get("would_terminate")

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=120.0,
        stdout_age_sec=120.0,
        stdout_lines=0,
        stdout_bytes=0,
        process_tree_cpu={"total_cpu_pct": 300.0, "busy_child_count": 2},
    )
    assert decision["would_terminate"] is True
    assert decision["terminate"] is False
    assert decision["requires_llm_decision"] is True
    assert decision["resource_review_boundary"]["kind"] == "route_value"

    arbiter_decision = asyncio.run(observer.arbiter_decide(job_id, proposal=decision["proposal"], decision_preview=decision))

    assert arbiter_decision["action"] == "KILL_AND_REPLAN"
    assert arbiter_decision["raw_action"] == "KILL_AND_REPLAN"
    assert arbiter_decision["execution_outcome"] == "KILL"
    assert arbiter_decision["terminate"] is True
    kill_intent = arbiter_decision["decision"]["kill_intent_snapshot"]
    assert kill_intent["snapshot_id"]
    assert kill_intent["metric_scope_key"] == ""
    assert arbiter_decision["suppress_main_agent_feedback"] is False
    assert "approved kill" in arbiter_decision["feedback"]
    assert observer._review_states[job_id].job_state_bucket == "ACTIONED"
    events = _resource_events(tmp_path)
    event_types = [event.get("event_type") for event in events]
    assert "main_agent_advisory" in event_types
    outcome_event = next(event for event in events if event.get("event_type") == "resource_review_outcome")
    assert outcome_event["payload"]["action"] == "KILL"
    assert outcome_event["payload"]["execution_outcome"] == "KILL"
    assert outcome_event["payload"]["raw_action"] == "KILL_AND_REPLAN"
    assert kill_intent["kill_basis"] == "value_stagnation"
    pending_event = next(
        event
        for event in events
        if event.get("event_type") == "execution"
        and event.get("payload", {}).get("execution_status") == "pending"
    )
    proposal_id = str(pending_event.get("proposal_id") or "")
    resource_state = json.loads((tmp_path / "resource" / "resource_state.json").read_text(encoding="utf-8"))
    assert resource_state["active_proposals"][proposal_id]["execution_status"] == "pending"

    observer.resource_guard_action(
        job_id,
        action="terminate_by_arbiter",
        reason="approved_test_kill",
        elapsed_sec=121.0,
    )

    resource_state = json.loads((tmp_path / "resource" / "resource_state.json").read_text(encoding="utf-8"))
    assert resource_state.get("active_proposals") == {}
    executed_event = next(
        event
        for event in reversed(_resource_events(tmp_path))
        if event.get("event_type") == "execution"
        and event.get("payload", {}).get("execution_status") == "executed"
    )
    assert executed_event["payload"]["raw_action"] == "terminate_by_arbiter"

def test_llm_budget_exhaustion_kills_repeated_high_confidence_no_work_stall(tmp_path) -> None:
    async def deny(_proposal):
        return {"action": "DENY_KILL", "reason": "still useful", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W01",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
        arbiter_job_llm_call_cap=1,
        arbiter_job_token_cap=100000,
    )
    job_id = "W01:bash:00151"
    observer._job_llm_call_count[job_id] = 1
    proposal = {
        "proposal_id": "rp_budget_stalled",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "command_id": job_id,
        "progress_snapshot": {
            "runtime_sec": 30663.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "stdout_last_line_age_sec": 30663.0,
            "artifact_last_update_age_sec": 30663.0,
            "metric_last_update_age_sec": 30663.0,
            "process_tree_cpu": {
                "available": True,
                "busy_child_count": 0,
                "total_cpu_pct": 0.0,
                "child_cpu_pct": 0.0,
            },
        },
        "execution_facts": {
            "assigned_resource_idle": True,
            "cpu_busy": False,
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
        "resource_snapshot": {},
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["action"] == "KILL_AND_REPLAN"
    assert result["terminate"] is True
    assert result["decision"]["gate"]["allowed"] is True
    assert result["decision"]["gate"]["deterministic_support"] is True
    assert "llm_budget_exhausted" in result["decision"]["reason"]

def test_stalled_guard_includes_cpu_fact_and_policy_kills_no_work(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=300.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        review_state_enabled=False,
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.stalled_guard_decision(
        job_id,
        elapsed_sec=1800.0,
        stdout_age_sec=1800.0,
        process_tree_cpu={"available": True, "busy_child_count": 0, "total_cpu_pct": 0.0, "child_cpu_pct": 0.0},
    )
    proposal = decision["proposal"]

    assert proposal["progress_snapshot"]["process_tree_cpu"]["busy_child_count"] == 0
    assert proposal["progress_snapshot"]["progress_signal"] == "stalled"

    arbiter = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview=decision))

    assert arbiter["action"] == "KILL_AND_REPLAN"
    assert arbiter["terminate"] is True
    assert arbiter["decision"]["gate"]["allowed"] is True
    assert arbiter["decision"]["gate"]["deterministic_support"] is True

def test_policy_arbiter_requests_advisory_for_low_progress_kill(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
    )
    job_id = _heavy_job(observer, tmp_path)
    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1200.0,
        stdout_age_sec=1200.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
    )
    assert decision["would_terminate"] is True
    assert decision["terminate"] is False
    assert decision["arbiter_enabled"] is True

    arbiter = asyncio.run(observer.arbiter_decide(job_id, proposal=decision["proposal"], decision_preview=decision))

    assert arbiter["action"] == "OBSERVE_MORE"
    assert arbiter["terminate"] is False
    assert "gate" not in arbiter["decision"]
    events = _resource_events(tmp_path)
    types = [event["event_type"] for event in events]
    assert "kill_proposal" in types
    assert "arbiter_input" in types
    assert "decision" in types
    assert "execution" in types
    execution = next(event for event in events if event["event_type"] == "execution")
    assert execution["payload"]["action"] == "NO_ACTION"
    assert execution["payload"]["execution_outcome"] == "NO_ACTION"
    assert execution["payload"]["raw_action"] == "OBSERVE_MORE"

def test_llm_arbiter_callback_can_deny_kill(tmp_path) -> None:
    async def deny(_proposal):
        return {"action": "DENY_KILL", "reason": "artifact chunks are still updating", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=0.1,
        low_progress_no_artifact_sec=0.1,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
    )
    job_id = _heavy_job(observer, tmp_path)
    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=1200.0,
        stdout_age_sec=1200.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
    )

    arbiter = asyncio.run(observer.arbiter_decide(job_id, proposal=decision["proposal"], decision_preview=decision))

    assert arbiter["action"] == "DENY_KILL"
    assert arbiter["raw_action"] == "DENY_KILL"
    assert arbiter["execution_outcome"] == "NO_ACTION"
    assert arbiter["terminate"] is False
    assert arbiter["value_review_outcome"] == "NO_ACTION"
    assert arbiter["suppress_main_agent_feedback"] is True
    assert arbiter["feedback"] == ""

def test_policy_fallback_kills_replan_advisory_with_stalled_no_output() -> None:
    proposal = {
        "proposal_id": "p-advisory-replan",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_route_value",
        "main_agent_advisory": {
            "preference": "replan",
            "confidence": "high",
            "advisory_status": "captured",
        },
        "progress_snapshot": {
            "runtime_sec": 3527.0,
            "stdout_last_line_age_sec": 3100.0,
            "stdout_lines": 16,
            "stdout_bytes": 575,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "metric_history_line_count": 0,
            "artifact_updates": [],
            "near_submission": False,
            "recoverable_artifact_on_disk": False,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert "main-agent advisory" in decision["reason"]
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["advisory_support"] is True

def test_policy_fallback_keeps_replan_advisory_from_killing_with_progress_artifact() -> None:
    proposal = {
        "proposal_id": "p-advisory-artifact",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_route_value",
        "main_agent_advisory": {
            "preference": "replan",
            "confidence": "high",
            "advisory_status": "captured",
        },
        "progress_snapshot": {
            "runtime_sec": 1800.0,
            "stdout_last_line_age_sec": 1200.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "metric_history_line_count": 0,
            "artifact_updates": [{"path": "artifacts/best_model.pt", "recoverable_artifact_on_disk": True}],
            "near_submission": False,
            "recoverable_artifact_on_disk": True,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": True,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] != "KILL_AND_REPLAN"

def test_policy_fallback_keeps_replan_advisory_from_killing_near_submission() -> None:
    proposal = {
        "proposal_id": "p-advisory-near-submission",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_route_value",
        "main_agent_advisory": {
            "preference": "replan",
            "confidence": "high",
            "advisory_status": "captured",
        },
        "progress_snapshot": {
            "runtime_sec": 1800.0,
            "stdout_last_line_age_sec": 1200.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "metric_history_line_count": 0,
            "artifact_updates": [],
            "near_submission": True,
            "recoverable_artifact_on_disk": False,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "DENY_KILL"

def test_policy_arbiter_ignores_completed_deliverable_as_kill_reason() -> None:
    decision = fallback_policy_decision(
        {
            "reason_code": "active_intervention:deliverable_complete_resource_hold",
            "progress_snapshot": {"progress_confidence": "high", "near_submission": True, "runtime_sec": 3600},
            "resource_snapshot": {"pressure": "green"},
        }
    )

    assert decision["action"] == "DENY_KILL"
    assert "near submission" in decision["reason"]

def test_policy_fallback_kills_late_log_only_gpu_idle_cpu_busy_work() -> None:
    proposal = {
        "proposal_id": "p-log-only",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:sm_timebox_expired",
        "progress_snapshot": {
            "runtime_sec": 2400.0,
            "progress_signal": "active",
            "progress_confidence": "medium",
            "artifact_updates": [{"path": "tmp/train_log.txt", "recoverable_artifact_on_disk": False}],
            "artifact_last_update_age_sec": 0.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
        },
        "execution_facts": {
            "intent_device_mismatch": True,
            "assigned_resource_idle": True,
            "cpu_busy": True,
            "activity_without_metric_submission_or_stdout": True,
            "has_recent_metric_or_submission": False,
        },
        "resource_snapshot": {"pressure": "green"},
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["deterministic_support"] is True
    assert gated["gate"]["kill_class"] == "deterministic_resource"

def test_kill_proposal_cooldown_suppresses_unchanged_boundary(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    observer.kill_proposal_cooldown_sec = 600.0
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]

    first_decision, first_signal = _route_value_review_payload(2)
    second_decision, second_signal = _route_value_review_payload(2)

    first = observer._record_kill_proposal_event(job, first_decision, first_signal, source="resource_review_state")
    second = observer._record_kill_proposal_event(job, second_decision, second_signal, source="resource_review_state")

    assert first is not None
    assert second is None
    events = _resource_events(tmp_path)
    assert sum(1 for event in events if event.get("event_type") == "kill_proposal") == 1
    suppressed = [event for event in events if event.get("event_type") == "suppressed_proposal"]
    assert suppressed
    assert suppressed[-1]["payload"]["suppressed_reason"] == "cooldown_same_unchanged_boundary"

def test_kill_proposal_carries_resource_metric_value_at_top_level(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]
    decision, signal = _route_value_review_payload(2)
    signal["resource_metric_value"] = {
        "metric_name": "val_w_auc",
        "value": 0.544124,
        "best_value": 0.490066,
        "status": "above_observed_best",
        "useful": True,
    }

    proposal = observer._record_kill_proposal_event(
        job,
        decision,
        signal,
        source="resource_review_state",
    )

    assert proposal is not None
    assert proposal["resource_metric_value"] == signal["resource_metric_value"]

def test_kill_proposal_cooldown_allows_no_progress_escalation(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
        review_value_windows=2,
    )
    observer.kill_proposal_cooldown_sec = 600.0
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]

    first_decision, first_signal = _route_value_review_payload(2)
    second_decision, second_signal = _route_value_review_payload(3)

    first = observer._record_kill_proposal_event(job, first_decision, first_signal, source="resource_review_state")
    second = observer._record_kill_proposal_event(job, second_decision, second_signal, source="resource_review_state")

    assert first is not None
    assert second is not None
    assert second["cooldown_bypassed_by_escalation"] is True
    assert second["boundary_state"]["no_useful_progress_windows"] == 3
    events = _resource_events(tmp_path)
    proposals = [event for event in events if event.get("event_type") == "kill_proposal"]
    assert len(proposals) == 2
    assert proposals[-1]["payload"]["cooldown_bypassed_by_escalation"] is True

def test_kill_proposal_cooldown_allows_repeated_timebox_expired(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    observer.kill_proposal_cooldown_sec = 600.0
    job_id = _heavy_job(observer, tmp_path)
    job = observer._jobs[job_id]
    boundary = {"kind": "timebox_expired", "reason": "timebox_expired"}
    review_state = {
        "job_state_bucket": "TIMEBOX_ACTIVE",
        "no_useful_progress_windows": 0,
        "timebox_id": "tb_test",
        "timebox_windows": 2,
        "blocked_worker_count": 0,
        "active_waiter_pressure": False,
    }
    signal = {
        "elapsed_sec": 360.0,
        "stdout_lines": 5,
        "stdout_bytes": 200,
        "process_tree_cpu": {"total_cpu_pct": 600.0, "busy_child_count": 4},
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
        "metric_history_text": "",
        "metric_history_line_count": 0,
    }
    decision = {
        "enabled": True,
        "terminate": False,
        "would_terminate": True,
        "requires_llm_decision": True,
        "arbiter_enabled": True,
        "reason": "active_intervention:sm_timebox_expired",
        "resource_review_boundary": boundary,
        "resource_review_state": review_state,
    }

    first = observer._record_kill_proposal_event(job, dict(decision), dict(signal), source="resource_review_state")
    second = observer._record_kill_proposal_event(job, dict(decision), dict(signal), source="resource_review_state")

    assert first is not None
    assert second is not None
    assert second["cooldown_bypassed_by_escalation"] is True
    events = _resource_events(tmp_path)
    proposals = [event for event in events if event.get("event_type") == "kill_proposal"]
    assert len(proposals) == 2
    assert not [event for event in events if event.get("event_type") == "suppressed_proposal"]

def test_arbiter_gate_downgrades_low_confidence_kill_even_with_advisory() -> None:
    proposal = {
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
        },
        "main_agent_advisory": {"preference": "safe_to_stop", "confidence": "high"},
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "medium", "reason": "advisory agrees"},
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["blocked_reason"] == "arbiter_confidence_not_high"

def test_arbiter_gate_requires_advisory_for_discretionary_kill() -> None:
    proposal = {
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 2,
        }
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "stalled"},
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["allowed"] is False
    assert gated["gate"]["blocked_reason"] == "owning_agent_advisory_missing"
    assert gated["gate"]["multi_window_progress"] is True

    with_advisory = dict(proposal)
    with_advisory["main_agent_advisory"] = {
        "preference": "safe_to_stop",
        "confidence": "high",
        "advisory_status": "captured",
    }
    allowed = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "advisory agrees"},
        with_advisory,
    )
    assert allowed["action"] == "KILL_AND_REPLAN"
    assert allowed["gate"]["allowed"] is True
    assert allowed["gate"]["advisory_support"] is True

def test_repeated_high_confidence_stall_escalates_observe_to_kill() -> None:
    proposal = {
        "proposal_id": "rp_stall",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "progress_snapshot": {
            "runtime_sec": 1800.0,
            "stdout_last_line_age_sec": 1800.0,
            "artifact_last_update_age_sec": 1800.0,
            "metric_last_update_age_sec": 1800.0,
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 3,
            "multi_window_low_progress": True,
            "process_tree_cpu": {"available": True, "busy_child_count": 0, "total_cpu_pct": 0.0, "child_cpu_pct": 0.0},
        },
        "review_history": {"last_action": "OBSERVE_MORE", "observe_count": 2},
    }

    assert proposal_has_severe_stalled_no_work(proposal) is True
    decision = enforce_repeated_stall_escalation(
        {"action": "OBSERVE_MORE", "confidence": "medium", "reason": "collect one more window"},
        proposal,
    )
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert decision["stall_escalation"]["from_action"] == "OBSERVE_MORE"
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["deterministic_support"] is True

def test_policy_fallback_kills_quick_probe_overrun_without_value_signal() -> None:
    proposal = {
        "proposal_id": "p-quick-probe",
        "proposal_type": "quick_probe_review",
        "reason_code": "quick_probe_runtime_exceeded",
        "progress_snapshot": {
            "runtime_sec": 640.0,
            "quick_probe_hard_review_sec": 420.0,
            "quick_probe_candidate": True,
            "progress_confidence": "high",
            "progress_signal": "stalled",
            "metric_history_line_count": 0,
            "stdout_lines": 0,
            "stdout_bytes": 0,
            "artifact_updates": [],
            "near_submission": False,
            "meaningful_stdout": False,
            "recoverable_artifact_on_disk": False,
        },
        "execution_facts": {
            "assigned_resource_idle": True,
            "cpu_busy": True,
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["deterministic_support"] is True

def test_policy_fallback_kills_active_but_unaffordable_under_finalization() -> None:
    proposal = {
        "proposal_id": "p-active-unaffordable",
        "proposal_type": "kill_proposal",
        "reason_code": "deadline_finalization_reserve",
        "progress_snapshot": {
            "runtime_sec": 5820.0,
            "progress_confidence": "high",
            "progress_signal": "active",
            "metric_history_line_count": 0,
            "stdout_lines": 100,
            "stdout_bytes": 4096,
            "artifact_updates": [],
            "near_submission": False,
            "meaningful_stdout": True,
            "recoverable_artifact_on_disk": False,
            "deadline_event": True,
            "deadline_remaining_sec": 800.0,
            "finalization_reserve_sec": 900.0,
            "eta_confidence": "high",
            "eta_to_current_phase_end_sec": 63760.0,
            "eta_to_deliverable_sec": 63760.0,
            "finish_feasible": False,
        },
        "execution_facts": {
            "has_recent_metric_or_submission": False,
            "has_recent_useful_artifact": False,
        },
    }

    decision = fallback_policy_decision(proposal)
    gated = enforce_arbiter_kill_gate(decision, proposal)

    assert decision["action"] == "KILL_AND_REPLAN"
    assert decision["confidence"] == "high"
    assert "deterministic resource facts" in decision["reason"]
    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["deterministic_support"] is True
