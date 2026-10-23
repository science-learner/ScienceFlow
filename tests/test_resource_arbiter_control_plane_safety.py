"""Resource Arbiter Control Plane contracts: safety."""

from __future__ import annotations

from tests._resource_arbiter_control_plane_support import *  # noqa: F401,F403


def test_arbiter_gate_allows_hard_safety_without_advisory() -> None:
    proposal = {
        "kill_class": "hard_safety",
        "reason_code": "gpu_boundary_violation",
        "progress_snapshot": {"progress_signal": "stalled", "progress_confidence": "high"},
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "KILL_AND_REPLAN", "confidence": "high", "reason": "GPU boundary violation"},
        proposal,
    )

    assert gated["action"] == "KILL_AND_REPLAN"
    assert gated["gate"]["allowed"] is True
    assert gated["gate"]["kill_class"] == "hard_safety"

def test_arbiter_gate_downgrades_stop_after_for_valid_deliverable_without_low_progress() -> None:
    proposal = {
        "reason_code": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "deliverable_validity": "produced_valid",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "STOP_AFTER_DELIVERABLE_AND_RELEASE",
            "confidence": "high",
            "reason": "valid deliverable already exists; release resources",
        },
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"

def test_arbiter_gate_downgrades_stop_after_for_invalid_deliverable() -> None:
    proposal = {
        "reason_code": "active_intervention:deliverable_complete_resource_hold",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "deliverable_validity": "produced_invalid",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {
            "action": "STOP_AFTER_DELIVERABLE_AND_RELEASE",
            "confidence": "high",
            "reason": "deliverable exists",
        },
        proposal,
    )

    assert gated["action"] == "OBSERVE_MORE"
    assert gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"

def test_advisory_commitment_missed_timebox_advances_proof_window_without_llm(tmp_path: Path) -> None:
    async def advisory(_proposal):
        raise AssertionError("missed proof-window retry must not ask main agent")

    async def arbiter(_proposal):
        raise AssertionError("missed proof-window retry must not call arbiter LLM")

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.runtime.safety.resource import new_review_state

    observer._review_states[job_id] = new_review_state(job_id)
    proposal = {
        "proposal_id": "rp_expired_timebox",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_intervention:sm_timebox_expired",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "timebox_expired", "reason": "timebox_expired"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    observer._review_states[job_id] = replace(observer._review_states[job_id], failed_proof_window_count=1, proof_window_index=1)
    observer._review_states[job_id] = replace(
        observer._review_states[job_id],
        proof_window_source="negotiated",
        timebox_success_condition="metric_update",
        timebox_id="tb_missed",
        timebox_failure_recorded=True,
    )

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "TIMEBOX"
    assert result["terminate"] is False
    assert result["decision"].get("reason_code") == "proof_window_retry"
    assert observer._review_states[job_id].job_state_bucket == "TIMEBOX_ACTIVE"
    assert observer._review_states[job_id].proof_window_index == 2
    assert observer._review_states[job_id].failed_proof_window_count == 1
    assert observer._review_states[job_id].proof_window_source == "proof_retry"

def test_repeated_failed_proof_windows_final_review_can_timebox(tmp_path: Path) -> None:
    async def advisory(_proposal):
        raise AssertionError("final resource review must not ask main agent")

    async def arbiter(proposal):
        assert proposal["resource_budget_escalation"]["kind"] == "final_resource_review"
        return {
            "outcome": "TIMEBOX",
            "reason_code": "final_resource_review_one_last_window",
            "reason": "resource arbiter allows one last bounded observation window",
            "confidence": "medium",
            "clear_on": "progress_advance",
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=arbiter,
        arbiter_job_llm_call_cap=0,
        arbiter_job_token_cap=0,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
        review_timebox_windows=1,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.runtime.safety.resource import new_review_state

    observer._review_states[job_id] = replace(
        new_review_state(job_id),
        failed_proof_window_count=3,
        proof_window_index=3,
    )
    proposal = {
        "proposal_id": "rp_final_proof_override",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_intervention:sm_timebox_expired",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "timebox_expired", "reason": "timebox_expired"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "TIMEBOX"
    assert result["terminate"] is False
    assert result["decision"]["reason_code"] == "final_resource_review_one_last_window"
    assert result["decision"].get("source") == "resource_arbiter_llm"
    assert observer._review_states[job_id].job_state_bucket == "TIMEBOX_ACTIVE"
