"""Resource Arbiter Control Plane contracts: replay."""

from __future__ import annotations

from tests._resource_arbiter_control_plane_support import *  # noqa: F401,F403


def test_resource_arbiter_prompt_includes_review_history_for_stall_decisions() -> None:
    prompt = build_resource_arbiter_prompt({
        "proposal_id": "p1",
        "review_history": {"last_action": "OBSERVE_MORE", "observe_count": 2},
        "budget_priority": {"level": "high"},
    })

    assert "review_history" in prompt
    assert "budget_priority" in prompt
    assert "repeated OBSERVE_MORE or DENY_KILL" in prompt

def test_review_history_escalates_repeated_safe_to_stop_advisory_to_structured_support(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path)
    proposal = {
        "command_id": "W00:bash:00123",
        "proposal_id": "rp1",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "progress_snapshot": {
            "progress_signal": "unknown",
            "progress_confidence": "low",
            "deadline_event": False,
        },
        "main_agent_advisory": {"preference": "safe_to_stop", "confidence": "low"},
    }

    first = observer._attach_review_history(proposal, now=1.0)
    observer._note_review_decision_history(first, {"action": "OBSERVE_MORE"})
    second = observer._attach_review_history(proposal, now=2.0)
    observer._note_review_decision_history(second, {"action": "OBSERVE_MORE"})
    third = observer._attach_review_history(proposal, now=3.0)

    history = third["review_history"]
    assert history["repeated_observe_support"] is True
    assert history["advisory_opportunity_support"] is True
    assert history["structured_opportunity_cost_support"] is True
    assert third["structured_opportunity_cost_support"] is True

def test_new_metric_history_triggers_fresh_llm_review(tmp_path) -> None:
    calls: list[dict] = []

    async def decide(proposal):
        calls.append(dict(proposal))
        return {"action": "DENY_KILL", "reason": "reviewed latest metric", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=decide,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
    )
    base = {
        "proposal_id": "rp_metric_generation",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_holder_observe",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {
            "runtime_sec": 1200,
            "progress_signal": "active",
            "progress_confidence": "high",
            "metric_history_text": "Epoch 1 val_auc=0.800",
        },
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
    }

    first = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=base))
    repeated = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=base))
    updated = dict(base)
    updated["progress_snapshot"] = dict(base["progress_snapshot"])
    updated["progress_snapshot"]["metric_history_text"] += "\nEpoch 2 val_auc=0.790"
    third = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=updated))

    assert first["action"] == "DENY_KILL"
    assert repeated["action"] == "OBSERVE_MORE"
    assert third["action"] == "DENY_KILL"
    assert len(calls) == 2
    assert calls[-1]["budget_priority"]["level"] == "low"
    assert "new_metric_history_since_llm_review" in calls[-1]["budget_priority"]["reasons"]
