"""Resource Arbiter Control Plane contracts: policy."""

from __future__ import annotations

from tests._resource_arbiter_control_plane_support import *  # noqa: F401,F403


def test_vague_advisory_commitment_does_not_create_timebox(tmp_path: Path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "continue",
            "confidence": "medium",
            "reason": "I think it may work.",
            "commitment": "it should improve soon",
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=lambda _proposal: {"action": "OBSERVE_MORE", "reason": "uncertain", "confidence": "medium"},
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=0.0,
    )
    job_id = _heavy_job(observer, tmp_path)
    from scienceflow.runtime.safety.resource import new_review_state

    observer._review_states[job_id] = replace(new_review_state(job_id), failed_proof_window_count=1, proof_window_index=1)
    proposal = {
        "proposal_id": "rp_vague_commitment",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_intervention:sm_timebox_expired",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "timebox_expired", "reason": "timebox_expired"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "NO_ACTION"
    assert result["decision"].get("reason_code") != "main_agent_advisory_commitment_timebox"
    assert observer._review_states[job_id].proof_window_source == "no_action"

def test_high_confidence_stop_advisory_is_reused_until_value_changes(tmp_path: Path) -> None:
    calls: list[dict] = []

    async def advisory(proposal):
        calls.append(dict(proposal))
        return {
            "preference": "safe_to_stop",
            "confidence": "high",
            "reason": "route remains below the observed best",
            "ttl_sec": 600.0,
            "observed_phase": "train",
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
        main_agent_advisory_min_interval_sec=600.0,
        arbiter_job_advisory_call_cap=1,
    )
    base = {
        "proposal_id": "rp_advisory_first",
        "proposal_type": "kill_proposal",
        "reason_code": "periodic_efficiency:low_progress_review",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {
            "known_stage": "train",
            "metric_scope_key": "route|train|auc",
            "near_submission": False,
            "metric_history_text": "val_auc=0.70",
        },
        "resource_metric_value": {
            "metric_scope_key": "route|train|auc",
            "useful": False,
            "status": "below_observed_best",
        },
    }

    first = asyncio.run(observer._maybe_attach_main_agent_advisory(base, job_key="W00:bash:00001"))
    repeated = dict(base)
    repeated["proposal_id"] = "rp_advisory_repeated"
    second = asyncio.run(observer._maybe_attach_main_agent_advisory(repeated, job_key="W00:bash:00001"))

    assert first["main_agent_advisory"]["source"] == "main_agent_advisory"
    assert second["main_agent_advisory"]["source"] == "main_agent_advisory_cache"
    assert second["main_agent_advisory"]["reused"] is True
    assert len(calls) == 1
    assert "main_agent_advisory_reused" in {
        event["event_type"] for event in _resource_events(tmp_path)
    }

    improved = dict(repeated)
    improved["proposal_id"] = "rp_advisory_improved"
    improved["progress_snapshot"] = dict(repeated["progress_snapshot"])
    improved["progress_snapshot"]["metric_history_text"] = "val_auc=0.70\nval_auc=0.71"
    improved["resource_metric_value"] = {
        "metric_scope_key": "route|train|auc",
        "useful": False,
        "status": "below_observed_best",
    }
    third = asyncio.run(observer._maybe_attach_main_agent_advisory(improved, job_key="W00:bash:00001"))

    assert "main_agent_advisory" not in third
    assert len(calls) == 1
