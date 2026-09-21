"""Resource Arbiter Control Plane contracts: observation."""

from __future__ import annotations

from tests._resource_arbiter_control_plane_support import *  # noqa: F401,F403


def test_main_agent_feedback_includes_fact_only_resource_signal() -> None:
    feedback = main_agent_feedback(
        {
            "action": "KILL_AND_REPLAN",
            "reason": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
            "confidence": "high",
        },
        proposal={
            "execution_facts": {
                "intent_device_mismatch": True,
                "declared_device": "gpu",
                "observed_device": "cpu",
                "assigned_resource_idle": True,
                "assigned_gpu_max_util_pct": 0.0,
                "assigned_gpu_max_mem_mb": 0.0,
                "process_cpu_pct": 640.0,
                "cpu_busy": True,
                "has_recent_metric_or_submission": False,
                "has_recent_useful_artifact": False,
            },
            "progress_snapshot": {
                "progress_signal": "stalled",
                "progress_confidence": "high",
                "output_pattern": "unknown",
                "stop_cost": "high",
                "finish_feasible": False,
                "eta_confidence": "high",
                "eta_to_deliverable_sec": 3600.0,
                "remaining_useful_budget_sec": 1200.0,
                "runtime_sec": 900.0,
            },
            "budget_context": {
                "remaining_budget_sec": 2400.0,
            },
            "score_context": {
                "record_count": 3,
                "valid_record_count": 1,
                "valid_best_score": {"value": 0.44, "validity": "valid_comparable"},
            },
            "review_history": {
                "observe_count": 2,
                "last_action": "OBSERVE_MORE",
                "unchanged_bad_fact_windows": 2,
                "repeated_observe_support": True,
                "structured_opportunity_cost_support": True,
                "advisory_preference": "safe_to_stop",
                "advisory_confidence": "medium",
            },
        },
    )

    assert feedback.startswith("Resource arbiter approved kill because")
    assert "RESOURCE_RESEARCH_SIGNAL:" in feedback
    assert "outcome=KILL_AND_REPLAN" in feedback
    assert "review_facts=" in feedback
    assert "prior_observe_count=2" in feedback
    assert "unchanged_windows=2" in feedback
    assert "last_action=OBSERVE_MORE" in feedback
    assert "opportunity_cost=true" in feedback
    assert "resource_facts=" in feedback
    assert "intent_device_mismatch" in feedback
    assert "device=gpu_to_cpu" in feedback
    assert "progress_facts=" in feedback
    assert "progress_signal=stalled" in feedback
    assert "finish_feasible=false" in feedback
    assert "budget_facts=" in feedback
    assert "eta_to_deliverable_sec=3600" in feedback
    assert "remaining_useful_budget_sec=1200" in feedback
    assert "value_facts=" in feedback
    assert "valid_best=0.44" in feedback
    assert "execution_focus=change_method_search_space_schedule_validation_target_or_stopping_condition" in feedback
    assert "category=" not in feedback
    assert "decision_hint=" not in feedback
    assert "route_action=" not in feedback
    assert "DataLoader" not in feedback

def test_observer_execution_facts_do_not_expect_gpu_for_cpu_job(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])
    job_id = observer.job_created(
        command="SCIENCEFLOW_RESOURCE_INTENT=cpu_support python3 train_s13_ensemble.py",
        inferred_class="heavy_cpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800.0,
        workspace_dir=tmp_path,
    )

    assert job_id is not None
    job = observer._jobs[job_id]
    facts = observer._execution_facts_for_job(
        job,
        resource_snapshot={
            "resource_class_declared": "heavy_cpu_candidate",
            "resources": [
                {"resource_type": "gpu", "id": "0", "utilization_gpu_pct": 0.0, "memory_used_mb": 0.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 758.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 786.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 758.0},
        },
    )

    assert job.gpu_ids == ["0"]
    assert facts["gpu_expected"] is False
    assert facts["task_device_mode"] == "cpu_only"
    assert facts["assigned_resource_idle"] is False

def test_execution_facts_do_not_treat_zero_metric_age_without_metric_as_progress() -> None:
    facts = build_execution_facts(
        command="SCIENCEFLOW_RESOURCE_INTENT=gpu_train python3 train.py",
        assigned_gpu_ids=["6"],
        resource_snapshot={
            "resource_class_declared": "heavy_gpu_train",
            "lease": {"active": True, "resource_ids": ["6"]},
            "resources": [
                {"resource_type": "gpu", "id": "6", "utilization_gpu_pct": 0.0, "memory_used_mb": 1200.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 240.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 2400.0,
            "artifact_updates": [{"path": "tmp/train_log.txt", "recoverable_artifact_on_disk": False}],
            "artifact_last_update_age_sec": 0.0,
            "metric_last_update_age_sec": 0.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 240.0},
        },
    )

    assert facts["has_recent_metric_or_submission"] is False
    assert facts["artifact_updates_log_only"] is True
    assert facts["has_recent_useful_artifact"] is False
    assert facts["activity_without_metric_submission_or_stdout"] is True

def test_llm_arbiter_budget_exhaustion_downgrades_to_observe_more(tmp_path) -> None:
    async def deny(_proposal):
        return {"action": "DENY_KILL", "reason": "still useful", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
        arbiter_job_llm_call_cap=1,
        arbiter_job_token_cap=100000,
    )
    proposal = {
        "proposal_id": "rp_budget",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
    }

    first = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))
    second = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert first["action"] == "DENY_KILL"
    assert second["action"] == "OBSERVE_MORE"
    assert "llm_budget_exhausted" in second["decision"]["reason"]
    assert any(event["event_type"] == "llm_budget_exhausted" for event in _resource_events(tmp_path))

def test_policy_fallback_approves_dataloader_bottleneck_without_artifact_progress() -> None:
    proposal = {
        "reason_code": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
        "severity": "red",
        "progress_snapshot": {
            "runtime_sec": 1200.0,
            "stdout_last_line_age_sec": 20.0,
            "progress_confidence": "medium",
            "artifact_updates": [],
            "near_submission": False,
        },
        "resource_snapshot": {},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "OBSERVE_MORE"
    assert "input preprocessing" in decision["reason"]
    assert "advisory" in decision["reason"]

def test_policy_arbiter_approves_invalid_training_metrics() -> None:
    decision = fallback_policy_decision(
        {
            "reason_code": "active_intervention:invalid_training_metrics_nan_inf",
            "progress_snapshot": {"progress_confidence": "high", "artifact_updates": ["checkpoint.pt"], "runtime_sec": 1200},
            "resource_snapshot": {"pressure": "green"},
        }
    )

    assert decision["action"] == "OBSERVE_MORE"
    assert "metrics appear invalid" in decision["reason"]
    assert "advisory" in decision["reason"]

def test_progress_signal_marks_multi_window_stalled() -> None:
    state = classify_progress_signal(
        {
            "elapsed_sec": 2400.0,
            "stdout_age_sec": 1300.0,
            "stdout_lines": 0,
            "saw_training_progress": False,
            "saw_final_score": False,
        },
        progress_age_sec=2400.0,
        artifact_age_sec=2400.0,
        low_progress_warmup_sec=600.0,
        stalled_stdout_sec=600.0,
        no_progress_sec=600.0,
        no_artifact_sec=600.0,
        min_confidence_windows=2,
    )

    assert state["progress_signal"] == "stalled"
    assert state["progress_confidence"] == "high"
    assert state["multi_window_low_progress"] is True

def test_progress_signal_does_not_treat_unadvanced_structured_heartbeat_as_active() -> None:
    state = classify_progress_signal(
        {
            "elapsed_sec": 240.0,
            "stdout_age_sec": 10.0,
            "stdout_lines": 13,
            "saw_training_progress": True,
            "structured_progress": {"current": 0.0, "total": 103.0, "advanced": False},
            "structured_progress_advanced": False,
        },
        progress_age_sec=10.0,
        artifact_age_sec=240.0,
        low_progress_warmup_sec=0.0,
        stalled_stdout_sec=600.0,
        no_progress_sec=600.0,
        no_artifact_sec=600.0,
        idle_samples=3,
        min_confidence_windows=1,
    )

    assert state["progress_signal"] == "degraded"
    assert "recent_progress_heartbeat" not in state["progress_signal_reason"]
    assert "idle_gpu_samples" in state["progress_signal_reason"]

def test_progress_signal_keeps_advanced_structured_heartbeat_active() -> None:
    state = classify_progress_signal(
        {
            "elapsed_sec": 240.0,
            "stdout_age_sec": 10.0,
            "stdout_lines": 13,
            "saw_training_progress": True,
            "structured_progress": {"current": 25.0, "total": 103.0, "advanced": True},
            "structured_progress_advanced": True,
        },
        progress_age_sec=10.0,
        artifact_age_sec=240.0,
        low_progress_warmup_sec=0.0,
        stalled_stdout_sec=600.0,
        no_progress_sec=600.0,
        no_artifact_sec=600.0,
        idle_samples=3,
        min_confidence_windows=1,
    )

    assert state["progress_signal"] == "active"
    assert "recent_progress_heartbeat" in state["progress_signal_reason"]

def test_value_review_observe_more_suppresses_main_agent_feedback(tmp_path: Path) -> None:
    observer, _ = make_observer(
        tmp_path,
        min_register_sec=0.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_proposal_coalesce_window_sec=0.0,
        review_warmup_windows=0,
    )
    job_id = _heavy_job(observer, tmp_path)
    proposal = {
        "proposal_id": "rp_suppress_no_action",
        "proposal_type": "kill_proposal",
        "command_id": job_id,
        "reason_code": "active_intervention:sm_progress_window",
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "progress_signal_windows": 1,
        },
        "decision_preview": {"job_id": job_id, "reason": "active_intervention:sm_progress_window"},
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview=proposal["decision_preview"]))

    assert result["enabled"] is True
    assert result["action"] in {"OBSERVE_MORE", "DENY_KILL", "CONTINUE", "MARK_STALLED_NO_KILL"}
    assert result["raw_action"] == result["action"]
    assert result["execution_outcome"] == "NO_ACTION"
    assert result["value_review_outcome"] == "NO_ACTION"
    assert result["suppress_main_agent_feedback"] is True
    assert result["feedback"] == ""
    events = _resource_events(tmp_path)
    outcome_event = next(e for e in events if e.get("event_type") == "resource_review_outcome")
    assert outcome_event["payload"]["action"] == "NO_ACTION"
    assert outcome_event["payload"]["execution_outcome"] == "NO_ACTION"
    assert outcome_event["payload"]["raw_action"] == result["action"]
    assert any(e.get("event_type") == "decision" for e in events)

def test_lhr_observer_progress_signal_enters_proposal(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        min_register_sec=0.0,
        stalled_stdout_sec=600.0,
        low_progress_enabled=True,
        low_progress_warmup_sec=0.0,
        low_progress_no_heartbeat_sec=600.0,
        low_progress_no_artifact_sec=600.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
    )
    job_id = _heavy_job(observer, tmp_path)

    decision = observer.active_intervention_decision(
        job_id,
        elapsed_sec=2400.0,
        stdout_age_sec=1300.0,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="training",
    )

    progress = decision["proposal"]["progress_snapshot"]
    assert progress["progress_signal"] == "stalled"
    assert progress["progress_confidence"] == "high"
    assert progress["multi_window_low_progress"] is True

def test_resource_observer_score_context_fact_card_hides_paths(tmp_path: Path) -> None:
    (tmp_path / "lhr_stage_performance.csv").write_text(
        "row_order,candidate_id,worker_id,stage_id,metric_value,metric_name,lower_is_better,validation_ok,val_score_type,selection_eligible,selection_score,metric_source_note,brief,why,submission_snapshot,candidate_ready,submission_status,snapshot_path\n"
        "1,W00:L01:S01,W00,S01,0.894680,Final Validation Score,0,1,holdout,1,0.894680,type=holdout,valid auc,ok,snapshots/W00/s01.csv,1,ready,/abs/snapshots/W00/S01\n",
        encoding="utf-8",
    )
    observer, _ = make_observer(tmp_path)

    card = observer._score_context_fact_card()

    assert card["valid_best_score"]["value"] == 0.89468
    assert card["valid_best_score"]["lower_is_better"] is False
    assert "artifact_path" not in card["valid_best_score"]
    assert "snapshot_path" not in card["valid_best_score"]

def test_budget_priority_marks_waiter_low_progress_as_high(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        arbiter_contention_min_waiter_age_sec=300.0,
    )
    priority = observer._budget_priority_for_proposal({
        "proposal_id": "rp_budget_priority",
        "proposal_type": "resource_contention_review",
        "reason_code": "low_progress_with_waiter",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"progress_signal": "unknown", "progress_confidence": "low"},
        "waiters": [{"job_id": "W01:bash:00002", "queue_age_sec": 480.0}],
        "suggested_actions": ["OBSERVE_MORE", "KILL_AND_REPLAN"],
    })

    assert priority["level"] == "high"
    assert "waiter_blocked" in priority["reasons"]
    assert "reason_indicates_low_value_or_stall" in priority["reasons"]

def test_policy_fallback_observes_quick_probe_overrun_with_stdout_signal() -> None:
    proposal = {
        "proposal_id": "p-quick-probe-stdout",
        "proposal_type": "quick_probe_review",
        "reason_code": "quick_probe_runtime_exceeded",
        "progress_snapshot": {
            "runtime_sec": 640.0,
            "quick_probe_hard_review_sec": 420.0,
            "quick_probe_candidate": True,
            "metric_history_line_count": 0,
            "stdout_lines": 3,
            "stdout_bytes": 128,
            "artifact_updates": [],
            "near_submission": False,
            "meaningful_stdout": True,
            "recoverable_artifact_on_disk": False,
        },
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "OBSERVE_MORE"
    assert decision["confidence"] == "low"
