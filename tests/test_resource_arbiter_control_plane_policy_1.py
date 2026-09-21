"""Resource Arbiter Control Plane contracts: policy."""

from __future__ import annotations

from tests._resource_arbiter_control_plane_support import *  # noqa: F401,F403


def test_resource_feedback_key_collapses_llm_reworded_pending_reasons() -> None:
    base = {
        "status": "PENDING",
        "scope": "per_gpu",
        "resource_mode": "YELLOW",
        "blocked_class": "heavy_gpu_train",
        "gpu_ids": ["0"],
        "allowed_classes": ["gpu_tt_light", "pure_tt_cpu"],
        "unlock_condition": "holder_released",
        "blocked_until_unlock": True,
    }
    key_a = LHRResourceObserver._resource_feedback_state_key(
        reason="GPU slot unavailable due to high utilization and resource class contention in YELLOW mode",
        **base,
    )
    key_b = LHRResourceObserver._resource_feedback_state_key(
        reason="GPU slot unavailable and lease not grantable by LLM; wait for availability",
        **base,
    )

    assert key_a == key_b
    assert "gpu_resource_wait" in key_a

def test_resource_arbiter_prompt_lists_allowed_action_values() -> None:
    prompt = build_resource_arbiter_prompt({"proposal_id": "p1"})

    assert "Allowed canonical action values:" in prompt
    assert "CONTINUE | DENY_KILL | OBSERVE_MORE | MARK_STALLED_NO_KILL | KILL_AND_REPLAN | RELEASE_IDLE_LEASE | GRANT_SHARED_GPU_LEASE | DENY_SHARE_USE_CPU_SUPPORT | CONTINUE_SHARED_OBSERVE | STOP_SECONDARY_SHARED_JOB | REVOKE_SHARED_LEASE" in prompt
    assert "finish_feasible=false" in prompt
    assert "liveness only" in prompt
    assert "Do not equate low-level liveness with research value" in prompt
    assert "root submission.csv" in prompt
    assert "STOP_AFTER_DELIVERABLE_AND_RELEASE" not in prompt

def test_resource_feedback_guidance_loads_from_markdown() -> None:
    assert resource_feedback_guidance_value("execution_optimization_focus") == "change_method_search_space_schedule_validation_target_or_stopping_condition"
    assert "method" in resource_feedback_guidance_value("advisory_stop_boundary_note")
    assert "search space" in resource_feedback_guidance_value("advisory_stop_boundary_note")
    assert "low-level liveness" in resource_feedback_guidance_value("budget_deliverable_value_note")
    assert "valid task deliverable" in resource_feedback_guidance_value("budget_deliverable_value_note")

def test_resource_advisory_prompt_separates_stop_from_route_abandonment() -> None:
    prompt = build_inline_resource_advisory_prompt({
        "proposal_id": "p-low-progress",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:low_progress_heartbeat_stalled",
    })

    assert "safe_to_stop means the current command can be interrupted" in prompt
    assert "does not mean abandon the route" in prompt
    assert "change the method" in prompt
    assert "search space" in prompt
    assert "Do not equate low-level liveness with research value" in prompt
    assert "next valid deliverable" in prompt

def test_resource_arbiter_prompt_tells_share_reviews_to_return_action() -> None:
    prompt = build_resource_arbiter_prompt({
        "proposal_id": "p-share",
        "proposal_type": "task_gpu_share_review",
        "share_decision_facts": {"share_eligible": True, "primary_gpu_util_p90_pct": 0.0},
    })

    assert "For task_gpu_share_review, answer with action, not outcome" in prompt
    assert "share_decision_facts" in prompt
    assert "GRANT_SHARED_GPU_LEASE" in prompt

def test_task_gpu_share_explicit_action_wins_over_canonical_outcome() -> None:
    proposal = {"proposal_id": "p-share", "proposal_type": "task_gpu_share_review"}

    decision = normalize_arbiter_decision(
        {
            "outcome": "NO_ACTION",
            "action": "GRANT_SHARED_GPU_LEASE",
            "reason": "low util and enough headroom",
            "confidence": "high",
        },
        proposal=proposal,
    )

    assert decision["action"] == "GRANT_SHARED_GPU_LEASE"
    assert decision["original_action"] == "GRANT_SHARED_GPU_LEASE"
    assert "canonical_outcome" not in decision

def test_main_agent_feedback_cpu_only_omits_gpu_mismatch_tokens() -> None:
    feedback = main_agent_feedback(
        {
            "action": "KILL_AND_REPLAN",
            "reason": "active_intervention:sm_timebox_expired",
            "confidence": "high",
        },
        proposal={
            "execution_facts": {
                "task_device_mode": "cpu_only",
                "gpu_expected": False,
                "declared_device": "gpu",
                "observed_device": "cpu",
                "intent_device_mismatch": False,
                "assigned_resource_idle": False,
                "process_cpu_pct": 640.0,
                "cpu_busy": True,
                "activity_without_metric_submission_or_stdout": True,
                "has_recent_metric_or_submission": False,
                "has_recent_useful_artifact": False,
            },
            "progress_snapshot": {
                "progress_signal": "stalled",
                "progress_confidence": "high",
                "runtime_sec": 1800.0,
            },
        },
    )

    assert "device_mode=cpu_only" in feedback
    assert "intent_device_mismatch" not in feedback
    assert "device=gpu_to_cpu" not in feedback
    assert "assigned_resource_idle=true" not in feedback
    assert "progress_facts=" in feedback
    assert "progress_signal=stalled" in feedback
    assert "value_facts=" in feedback
    assert "score_context=unavailable" in feedback
    assert "execution_focus=add_flushed_scienceflow_hb_or_metric_callback_then_resume" in feedback
    assert "change_method_search_space_schedule_validation_target_or_stopping_condition" not in feedback

def test_execution_facts_detect_gpu_intent_running_cpu_only() -> None:
    facts = build_execution_facts(
        command="SCIENCEFLOW_RESOURCE_INTENT=gpu_tt python3 score.py",
        assigned_gpu_ids=["2"],
        gpu_expected=True,
        resource_snapshot={
            "resource_class_declared": "gpu_tt_light",
            "lease": {"active": True, "resource_ids": ["2"]},
            "resources": [
                {"resource_type": "gpu", "id": "2", "utilization_gpu_pct": 0.0, "memory_used_mb": 256.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 640.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 1800.0,
            "artifact_updates": [{"path": "tmp/score.log"}],
            "artifact_last_update_age_sec": 10.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 640.0},
        },
    )

    assert facts["declared_device"] == "gpu"
    assert facts["observed_device"] == "cpu"
    assert facts["intent_device_mismatch"] is True
    assert facts["activity_without_metric_submission_or_stdout"] is True

def test_execution_facts_cpu_only_task_suppresses_gpu_mismatch() -> None:
    facts = build_execution_facts(
        command="SCIENCEFLOW_RESOURCE_INTENT=gpu_tt python3 score.py",
        assigned_gpu_ids=["2"],
        gpu_expected=False,
        resource_snapshot={
            "resource_class_declared": "gpu_tt_light",
            "resources": [
                {"resource_type": "gpu", "id": "2", "utilization_gpu_pct": 0.0, "memory_used_mb": 256.0},
                {"resource_type": "cpu", "id": "process", "process_tree_cpu": {"total_cpu_pct": 640.0}},
            ],
        },
        progress_snapshot={
            "runtime_sec": 1800.0,
            "metric_history_line_count": 0,
            "meaningful_stdout": False,
            "near_submission": False,
            "process_tree_cpu": {"total_cpu_pct": 640.0},
        },
    )

    assert facts["task_device_mode"] == "cpu_only"
    assert facts["gpu_expected"] is False
    assert facts["declared_device"] == "gpu"
    assert facts["observed_device"] == "cpu"
    assert facts["intent_device_mismatch"] is False
    assert facts["assigned_resource_idle"] is False
    assert facts["activity_without_metric_submission_or_stdout"] is True

def test_resource_arbiter_prompt_includes_execution_facts() -> None:
    prompt = build_resource_arbiter_prompt({
        "proposal_id": "p-exec",
        "execution_facts": {"intent_device_mismatch": True, "observed_device": "cpu"},
    })

    assert "execution_facts" in prompt
    assert "intent_device_mismatch" in prompt
    assert "CPU busy" in prompt

def test_arbiter_downgrades_action_not_allowed_for_proposal_type() -> None:
    proposal = {
        "proposal_id": "p1",
        "proposal_type": "task_gpu_share_review",
        "reason_code": "task_gpu_share_review:wrong_action",
        "progress_snapshot": {
            "progress_signal": "stalled",
            "progress_confidence": "high",
            "progress_signal_windows": 2,
            "deliverable_validity": "produced_valid",
        },
    }

    gated = enforce_arbiter_kill_gate(
        {"action": "STOP_AFTER_DELIVERABLE_AND_RELEASE", "confidence": "high", "reason": "wrong proposal"},
        proposal,
    )
    gated = enforce_proposal_action_allowlist(gated, proposal)

    assert gated["action"] == "CONTINUE_SHARED_OBSERVE"
    assert gated["gate"]["blocked_reason"] == "proposal_action_not_allowed"
    assert gated["gate"]["fallback_action"] == "CONTINUE_SHARED_OBSERVE"

def test_contention_review_emits_non_terminating_arbiter_proposal(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        assignment="env_only",
        min_register_sec=0.0,
        stalled_stdout_sec=0.0,
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        arbiter_contention_review_enabled=True,
        arbiter_contention_min_runtime_sec=0.0,
        arbiter_contention_min_waiter_age_sec=0.0,
        arbiter_contention_min_interval_sec=1.0,
        arbiter_proposal_coalesce_window_sec=0.0,
    )
    blocker = _heavy_job(observer, tmp_path)
    acquired = observer.queue_try_acquire(blocker, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired["acquired"] is True
    (tmp_path / "train_waiter.py").write_text("import torch\ntorch.cuda.is_available()\nprint('waiter')\n", encoding="utf-8")
    waiter = observer.job_created(
        command="python3 train_waiter.py --device cuda --epochs 3",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=100.0,
        workspace_dir=tmp_path,
    )
    assert waiter is not None
    queued = observer.queue_try_acquire(waiter, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert queued["acquired"] is False

    decision = observer.active_intervention_decision(
        blocker,
        elapsed_sec=1200.0,
        stdout_age_sec=1.0,
        stdout_lines=10,
        stdout_bytes=100,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )

    assert decision["arbiter_review"] is True
    assert decision["would_terminate"] is False
    proposal = decision["proposal"]
    assert proposal["proposal_type"] == "resource_contention_review"
    assert proposal["waiters"][0]["job_id"] == waiter
    assert "near_deliverable" not in proposal["blocker"]
    assert "deliverable_completion_state" not in proposal["blocker"]

    arbiter = asyncio.run(observer.arbiter_decide(blocker, proposal=proposal, decision_preview=decision))

    assert arbiter["enabled"] is True
    assert arbiter["terminate"] is False
    events = _resource_events(tmp_path)
    types = [event["event_type"] for event in events]
    assert "resource_review_proposal" in types
    assert "arbiter_input" in types
    assert "decision" in types

def test_main_agent_advisory_is_attached_as_evidence(tmp_path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "safe_to_stop",
            "confidence": "high",
            "reason": "route is no longer useful",
            "advisory_mode": "inline_memory_edit",
            "memory_edit_applied": True,
            "_audit": {
                "event_version": 1,
                "advisory_mode": "inline_memory_edit",
                "status": "captured",
                "raw_response": "RESOURCE_ADVISORY_RESPONSE_BEGIN...",
            },
        }

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
        main_agent_advisory_enabled=True,
        main_agent_advisory_decider=advisory,
    )
    proposal = {
        "proposal_id": "rp_advisory",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "periodic_efficiency:low_progress_review",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "unknown", "progress_confidence": "low"},
        "resource_snapshot": {},
    }

    arbiter = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert arbiter["enabled"] is True
    events = _resource_events(tmp_path)
    audit_event = next(event for event in events if event["event_type"] == "resource_advisory_audit")
    assert audit_event["payload"]["advisory_mode"] == "inline_memory_edit"
    assert audit_event["payload"]["job_id"] == "W00:bash:00001"
    advisory_event = next(event for event in events if event["event_type"] == "main_agent_advisory")
    assert advisory_event["payload"]["advisory"]["preference"] == "safe_to_stop"
    assert advisory_event["payload"]["advisory"]["memory_edit_applied"] is True
    decision_event = next(event for event in events if event["event_type"] == "decision")
    assert decision_event["payload"]["decision"]["proposal_id"] == "rp_advisory"

def test_arbiter_job_lookup_does_not_reuse_mismatched_active_proposal(tmp_path) -> None:
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="policy",
    )
    assert observer.resource_runtime is not None
    observer.resource_runtime.update_active_resource_proposal(
        "rp_W00:bash:00001_share",
        {
            "proposal_id": "rp_W00:bash:00001_share",
            "proposal_type": "task_gpu_share_review",
            "reason_code": "task_gpu_share_review:eligible_secondary_waiter",
            "command_id": "W00:bash:00001",
            "decision_preview": {
                "job_id": "W00:bash:00001",
                "reason": "task_gpu_share_review:eligible_secondary_waiter",
            },
            "progress_snapshot": {"progress_signal": "active", "progress_confidence": "high"},
            "resource_snapshot": {},
            "share_observation": {"share_eligible": True},
        },
    )

    result = asyncio.run(
        observer.arbiter_decide(
            "W00:bash:00001",
            decision_preview={
                "job_id": "W00:bash:00001",
                "reason": "active_intervention:dataloader_bottleneck_low_gpu_high_cpu",
                "would_terminate": True,
            },
        )
    )

    assert result["enabled"] is False
    assert result["reason"] == "proposal_missing"

def test_sync_llm_arbiter_timeout_records_fallback_event(tmp_path) -> None:
    def slow(_proposal):
        time.sleep(1.2)
        return {"action": "KILL_AND_REPLAN", "reason": "too late", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=slow,
        arbiter_timeout_sec=0.05,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
    )
    proposal = {
        "proposal_id": "rp_timeout",
        "proposal_type": "kill_proposal",
        "reason_code": "stalled_stdout",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "unknown", "progress_confidence": "low"},
        "resource_snapshot": {},
    }

    decision = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert decision["action"] == "OBSERVE_MORE"
    assert decision["decision"]["reason"] == "arbiter_llm_timeout"
    events = _resource_events(tmp_path)
    assert any(event["event_type"] == "resource_arbiter_timeout" for event in events)
    assert any(event["event_type"] == "decision" for event in events)

def test_cpu_sidecar_backfill_writes_review_report(tmp_path) -> None:
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        sidecar_enabled=True,
        sidecar_min_parent_runtime_sec=1.0,
    )
    job_id = _heavy_job(observer, tmp_path)

    result = observer.maybe_run_sidecar_backfill(job_id, elapsed_sec=5.0, parent_state="healthy_running")

    assert result["started"] is True
    report_path = tmp_path / "resource" / result["report"]["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["quality_gate"] == "review"
    assert report["sanity_checks"]["sidecar_writes_isolated"] is True
    types = [event["event_type"] for event in _resource_events(tmp_path)]
    assert "sidecar_forked" in types
    assert "sidecar_joined" in types

def test_estra_magent_sidecar_writes_join_packet_and_resource_hint(tmp_path) -> None:
    from scienceflow.research.solver.lnr.transitions.estra_magent.join_gate import load_join_packets
    from scienceflow.research.solver.lnr.transitions.estra_magent.prompt_blocks import format_magent_recommendations

    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        estra_magent_enabled=True,
        estra_magent_sidecar_enabled=True,
        estra_magent_min_parent_runtime_sec=1.0,
    )
    job_id = _heavy_job(observer, tmp_path)

    result = observer.maybe_run_sidecar_backfill(job_id, elapsed_sec=5.0, parent_state="healthy_running")

    assert result["started"] is True
    join_packet = result["join_packet"]
    assert join_packet["quality_gate"] == "review"
    packet_path = tmp_path / "resource" / join_packet["packet_path"]
    assert packet_path.exists()
    packets = load_join_packets(tmp_path / "resource", parent_worker_id="W00")
    block = format_magent_recommendations(packets)
    assert "magent_recommendations:" in block
    assert "parent_summary:" in block
    assert "submission_checker" in block
    types = [event["event_type"] for event in _resource_events(tmp_path)]
    assert "magent_train_observed" in types
    assert "magent_fork_considered" in types
    assert "magent_sidecar_started" in types
    assert "magent_join_packet_ready" in types

def test_estra_magent_recommendations_do_not_fallback_to_estra_summary() -> None:
    from scienceflow.research.solver.lnr.transitions.estra_magent.prompt_blocks import format_magent_recommendations

    block = format_magent_recommendations([
        {
            "sidecar_id": "sc1",
            "quality_gate": "review",
            "summary_for_parent": "",
            "summary_for_estra": "route evidence only",
        }
    ])

    assert "route evidence only" not in block
    assert "parent_summary:" not in block

def test_estra_magent_recommendations_skip_whole_items_over_char_budget() -> None:
    from scienceflow.research.solver.lnr.transitions.estra_magent.prompt_blocks import format_magent_recommendations

    packets = [
        {
            "sidecar_id": "sc1",
            "quality_gate": "review",
            "summary_for_parent": "usable parent advice",
        },
        {
            "sidecar_id": "sc2",
            "quality_gate": "review",
            "summary_for_parent": "second item should not be partially rendered",
            "artifact_refs": [{"type": "report", "path": "reports/very_long_second_item.md"}],
        },
    ]

    block = format_magent_recommendations(packets, max_items=2, max_chars=110)

    assert "sc1" in block
    assert "usable parent advice" in block
    assert "sc2" not in block
    assert "second item" not in block
    assert not block.endswith("...")
    assert format_magent_recommendations(packets, max_items=2, max_chars=0) == ""

def test_checkpoint_to_submission_guard_records_gap(tmp_path) -> None:
    observer, _sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    job_id = _heavy_job(observer, tmp_path)
    gap = {
        "reason": "checkpoint_without_submission",
        "checkpoint_artifacts": [{"path": "model.pt", "size_bytes": 4}],
        "submission_updated": False,
    }

    feedback = observer.checkpoint_to_submission_guard(job_id, gap=gap, elapsed_sec=12.0)

    assert feedback.startswith("RESOURCE_FEEDBACK: checkpoint_without_submission because")
    event = next(event for event in _resource_events(tmp_path) if event["event_type"] == "checkpoint_to_submission_guard")
    assert event["payload"]["checkpoint_artifacts"][0]["path"] == "model.pt"

def test_checkpoint_guard_uses_configured_candidate_artifact_feedback(tmp_path) -> None:
    observer, _sm = make_observer(tmp_path, worker_id="W00", gpu_pool=["0"])
    job_id = _heavy_job(observer, tmp_path)
    gap = {
        "reason": "checkpoint_without_candidate_artifact",
        "checkpoint_artifacts": [{"path": "tmp/model.npy", "size_bytes": 4}],
        "artifact_path": "artifacts/best_solution.json",
        "artifact_updated": False,
    }

    feedback = observer.checkpoint_to_submission_guard(job_id, gap=gap, elapsed_sec=12.0)

    assert feedback.startswith("RESOURCE_FEEDBACK: checkpoint_without_candidate_artifact because")
    assert "artifacts/best_solution.json" in feedback
    assert "root submission.csv" not in feedback

def test_policy_fallback_uses_stdout_last_line_age_field() -> None:
    proposal = {
        "reason_code": "stalled_stdout",
        "severity": "red",
        "progress_snapshot": {
            "runtime_sec": 1200.0,
            "stdout_last_line_age_sec": 1200.0,
            "progress_confidence": "low",
            "artifact_updates": [],
            "near_submission": False,
        },
        "resource_snapshot": {},
    }

    decision = fallback_policy_decision(proposal)

    assert decision["action"] == "OBSERVE_MORE"
    assert "advisory" in decision["reason"]

def test_gpu_visibility_probe_allows_only_low_cost_cuda_checks() -> None:
    assert _is_gpu_visibility_probe('python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"')
    assert _is_gpu_visibility_probe("nvidia-smi --query-gpu=index --format=csv")
    assert not _is_gpu_visibility_probe("python3 train.py --epochs 10")
    assert not _is_gpu_visibility_probe('python3 -c "import torch; model.train(); loss.backward(); optimizer.step()"')

def test_active_resource_proposals_coalesce_same_command_type_reason(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path, gpu_pool=["0"])
    runtime = observer.resource_runtime
    assert runtime is not None

    first = {
        "proposal_id": "rp_old",
        "proposal_type": "kill_proposal",
        "command_id": "W00:bash:00034",
        "reason_code": "active_intervention:sm_timebox_expired",
    }
    second = {
        "proposal_id": "rp_new",
        "proposal_type": "kill_proposal",
        "command_id": "W00:bash:00034",
        "reason_code": "active_intervention:sm_timebox_expired",
        "boundary_state": {"timebox_windows": 7},
    }

    runtime.update_active_resource_proposal("rp_old", first)
    runtime.update_active_resource_proposal("rp_new", second)

    state = json.loads((tmp_path / "resource" / "resource_state.json").read_text(encoding="utf-8"))
    active = state.get("active_proposals") or {}
    assert list(active) == ["rp_new"]
    assert active["rp_new"]["boundary_state"]["timebox_windows"] == 7

def test_repeated_stall_escalation_respects_active_cpu_counterevidence() -> None:
    proposal = {
        "proposal_id": "rp_busy",
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
            "process_tree_cpu": {"available": True, "busy_child_count": 1, "total_cpu_pct": 120.0, "child_cpu_pct": 120.0},
        },
        "review_history": {"last_action": "OBSERVE_MORE", "observe_count": 2},
    }

    assert proposal_has_severe_stalled_no_work(proposal) is False
    decision = enforce_repeated_stall_escalation(
        {"action": "OBSERVE_MORE", "confidence": "medium", "reason": "silent but busy"},
        proposal,
    )

    assert decision["action"] == "OBSERVE_MORE"

def test_arbiter_gate_distinguishes_stop_after_missing_from_unknown_evidence() -> None:
    missing = {
        "reason_code": "active_intervention:deliverable_complete_resource_hold",
        "progress_snapshot": {"progress_signal": "active", "progress_confidence": "medium"},
    }
    unknown = {
        "reason_code": "active_intervention:deliverable_complete_resource_hold",
        "progress_snapshot": {
            "progress_signal": "active",
            "progress_confidence": "medium",
            "deliverable_validity": "produced_unknown",
        },
    }

    missing_gated = enforce_arbiter_kill_gate(
        {"action": "STOP_AFTER_DELIVERABLE_AND_RELEASE", "confidence": "high", "reason": "complete"},
        missing,
    )
    unknown_gated = enforce_arbiter_kill_gate(
        {"action": "STOP_AFTER_DELIVERABLE_AND_RELEASE", "confidence": "high", "reason": "complete"},
        unknown,
    )

    assert missing_gated["action"] == "OBSERVE_MORE"
    assert missing_gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"
    assert unknown_gated["action"] == "OBSERVE_MORE"
    assert unknown_gated["gate"]["blocked_reason"] == "deliverable_not_resource_kill_evidence"

def test_resource_arbiter_prompt_includes_score_context() -> None:
    prompt = build_resource_arbiter_prompt(
        {
            "proposal_id": "p-score",
            "score_context": {
                "valid_best_score": {"value": 0.89468, "lower_is_better": False},
                "capture_gap": False,
            },
        }
    )

    assert "score_context" in prompt
    assert "0.89468" in prompt

def test_resource_intervention_summary_uses_trusted_valid_best(tmp_path: Path) -> None:
    (tmp_path / "lhr_stage_performance.csv").write_text(
        "row_order,candidate_id,worker_id,stage_id,metric_value,metric_name,lower_is_better,validation_ok,val_score_type,selection_eligible,selection_score,metric_source_note,brief,why,submission_snapshot,candidate_ready,submission_status,snapshot_path\n"
        "1,W01:L01:S10,W01,S10,0.894680,Final Validation Score,0,1,holdout,1,0.894680,type=holdout,valid auc,ok,snapshots/W01/s10.csv,1,ready,/abs/snapshots/W01/S10\n",
        encoding="utf-8",
    )
    observer, _ = make_observer(tmp_path)

    feedback = observer._append_resource_intervention_summary(
        "RESOURCE_FEEDBACK: terminated_low_progress_command because no heartbeat.\n",
        action="KILL_AND_REPLAN",
        reason="active_intervention:low_progress",
    )

    assert "RESOURCE_INTERVENTION_SUMMARY:" in feedback
    assert "action=KILL_AND_REPLAN" in feedback
    assert "current_valid_best=0.89468" in feedback
    assert "current_valid_best_source=lhr_stage_performance.csv" in feedback
    assert "current_valid_best_validity=valid_comparable" in feedback
    assert "lower_is_better=false" in feedback
    assert "best_worker=W01" in feedback
    assert "best_stage=S10" in feedback

def test_resource_intervention_summary_skips_same_validation_meta_best(tmp_path: Path) -> None:
    (tmp_path / "lhr_stage_performance.csv").write_text(
        "row_order,candidate_id,worker_id,stage_id,metric_value,metric_name,lower_is_better,validation_ok,"
        "val_score_type,selection_eligible,selection_score,metric_validity,metric_validity_note,"
        "metric_validity_reason_code,metric_source_note,evaluator_backend,evaluator_status,brief,why,"
        "submission_snapshot,candidate_ready,submission_status,snapshot_path\n"
        "1,W00:L02:S24,W00,S24,0.708300,Final Validation Score,0,1,holdout,1,0.708300,high,"
        "comparable_holdout: clean holdout,comparable_holdout,Submission is valid.,local_csv,ok,valid ensemble,"
        "clean held-out score,snapshots/W00/s24.csv,1,ready,/abs/snapshots/W00/S24\n"
        "2,W00:L05:S12,W00,S12,0.967800,Final Validation Score,0,1,holdout,1,0.967800,medium,"
        "same_validation_meta_fit: validation set tuned ensemble weights,same_validation_meta_fit,"
        "Submission is valid.,local_csv,ok,contaminated high score,validation reuse inflated score,"
        "snapshots/W00/s12.csv,1,ready,/abs/snapshots/W00/S12\n",
        encoding="utf-8",
    )
    observer, _ = make_observer(tmp_path)

    feedback = observer._append_resource_intervention_summary(
        "RESOURCE_FEEDBACK: terminated_low_progress_command because no heartbeat.\n",
        action="KILL_AND_REPLAN",
        reason="active_intervention:low_progress",
    )

    assert "RESOURCE_INTERVENTION_SUMMARY:" in feedback
    assert "current_valid_best=0.7083" in feedback
    assert "best_stage=S24" in feedback
    assert "current_valid_best=0.9678" not in feedback
    assert "best_stage=S12" not in feedback

def test_resource_intervention_summary_marks_untrusted_best(tmp_path: Path) -> None:
    observer, _ = make_observer(tmp_path)

    feedback = observer._append_resource_intervention_summary(
        "RESOURCE_FEEDBACK: terminated_low_progress_command because no heartbeat.\n",
        action="KILL_AND_REPLAN",
        reason="active_intervention:low_progress",
    )

    assert "RESOURCE_INTERVENTION_SUMMARY:" in feedback
    assert "current_valid_best=unknown" in feedback
    assert "current_valid_best_status=best_unreliable" in feedback
    assert "best_unreliable_reason=metric_direction_or_source_uncertain" in feedback

def test_low_priority_repeated_llm_review_is_deferred(tmp_path) -> None:
    calls: list[dict] = []

    async def deny(proposal):
        calls.append(dict(proposal))
        return {"action": "DENY_KILL", "reason": "active and useful", "confidence": "high"}

    observer, _sm = make_observer(
        tmp_path,
        worker_id="W00",
        gpu_pool=["0"],
        kill_mode="arbiter",
        arbiter_enabled=True,
        arbiter_mode="llm",
        arbiter_decider=deny,
        arbiter_job_llm_call_cap=10,
        arbiter_job_token_cap=100000,
    )
    proposal = {
        "proposal_id": "rp_low_budget",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_holder_observe",
        "command_id": "W00:bash:00001",
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    first = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))
    second = asyncio.run(observer.arbiter_decide("W00:bash:00001", proposal=proposal, decision_preview={"job_id": "W00:bash:00001"}))

    assert first["action"] == "DENY_KILL"
    assert second["action"] == "OBSERVE_MORE"
    assert "llm_budget_deferred" in second["decision"]["reason"]
    assert len(calls) == 1
    events = _resource_events(tmp_path)
    deferred = [event for event in events if event["event_type"] == "llm_budget_deferred"]
    assert deferred
    assert deferred[-1]["payload"]["budget_priority"]["level"] == "low"

def test_advisory_continue_commitment_becomes_negotiated_timebox(tmp_path: Path) -> None:
    async def advisory(_proposal):
        return {
            "preference": "continue",
            "confidence": "medium",
            "reason": "Need one clean held-out validation metric before judging route value.",
            "commitment": "print clean held-out validation RMSLE",
            "expected_next_artifact": "tmp/train_s03.log with validation RMSLE",
        }

    arbiter_calls: list[dict] = []

    async def arbiter(proposal):
        arbiter_calls.append(dict(proposal))
        return {"action": "DENY_KILL", "reason": "active and useful", "confidence": "high"}

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
        "proposal_id": "rp_negotiated_timebox",
        "proposal_type": "periodic_efficiency_review",
        "reason_code": "active_holder_observe",
        "command_id": job_id,
        "resource_review_boundary": {"kind": "progress_window", "reason": "progress_window_elapsed>=1"},
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "active", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "TIMEBOX"
    assert result["value_review_outcome"] == "TIMEBOX"
    assert result["suppress_main_agent_feedback"] is True
    assert result["decision"]["reason_code"] == "main_agent_advisory_commitment_timebox"
    assert result["decision"]["clear_on"] == "metric_update"
    assert result["computed_timebox_sec"] is not None
    assert observer._review_states[job_id].job_state_bucket == "TIMEBOX_ACTIVE"
    assert observer._review_states[job_id].timebox_success_condition == "metric_update"
    assert arbiter_calls == []
    events = _resource_events(tmp_path)
    outcome_event = next(event for event in events if event["event_type"] == "resource_review_outcome")
    assert outcome_event["payload"]["action"] == "TIMEBOX"
    assert outcome_event["payload"]["reason_code"] == "main_agent_advisory_commitment_timebox"
    timebox_id = observer._review_states[job_id].timebox_id
    proposal_with_advisory = dict(proposal)
    proposal_with_advisory["main_agent_advisory"] = {
        "preference": "continue",
        "confidence": "medium",
        "commitment": "print clean held-out validation RMSLE",
        "expected_next_artifact": "tmp/train_s03.log with validation RMSLE",
    }

    second = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal_with_advisory, decision_preview={"job_id": job_id}))

    assert second["execution_outcome"] == "NO_ACTION"
    assert second["decision"]["reason_code"] == "advisory_timebox_already_active"
    assert observer._review_states[job_id].timebox_id == timebox_id

    third = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert third["execution_outcome"] == "NO_ACTION"
    assert third["decision"].get("active_timebox_preserved") is True
    assert observer._review_states[job_id].timebox_id == timebox_id

def test_recorded_negotiated_failure_skips_advisory_but_uses_resource_arbiter(tmp_path: Path) -> None:
    async def advisory(_proposal):
        raise AssertionError("final resource review must not ask main agent")

    arbiter_seen: dict[str, object] = {}

    async def arbiter(proposal):
        arbiter_seen.update(proposal)
        return {
            "outcome": "KILL",
            "reason_code": "final_resource_budget_review",
            "reason": "proof windows are exhausted and resource budget no longer supports another retry",
            "confidence": "high",
        }

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

    observer._review_states[job_id] = replace(
        new_review_state(job_id),
        failed_proof_window_count=3,
        proof_window_index=3,
        proof_window_source="negotiated",
        timebox_id="tb_missed",
        timebox_success_condition="metric_update",
        timebox_failure_recorded=True,
    )
    proposal = {
        "proposal_id": "rp_later_low_progress",
        "proposal_type": "kill_proposal",
        "reason_code": "active_intervention:low_progress_heartbeat_stalled",
        "command_id": job_id,
        "progress_snapshot": {"runtime_sec": 1200, "progress_signal": "stalled", "progress_confidence": "high"},
        "resource_snapshot": {},
        "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
    }

    result = asyncio.run(observer.arbiter_decide(job_id, proposal=proposal, decision_preview={"job_id": job_id}))

    assert result["execution_outcome"] == "KILL"
    assert result["terminate"] is True
    assert result["decision"].get("reason_code") == "final_resource_budget_review"
    assert result["decision"].get("source") == "resource_arbiter_llm"
    assert arbiter_seen["resource_budget_escalation"]["kind"] == "final_resource_review"
    assert observer._review_states[job_id].job_state_bucket == "ACTIONED"
