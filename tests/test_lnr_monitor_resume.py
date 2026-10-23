"""Process resume preserves durable monitor totals without reviving old leases."""

import json

from scienceflow.research.solver.lnr.orchestration.state_machine import (
    LHRStateMachineStore,
)


def test_monitor_resume_keeps_stage_best_and_estra_without_reemitting(tmp_path):
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    store.append_event(
        "stage_captured",
        payload={
            "stage_id": "S01",
            "snapshot_id": "W00-L01-S01",
            "metric_event": {
                "metric_value": 2.6,
                "metric_name": "radii_sum",
                "lower_is_better": False,
                "validation_ok": True,
                "candidate_ready": True,
                "selection_eligible": True,
                "metric_validity": "high",
                "evaluator_backend": "task_package",
                "evaluator_status": "ok",
                "submission_status": "ready",
            },
        },
    )
    store.append_event("estra_decision", payload={"action": "keep_current"})
    store.mark_run_status("finished")
    original = store.events_path.read_bytes()
    resumed = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    assert resumed.events_path.read_bytes() == original
    assert resumed.stage_count == 1
    assert resumed.estra_decision_count == 1
    assert resumed.global_best == store.global_best
    assert resumed.global_best["metric_value"] == 2.6
    assert resumed.event_count == store.event_count
    resumed.mark_run_status("running")
    state = json.loads(resumed.state_path.read_text())
    assert state["stage_count"] == 1
    assert state["global_best"]["metric_value"] == 2.6
    resumed.append_event("estra_decision", payload={"action": "keep_current"})
    again = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    assert again.stage_count == 1
    assert again.estra_decision_count == 2
    assert again.resource_active_leases == {}


def test_monitor_does_not_attribute_other_worker_history(tmp_path):
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W01")
    store.append_event("stage_captured", payload={"stage_id": "S01"})
    other = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    assert other.stage_count == 0
    assert other.event_count == 0


def test_stage_csv_preserves_close_metric_ordering():
    from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
        _fmt_csv_value,
    )

    first, second = 0.0556279981, 0.0556279982
    assert float(_fmt_csv_value(first)) == first
    assert float(_fmt_csv_value(second)) == second
    assert float(_fmt_csv_value(first)) < float(_fmt_csv_value(second))
