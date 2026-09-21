from __future__ import annotations

import pytest

from scienceflow.research.control.resources import (
    ResourceReviewEvent,
    ResourceReviewEventKind,
    ResourceReviewMachine,
)
from scienceflow.runtime.core.kernel import TransitionVersionConflictError
from scienceflow.runtime.safety.resource.review.review_signal import build_review_signal
from scienceflow.runtime.safety.resource.review.review_state import (
    ResourceReviewConfig,
    advance_review_state,
    new_review_state,
)
from scienceflow.research.solver.lnr.orchestration.state_machine import LHRStateMachineStore
from scienceflow.research.solver.lnr.orchestration.runtime.services.worker_services import LegacyRunStatus


def test_resource_review_machine_preserves_legacy_reducer_projection() -> None:
    initial = new_review_state("job-1")
    config = ResourceReviewConfig(warmup_windows=0, inactive_windows=3)
    signal = build_review_signal(
        elapsed_sec=60,
        stdout_bytes=100,
        process_tree_cpu={"total_cpu_pct": 120.0, "busy_child_count": 1},
    )
    expected = advance_review_state(initial, signal, config=config)
    machine = ResourceReviewMachine(job_id="job-1", initial_state=initial)
    event = ResourceReviewEvent(
        kind=ResourceReviewEventKind.HEARTBEAT,
        signal=signal,
        config=config,
    )

    first = machine.advance(
        event,
        event_id="job-1:heartbeat:1",
        expected_version=0,
    )
    duplicate = machine.advance(
        event,
        event_id="job-1:heartbeat:1",
        expected_version=0,
    )

    assert first.current == expected
    assert first.current.to_json() == expected.to_json()
    assert (first.machine_type, first.machine_id) == ("resource_review", "job-1")
    assert duplicate.duplicate is True
    assert machine.version == 1


def test_resource_review_machine_rejects_stale_expected_version() -> None:
    machine = ResourceReviewMachine(job_id="job-1")
    signal = build_review_signal(elapsed_sec=60)
    machine.advance(
        ResourceReviewEvent(
            kind=ResourceReviewEventKind.HEARTBEAT,
            signal=signal,
        ),
        event_id="heartbeat-1",
        expected_version=0,
    )

    with pytest.raises(TransitionVersionConflictError):
        machine.advance(
            ResourceReviewEvent(
                kind=ResourceReviewEventKind.HEARTBEAT,
                signal=build_review_signal(elapsed_sec=120),
            ),
            event_id="heartbeat-2",
            expected_version=0,
        )


def test_lhr_store_accepts_only_typed_legacy_projection_statuses(tmp_path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")

    store.mark_run_status(LegacyRunStatus.RUNNING)
    store.mark_run_status("finished")

    assert store.run_status == "finished"
    with pytest.raises(ValueError):
        store.mark_run_status("arbitrary_status")
