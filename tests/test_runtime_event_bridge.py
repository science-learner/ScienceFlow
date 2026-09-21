from __future__ import annotations

import json

from scienceflow.runtime.events import (
    ScienceFlowEventBridge,
    event_bridge_from_env,
    legacy_event_from_runtime,
    runtime_event_from_legacy,
)
from scienceflow.research.solver.lnr.orchestration import state_machine
from scienceflow.research.solver.lnr.orchestration.state_machine import LHRStateMachineStore


def test_event_bridge_is_disabled_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SCIENCEFLOW_RUNTIME_EVENTS", raising=False)
    monkeypatch.delenv("SCIENCEFLOW_RUNTIME_EVENT_LOG", raising=False)
    assert event_bridge_from_env(log_dir=tmp_path, worker_id="W00") is None


def test_generic_environment_uses_generic_default_filename(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SCIENCEFLOW_RUNTIME_EVENTS", "1")
    bridge = event_bridge_from_env(log_dir=tmp_path, worker_id="W00")

    assert bridge is not None
    assert bridge.path == tmp_path / "agent_runtime_events.jsonl"


def test_lhr_dual_write_preserves_legacy_event_and_adds_envelope(tmp_path) -> None:
    runtime_path = tmp_path / "agent_runtime_events.jsonl"
    bridge = ScienceFlowEventBridge(
        runtime_path,
        session_id="session-1",
        run_id="run-1",
        agent_id="W00",
    )
    store = LHRStateMachineStore(
        log_dir=tmp_path,
        worker_id="W00",
        event_observer=bridge.emit_legacy,
    )

    store.append_event(
        "stage_captured",
        task_type="stage_commit",
        task_id="stage_commit:W00:S01",
        status="succeeded",
        payload={"stage_id": "S01", "metric": 2.5},
    )

    legacy = json.loads((tmp_path / "lhr_events.jsonl").read_text())
    mirrored = json.loads(runtime_path.read_text())
    assert legacy["schema_version"] == 1
    assert legacy["event"] == "stage_captured"
    assert legacy["payload"] == {"metric": 2.5, "stage_id": "S01"}
    assert mirrored["schema_version"] == "1.0"
    assert mirrored["sequence"] == 1
    assert mirrored["type"] == "scienceflow.stage_captured"
    assert mirrored["payload"]["data"] == legacy["payload"]


def test_runtime_event_is_authoritative_without_changing_legacy_shape() -> None:
    legacy = {
        "schema_version": 1,
        "timestamp": 123.5,
        "timestamp_utc": "2000-01-01T00:00:00Z",
        "event": "worker_status",
        "task_type": "repl_search",
        "task_id": "repl_search:W00",
        "status": "running",
        "worker_id": "W00",
        "worker_index": 0,
        "payload": {"mode": "single_worker"},
    }
    runtime = runtime_event_from_legacy(
        legacy,
        session_id="session",
        run_id="run",
        agent_id="W00",
    )
    assert runtime.type == "scienceflow.worker_status"
    assert legacy_event_from_runtime(runtime) == legacy


def test_observer_failure_never_changes_legacy_write(tmp_path) -> None:
    def broken(_event):
        raise RuntimeError("observer failed")

    store = LHRStateMachineStore(
        log_dir=tmp_path,
        worker_id="W00",
        event_observer=broken,
    )
    store.append_event("worker_status", status="running")

    legacy = json.loads((tmp_path / "lhr_events.jsonl").read_text())
    assert legacy["event"] == "worker_status"
    assert legacy["status"] == "running"


def test_dual_write_keeps_legacy_files_byte_identical(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(state_machine, "_now_ts", lambda: 1234.5)
    plain_dir = tmp_path / "plain"
    dual_dir = tmp_path / "dual"
    bridge = ScienceFlowEventBridge(
        dual_dir / "agent_runtime_events.jsonl",
        session_id="s",
        run_id="r",
        agent_id="W00",
    )
    plain = LHRStateMachineStore(log_dir=plain_dir, worker_id="W00")
    dual = LHRStateMachineStore(
        log_dir=dual_dir,
        worker_id="W00",
        event_observer=bridge.emit_legacy,
    )

    for store in (plain, dual):
        store.append_event(
            "stage_captured",
            task_type="stage_commit",
            status="succeeded",
            payload={"stage_id": "S01", "metric": 2.5},
        )

    assert (plain_dir / "lhr_events.jsonl").read_bytes() == (
        dual_dir / "lhr_events.jsonl"
    ).read_bytes()
    assert (plain_dir / "lhr_state.json").read_bytes() == (
        dual_dir / "lhr_state.json"
    ).read_bytes()


def test_bridge_restart_repairs_partial_tail_and_continues_sequence(tmp_path) -> None:
    path = tmp_path / "agent_runtime_events.jsonl"
    event = {
        "schema_version": 1,
        "timestamp": 1.0,
        "timestamp_utc": "2000-01-01T00:00:00Z",
        "event": "worker_status",
        "worker_id": "W00",
        "worker_index": 0,
        "payload": {},
    }
    ScienceFlowEventBridge(
        path, session_id="s", run_id="r", agent_id="W00"
    ).emit_legacy(event)
    with path.open("ab") as handle:
        handle.write(b"{partial")
    ScienceFlowEventBridge(
        path, session_id="s", run_id="r", agent_id="W00"
    ).emit_legacy(event)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2]
