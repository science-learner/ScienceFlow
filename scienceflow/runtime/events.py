"""Opt-in bridge from existing ScienceFlow events to the runtime JSONL envelope.

The legacy event is always written first by its original producer. Bridge failures are
isolated and therefore cannot change ScienceFlow control flow or existing workspace
files during the dual-write migration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from inquirycraft.events import RuntimeEvent, SyncJsonlEventSink

_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENABLED_ENV = "SCIENCEFLOW_RUNTIME_EVENTS"
_PATH_ENV = "SCIENCEFLOW_RUNTIME_EVENT_LOG"


def runtime_event_from_legacy(
    legacy_event: Mapping[str, Any],
    *,
    session_id: str,
    run_id: str,
    agent_id: str,
    sequence: int = 0,
) -> RuntimeEvent:
    """Promote one historical ScienceFlow event into the runtime envelope."""

    event_name = str(legacy_event.get("event") or "event")
    return RuntimeEvent(
        type=f"scienceflow.{event_name}",
        session_id=str(session_id or run_id or "scienceflow"),
        run_id=str(run_id or session_id or "scienceflow"),
        agent_id=str(agent_id or legacy_event.get("worker_id") or "W00"),
        sequence=sequence,
        timestamp=str(legacy_event.get("timestamp_utc") or ""),
        payload={
            "legacy_schema_version": legacy_event.get("schema_version"),
            "legacy_timestamp": legacy_event.get("timestamp"),
            "task_type": legacy_event.get("task_type", ""),
            "task_id": legacy_event.get("task_id", ""),
            "status": legacy_event.get("status", ""),
            "worker_id": legacy_event.get("worker_id", agent_id),
            "worker_index": legacy_event.get("worker_index", 0),
            "data": dict(legacy_event.get("payload") or {}),
        },
    )


def legacy_event_from_runtime(event: RuntimeEvent) -> dict[str, Any]:
    """Render the byte-contract fields consumed by historical SF log readers."""

    payload = dict(event.payload)
    event_name = event.type.removeprefix("scienceflow.")
    return {
        "schema_version": payload.get("legacy_schema_version", 1),
        "timestamp": payload.get("legacy_timestamp"),
        "timestamp_utc": event.timestamp,
        "event": event_name,
        "task_type": payload.get("task_type", ""),
        "task_id": payload.get("task_id", ""),
        "status": payload.get("status", ""),
        "worker_id": payload.get("worker_id", event.agent_id),
        "worker_index": payload.get("worker_index", 0),
        "payload": dict(payload.get("data") or {}),
    }


class ScienceFlowEventBridge:
    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str,
        run_id: str,
        agent_id: str,
    ) -> None:
        self.path = Path(path)
        self.session_id = str(session_id or run_id or "scienceflow")
        self.run_id = str(run_id or session_id or "scienceflow")
        self.agent_id = str(agent_id or "W00")
        self._sink = SyncJsonlEventSink(self.path)

    def emit_legacy(self, legacy_event: Mapping[str, Any]) -> None:
        event = runtime_event_from_legacy(
            legacy_event,
            session_id=self.session_id,
            run_id=self.run_id,
            agent_id=self.agent_id,
        )
        emitted = self._sink.emit(event)
        if emitted.type.removeprefix("scienceflow.") in {
            "worker_status",
            "stage_captured",
            "stage_commit_ok",
        }:
            self._sink.flush()


def event_bridge_from_env(
    *, log_dir: Path, worker_id: str
) -> ScienceFlowEventBridge | None:
    enabled = os.environ.get(_ENABLED_ENV, "").strip().lower()
    explicit_path = os.environ.get(_PATH_ENV, "").strip()
    if enabled not in _TRUE_VALUES and not explicit_path:
        return None
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = log_dir / path
    else:
        path = log_dir / "agent_runtime_events.jsonl"
    run_id = os.environ.get("SCIENCEFLOW_RUN_ID", "").strip() or log_dir.parent.name
    session_id = os.environ.get("SCIENCEFLOW_SESSION_ID", "").strip() or run_id
    return ScienceFlowEventBridge(
        path,
        session_id=session_id,
        run_id=run_id,
        agent_id=worker_id,
    )


__all__ = [
    "ScienceFlowEventBridge",
    "event_bridge_from_env",
    "legacy_event_from_runtime",
    "runtime_event_from_legacy",
]
