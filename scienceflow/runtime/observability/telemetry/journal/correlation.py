"""Append-only, idempotent cross-component correlation journal."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from scienceflow.runtime.observability.telemetry.journal.contracts import CorrelationEnvelope


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CorrelationJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._ids: set[str] = set()
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            correlation_id = str(row.get("correlation_id") or "") if isinstance(row, dict) else ""
            if correlation_id:
                self._ids.add(correlation_id)

    def record(
        self,
        *,
        source: str,
        event_type: str,
        payload: Mapping[str, Any],
        run_id: str = "",
        worker_id: str = "",
        process_id: str = "",
        stage_id: str = "",
        event_id: str = "",
        sequence: int = 0,
    ) -> CorrelationEnvelope:
        normalized = _jsonable(dict(payload or {}))
        payload_hash = _hash(normalized)
        identity = {
            "source": source,
            "event_type": event_type,
            "run_id": run_id,
            "worker_id": worker_id,
            "process_id": process_id,
            "stage_id": stage_id,
            "event_id": event_id,
            "sequence": int(sequence or 0),
            "payload_hash": payload_hash,
        }
        envelope = CorrelationEnvelope(
            correlation_id="corr_" + _hash(identity)[:24],
            source=str(source or "unknown"),
            event_type=str(event_type or "event"),
            run_id=str(run_id or ""),
            worker_id=str(worker_id or ""),
            process_id=str(process_id or ""),
            stage_id=str(stage_id or ""),
            event_id=str(event_id or ""),
            sequence=max(0, int(sequence or 0)),
            payload_hash=payload_hash,
            payload=normalized,
        )
        self.append(envelope)
        return envelope

    def append(self, envelope: CorrelationEnvelope) -> bool:
        with self._lock:
            if envelope.correlation_id in self._ids:
                return False
            payload = _jsonable(asdict(envelope))
            payload["schema_version"] = envelope.contract_schema_version()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            self._ids.add(envelope.correlation_id)
            return True


__all__ = ["CorrelationJournal"]
