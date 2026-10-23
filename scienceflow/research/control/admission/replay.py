"""Append-only Admission decision replay archive."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scienceflow.research.control.admission.contracts import AdmissionDecisionRecord


class AdmissionReplayArchive:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
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
            record_id = str(row.get("record_id") or "") if isinstance(row, dict) else ""
            if record_id:
                self._records.setdefault(record_id, row)

    def append(self, record: AdmissionDecisionRecord) -> bool:
        if record.record_id in self._records:
            return False
        payload = asdict(record)
        payload["schema_version"] = record.contract_schema_version()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        self._records[record.record_id] = payload
        return True

    def get(self, record_id: str) -> dict[str, Any] | None:
        row = self._records.get(str(record_id or ""))
        return dict(row) if row is not None else None

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._records.values())


__all__ = ["AdmissionReplayArchive"]
