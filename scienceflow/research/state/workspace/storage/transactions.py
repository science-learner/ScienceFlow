# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Crash-safe Stage commit transactions owned by Workspace."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from inquirycraft.tools import atomic_write

from scienceflow.research.state.workspace.adapters.lnr import read_ledger
from scienceflow.research.state.workspace.session.lifecycle import (
    WorkspaceTransactionEvent,
    WorkspaceTransactionMachine,
)

logger = logging.getLogger("scienceflow")

PendingTransaction = dict[str, Any]
WorkspaceEventSink = Callable[[dict[str, Any]], None]


class StageTransactionService:
    """Persist and recover the ledger-to-snapshot commit boundary."""

    def __init__(self, *, ledger_path: str | Path, transaction_root: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.transaction_root = Path(transaction_root)
        self._machines: dict[str, WorkspaceTransactionMachine] = {}

    @property
    def machines(self) -> tuple[WorkspaceTransactionMachine, ...]:
        return tuple(self._machines[key] for key in sorted(self._machines))

    def prepare(
        self,
        *,
        transaction_id: str,
        stage_id: str,
        lineage_id: str,
        ledger_before: str,
        ledger_after: str,
        metric_event: dict[str, Any],
    ) -> PendingTransaction:
        metric_event["stage_transaction_id"] = transaction_id
        backup = self.transaction_root / f"{transaction_id}.ledger_before.md"
        manifest = self.transaction_root / f"{transaction_id}.json"
        atomic_write(backup, ledger_before)
        payload = {
            "version": 2,
            "transaction_id": transaction_id,
            "stage_id": stage_id,
            "lineage_id": lineage_id,
            "status": "prepared",
            "prepared_at": time.time(),
            "ledger_before_path": backup.name,
            "ledger_before_sha256": _sha256(ledger_before),
            "ledger_after_sha256": _sha256(ledger_after),
            "artifact_sha": str(
                metric_event.get("artifact_sha")
                or metric_event.get("submission_sha")
                or ""
            ),
            "gate_accepted": metric_event.get("gate_accepted"),
            "gate_reason_code": str(metric_event.get("gate_reason_code") or ""),
        }
        _write_manifest(manifest, payload)
        machine = WorkspaceTransactionMachine(transaction_id=transaction_id)
        machine.advance(
            WorkspaceTransactionEvent.PREPARE,
            event_id=f"{transaction_id}:prepare",
            expected_version=0,
        )
        self._machines[transaction_id] = machine
        return {"manifest": manifest, "payload": payload}

    def complete(
        self,
        pending: PendingTransaction | None,
        *,
        stage_id: str,
        snapshot: Any,
    ) -> tuple[PendingTransaction | None, bool]:
        manifest, payload = _pending_parts(pending)
        if (
            manifest is None
            or str(payload.get("stage_id") or "") != stage_id
            or str(payload.get("status") or "") != "prepared"
        ):
            return pending, False
        payload.update(
            {
                "status": "committed",
                "committed_at": time.time(),
                "snapshot_id": str(getattr(snapshot, "snapshot_id", "") or ""),
                "snapshot_path": str(getattr(snapshot, "snapshot_path", "") or ""),
            }
        )
        _write_manifest(manifest, payload)
        self._machine_for(payload).advance(
            WorkspaceTransactionEvent.COMMIT,
            event_id=f"{payload.get('transaction_id')}:commit",
            expected_version=1,
        )
        return None, True

    def rollback(
        self,
        pending: PendingTransaction | None,
        *,
        stage_id: str,
        reason: str,
    ) -> tuple[PendingTransaction | None, bool]:
        manifest, payload = _pending_parts(pending)
        if (
            manifest is None
            or str(payload.get("stage_id") or "") != stage_id
            or str(payload.get("status") or "") != "prepared"
        ):
            return pending, False
        backup = manifest.parent / str(payload.get("ledger_before_path") or "")
        try:
            atomic_write(self.ledger_path, backup.read_text(encoding="utf-8"))
            payload.update(
                {
                    "status": "rolled_back",
                    "rolled_back_at": time.time(),
                    "rollback_reason": reason,
                }
            )
            _write_manifest(manifest, payload)
            self._machine_for(payload).advance(
                WorkspaceTransactionEvent.ROLLBACK,
                event_id=f"{payload.get('transaction_id')}:rollback",
                expected_version=1,
            )
            return None, True
        except OSError:
            logger.exception("[workspace] failed to roll back stage transaction %s", stage_id)
            return pending, False

    def recover(
        self,
        *,
        snapshots: Iterable[Any],
        event_sink: WorkspaceEventSink | None = None,
    ) -> None:
        if not self.transaction_root.is_dir():
            return
        candidates = list(snapshots)
        for manifest in sorted(self.transaction_root.glob("*.json")):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("status") != "prepared":
                continue
            stage_id = str(payload.get("stage_id") or "").strip().upper()
            snapshot = _matching_snapshot(payload, stage_id=stage_id, snapshots=candidates)
            if snapshot is not None:
                payload.update(
                    {
                        "status": "committed",
                        "committed_at": time.time(),
                        "recovered_after_restart": True,
                        "snapshot_id": str(getattr(snapshot, "snapshot_id", "") or ""),
                        "snapshot_path": str(getattr(snapshot, "snapshot_path", "") or ""),
                    }
                )
                _write_manifest(manifest, payload)
                self._machine_for(payload, restored_status="prepared").advance(
                    WorkspaceTransactionEvent.COMMIT,
                    event_id=f"{payload.get('transaction_id')}:commit",
                    expected_version=1,
                )
                continue

            backup = manifest.parent / str(payload.get("ledger_before_path") or "")
            try:
                ledger_before = backup.read_text(encoding="utf-8")
                current = read_ledger(self.ledger_path)
            except OSError:
                continue
            if _sha256(current) != str(payload.get("ledger_after_sha256") or ""):
                continue
            atomic_write(self.ledger_path, ledger_before)
            payload.update(
                {
                    "status": "rolled_back",
                    "rolled_back_at": time.time(),
                    "rollback_reason": "resume_missing_snapshot",
                }
            )
            _write_manifest(manifest, payload)
            self._machine_for(payload, restored_status="prepared").advance(
                WorkspaceTransactionEvent.ROLLBACK,
                event_id=f"{payload.get('transaction_id')}:rollback",
                expected_version=1,
            )
            if event_sink is not None:
                event_sink(
                    {
                        "event": "incomplete_stage_transaction_rolled_back",
                        "stage_id": stage_id,
                        "transaction_id": payload.get("transaction_id"),
                    }
                )

    def _machine_for(
        self,
        payload: dict[str, Any],
        *,
        restored_status: str = "",
    ) -> WorkspaceTransactionMachine:
        transaction_id = str(payload.get("transaction_id") or "")
        machine = self._machines.get(transaction_id)
        if machine is not None:
            return machine
        machine = WorkspaceTransactionMachine(transaction_id=transaction_id)
        status = restored_status or str(payload.get("status") or "")
        if status in {"prepared", "committed", "rolled_back"}:
            machine.advance(
                WorkspaceTransactionEvent.PREPARE,
                event_id=f"{transaction_id}:prepare",
                expected_version=0,
            )
        if status == "committed":
            machine.advance(
                WorkspaceTransactionEvent.COMMIT,
                event_id=f"{transaction_id}:commit",
                expected_version=1,
            )
        elif status == "rolled_back":
            machine.advance(
                WorkspaceTransactionEvent.ROLLBACK,
                event_id=f"{transaction_id}:rollback",
                expected_version=1,
            )
        self._machines[transaction_id] = machine
        return machine


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _pending_parts(
    pending: PendingTransaction | None,
) -> tuple[Path | None, dict[str, Any]]:
    if not isinstance(pending, dict):
        return None, {}
    payload = pending.get("payload")
    manifest = pending.get("manifest")
    return (
        manifest if isinstance(manifest, Path) else None,
        payload if isinstance(payload, dict) else {},
    )


def _matching_snapshot(
    payload: dict[str, Any],
    *,
    stage_id: str,
    snapshots: Iterable[Any],
) -> Any | None:
    expected_lineage = str(payload.get("lineage_id") or "").strip()
    expected_artifact = str(payload.get("artifact_sha") or "").strip()
    expected_transaction = str(payload.get("transaction_id") or "").strip()
    for candidate in snapshots:
        if str(getattr(candidate, "stage_id", "") or "").strip().upper() != stage_id:
            continue
        raw_source = getattr(candidate, "source_event", {})
        source = raw_source if isinstance(raw_source, dict) else {}
        candidate_artifact = str(
            source.get("artifact_sha") or source.get("submission_sha") or ""
        ).strip()
        candidate_transaction = str(source.get("stage_transaction_id") or "").strip()
        if expected_transaction and candidate_transaction != expected_transaction:
            continue
        if expected_lineage and str(
            getattr(candidate, "lineage_id", "") or ""
        ).strip() != expected_lineage:
            continue
        if expected_artifact and candidate_artifact != expected_artifact:
            continue
        return candidate
    return None


__all__ = ["PendingTransaction", "StageTransactionService"]
