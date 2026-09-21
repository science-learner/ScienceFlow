# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Idempotent append-only archive for replayable EStra envelopes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scienceflow.foundation.contracts import EstraContext, EstraDecision
from scienceflow.research.control.estra.runtime.commands import map_estra_runtime_command
from scienceflow.research.control.estra.planning.contracts import (
    EstraDecisionEnvelope,
    EstraRecordedPlan,
)


class EstraArchiveStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._envelope_ids = self._load_ids()

    def append(self, envelope: EstraDecisionEnvelope) -> bool:
        envelope_id = str(envelope.envelope_id or "").strip()
        if not envelope_id:
            raise ValueError("envelope_id is required")
        if envelope_id in self._envelope_ids:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(self._payload(envelope), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
        self._envelope_ids.add(envelope_id)
        return True

    def record(
        self,
        *,
        request_id: str,
        context: EstraContext,
        decision: EstraDecision,
        decision_mode: str,
        raw: str,
        expected_stage: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> EstraRecordedPlan:
        input_hash = _hash(
            {
                "context": {
                    "latest_stage": context.latest_stage,
                    "switch_candidates": context.switch_candidates,
                    "trigger_source": context.trigger_source,
                    "metadata": context.metadata,
                },
                "raw": raw,
            }
        )
        output_hash = _hash(decision.to_dict())
        envelope_id = "estra:" + _hash(
            {
                "request_id": request_id,
                "input_hash": input_hash,
                "output_hash": output_hash,
            }
        )[:24]
        command = map_estra_runtime_command(
            decision,
            command_id=f"command:{envelope_id}",
            expected_stage=expected_stage,
            evidence_refs=(envelope_id,),
        )
        command_payload = {
            "schema_version": command.schema_version,
            "command_id": command.command_id,
            "kind": command.kind.value,
            "target_stage": command.target_stage,
            "reason_code": command.reason_code,
            "evidence_refs": list(command.evidence_refs),
            "expected_stage": command.expected_stage,
            "parameters": dict(command.parameters),
        }
        envelope = EstraDecisionEnvelope(
            envelope_id=envelope_id,
            request_id=request_id,
            context=context,
            decision=decision,
            decision_mode=decision_mode,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                **dict(metadata or {}),
                "runtime_command": command_payload,
            },
        )
        return EstraRecordedPlan(
            envelope=envelope,
            command=command,
            archived=self.append(envelope),
        )

    def replay(self) -> tuple[dict[str, object], ...]:
        if not self.path.is_file():
            return ()
        rows: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return tuple(rows)

    def _load_ids(self) -> set[str]:
        return {
            str(row.get("envelope_id") or "")
            for row in self.replay()
            if row.get("envelope_id")
        }

    @staticmethod
    def _payload(envelope: EstraDecisionEnvelope) -> dict[str, object]:
        return {
            "schema_version": envelope.schema_version,
            "envelope_id": envelope.envelope_id,
            "request_id": envelope.request_id,
            "context": {
                "latest_stage": envelope.context.latest_stage,
                "switch_candidates": list(envelope.context.switch_candidates),
                "trigger_source": envelope.context.trigger_source,
                "metadata": dict(envelope.context.metadata),
            },
            "decision": envelope.decision.to_dict(),
            "decision_mode": envelope.decision_mode,
            "input_hash": envelope.input_hash,
            "output_hash": envelope.output_hash,
            "metadata": dict(envelope.metadata),
        }


__all__ = ["EstraArchiveStore"]


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
