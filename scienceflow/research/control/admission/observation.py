"""Deterministic Admission observation builder."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from scienceflow.research.control.admission.contracts import AdmissionPolicyObservation


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class AdmissionObservationBuilder:
    """Copy public facts into a stable policy input without applying an effect."""

    def build(
        self,
        *,
        task_card: Mapping[str, Any],
        rule_result: Mapping[str, Any],
        mode: str,
    ) -> AdmissionPolicyObservation:
        card = copy.deepcopy(dict(task_card or {}))
        rule = copy.deepcopy(dict(rule_result or {}))
        clean_mode = str(mode or "low_confidence").strip().lower()
        payload = {"mode": clean_mode, "task_card": card, "rule_result": rule}
        digest = semantic_hash(payload)
        command_id = str(card.get("job_id") or rule.get("job_id") or "")
        return AdmissionPolicyObservation(
            observation_id=f"adobs_{digest[:16]}",
            command_id=command_id,
            mode=clean_mode,
            task_card=card,
            rule_result=rule,
            input_hash=digest,
        )


__all__ = ["AdmissionObservationBuilder", "canonical_json", "semantic_hash"]
