"""Run-level Execution Value source selection with deterministic shadow diff."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Mapping

from scienceflow.foundation.contracts import ExecutionDecision, ExecutionObservation, ValueAssessment
from scienceflow.research.control.execution_value.decision.contracts import (
    ExecutionEvidenceRef,
    ExecutionSafetyAudit,
    ExecutionValueDiff,
    ExecutionValueRecord,
)
from scienceflow.research.control.execution_value.evidence.replay import ExecutionValueReplayArchive
from scienceflow.research.control.execution_value.decision.service import ExecutionValueService


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _payload(decision: ExecutionDecision, observation: ExecutionObservation) -> dict[str, Any]:
    payload = asdict(decision)
    payload["schema_version"] = decision.contract_schema_version()
    payload["observation"] = asdict(observation)
    return payload


def _legacy_decision(observation: ExecutionObservation) -> ExecutionDecision:
    """Explicit compatibility comparator for the pre-policy no-action behavior."""

    if observation.budget_remaining_sec <= 0:
        action = "STOP"
        reason = "budget_exhausted"
        score = -1.0
        confidence = "high"
    elif observation.metric_improved is True or observation.artifact_progress:
        action = "CONTINUE"
        reason = "value_progress_observed"
        score = 1.0 if observation.metric_improved is True else 0.4
        confidence = "high" if observation.metric_improved is True else "medium"
    else:
        action = "CONTINUE"
        reason = "legacy_no_action"
        score = -0.4 if observation.metric_improved is False else 0.0
        confidence = "medium" if observation.metric_improved is False else "low"
    return ExecutionDecision(
        action=action,
        reason_code=reason,
        assessment=ValueAssessment(
            score=score,
            confidence=confidence,
            reasons=(reason,),
        ),
    )


class ExecutionValueDecisionRouter:
    _SOURCES = {"component", "legacy", "shadow"}

    def __init__(
        self,
        *,
        source: str = "component",
        service: ExecutionValueService | None = None,
        archive: ExecutionValueReplayArchive | None = None,
    ) -> None:
        clean = str(source or "component").strip().lower()
        if clean not in self._SOURCES:
            raise ValueError(f"unsupported execution-value decision source: {source}")
        self.source = clean
        self.service = service or ExecutionValueService()
        self.archive = archive

    def decide(
        self,
        observation: ExecutionObservation,
        *,
        signal: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ExecutionValueRecord]:
        component = _payload(self.service.decide(observation), observation)
        legacy = _payload(_legacy_decision(observation), observation)
        selected_source = "legacy" if self.source in {"legacy", "shadow"} else "component"
        selected = legacy if selected_source == "legacy" else component
        component_hash = _hash(component)
        legacy_hash = _hash(legacy)
        changed = tuple(
            sorted(
                key
                for key in set(component) | set(legacy)
                if component.get(key) != legacy.get(key)
            )
        )
        facts = dict(signal or {})
        near_deliverable = bool(
            facts.get("saw_final_score")
            or facts.get("near_deliverable")
            or facts.get("deliverable_complete")
        )
        selected_action = str(selected.get("action") or "").upper()
        audit_reasons = ["execution_value_is_advisory_only"]
        if near_deliverable:
            audit_reasons.append("near_deliverable_veto_remains_external")
        if selected_action in {"STOP", "KILL", "RELEASE"}:
            audit_reasons.append("kill_intent_revalidation_required_before_effect")
        observation_hash = _hash(asdict(observation))
        seed = f"{observation_hash}:{self.source}:{component_hash}:{legacy_hash}"
        record = ExecutionValueRecord(
            record_id="evrec_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
            observation_hash=observation_hash,
            selected_source=selected_source,
            selected=selected,
            component=component,
            legacy=legacy,
            diff=ExecutionValueDiff(
                equal=not changed,
                changed_fields=changed,
                component_hash=component_hash,
                legacy_hash=legacy_hash,
            ),
            safety_audit=ExecutionSafetyAudit(
                effect_free=True,
                safety_veto_required=selected_action in {"STOP", "KILL", "RELEASE"},
                near_deliverable=near_deliverable,
                kill_intent_revalidation_required=selected_action in {"STOP", "KILL", "RELEASE"},
                reasons=tuple(audit_reasons),
            ),
            evidence=(
                ExecutionEvidenceRef(
                    kind="execution_observation",
                    reference=observation.command_id,
                    digest=observation_hash,
                ),
            ),
        )
        if self.archive is not None:
            try:
                self.archive.append(record)
            except OSError:
                # Advisory replay persistence must not alter process control.
                pass
        return selected, record


__all__ = ["ExecutionValueDecisionRouter"]
