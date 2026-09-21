"""Pure Admission policy service and run-level decision source router."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping

from scienceflow.research.control.admission.contracts import AdmissionDecisionRecord, AdmissionPolicyDiff
from scienceflow.research.control.admission.observation import AdmissionObservationBuilder, semantic_hash
from scienceflow.research.control.admission.policy import (
    normalize_admission_decision,
    should_request_admission_llm,
)
from scienceflow.research.control.admission.replay import AdmissionReplayArchive


LegacyNormalizer = Callable[..., dict[str, Any]]


class AdmissionPolicyService:
    """Interpret Admission facts; never acquires, releases, queues, or kills."""

    def __init__(self, builder: AdmissionObservationBuilder | None = None) -> None:
        self.builder = builder or AdmissionObservationBuilder()

    def should_review(self, rule_result: Mapping[str, Any], *, mode: str) -> bool:
        return should_request_admission_llm(dict(rule_result), mode=mode)

    def normalize(
        self,
        raw: Mapping[str, Any],
        *,
        task_card: Mapping[str, Any],
        rule_result: Mapping[str, Any],
        source: str,
    ) -> dict[str, Any]:
        return normalize_admission_decision(
            dict(raw),
            task_card=dict(task_card),
            rule_result=dict(rule_result),
            source=source,
        )


class AdmissionDecisionRouter:
    """Select component/legacy output once per run while always producing a diff."""

    _SOURCES = {"component", "legacy", "shadow"}

    def __init__(
        self,
        *,
        source: str = "component",
        service: AdmissionPolicyService | None = None,
        archive: AdmissionReplayArchive | None = None,
        legacy_normalizer: LegacyNormalizer | None = None,
    ) -> None:
        clean = str(source or "component").strip().lower()
        if clean not in self._SOURCES:
            raise ValueError(f"unsupported admission decision source: {source}")
        self.source = clean
        self.service = service or AdmissionPolicyService()
        self.archive = archive
        self.legacy_normalizer = legacy_normalizer or normalize_admission_decision

    def should_review(
        self,
        rule_result: Mapping[str, Any],
        *,
        mode: str,
    ) -> bool:
        component = self.service.should_review(rule_result, mode=mode)
        legacy = should_request_admission_llm(dict(rule_result), mode=mode)
        return legacy if self.source in {"legacy", "shadow"} else component

    def decide(
        self,
        raw: Mapping[str, Any],
        *,
        task_card: Mapping[str, Any],
        rule_result: Mapping[str, Any],
        mode: str,
        source: str,
    ) -> AdmissionDecisionRecord:
        observation = self.service.builder.build(
            task_card=task_card,
            rule_result=rule_result,
            mode=mode,
        )
        component = self.service.normalize(
            raw,
            task_card=observation.task_card,
            rule_result=observation.rule_result,
            source=source,
        )
        legacy = self.legacy_normalizer(
            dict(raw),
            task_card=dict(observation.task_card),
            rule_result=dict(observation.rule_result),
            source=source,
        )
        component_hash = semantic_hash(component)
        legacy_hash = semantic_hash(legacy)
        changed = tuple(
            sorted(
                key
                for key in set(component) | set(legacy)
                if component.get(key) != legacy.get(key)
            )
        )
        selected_source = "legacy" if self.source in {"legacy", "shadow"} else "component"
        selected = legacy if selected_source == "legacy" else component
        record_seed = f"{observation.input_hash}:{semantic_hash(raw)}:{self.source}:{component_hash}:{legacy_hash}"
        record_id = "adrec_" + hashlib.sha256(record_seed.encode("utf-8")).hexdigest()[:20]
        record = AdmissionDecisionRecord(
            record_id=record_id,
            observation_id=observation.observation_id,
            selected_source=selected_source,
            selected=selected,
            component=component,
            legacy=legacy,
            diff=AdmissionPolicyDiff(
                equal=not changed,
                changed_fields=changed,
                component_hash=component_hash,
                legacy_hash=legacy_hash,
            ),
            safety_veto_preserved=not (
                str(selected.get("admission_action") or "").upper()
                in {"RUN_NOW", "OBSERVE_THEN_RUN"}
                and not bool(selected.get("lease_grantable_by_llm"))
            ),
        )
        if self.archive is not None:
            try:
                self.archive.append(record)
            except OSError:
                # Decision persistence is observable telemetry, never an effect gate.
                pass
        return record


__all__ = ["AdmissionDecisionRouter", "AdmissionPolicyService"]
