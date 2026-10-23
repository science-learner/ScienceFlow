# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FinalCoveragePlan:
    """Reserve one final for a valid medium-confidence candidate route."""

    policy: str = "metric_validity_coverage"
    version: str = "1"
    reserved_slots: int = 0
    high_candidate_count: int = 0
    medium_candidate_count: int = 0
    coverage_candidate_id: str = ""
    coverage_artifact_sha: str = ""
    coverage_reason_code: str = ""

    @property
    def enabled(self) -> bool:
        return self.reserved_slots > 0 and bool(self.coverage_candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "enabled": self.enabled}


def _validity(candidate: dict[str, Any]) -> str:
    return str(candidate.get("metric_validity") or "").strip().lower()


def _artifact_sha(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("artifact_sha") or candidate.get("submission_sha") or ""
    ).strip()


def build_final_coverage_plan(
    ranked: Iterable[dict[str, Any]],
    *,
    final_slots: int,
) -> FinalCoveragePlan:
    """Plan coverage only when high and medium evidence can share the final set."""

    candidates = list(ranked)
    high = [candidate for candidate in candidates if _validity(candidate) == "high"]
    medium = [candidate for candidate in candidates if _validity(candidate) == "medium"]
    if int(final_slots) < 2 or not high or not medium:
        return FinalCoveragePlan(
            high_candidate_count=len(high),
            medium_candidate_count=len(medium),
        )
    coverage = medium[0]
    candidate_id = str(coverage.get("candidate_id") or "").strip()
    if not candidate_id:
        return FinalCoveragePlan(
            high_candidate_count=len(high),
            medium_candidate_count=len(medium),
        )
    return FinalCoveragePlan(
        reserved_slots=1,
        high_candidate_count=len(high),
        medium_candidate_count=len(medium),
        coverage_candidate_id=candidate_id,
        coverage_artifact_sha=_artifact_sha(coverage),
        coverage_reason_code=str(
            coverage.get("metric_validity_reason_code") or ""
        ).strip(),
    )


def order_candidates_for_coverage(
    ranked: Iterable[dict[str, Any]],
    *,
    plan: FinalCoveragePlan,
    coverage_first: bool,
) -> list[dict[str, Any]]:
    """Place the coverage route in a reserved slot without changing hard gates."""

    candidates = list(ranked)
    if not plan.enabled:
        return candidates
    coverage = next(
        (
            candidate
            for candidate in candidates
            if str(candidate.get("candidate_id") or "").strip()
            == plan.coverage_candidate_id
        ),
        None,
    )
    if coverage is None:
        return candidates
    remaining = [candidate for candidate in candidates if candidate is not coverage]
    if coverage_first or not remaining:
        return [coverage, *remaining]
    return [remaining[0], coverage, *remaining[1:]]


__all__ = [
    "FinalCoveragePlan",
    "build_final_coverage_plan",
    "order_candidates_for_coverage",
]
