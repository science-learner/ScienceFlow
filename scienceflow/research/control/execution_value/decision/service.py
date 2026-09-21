# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Pure execution-value assessment; actions are Runtime recommendations."""

from __future__ import annotations

from scienceflow.foundation.contracts import (
    ExecutionDecision,
    ExecutionObservation,
    ValueAssessment,
)


class ExecutionValueService:
    def __init__(self, *, no_progress_timebox_sec: float = 300.0) -> None:
        self.no_progress_timebox_sec = max(1.0, float(no_progress_timebox_sec))

    def assess(self, observation: ExecutionObservation) -> ValueAssessment:
        score = 0.0
        reasons: list[str] = []
        confidence = "low"
        if observation.metric_improved is True:
            score += 1.0
            reasons.append("verified_metric_improvement")
            confidence = "high"
        elif observation.metric_improved is False:
            score -= 0.4
            reasons.append("verified_metric_not_improved")
            confidence = "medium"
        if observation.artifact_progress:
            score += 0.4
            reasons.append("artifact_progress")
            if confidence == "low":
                confidence = "medium"
        if observation.recoverable_artifact:
            score += 0.2
            reasons.append("recoverable_artifact")
        if observation.budget_remaining_sec <= 0:
            score -= 1.0
            reasons.append("budget_exhausted")
            confidence = "high"
        return ValueAssessment(
            score=score,
            confidence=confidence,
            reasons=tuple(reasons or ["insufficient_value_evidence"]),
        )

    def decide(self, observation: ExecutionObservation) -> ExecutionDecision:
        assessment = self.assess(observation)
        if observation.budget_remaining_sec <= 0:
            return ExecutionDecision(
                action="STOP",
                reason_code="budget_exhausted",
                assessment=assessment,
            )
        if observation.metric_improved is True or observation.artifact_progress:
            return ExecutionDecision(
                action="CONTINUE",
                reason_code="value_progress_observed",
                assessment=assessment,
            )
        if observation.elapsed_sec >= self.no_progress_timebox_sec:
            return ExecutionDecision(
                action="TIMEBOX",
                reason_code="no_verified_value_progress",
                assessment=assessment,
                timebox_sec=min(
                    self.no_progress_timebox_sec,
                    max(1.0, observation.budget_remaining_sec),
                ),
            )
        return ExecutionDecision(
            action="CONTINUE",
            reason_code="observation_window_incomplete",
            assessment=assessment,
        )


__all__ = ["ExecutionValueService"]
