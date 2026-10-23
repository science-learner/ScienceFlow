# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Public service facade for candidate measurement and admission."""

from __future__ import annotations

from typing import Any

from scienceflow.research.quality.assessment.pipeline import (
    CandidateAssessmentPipeline,
    _boolish,
    _non_full_result_reason,
    _optional_bool,
    _optional_float,
    _stage_result_signal_event,
)
from scienceflow.foundation.contracts import (
    EvaluationOutcome,
    EvaluationRequest,
    MetricEvent,
)


class CandidateAssessmentService:
    """Stable service API over the evaluator-to-gate assessment pipeline."""

    def __init__(
        self,
        evaluator_manager: Any | None = None,
        gate_manager: Any | None = None,
    ) -> None:
        self.pipeline = CandidateAssessmentPipeline(
            evaluator_manager=evaluator_manager,
            gate_manager=gate_manager,
        )
        self.evaluator_manager = self.pipeline.evaluator_manager
        self.gate_manager = self.pipeline.gate_manager

    @property
    def manager(self) -> Any:
        """Return the evaluator plugin manager owned by the pipeline."""

        return self.evaluator_manager

    @classmethod
    def default(cls) -> "CandidateAssessmentService":
        return cls()

    def build_prompt_contract(self, request: EvaluationRequest) -> str:
        return self.pipeline.build_prompt_contract(request)

    def evaluate(self, request: EvaluationRequest) -> list[EvaluationOutcome]:
        return self.pipeline.assess(request)

    def _evaluate_event(
        self,
        request: EvaluationRequest,
        raw_event: MetricEvent,
    ) -> EvaluationOutcome:
        return self.pipeline._evaluate_event(request, raw_event)


__all__ = [
    "CandidateAssessmentService",
    "_boolish",
    "_non_full_result_reason",
    "_optional_bool",
    "_optional_float",
    "_stage_result_signal_event",
]
