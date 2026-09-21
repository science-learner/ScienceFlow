# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Independent evaluator API.

Evaluator implementations, providers, cache, and event logging are owned by
this package.  Cross-module payloads remain in :mod:`scienceflow.foundation.contracts`.
"""

from scienceflow.foundation.contracts import (
    CandidateRef,
    EvalContext,
    EvaluationRequest,
    EvaluationResult,
    MetricEvent,
)
from scienceflow.research.quality.evaluator.manager import EvaluatorManager

__all__ = [
    "CandidateRef",
    "EvalContext",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluatorManager",
    "MetricEvent",
]
