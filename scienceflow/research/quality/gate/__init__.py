# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Independent candidate-admission policy API.

Gate measures no artifacts and owns no evaluator lifecycle. Candidate
measurement and admission are composed by :mod:`scienceflow.research.quality.assessment`.
"""

from scienceflow.foundation.contracts import GateDecision
from scienceflow.research.quality.gate.feedback import format_invalid_evaluator_feedback
from scienceflow.research.quality.gate.policy import (
    DefaultGatePolicy,
    GateManager,
    GatePolicy,
    OptimizationFeasibilityGatePolicy,
    decide_metric_event,
    legacy_score_contract_enabled,
    normalize_gate_name,
    resolve_gate_config,
)

__all__ = [
    "DefaultGatePolicy",
    "GateDecision",
    "GateManager",
    "GatePolicy",
    "OptimizationFeasibilityGatePolicy",
    "decide_metric_event",
    "format_invalid_evaluator_feedback",
    "legacy_score_contract_enabled",
    "normalize_gate_name",
    "resolve_gate_config",
]
