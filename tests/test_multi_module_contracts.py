# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Contracts that protect the independently owned module boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from scienceflow.research.quality.assessment import (
    CandidateAssessmentPipeline,
    CandidateAssessmentService,
)
from scienceflow.foundation.contracts import (
    CandidateAssessment,
    AdmissionDecision,
    CandidateRef,
    EvalContext,
    EvaluationOutcome,
    EvaluationRequest,
    EvaluationResult,
    EstraContext,
    EstraDecision,
    ExecutionDecision,
    ExecutionObservation,
    GateDecision,
    MetricEvent,
    MemoryView,
    ResourceObservation,
    ResourceRequest,
    StageCommitted,
    StageRecord,
    ValueAssessment,
)
from scienceflow.research.quality.evaluator import EvaluatorManager
from scienceflow.research.quality.gate import GateManager


def test_evaluation_outcome_remains_the_candidate_assessment_contract() -> None:
    assert EvaluationOutcome is CandidateAssessment


def test_removed_gate_compatibility_namespace_stays_absent() -> None:
    package_root = Path(__file__).parents[1] / "scienceflow"
    assert not (package_root / "gates").exists()


def test_cross_module_contracts_expose_schema_version_without_payload_drift() -> None:
    classes = (
        CandidateRef,
        EvalContext,
        EvaluationRequest,
        EvaluationResult,
        MetricEvent,
        GateDecision,
        CandidateAssessment,
        StageRecord,
        StageCommitted,
        MemoryView,
        EstraContext,
        EstraDecision,
        ResourceRequest,
        ResourceObservation,
        AdmissionDecision,
        ExecutionObservation,
        ValueAssessment,
        ExecutionDecision,
    )
    assert {cls.contract_schema_version() for cls in classes} == {"1.0"}

    decision = GateDecision(
        action="accept",
        accepted=True,
        candidate_ready=True,
        selection_eligible=True,
        reason_code="accepted",
    )
    assert decision.to_dict() == {
        "action": "accept",
        "accepted": True,
        "candidate_ready": True,
        "selection_eligible": True,
        "reason_code": "accepted",
        "message": "",
    }


def test_new_module_entrypoints_construct_independently() -> None:
    evaluator = EvaluatorManager.default()
    gate = GateManager.default()
    pipeline = CandidateAssessmentPipeline(evaluator, gate)
    service = CandidateAssessmentService(evaluator, gate)

    assert pipeline.evaluator_manager is evaluator
    assert pipeline.gate_manager is gate
    assert service.pipeline.evaluator_manager is evaluator
    assert service.pipeline.gate_manager is gate


def test_contracts_do_not_import_evaluator_or_gate_implementations() -> None:
    root = Path(__file__).parents[1] / "scienceflow" / "foundation" / "contracts"
    forbidden = (
        "scienceflow.research.quality.evaluator",
        "scienceflow.research.quality.gate",
        "scienceflow.research.quality.assessment",
        "scienceflow.research.state.knowledge.memory",
        "scienceflow.research.state.workspace",
        "scienceflow.research.control.resources",
        "scienceflow.research.control.execution_value",
    )
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = str(node.module or "")
                assert not module.startswith(forbidden), (path, module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), (path, alias.name)
