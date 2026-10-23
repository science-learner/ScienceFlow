# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Composition root for independently evolving ScienceFlow modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scienceflow.research.control.admission import AdmissionPolicyService
from scienceflow.research.quality.assessment import CandidateAssessmentPipeline
from scienceflow.research.quality.evaluator import EvaluatorManager
from scienceflow.research.control.execution_value import ExecutionValueService
from scienceflow.research.quality.gate import GateManager
from scienceflow.research.state.knowledge.memory import MemoryService
from scienceflow.research.state.knowledge.prompt import PromptContextBuilder
from scienceflow.research.control.resources import ResourceManagementService
from scienceflow.runtime.core.stage import StageLifecycleCoordinator
from scienceflow.runtime.observability.telemetry import CorrelationJournal
from scienceflow.research.control.estra import EstraArchiveStore, EstraPlanner, EstraService
from scienceflow.research.quality.finalization import FinalizationService
from scienceflow.research.state.workspace import WorkspaceService


@dataclass(frozen=True, slots=True)
class LnrModuleGraph:
    workspace: WorkspaceService
    memory: MemoryService
    evaluator: EvaluatorManager
    gate: GateManager
    assessment: CandidateAssessmentPipeline
    estra: EstraService
    estra_planner: EstraPlanner
    estra_archive: EstraArchiveStore
    resource_management: ResourceManagementService
    admission: AdmissionPolicyService
    execution_value: ExecutionValueService
    stage_lifecycle: StageLifecycleCoordinator
    finalization: FinalizationService
    prompt_context: PromptContextBuilder
    telemetry: CorrelationJournal


def build_lnr_module_graph(
    *,
    root_dir: str | Path,
    workspace_dir: str | Path,
    ledger_path: str | Path,
    module_state_dir: str | Path | None = None,
) -> LnrModuleGraph:
    evaluator = EvaluatorManager.default()
    gate = GateManager.default()
    estra = EstraService()
    state_root = Path(module_state_dir or Path(workspace_dir) / ".module_state")
    return LnrModuleGraph(
        workspace=WorkspaceService(
            root_dir=root_dir,
            workspace_dir=workspace_dir,
            ledger_path=ledger_path,
        ),
        memory=MemoryService(workspace_dir=workspace_dir),
        evaluator=evaluator,
        gate=gate,
        assessment=CandidateAssessmentPipeline(evaluator, gate),
        estra=estra,
        estra_planner=EstraPlanner(estra),
        estra_archive=EstraArchiveStore(
            state_root
            / "estra"
            / "decisions.jsonl"
        ),
        resource_management=ResourceManagementService(),
        admission=AdmissionPolicyService(),
        execution_value=ExecutionValueService(),
        stage_lifecycle=StageLifecycleCoordinator(),
        finalization=FinalizationService(),
        prompt_context=PromptContextBuilder(),
        telemetry=CorrelationJournal(
            state_root / "telemetry" / "correlation.jsonl"
        ),
    )


__all__ = ["LnrModuleGraph", "build_lnr_module_graph"]
