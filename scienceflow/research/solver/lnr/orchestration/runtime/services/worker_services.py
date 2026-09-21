# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Narrow capability contracts for single- and multi-worker runners."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from scienceflow.runtime.core.kernel import (
    WorkerLifecycleEvent,
    WorkerLifecycleMachine,
)
from scienceflow.research.state.knowledge.prompt import PromptContextProjection, PromptContextRequest


class LegacyRunStatus(str, Enum):
    """Stable status values projected to the existing on-disk state store."""

    RUNNING = "running"
    REDUCING = "reducing"
    FINISHED = "finished"
    FAILED = "failed"


class WorkerSearchOutcome(str, Enum):
    CONTINUE_SEARCH = "continue_search"
    FINALIZE_CANDIDATE = "finalize_candidate"
    STOP_NO_CANDIDATE = "stop_no_candidate"
    STAGE_SWITCH = "stage_switch"


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """Immutable request used to start one coordinator-owned worker."""

    worker_index: int
    worker_count: int

    @property
    def worker_id(self) -> str:
        return f"W{self.worker_index:02d}"


class WorkerResult(TypedDict, total=False):
    """Stable common fields; domain-specific metrics may remain additive."""

    worker_id: str
    worker_index: int
    status: str
    error: str
    stop_reason: str
    outcome: str
    failure_kind: str
    best_stage: str
    best_metric: Any
    main_tokens_input: int
    main_tokens_cached: int
    main_tokens_output: int
    main_llm_calls: int
    stage_tokens_input: int
    stage_tokens_cached: int
    stage_llm_calls: int
    estra_tokens_input: int
    estra_tokens_cached: int
    estra_llm_calls: int


@dataclass(frozen=True, slots=True)
class LegacyStatusProjection:
    write: Callable[[LegacyRunStatus, dict[str, Any]], None]


class WorkerLifecycleController:
    """Own canonical worker transitions and project compatible status files."""

    def __init__(
        self,
        *,
        worker_id: str,
        projection: LegacyStatusProjection,
    ) -> None:
        self.machine = WorkerLifecycleMachine(worker_id=worker_id)
        self._worker_id = worker_id
        self._projection = projection

    def mark_running(self, payload: dict[str, Any]) -> None:
        self.machine.advance(
            WorkerLifecycleEvent.START,
            event_id=f"{self._worker_id}:start",
            expected_version=0,
        )
        self.machine.advance(
            WorkerLifecycleEvent.READY,
            event_id=f"{self._worker_id}:ready",
            expected_version=1,
        )
        self._projection.write(LegacyRunStatus.RUNNING, payload)

    def mark_finished(self, payload: dict[str, Any]) -> None:
        self._projection.write(LegacyRunStatus.FINISHED, payload)
        self.machine.advance(
            WorkerLifecycleEvent.SUCCEED,
            event_id=f"{self._worker_id}:succeed",
            expected_version=self.machine.version,
        )

    def mark_failed(self, payload: dict[str, Any]) -> None:
        self.machine.advance(
            WorkerLifecycleEvent.FAIL,
            event_id=f"{self._worker_id}:fail",
            expected_version=self.machine.version,
        )
        self._projection.write(LegacyRunStatus.FAILED, payload)


@dataclass(frozen=True, slots=True)
class WorkerRuntimeSpec:
    worker_id: str
    task_description: str
    wall_clock_budget_sec: int
    seed: int
    max_steps: int
    memory_dir: Path


@dataclass(frozen=True, slots=True)
class WorkerRuntimeServices:
    """Capabilities used inside one worker's agent loop."""

    spec: WorkerRuntimeSpec
    lifecycle: WorkerLifecycleController
    prepare_workspace: Callable[[], None]
    load_existing_stage_snapshots: Callable[[], None]
    initialize_snapshot_baseline: Callable[[], bool]
    deadline: Callable[[], float]
    initial_workspace_state: Callable[[], Any]
    parallel_worker_snapshot_for_prompt: Callable[[], Any]
    resource_context_for_prompt: Callable[[], Any]
    skill_hint: Callable[[], Any]
    evaluator_prompt_contract: Callable[[], Any]
    evaluator_task_profile: Callable[[], Any]
    task_runtime_prompt_contract: Callable[[], Any]
    build_prompt_context: Callable[[PromptContextRequest], PromptContextProjection]
    make_agent: Callable[[bool], Any]
    append_event: Callable[[str, dict[str, Any]], None]
    pending_stage_commit_active: Callable[[], bool]
    pending_stage_commit: Callable[[], Any]
    extend_stage_commit_deadline: Callable[[Any], None]
    accumulate_main_run_tokens: Callable[[Any], None]
    evaluator_stop_requested: Callable[[], bool]
    evaluator_stop_reason: Callable[[], str]
    pending_estra: Callable[[], bool]
    restore_pending_estra: Callable[[], Awaitable[None]]
    build_result: Callable[[str], dict[str, Any]]
    retain_live_agent: Callable[[Any], None]
    abandon_pending_stage_commit: Callable[[Any], None]
    metric_feedback_llm: Callable[[], Any | None]


@dataclass(frozen=True, slots=True)
class CoordinatorSpec:
    worker_count: int
    final_artifact_mode: str
    merge_enabled: bool
    root_dir: Path
    log_dir: Path
    solver_name: str
    ledger_filename: str


@dataclass(frozen=True, slots=True)
class CoordinatorServices:
    """Capabilities used to coordinate workers and finalize their results."""

    spec: CoordinatorSpec
    status_projection: LegacyStatusProjection
    append_event: Callable[[str, dict[str, Any]], None]
    run_one_worker: Callable[[WorkerRequest], Awaitable[WorkerResult]]
    aggregate_worker_state_periodically: Callable[
        [int, LegacyRunStatus], Awaitable[None]
    ]
    aggregate_worker_state: Callable[[int, LegacyRunStatus], None]
    close_merge_owner_agent: Callable[[], Awaitable[None]]
    load_worker_candidates: Callable[[WorkerResult, bool], list[dict[str, Any]]]
    write_stage_collection_outputs: Callable[
        [list[WorkerResult], list[dict[str, Any]]], Path
    ]
    write_merge_outputs: Callable[
        [list[WorkerResult], list[dict[str, Any]]],
        Awaitable[dict[str, Any]],
    ]
    merge_dir: Callable[[], Path]
    evaluator_candidate_artifact: Callable[[], str]
    refresh_merge_owner_result: Callable[[list[WorkerResult]], None]
    worker_root: Callable[[int], Path]
    refresh_submission_links: Callable[[int, bool], None]
    write_global_time_trace: Callable[[list[WorkerResult]], None]
    cleanup_coordinator_workspace_shell: Callable[[], None]
    multi_worker_failure_kind: Callable[[list[WorkerResult]], tuple[str, list[str]]]
    multi_worker_stop_reason: Callable[[bool, list[WorkerResult]], tuple[str, Any]]


__all__ = [
    "LegacyRunStatus",
    "LegacyStatusProjection",
    "CoordinatorServices",
    "CoordinatorSpec",
    "WorkerRuntimeServices",
    "WorkerRuntimeSpec",
    "WorkerRequest",
    "WorkerResult",
    "WorkerSearchOutcome",
    "WorkerLifecycleController",
]
