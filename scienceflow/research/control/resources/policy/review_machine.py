# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Versioned transition adapter for the existing ResourceReview pure reducers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from scienceflow.runtime.core.kernel.state_machines import TransitionEngine
from scienceflow.runtime.safety.resource.review.review_boundary import ReviewBoundary
from scienceflow.runtime.safety.resource.review.review_signal import ResourceReviewSignal
from scienceflow.runtime.safety.resource.review.review_state import (
    ResourceReviewConfig,
    ResourceReviewState,
    advance_review_state,
    apply_kill,
    apply_no_action,
    apply_timebox,
    mark_review_emitted,
    new_review_state,
)


class ResourceReviewEventKind(str, Enum):
    HEARTBEAT = "heartbeat"
    REVIEW_EMITTED = "review_emitted"
    APPLY_TIMEBOX = "apply_timebox"
    APPLY_NO_ACTION = "apply_no_action"
    APPLY_KILL = "apply_kill"
    RESET_STALE_KILL = "reset_stale_kill"


@dataclass(frozen=True)
class ResourceReviewEvent:
    kind: ResourceReviewEventKind
    signal: ResourceReviewSignal | None = None
    config: ResourceReviewConfig | None = None
    boundary: ReviewBoundary | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


def _advance(
    state: ResourceReviewState,
    event: ResourceReviewEvent,
) -> ResourceReviewState:
    if event.kind is ResourceReviewEventKind.HEARTBEAT and event.signal is not None:
        return advance_review_state(state, event.signal, config=event.config)
    if event.kind is ResourceReviewEventKind.REVIEW_EMITTED and event.boundary is not None:
        return mark_review_emitted(state, event.boundary)
    if event.kind is ResourceReviewEventKind.APPLY_TIMEBOX:
        return apply_timebox(state, **dict(event.parameters))
    if event.kind is ResourceReviewEventKind.APPLY_NO_ACTION:
        return apply_no_action(state, **dict(event.parameters))
    if event.kind is ResourceReviewEventKind.APPLY_KILL:
        return apply_kill(state)
    if event.kind is ResourceReviewEventKind.RESET_STALE_KILL:
        scope_memory = dict(state.metric_scope_memory or {})
        scope_memory.pop(str(state.metric_scope_key or ""), None)
        return replace(
            state,
            job_state_bucket="RUNNING_HEALTHY",
            no_useful_progress_windows=0,
            timebox_id="",
            timebox_windows=0,
            timebox_deadline_windows=0,
            timebox_success_condition="",
            timebox_failure_recorded=False,
            proof_window_index=0,
            failed_proof_window_count=0,
            proof_window_source="",
            metric_scope_memory=scope_memory,
            last_outcome="STALE_KILL_INTENT",
        )
    raise ValueError(f"invalid resource review event: {event.kind.value}")


class ResourceReviewMachine(
    TransitionEngine[ResourceReviewState, ResourceReviewEvent]
):
    def __init__(
        self,
        *,
        job_id: str,
        initial_state: ResourceReviewState | None = None,
    ) -> None:
        super().__init__(
            initial_state=initial_state or new_review_state(job_id),
            reducer=_advance,
            machine_type="resource_review",
            machine_id=job_id,
        )


__all__ = [
    "ResourceReviewEvent",
    "ResourceReviewEventKind",
    "ResourceReviewMachine",
]
