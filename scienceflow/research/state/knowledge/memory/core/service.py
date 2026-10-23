# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Idempotent memory projection service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from scienceflow.foundation.contracts import MemoryView, StageCommitted
from scienceflow.research.state.knowledge.memory.core.compactor import MemoryCompactor
from scienceflow.research.state.knowledge.memory.core.contracts import (
    MemoryCompactionRequest,
    MemoryCompactionResult,
    ProtectedContextRequest,
    ProtectedContextResult,
)
from scienceflow.research.state.knowledge.memory.context.protected_context import ProtectedContextAdapter
from scienceflow.research.state.knowledge.memory.adapters.lnr_stage import (
    StageMemoryView,
    build_stage_memory_view,
    sync_current_segment,
)


class MemoryService:
    """Build cognitive views from events; never writes Workspace facts."""

    def __init__(self, *, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self._stage_events: dict[str, StageCommitted] = {}
        self.compactor = MemoryCompactor()
        self.protected_context = ProtectedContextAdapter()

    def consume(self, event: StageCommitted) -> bool:
        if event.event_id in self._stage_events:
            return False
        self._stage_events[event.event_id] = event
        return True

    def rebuild(self, events: Iterable[StageCommitted]) -> MemoryView:
        self._stage_events.clear()
        for event in sorted(events, key=lambda item: (item.sequence, item.event_id)):
            self.consume(event)
        ordered = sorted(
            self._stage_events.values(), key=lambda item: (item.sequence, item.event_id)
        )
        lines = [
            f"{event.stage.stage_id}: metric={event.stage.metric or '-'}; "
            f"brief={event.stage.brief or '-'}"
            for event in ordered
        ]
        return MemoryView(
            text="\n".join(lines),
            source_event_ids=tuple(event.event_id for event in ordered),
            last_sequence=max((event.sequence for event in ordered), default=0),
        )

    def build_stage_view(
        self,
        cards: list[Any],
        *,
        context_budget_chars: int,
        rebuild_on_stale: bool,
        target_stage: str = "",
        latest_stage: str = "",
        best_stage: str = "",
    ) -> StageMemoryView:
        return build_stage_memory_view(
            self.workspace_dir,
            cards,
            context_budget_chars=context_budget_chars,
            rebuild_on_stale=rebuild_on_stale,
            target_stage=target_stage,
            latest_stage=latest_stage,
            best_stage=best_stage,
        )

    def sync_current_segment(self, cards: Iterable[Any]) -> None:
        sync_current_segment(self.workspace_dir, cards)

    def compact(self, request: MemoryCompactionRequest) -> MemoryCompactionResult:
        return self.compactor.compact(request)

    def protect_context(
        self, context: Any, request: ProtectedContextRequest
    ) -> ProtectedContextResult:
        return self.protected_context.apply(context, request)


__all__ = ["MemoryService"]
