# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Workspace service boundary over the existing LNR stores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scienceflow.foundation.contracts import StageCommitted, StageRecord
from scienceflow.research.state.workspace.adapters.lnr import (
    InitWorkspaceResult,
    initialize_workspace_from_path,
    parse_stage_cards,
    read_ledger,
)
from scienceflow.research.state.workspace.storage.transactions import StageTransactionService


@dataclass(slots=True)
class WorkspaceService:
    """Read authoritative stage facts without depending on Memory."""

    root_dir: Path
    workspace_dir: Path
    ledger_path: Path

    def __init__(
        self,
        *,
        root_dir: str | Path,
        workspace_dir: str | Path,
        ledger_path: str | Path,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.workspace_dir = Path(workspace_dir)
        self.ledger_path = Path(ledger_path)

    def prepare_from(self, source_workspace: str | Path) -> InitWorkspaceResult:
        return initialize_workspace_from_path(
            source_workspace=source_workspace,
            target_workspace=self.workspace_dir,
        )

    def list_stages(self) -> tuple[StageRecord, ...]:
        cards = parse_stage_cards(read_ledger(self.ledger_path))
        return tuple(
            StageRecord(
                stage_id=card.stage_id,
                metric=card.metric,
                lower_is_better=card.lower_is_better,
                metric_validity=card.metric_validity,
                selection_eligible=card.selection_eligible,
                brief=card.brief,
                why=card.why,
                files=card.files,
                body=card.body,
                metadata={"stage_events": list(card.stage_events)},
            )
            for card in cards
        )

    def stage_committed_events(self) -> tuple[StageCommitted, ...]:
        workspace_id = self._workspace_id()
        events: list[StageCommitted] = []
        for sequence, stage in enumerate(self.list_stages(), start=1):
            digest = hashlib.sha256(
                json.dumps(
                    stage.to_dict(), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            events.append(
                StageCommitted(
                    event_id=f"stage-committed:{workspace_id}:{stage.stage_id}:{digest[:16]}",
                    workspace_id=workspace_id,
                    stage=stage,
                    sequence=sequence,
                )
            )
        return tuple(events)

    def stage_transactions(
        self, *, transaction_root: str | Path
    ) -> StageTransactionService:
        return StageTransactionService(
            ledger_path=self.ledger_path,
            transaction_root=transaction_root,
        )

    def _workspace_id(self) -> str:
        identity = str(self.workspace_dir.resolve(strict=False))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


__all__ = ["WorkspaceService"]
