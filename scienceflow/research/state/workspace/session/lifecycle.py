# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Domain-owned lifecycle for crash-safe Workspace stage transactions."""

from __future__ import annotations

from enum import Enum

from scienceflow.runtime.core.kernel.state_machines import (
    InvalidTransitionError,
    TransitionEngine,
)


class WorkspaceTransactionState(str, Enum):
    CREATED = "created"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


class WorkspaceTransactionEvent(str, Enum):
    PREPARE = "prepare"
    COMMIT = "commit"
    ROLLBACK = "rollback"


_TRANSITIONS = {
    (
        WorkspaceTransactionState.CREATED,
        WorkspaceTransactionEvent.PREPARE,
    ): WorkspaceTransactionState.PREPARED,
    (
        WorkspaceTransactionState.PREPARED,
        WorkspaceTransactionEvent.COMMIT,
    ): WorkspaceTransactionState.COMMITTED,
    (
        WorkspaceTransactionState.PREPARED,
        WorkspaceTransactionEvent.ROLLBACK,
    ): WorkspaceTransactionState.ROLLED_BACK,
}


def _advance(
    state: WorkspaceTransactionState,
    event: WorkspaceTransactionEvent,
) -> WorkspaceTransactionState:
    try:
        return _TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise InvalidTransitionError(
            f"illegal workspace transaction transition: {state.value} + {event.value}"
        ) from exc


class WorkspaceTransactionMachine(
    TransitionEngine[WorkspaceTransactionState, WorkspaceTransactionEvent]
):
    def __init__(self, *, transaction_id: str) -> None:
        super().__init__(
            initial_state=WorkspaceTransactionState.CREATED,
            reducer=_advance,
            machine_type="workspace_transaction",
            machine_id=transaction_id,
        )


__all__ = [
    "WorkspaceTransactionEvent",
    "WorkspaceTransactionMachine",
    "WorkspaceTransactionState",
]
