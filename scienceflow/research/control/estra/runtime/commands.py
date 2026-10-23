# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Pure mapping from EStra decisions to Runtime command proposals."""

from __future__ import annotations

from scienceflow.foundation.contracts import EstraDecision
from scienceflow.research.control.estra.planning.contracts import (
    EstraCommandKind,
    EstraRuntimeCommand,
)


def map_estra_runtime_command(
    decision: EstraDecision,
    *,
    command_id: str,
    expected_stage: str,
    evidence_refs: tuple[str, ...] = (),
) -> EstraRuntimeCommand:
    kind = (
        EstraCommandKind.RESTORE_STAGE
        if decision.action == "switch_stage"
        else EstraCommandKind.COMPACT_CURRENT
    )
    reason_code = {
        "switch_stage": "estra_switch_stage",
        "keep_but_redirect": "estra_redirect_current",
        "keep_current": "estra_keep_current",
    }.get(decision.action, "estra_unknown_action")
    return EstraRuntimeCommand(
        command_id=str(command_id or ""),
        kind=kind,
        target_stage=decision.target_stage,
        reason_code=reason_code,
        evidence_refs=tuple(evidence_refs),
        expected_stage=str(expected_stage or ""),
        parameters={
            "action": decision.action,
            "startpoint": decision.startpoint,
            "intent": decision.intent,
            "compact": decision.compact,
            "reason": decision.reason,
            "diagnostics": dict(decision.diagnostics),
        },
    )


__all__ = ["map_estra_runtime_command"]
