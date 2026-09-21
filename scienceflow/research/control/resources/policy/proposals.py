# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Read-only resource proposals separated from mechanism effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from scienceflow.foundation.contracts import AdmissionDecision


@dataclass(frozen=True, slots=True)
class ResourceActionProposal:
    proposal_id: str
    action: str
    command_id: str
    reason_code: str
    facts: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


class AdmissionProposalProjector:
    """Expose admission output as a proposal; never applies an effect."""

    def project(self, decision: AdmissionDecision, *, proposal_id: str, command_id: str) -> ResourceActionProposal:
        return ResourceActionProposal(
            proposal_id=str(proposal_id or ""),
            action=str(decision.action or ""),
            command_id=str(command_id or ""),
            reason_code=str(decision.reason_code or ""),
            facts={
                "admitted": bool(decision.admitted),
                "retry_after_sec": float(decision.retry_after_sec or 0.0),
                "lease_id": str(decision.lease_id or ""),
                "metadata": dict(decision.metadata),
            },
        )


__all__ = ["AdmissionProposalProjector", "ResourceActionProposal"]
