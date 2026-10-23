# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Pure EStra command service."""

from __future__ import annotations

from typing import Any

from scienceflow.foundation.contracts import EstraContext, EstraDecision
from scienceflow.research.control.estra.planning.parser import parse_estra_decision
from scienceflow.research.control.estra.planning.policy import normalize_decision


class EstraService:
    def parse(self, text: str, *, context: EstraContext) -> dict[str, Any]:
        return parse_estra_decision(
            text, switch_candidates=list(context.switch_candidates)
        )

    def normalize(
        self, parsed: dict[str, Any], *, context: EstraContext
    ) -> EstraDecision | None:
        return normalize_decision(parsed, context=context)

    def decide_from_text(
        self, text: str, *, context: EstraContext
    ) -> EstraDecision | None:
        return self.normalize(self.parse(text, context=context), context=context)


__all__ = ["EstraService"]
