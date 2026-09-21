# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""LLM-agnostic EStra planner with safe invalid/error fallback."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

from scienceflow.foundation.contracts import EstraDecision
from scienceflow.research.control.estra.planning.contracts import (
    EstraPlanAttempt,
    EstraPlanRequest,
    EstraPlanResult,
)
from scienceflow.research.control.estra.runtime.service import EstraService


EstraModelPort = Callable[[EstraPlanRequest], str | Awaitable[str]]


class EstraPlanner:
    """Own model-output normalization and safe fallback semantics."""

    def __init__(self, service: EstraService | None = None) -> None:
        self.service = service or EstraService()

    async def plan(
        self,
        request: EstraPlanRequest,
        primary: EstraModelPort,
        *,
        primary_mode: str = "isolated",
        fallback: EstraModelPort | None = None,
        fallback_mode: str = "isolated_fallback",
    ) -> EstraPlanResult:
        attempts: list[EstraPlanAttempt] = []
        raw = ""
        decision: EstraDecision | None = None
        mode = primary_mode
        try:
            raw = str(await _call(primary, request) or "")
            decision = self.service.decide_from_text(raw, context=request.context)
            attempts.append(
                EstraPlanAttempt(
                    mode=primary_mode,
                    outcome="valid" if decision is not None else "invalid",
                    raw=raw,
                )
            )
        except Exception as exc:
            attempts.append(
                EstraPlanAttempt(
                    mode=primary_mode,
                    outcome="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        if decision is None and attempts[-1].outcome == "invalid" and fallback:
            try:
                raw = str(await _call(fallback, request) or "")
                decision = self.service.decide_from_text(raw, context=request.context)
                attempts.append(
                    EstraPlanAttempt(
                        mode=fallback_mode,
                        outcome="valid" if decision is not None else "invalid",
                        raw=raw,
                    )
                )
                if decision is not None:
                    mode = fallback_mode
            except Exception as exc:
                attempts.append(
                    EstraPlanAttempt(
                        mode=fallback_mode,
                        outcome="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        if decision is None:
            error = next(
                (attempt.error for attempt in reversed(attempts) if attempt.error),
                "",
            )
            decision = EstraDecision(
                action="keep_current",
                startpoint="current_workspace",
                intent="continue",
                target_stage=request.context.latest_stage,
                compact=True,
                reason=f"estra llm error: {error}" if error else "invalid estra response",
            )
            mode = "safe_fallback"
        input_hash = _hash(
            {
                "request_id": request.request_id,
                "context": {
                    "latest_stage": request.context.latest_stage,
                    "switch_candidates": request.context.switch_candidates,
                    "trigger_source": request.context.trigger_source,
                    "metadata": request.context.metadata,
                },
                "prompt": request.prompt,
                "metadata": request.metadata,
            }
        )
        output_hash = _hash(decision.to_dict())
        return EstraPlanResult(
            request_id=request.request_id,
            decision=decision,
            decision_mode=mode,
            attempts=tuple(attempts),
            input_hash=input_hash,
            output_hash=output_hash,
            used_fallback=len(attempts) > 1,
        )


async def _call(port: EstraModelPort, request: EstraPlanRequest) -> Any:
    value = port(request)
    return await value if inspect.isawaitable(value) else value


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["EstraModelPort", "EstraPlanner"]
