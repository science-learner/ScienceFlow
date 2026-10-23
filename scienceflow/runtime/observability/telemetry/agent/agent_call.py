"""ScienceFlow telemetry adapter for InquiryCraft provider observations."""

from __future__ import annotations

import logging
from typing import Any

from inquirycraft.runtime import build_llm_call_record

logger = logging.getLogger("scienceflow")


def record_llm_call(
    self,
    operation: str,
    duration_sec: float,
    round_idx: int | None,
    status: str = "ok",
    *,
    recovery: bool = False,
    turn_kind: str | None = None,
    first_tool_name: str | None = None,
    first_bash_kind: str | None = None,
    llm_override: Any | None = None,
    llm_role: str | None = None,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    tokens_cached: int | None = None,
    ttft_sec: float | None = None,
    tpot_ms: float | None = None,
) -> None:
    """Invoke ``on_llm_call`` with one row of per-call stats (must not raise)."""
    if self._on_llm_call is None:
        return
    self._call_seq += 1
    source_llm = llm_override or self.llm
    payload = build_llm_call_record(
        source_llm,
        operation=operation,
        duration_sec=duration_sec,
        call_seq=self._call_seq,
        round_idx=round_idx,
        status=status,
        recovery=recovery,
        turn_kind=turn_kind,
        first_tool_name=first_tool_name,
        first_bash_kind=first_bash_kind,
        llm_role=llm_role,
    )
    overrides = {
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_cached": tokens_cached,
        "ttft_sec": ttft_sec,
        "tpot_ms": tpot_ms,
    }
    payload.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    try:
        self._on_llm_call(payload)
    except Exception:
        logger.debug("on_llm_call hook failed", exc_info=True)


__all__ = ["record_llm_call"]
