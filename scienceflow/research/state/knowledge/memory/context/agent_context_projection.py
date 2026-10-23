"""Context preparation and compaction orchestration for one runtime round."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from inquirycraft.memory import Message

from scienceflow.agent.core.ports.callback_ports import resolve_agent_callback
from scienceflow.research.state.knowledge.memory.context.agent_context_rules import (
    _should_append_run_request,
)
from scienceflow.research.control.agent_session_state import RunSessionState


logger = logging.getLogger("scienceflow")


@dataclass(frozen=True, slots=True)
class ContextPreparationResult:
    messages: list[Any]
    early_result: str | None = None


@dataclass(slots=True)
class _CompactionState:
    omitted_before: int
    compact_on_threshold: bool
    auto_compact: bool
    mode: str
    recent_keep: int
    tokens_before: dict[str, int]
    output: str = ""
    omitted_after: int = 0
    summary_chars: int | None = None


class _CompactionEvents:
    def __init__(self, agent: Any, round_number: int) -> None:
        self._callback = resolve_agent_callback(agent, "context_compact_observer")
        self._round_number = round_number

    def emit(self, phase: str, **payload: Any) -> bool:
        if not callable(self._callback):
            return False
        try:
            self._callback(phase=phase, round=self._round_number, **payload)
            return True
        except Exception:
            logger.debug("[context-compact] event callback failed", exc_info=True)
            return False


def _compact_tokens(agent: Any) -> dict[str, int]:
    return {
        "tokens_in": int(getattr(agent, "_run_compact_tokens_in", 0) or 0),
        "tokens_out": int(getattr(agent, "_run_compact_tokens_out", 0) or 0),
        "tokens_cached": int(getattr(agent, "_run_compact_tokens_cached", 0) or 0),
    }


def _compact_status(output: str) -> str:
    if output.startswith("Compact failed"):
        return "failed"
    return "skipped" if output == "Nothing to compact." else "ok"


def _restore_request_after_compaction(
    agent: Any,
    request: str | None,
    output: str,
) -> None:
    if not output.startswith("Compacted "):
        return
    agent._mid_run_compacted = True
    if not request:
        return
    try:
        stored = agent.memory.chat_history_memory.retrieve(window_size=None)
        has_current_request = any(
            record.memory_record.message.role == "user"
            and record.memory_record.message.content == request
            for record in stored
        )
    except Exception:
        has_current_request = False
    if not has_current_request and _should_append_run_request(agent.memory, request):
        agent.memory.add_message(Message.user_message(str(request)))


async def _try_context_limit_estra(
    agent: Any,
    session: RunSessionState,
    state: _CompactionState,
    events: _CompactionEvents,
) -> str | None:
    callback = resolve_agent_callback(agent, "context_limit_estra")
    if not state.auto_compact or not callable(callback):
        return None
    events.emit(
        "estra_check",
        mode=state.mode,
        reason="context_limit",
        omitted_before=state.omitted_before,
        recent_keep=state.recent_keep,
    )
    try:
        result = callback(
            agent=agent,
            round_idx=session.round_index,
            max_steps=session.effective_max_steps,
            omitted=state.omitted_before,
        )
        if asyncio.iscoroutine(result):
            result = await result
    except Exception:
        logger.debug(
            "[context-compact] context-limit estra callback failed",
            exc_info=True,
        )
        result = None
    if not result:
        return None
    events.emit(
        "deferred",
        mode=state.mode,
        reason="estra_restore",
        omitted_before=state.omitted_before,
        result=str(result)[:240],
    )
    agent._sync_last_run_token_totals()
    return str(result)


def _emit_agent_finished(
    agent: Any,
    events: _CompactionEvents,
    state: _CompactionState,
    *,
    tokens_before: dict[str, int],
    mode: str | None = None,
) -> None:
    tokens_after = _compact_tokens(agent)
    events.emit(
        "finished",
        mode=mode or state.mode,
        status=_compact_status(state.output),
        omitted_before=state.omitted_before,
        omitted_after=state.omitted_after,
        summary_chars=state.summary_chars,
        tokens_in=tokens_after["tokens_in"] - tokens_before["tokens_in"],
        tokens_out=tokens_after["tokens_out"] - tokens_before["tokens_out"],
        tokens_cached=(tokens_after["tokens_cached"] - tokens_before["tokens_cached"]),
        result=state.output[:240],
    )


async def _run_primary_compaction(
    agent: Any,
    request: str | None,
    state: _CompactionState,
    events: _CompactionEvents,
) -> list[Any]:
    emitted_start = events.emit(
        "started",
        mode=state.mode,
        reason="context_limit" if state.auto_compact else "mid_run",
        omitted_before=state.omitted_before,
        recent_keep=state.recent_keep,
    )
    if not emitted_start:
        agent._log_info(
            "[context-compact] context threshold exceeded; %d message(s) would "
            "be omitted. running %s compact before the next LLM request",
            state.omitted_before,
            state.mode,
        )
    result = (
        await agent.compact_inband(mid_run=True)
        if state.auto_compact
        else await agent.compact(mid_run=True)
    )
    state.output = (result or "").strip()
    if state.output.startswith("Compact failed") and state.compact_on_threshold:
        events.emit(
            "failed",
            mode="inband",
            status="failed",
            omitted_before=state.omitted_before,
            reason_detail=state.output[:240],
            fallback="durable",
        )
        agent._log_warning(
            "[context-compact] in-band compact failed in compact-only mode; "
            "falling back to durable compact: %s",
            state.output[:200],
        )
        events.emit(
            "started",
            mode="durable",
            reason="fallback",
            omitted_before=state.omitted_before,
            recent_keep=8,
        )
        result = await agent.compact(mid_run=True)
        state.output = (result or "").strip()
        state.mode = "durable"
    if state.output.startswith("Compact failed"):
        agent._log_warning(
            "[context-compact] compact did not replace history: %s",
            state.output[:200],
        )
    _restore_request_after_compaction(agent, request, state.output)
    messages, state.omitted_after = (
        agent._memory_ctx.build_messages_for_llm_with_stats()
    )
    match = re.search(r"\((\d+)\s+chars\)", state.output)
    if match:
        try:
            state.summary_chars = int(match.group(1))
        except ValueError:
            state.summary_chars = None
    _emit_agent_finished(agent, events, state, tokens_before=state.tokens_before)
    return messages


async def _run_post_compact_durable_fallback(
    agent: Any,
    state: _CompactionState,
    events: _CompactionEvents,
) -> list[Any] | None:
    if not (
        state.compact_on_threshold
        and state.omitted_after > 0
        and state.mode != "durable"
    ):
        return None
    agent._log_warning(
        "[context-compact] %s compact still omitted %d message(s); "
        "retrying durable compact",
        state.mode,
        state.omitted_after,
    )
    events.emit(
        "started",
        mode="durable",
        reason="post_compact_omitted_fallback",
        omitted_before=state.omitted_after,
        recent_keep=0,
    )
    fallback_tokens_before = _compact_tokens(agent)
    result = await agent.compact(mid_run=True)
    state.output = (result or "").strip()
    state.mode = "durable"
    messages, state.omitted_after = (
        agent._memory_ctx.build_messages_for_llm_with_stats()
    )
    _emit_agent_finished(
        agent,
        events,
        state,
        tokens_before=fallback_tokens_before,
        mode="durable",
    )
    return messages


async def prepare_context_for_round(
    agent: Any,
    session: RunSessionState,
    request: str | None,
) -> ContextPreparationResult:
    messages, omitted = agent._memory_ctx.build_messages_for_llm_with_stats()
    compact_on_threshold = bool(
        getattr(agent, "_lnr_compact_on_context_threshold", False)
    )
    auto_compact = omitted > 0 and (
        compact_on_threshold
        or not bool(getattr(agent, "_lnr_superloop_enabled", False))
    )
    legacy_compact = (
        agent._mid_run_compact_enabled and not agent._mid_run_compacted and omitted > 0
    )
    if not (auto_compact or legacy_compact):
        return ContextPreparationResult(messages=messages)
    state = _CompactionState(
        omitted_before=omitted,
        compact_on_threshold=compact_on_threshold,
        auto_compact=auto_compact,
        mode="inband" if auto_compact else "durable",
        recent_keep=0 if compact_on_threshold else 8,
        tokens_before=_compact_tokens(agent),
    )
    events = _CompactionEvents(agent, session.round_index + 1)
    early_result = await _try_context_limit_estra(agent, session, state, events)
    if early_result is not None:
        return ContextPreparationResult(messages=messages, early_result=early_result)
    messages = await _run_primary_compaction(agent, request, state, events)
    fallback_messages = await _run_post_compact_durable_fallback(agent, state, events)
    if fallback_messages is not None:
        messages = fallback_messages
    if state.compact_on_threshold and state.omitted_after > 0:
        raise RuntimeError(
            "lnr compact did not fit context; refusing to send an "
            "omitted-history main-agent request"
        )
    return ContextPreparationResult(messages=messages)


__all__ = ("ContextPreparationResult", "prepare_context_for_round")
