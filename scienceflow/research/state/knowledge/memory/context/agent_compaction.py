# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""ScienceFlow memory compaction entry points."""

from __future__ import annotations

from inquirycraft.memory import Message


async def compact(self, *, mid_run: bool = False) -> str:
    """Run LLM summarization and reset chat history (see REPL /compact).

    *mid_run* uses a shorter, debugging-oriented summary prompt (in-session continuation).
    """

    async def complete(prompt: str, system_prompt: str) -> str:
        return await self.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="compact",
            base_messages=[],
            system_messages=[Message.system_message(system_prompt)],
        )

    out = await self._memory_ctx.compact(
        mid_run=mid_run,
        completion=complete,
    )
    if (out or "").strip() == "Nothing to compact.":
        return out
    self._sync_last_run_token_totals()
    # compact() replaces chat history with a short summary; reset the monotone
    # window-k so the next build_messages_for_llm_with_stats re-evaluates from 0.
    try:
        self._memory_ctx._last_window_k = 0
    except AttributeError:
        pass
    return out


async def compact_inband(self, *, mid_run: bool = True) -> str:
    """Compact REPL history through the normal system prompt, not a summarizer prompt."""
    compact_messages, total_messages = self._memory_ctx.build_inband_compact_messages(
        mid_run=mid_run,
    )
    if total_messages <= 0 or not compact_messages:
        return "Nothing to compact."
    from scienceflow.research.control.ephemeral_agent_session import (
        run_ephemeral_science_agent,
    )

    assistant_msg = await run_ephemeral_science_agent(
        self,
        str(compact_messages[-1].content or ""),
        trigger="compact_inband",
        base_messages=compact_messages[:-1],
        system_messages=[Message.system_message(self._build_system_prompt())],
    )
    self._sync_last_run_token_totals()

    tool_calls = getattr(assistant_msg, "tool_calls", None) or []
    if tool_calls:
        return "Compact failed: summary turn emitted tool call(s)."
    summary = (getattr(assistant_msg, "content", None) or "").strip()
    if not summary:
        summary = (getattr(assistant_msg, "reasoning_content", None) or "").strip()
    recent_messages = (
        0
        if bool(
            getattr(self, "_lnr_compact_on_context_threshold", False),
        )
        else 8
    )
    return self._memory_ctx.replace_history_with_compacted_summary(
        summary,
        recent_messages=recent_messages,
    )
