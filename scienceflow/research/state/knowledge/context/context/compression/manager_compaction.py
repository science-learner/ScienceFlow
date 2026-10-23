"""MemoryContextManager responsibility: compaction."""

from __future__ import annotations

from inquirycraft.memory import Message

from scienceflow.research.state.knowledge.context.context.results.base import (
    COMPACTED_CONVERSATION_SUMMARY_MARKER,
    logger,
)
from scienceflow.research.state.knowledge.context.context.compression.clone_compression import _clear_memory_and_long_term_log
from scienceflow.research.state.knowledge.context.context.compression.compaction import (
    MSG0_BODY_TRUNCATED_PLACEHOLDER,
    _compress_tool_output_for_mechanical_compact,
    _format_message_excerpt_for_compact,
    _split_msg0_head_body,
    _split_recent_turn_suffix,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _all_messages,
    _message_char_len,
    _message_text_for_chars,
    _message_with_text,
)


def mechanical_compress_old_messages(self, *, keep_recent_turns: int = 3) -> dict[str, int]:
    """Deterministically compress older chat messages (no LLM call).

    Keeps the latest ``keep_recent_turns`` assistant turns untouched and compresses only
    older content, mainly successful tool outputs.
    """
    messages = _all_messages(self._memory)
    if not messages:
        return {"total": 0, "kept": 0, "changed": 0, "dropped": 0}

    old, recent = _split_recent_turn_suffix(
        messages,
        keep_recent_turns=max(0, int(keep_recent_turns)),
    )

    # Leading-user pin: indices of consecutive ``user`` messages at the head of ``old``.
    # Mirrors the teleport cross-fork compress (see
    # ``scienceflow/research/solver/lnr/_internal/teleport_inherit_compress.py``)
    # so that the task pin (policy + ensemble-fusion format + lite title) is never
    # truncated to 400 chars and thus the model never loses constraints like
    # "Do NOT ensemble / stack / blend".
    leading_pin: set[int] = set()
    for j, mm in enumerate(old):
        if getattr(mm, "role", None) == "user":
            leading_pin.add(j)
        else:
            break
    msg0_idx = 0 if (old and getattr(old[0], "role", None) == "user") else None

    changed = 0
    dropped = 0
    compressed_old: list[Message] = []
    for i, m in enumerate(old):
        role = getattr(m, "role", "") or ""
        text = _message_text_for_chars(m)
        if role == "tool":
            new_text = _compress_tool_output_for_mechanical_compact(text)
            if new_text != text:
                changed += 1
            compressed_old.append(_message_with_text(m, new_text))
            continue

        if role == "assistant":
            if len(text) > 500:
                changed += 1
                compressed_old.append(
                    _message_with_text(
                        m,
                        text[:500].rstrip() + "\n...[assistant message truncated for context]",
                    ),
                )
            else:
                compressed_old.append(m)
            continue

        if role == "user":
            # msg[0]: by default kept verbatim (whole user pin). When the
            # ``msg0_compress_body`` switch is on, the body after
            # ``## Task description`` is replaced with a placeholder while
            # the head (policy + ensemble fusion + lite title) stays intact.
            if i == msg0_idx:
                if self._msg0_compress_body:
                    head, body = _split_msg0_head_body(text)
                    if body:
                        new_text = (
                            head.rstrip("\n")
                            + "\n\n## Task description\n"
                            + MSG0_BODY_TRUNCATED_PLACEHOLDER
                        )
                        if new_text != text:
                            changed += 1
                            compressed_old.append(_message_with_text(m, new_text))
                            continue
                compressed_old.append(m)
                continue
            # Other leading user messages: preserve verbatim (cross-branch overview,
            # additional pinned turns), aligned with teleport cross-fork compress.
            if i in leading_pin:
                compressed_old.append(m)
                continue
            if (
                text.startswith("[Context budget:")
                or "## Time Budget" in text[:1200]
                or "## Trajectory branch history" in text[:1200]
                or "## Cross-branch overview" in text[:1200]
            ):
                dropped += 1
                changed += 1
                continue
            if len(text) > 400:
                changed += 1
                compressed_old.append(
                    _message_with_text(
                        m,
                        text[:400].rstrip() + "\n...[user message truncated for context]",
                    ),
                )
            else:
                compressed_old.append(m)
            continue

        compressed_old.append(m)

    merged = compressed_old + recent
    _clear_memory_and_long_term_log(self._memory)
    for m in merged:
        self._memory.add_message(m)

    return {
        "total": len(messages),
        "kept": len(merged),
        "changed": changed,
        "dropped": dropped,
    }


def _leading_user_messages_for_compact(self, messages: list[Message]) -> list[Message]:
    """Leading user/task messages to preserve verbatim across a compact rewrite."""
    pinned_leading: list[Message] = []
    for j, mm in enumerate(messages):
        if getattr(mm, "role", None) != "user":
            break
        text = _message_text_for_chars(mm)
        if text.startswith(COMPACTED_CONVERSATION_SUMMARY_MARKER):
            break
        if j == 0 and self._msg0_compress_body:
            head, body = _split_msg0_head_body(text)
            if body:
                pinned_leading.append(
                    _message_with_text(
                        mm,
                        head.rstrip("\n")
                        + "\n\n## Task description\n"
                        + MSG0_BODY_TRUNCATED_PLACEHOLDER,
                    ),
                )
                continue
        pinned_leading.append(mm)
    return pinned_leading


def set_protected_raw_prefix(
    self,
    end_index: int,
    *,
    warn_chars: int = 50_000,
    label: str = "protected raw prefix",
) -> dict[str, int | str]:
    """Preserve messages before *end_index* verbatim across future compact calls.

    Used by long-horizon REPL to keep early EDA evidence raw while compacting
    later exploration.  The first user/task pin is already protected separately;
    this marker protects the dynamic records immediately after that pin.
    If the fixed prefix exceeds ``warn_chars`` we warn but still keep it verbatim.
    """
    messages = _all_messages(self._memory)
    end = max(0, min(int(end_index), len(messages)))
    self._protected_raw_prefix_end_index = end
    self._protected_raw_prefix_warn_chars = max(0, int(warn_chars or 0))
    self._protected_raw_prefix_label = str(label or "protected raw prefix")
    pinned = self._leading_user_messages_for_compact(messages)
    protected, _rest, chars = self._protected_raw_prefix_parts(messages, len(pinned))
    return {
        "end_index": end,
        "message_count": len(protected),
        "chars": chars,
        "warn_chars": self._protected_raw_prefix_warn_chars,
        "label": self._protected_raw_prefix_label,
    }


def _protected_raw_prefix_parts(
    self,
    messages: list[Message],
    pinned_len: int,
) -> tuple[list[Message], list[Message], int]:
    end = max(0, int(getattr(self, "_protected_raw_prefix_end_index", 0) or 0))
    end = min(end, len(messages))
    start = max(0, min(int(pinned_len), len(messages)))
    if end <= start:
        return [], list(messages[start:]), 0
    protected = list(messages[start:end])
    rest = list(messages[end:])
    chars = 0
    for m in protected:
        role = str(getattr(m, "role", "") or "")
        chars += len(role) + 3 + _message_char_len(m)
    warn_chars = max(0, int(getattr(self, "_protected_raw_prefix_warn_chars", 0) or 0))
    if warn_chars and chars > warn_chars:
        label = str(getattr(self, "_protected_raw_prefix_label", "protected raw prefix") or "protected raw prefix")
        logger.warning(
            "%s is %d chars, exceeding warning threshold %d; keeping it verbatim as fixed prefix",
            label,
            chars,
            warn_chars,
        )
    return protected, rest, chars


def replace_protected_raw_prefix_with_summary(
    self,
    end_index: int,
    summary: str,
    *,
    warn_chars: int = 50_000,
    label: str = "protected summary prefix",
) -> dict[str, int | str]:
    """Replace a raw protected prefix with one compact protected summary card.

    Long-horizon REPL uses this after the first metric-backed stage: early EDA
    facts stay as a fixed prefix, but scratch Python, full heredocs, and probe
    tool-call arguments are removed from the stable LLM context. Raw audit logs
    remain on disk under workspace ``.logs`` and stage snapshots.
    """
    messages = _all_messages(self._memory)
    if not messages:
        return {
            "end_index": 0,
            "message_count": 0,
            "chars": 0,
            "warn_chars": max(0, int(warn_chars or 0)),
            "label": str(label or "protected summary prefix"),
            "original_message_count": 0,
            "original_chars": 0,
            "mode": "summary",
        }
    end = max(0, min(int(end_index), len(messages)))
    pinned = self._leading_user_messages_for_compact(messages)
    start = max(0, min(len(pinned), len(messages)))
    protected_original = list(messages[start:end]) if end > start else []
    rest = list(messages[end:])
    original_chars = 0
    for m in protected_original:
        role = str(getattr(m, "role", "") or "")
        original_chars += len(role) + 3 + _message_char_len(m)

    body = (summary or "").strip()
    if not body:
        body = "No compact EDA facts were extracted; use dataset files and workspace logs for details."
    card_label = str(label or "protected summary prefix")
    summary_msg = Message.assistant_message(f"[{card_label}]\n{body}")

    _clear_memory_and_long_term_log(self._memory)
    for pm in pinned:
        self._memory.add_message(pm)
    self._memory.add_message(summary_msg)
    for m in rest:
        self._memory.add_message(m)

    self._protected_raw_prefix_end_index = len(pinned) + 1
    self._protected_raw_prefix_warn_chars = max(0, int(warn_chars or 0))
    self._protected_raw_prefix_label = card_label
    chars = len("assistant") + 3 + _message_char_len(summary_msg)
    if self._protected_raw_prefix_warn_chars and chars > self._protected_raw_prefix_warn_chars:
        logger.warning(
            "%s is %d chars, exceeding warning threshold %d; keeping it verbatim as fixed prefix",
            card_label,
            chars,
            self._protected_raw_prefix_warn_chars,
        )
    return {
        "end_index": self._protected_raw_prefix_end_index,
        "message_count": 1,
        "chars": chars,
        "warn_chars": self._protected_raw_prefix_warn_chars,
        "label": card_label,
        "original_message_count": len(protected_original),
        "original_chars": original_chars,
        "mode": "summary",
    }


def build_inband_compact_messages(
    self,
    *,
    mid_run: bool = True,
    max_history_chars: int | None = None,
) -> tuple[list[Message], int]:
    """Build a same-session compact request using the normal agent system prompt.

    The returned messages are meant to be sent with the main system prompt and
    ``tool_choice=none``. This avoids the old separate summarizer system prompt
    while still letting the model produce a concise continuation state.
    """
    messages = _all_messages(self._memory)
    if not messages:
        return [], 0

    pinned_leading = self._leading_user_messages_for_compact(messages)
    protected_raw, dynamic_messages, protected_chars = self._protected_raw_prefix_parts(
        messages,
        len(pinned_leading),
    )
    history_parts: list[str] = []
    for m in dynamic_messages:
        role = getattr(m, "role", "") or ""
        text = _message_text_for_chars(m)
        if text.startswith("[Context budget:"):
            continue
        excerpt = _format_message_excerpt_for_compact(m)
        history_parts.append(f"[{role}]: {excerpt}")

    if max_history_chars is None:
        history_cap = max(8000, min(45000, int(self._budget_chars * 0.70)))
    else:
        history_cap = max(2000, int(max_history_chars))

    kept_rev: list[str] = []
    used = 0
    for part in reversed(history_parts):
        part_len = len(part) + 1
        if kept_rev and used + part_len > history_cap:
            break
        if not kept_rev and part_len > history_cap:
            part = part[-history_cap:]
            part_len = len(part) + 1
        kept_rev.append(part)
        used += part_len
    kept = list(reversed(kept_rev))
    omitted = len(history_parts) - len(kept)
    if omitted > 0:
        kept.insert(
            0,
            "[Earlier dynamic history omitted from this compact request: "
            f"{omitted} message(s). Leading user/task messages are preserved verbatim.]",
        )
    history_text = (
        "\n".join(kept).strip()
        or "(No dynamic history beyond the preserved leading user/task messages.)"
    )

    workspace_state: list[str] = []
    sol_path = self._workspace / "solution.py"
    if sol_path.is_file():
        try:
            sol_code = sol_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            sol_code = ""
        if sol_code.strip():
            if len(sol_code) > 6000:
                sol_code = sol_code[:6000] + "\n... [truncated]"
            workspace_state.append(
                "[CURRENT solution.py]\n```python\n" + sol_code + "\n```",
            )
    result_path = self._workspace / "result.md"
    if result_path.is_file():
        try:
            rt = result_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            rt = ""
        if rt.strip():
            workspace_state.append("[CURRENT result.md]\n" + rt[:3000])

    compact_kind = "same REPL session" if mid_run else "next continuation session"
    protected_note = ""
    if protected_raw:
        protected_note = (
            f"The runtime also preserves {len(protected_raw)} early EDA/tool message(s) "
            f"verbatim ({protected_chars} chars); do not repeat or paraphrase them. "
        )
    prompt = (
        "Internal context maintenance request: summarize the conversation state so the "
        f"{compact_kind} can continue without losing task intent. Do not call tools. "
        "Return only the compact state summary.\n\n"
        "The leading user/task messages immediately before this request are preserved "
        "verbatim by the runtime; do not repeat or paraphrase them. "
        f"{protected_note}"
        "Preserve concrete "
        "ML research state: EDA/data insights, current approach, validation/submission "
        "status, metrics, files changed, recent failures, leakage risks, and the next "
        "3-6 actions. "
        "Quote still-active explicit user prohibitions only if they are needed to avoid "
        "a mistake. Max ~700 words.\n\n"
        "[Dynamic conversation to compact]\n"
        f"{history_text}"
    )
    if workspace_state:
        prompt += "\n\n[Ground-truth workspace state]\n" + "\n\n".join(workspace_state)
    return [*pinned_leading, Message.user_message(prompt)], len(messages)


def replace_history_with_compacted_summary(
    self,
    summary: str,
    *,
    recent_messages: int = 8,
) -> str:
    """Rewrite chat history as stable leading prefix + compact summary + recent tail."""
    messages = _all_messages(self._memory)
    body = (summary or "").strip()
    if not messages:
        return "Nothing to compact."
    if not body:
        return "Compact failed: empty summary."

    pinned_leading = self._leading_user_messages_for_compact(messages)
    protected_raw, dynamic_messages, protected_chars = self._protected_raw_prefix_parts(
        messages,
        len(pinned_leading),
    )
    n_recent = max(0, int(recent_messages))
    recent = list(dynamic_messages[-n_recent:]) if n_recent else []
    recent = [
        m
        for m in recent
        if COMPACTED_CONVERSATION_SUMMARY_MARKER not in _message_text_for_chars(m)
        and not _message_text_for_chars(m).startswith("[Context budget:")
    ]
    while recent and getattr(recent[0], "role", None) == "tool":
        recent.pop(0)

    _clear_memory_and_long_term_log(self._memory)
    for pm in pinned_leading:
        self._memory.add_message(pm)
    for pm in protected_raw:
        self._memory.add_message(pm)
    self._memory.add_message(
        Message.user_message(
            f"{COMPACTED_CONVERSATION_SUMMARY_MARKER}\n{body}",
        ),
    )
    for m in recent:
        self._memory.add_message(m)
    if protected_raw:
        self._protected_raw_prefix_end_index = len(pinned_leading) + len(protected_raw)
    self._last_window_k = 0
    if protected_raw:
        return (
            f"Compacted {len(messages)} messages into in-band summary "
            f"({len(body)} chars); kept {len(pinned_leading)} leading + "
            f"{len(protected_raw)} protected raw + {len(recent)} recent."
        )
    return (
        f"Compacted {len(messages)} messages into in-band summary "
        f"({len(body)} chars); kept {len(pinned_leading)} leading + {len(recent)} recent."
    )
