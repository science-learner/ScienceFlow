"""MemoryContextManager responsibility: projection."""

from __future__ import annotations

import hashlib

from inquirycraft.memory import Message

from scienceflow.research.state.knowledge.context.context.results.base import (
    FileSnapshotInfo,
)
from scienceflow.research.state.knowledge.context.context.projection.bash_projection import _tool_call_signature
from scienceflow.research.state.knowledge.context.context.compression.clone_compression import (
    _clear_memory_and_long_term_log,
)
from scienceflow.research.state.knowledge.context.context.compression.clone_inheritance import _write_tool_feedback_is_success
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _all_messages,
    _apply_pinned_budget,
    _best_suffix_for_budget_priority,
    _finalize_messages_for_llm,
    _message_char_len,
    _project_messages_for_llm_view,
    _split_leading_user_prefix,
)
from scienceflow.research.state.knowledge.context.context.results.tool_results import _tool_args_dict_for_tool_message
from scienceflow.research.state.knowledge.context.source_snapshot import _tool_output_raw_id_from_content


def _update_snapshot_from_path(self, rel_path: str) -> None:
    try:
        p = self._guard.resolve(rel_path)
    except ValueError:
        return
    if not p.is_file():
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    raw = text.encode("utf-8")
    short = hashlib.sha256(raw).hexdigest()[:16]
    self._file_snapshots[rel_path] = FileSnapshotInfo(
        lines=len(text.splitlines()),
        sha256_short=short,
    )
    self._trim_snapshots()


@property
def file_state_summary(self) -> str:
    if not self._file_snapshots:
        return ""
    lines = ["[Workspace file state]"]
    for path, info in sorted(self._file_snapshots.items()):
        lines.append(
            f"  {path}: {info.lines} lines, hash~{info.sha256_short}",
        )
    return "\n".join(lines)


def pin_message(self, msg: Message) -> None:
    """Append a pinned user message (L1). Pinned messages are always sent in full
    before the sliding-window slice of chat history."""
    self._pinned_messages.append(msg)


@property
def pinned_messages(self) -> list[Message]:
    """Copy of pinned messages for auditing."""
    return list(self._pinned_messages)


def build_messages_for_llm_with_stats(self) -> tuple[list[Message], int]:
    """Return chat messages for the next LLM call and how many raw messages were dropped.

    *omitted* counts messages from the in-memory conversation that did not fit the sliding
    window (before the synthetic ``[Context budget: … omitted]`` user message is added).
    """
    raw_messages = _all_messages(self._memory)
    messages = (
        _project_messages_for_llm_view(raw_messages)
        if self._tool_memory_compression
        else list(raw_messages)
    )
    stable_leading_prefix, messages = _split_leading_user_prefix(messages)
    pinned_base = _apply_pinned_budget(
        list(self._pinned_messages),
        budget_chars=self._budget_chars,
        pinned_budget_ratio=self._pinned_budget_ratio,
    )
    pinned = [*pinned_base, *stable_leading_prefix]

    if not messages and not pinned:
        return [], 0

    budget = self._budget_chars
    if budget <= 0:
        out = pinned + messages
        return _finalize_messages_for_llm(out), 0

    leading, tail_messages = _split_leading_user_prefix(messages)
    pinned_len = sum(_message_char_len(m) for m in pinned)
    leading_len = sum(_message_char_len(m) for m in leading)
    budget_rest = max(0, budget - pinned_len - leading_len)

    if not tail_messages:
        return _finalize_messages_for_llm([*pinned, *leading]), 0

    if self._sliding_window_priority_enabled:
        chosen, new_k = _best_suffix_for_budget_priority(
            tail_messages, budget_rest, min_k=self._last_window_k,
        )
        self._last_window_k = new_k
    else:
        total = 0
        chosen_rev: list[Message] = []
        fallback_k = len(tail_messages)
        for i, m in enumerate(reversed(tail_messages)):
            lc = _message_char_len(m)
            if chosen_rev and total + lc > budget_rest:
                break
            chosen_rev.append(m)
            total += lc
            fallback_k = len(tail_messages) - 1 - i
        chosen = list(reversed(chosen_rev))
        self._last_window_k = max(
            self._last_window_k,
            fallback_k if chosen else len(tail_messages),
        )

    selected_count_before_format_cleanup = len(chosen)
    chosen = _finalize_messages_for_llm(chosen)

    omitted = len(tail_messages) - selected_count_before_format_cleanup
    # OpenAI-style chat payloads cannot start a visible suffix with a tool
    # result. Dropping those leading tool messages is format cleanup, not a
    # context-budget omission; otherwise a successful compact can be
    # misreported as still omitting history.
    while chosen and chosen[0].role == "tool":
        chosen.pop(0)

    result: list[Message] = [*pinned, *leading]
    if omitted > 0:
        # Fixed placeholder text (no variable count) keeps the prefix byte-identical
        # across consecutive calls, maximising LLM prefix-cache hit rates.
        result.append(
            Message.user_message(
                "[Context budget: earlier messages omitted from this request. "
                "Use tools to re-read files if needed.]",
            ),
        )
    result.extend(chosen)
    return _finalize_messages_for_llm(result), omitted


def build_messages_for_llm(self) -> list[Message]:
    """Return chat messages for the next LLM call: pinned (L1) + conversation (L2/L3).

    File snapshot metadata is no longer a synthetic leading user message; it is merged
    into the system prompt by the ScienceFlow prompt projection.
    """
    return self.build_messages_for_llm_with_stats()[0]


def rewrite_messages(self, messages: list[Message]) -> None:
    """Replace chat memory and truncate the append-only long-term log."""
    _clear_memory_and_long_term_log(self._memory)
    for message in messages:
        self._memory.add_message(message)
    self._last_window_k = 0


def replay_state_from_inherited_memory(self) -> dict[str, int]:
    """Rebuild session-local guard state from already-loaded inherited memory.

    Clone-continue starts a fresh :class:`MemoryContextManager`, so in-session state such as
    read coverage and repeated read/bash signatures would otherwise be empty even though the
    inherited transcript contains the same tool calls. Replay only deterministic metadata:
    no messages are added or rewritten here.
    """
    messages = _all_messages(self._memory)
    if not messages:
        return {
            "read_coverage": 0,
            "file_snapshots": 0,
            "tool_signatures": 0,
        }

    self._read_coverage.clear()
    self._read_symbol_coverage.clear()
    self._file_snapshots.clear()
    self._seen_tool_signatures.clear()
    self._released_full_signatures.clear()

    read_coverage = 0
    file_snapshots = 0
    tool_signatures = 0
    for i, m in enumerate(messages):
        role = getattr(m, "role", "") or ""
        if role != "tool":
            continue
        tcid = str(getattr(m, "tool_call_id", "") or "")
        if tcid.startswith("inherited_fullrun_"):
            continue
        name = str(getattr(m, "name", "") or "")
        args_d = _tool_args_dict_for_tool_message(messages, i)
        if not isinstance(args_d, dict):
            continue

        sig = _tool_call_signature(name, args_d)
        if sig is not None:
            self._seen_tool_signatures[sig] = i
            tool_signatures += 1

        rel = str(args_d.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel:
            continue

        content = m.content if isinstance(m.content, str) else str(m.content or "")
        if name == "read":
            before = dict(self._read_coverage)
            raw_id = _tool_output_raw_id_from_content(content)
            self._update_read_coverage_and_is_redundant(
                rel,
                args_d,
                raw_id=raw_id,
            )
            if self._read_coverage != before or rel in self._read_symbol_coverage:
                read_coverage += 1
            continue

        if name in ("write", "edit"):
            ok = (
                name == "write" and _write_tool_feedback_is_success(content)
            ) or (
                name == "edit"
                and content.startswith("Edited ")
                and "1 replacement OK." in content.split("\n", 1)[0]
            )
            if ok:
                self._read_coverage.pop(rel, None)
                try:
                    resolved = self._guard.resolve(rel)
                    if resolved.is_file():
                        self._prune_symbol_read_coverage_after_write(
                            rel,
                            resolved.read_text(encoding="utf-8", errors="replace"),
                        )
                except (ValueError, OSError):
                    self._read_symbol_coverage.pop(rel, None)
                self._update_snapshot_from_path(rel)
                if rel in self._file_snapshots:
                    file_snapshots += 1

    return {
        "read_coverage": read_coverage,
        "file_snapshots": file_snapshots,
        "tool_signatures": tool_signatures,
    }
