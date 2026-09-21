# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from inquirycraft.memory import Memory, Message
from inquirycraft.memory import (
    dedupe_message_records,
    message_record_content,
    message_record_has_tool_calls,
    message_record_message,
    message_record_role,
    message_record_window_score,
    read_message_records,
    repair_message_record_window,
    write_message_records,
)
from inquirycraft.memory import BaseChatHistoryMemory, BaseContextCreator
from inquirycraft.memory import JsonKeyValueStorage


logger = logging.getLogger("scienceflow")


def create_agent_memory(memory_dir: Path, agent_name: str, max_messages: int) -> Memory:
    agent_dir = memory_dir / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    return Memory(
        max_messages=max_messages,
        chat_history_memory=BaseChatHistoryMemory(
            storage=JsonKeyValueStorage(
                path=str(agent_dir / "short_term.json"), mode="w"
            )
        ),
        context_creator=BaseContextCreator(),
        long_term_log=str(agent_dir / "long_term.jsonl"),
    )


def load_agent_memory(
    memory_dir: Path,
    agent_name: str,
    max_messages: int,
    *,
    recent_rounds: int = 0,
) -> Memory:
    """Load existing agent memory from disk (clone-continue mode).

    Parameters
    ----------
    recent_rounds:
        If > 0, keep only the last *recent_rounds* tool-call rounds
        (each round ≈ 1 assistant + 1 tool message). 0 = keep all.
    """
    agent_dir = memory_dir / agent_name
    if not agent_dir.is_dir():
        return create_agent_memory(memory_dir, agent_name, max_messages)
    _repair_agent_memory_short_term(agent_dir, max_messages=max_messages)
    memory = Memory(
        max_messages=max_messages,
        chat_history_memory=BaseChatHistoryMemory(
            storage=JsonKeyValueStorage(
                path=str(agent_dir / "short_term.json"), mode="a"
            )
        ),
        context_creator=BaseContextCreator(),
        long_term_log=str(agent_dir / "long_term.jsonl"),
    )
    if recent_rounds > 0:
        _prune_memory_to_recent_rounds(memory, recent_rounds)
    return memory


def select_prefix_safe_agent_memory_records(
    agent_dir: Path,
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    """Return the best prefix-safe snapshot from an agent memory directory.

    ``short_term.json`` is bounded by ``max_messages`` and can lose the leading
    task/user prefix. ``long_term.jsonl`` is usually a better recovery source,
    but it is not guaranteed to exist on every path. This helper prefers the
    source that still has a leading system/user prefix and then trims it while
    preserving that prefix.
    """
    agent_dir = Path(agent_dir)
    short_records = _read_memory_jsonl(agent_dir / "short_term.json")
    long_records = _read_memory_jsonl(agent_dir / "long_term.jsonl")
    short_repaired = _filter_agentic_route_memory_records(
        _repair_memory_record_window(short_records, max_messages=max_messages),
    )
    long_repaired = _filter_agentic_route_memory_records(
        _repair_memory_record_window(long_records, max_messages=max_messages),
    )
    source, records = max(
        [("short_term", short_repaired), ("long_term", long_repaired)],
        key=lambda item: _memory_record_score(item[1]),
    )
    if source == "long_term":
        logger.info(
            "[memory] using long_term.jsonl as prefix-safe recovery snapshot: %s records=%d",
            agent_dir,
            len(records),
        )
    return list(records)


def write_agent_memory_record_files(
    agent_dir: Path,
    records: list[dict[str, Any]],
    *,
    write_long_term: bool = False,
) -> None:
    """Persist raw chat-history records for an agent memory directory."""
    agent_dir = Path(agent_dir)
    _write_memory_jsonl(agent_dir / "short_term.json", records)
    if write_long_term:
        _write_memory_jsonl(agent_dir / "long_term.jsonl", records)


def _repair_agent_memory_short_term(agent_dir: Path, *, max_messages: int) -> None:
    short_path = Path(agent_dir) / "short_term.json"
    before = _read_memory_jsonl(short_path)
    after = select_prefix_safe_agent_memory_records(
        agent_dir, max_messages=max_messages
    )
    if before == after:
        return
    _write_memory_jsonl(short_path, after)
    logger.info(
        "[memory] repaired short_term prefix window: %s before=%d after=%d first_role=%s",
        short_path,
        len(before),
        len(after),
        _record_role(after[0]) if after else "",
    )


def _read_memory_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_message_records(path, logger=logger)


def _write_memory_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    write_message_records(path, records)


def _dedupe_memory_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return dedupe_message_records(records)


def _record_message_dict(record: dict[str, Any]) -> dict[str, Any]:
    return message_record_message(record)


def _record_role(record: dict[str, Any]) -> str:
    return message_record_role(record)


def _record_content(record: dict[str, Any]) -> str:
    return message_record_content(record)


def _record_has_tool_calls(record: dict[str, Any]) -> bool:
    return message_record_has_tool_calls(record)


def _is_agentic_route_prompt_record(record: dict[str, Any]) -> bool:
    if _record_role(record) != "user":
        return False
    text = _record_content(record)
    markers = (
        "You returned pure text without a tool call",
        "Make the second-stage route decision now",
        "missing a parseable `## Next Search Decision`",
        "previous loop-local route reply is missing parseable JSON",
        "says `exit_current_branch`",
    )
    return any(marker in text for marker in markers)


def _is_agentic_route_response_record(record: dict[str, Any]) -> bool:
    if _record_role(record) != "assistant":
        return False
    if _record_has_tool_calls(record):
        return False
    text = _record_content(record)
    if not text.strip():
        return False
    route_tokens = (
        '"action"',
        "'action'",
        "rewind_to_step",
        "rewind_to_node",
        "continue_current",
        "exit_current_branch",
        "new_branch",
        "## Next Search Decision",
    )
    return any(token in text for token in route_tokens)


def _filter_agentic_route_memory_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        return []
    out: list[dict[str, Any]] = []
    skip_route_response = False
    for rec in records:
        if _is_agentic_route_prompt_record(rec):
            skip_route_response = True
            continue
        if skip_route_response:
            if _is_agentic_route_response_record(rec):
                skip_route_response = False
                continue
            skip_route_response = False
        out.append(rec)
    return out


def _repair_memory_record_window(
    records: list[dict[str, Any]],
    *,
    max_messages: int,
) -> list[dict[str, Any]]:
    return repair_message_record_window(records, max_messages=max_messages)


def _memory_record_score(records: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    return message_record_window_score(records)


_WSP_ABS_PREFIX_RE = re.compile(r"/\S+/wsp/[0-9a-f]{32}/")


def _sanitize_wsp_paths_in_text(text: str) -> str:
    """Replace ``/.../wsp/<32hex>/`` with ``./`` (clone-continue path hygiene)."""
    if not text:
        return text
    return _WSP_ABS_PREFIX_RE.sub("./", text)


def rewrite_workspace_abs_to_relative(text: str, workspace_abs: str | None) -> str:
    """Replace runtime-absolute *workspace* prefix with ``./`` (teleport path hygiene).

    Used at tool-result emission to keep LLM-visible paths stable as the
    underlying node directory rotates. Combined with the historical
    ``_sanitize_wsp_paths_in_text`` (which still mops up ``wsp/<32hex>``
    leftovers in inherited memory).

    Empty / falsy ``workspace_abs`` is a no-op so the helper is safe to call
    unconditionally from teleport-aware tool execution paths.
    """
    if not text or not workspace_abs:
        return text
    prefix = str(workspace_abs).rstrip("/")
    if not prefix or prefix == "/" or prefix not in text:
        return text
    return text.replace(prefix + "/", "./").replace(prefix, ".")


def _sanitize_message_wsp_paths(m: Message) -> Message:
    """Rewrite absolute workspace paths inside one message (content + tool call JSON)."""
    from inquirycraft.memory import Function as ToolFunction

    updates: dict[str, Any] = {}
    if isinstance(m.content, str):
        c = _sanitize_wsp_paths_in_text(m.content)
        if c != m.content:
            updates["content"] = c
    if m.tool_calls:
        new_tcs = []
        tc_changed = False
        for tc in m.tool_calls:
            args = _sanitize_wsp_paths_in_text(tc.function.arguments)
            if args != tc.function.arguments:
                tc_changed = True
                new_tc = tc.model_copy(
                    update={
                        "function": ToolFunction(
                            name=tc.function.name,
                            arguments=args,
                        )
                    }
                )
                new_tcs.append(new_tc)
            else:
                new_tcs.append(tc)
        if tc_changed:
            updates["tool_calls"] = new_tcs

    if not updates:
        return m
    return m.model_copy(update=updates)


def sanitize_inherited_memory_workspace_paths(memory: Memory) -> None:
    """Replace absolute ``.../wsp/<uuid>/...`` paths with ``./...`` in all chat messages.

    Run after loading clone-inherited memory so the model does not copy parent node paths
    from stored history (assistant tool_calls and tool outputs).
    """
    messages = [
        record.memory_record.message
        for record in memory.chat_history_memory.retrieve(window_size=None)
    ]
    rewritten = [_sanitize_message_wsp_paths(message) for message in messages]
    if all(after is before for before, after in zip(messages, rewritten, strict=True)):
        return
    memory.chat_history_memory.storage.clear()
    for message in rewritten:
        memory.add_message(message)

def _prune_memory_to_recent_rounds(memory: Memory, rounds: int) -> None:
    """Keep only the last *rounds* tool-call rounds in memory.

    A "round" ≈ 1 assistant message (with tool_calls) + 1 tool message.
    Also preserves leading system/user messages (task instructions).
    """
    records = memory.chat_history_memory.retrieve(window_size=None)
    messages = [record.memory_record.message for record in records]
    prefix_end = 0
    for index, message in enumerate(messages):
        if message.role in {"system", "user"}:
            prefix_end = index + 1
        else:
            break
    suffix = messages[prefix_end:]
    start = max(0, len(suffix) - rounds * 2)
    while start > 0 and suffix[start].role == "tool":
        start -= 1
    kept = suffix[start:]
    while kept and kept[0].role == "tool":
        kept = kept[1:]
    memory.chat_history_memory.storage.clear()
    for message in messages[:prefix_end] + kept:
        memory.add_message(message)
