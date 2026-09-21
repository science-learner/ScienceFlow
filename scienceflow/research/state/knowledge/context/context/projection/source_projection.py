"""Memory-context responsibility: source projection."""

from __future__ import annotations

import json
import re
from typing import Any

from inquirycraft.memory import Function, Memory, Message, ToolCall
from inquirycraft.tools import looks_like_write_placeholder_mimicry

from scienceflow.research.state.knowledge.context.context.results.base import (
    COMPACTED_CONVERSATION_SUMMARY_MARKER,
    _PINNED_TRUNC_SUFFIX,
    logger,
)
from scienceflow.research.state.knowledge.context.source_snapshot import (
    _AUTO_SNAPSHOT_PREFIX,
    _CHANGED_RANGE_PREFIX,
    _CODE_MAP_PREFIX,
    _SYMBOL_SUMMARY_PREFIX,
)

def _should_omit_write_body_from_llm_projection(content: str) -> bool:
    return looks_like_write_placeholder_mimicry(content)

def _message_text_for_chars(message: Message) -> str:
    c = message.content
    if isinstance(c, str):
        return c
    return str(c or "")

def _message_with_text(message: Message, text: str) -> Message:
    return message.model_copy(update={"content": text})

def _apply_pinned_budget(
    pinned: list[Message],
    *,
    budget_chars: int,
    pinned_budget_ratio: float,
) -> list[Message]:
    """Trim pinned list so total char length stays within ``budget_chars * pinned_budget_ratio``.

    The first pinned message is never truncated (task + data preview); overflow is taken from
    subsequent pinned messages, each truncated in order with :data:`_PINNED_TRUNC_SUFFIX`.

    When ``pinned_budget_ratio`` is ``<= 0``, pinned content is not subject to this cap.
    """
    if pinned_budget_ratio <= 0 or not pinned:
        return pinned
    cap = max(0, int(budget_chars * float(pinned_budget_ratio)))
    total = sum(_message_char_len(m) for m in pinned)
    if total <= cap:
        return pinned

    first = pinned[0]
    first_len = _message_char_len(first)
    out: list[Message] = [first]
    rest = pinned[1:]
    if not rest:
        return out

    remainder_cap = max(0, cap - first_len)
    suffix = "\n\n" + _PINNED_TRUNC_SUFFIX
    for m in rest:
        lc = _message_char_len(m)
        if lc <= remainder_cap:
            out.append(m)
            remainder_cap -= lc
            continue
        text = _message_text_for_chars(m)
        if remainder_cap <= len(suffix) + 1:
            out.append(_message_with_text(m, _PINNED_TRUNC_SUFFIX))
            remainder_cap = 0
            continue
        max_body = remainder_cap - len(suffix)
        new_text = text[:max_body].rstrip() + suffix
        out.append(_message_with_text(m, new_text))
        remainder_cap = 0
    return out

def _split_leading_user_prefix(
    messages: list[Message],
    *,
    max_count: int = 4,
) -> tuple[list[Message], list[Message]]:
    """Return leading user turns as a stable prefix and the remaining dynamic tail.

    LNR's cache-first path relies on keeping early task/user contract bytes fixed.
    REPL uses the same shape: task_description may be pinned separately, while the
    first auto user query lives in chat history. Treating a small number of
    consecutive leading user messages as an L1.5 prefix preserves the visible order
    without changing prompt text, and prevents sliding-window suffix selection from
    dropping or moving them. The cap avoids pathological user-only histories turning
    the whole conversation into pinned context.
    """
    if not messages or max_count <= 0:
        return [], messages
    prefix: list[Message] = []
    limit = max(0, int(max_count))
    for m in messages:
        if len(prefix) >= limit:
            break
        if getattr(m, "role", None) != "user":
            break
        prefix.append(m)
        text = _message_text_for_chars(m)
        if text.startswith(COMPACTED_CONVERSATION_SUMMARY_MARKER):
            break
    if not prefix:
        return [], messages
    return prefix, messages[len(prefix):]

def _message_priority_score(message: Message) -> int:
    """Higher = more important to keep when trimming the sliding window (errors/EDA > write/edit OK)."""
    role = getattr(message, "role", "") or ""
    content = message.content if isinstance(message.content, str) else str(message.content or "")
    if role == "tool":
        tcid = str(getattr(message, "tool_call_id", "") or "")
        if tcid.startswith("inherited_fullrun_"):
            return 88
        first = content.splitlines()[0] if content else ""
        lower = content.lower()
        if "traceback" in lower or "error:" in lower[:2000]:
            return 95
        if first.startswith("[exit="):
            head = first[:120]
            if "exit=0" in head or " exit=0," in head:
                # Successful bash — lower priority than failures / reads
                return 22
            return 92
        if first.startswith("[") and ("lines total" in first or "showing" in first):
            return 72
        if first.startswith("Edited ") and "1 replacement OK." in first:
            return 28
        if "wrote " in lower and "bytes" in lower[:200]:
            if "solution.py" in lower[:300]:
                return 75
            return 28
        return 48
    return 50

def _message_char_len(message: Message) -> int:
    """Approximate serialized size for sliding-window budget (content + tool_calls JSON)."""
    n = 0
    c = message.content
    if isinstance(c, str):
        n += len(c)
    elif c is not None:
        n += len(str(c))
    tcs = getattr(message, "tool_calls", None)
    if tcs:
        try:
            n += len(json.dumps(tcs, ensure_ascii=False))
        except (TypeError, ValueError):
            n += len(str(tcs))
    return n

def _tool_call_ids_from_assistant_message(message: Message) -> list[str]:
    """Extract tool call ids from an assistant message (dict or model objects)."""
    tcs = getattr(message, "tool_calls", None) or []
    out: list[str] = []
    for tc in tcs:
        if isinstance(tc, dict):
            out.append(str(tc.get("id") or ""))
        else:
            out.append(str(getattr(tc, "id", "") or ""))
    return out

def _ensure_complete_tool_turn_prefix(messages: list[Message]) -> list[Message]:
    """Drop leading messages until the list starts with a valid chat prefix.

    Sliding-window truncation can leave an ``assistant`` message with ``tool_calls`` but
    fewer following ``role=tool`` responses than required; OpenAI rejects such requests.
    Also drops orphaned leading ``tool`` messages (handled here for robustness).
    """
    msgs = list(messages)
    while msgs:
        if msgs[0].role == "tool":
            msgs.pop(0)
            continue
        m0 = msgs[0]
        tcs = getattr(m0, "tool_calls", None) or []
        if m0.role != "assistant" or not tcs:
            break
        n = len(tcs)
        ids = _tool_call_ids_from_assistant_message(m0)
        if len(msgs) < 1 + n:
            msgs.pop(0)
            continue
        valid = True
        for j in range(n):
            tm = msgs[1 + j]
            if tm.role != "tool":
                valid = False
                break
            tcid = getattr(tm, "tool_call_id", None) or ""
            if ids[j] and tcid and tcid != ids[j]:
                valid = False
                break
        if valid:
            break
        msgs.pop(0)
    return msgs

def _assistant_without_tool_calls_for_broken_turn(message: Message) -> Message:
    content = message.content if isinstance(message.content, str) else str(message.content or "")
    note = (
        "[Tool-call history note: an earlier assistant tool-call turn was converted "
        "to text because its tool responses were not contiguous in memory.]"
    )
    text = (content.strip() + "\n\n" + note).strip() if content.strip() else note
    return Message.assistant_message(
        text,
        reasoning_content=getattr(message, "reasoning_content", None),
    )

def _sanitize_incomplete_tool_call_turns(messages: list[Message]) -> list[Message]:
    """Convert non-contiguous assistant tool-call turns to text-only history.

    OpenAI-compatible APIs require an assistant message with N ``tool_calls`` to be
    followed immediately by N matching ``role=tool`` messages. Legacy guard/user
    injections can leave the matching tool output later in the transcript; preserve
    the information as text, but remove executable ``tool_calls`` from that turn.
    """
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        tcs = getattr(m, "tool_calls", None) or []
        if getattr(m, "role", None) != "assistant" or not tcs:
            out.append(m)
            i += 1
            continue

        expected_ids = _tool_call_ids_from_assistant_message(m)
        complete = len(messages) >= i + 1 + len(tcs)
        if complete:
            for j, expected_id in enumerate(expected_ids):
                tm = messages[i + 1 + j]
                if getattr(tm, "role", None) != "tool":
                    complete = False
                    break
                actual_id = str(getattr(tm, "tool_call_id", "") or "")
                if expected_id and actual_id and actual_id != expected_id:
                    complete = False
                    break
        if complete:
            out.append(m)
        else:
            out.append(_assistant_without_tool_calls_for_broken_turn(m))
        i += 1
    return out

def _sanitize_orphan_tool_messages(messages: list[Message]) -> list[Message]:
    """Downgrade ``role=tool`` messages that do not match pending assistant ``tool_calls``.

    Strict OpenAI-compatible APIs (e.g. DashScope) reject requests where a ``tool`` output
    has no matching ``tool_call`` id on the preceding assistant turn (in order). This can
    happen if legacy code injected a synthetic ``tool`` without a corresponding call.
    """
    out: list[Message] = []
    pending: list[str] = []

    for m in messages:
        if m.role == "assistant":
            out.append(m)
            tcs = getattr(m, "tool_calls", None) or []
            pending = _tool_call_ids_from_assistant_message(m) if tcs else []
        elif m.role == "tool":
            tcid = getattr(m, "tool_call_id", None) or ""
            if pending and tcid == pending[0]:
                pending.pop(0)
                out.append(m)
            else:
                content = m.content if isinstance(m.content, str) else str(m.content or "")
                out.append(
                    Message.user_message(
                        "[Tool output downgraded to user message "
                        "(no matching assistant tool_call; strict API compatibility)]\n"
                        + content,
                    ),
                )
        else:
            out.append(m)
            pending = []

    return out

def _tool_call_name_args_for_projection(tc: Any) -> tuple[str, dict[str, Any], str, str]:
    """Return ``(name, args, id, type)`` from a raw stored tool call."""
    tcid = str(tc.get("id") or "") if isinstance(tc, dict) else str(getattr(tc, "id", "") or "")
    tctype = str(tc.get("type") or "function") if isinstance(tc, dict) else str(getattr(tc, "type", "function") or "function")
    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
    if fn is None:
        return "", {}, tcid, tctype
    name = str(fn.get("name") or "") if isinstance(fn, dict) else str(getattr(fn, "name", "") or "")
    raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
    try:
        args = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args, tcid, tctype

def _tool_call_from_projection(
    tc: Any,
    *,
    name: str,
    args: dict[str, Any],
    tcid: str,
    tctype: str,
) -> ToolCall:
    """Build a validated tool call for the LLM view without mutating raw storage."""
    return ToolCall(
        id=tcid or _tool_call_id_from_raw(tc),
        type=tctype or "function",
        function=Function(
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
        ),
    )

def _coerce_tool_call_model_for_llm_projection(tc: Any) -> ToolCall:
    """Return a ``ToolCall`` model so downstream serializers never see raw dicts."""
    if isinstance(tc, ToolCall):
        return tc
    name, args, tcid, tctype = _tool_call_name_args_for_projection(tc)
    return _tool_call_from_projection(tc, name=name, args=args, tcid=tcid, tctype=tctype)

def _assistant_message_with_tool_calls_for_llm_projection(
    message: Message,
    tool_calls: list[Any],
) -> Message:
    """Copy an assistant message while validating projected tool calls."""
    return Message(
        role="assistant",
        content=message.content,
        tool_calls=[
            _coerce_tool_call_model_for_llm_projection(tc)
            for tc in tool_calls
        ],
        name=message.name,
        tool_call_id=message.tool_call_id,
        reasoning_content=message.reasoning_content,
    )

def _project_regular_tool_call_for_llm(tc: Any) -> Any:
    """Strip non-essential args from ordinary tool calls in the LLM-visible view."""
    name, args, tcid, tctype = _tool_call_name_args_for_projection(tc)
    if not name:
        return tc
    changed = False
    if args.pop("config", None) is not None:
        changed = True
    if args.pop("thought", None) is not None:
        changed = True
    if not changed:
        return tc
    return _tool_call_from_projection(tc, name=name, args=args, tcid=tcid, tctype=tctype)

def _write_tool_call_needs_llm_projection(tc: Any) -> bool:
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    return (
        name == "write"
        and isinstance(args.get("content"), str)
        and _should_omit_write_body_from_llm_projection(args["content"])
    )

def _edit_tool_call_needs_llm_projection(tc: Any) -> bool:
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    if name != "edit":
        return False
    for key in ("old_str", "new_str", "old_string", "new_string"):
        val = args.get(key)
        if not isinstance(val, str):
            continue
        if "<<<MEMORY_COMPRESSED" in val or len(val) > 240:
            return True
    return False

def _tool_call_needs_llm_projection(tc: Any) -> bool:
    return _write_tool_call_needs_llm_projection(tc) or _edit_tool_call_needs_llm_projection(tc)

def _tool_result_first_line(tool_msg: Message | None) -> str:
    if tool_msg is None:
        return "(no paired tool result in visible window)"
    content = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content or "")
    first = content.splitlines()[0].strip() if content else ""
    return first or "(empty tool result)"

def _extract_code_map_block_from_tool_feedback(content: str, *, max_chars: int = 1800) -> str:
    if not content or _CODE_MAP_PREFIX not in content:
        return ""
    lines = content.splitlines()
    start = -1
    for i, line in enumerate(lines):
        if line.startswith(_CODE_MAP_PREFIX):
            start = i
            break
    if start < 0:
        return ""
    picked: list[str] = []
    for line in lines[start:]:
        if picked and line.startswith("[source-excerpt"):
            break
        if picked and line.startswith(_CHANGED_RANGE_PREFIX):
            break
        if picked and line.startswith(_SYMBOL_SUMMARY_PREFIX):
            break
        if picked and line.startswith(_AUTO_SNAPSHOT_PREFIX):
            break
        picked.append(line)
    block = "\n".join(picked).strip()
    if len(block) > max_chars:
        block = block[: max_chars - 32].rstrip() + "\n...[code-map truncated]"
    return block

def _historical_write_completion_summary(tool_msg: Message | None) -> str:
    """Stable LLM-visible completion summary for hidden historical write args."""
    if tool_msg is None:
        return ""
    content = tool_msg.content if isinstance(tool_msg.content, str) else str(tool_msg.content or "")
    if not content:
        return ""
    first = _tool_result_first_line(tool_msg)
    code_map = _extract_code_map_block_from_tool_feedback(content)
    if code_map:
        return first + "\n" + code_map
    return first

def _extract_changed_range_line_from_tool_feedback(content: str) -> str:
    if not content:
        return ""
    for line in content.splitlines():
        if line.startswith(_CHANGED_RANGE_PREFIX):
            return line.strip()
    return ""

def _project_historical_tool_result_for_llm(tc: Any, tool_msg: Message | None) -> Message:
    """Convert a historical tool turn into ordinary text when a write payload is hidden."""
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    path = str(args.get("path") or "").replace("\\", "/").lstrip("/")
    first = _tool_result_first_line(tool_msg)
    if name == "write" and isinstance(args.get("content"), str):
        completion = _historical_write_completion_summary(tool_msg)
        lines = [
            "[Historical write tool call omitted from executable LLM context]",
            "Tool: write",
            f"Path: {path or '(unknown)'}",
            f"Result: {first}",
            (
                "This historical write payload looked like a placeholder or memory "
                "compression marker, so it is not shown as callable tool JSON here."
            ),
        ]
        if completion and completion != first:
            lines.extend(["Tool completion summary:", completion])
        lines.append(
            "For exact edit anchors, use a targeted read around the listed line range."
        )
        return Message.user_message("\n".join(lines))

    if name == "edit":
        content = tool_msg.content if tool_msg and isinstance(tool_msg.content, str) else ""
        changed = _extract_changed_range_line_from_tool_feedback(content)
        code_map = _extract_code_map_block_from_tool_feedback(content, max_chars=900)
        lines = [
            "[Historical edit tool call omitted from executable LLM context]",
            "Tool: edit",
            f"Path: {path or '(unknown)'}",
            f"Result: {first}",
            (
                "The exact old_str/new_str payload was large and is not shown as "
                "callable JSON to avoid copying memory-compression markers."
            ),
        ]
        if changed:
            lines.append(f"Changed range: {changed}")
        if code_map:
            lines.extend(["Tool completion summary:", code_map])
        lines.append("For exact edit anchors, use a targeted read around the current symbol range.")
        return Message.user_message("\n".join(lines))

    safe_args = dict(args)
    safe_args.pop("config", None)
    safe_args.pop("thought", None)
    for key in ("content", "old_str", "new_str"):
        if isinstance(safe_args.get(key), str) and len(safe_args[key]) > 240:
            safe_args[key] = safe_args[key][:200] + "...[truncated]"
    try:
        args_text = json.dumps(safe_args, ensure_ascii=False)
    except (TypeError, ValueError):
        args_text = "{}"
    return Message.user_message(
        "[Historical tool result from a turn whose large write payload was hidden]\n"
        f"Tool: {name or '(unknown)'}\n"
        f"Args: {args_text}\n"
        f"Result: {first}"
    )

def _project_messages_for_llm_view(messages: list[Message]) -> list[Message]:
    """Return an LLM-only projection, leaving the stored event log untouched.

    Assistant-authored write/edit/bash arguments are preserved for cacheable history.
    Only unsafe placeholder-like writes or legacy compressed edit payloads are converted
    to ordinary text so the model does not copy memory-compression markers as code.
    """
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if getattr(m, "role", None) != "assistant" or not getattr(m, "tool_calls", None):
            out.append(m)
            i += 1
            continue

        tcs = list(getattr(m, "tool_calls", None) or [])
        needs_projection = any(_tool_call_needs_llm_projection(tc) for tc in tcs)
        if not needs_projection:
            projected_tcs = [_project_regular_tool_call_for_llm(tc) for tc in tcs]
            out.append(_assistant_message_with_tool_calls_for_llm_projection(m, projected_tcs))
            i += 1
            continue

        tool_msgs: list[Message] = []
        j = i + 1
        expected_ids = [_tool_call_id_from_raw(tc) for tc in tcs]
        while j < len(messages) and getattr(messages[j], "role", None) == "tool":
            tool_msgs.append(messages[j])
            j += 1
            if len(tool_msgs) >= len(tcs):
                break

        content = m.content if isinstance(m.content, str) else str(m.content or "")
        summary = content.strip()
        note = (
            "[LLM memory view: one or more unsafe historical write payloads or "
            "compressed edit payloads from this assistant turn are summarized "
            "below as text, not exposed as tool-call JSON.]"
        )
        if summary:
            summary = summary + "\n\n" + note
        else:
            summary = note
        out.append(Message.assistant_message(summary, reasoning_content=getattr(m, "reasoning_content", None)))

        by_id = {
            str(getattr(tm, "tool_call_id", "") or ""): tm
            for tm in tool_msgs
        }
        for idx, tc in enumerate(tcs):
            tm = by_id.get(expected_ids[idx]) if idx < len(expected_ids) else None
            if tm is None and idx < len(tool_msgs):
                tm = tool_msgs[idx]
            out.append(_project_historical_tool_result_for_llm(tc, tm))
        i = j

    return out

def _finalize_messages_for_llm(messages: list[Message]) -> list[Message]:
    """Prefix fix + strict tool-call adjacency sanitization for LLM APIs."""
    return _sanitize_orphan_tool_messages(
        _sanitize_incomplete_tool_call_turns(
            _ensure_complete_tool_turn_prefix(messages),
        ),
    )

def _best_suffix_for_budget_priority(
    rest: list[Message],
    budget_rest: int,
    *,
    min_k: int = 0,
) -> tuple[list[Message], int]:
    """Pick ``rest[k:]`` that maximizes sum of :func:`_message_priority_score` within ``budget_rest`` chars.

    *min_k* enforces a monotone lower bound on the suffix start so the prefix sent to the LLM
    never grows backwards — improving prefix-cache hit rates across consecutive calls.

    Tie-break: prefer smaller *k* (longer suffix, more recent context). If no non-empty suffix fits,
    fall back to newest-first greedy (same as legacy sliding window).

    Returns ``(chosen_messages, actual_k)``.
    """
    n = len(rest)
    # Safety: if previous min_k is beyond current history length, reset to 0.
    if min_k > n:
        min_k = 0
    best_k = n
    best_score = -1
    for k in range(min_k, n + 1):
        suffix = rest[k:]
        total_len = sum(_message_char_len(m) for m in suffix)
        if total_len > budget_rest:
            continue
        score = sum(_message_priority_score(m) for m in suffix)
        if score > best_score or (score == best_score and k < best_k):
            best_score = score
            best_k = k
    if best_score >= 0:
        return rest[best_k:], best_k
    # Fallback: newest-first greedy, still constrained to >= min_k.
    chosen_rev: list[Message] = []
    tot = 0
    fallback_k = n
    for i, m in enumerate(reversed(rest)):
        lc = _message_char_len(m)
        if chosen_rev and tot + lc > budget_rest:
            break
        chosen_rev.append(m)
        tot += lc
        fallback_k = n - 1 - i
    actual_k = max(min_k, fallback_k) if chosen_rev else n
    return list(reversed(chosen_rev)), actual_k

_EDIT_SNAPSHOT_MARKER = "--- Current file snapshot (after edit) ---"

def compress_edit_success_output_for_memory(obs: str) -> str:
    """Drop multi-line context preview from successful edit tool output (keep header + hash line).

    Never raises: failures fall back to *obs* unchanged.
    """
    try:
        body = obs
        had_snapshot = _EDIT_SNAPSHOT_MARKER in obs
        if had_snapshot:
            body = obs.split(_EDIT_SNAPSHOT_MARKER, 1)[0].rstrip()
        lines = body.splitlines()
        if len(lines) < 2:
            return body if had_snapshot else obs
        first, last = lines[0], lines[-1]
        if not first.startswith("Edited ") or "1 replacement OK." not in first:
            return body if had_snapshot else obs
        # Allow variable hash length and optional trailing metadata on the summary line.
        if not re.match(r"^\(\d+ lines, sha256~[a-f0-9]+", last):
            return body if had_snapshot else obs
        return f"{first}\n{last}"
    except Exception:
        logger.debug("compress_edit_success_output_for_memory failed", exc_info=True)
        return obs

def _parse_write_success_metadata_from_output(text: str) -> tuple[int | None, str | None]:
    """Parse (lines, sha256_short) from WriteTool / no-op success text."""
    if not (text and isinstance(text, str)):
        return None, None
    m = re.search(
        r"\((\d+)\s+lines,.*?sha256~([a-f0-9]+)",
        text[:4000],
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), m.group(2)
    return None, None

def write_success_feedback_for_memory(
    rel_path: str,
    *,
    lines: int | None = None,
    sha256_short: str | None = None,
) -> str:
    """Short success line for write tool results stored in memory.

    Includes line count + content fingerprint when available so the model can
    trust the write; avoids suggesting unnecessary reads.
    """
    p = (rel_path or "").replace("\\", "/").lstrip("/").strip() or "file"
    if lines is not None and sha256_short:
        return (
            f"File `{p}` written successfully ({lines} lines, sha256~{sha256_short}). "
            "Syntax check passed; file is complete on disk. Use bash to test when ready. "
            "Do not rewrite the same content without changes."
        )
    return (
        f"File `{p}` written successfully; syntax check passed. "
        "File is on disk. Use bash to test when ready."
    )


def _tool_call_id_from_raw(tc: Any) -> str:
    """Return the stable tool-call ID used by LLM-view projection."""

    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def _all_messages(memory: Memory) -> list[Message]:
    records = memory.chat_history_memory.retrieve(window_size=None)
    return [cr.memory_record.message for cr in records]

__all__ = tuple(name for name in globals() if not name.startswith("__"))
