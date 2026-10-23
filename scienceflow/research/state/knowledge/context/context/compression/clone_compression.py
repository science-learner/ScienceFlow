"""Memory-context responsibility: clone compression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inquirycraft.memory import Memory, MemoryRecord, Message

from scienceflow.research.state.knowledge.context.context.results.base import (
    logger,
)
from scienceflow.research.state.knowledge.context.context.compression.clone_inheritance import (
    _compile_clone_inherit_signal_patterns,
    _compress_inherited_bash_output,
    _compress_protected_user_message_for_inherit,
    _compress_stale_fork_user_message_for_inherit,
    _estimate_transcript_chars,
    _is_inherited_auto_snapshot_tool_message,
    _is_inherited_code_file_path,
    _maybe_compress_assistant_tool_calls_for_inherit,
    _norm_code_extensions_for_inherit,
    _snapshot_path_from_auto_snapshot_content,
    _write_tool_feedback_is_success,
    compress_inherited_tool_message_content,
    slice_clone_minimal_inherited_messages,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _all_messages,
    _tool_call_id_from_raw,
    _tool_call_name_args_for_projection,
)
from scienceflow.research.state.knowledge.context.context.results.tool_results import _tool_args_dict_for_tool_message
from scienceflow.research.state.knowledge.context.source_snapshot import (
    _AUTO_SNAPSHOT_PREFIX,
    extract_write_auto_snapshot_block,
)

def _clear_memory_and_long_term_log(memory: Memory) -> None:
    """Clear chat storage and truncate ``long_term.jsonl`` before bulk re-import.

    ``JsonKeyValueStorage.clear()`` / ``BaseChatHistoryMemory.clear()`` clears
    ``short_term.json`` but leaves ``long_term.jsonl`` append-only; subsequent
    :meth:`Memory.add_message` would duplicate every record in the JSONL.
    """
    memory.chat_history_memory.clear()
    lt = getattr(memory, "long_term_log", None)
    if lt:
        p = Path(str(lt))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
        except OSError:
            logger.debug("truncate long_term.jsonl failed: %s", lt, exc_info=True)
    else:
        storage = getattr(memory.chat_history_memory, "storage", None)
        jp = getattr(storage, "json_path", None) if storage is not None else None
        if jp is not None:
            alt = Path(jp).parent / "long_term.jsonl"
            try:
                if alt.is_file():
                    alt.write_text("", encoding="utf-8")
            except OSError:
                logger.debug("truncate fallback long_term failed: %s", alt, exc_info=True)

def _tool_call_targets_result_md(tc: Any) -> bool:
    name, args, _tcid, _tctype = _tool_call_name_args_for_projection(tc)
    if name not in {"write", "edit"}:
        return False
    raw_path = str(args.get("path") or "").replace("\\", "/").strip()
    raw_path = raw_path.lstrip("./")
    return raw_path == "result.md" or raw_path.endswith("/result.md")

def _assistant_writes_result_md(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "assistant":
        return False
    return any(
        _tool_call_targets_result_md(tc)
        for tc in (getattr(msg, "tool_calls", None) or [])
    )

def _is_result_md_recovery_user_message(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "user":
        return False
    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    lower = content.lower()
    if "result.md" not in lower:
        return False
    if "write" not in lower and "must run a full training pass" not in lower:
        return False
    return (
        "result.md is missing" in lower
        or "result.md does not exist yet" in lower
        or "write result.md" in lower
        or "write it now" in lower
    )

def _is_science_agent_stop_summary(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "assistant":
        return False
    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    return content.strip().startswith("[ScienceAgent] Stopped after")

def _is_result_md_interstitial_guard_user_message(msg: Message) -> bool:
    """Return true for guard notes injected while the result.md writer is pending."""
    if (getattr(msg, "role", "") or "") != "user":
        return False
    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    stripped = content.strip()
    return (
        stripped.startswith("[Guard] No meaningful progress detected for multiple rounds")
        or stripped.startswith("[System] No meaningful progress detected for multiple rounds")
        or stripped.startswith("[Guard] No successful `python3 solution.py`")
        or stripped.startswith("[System] No successful `python3 solution.py`")
        or stripped.startswith("[Guard] Detected repeated single-`read` rounds.")
        or stripped.startswith("[LNR] Detected repeated single-`read` rounds.")
    )

def _is_text_only_assistant(msg: Message) -> bool:
    return (
        (getattr(msg, "role", "") or "") == "assistant"
        and not (getattr(msg, "tool_calls", None) or [])
    )

def _skip_assistant_tool_round(messages: list[Message], assistant_idx: int) -> int:
    """Return the index after one assistant tool-call turn.

    Strict LLM APIs require every assistant ``tool_calls`` message to be followed by
    matching ``role=tool`` responses. When dropping a report write, drop the
    assistant message and its matching tool responses together.
    """
    msg = messages[assistant_idx]
    tcs = list(getattr(msg, "tool_calls", None) or [])
    expected = [_tool_call_id_from_raw(tc) for tc in tcs]
    expected_set = {x for x in expected if x}
    j = assistant_idx + 1
    seen = 0
    while j < len(messages) and getattr(messages[j], "role", None) == "tool":
        tcid = str(getattr(messages[j], "tool_call_id", "") or "")
        if expected_set and tcid and tcid not in expected_set:
            break
        seen += 1
        j += 1
        if tcs and seen >= len(tcs):
            break
    return j

def _skip_result_md_recovery_segment(messages: list[Message], start_idx: int) -> int:
    """Return the index after a result.md recovery segment.

    The recovery prompt is bookkeeping for closing the current node. It should
    not be inherited by the next node, and neither should the immediate tool
    attempts made only to satisfy that prompt. Guard notes can be injected as
    user messages while the report writer is pending; keep treating those as
    part of the bookkeeping segment so the child inherits the preceding
    successful ``python3 solution.py`` tool result as the natural transcript tail.
    """
    j = start_idx + 1
    while j < len(messages):
        if _assistant_writes_result_md(messages[j]):
            j = _skip_assistant_tool_round(messages, j)
            while j < len(messages) and (
                _is_science_agent_stop_summary(messages[j])
                or _is_text_only_assistant(messages[j])
            ):
                j += 1
            return j
        if (
            getattr(messages[j], "role", None) == "user"
            and not _is_result_md_interstitial_guard_user_message(messages[j])
        ):
            break
        j += 1
    while j < len(messages) and _is_science_agent_stop_summary(messages[j]):
        j += 1
    return j

def drop_result_md_summary_rounds_for_inherit(memory: Memory) -> dict[str, Any]:
    """Remove result-report summary turns from clone-inherited chat memory.

    ``result.md`` is a node artifact for tracking/journal context, not part of
    the next node's natural tool trajectory. This removes recovery prompts and
    assistant ``write/edit result.md`` tool turns as whole assistant+tool groups,
    preserving valid tool-call pairing in the remaining transcript. The preceding
    successful ``python3 solution.py`` bash turn is intentionally kept so clone
    children inherit the parent's final score and run-control evidence naturally.
    """
    all_records = memory.chat_history_memory.retrieve(window_size=None)
    if not all_records:
        return {"changed": False, "messages_dropped": 0, "rounds_dropped": 0}
    messages = [cr.memory_record.message for cr in all_records]
    new_messages: list[Message] = []
    dropped = 0
    rounds = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if _is_result_md_recovery_user_message(msg):
            end = _skip_result_md_recovery_segment(messages, i)
            dropped += end - i
            rounds += 1
            i = end
            continue
        if _assistant_writes_result_md(msg):
            end = _skip_assistant_tool_round(messages, i)
            while end < len(messages) and _is_science_agent_stop_summary(messages[end]):
                end += 1
            dropped += end - i
            rounds += 1
            i = end
            continue
        new_messages.append(msg)
        i += 1

    if not dropped:
        return {"changed": False, "messages_dropped": 0, "rounds_dropped": 0}
    _clear_memory_and_long_term_log(memory)
    for msg in new_messages:
        memory.add_message(msg)
    return {
        "changed": True,
        "messages_dropped": dropped,
        "rounds_dropped": rounds,
    }

def _supersede_stale_compressed_records(
    *,
    memory: Memory,
    released_sigs: set[tuple[str, str]],
    seen_sigs: dict[tuple[str, str], int],
) -> None:
    """Replace first-call compressed tool rows with a one-line note; drop tracking state."""
    if not released_sigs:
        return
    storage = getattr(getattr(memory, "chat_history_memory", None), "storage", None)
    if storage is None:
        released_sigs.clear()
        return
    try:
        records = storage.load()
    except Exception:
        logger.debug("supersede: storage.load failed", exc_info=True)
        released_sigs.clear()
        return
    if not isinstance(records, list) or not records:
        released_sigs.clear()
        return

    changed = False
    for sig in list(released_sigs):
        idx = seen_sigs.get(sig)
        if idx is None or idx < 0 or idx >= len(records):
            seen_sigs.pop(sig, None)
            continue
        rec = records[idx]
        if not isinstance(rec, dict):
            seen_sigs.pop(sig, None)
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            seen_sigs.pop(sig, None)
            continue
        msg["content"] = (
            f"[superseded by later {sig[0]} call with identical args — compressed output collapsed]"
        )
        changed = True
        seen_sigs.pop(sig, None)

    released_sigs.clear()

    if not changed:
        return

    chm = memory.chat_history_memory
    chm.clear()
    for rec in records:
        chm.add_record(MemoryRecord.from_dict(rec))

@dataclass(frozen=True)
class CloneCompressionPolicy:
    patterns: list[Any]
    code_extensions: set[str]
    read_max: int
    bash_tail: int
    omit_solution_reads: bool
    write_compact: bool
    edit_compact: bool
    tool_call_budget: int
    snapshot_max_lines: int
    snapshot_max_chars: int
    exempt_last_state_write: bool


def _clone_compression_noop() -> dict[str, Any]:
    return {
        "changed": False,
        "user_msgs": 0,
        "tool_msgs": 0,
        "assistant_tc_args": 0,
        "snapshots_dropped": 0,
        "chars_before": 0,
        "chars_after": 0,
    }


def _clone_compression_policy(
    cfg: Any,
    *,
    repl_mode: bool,
    preserve_initial_eda: bool,
) -> CloneCompressionPolicy:
    raw_patterns = getattr(cfg, "clone_inherit_signal_patterns", None)
    if raw_patterns is not None and not isinstance(raw_patterns, list):
        raw_patterns = None
    patterns = _compile_clone_inherit_signal_patterns(
        [str(item) for item in raw_patterns] if raw_patterns else None
    )
    return CloneCompressionPolicy(
        patterns=patterns,
        code_extensions=_norm_code_extensions_for_inherit(
            getattr(cfg, "write_auto_snapshot_code_extensions", None)
        ),
        read_max=int(getattr(cfg, "clone_inherit_read_max_lines", 80) or 80),
        bash_tail=int(getattr(cfg, "clone_inherit_bash_tail_lines", 20) or 20),
        omit_solution_reads=bool(getattr(cfg, "clone_inherit_omit_solution_py_reads", True)),
        write_compact=bool(getattr(cfg, "clone_inherit_write_compact", True)),
        edit_compact=bool(getattr(cfg, "clone_inherit_edit_compact", True)),
        tool_call_budget=int(
            getattr(cfg, "clone_inherit_tool_args_head_tail_chars", 2000) or 0
        ),
        snapshot_max_lines=int(
            getattr(cfg, "clone_inherit_write_snapshot_max_lines", 0) or 0
        ),
        snapshot_max_chars=int(
            getattr(cfg, "clone_inherit_write_snapshot_max_chars", 0) or 0
        ),
        exempt_last_state_write=bool(
            getattr(cfg, "clone_inherit_last_state_write_exempt", True)
        ) and repl_mode and not preserve_initial_eda,
    )


def _assistant_write_path(tool_call: Any) -> str:
    name = str(getattr(getattr(tool_call, "function", None), "name", "") or "")
    if name != "write":
        return ""
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return ""
    return str(args.get("path") or "") if isinstance(args, dict) else ""


def _clone_compression_boundaries(
    messages: list[Message],
    policy: CloneCompressionPolicy,
    *,
    preserve_initial_eda: bool,
) -> tuple[int, tuple[int, int] | None, dict[str, int]]:
    protected_end = len(messages) if preserve_initial_eda else 0
    last_state_write: tuple[int, int] | None = None
    for message_index, message in enumerate(messages):
        if (getattr(message, "role", "") or "") != "assistant" or not message.tool_calls:
            continue
        for call_index, tool_call in enumerate(message.tool_calls):
            path = _assistant_write_path(tool_call)
            if not path or not _is_inherited_code_file_path(path, policy.code_extensions):
                continue
            if preserve_initial_eda and protected_end == len(messages):
                protected_end = message_index
            if policy.exempt_last_state_write:
                last_state_write = (message_index, call_index)
    latest_snapshots: dict[str, int] = {}
    if preserve_initial_eda:
        for index, message in enumerate(messages[protected_end:], start=protected_end):
            if not _is_inherited_auto_snapshot_tool_message(message):
                continue
            args = _tool_args_dict_for_tool_message(messages, index)
            content = message.content if isinstance(message.content, str) else str(message.content or "")
            rel = str((args or {}).get("path") or "").replace("\\", "/").lstrip("/")
            rel = rel or _snapshot_path_from_auto_snapshot_content(content)
            if rel and _is_inherited_code_file_path(rel, policy.code_extensions):
                latest_snapshots[rel] = index
    return protected_end, last_state_write, latest_snapshots


def _compress_clone_messages(
    messages: list[Message],
    policy: CloneCompressionPolicy,
    *,
    repl_mode: bool,
    preserve_initial_eda: bool,
    protected_end: int,
    last_state_write: tuple[int, int] | None,
    latest_snapshots: dict[str, int],
) -> tuple[list[Message], int, int, int, int]:
    new_messages: list[Message] = []
    user_changed = tool_changed = assistant_changed = snapshots_dropped = 0
    for index, message in enumerate(messages):
        projected = message
        role = getattr(message, "role", "") or ""
        if preserve_initial_eda and index < protected_end:
            if role == "user" and isinstance(message.content, str):
                content = _compress_protected_user_message_for_inherit(message.content)
                if content != message.content:
                    user_changed += 1
                    projected = message.model_copy(update={"content": content})
            new_messages.append(projected)
            continue
        if role == "user" and preserve_initial_eda and isinstance(message.content, str):
            content = _compress_stale_fork_user_message_for_inherit(message.content)
            if content != message.content:
                user_changed += 1
                projected = message.model_copy(update={"content": content})
        elif role == "assistant" and message.tool_calls and policy.tool_call_budget > 0:
            exempt_index = (
                last_state_write[1]
                if last_state_write is not None and last_state_write[0] == index
                else None
            )
            projected = _maybe_compress_assistant_tool_calls_for_inherit(
                message,
                max_total_chars=policy.tool_call_budget,
                exempt_write_call_index=exempt_index,
            )
            if projected is not message:
                assistant_changed += 1
        elif role == "tool" and isinstance(message.content, str):
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            if tool_call_id.startswith("inherited_fullrun_"):
                new_messages.append(projected)
                continue
            name = str(getattr(message, "name", "") or "")
            if repl_mode:
                if name == "bash":
                    content = _compress_inherited_bash_output(
                        message.content,
                        tail_lines=policy.bash_tail,
                        patterns=policy.patterns,
                        repl_mode=True,
                    )
                    if content != message.content:
                        tool_changed += 1
                        projected = message.model_copy(update={"content": content})
            else:
                args = _tool_args_dict_for_tool_message(messages, index)
                rel = str((args or {}).get("path") or "").replace("\\", "/").lstrip("/")
                rel = rel or _snapshot_path_from_auto_snapshot_content(message.content)
                keep_snapshot = (
                    not preserve_initial_eda
                    or not rel
                    or latest_snapshots.get(rel) == index
                )
                content = compress_inherited_tool_message_content(
                    message.content,
                    tool_name=name,
                    tool_args=args,
                    read_max_lines=policy.read_max,
                    bash_tail_lines=policy.bash_tail,
                    omit_solution_py_reads=policy.omit_solution_reads,
                    write_compact=policy.write_compact,
                    edit_compact=policy.edit_compact,
                    patterns=policy.patterns,
                    write_snapshot_inherit_max_lines=policy.snapshot_max_lines,
                    write_snapshot_inherit_max_chars=policy.snapshot_max_chars,
                    keep_auto_snapshot=keep_snapshot,
                    code_extensions=policy.code_extensions,
                )
                if content != message.content:
                    tool_changed += 1
                    if not keep_snapshot and _is_inherited_auto_snapshot_tool_message(message):
                        snapshots_dropped += 1
                    projected = message.model_copy(update={"content": content})
        new_messages.append(projected)
    return new_messages, user_changed, tool_changed, assistant_changed, snapshots_dropped


def apply_clone_inherit_compression(
    memory: Memory,
    cfg: Any,
    *,
    repl_mode: bool = False,
    preserve_initial_eda: bool = False,
) -> dict[str, Any]:
    """Rewrite inherited chat messages in place using the configured projection policy."""
    if not bool(getattr(cfg, "clone_inherit_compress", True)):
        return _clone_compression_noop()
    records = memory.chat_history_memory.retrieve(window_size=None)
    if not records:
        return _clone_compression_noop()
    messages = [record.memory_record.message for record in records]
    policy = _clone_compression_policy(
        cfg,
        repl_mode=repl_mode,
        preserve_initial_eda=preserve_initial_eda,
    )
    protected_end, last_state_write, latest_snapshots = _clone_compression_boundaries(
        messages,
        policy,
        preserve_initial_eda=preserve_initial_eda,
    )
    projected, user_count, tool_count, assistant_count, snapshots_dropped = (
        _compress_clone_messages(
            messages,
            policy,
            repl_mode=repl_mode,
            preserve_initial_eda=preserve_initial_eda,
            protected_end=protected_end,
            last_state_write=last_state_write,
            latest_snapshots=latest_snapshots,
        )
    )
    changed = bool(user_count or tool_count or assistant_count)
    if changed:
        _clear_memory_and_long_term_log(memory)
        for message in projected:
            memory.add_message(message)
    return {
        "changed": changed,
        "user_msgs": user_count,
        "tool_msgs": tool_count,
        "assistant_tc_args": assistant_count,
        "snapshots_dropped": snapshots_dropped,
        "chars_before": _estimate_transcript_chars(messages),
        "chars_after": _estimate_transcript_chars(projected),
    }


def _assistant_tool_call_writes_solution(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "assistant":
        return False
    for tc in getattr(msg, "tool_calls", None) or []:
        name = ""
        raw_args = ""
        if isinstance(tc, dict):
            tc_fn = tc.get("function") or {}
            if isinstance(tc_fn, dict):
                name = str(tc_fn.get("name") or "")
                raw_args = str(tc_fn.get("arguments") or "")
        else:
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "") if fn is not None else ""
            raw_args = str(getattr(fn, "arguments", "") or "") if fn is not None else ""
        if name not in {"write", "edit"}:
            continue
        try:
            args = json.loads(raw_args or "{}")
        except (TypeError, json.JSONDecodeError):
            args = {}
        if isinstance(args, dict) and _is_inherited_code_file_path(str(args.get("path") or "")):
            return True
    return False

def _first_solution_write_assistant_index(messages: list[Message]) -> int:
    for idx, msg in enumerate(messages):
        if _assistant_tool_call_writes_solution(msg):
            return idx
    return len(messages)

def _assistant_index_for_tool_index(messages: list[Message], tool_idx: int) -> int | None:
    tcid = str(getattr(messages[tool_idx], "tool_call_id", "") or "")
    j = tool_idx - 1
    while j >= 0 and getattr(messages[j], "role", None) == "tool":
        j -= 1
    if j < 0 or getattr(messages[j], "role", None) != "assistant":
        return None
    if not tcid:
        return j
    for tc in getattr(messages[j], "tool_calls", None) or []:
        if _tool_call_id_from_raw(tc) == tcid:
            return j
    return j

def _assistant_round_indices(messages: list[Message], assistant_idx: int | None) -> set[int]:
    if assistant_idx is None or assistant_idx < 0 or assistant_idx >= len(messages):
        return set()
    out = {assistant_idx}
    expected = _tool_call_id_set_for_message(messages[assistant_idx])
    j = assistant_idx + 1
    while j < len(messages) and getattr(messages[j], "role", None) == "tool":
        tcid = str(getattr(messages[j], "tool_call_id", "") or "")
        if not expected or not tcid or tcid in expected:
            out.add(j)
        j += 1
    return out

def _tool_call_id_set_for_message(msg: Message) -> set[str]:
    ids: set[str] = set()
    for tc in getattr(msg, "tool_calls", None) or []:
        tcid = _tool_call_id_from_raw(tc)
        if tcid:
            ids.add(tcid)
    return ids

def _last_solution_snapshot_tool_index(messages: list[Message]) -> int | None:
    by_path = _last_code_snapshot_tool_indices_by_path(messages)
    return max(by_path.values(), default=None)

def _last_code_snapshot_tool_indices_by_path(messages: list[Message]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if (getattr(msg, "role", "") or "") != "tool":
            continue
        name = str(getattr(msg, "name", "") or "")
        if name not in {"write", "edit"}:
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        args = _tool_args_dict_for_tool_message(messages, idx) or {}
        rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
        if not rel:
            rel = _snapshot_path_from_auto_snapshot_content(content)
        if not _is_inherited_code_file_path(rel):
            continue
        if _AUTO_SNAPSHOT_PREFIX in content or _write_tool_feedback_is_success(content) or content.startswith("Edited "):
            out.setdefault(rel, idx)
    return out

def _last_signal_bash_tool_index(messages: list[Message]) -> int | None:
    fallback: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if (getattr(msg, "role", "") or "") != "tool":
            continue
        if str(getattr(msg, "name", "") or "") != "bash":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if not content:
            continue
        if fallback is None:
            fallback = idx
        if "Final Validation Score" in content or "METRIC:" in content:
            return idx
        first = content.splitlines()[0] if content else ""
        if first.startswith("[exit=0"):
            return idx
    return fallback

def _message_chars(msg: Message) -> int:
    return _estimate_transcript_chars([msg])

def _selected_chars(messages: list[Message], indices: set[int]) -> int:
    return sum(_message_chars(messages[i]) for i in sorted(indices))

def _truncate_for_capsule(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    keep = max(0, max_chars - 32)
    head = max(0, keep // 2)
    tail = max(0, keep - head)
    return s[:head].rstrip() + "\n...[truncated]...\n" + s[-tail:].lstrip()

def build_shallow_continuity_capsule(
    memory: Memory,
    *,
    max_chars: int = 12000,
) -> str:
    """Bounded B_DEEP handoff from the most recent shallow memory.

    The capsule is deterministic: it preserves recent intent, current code state,
    and last validation signal without copying the whole branch transcript.
    """
    messages = _all_messages(memory)
    if not messages or max_chars <= 0:
        return ""
    first_write = _first_solution_write_assistant_index(messages)
    latest_snapshot_indices = _last_code_snapshot_tool_indices_by_path(messages)
    latest_bash_idx = _last_signal_bash_tool_index(messages)

    parts: list[str] = ["## Recent shallow continuity capsule"]
    if first_write < len(messages):
        parts.append("- scope: latest shallow state after initial EDA and solution creation")
    else:
        parts.append("- scope: recent shallow state; no solution write found in inherited memory")

    last_intent = ""
    for msg in reversed(messages[first_write:]):
        if (getattr(msg, "role", "") or "") != "assistant":
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if content.strip():
            last_intent = _truncate_for_capsule(content, 1000)
            break
    if last_intent:
        parts.append("\n### Recent assistant intent\n" + last_intent)

    if latest_snapshot_indices:
        snap_parts: list[str] = []
        for rel, idx in sorted(latest_snapshot_indices.items()):
            msg = messages[idx]
            content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            snap = extract_write_auto_snapshot_block(content) or content
            snap_parts.append(f"#### {rel}\n" + _truncate_for_capsule(snap, 1200))
        parts.append("\n### Current code state\n" + _truncate_for_capsule("\n\n".join(snap_parts), 2600))

    if latest_bash_idx is not None:
        msg = messages[latest_bash_idx]
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        parts.append("\n### Latest shallow run signal\n" + _truncate_for_capsule(content, 1800))

    last_gate = ""
    for msg in reversed(messages[first_write:]):
        if (getattr(msg, "role", "") or "") != "user":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if "NODE-GATE" in content or "Phase budget" in content or "result.md" in content:
            last_gate = _truncate_for_capsule(content, 900)
            break
    if last_gate:
        parts.append("\n### Recent gate / budget signal\n" + last_gate)
    out = "\n".join(parts).strip()
    return _truncate_for_capsule(out, max_chars)

def apply_clone_non_a_memory_budget(
    memory: Memory,
    cfg: Any,
    *,
    fork_class: str,
) -> dict[str, Any]:
    """Hard-budget B/C inherited memory while preserving draft EDA and current state.

    This runs after clone inheritance compression. It is intentionally not used for
    A_TELEPORT, whose cache behavior depends on prefix-safe transcript continuity.
    """
    budget = int(getattr(cfg, "clone_non_a_inherit_budget_chars", 0) or 0)
    if budget <= 0:
        return {
            "changed": False,
            "chars_before": 0,
            "chars_after": 0,
            "budget": budget,
            "messages_before": 0,
            "messages_after": 0,
        }
    messages = _all_messages(memory)
    if not messages:
        return {
            "changed": False,
            "chars_before": 0,
            "chars_after": 0,
            "budget": budget,
            "messages_before": 0,
            "messages_after": 0,
        }
    before = _estimate_transcript_chars(messages)
    if before <= budget:
        return {
            "changed": False,
            "chars_before": before,
            "chars_after": before,
            "budget": budget,
            "messages_before": len(messages),
            "messages_after": len(messages),
        }

    fork = str(fork_class or "").upper()
    first_write = _first_solution_write_assistant_index(messages)
    selected: set[int] = set(range(0, first_write))

    latest_code_indices = _last_code_snapshot_tool_indices_by_path(messages)
    latest_bash_idx = _last_signal_bash_tool_index(messages)

    essential_groups: list[set[int]] = []
    for latest_solution_idx in sorted(set(latest_code_indices.values())):
        essential_groups.append(
            _assistant_round_indices(messages, _assistant_index_for_tool_index(messages, latest_solution_idx)),
        )
    if latest_bash_idx is not None:
        essential_groups.append(
            _assistant_round_indices(messages, _assistant_index_for_tool_index(messages, latest_bash_idx)),
        )

    for group in essential_groups:
        selected.update(group)

    optional_groups: list[set[int]] = []
    if fork == "B_DEEP":
        # Preserve recent shallow continuity as rounds, newest first. This is a
        # bounded tail, not the whole branch transcript.
        idx = len(messages) - 1
        while idx >= first_write:
            role = getattr(messages[idx], "role", "") or ""
            if role == "tool":
                aidx = _assistant_index_for_tool_index(messages, idx)
                group = _assistant_round_indices(messages, aidx)
                if group:
                    optional_groups.append(group)
                    idx = min(group) - 1
                    continue
            optional_groups.append({idx})
            idx -= 1
            if len(optional_groups) >= 12:
                break
    else:
        kept_users = 0
        for idx in range(len(messages) - 1, first_write - 1, -1):
            msg = messages[idx]
            if (getattr(msg, "role", "") or "") == "user":
                optional_groups.append({idx})
                kept_users += 1
                if kept_users >= 2:
                    break

    for group in optional_groups:
        if not group or group.issubset(selected):
            continue
        trial = selected | group
        if _selected_chars(messages, trial) <= budget:
            selected = trial

    new_messages = [messages[i] for i in sorted(selected)]
    after = _estimate_transcript_chars(new_messages)
    changed = len(new_messages) != len(messages) or after != before
    if changed:
        _clear_memory_and_long_term_log(memory)
        for msg in new_messages:
            memory.add_message(msg)
    return {
        "changed": bool(changed),
        "chars_before": before,
        "chars_after": after,
        "budget": budget,
        "messages_before": len(messages),
        "messages_after": len(new_messages),
        "budget_overflow": after > budget,
        "protected_prefix_messages": first_write,
    }

def apply_clone_minimal_slice_to_memory(memory: Memory) -> bool:
    """Drop middle transcript; keep leading users + suffix from last assistant."""
    msgs = _all_messages(memory)
    sliced = slice_clone_minimal_inherited_messages(msgs)
    if sliced is msgs:
        return False
    if len(sliced) == len(msgs) and all(a is b for a, b in zip(sliced, msgs, strict=False)):
        return False
    _clear_memory_and_long_term_log(memory)
    for m in sliced:
        memory.add_message(m)
    return True

__all__ = tuple(name for name in globals() if not name.startswith("__"))
