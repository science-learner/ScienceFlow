"""Memory-context responsibility: tool results."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from inquirycraft.memory import Message

from scienceflow.runtime.core.process.commands import looks_like_bare_solution_run
from scienceflow.research.state.knowledge.context.context.results.base import (
    _TOOL_FEEDBACK_TRIMMED_PREFIX,
    logger,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _parse_write_success_metadata_from_output,
    _tool_call_id_from_raw,
    compress_edit_success_output_for_memory,
    write_success_feedback_for_memory,
)
from scienceflow.research.state.knowledge.context.source_snapshot import (
    _AUTO_SNAPSHOT_PREFIX,
    _build_source_code_map,
    extract_write_auto_snapshot_block,
)


def _rel_path_for_write_compact(
    tool_args: dict[str, Any] | None,
    content: str,
) -> str:
    if tool_args and tool_args.get("path"):
        return str(tool_args["path"]).replace("\\", "/").lstrip("/").strip() or "file"
    first = content.split("\n", 1)[0] if content else ""
    match = re.match(r"^Written\s+(\S+)", first.strip())
    if match:
        return match.group(1).strip().replace("\\", "/").lstrip("/") or "file"
    return "file"

def _strip_tool_feedback_trim_header(content: str) -> str:
    """Drop the generic LLM-context trim header before inherited compaction."""
    if not content.startswith(_TOOL_FEEDBACK_TRIMMED_PREFIX):
        return content
    lines = content.splitlines()
    if not lines:
        return content
    return "\n".join(lines[1:]).lstrip()

def _snapshot_inherit_limits(
    *,
    read_max_lines: int,
    write_snapshot_inherit_max_lines: int,
    write_snapshot_inherit_max_chars: int,
) -> tuple[int, int]:
    wlim = (
        int(write_snapshot_inherit_max_lines)
        if int(write_snapshot_inherit_max_lines) > 0
        else max(int(read_max_lines or 0), 120)
    )
    wch = (
        int(write_snapshot_inherit_max_chars)
        if int(write_snapshot_inherit_max_chars) > 0
        else 24000
    )
    return wlim, wch

def _edit_success_feedback_for_inherit(content: str, rel: str) -> str:
    body = _strip_tool_feedback_trim_header(content).strip()
    prefix = body.split(_AUTO_SNAPSHOT_PREFIX, 1)[0].rstrip()
    compact = compress_edit_success_output_for_memory(prefix)
    if compact.strip():
        return compact.strip()
    ln, sha = _parse_write_success_metadata_from_output(body)
    p = (rel or "file").replace("\\", "/").lstrip("/")
    if ln is not None and sha:
        return f"Edited `{p}` successfully ({ln} lines, sha256~{sha})."
    return f"Edited `{p}` successfully; file is complete on disk."

def _compact_snapshot_tool_feedback_for_inherit(
    content: str,
    *,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    read_max_lines: int,
    write_snapshot_inherit_max_lines: int,
    write_snapshot_inherit_max_chars: int,
    keep_auto_snapshot: bool = True,
) -> str | None:
    """Compact write/edit feedback that carries a canonical auto-snapshot."""
    snap = extract_write_auto_snapshot_block(content)
    if not snap:
        return None
    stripped = _strip_tool_feedback_trim_header(content)
    rel = _rel_path_for_write_compact(tool_args, stripped)
    if tool_name == "edit":
        base = _edit_success_feedback_for_inherit(stripped, rel)
    else:
        ln, sha = _parse_write_success_metadata_from_output(stripped)
        base = write_success_feedback_for_memory(rel, lines=ln, sha256_short=sha)
    if not keep_auto_snapshot:
        return (
            base
            + "\n[inherited] auto-snapshot omitted; superseded by a later "
            "solution.py snapshot or the current child fork instruction."
        )
    wlim, wch = _snapshot_inherit_limits(
        read_max_lines=read_max_lines,
        write_snapshot_inherit_max_lines=write_snapshot_inherit_max_lines,
        write_snapshot_inherit_max_chars=write_snapshot_inherit_max_chars,
    )
    return base + "\n\n" + truncate_inherited_snapshot_block(
        snap,
        max_lines=wlim,
        max_chars=wch,
    )

def trim_tool_feedback_for_llm_context(feedback: str, *, max_chars: int) -> str:
    """Trim tool feedback before storing it in chat memory.

    Ordinary tool output uses the legacy head+tail trim. Write auto-snapshots are different:
    they are deliberately injected as line-numbered source context so the next turn does not
    need to re-read ``solution.py``. Preserve the snapshot block intact and trim only the
    prefix around it; the snapshot itself is already bounded by ``write_auto_snapshot_max_chars``.
    """
    if max_chars <= 0 or not feedback or len(feedback) <= max_chars:
        return feedback

    orig_len = len(feedback)
    orig_lines = len(feedback.splitlines())

    if _AUTO_SNAPSHOT_PREFIX in feedback:
        idx = feedback.find(_AUTO_SNAPSHOT_PREFIX)
        prefix = feedback[:idx].rstrip()
        snapshot = feedback[idx:].lstrip()
        trimmed_prefix = prefix
        if len(trimmed_prefix) > max_chars:
            half = max(1, max_chars // 2)
            trimmed_prefix = (
                trimmed_prefix[:half].rstrip()
                + "\n...[prefix truncated; auto-snapshot preserved]...\n"
                + trimmed_prefix[-half:].lstrip()
            )
        header = (
            f"{_TOOL_FEEDBACK_TRIMMED_PREFIX} {orig_len} chars, "
            f"{orig_lines} lines; auto-snapshot preserved]"
        )
        body = (trimmed_prefix + "\n\n" + snapshot).strip()
        return header + "\n" + body

    half = max(1, max_chars // 2)
    return (
        f"{_TOOL_FEEDBACK_TRIMMED_PREFIX} {orig_len} chars, "
        f"{orig_lines} lines -> head+tail ~{max_chars} chars]\n"
        + feedback[:half]
        + "\n...[truncated]...\n"
        + feedback[-half:]
    )

def truncate_inherited_snapshot_block(
    snap: str,
    *,
    max_lines: int,
    max_chars: int,
) -> str:
    """Trim a snapshot block for clone-inherited memory."""
    if max_lines <= 0 or max_chars <= 0:
        return snap
    lines = snap.splitlines()
    if len(lines) <= max_lines + 6 and len(snap) <= max_chars:
        return snap
    # Keep intro + meta line(s) heuristically: first block until a line starting with "     1|"
    body_start = 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*1\|", ln):
            body_start = i
            break
    head = "\n".join(lines[:body_start])
    body_lines = lines[body_start:]
    if len(body_lines) <= max_lines:
        out = head + "\n" + "\n".join(body_lines)
    else:
        keep = body_lines[:max_lines]
        out = head + "\n" + "\n".join(keep) + "\n...[inherited snapshot truncated; use read tool]..."
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n...[truncated]...\n" + out[-(max_chars // 2) :]
    return out

def compress_read_output_for_memory(obs: str, *, max_lines: int = 80) -> str:
    """Truncate long read output keeping header + first *max_lines* content lines.

    Never raises: failures fall back to *obs* unchanged.
    """
    try:
        if max_lines <= 0:
            return obs
        lines = obs.splitlines()
        if not lines:
            return obs
        # Header line looks like: [path: N lines total, showing 1-200]
        first = lines[0]
        if not first.startswith("["):
            return obs
        body_lines = lines[1:]
        n = len(body_lines)
        if n <= max_lines:
            return obs
        kept = body_lines[:max_lines]
        return (
            first + "\n"
            + "\n".join(kept) + "\n"
            + f"...[read output truncated for LLM context: showing first {max_lines} of {n} content lines. "
            "Use offset parameter to read further.]"
        )
    except Exception:
        logger.debug("compress_read_output_for_memory failed", exc_info=True)
        return obs

def _read_args_has_explicit_range(args: dict[str, Any]) -> bool:
    return "offset" in args or "limit" in args

def _read_args_limit(args: dict[str, Any]) -> int:
    try:
        return int(args.get("limit") or 200)
    except (TypeError, ValueError):
        return 200

def _read_header_line(obs: str) -> str:
    return obs.splitlines()[0] if obs else ""

def _build_read_code_map_summary_for_memory(
    obs: str,
    *,
    rel: str,
    text: str,
    raw_id: str = "",
    max_chars: int = 2200,
) -> str:
    first = _read_header_line(obs)
    code_map = _build_source_code_map(rel, text, max_chars=max(800, max_chars - 500))
    lines = text.splitlines()
    parts = [
        first or f"[{rel}: {len(lines)} lines total]",
        (
            f"[read compressed: reducer=read_code_map_v2 "
            f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
        ),
        code_map,
        (
            "[full read body omitted from memory; use read with explicit offset/limit "
            "for the target function or line range.]"
        ),
    ]
    if raw_id:
        parts.append(f"[exact raw output: {raw_id}]")
    out = "\n".join(p for p in parts if p)
    if len(out) > max_chars:
        out = out[: max(1, max_chars - 28)].rstrip() + "\n...[read code-map truncated]"
    return out

def _build_snapshot_ref_output_for_memory(
    obs: str,
    *,
    rel: str,
    sha: str,
    raw_id: str = "",
    reason: str = "source body omitted to avoid duplicate context",
) -> str:
    first = _read_header_line(obs)
    lines = [
        first or f"[{rel}: current file snapshot available]",
        (
            f"[tool-output compressed: reducer=snapshot_ref_v1 "
            f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
        ),
        f"[see current snapshot: {rel} sha~{sha}; {reason}.]",
    ]
    if raw_id:
        lines.append(f"[exact raw output: {raw_id}]")
    return "\n".join(lines)

def _strip_bash_stderr_section_from_block(obs: str) -> str:
    """Drop trailing ``[stderr]`` section from a bash tool block (stdout-only for memory)."""
    lines = obs.splitlines()
    if len(lines) < 2 or not lines[0].startswith("[exit="):
        return obs
    body = "\n".join(lines[1:])
    marker = "\n[stderr]\n"
    idx = body.find(marker)
    if idx == -1:
        return obs
    new_body = body[:idx].rstrip()
    return f"{lines[0]}\n{new_body}" if new_body else lines[0]

def _bash_tool_feedback_header_is_success(feedback: str) -> bool:
    first = feedback.splitlines()[0] if feedback else ""
    if not first.startswith("[exit="):
        return False
    return first.startswith("[exit=0,") or first.startswith("[exit=0]")

def _bash_tool_feedback_is_failed_wrapped(feedback: str) -> bool:
    return bool(feedback and feedback.lstrip().startswith("Error:"))

def find_last_real_bare_solution_bash_tool_feedback(messages: Sequence[Message]) -> str | None:
    """Scan *messages* (chronological) for the last real ``bash`` tool result for a bare
    ``python3 solution.py``-class command.

    Skips synthetic :data:`inherited_fullrun_*` tool_call ids. Returns the tool **content**
    string as stored in memory (already stdout-only / verbatim for successful parent runs).
    """
    msgs = list(messages)
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if getattr(m, "role", None) != "tool":
            continue
        if (getattr(m, "name", None) or "") != "bash":
            continue
        tcid = str(getattr(m, "tool_call_id", "") or "")
        if tcid.startswith("inherited_fullrun_"):
            continue
        cmd = _bash_command_for_tool_message(msgs, i)
        if not cmd or not looks_like_bare_solution_run(cmd):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content or "")
        if not content.strip():
            continue
        return content
    return None

def clone_memory_has_successful_bare_solution_bash(messages: Sequence[Message]) -> bool:
    """True if the last bare ``solution.py`` bash in *messages* is a successful run (for skipping
    synthetic full-run tail inject).
    """
    fb = find_last_real_bare_solution_bash_tool_feedback(messages)
    if not fb:
        return False
    if _bash_tool_feedback_is_failed_wrapped(fb):
        return False
    return _bash_tool_feedback_header_is_success(fb)

def memory_has_inherited_fullrun_synthetic_round(messages: Sequence[Message]) -> bool:
    """True if a fork-ready synthetic full-run recap exists (``tool_call_id`` ``inherited_fullrun_*``).

    Used for idempotent parent persistence and to avoid duplicate child fallback injects.
    """
    for m in reversed(list(messages)):
        if getattr(m, "role", None) != "tool":
            continue
        tid = str(getattr(m, "tool_call_id", "") or "")
        if tid.startswith("inherited_fullrun_"):
            return True
    return False

def _bash_command_from_tool_call(tc: Any) -> str | None:
    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
    if fn is None:
        return None
    name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
    if name != "bash":
        return None
    raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
    try:
        args = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(args, dict):
        return None
    return str(args.get("command") or "")

def _bash_command_for_tool_message(messages: list[Message], tool_idx: int) -> str | None:
    tcid = str(getattr(messages[tool_idx], "tool_call_id", "") or "")
    j = tool_idx - 1
    while j >= 0 and getattr(messages[j], "role", None) == "tool":
        j -= 1
    if j < 0:
        return None
    assistant = messages[j]
    if getattr(assistant, "role", None) != "assistant":
        return None
    for tc in getattr(assistant, "tool_calls", None) or []:
        if _tool_call_id_from_raw(tc) == tcid:
            return _bash_command_from_tool_call(tc)
    return None

def _tool_args_dict_for_tool_message(messages: list[Message], tool_idx: int) -> dict[str, Any] | None:
    tcid = str(getattr(messages[tool_idx], "tool_call_id", "") or "")
    j = tool_idx - 1
    while j >= 0 and getattr(messages[j], "role", None) == "tool":
        j -= 1
    if j < 0:
        return None
    assistant = messages[j]
    if getattr(assistant, "role", None) != "assistant":
        return None
    for tc in getattr(assistant, "tool_calls", None) or []:
        if _tool_call_id_from_raw(tc) != tcid:
            continue
        fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
        if fn is None:
            return None
        raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
        try:
            args = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            return None
        if isinstance(args, dict):
            return args
        return None
    return None

__all__ = tuple(name for name in globals() if not name.startswith("__"))
