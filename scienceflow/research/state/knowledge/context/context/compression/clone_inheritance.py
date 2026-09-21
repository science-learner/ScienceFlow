"""Memory-context responsibility: clone inheritance."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any

from inquirycraft.memory import Function, Message, ToolCall

from scienceflow.research.state.knowledge.context.context.results.base import (
    logger,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _parse_write_success_metadata_from_output,
    compress_edit_success_output_for_memory,
    write_success_feedback_for_memory,
)
from scienceflow.research.state.knowledge.context.context.results.tool_results import (
    _compact_snapshot_tool_feedback_for_inherit,
    _rel_path_for_write_compact,
    _snapshot_inherit_limits,
    _strip_bash_stderr_section_from_block,
    compress_read_output_for_memory,
    truncate_inherited_snapshot_block,
)
from scienceflow.research.state.knowledge.context.source_snapshot import (
    _AUTO_SNAPSHOT_PREFIX,
    extract_write_auto_snapshot_block,
)

_DEFAULT_CLONE_INHERIT_SIGNAL_PATTERNS: tuple[str, ...] = (
    r"Final Validation Score",
    r"Validation MCC",
    r"Best threshold",
    r"(?i)submission is valid",
    r"(?i)is_valid\s*=\s*True",
    r"Traceback",
    r"(?i)Error:",
    r"(?i)Killed",
    r"(?i)\bOOM\b",
    r"(?i)out of memory",
    r"(?i)cuda out of memory",
)

def _compile_clone_inherit_signal_patterns(raw: list[str] | None) -> list[re.Pattern[str]]:
    src = list(raw) if raw else list(_DEFAULT_CLONE_INHERIT_SIGNAL_PATTERNS)
    out: list[re.Pattern[str]] = []
    for s in src:
        try:
            out.append(re.compile(str(s)))
        except re.error:
            logger.warning("Invalid clone_inherit_signal_patterns entry %r; skipped", s)
    return out

def slice_clone_minimal_inherited_messages(msgs: list[Message]) -> list[Message]:
    """Keep leading user block + tail from parent node's last assistant (inclusive).

    Drops middle transcript for token savings while preserving the original task user
    turn(s) and the latest model reasoning plus any following tool results.
    """
    if len(msgs) < 2:
        return msgs
    j = 0
    while j < len(msgs) and getattr(msgs[j], "role", "") == "user":
        j += 1
    leading = msgs[:j]
    last_a: int | None = None
    for i in range(len(msgs) - 1, -1, -1):
        if getattr(msgs[i], "role", "") == "assistant":
            last_a = i
            break
    if last_a is None:
        return msgs
    if last_a < j:
        return msgs
    tail = msgs[last_a:]
    if not leading:
        return tail
    return leading + tail

def _is_inherited_write_path_solution_py(path: str) -> bool:
    """True when a write tool targets ``solution.py`` (any relative path ending with it)."""
    p = str(path or "").replace("\\", "/").strip().lstrip("/")
    if not p:
        return False
    return p.endswith("solution.py") or p.split("/")[-1] == "solution.py"

def _norm_code_extensions_for_inherit(raw: Any = None) -> tuple[str, ...]:
    vals = raw if isinstance(raw, (list, tuple)) else (".py",)
    out = tuple(
        e if e.startswith(".") else f".{e}"
        for e in (str(x).strip().lower() for x in vals)
        if e
    )
    return out or (".py",)

def _is_inherited_code_file_path(
    path: str,
    code_extensions: tuple[str, ...] = (".py",),
) -> bool:
    p = str(path or "").replace("\\", "/").strip().lstrip("/")
    return bool(p and PurePosixPath(p).suffix.lower() in code_extensions)

def _snapshot_path_from_auto_snapshot_content(content: str) -> str:
    if not content or _AUTO_SNAPSHOT_PREFIX not in content:
        return ""
    tail = content.split(_AUTO_SNAPSHOT_PREFIX, 1)[1]
    first = tail.split("\n", 1)[0].strip()
    if first.endswith("]"):
        first = first[:-1].strip()
    return first.replace("\\", "/").lstrip("/")

def _read_path_from_content_or_args(content: str, tool_args: dict[str, Any] | None) -> str:
    if tool_args:
        rel = str(tool_args.get("path") or "").replace("\\", "/").strip().lstrip("/")
        if rel:
            return rel
    first = content.splitlines()[0] if content else ""
    m = re.match(r"^\[([^:\]]+):\s+\d+\s+lines total", first)
    if m:
        return m.group(1).replace("\\", "/").strip().lstrip("/")
    return ""

def _is_read_of_solution_py(content: str, tool_args: dict[str, Any] | None) -> bool:
    rel = _read_path_from_content_or_args(content, tool_args)
    base = rel.split("/")[-1] if rel else ""
    if base == "solution.py" or rel.endswith("/solution.py"):
        return True
    if not content:
        return False
    first = content.splitlines()[0] if content else ""
    if "solution.py" in first and ("lines total" in first or "showing" in first):
        return True
    head = content[:2000]
    return "solution.py" in head and ("lines total" in head or "showing" in head)

def _is_read_of_code_file(
    content: str,
    tool_args: dict[str, Any] | None,
    *,
    code_extensions: tuple[str, ...],
) -> bool:
    rel = _read_path_from_content_or_args(content, tool_args)
    if rel:
        return _is_inherited_code_file_path(rel, code_extensions)
    return _is_read_of_solution_py(content, tool_args)

def _lines_matching_signal_patterns(
    lines: list[str],
    patterns: list[re.Pattern[str]],
) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if ln in seen:
            continue
        if any(p.search(ln) for p in patterns):
            picked.append(ln)
            seen.add(ln)
    return picked

def _truncate_middle_on_inherit(s: str, max_total: int) -> str:
    if max_total <= 0 or len(s) <= max_total:
        return s
    half = max_total // 2
    omitted = len(s) - max_total
    return (
        f"{s[:half]}\n…[args truncated on inherit: {omitted} chars]…\n{s[len(s) - half :]}"
    )

def _bash_exit_header_success(first: str) -> bool:
    return first.startswith("[exit=0]") or first.startswith("[exit=0,")

def _compress_inherited_bash_output(
    obs: str,
    *,
    tail_lines: int,
    patterns: list[re.Pattern[str]],
    repl_mode: bool = False,
) -> str:
    """Compress a bash tool output for inherited memory.

    Default (``repl_mode=False``): errors are kept verbatim (< 32 kB) so the *same* node can
    debug them; only successful long outputs are tail-truncated with signal-line rescue.

    When ``repl_mode=True`` (cross-node REPL-style inheritance): both errors *and* successes are
    tail-truncated with signal-line rescue. Old verbose tracebacks from a previous node's approach
    are not useful to a child node that will use a different method — only the key error type and
    metric lines matter (signal-to-noise improvement).
    """
    if not obs or not isinstance(obs, str):
        return obs
    if not repl_mode:
        low = obs.lower()
        if "traceback" in low or "error:" in obs[:4000]:
            if len(obs) > 32000:
                return obs[:16000] + "\n...[truncated]...\n" + obs[-16000:]
            return obs
        if obs.lstrip().startswith("Error:"):
            if len(obs) > 32000:
                return obs[:16000] + "\n...[truncated]...\n" + obs[-16000:]
            return obs
    stripped = _strip_bash_stderr_section_from_block(obs)
    lines = stripped.splitlines()
    if not lines:
        return obs
    first = lines[0]
    if not first.startswith("[exit="):
        if len(stripped) > 12000:
            return stripped[:6000] + "\n...[truncated]...\n" + stripped[-6000:]
        return stripped
    if not repl_mode and not _bash_exit_header_success(first):
        if len(stripped) > 32000:
            return stripped[:16000] + "\n...[truncated]...\n" + stripped[-16000:]
        return stripped
    if "[bash:" in first and "showing last" in first:
        return stripped
    body = lines[1:]
    if tail_lines <= 0 or len(body) <= tail_lines:
        return stripped
    tail = body[-tail_lines:]
    dropped = body[:-tail_lines]
    extra = _lines_matching_signal_patterns(dropped, patterns)
    tail_set = set(tail)
    extra = [ln for ln in extra if ln not in tail_set]
    parts: list[str] = [first]
    if extra:
        parts.append("---")
        parts.append("[preserved signal lines from truncated bash stdout]")
        parts.extend(extra)
    parts.extend(tail)
    return "\n".join(parts)

def _compress_inherited_read_output(
    content: str,
    *,
    max_lines: int,
    patterns: list[re.Pattern[str]],
) -> str:
    if not content or max_lines <= 0:
        return content
    lines = content.splitlines()
    if not lines:
        return content
    first = lines[0]
    if not first.startswith("["):
        return content
    body_lines = lines[1:]
    if len(body_lines) <= max_lines:
        return content
    dropped = body_lines[max_lines:]
    extra = _lines_matching_signal_patterns(dropped, patterns)
    base = compress_read_output_for_memory(content, max_lines=max_lines)
    if not extra:
        return base
    tail_set = set(base.splitlines())
    extra = [ln for ln in extra if ln not in tail_set]
    if not extra:
        return base
    return base + "\n---\n[preserved signal lines from truncated read]\n" + "\n".join(extra)

def _write_tool_feedback_is_success(content: str) -> bool:
    if not content or not isinstance(content, str):
        return False
    if content.startswith("Error:") or content.lstrip().startswith("Error:"):
        return False
    first = content.split("\n", 1)[0]
    if first.startswith("File `") and (
        "updated." in first or "written successfully" in first
    ):
        return True
    if first.startswith("Written ") and "lines" in first:
        return True
    return False

def compress_inherited_tool_message_content(
    content: str,
    *,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    read_max_lines: int,
    bash_tail_lines: int,
    omit_solution_py_reads: bool,
    write_compact: bool,
    edit_compact: bool,
    patterns: list[re.Pattern[str]],
    write_snapshot_inherit_max_lines: int = 0,
    write_snapshot_inherit_max_chars: int = 0,
    keep_auto_snapshot: bool = True,
    code_extensions: tuple[str, ...] = (".py",),
) -> str:
    """Shorten one tool message body when loading clone-inherited memory."""
    if not content or not isinstance(content, str):
        return content
    if tool_name == "read":
        if omit_solution_py_reads and _is_read_of_code_file(
            content,
            tool_args,
            code_extensions=code_extensions,
        ):
            rel = _read_path_from_content_or_args(content, tool_args) or "code file"
            return (
                f"[inherited] {rel} read omitted — file is unchanged on disk; "
                "use the `read` tool with `offset`/`limit` to fetch a fresh range only "
                "if you need to inspect specific lines."
            )
        return _compress_inherited_read_output(
            content, max_lines=read_max_lines, patterns=patterns,
        )
    if tool_name == "bash":
        return _compress_inherited_bash_output(
            content, tail_lines=bash_tail_lines, patterns=patterns,
        )
    if tool_name in ("write", "edit") and (
        (tool_name == "write" and write_compact)
        or (tool_name == "edit" and edit_compact)
    ):
        compacted = _compact_snapshot_tool_feedback_for_inherit(
            content,
            tool_name=tool_name,
            tool_args=tool_args,
            read_max_lines=read_max_lines,
            write_snapshot_inherit_max_lines=write_snapshot_inherit_max_lines,
            write_snapshot_inherit_max_chars=write_snapshot_inherit_max_chars,
            keep_auto_snapshot=keep_auto_snapshot,
        )
        if compacted is not None:
            return compacted
    if tool_name == "write" and write_compact and _write_tool_feedback_is_success(content):
        rel = _rel_path_for_write_compact(tool_args, content)
        ln, sha = _parse_write_success_metadata_from_output(content)
        base = write_success_feedback_for_memory(rel, lines=ln, sha256_short=sha)
        snap = extract_write_auto_snapshot_block(content)
        if snap:
            wlim, wch = _snapshot_inherit_limits(
                read_max_lines=read_max_lines,
                write_snapshot_inherit_max_lines=write_snapshot_inherit_max_lines,
                write_snapshot_inherit_max_chars=write_snapshot_inherit_max_chars,
            )
            return base + "\n\n" + truncate_inherited_snapshot_block(
                snap, max_lines=wlim, max_chars=wch,
            )
        return base
    if tool_name == "edit" and edit_compact:
        if content.startswith("Edited ") and "1 replacement OK." in content.split("\n", 1)[0]:
            return compress_edit_success_output_for_memory(content)
        return content
    return content

def _estimate_transcript_chars(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        c = m.content
        if isinstance(c, str):
            total += len(c)
        elif c is not None:
            total += len(str(c))
        for tc in getattr(m, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is not None:
                raw = getattr(fn, "arguments", "") or ""
                total += len(raw)
    return total

def _compress_stale_fork_user_message_for_inherit(content: str) -> str:
    """Replace old fork-control prompts after EDA with a short marker."""
    if not content or not isinstance(content, str):
        return content
    stripped = content.strip()
    if stripped.startswith("## Trajectory branch history"):
        return (
            "[inherited fork-control prompt omitted: prior C_BACKTRACK trajectory "
            "selection. The current child receives a fresh fork instruction.]"
        )
    if stripped.startswith("## Reference peer trajectories"):
        return (
            "[inherited peer trajectory prompt omitted: prior B_ENSEMBLE peer list. "
            "The current child receives a bounded trajectory rollup / peer artifact summary.]"
        )
    if stripped.startswith("Switch to **deep** mode:"):
        return (
            "[inherited fork-control prompt omitted: prior B_DEEP full-data "
            "instruction. The current child receives a fresh fork instruction.]"
        )
    if stripped.startswith("**Phase override (ensemble):**"):
        return (
            "[inherited fork-control prompt omitted: prior B_ENSEMBLE fusion "
            "instruction. The current child receives a fresh fork instruction.]"
        )
    if stripped.startswith("[Guard] Detected repeated single-`read` rounds."):
        return (
            "[inherited guard note omitted: prior repeated-read warning. Current "
            "workspace state and tool guards remain active.]"
        )
    if stripped.startswith("[LNR] Detected repeated single-`read` rounds."):
        return (
            "[inherited guard note omitted: prior repeated-read warning. Current "
            "workspace state and tool guards remain active.]"
        )
    if stripped.startswith("[Guard] No meaningful progress detected for multiple rounds"):
        return (
            "[inherited guard note omitted: prior no-progress warning. Current "
            "child should use the latest workspace state.]"
        )
    if stripped.startswith("[System] No meaningful progress detected for multiple rounds"):
        return (
            "[inherited guard note omitted: prior no-progress warning. Current "
            "child should use the latest workspace state.]"
        )
    return content

def _compress_protected_user_message_for_inherit(content: str) -> str:
    """Compress inherited non-EDA user prose while preserving EDA tool outputs."""
    if not content or not isinstance(content, str):
        return content
    stripped = content.strip()
    if stripped.startswith("[fresh-workspace]") or stripped.startswith("[Guard] Fresh workspace note:"):
        return (
            "[inherited setup hint omitted: the parent started from a fresh workspace, "
            "but this child inherits the current workspace and solution state.]"
        )
    if stripped.startswith("Design a **gold** strong single pipeline for this node."):
        task_start = stripped.find("# Nomad2018")
        if task_start < 0:
            task_start = stripped.find("## Task objective")
        if task_start < 0:
            task_start = stripped.find("# ")
        task_brief = stripped[task_start:].strip() if task_start >= 0 else ""
        if task_brief:
            return (
                "## Inherited task brief (compressed)\n\n"
                "The original draft single-solution policy prose is omitted here; "
                "the current branch first-user prompt carries the active fork policy.\n\n"
                + task_brief
            )
    return content

def _is_inherited_auto_snapshot_tool_message(msg: Message) -> bool:
    if (getattr(msg, "role", "") or "") != "tool":
        return False
    if str(getattr(msg, "name", "") or "") not in ("write", "edit"):
        return False
    content = msg.content
    return isinstance(content, str) and _AUTO_SNAPSHOT_PREFIX in content

def _maybe_compress_assistant_tool_calls_for_inherit(
    msg: Message,
    *,
    max_total_chars: int,
    exempt_write_call_index: int | None = None,
) -> Message:
    if max_total_chars <= 0 or not msg.tool_calls:
        return msg
    new_tcs: list[ToolCall] = []
    changed = False
    for j, tc in enumerate(msg.tool_calls):
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
            new_tcs.append(tc)
            continue
        if not isinstance(args, dict):
            new_tcs.append(tc)
            continue
        args_changed = False
        if name == "write" and j == exempt_write_call_index:
            new_tcs.append(tc)
            continue
        if name == "write" and isinstance(args.get("content"), str):
            new_c = _truncate_middle_on_inherit(args["content"], max_total_chars)
            if new_c != args["content"]:
                args["content"] = new_c
                args_changed = True
        elif name == "edit":
            for key in ("new_string", "old_string", "new_str", "old_str"):
                if isinstance(args.get(key), str):
                    new_s = _truncate_middle_on_inherit(args[key], max_total_chars)
                    if new_s != args[key]:
                        args[key] = new_s
                        args_changed = True
        if args_changed:
            changed = True
            try:
                raw = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                new_tcs.append(tc)
                continue
            new_tcs.append(
                tc.model_copy(
                    update={"function": Function(name=name, arguments=raw)},
                ),
            )
        else:
            new_tcs.append(tc)
    if not changed:
        return msg
    return msg.model_copy(update={"tool_calls": new_tcs})

_READ_ONLY_BASH_CMDS: frozenset[str] = frozenset(
    # ``sed`` is common in read-only ``cat … | sed -n 'a,bp'`` pipelines; ``-i`` is rejected below.
    {"cat", "head", "tail", "less", "more", "grep", "egrep", "rg", "sed"},
)

_BASH_WRITE_REDIRECT_RE = re.compile(r"(?:^|\s)(?:\d?>|&>|>>)")

__all__ = tuple(name for name in globals() if not name.startswith("__"))
