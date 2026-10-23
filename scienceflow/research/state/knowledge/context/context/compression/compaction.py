"""Memory-context responsibility: compaction."""

from __future__ import annotations

import re

from inquirycraft.memory import Memory, Message

from scienceflow.runtime.core.process.commands import looks_like_bare_solution_run
from scienceflow.research.state.knowledge.context.context.results.base import (
    COMPACTED_CONVERSATION_SUMMARY_MARKER,
    logger,
)
from scienceflow.research.state.knowledge.context.context.projection.bash_projection import (
    _bash_command_looks_like_install,
    _bash_command_looks_like_test,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _all_messages,
    compress_edit_success_output_for_memory,
)
from scienceflow.research.state.knowledge.context.context.results.tool_results import compress_read_output_for_memory
from scienceflow.runtime.observability.interaction_log import (
    collapse_consecutive_repeated_lines_in_text,
)

_BASH_TRAINING_SIGNAL_RE = re.compile(
    r"(score|metric|rmse|rmsle|mae|mse|accuracy|acc\b|auc|loss|epoch|fold|best|"
    r"valid|validation|final|saved|written|submission|artifact|model|csv|pkl)",
    re.IGNORECASE,
)

_BASH_PYTEST_SUMMARY_RE = re.compile(
    r"(=+\s*.*(?:passed|failed|errors?|warnings?|skipped|xfailed|xpassed).*=+|"
    r"\b\d+\s+(?:passed|failed|errors?|warnings?|skipped)\b)",
    re.IGNORECASE,
)

_BASH_PYTEST_FAILURE_RE = re.compile(
    r"(^FAILED\s+|^ERROR\s+|FAILURES|ERRORS|short test summary info|"
    r"\bFAILED\b|\bERROR\b|assert\s|^E\s+|^>\s+|Traceback|File \")",
)

_BASH_INSTALL_SIGNAL_RE = re.compile(
    r"(successfully installed|installed|resolved|prepared|done|success|error|failed|failure)",
    re.IGNORECASE,
)

def _dedupe_recent_nonempty_lines(lines: list[str], *, limit: int) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for line in reversed(lines):
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        picked.append(line)
        if len(picked) >= limit:
            break
    picked.reverse()
    return picked

def _dedupe_preserve_order(lines: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= limit:
            break
    return out

_PY_TRACEBACK_START = "Traceback (most recent call last):"

_PY_EXCEPTION_LINE_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*\.)*[A-Za-z_]\w*(?:Error|Exception|Warning):\s+.+"
)

def _extract_recent_python_error_block(lines: list[str], *, max_lines: int = 12) -> list[str]:
    """Return the most recent Python traceback/exception block from bash output."""
    if not lines:
        return []
    start: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        if _PY_TRACEBACK_START in lines[i]:
            start = i
            break
    if start is not None:
        end = min(len(lines), start + max_lines)
        return lines[start:end]

    for i in range(len(lines) - 1, -1, -1):
        if _PY_EXCEPTION_LINE_RE.search(lines[i]):
            lo = max(0, i - 3)
            hi = min(len(lines), i + 2)
            return lines[lo:hi]
    return []

def _bash_memory_marker(first: str, reducer: str, obs: str) -> list[str]:
    return [
        first,
        (
            f"[tool-output compressed: reducer={reducer} "
            f"raw_chars={len(obs)} raw_lines={len(obs.splitlines())}]"
        ),
    ]

def _compress_training_bash_output_for_memory(obs: str, *, tail_lines: int = 5) -> str:
    lines = obs.splitlines()
    if not lines:
        return obs
    first, body = lines[0], lines[1:]
    if len(body) <= 20:
        return obs
    tail_n = max(1, int(tail_lines or 5))
    tail = body[-tail_n:]
    tail_keys = {ln.strip() for ln in tail}
    error_block = _extract_recent_python_error_block(body)
    error_keys = {ln.strip() for ln in error_block}
    signals = _dedupe_recent_nonempty_lines(
        [ln for ln in body if _BASH_TRAINING_SIGNAL_RE.search(ln)],
        limit=15,
    )
    signals = [
        ln for ln in signals
        if ln.strip() not in tail_keys and ln.strip() not in error_keys
    ]
    parts = _bash_memory_marker(first, "bash_training_signal_v2", obs)
    if error_block:
        parts.append("--- preserved error/traceback lines ---")
        parts.extend(error_block)
    if signals:
        parts.append("--- preserved training signal lines ---")
        parts.extend(signals)
    parts.append(f"--- tail last {tail_n} lines ---")
    parts.extend(tail)
    return "\n".join(parts)

def _compress_pytest_bash_output_for_memory(obs: str) -> str:
    lines = obs.splitlines()
    if not lines:
        return obs
    if len(obs) <= 3600 and len(lines) <= 80:
        return obs
    first, body = lines[0], lines[1:]
    picked: list[str] = []
    for ln in body:
        if " PASSED " in f" {ln} " and not _BASH_PYTEST_SUMMARY_RE.search(ln):
            continue
        if _BASH_PYTEST_SUMMARY_RE.search(ln) or _BASH_PYTEST_FAILURE_RE.search(ln):
            picked.append(ln)
    picked = _dedupe_preserve_order([ln for ln in picked if ln.strip()], limit=80)
    if not picked:
        picked = body[-8:]
    parts = _bash_memory_marker(first, "bash_pytest_summary_v2", obs)
    parts.extend(picked[:80])
    return "\n".join(parts)

def _compress_install_bash_output_for_memory(obs: str) -> str:
    lines = obs.splitlines()
    if not lines:
        return obs
    first, body = lines[0], lines[1:]
    if len(body) <= 5:
        return obs
    signals = _dedupe_recent_nonempty_lines(
        [ln for ln in body if _BASH_INSTALL_SIGNAL_RE.search(ln)],
        limit=4,
    )
    tail = body[-1:] if body else []
    kept = _dedupe_preserve_order([*signals, *tail], limit=5)
    parts = _bash_memory_marker(first, "bash_install_summary_v2", obs)
    parts.extend(kept)
    return "\n".join(parts)

def compress_bash_tool_output_for_memory(
    obs: str,
    *,
    tail_lines: int = 20,
    command: str = "",
    dedup_enabled: bool = True,
    dedup_min_repeat: int = 3,
    dedup_summary_prefix: str = "[log-dedup]",
) -> str:
    """Keep bash header + last *tail_lines* lines of stdout/stderr (success paths only).

    When *dedup_enabled*, consecutive duplicate lines are collapsed before tail truncation.

    Never raises: failures fall back to *obs* unchanged.
    """
    try:
        if tail_lines <= 0:
            return obs
        obs_work = obs
        if dedup_enabled:
            obs_work = collapse_consecutive_repeated_lines_in_text(
                obs_work,
                min_repeat=int(dedup_min_repeat),
                summary_prefix=dedup_summary_prefix,
            )
        lines = obs_work.splitlines()
        if not lines:
            return obs
        first = lines[0]
        if not first.startswith("[exit="):
            return obs_work
        if looks_like_bare_solution_run(command):
            return _compress_training_bash_output_for_memory(
                obs_work,
                tail_lines=tail_lines,
            )
        if _bash_command_looks_like_test(command):
            return _compress_pytest_bash_output_for_memory(obs_work)
        if _bash_command_looks_like_install(command):
            return _compress_install_bash_output_for_memory(obs_work)
        body_lines = lines[1:]
        n = len(body_lines)
        if n <= tail_lines:
            return obs_work
        tail = body_lines[-tail_lines:]
        # Do not use str.strip("[]") — that strips any leading/trailing [ or ] chars, not a pair.
        meta = (
            first[1:-1]
            if len(first) >= 2 and first.startswith("[") and first.endswith("]")
            else first
        )
        return (
            f"[{meta}] [bash: {n} lines total, showing last {tail_lines}]\n"
            + "\n".join(tail)
        )
    except Exception:
        logger.debug("compress_bash_tool_output_for_memory failed", exc_info=True)
        return obs

def _split_recent_turn_suffix(
    messages: list[Message],
    *,
    keep_recent_turns: int,
) -> tuple[list[Message], list[Message]]:
    """Split messages into (older, recent suffix) by assistant-turn count."""
    if keep_recent_turns <= 0 or not messages:
        return list(messages), []
    idx = len(messages)
    turns = 0
    while idx > 0 and turns < keep_recent_turns:
        idx -= 1
        if messages[idx].role == "assistant":
            turns += 1
    return messages[:idx], messages[idx:]

_TASK_DESC_BODY_RE = re.compile(r"(?m)^##\s+Task description\b")

MSG0_BODY_TRUNCATED_PLACEHOLDER = "[task description body truncated for context]"

def _split_msg0_head_body(text: str) -> tuple[str, str]:
    """Split the leading user message into (head, body) at ``## Task description``.

    Head is everything strictly before the first line that starts with
    ``## Task description``; body is that line plus everything after it.
    When the anchor is absent (older tasks, custom callers, or messages that
    are not the lite-task pin), the whole text is treated as ``head`` and
    body is empty — preserving the existing behavior of leaving such
    messages untouched.
    """
    if not text:
        return "", ""
    m = _TASK_DESC_BODY_RE.search(text)
    if not m:
        return text, ""
    head = text[: m.start()].rstrip("\n")
    body = text[m.start():]
    return head, body

def _compress_tool_output_for_mechanical_compact(content: str) -> str:
    """Deterministic compression for old tool outputs (mid-run, no LLM)."""
    try:
        if not content:
            return content
        low = content.lower()
        first = content.splitlines()[0] if content else ""

        # Keep failures/errors intact for debugging.
        if "traceback" in low or "error:" in low[:2000] or "non-zero exit code" in low[:2000]:
            return content

        if first.startswith("[exit="):
            return compress_bash_tool_output_for_memory(content, tail_lines=3)

        if first.startswith("[") and ("lines total" in first or "showing" in first):
            return compress_read_output_for_memory(content, max_lines=5)

        if content.startswith("Edited ") and "1 replacement OK." in content:
            return compress_edit_success_output_for_memory(content)

        if "wrote " in low and "bytes" in low[:200]:
            lines = content.splitlines()
            if not lines:
                return content
            keep = [lines[0]]
            for ln in lines[1:]:
                if "sha256~" in ln:
                    keep.append(ln)
                    break
            return "\n".join(keep)

        if first.startswith("[skill:") or "[skill:" in low[:120]:
            if len(content) <= 200:
                return content
            return content[:200].rstrip() + "\n...[skill output truncated for context]"

        if len(content) > 800:
            return content[:800].rstrip() + "\n...[tool output truncated for context]"
        return content
    except Exception:
        logger.debug("_compress_tool_output_for_mechanical_compact failed", exc_info=True)
        return content

def _format_message_excerpt_for_compact(message: Message) -> str:
    """Build a longer excerpt per role for compact input (was fixed 500 chars)."""
    text = message.content if isinstance(message.content, str) else str(message.content or "")
    role = getattr(message, "role", "") or ""

    if role == "tool":
        # Prefer command + head of output for bash-like tool results
        lines = text.splitlines()
        head_n = 40
        if len(lines) <= head_n:
            body = text
        else:
            body = "\n".join(lines[:head_n]) + f"\n... ({len(lines)} lines total) ..."
        return body[:6000]

    # user / assistant / system: more room for code and reasoning
    cap = 2000
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[truncated, total {len(text)} chars]"

def extract_compacted_summary_body(memory: Memory) -> str | None:
    """Return text after :data:`COMPACTED_CONVERSATION_SUMMARY_MARKER` in a post-compact message.

    New compacts store this as ``role=user`` (Gemini-compatible handoff); legacy workspaces
    may still have ``role=system``.
    """
    marker = COMPACTED_CONVERSATION_SUMMARY_MARKER
    for m in _all_messages(memory):
        if getattr(m, "role", None) not in ("system", "user"):
            continue
        c = m.content if isinstance(m.content, str) else str(m.content or "")
        if marker not in c:
            continue
        idx = c.find(marker)
        rest = c[idx + len(marker) :].lstrip()
        if rest.startswith("\n"):
            rest = rest[1:]
        rest = rest.strip()
        return rest or None
    return None

__all__ = tuple(name for name in globals() if not name.startswith("__"))
