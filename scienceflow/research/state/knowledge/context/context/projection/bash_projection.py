"""Memory-context responsibility: bash projection."""

from __future__ import annotations

import re
import shlex
from typing import Any

from scienceflow.runtime.core.process.commands import looks_like_bare_solution_run
from scienceflow.research.state.knowledge.context.context.compression.clone_inheritance import (
    _BASH_WRITE_REDIRECT_RE,
    _READ_ONLY_BASH_CMDS,
)

def _bash_leading_token(segment: str) -> str:
    """First command token in a segment, after stripping ``VAR=value`` prefixes."""
    parts = segment.strip().split()
    i = 0
    while i < len(parts):
        tok = parts[i]
        if "=" in tok:
            key, _, _ = tok.partition("=")
            if key and key.replace("_", "").isupper():
                i += 1
                continue
        return tok
    return ""

def _bash_first_command_words(cmd: str) -> list[str]:
    """Return first shell segment words after stripping leading env assignments."""
    s = (cmd or "").strip()
    if not s:
        return []
    first_segment = re.split(r"\s*(?:\|\||&&|\||;|&)\s*", s, maxsplit=1)[0].strip()
    if not first_segment:
        return []
    try:
        words = shlex.split(first_segment)
    except ValueError:
        words = first_segment.split()
    out = list(words)
    while out:
        tok = out[0]
        if tok == "env":
            out.pop(0)
            continue
        if "=" in tok:
            key, _, _ = tok.partition("=")
            if key and key.replace("_", "").isupper():
                out.pop(0)
                continue
        break
    return out

def _bash_command_looks_like_test(cmd: str) -> bool:
    words = _bash_first_command_words(cmd)
    if not words:
        return False
    first = words[0]
    if first == "pytest":
        return True
    if first in {"python", "python3"} and len(words) >= 3:
        return words[1] == "-m" and words[2] == "pytest"
    if first == "uv" and len(words) >= 3 and words[1] == "run":
        rest = words[2:]
        if not rest:
            return False
        if rest[0] == "pytest":
            return True
        return len(rest) >= 3 and rest[0] in {"python", "python3"} and rest[1] == "-m" and rest[2] == "pytest"
    return False

def _bash_command_looks_like_install(cmd: str) -> bool:
    words = _bash_first_command_words(cmd)
    if len(words) < 2:
        return False
    first = words[0]
    if first in {"pip", "pip3"}:
        return words[1] == "install"
    if first == "uv":
        return words[1] in {"sync", "lock"} or (
            len(words) >= 3 and words[1] == "pip" and words[2] == "install"
        )
    if first == "npm":
        return words[1] == "install"
    if first == "yarn":
        return words[1] in {"install", "add"}
    if first == "apt":
        return words[1] == "install"
    return False

def _bash_is_pure_read_only(cmd: str) -> bool:
    """True if every ``||``, ``&&``, ``|``, ``;``, ``&`` segment starts with a whitelisted command."""
    if not cmd.strip():
        return False
    segments = re.split(r"\s*(?:\|\||&&|\||;|&)\s*", cmd.strip())
    for seg in segments:
        if not seg.strip():
            continue
        seg_for_redirect = (
            seg.replace("2>&1", "")
            .replace("1>&2", "")
            .replace("2> /dev/null", "")
            .replace("2>/dev/null", "")
        )
        if "<<" in seg_for_redirect or _BASH_WRITE_REDIRECT_RE.search(seg_for_redirect):
            return False
        first = _bash_leading_token(seg)
        if not first or first not in _READ_ONLY_BASH_CMDS:
            return False
        if first == "sed" and re.search(r"(?<![A-Za-z0-9_])-i", seg):
            # in-place / backup suffix forms are not read-only
            return False
    return True

def _bash_success_tail_lines_for_command(
    cmd: str,
    *,
    fallback: int,
    solution: int,
    test: int,
    readonly: int,
    install: int,
) -> int:
    if looks_like_bare_solution_run(cmd):
        return min(int(fallback), int(solution))
    if _bash_command_looks_like_test(cmd):
        return int(test)
    if _bash_is_pure_read_only(cmd):
        return int(readonly)
    if _bash_command_looks_like_install(cmd):
        return int(install)
    return int(fallback)

_BASH_SOURCE_DUMP_CMDS: frozenset[str] = frozenset({"cat", "head", "tail", "sed"})

_PYPATH_IN_BASH_CMD = re.compile(
    r"(?:^|[\s/])([A-Za-z0-9_.-]+\.py)\b",
)

def bash_command_dumps_python_source(cmd: str) -> bool:
    """True if *cmd* is read-only and uses cat/head/tail/sed to show ``*.py`` / known entry files.

    Used to discourage repeated raw source dumps; targeted bash snippets remain acceptable in
    bash-first mode, and ``read`` is only a fallback for exact line-numbered anchors.
    """
    if not _bash_is_pure_read_only(cmd):
        return False
    # Any ``*.py`` path anywhere in the command (pipelines may only name the file in the first segment).
    if not _PYPATH_IN_BASH_CMD.search(cmd):
        return False
    segments = re.split(r"\s*(?:\|\||&&|\||;|&)\s*", cmd.strip())
    for seg in segments:
        s = seg.strip()
        if not s:
            continue
        first = _bash_leading_token(s)
        if first in _BASH_SOURCE_DUMP_CMDS:
            return True
    return False

_BASH_READ_PY_SOURCE_COACHING = (
    "\n\n[Guard] Bash source dump detected. Keep source inspection compact: use one "
    "bounded snippet/search command, or targeted `read` tool only when exact line-numbered "
    "anchors are needed. Do not spend rounds dumping code."
)

def _tool_call_signature(tool_name: str, args: dict[str, Any]) -> tuple[str, str] | None:
    """Stable signature for repeat-detection (read-only ``bash`` / ``read`` only)."""
    if tool_name == "bash":
        cmd = " ".join(str(args.get("command") or "").split())
        if not cmd or not _bash_is_pure_read_only(cmd):
            return None
        return ("bash", cmd)
    if tool_name == "read":
        path = str(args.get("path") or "").replace("\\", "/").strip().lstrip("/")
        if not path:
            return None
        offset = args.get("offset")
        limit = args.get("limit")
        return ("read", f"{path}|{offset}|{limit}")
    return None

__all__ = tuple(name for name in globals() if not name.startswith("__"))
