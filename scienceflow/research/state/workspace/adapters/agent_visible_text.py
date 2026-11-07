# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Project workspace text into the form the agent is allowed to see.

This module owns two orthogonal transforms:

* :func:`rewrite_agent_visible_paths` rewrites host/workspace absolute paths to
  stable workspace-relative forms.  It never hides anything.
* :func:`redact_hidden_paths` removes references to control-plane files and
  directories.  It never emits path placeholders.

The two transforms are intentionally separate so tool results can be filtered
before any hidden mention is rewritten, and so prompt/tool-schema surfaces can
be path-rewritten without turning ordinary prose (for example the word
``logs``) into a placeholder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scienceflow.research.state.workspace.session.agent_contracts import (
    _HOST_ABSOLUTE_VISIBLE_PATH_RE,
    _WSP_ABSOLUTE_VISIBLE_PREFIX_RE,
)

MatchContext = Literal["command", "listing", "prose", "tool_output"]
Surface = Literal["assistant", "log", "prompt", "tool_output"]

# A path token starts at a path boundary, may be prefixed with ``./`` or ``/``,
# and may contain multiple ``/`` separated components.  It deliberately excludes
# surrounding quotes/punctuation so a comma-separated listing is tokenized too.
_PATH_TOKEN_RE = re.compile(
    r"(?<![\w.-])(?:\./|/)?[\w.\-]+(?:/[\w.\-]+)*/?(?![\w./-])"
)
_FILE_DIR_RE = re.compile(r"^(?:FILE|DIR)\s+")
_LS_LONG_RE = re.compile(r"^[-dlpscb][rwxStTs-]{9}[@+.]?\s")
_LISTING_COMMAND_RE = re.compile(
    r"(?:^|[;&|])\s*(?:ls|find|tree|du|stat|rg\s+--files)\b"
)
_EMPTY_AFTER_REDACT_RE = re.compile(r"^[\s,;|│├└─]*$")
_LEGACY_PLACEHOLDER_RE = re.compile(r"<hidden-control-(?:path|file)>")


@dataclass(frozen=True)
class HiddenPathMatch:
    """One hidden path token found inside a single output line."""

    start: int
    end: int
    token: str


@dataclass(frozen=True)
class HiddenPathPolicy:
    """Hidden workspace paths shared by access denial and output filtering."""

    prefixes: tuple[str, ...]
    filenames: tuple[str, ...]
    denied_prefixes: tuple[str, ...] = ()
    bare_names_require_path_context: bool = True

    @classmethod
    def from_agent(cls, agent: Any) -> HiddenPathPolicy:
        cached = getattr(agent, "_hidden_path_policy", None)
        if isinstance(cached, cls):
            return cached
        return cls(
            prefixes=_normalise_entries(
                getattr(agent, "_agent_hidden_workspace_path_prefixes", ())
            ),
            filenames=_normalise_filenames(
                getattr(agent, "_agent_hidden_workspace_filenames", ())
            ),
        )


def _normalise_entry(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip().strip("'\"")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.strip("/")


def _normalise_entries(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        raw = (raw,)
    try:
        iterator = iter(raw)
    except TypeError:
        iterator = iter(())
    out: list[str] = []
    for item in iterator:
        value = _normalise_entry(item)
        if value and value not in out:
            out.append(value)
    return tuple(out)


def _normalise_filenames(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        raw = (raw,)
    try:
        iterator = iter(raw)
    except TypeError:
        iterator = iter(())
    out: list[str] = []
    for item in iterator:
        name = Path(str(item or "").replace("\\", "/")).name.strip()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _matches_hidden_filename(parts: list[str], policy: HiddenPathPolicy) -> bool:
    return any(part in policy.filenames for part in parts)


def _matches_hidden_prefix(
    parts: list[str],
    policy: HiddenPathPolicy,
) -> bool:
    for prefix in policy.prefixes:
        prefix_parts = [part for part in prefix.split("/") if part]
        if not prefix_parts:
            continue
        if parts[: len(prefix_parts)] == prefix_parts:
            return True
    return False


def _line_is_listing_entry(line: str) -> bool:
    stripped = line.strip()
    return bool(
        _FILE_DIR_RE.match(stripped) or _LS_LONG_RE.match(stripped)
    )


def _tool_output_entry_tail(after: str) -> bool:
    """Return True when *after* looks like the tail of a path listing entry."""

    stripped = after.lstrip()
    if not stripped:
        return True
    if stripped.startswith(","):
        return True
    # ``grep -n`` style: ``logs/file:12:content``.
    return re.match(r"^:\d+(?::|$)", stripped) is not None


def _bare_name_context_allows(
    context: MatchContext,
    *,
    at_line_start: bool,
    listing_line: bool,
    in_quotes: bool,
    after: str,
) -> bool:
    if context in {"command", "listing"} or in_quotes:
        return True
    if context == "tool_output":
        return listing_line or (
            at_line_start and _tool_output_entry_tail(after)
        )
    return False


def _inside_angle_placeholder(line: str, start: int, end: int) -> bool:
    open_index = line.rfind("<", 0, start)
    close_index = line.find(">", end)
    if open_index < 0 or close_index < 0:
        return False
    return " " not in line[open_index:close_index]


def _inside_url(line: str, start: int) -> bool:
    # ``start`` may point at the second slash of ``scheme://`` because the
    # path token regex can consume the trailing slash separately.
    scheme_index = line.rfind("://", 0, start + 2)
    if scheme_index < 0:
        return False
    return not any(char.isspace() for char in line[scheme_index + 3 : start])


def _iter_hidden_matches_in_line(
    line: str,
    policy: HiddenPathPolicy,
    *,
    context: MatchContext,
) -> list[HiddenPathMatch]:
    matches: list[HiddenPathMatch] = []
    listing_line = _line_is_listing_entry(line)
    for regex_match in _PATH_TOKEN_RE.finditer(line):
        if _inside_angle_placeholder(
            line, regex_match.start(), regex_match.end()
        ):
            continue
        if _inside_url(line, regex_match.start()):
            continue
        token = regex_match.group(0)
        if token.startswith("/"):
            # Workspace-absolute paths are normalized to relative form before
            # matching.  A remaining leading-slash token is an absolute or
            # URL-ish path, not a workspace-relative hidden entry.
            continue
        norm = token.strip("/")
        if not norm or norm in {".", ".."}:
            continue

        parts = [part for part in norm.split("/") if part and part != "."]
        if not parts:
            continue

        before = line[: regex_match.start()]
        after = line[regex_match.end() :]
        at_line_start = not before.strip()
        in_quotes = bool(
            before
            and after
            and before[-1] in "\"'"
            and after[0] == before[-1]
        )

        if _matches_hidden_filename(parts, policy):
            matches.append(
                HiddenPathMatch(regex_match.start(), regex_match.end(), token)
            )
            continue

        if not _matches_hidden_prefix(parts, policy):
            continue

        # Bare generic names (``logs``, ``submission_snapshots``) are only
        # treated as paths when the surrounding context is path-shaped.
        if (
            policy.bare_names_require_path_context
            and "/" not in norm
            and not norm.startswith(".")
            and not _bare_name_context_allows(
                context,
                at_line_start=at_line_start,
                listing_line=listing_line,
                in_quotes=in_quotes,
                after=after,
            )
        ):
            continue

        matches.append(
            HiddenPathMatch(regex_match.start(), regex_match.end(), token)
        )
    return matches


def _remove_matches(line: str, matches: list[HiddenPathMatch]) -> str:
    pieces: list[str] = []
    last = 0
    for match in matches:
        pieces.append(line[last : match.start])
        last = match.end
    pieces.append(line[last:])
    text = "".join(pieces)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^\s*[,;]\s*", "", text)
    text = re.sub(r"\s*,\s*(?=,|$)", "", text)
    text = re.sub(r"([\[\(\{]\s*)[,;]\s*", r"\1", text)
    text = re.sub(r"\s*[,;]\s*([\]\)\}])", r"\1", text)
    text = text.rstrip()
    if matches and not line[: matches[0].start].strip():
        text = text.lstrip()
    return text


def _has_visible_entry(text: str) -> bool:
    return _EMPTY_AFTER_REDACT_RE.match(text) is None


def _path_prefixed_content_line(
    line: str, matches: list[HiddenPathMatch]
) -> bool:
    first = min(matches, key=lambda match: match.start)
    if line[: first.start].strip():
        return False
    return ":" in line[first.end :]


def _should_drop_line(
    line: str,
    matches: list[HiddenPathMatch],
    *,
    context: MatchContext,
) -> bool:
    # Free-form assistant/memory prose is always redacted inline; only tool
    # output and listing surfaces may drop an entire line.
    if context not in {"listing", "tool_output"}:
        return False

    stripped = line.strip()

    # ``FILE x`` / ``DIR y`` entries and ``ls -l`` rows are whole entries.
    if _FILE_DIR_RE.match(stripped) or _LS_LONG_RE.match(stripped):
        return True

    # A line such as ``logs/train.log:12:content`` is grep output for a hidden
    # file; drop the whole line rather than keeping the content fragment.
    if _path_prefixed_content_line(line, matches):
        return True

    # After removing hidden tokens, a pure listing row is empty; a mixed line
    # (``.git, README.md``) keeps its visible entries.
    remaining = _remove_matches(line, matches)
    return not _has_visible_entry(remaining)


def redact_hidden_paths(
    agent: Any,
    text: str,
    *,
    context: MatchContext,
    marker: bool = False,
) -> str:
    """Remove hidden paths from *text* without emitting placeholders."""

    if not isinstance(text, str) or not text:
        return text

    # Older sessions may already contain these in persisted memory.
    text = _LEGACY_PLACEHOLDER_RE.sub("", text)

    policy = HiddenPathPolicy.from_agent(agent)
    if not policy.prefixes and not policy.filenames:
        return text

    kept: list[str] = []
    dropped = False
    for line in text.splitlines():
        matches = _iter_hidden_matches_in_line(line, policy, context=context)
        if not matches:
            kept.append(line)
            continue
        if _should_drop_line(line, matches, context=context):
            dropped = True
            continue
        kept.append(_remove_matches(line, matches))

    if marker and dropped:
        kept.append("[hidden control file entries omitted]")
    return "\n".join(kept)


def _host_abs_placeholder(value: str) -> str:
    raw = str(value or "")
    name = Path(raw.rstrip("/ ")).name or raw.strip("/") or "path"
    return f"<host-path:{name}>"


def _rewrite_abs_prefix_to_label(text: str, root: Path, label: str) -> str:
    root_s = str(root).rstrip("/")
    if not root_s or root_s == "/" or root_s not in text:
        return text
    text = text.replace(root_s + "/", label.rstrip("/") + "/")
    text = text.replace(root_s, label.rstrip("/"))
    return text


def rewrite_agent_visible_paths(agent: Any, text: str) -> str:
    """Rewrite absolute paths only; never hide a hidden path here."""

    if not isinstance(text, str) or not text:
        return text

    workspace_dir = getattr(agent, "_workspace_dir", None)
    if workspace_dir is None:
        return text

    relative_mode = bool(getattr(agent, "_workspace_relative_path_mode", False))
    teleport_mode = str(getattr(agent, "_teleport_mode", "off") or "off")
    if not (relative_mode or teleport_mode != "off"):
        return text

    from scienceflow.research.state.knowledge.memory.records.agent_records import (
        rewrite_workspace_abs_to_relative,
    )

    out = rewrite_workspace_abs_to_relative(text, str(workspace_dir))
    out = _WSP_ABSOLUTE_VISIBLE_PREFIX_RE.sub("./", out)

    dataset_link = Path(workspace_dir) / "dataset"
    try:
        if dataset_link.exists():
            out = _rewrite_abs_prefix_to_label(
                out,
                dataset_link.resolve(),
                "dataset",
            )
    except OSError:
        pass

    if relative_mode:
        out = _HOST_ABSOLUTE_VISIBLE_PATH_RE.sub(
            lambda match: _host_abs_placeholder(match.group(0)),
            out,
        )
    return out


def project_agent_visible_text(
    agent: Any,
    text: str,
    *,
    surface: Surface = "assistant",
) -> str:
    """Project assistant/log/prompt text for a specific surface."""

    out = rewrite_agent_visible_paths(agent, text)
    if surface == "prompt":
        return out
    context: MatchContext = "tool_output" if surface == "tool_output" else "prose"
    return redact_hidden_paths(agent, out, context=context, marker=False)


def project_log(agent: Any, text: str) -> str:
    return project_agent_visible_text(agent, text, surface="log")


def _is_listing_request(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name in {"ls", "glob"}:
        return True
    if tool_name == "bash":
        command = str((args or {}).get("command") or "")
        return bool(_LISTING_COMMAND_RE.search(command))
    return False


def project_tool_output(
    agent: Any,
    text: str,
    *,
    tool_name: str,
    args: dict[str, Any],
    marker: bool = True,
) -> str:
    """Project one tool result for model-visible feedback."""

    out = rewrite_agent_visible_paths(agent, text)
    listing = _is_listing_request(tool_name, args or {})
    return redact_hidden_paths(
        agent,
        out,
        context="listing" if listing else "tool_output",
        marker=marker,
    )


def path_mentions_hidden_file(agent: Any, value: str) -> bool:
    policy = HiddenPathPolicy.from_agent(agent)
    text = str(value or "").replace("\\", "/").strip().strip("'\"")
    if not text:
        return False

    workspace_dir = getattr(agent, "_workspace_dir", None)
    if text.startswith("/") and workspace_dir is not None:
        try:
            text = (
                Path(text)
                .resolve(strict=False)
                .relative_to(Path(workspace_dir).resolve(strict=False))
                .as_posix()
            )
        except (OSError, ValueError):
            pass

    return bool(_iter_hidden_matches_in_line(text, policy, context="command"))


def text_mentions_hidden_paths(
    agent: Any,
    text: str,
    *,
    context: MatchContext = "command",
) -> bool:
    policy = HiddenPathPolicy.from_agent(agent)
    for line in str(text or "").splitlines():
        if _iter_hidden_matches_in_line(line, policy, context=context):
            return True
    return False


def text_mentions_hidden_prefix(agent: Any, text: str, prefix: str) -> bool:
    normalised = _normalise_entry(prefix)
    if not normalised:
        return False
    policy = HiddenPathPolicy(prefixes=(normalised,), filenames=())
    for line in str(text or "").splitlines():
        if _iter_hidden_matches_in_line(line, policy, context="command"):
            return True
    return False


def request_mentions_hidden_paths(
    agent: Any,
    tool_name: str,
    args: dict[str, Any],
) -> bool:
    policy = HiddenPathPolicy.from_agent(agent)
    if not policy.prefixes and not policy.filenames:
        return False

    if tool_name in {"read", "write", "edit", "ls", "grep", "glob"}:
        return path_mentions_hidden_file(agent, str(args.get("path") or ""))

    if tool_name == "bash":
        return text_mentions_hidden_paths(
            agent,
            str(args.get("command") or ""),
            context="command",
        )

    return False


__all__ = [
    "HiddenPathMatch",
    "HiddenPathPolicy",
    "MatchContext",
    "Surface",
    "path_mentions_hidden_file",
    "project_agent_visible_text",
    "project_log",
    "project_tool_output",
    "redact_hidden_paths",
    "request_mentions_hidden_paths",
    "rewrite_agent_visible_paths",
    "text_mentions_hidden_paths",
    "text_mentions_hidden_prefix",
]
