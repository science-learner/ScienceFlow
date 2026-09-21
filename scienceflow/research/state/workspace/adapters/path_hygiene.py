# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Workspace path visibility adapter for ScienceAgent provider messages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from inquirycraft.memory import Message
from inquirycraft.tools import ToolResult

from scienceflow.research.state.workspace.session.agent_contracts import (
    _HOST_ABSOLUTE_VISIBLE_PATH_RE,
    _WSP_ABSOLUTE_VISIBLE_PREFIX_RE,
)
from scienceflow.research.state.knowledge.memory.agent.memory_utils import _assistant_message_from_api


def _maybe_workspace_git_auto_checkpoint(
    self,
    tool_name: str,
    tool_result: ToolResult,
) -> None:
    """Checkpoint tracked source/docs after successful REPL tool activity."""
    if not bool(getattr(self, "_workspace_git_auto_checkpoint_enabled", False)):
        return
    try:
        from scienceflow.research.state.workspace.storage.git import auto_checkpoint_workspace_source

        result = auto_checkpoint_workspace_source(
            self._workspace_dir,
            enabled=True,
            track_globs=getattr(self, "_workspace_git_track_globs", ("*.py", "*.md")),
            tool_name=tool_name,
            tool_error=bool(tool_result.error),
        )
    except Exception as exc:
        self._log_warning("[workspace-git] auto checkpoint failed: %s", exc)
        return
    if not result.ready:
        if result.message:
            self._log_warning(
                "[workspace-git] auto checkpoint unavailable: %s", result.message
            )
        return
    if result.committed or result.submission_snapshot:
        parts = []
        if result.committed:
            parts.append(f"commit={result.commit_sha[:12]}")
        if result.submission_snapshot:
            parts.append(f"submission={result.submission_snapshot}")
        if result.metric_value is not None:
            parts.append(f"metric={result.metric_value:.6g}")
        self._log_info("[workspace-git] auto checkpoint %s", " ".join(parts))


def set_repl_session_run_index(self, n: int | None) -> None:
    """Set outer REPL session counter for ``[repl-run N]`` logs (None to disable)."""
    self._repl_session_run_index = n


def _workspace_relative_path_mode_enabled(self) -> bool:
    return bool(getattr(self, "_workspace_relative_path_mode", False))


def _path_hygiene_enabled(self) -> bool:
    return _workspace_relative_path_mode_enabled(self) or (
        str(getattr(self, "_teleport_mode", "off") or "off") != "off"
    )


@staticmethod
def _host_abs_placeholder(value: str) -> str:
    raw = str(value or "")
    name = Path(raw.rstrip("/ ")).name or raw.strip("/") or "path"
    return f"<host-path:{name}>"


@staticmethod
def _rewrite_abs_prefix_to_label(text: str, root: Path, label: str) -> str:
    root_s = str(root).rstrip("/")
    if not root_s or root_s == "/" or root_s not in text:
        return text
    text = text.replace(root_s + "/", label.rstrip("/") + "/")
    text = text.replace(root_s, label.rstrip("/"))
    return text


def _get_agent_hidden_workspace_filenames(self) -> tuple[str, ...]:
    raw = getattr(self, "_agent_hidden_workspace_filenames", ())
    if isinstance(raw, str):
        raw = (raw,)
    names: list[str] = []
    try:
        iterator = iter(raw)
    except TypeError:
        iterator = iter(())
    for item in iterator:
        name = Path(str(item or "").replace("\\", "/")).name.strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _normalize_hidden_workspace_path(self, value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip().strip("'\"")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.lstrip("/")
    return raw.rstrip("/")


def _get_agent_hidden_workspace_path_prefixes(self) -> tuple[str, ...]:
    raw = getattr(self, "_agent_hidden_workspace_path_prefixes", ())
    if isinstance(raw, str):
        raw = (raw,)
    prefixes: list[str] = []
    try:
        iterator = iter(raw)
    except TypeError:
        iterator = iter(())
    for item in iterator:
        prefix = _normalize_hidden_workspace_path(self, str(item or ""))
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def _apply_agent_hidden_path_denials(self, prefixes: Sequence[str | Path]) -> None:
    denied = [
        _normalize_hidden_workspace_path(self, str(prefix or "")) for prefix in prefixes
    ]
    denied = [prefix for prefix in denied if prefix]
    tool_map = getattr(getattr(self, "availableTools", None), "tool_map", {}) or {}
    for tool in tool_map.values():
        if hasattr(tool, "path_guard_denied_prefixes"):
            try:
                tool.path_guard_denied_prefixes = list(denied)
            except Exception:
                continue


def _text_mentions_hidden_workspace_prefix(self, text: str, prefix: str) -> bool:
    raw = str(text or "").replace("\\", "/")
    prefix = str(prefix or "").strip("/")
    if not raw or not prefix:
        return False
    escaped = re.escape(prefix)
    pattern = rf"(^|[\s'\"=;:&|(<>{{}}\[])(?:\./)?{escaped}(?=($|[\s'\"=;:&|/)<>}}\]]))"
    return re.search(pattern, raw) is not None


def _hide_agent_hidden_workspace_filename_mentions(self, text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    for prefix in sorted(
        _get_agent_hidden_workspace_path_prefixes(self), key=len, reverse=True
    ):
        escaped = re.escape(prefix)
        out = re.sub(
            rf"(?<![\w./-])(?:\./)?{escaped}(?=($|[\s'\"=;:&|/)<>}}\]]))",
            "<hidden-control-path>",
            out,
        )
    for name in _get_agent_hidden_workspace_filenames(self):
        out = out.replace(name, "<hidden-control-file>")
    return out


def _path_mentions_hidden_workspace_file(self, value: str) -> bool:
    raw = str(value or "").replace("\\", "/").strip().strip("'\"")
    if not raw:
        return False
    norm = _normalize_hidden_workspace_path(self, raw)
    for prefix in _get_agent_hidden_workspace_path_prefixes(self):
        if (
            norm == prefix
            or norm.startswith(prefix + "/")
            or norm.endswith("/" + prefix)
            or f"/{prefix}/" in norm
        ):
            return True
    for name in _get_agent_hidden_workspace_filenames(self):
        if (
            norm == name
            or norm.endswith("/" + name)
            or raw == name
            or raw.endswith("/" + name)
        ):
            return True
    return False


def _tool_request_mentions_hidden_workspace_file(
    self, tool_name: str, args: dict[str, Any]
) -> bool:
    prefixes = _get_agent_hidden_workspace_path_prefixes(self)
    names = _get_agent_hidden_workspace_filenames(self)
    if not prefixes and not names:
        return False
    if tool_name in {"read", "write", "edit", "ls", "grep"}:
        if _path_mentions_hidden_workspace_file(self, str(args.get("path") or "")):
            return True
    if tool_name == "bash":
        cmd = str(args.get("command") or "")
        if any(
            _text_mentions_hidden_workspace_prefix(self, cmd, prefix)
            for prefix in prefixes
        ):
            return True
        return any(name in cmd for name in names)
    return False


def _hide_agent_hidden_workspace_file_lines(self, text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    names = _get_agent_hidden_workspace_filenames(self)
    prefixes = _get_agent_hidden_workspace_path_prefixes(self)
    if not names and not prefixes:
        return text
    kept = []
    removed = False
    for line in text.splitlines():
        if any(name in line for name in names) or any(
            _text_mentions_hidden_workspace_prefix(self, line, prefix)
            for prefix in prefixes
        ):
            removed = True
            continue
        kept.append(line)
    if removed:
        kept.append("[hidden control file entries omitted]")
    return "\n".join(kept)


def _sanitize_agent_visible_paths(self, text: str) -> str:
    """Rewrite model-visible host paths to workspace-relative or opaque labels."""
    if not isinstance(text, str) or not text:
        return text
    if not _path_hygiene_enabled(self):
        return _hide_agent_hidden_workspace_filename_mentions(self, text)
    from scienceflow.research.state.knowledge.memory.records.agent_records import rewrite_workspace_abs_to_relative

    out = rewrite_workspace_abs_to_relative(text, str(self._workspace_dir))
    out = _WSP_ABSOLUTE_VISIBLE_PREFIX_RE.sub("./", out)

    dataset_link = self._workspace_dir / "dataset"
    try:
        if dataset_link.exists():
            out = _rewrite_abs_prefix_to_label(out, dataset_link.resolve(), "dataset")
    except OSError:
        pass

    if _workspace_relative_path_mode_enabled(self):
        out = _HOST_ABSOLUTE_VISIBLE_PATH_RE.sub(
            lambda m: _host_abs_placeholder(m.group(0)),
            out,
        )
    return _hide_agent_hidden_workspace_filename_mentions(self, out)


def _sanitize_message_agent_visible_paths(self, message: Message) -> Message:
    """Sanitize message content and assistant tool-call arguments before memory use."""
    if not _path_hygiene_enabled(self):
        return message
    from inquirycraft.memory import Function as ToolFunction

    updates: dict[str, Any] = {}
    if isinstance(message.content, str):
        content = _sanitize_agent_visible_paths(self, message.content)
        if content != message.content:
            updates["content"] = content
    if message.tool_calls:
        new_calls = []
        changed = False
        for tc in message.tool_calls:
            args = tc.function.arguments
            new_args = (
                _sanitize_agent_visible_paths(self, args)
                if isinstance(args, str)
                else args
            )
            if new_args != args:
                changed = True
                new_calls.append(
                    tc.model_copy(
                        update={
                            "function": ToolFunction(
                                name=tc.function.name,
                                arguments=new_args,
                            ),
                        },
                    ),
                )
            else:
                new_calls.append(tc)
        if changed:
            updates["tool_calls"] = new_calls
    if not updates:
        return message
    return message.model_copy(update=updates)


def _add_assistant_api_message(self, assistant_msg: Any) -> None:
    self.memory.add_message(
        _sanitize_message_agent_visible_paths(
            self, _assistant_message_from_api(assistant_msg)
        ),
    )


def sanitize_existing_memory_agent_visible_paths(self) -> int:
    """Rewrite already-loaded chat memory for opt-in workspace-relative path mode."""
    if not _path_hygiene_enabled(self):
        return 0
    try:
        records = self.memory.chat_history_memory.retrieve(window_size=None)
    except Exception:
        return 0
    if not records:
        return 0
    messages = []
    changed = False
    for record in records:
        try:
            message = record.memory_record.message
        except Exception:
            continue
        new_message = _sanitize_message_agent_visible_paths(self, message)
        if new_message is not message:
            changed = True
        messages.append(new_message)
    if not changed or not messages:
        return 0
    try:
        rewrite = getattr(getattr(self, "_memory_ctx", None), "rewrite_messages", None)
        if callable(rewrite):
            rewrite(messages)
        else:
            self.memory.chat_history_memory.storage.clear()
            for message in messages:
                self.memory.add_message(message)
        return len(messages)
    except Exception:
        return 0


def _maybe_normalize_abs_path_to_workspace_relative(
    self, path_value: str
) -> str | None:
    try:
        p = Path(path_value).expanduser()
    except (TypeError, ValueError):
        return None
    if not p.is_absolute():
        return None
    try:
        rel = p.resolve(strict=False).relative_to(self._workspace_dir)
        return rel.as_posix() or "."
    except (OSError, ValueError):
        pass
    dataset_link = self._workspace_dir / "dataset"
    try:
        if dataset_link.exists():
            rel = p.resolve(strict=False).relative_to(dataset_link.resolve())
            return f"dataset/{rel.as_posix()}" if rel.as_posix() else "dataset"
    except (OSError, ValueError):
        pass
    return None


def _maybe_normalize_tool_input_paths(
    self, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """Normalize model-emitted path args to workspace-relative form when enabled."""
    if not isinstance(args, dict):
        return args
    if not _path_hygiene_enabled(self):
        return args
    path_val = args.get("path")
    if isinstance(path_val, str):
        new_path: str | None = None
        if path_val.startswith("/workspace/"):
            new_path = "./" + path_val[len("/workspace/") :]
        elif path_val == "/workspace":
            new_path = "."
        elif _workspace_relative_path_mode_enabled(self):
            new_path = _maybe_normalize_abs_path_to_workspace_relative(self, path_val)
        if new_path is not None and new_path != path_val:
            args = dict(args)
            args["path"] = new_path
    return args


def _maybe_rewrite_tool_result_paths(self, tool_result: ToolResult) -> ToolResult:
    """Rewrite model-visible tool output paths to stable relative forms."""
    if tool_result is None or not _path_hygiene_enabled(self):
        return tool_result
    out = tool_result.output
    new_out = _sanitize_agent_visible_paths(self, out) if isinstance(out, str) else out
    err = tool_result.error
    new_err = _sanitize_agent_visible_paths(self, err) if isinstance(err, str) else err
    if new_out is out and new_err is err:
        return tool_result
    # Mutate in place when possible (ToolResult is pydantic; fall back to copy)
    try:
        tool_result.output = new_out  # type: ignore[assignment]
        tool_result.error = new_err  # type: ignore[assignment]
        return tool_result
    except Exception:
        try:
            return tool_result.model_copy(update={"output": new_out, "error": new_err})
        except Exception:
            return tool_result
