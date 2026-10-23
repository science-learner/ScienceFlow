# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""ScienceFlow read-before-edit safety policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from inquirycraft.tools import PathGuard, ToolResult, coerce_tool_result

from scienceflow.foundation.config.runtime.agent_constants import (
    _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD,
)
from scienceflow.research.state.workspace.adapters.path_hygiene import (
    _maybe_normalize_tool_input_paths,
)


def _reset_edit_read_guard_state(self) -> None:
    """Clear read fingerprints for :class:`EditFailureGuard` hard re-read policy."""
    self._recent_read_history.clear()
    self._last_read_sha_by_path.clear()


@staticmethod
def _normalize_rel_workspace_path(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("/")


def _resolve_workspace_path(self, rel: str) -> Path | None:
    try:
        return PathGuard(
            self._workspace_dir,
            enabled=self._sandbox,
            extra_roots=self._path_guard_extra_roots,
        ).resolve(rel)
    except ValueError:
        return None


def _record_read_for_edit_guard(self, rel: str) -> None:
    norm = _normalize_rel_workspace_path(rel)
    resolved = _resolve_workspace_path(self, norm)
    if resolved is None:
        return
    sha = _sha256_file(resolved)
    if sha is None:
        return
    self._last_read_sha_by_path[norm] = sha
    self._recent_read_history.append((norm, sha))


def _record_successful_file_view_for_edit_guard(
    self,
    name: str,
    args: dict[str, Any],
    result: ToolResult,
) -> None:
    """Treat successful read/write/edit results as a fresh file view for edit safety."""
    if result.error or name not in {"read", "write", "edit"}:
        return
    path = str(args.get("path") or "")
    if path:
        _record_read_for_edit_guard(self, path)


def seed_edit_guard(self, rel: str) -> None:
    """Pre-seed read-before-edit guard for a path (e.g. clone parent solution on disk)."""
    _record_read_for_edit_guard(self, rel)


def _should_block_edit_for_stale_read(self, rel: str) -> bool:
    """True when ``edit`` must be blocked until a successful ``read`` matches on-disk content."""
    norm = _normalize_rel_workspace_path(rel)
    resolved = _resolve_workspace_path(self, norm)
    if resolved is None:
        return True
    cur = _sha256_file(resolved)
    if cur is None:
        return True
    if cur == self._last_read_sha_by_path.get(norm):
        return False
    for p, s in self._recent_read_history:
        if p == norm and s == cur:
            return False
    return True


async def _execute_tool_maybe_edit_guard(
    self,
    name: str,
    args: dict[str, Any],
) -> ToolResult:
    """Run non-bash tools; block ``edit`` until a recent ``read`` matches file contents."""
    # Teleport: normalize defensive ``/workspace/...`` prefix to ``./...`` so the
    # tool layer (rooted at the current node directory) sees a workable path.
    args = _maybe_normalize_tool_input_paths(self, name, args)
    if name == "edit":
        rel = str(args.get("path") or "")
        if _should_block_edit_for_stale_read(self, rel):
            return ToolResult(
                error=(
                    f"Edit blocked: re-read `{rel}` (full or around target) first; "
                    "file changed or not recently read."
                ),
            )
    result = await self.availableTools.execute(
        name=name,
        tool_input=args,
    )
    result = coerce_tool_result(result)
    _record_successful_file_view_for_edit_guard(self, name, args, result)
    return result


@staticmethod
def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()
    except OSError:
        return None


def _parent_solution_path(self) -> Path | None:
    """Improve nodes: parent's ``solution.py`` on disk (sibling workspace), if known."""
    from scienceflow.runtime.core.support.node_paths import find_node_context_path

    ws = self._workspace_dir
    ctx_path = find_node_context_path(ws)
    if ctx_path.is_file():
        try:
            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            ctx = {}
        else:
            pid = ctx.get("parent_node_id") or ctx.get("inherit_parent_id")
            if isinstance(pid, str) and pid.strip():
                p = (ws.parent / pid.strip() / "solution.py").resolve()
                if p.is_file():
                    return p
    leg = ws / "parent_workspace" / "solution.py"
    return leg if leg.is_file() else None


@staticmethod
def classify_tool_error_for_budget(
    name: str,
    args: dict[str, Any],
    tool_result: ToolResult,
) -> str:
    """``protocol`` = user/tool-args mistakes; ``runtime`` = command execution failures."""
    err = (tool_result.error or "") or ""
    el = err.lower()
    if name == "bash":
        return "runtime"
    if name == "write":
        if "syntax check failed" in el:
            return "protocol"
        c = args.get("content")
        if isinstance(c, str) and len(c) < _WRITE_FAIL_COACH_SHORT_CONTENT_THRESHOLD:
            return "protocol"
        return "runtime"
    if name == "grep":
        if "timed out" in el or "timeout" in el:
            return "runtime"
        return "protocol"
    if name in ("read", "edit"):
        return "protocol"
    return "runtime"
