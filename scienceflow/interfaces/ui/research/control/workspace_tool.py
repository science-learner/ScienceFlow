"""Bounded, shell-free inspection of the active tool workspace."""

import asyncio
import json
import os
from itertools import islice
from pathlib import Path

from inquirycraft.tools.base import BaseTool, ToolResult
from inquirycraft.tools.path_policy import PathPolicy


class WorkspaceInfoTool(BaseTool):
    """Inspect directories without granting command execution."""

    name = "workspace_info"
    description = (
        "Query the actual current tool workspace (pwd), list a directory, or inspect "
        "file metadata without shell permission. Paths are workspace-relative; "
        "listing is non-recursive and limited to 100 entries."
    )
    replay_policy = "safe"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["pwd", "list", "stat"]},
            "path": {"type": "string", "description": "Relative path; defaults to ."},
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    async def execute(self, arguments, context):
        try:
            result = await asyncio.to_thread(
                self._inspect, arguments, context.workspace
            )
            return ToolResult(output=json.dumps(result, ensure_ascii=False))
        except (OSError, ValueError, RuntimeError) as exc:
            return ToolResult(error=str(exc))

    @staticmethod
    def _inspect(arguments, workspace):
        root = Path(workspace).resolve()
        action = arguments.get("action")
        if action == "pwd":
            return {"workspace": str(root)}
        raw = arguments.get("path", ".")
        if not isinstance(raw, str) or not raw.strip() or Path(raw).is_absolute():
            raise ValueError("Use a non-empty workspace-relative path.")
        path = PathPolicy(root).resolve(raw)
        result = {"workspace": str(root), "path": str(path.relative_to(root))}
        if action == "stat":
            info = path.stat()
            result.update(
                kind="directory" if path.is_dir() else "file",
                bytes=info.st_size,
                modified_unix=info.st_mtime,
            )
        elif action == "list":
            with os.scandir(path) as entries:
                sample = list(islice(entries, 101))
            result["entries"] = [
                {
                    "name": entry.name,
                    "kind": "symlink"
                    if entry.is_symlink()
                    else "directory"
                    if entry.is_dir(follow_symlinks=False)
                    else "file",
                }
                for entry in sorted(sample[:100], key=lambda entry: entry.name)
            ]
            result["truncated"] = len(sample) > 100
        else:
            raise ValueError("action must be pwd, list, or stat")
        return result
