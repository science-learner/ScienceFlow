from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.runtime_parity.comparison.normalize import normalize_value, rule_registered


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture_workspace(root: Path) -> list[dict[str, Any]]:
    """Capture deterministic file content plus symlink contracts for small benchmark workspaces."""
    entries: list[dict[str, Any]] = []
    if not root.exists():
        return entries
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if rule_registered("runtime_audit_artifacts") and (
            relative == ".logs/agent_runtime_events.jsonl"
            or relative == ".logs/agent_provider_calls.jsonl"
            or relative.startswith(".logs/agent_runtime_operations")
        ):
            continue
        if path.is_symlink():
            entries.append(
                {"path": relative, "kind": "symlink", "target": os.readlink(path)}
            )
            continue
        if path.is_dir():
            entries.append({"path": relative, "kind": "directory"})
            continue
        data = path.read_bytes()
        entry: dict[str, Any] = {"path": relative, "kind": "file", "size": len(data)}
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            entry["sha256"] = _sha256(data)
        else:
            if path.suffix == ".json":
                try:
                    entry["content"] = normalize_value(json.loads(text), workspace=root)
                except json.JSONDecodeError:
                    entry["content"] = normalize_value(text, workspace=root)
            elif path.suffix == ".jsonl" or path.name.endswith(".jsonl"):
                rows: list[Any] = []
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        rows.append(line)
                if path.name == "events.jsonl" and rule_registered("event_id"):
                    from inquirycraft.events.replay import canonicalize_runtime_events

                    rows = canonicalize_runtime_events(rows)
                entry["content"] = normalize_value(rows, workspace=root)
            else:
                entry["content"] = normalize_value(text, workspace=root)
        entries.append(entry)
    return entries


__all__ = ["capture_workspace"]
