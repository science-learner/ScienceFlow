# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Exact-format JSON event journal and atomic state projection stores."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_text_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            writer(handle)
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


class JsonEventJournal:
    """Append and rebuild a newline-delimited JSON journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_json_safe(event), ensure_ascii=False, sort_keys=True) + "\n"
            )

    def rewrite(self, events: Sequence[Mapping[str, Any]]) -> None:
        def write(handle: Any) -> None:
            for event in events:
                handle.write(
                    json.dumps(_json_safe(event), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )

        _atomic_text_write(self.path, write)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            for line in self.path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    events.append(value)
        except (OSError, json.JSONDecodeError):
            return events
        return events


class JsonStateProjection:
    """Atomically replace a stable, sorted and indented JSON projection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, state: Mapping[str, Any]) -> None:
        def write(handle: Any) -> None:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        _atomic_text_write(self.path, write)


__all__ = ["JsonEventJournal", "JsonStateProjection"]
