#!/usr/bin/env python3
"""Capture a portable ScienceFlow workspace contract manifest.

The manifest records every externally visible path while avoiding host absolute
paths. It is intentionally descriptive: later compatibility tests may normalize
only the explicitly listed volatile fields, never unknown fields wholesale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "1.0"

CONTRACT_RULES = [
    {
        "pattern": "**/.agent_memory/**/short_term.json",
        "producer": "scienceflow.research.state.knowledge.memory",
        "consumers": ["scienceflow.resume", "scienceflow.replay"],
        "write_mode": "replace",
    },
    {
        "pattern": "**/.agent_memory/**/long_term.jsonl",
        "producer": "scienceflow.research.state.knowledge.memory",
        "consumers": ["scienceflow.resume", "scienceflow.replay"],
        "write_mode": "append",
    },
    {
        "pattern": "**/lhr_events.jsonl",
        "producer": "scienceflow.lnr",
        "consumers": ["scienceflow.monitor", "scienceflow.merge"],
        "write_mode": "append",
    },
    {
        "pattern": "**/interaction.log",
        "producer": "scienceflow.logging",
        "consumers": ["human", "scienceflow.trace"],
        "write_mode": "append",
    },
    {
        "pattern": "**/.run_results.md",
        "producer": "scienceflow.stage",
        "consumers": ["scienceflow.merge", "scienceflow.resume"],
        "write_mode": "append",
    },
    {
        "pattern": "**/resolved_config.yaml",
        "producer": "scienceflow.foundation.config",
        "consumers": ["scienceflow.runtime", "scienceflow.replay"],
        "write_mode": "replace",
    },
    {
        "pattern": "**/merge/**",
        "producer": "scienceflow.merge",
        "consumers": ["scienceflow.final_evaluator", "human"],
        "write_mode": "replace",
    },
]

VOLATILE_FIELDS = {
    "json_keys": [
        "timestamp",
        "timestamp_utc",
        "generated_at",
        "generated_at_utc",
        "started_at",
        "started_at_utc",
        "run_started_at",
        "elapsed_sec",
        "charged_elapsed_sec",
        "segment_elapsed_sec",
        "pid",
        "pgid",
    ],
    "text_patterns": [
        "ISO-8601 timestamps",
        "process IDs",
        "elapsed durations",
        "host absolute paths",
        "runtime-generated UUIDs and event IDs",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def object_shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": sorted(value),
            "key_types": {key: value_type(value[key]) for key in sorted(value)},
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "item_types": sorted({value_type(item) for item in value}),
        }
    return {"type": value_type(value)}


def jsonl_shape(path: Path) -> dict[str, Any]:
    row_count = 0
    invalid_rows = 0
    keys: set[str] = set()
    key_types: dict[str, set[str]] = {}
    event_types: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(row, dict):
                continue
            keys.update(row)
            for key, value in row.items():
                key_types.setdefault(key, set()).add(value_type(value))
            event_type = row.get("event") or row.get("type")
            if isinstance(event_type, str):
                event_types[event_type] += 1
    return {
        "format": "jsonl",
        "row_count": row_count,
        "invalid_rows": invalid_rows,
        "keys": sorted(keys),
        "key_types": {key: sorted(values) for key, values in sorted(key_types.items())},
        "event_type_counts": dict(sorted(event_types.items())),
    }


def structured_shape(path: Path) -> dict[str, Any] | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            return jsonl_shape(path)
        if suffix == ".json":
            return {"format": "json", **object_shape(json.loads(path.read_text(encoding="utf-8")))}
        if suffix in {".yaml", ".yml"}:
            return {"format": "yaml", **object_shape(yaml.safe_load(path.read_text(encoding="utf-8")))}
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                row_count = sum(1 for _ in reader)
            return {"format": "csv", "columns": header, "row_count": row_count}
    except (OSError, UnicodeError, ValueError, TypeError, yaml.YAMLError) as exc:
        return {"format": suffix.lstrip(".") or "unknown", "parse_error": type(exc).__name__}
    return None


def portable_link_target(path: Path, root: Path) -> dict[str, str]:
    raw = os.readlink(path)
    target = Path(raw)
    if not target.is_absolute():
        return {"target_kind": "relative", "target": raw}
    try:
        relative = target.resolve().relative_to(root.resolve())
    except ValueError:
        return {"target_kind": "external", "target_basename": target.name}
    return {"target_kind": "workspace", "target": relative.as_posix()}


def capture(root: Path) -> dict[str, Any]:
    root = root.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        entry: dict[str, Any] = {
            "path": relative,
            "mode": stat.S_IMODE(info.st_mode),
        }
        if path.is_symlink():
            entry.update(kind="symlink", **portable_link_target(path, root))
        elif path.is_dir():
            entry["kind"] = "directory"
        elif path.is_file():
            entry.update(kind="file", size=info.st_size, sha256=sha256(path))
            shape = structured_shape(path)
            if shape is not None:
                entry["shape"] = shape
        else:
            entry["kind"] = "other"
        entries.append(entry)

    counts = Counter(entry["kind"] for entry in entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "root_name": root.name,
        "entry_count": len(entries),
        "kind_counts": dict(sorted(counts.items())),
        "contract_rules": CONTRACT_RULES,
        "volatile_fields": VOLATILE_FIELDS,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"workspace root is not a directory: {args.root}")
    manifest = capture(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"captured {manifest['entry_count']} entries -> {args.output}")


if __name__ == "__main__":
    main()
