#!/usr/bin/env python3
"""Compare ScienceFlow workspace manifests without hiding unknown drift.

``strict`` requires the complete captured manifest to match (except ``root_name``).
``compatible`` normalizes only the runtime-generated path components listed below,
ignores content-addressed object payload paths, and compares every remaining path,
kind, mode, link contract and structured-file shape from the baseline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"

# These are value-derived storage locations, not public filenames. Their parent
# directories remain part of the contract and are still checked.
IGNORED_GENERATED_PATHS = (
    re.compile(r"^workers/w\d+/workspace/\.git/objects/(?!info$|pack$).+"),
    re.compile(r"^workers/w\d+/snapshots/objects/sha256/.+"),
    re.compile(r"^workers/w\d+/workspace/\.scienceflow_runs/.+"),
    re.compile(r"^workers/w\d+/workspace/tmp/.+"),
    re.compile(r"^workers/w\d+/workspace/\.scienceflow_global_merge/\.scienceflow_runs(?:/.*)?$"),
    re.compile(r"(?:^|/)tool_outputs/(?!index\.txt$).+"),
)

PATH_NORMALIZERS = (
    (re.compile(r"^workers/w\d+"), "workers/wXX"),
    (re.compile(r"/workers/w\d+"), "/workers/wXX"),
    (re.compile(r"(?:^|(?<=/))w\d+(?=/snapshots/)"), "wXX"),
    (re.compile(r"W\d+(?=-L\d+-S\d+)"), "WXX"),
    (re.compile(r"W\d+(?=_bash_\d+)"), "WXX"),
    (re.compile(r"(?<=_bash_)\d+"), "<run>"),
    (re.compile(r"(?<=tool_)\d+"), "<tool>"),
    (re.compile(r"(-S\d+-)[0-9a-f]{10}(?=$|[./])"), r"\1<snapshot>"),
    (
        re.compile(r"iter_\d+_\d{8}T\d{6}Z_([^/]+)_sha_[0-9a-f]{12}\.json$"),
        r"iter_<n>_<timestamp>_\1_sha_<hash>.json",
    ),
)


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read manifest {source}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"Unsupported workspace manifest: {source}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("entry_count") != len(entries):
        raise ManifestError(f"Invalid entry_count in workspace manifest: {source}")
    return manifest


def normalize_generated_path_parts(path: str) -> str:
    normalized = path.replace("\\", "/")
    for pattern, replacement in PATH_NORMALIZERS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def normalize_relative_link_target(target: str) -> str:
    normalized = normalize_generated_path_parts(target)
    # The global merge workspace may be hosted by the source worker or a peer.
    # Relative depth therefore changes while the conceptual snapshot target does not.
    normalized = re.sub(r"^(?:\.\./)+(?:wXX/)?", "", normalized)
    return normalized


def canonical_path(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    if any(pattern.search(normalized) for pattern in IGNORED_GENERATED_PATHS):
        return None
    return normalize_generated_path_parts(normalized)


def _shape_contract(shape: Any) -> Any:
    if not isinstance(shape, dict):
        return None
    contract = {key: value for key, value in shape.items() if key not in {"row_count", "length"}}
    # Event occurrence depends on scheduling and task trajectory. Strict mode still
    # compares the original complete shape; compatible mode protects envelope keys
    # and types while deterministic replay protects exact event sequence.
    contract.pop("event_type_counts", None)
    return contract


def _entry_contract(entry: dict[str, Any], *, include_content: bool) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "kind": entry.get("kind"),
        "mode": entry.get("mode"),
    }
    for key in ("target_kind", "target", "target_basename"):
        if key in entry:
            value = entry[key]
            if key == "target" and not include_content and isinstance(value, str):
                value = normalize_relative_link_target(value)
            contract[key] = value
    shape = entry.get("shape") if include_content else _shape_contract(entry.get("shape"))
    if shape is not None:
        contract["shape"] = shape
    if include_content:
        for key in ("size", "sha256"):
            if key in entry:
                contract[key] = entry[key]
    return contract


def _projection(
    manifest: dict[str, Any], *, include_content: bool
) -> dict[str, set[str]]:
    projected: dict[str, set[str]] = defaultdict(set)
    for raw in manifest["entries"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ManifestError("Workspace manifest contains an invalid entry")
        path = raw["path"] if include_content else canonical_path(raw["path"])
        if path is None:
            continue
        encoded = json.dumps(
            _entry_contract(raw, include_content=include_content),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        projected[path].add(encoded)
    return dict(projected)


def _worker_roots(manifest: dict[str, Any]) -> set[str]:
    roots: set[str] = set()
    for entry in manifest["entries"]:
        path = str(entry.get("path") or "")
        match = re.match(r"^workers/(w\d+)(?:/|$)", path)
        if match:
            roots.add(match.group(1))
    return roots


def compare_manifests(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    mode: str = "compatible",
) -> dict[str, Any]:
    if mode not in {"strict", "compatible"}:
        raise ValueError(f"Unsupported comparison mode: {mode}")
    strict = mode == "strict"
    expected = _projection(baseline, include_content=strict)
    actual = _projection(current, include_content=strict)

    missing_paths = sorted(set(expected) - set(actual))
    additional_paths = sorted(set(actual) - set(expected))
    mismatched: list[dict[str, Any]] = []
    for path in sorted(set(expected) & set(actual)):
        if strict:
            matches = expected[path] == actual[path]
        else:
            # Each baseline contract variant must remain available. Additional
            # variants are allowed so a later event type cannot mask a removal.
            matches = expected[path].issubset(actual[path])
        if not matches:
            mismatched.append(
                {
                    "path": path,
                    "expected": [json.loads(row) for row in sorted(expected[path])],
                    "actual": [json.loads(row) for row in sorted(actual[path])],
                }
            )

    expected_workers = len(_worker_roots(baseline))
    actual_workers = len(_worker_roots(current))
    worker_count_match = expected_workers == actual_workers
    compatible = not missing_paths and not mismatched and worker_count_match
    if strict:
        compatible = compatible and not additional_paths
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "compatible": compatible,
        "expected_worker_count": expected_workers,
        "actual_worker_count": actual_workers,
        "missing_paths": missing_paths,
        "additional_paths": additional_paths,
        "mismatched_entries": mismatched,
        "checked_path_count": len(expected),
    }


def _limited(items: Iterable[Any], limit: int) -> list[Any]:
    rows = list(items)
    if len(rows) <= limit:
        return rows
    return [*rows[:limit], {"truncated": len(rows) - limit}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--mode", choices=("strict", "compatible"), default="compatible")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--display-limit", type=int, default=25)
    args = parser.parse_args()

    report = compare_manifests(
        load_manifest(args.baseline),
        load_manifest(args.current),
        mode=args.mode,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    display = dict(report)
    for key in ("missing_paths", "additional_paths", "mismatched_entries"):
        display[key] = _limited(display[key], args.display_limit)
    print(json.dumps(display, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["compatible"] else 1)


if __name__ == "__main__":
    main()
