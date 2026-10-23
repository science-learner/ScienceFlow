# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Low-overhead task storage telemetry for long-running research."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:  # Linux production runs use process locks; keep package imports portable.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback is process-local.
    fcntl = None  # type: ignore[assignment]

STORAGE_EVENTS_FILENAME = "storage_events.jsonl"
STORAGE_LATEST_FILENAME = "storage_latest.json"
_STORAGE_LOCK_FILENAME = ".storage_usage.lock"
_DATASET_DIR_NAMES = frozenset({"dataset"})
_TELEMETRY_FILE_NAMES = frozenset(
    {
        STORAGE_EVENTS_FILENAME,
        STORAGE_LATEST_FILENAME,
        _STORAGE_LOCK_FILENAME,
    }
)
_PROCESS_LOCK = threading.Lock()


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _locked(log_dir: Path) -> Iterator[None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        with (log_dir / _STORAGE_LOCK_FILENAME).open("a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _append_event(log_dir: Path, payload: Mapping[str, Any]) -> None:
    with (log_dir / STORAGE_EVENTS_FILENAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def record_snapshot_storage(
    *,
    task_root_dir: str | Path,
    worker_id: str,
    stage_id: str,
    snapshot_id: str,
    stats: Mapping[str, Any],
    snapshot_kind: str = "stage",
) -> bool:
    """Record snapshot counters already computed by ``WorkspaceSnapshotStore``.

    This function performs no directory traversal. Concurrent workers serialize
    the small task-level projection with an advisory file lock.
    """

    task_root = Path(task_root_dir)
    log_dir = task_root / "task_logs"
    latest_path = log_dir / STORAGE_LATEST_FILENAME
    worker_key = str(worker_id or "w00").strip().lower() or "w00"
    now = _timestamp_utc()
    try:
        with _locked(log_dir):
            previous = _read_json(latest_path)
            if previous.get("exact") is True:
                previous = {
                    "schema_version": 1,
                    "baseline_bytes": _nonnegative_int(previous.get("total_bytes")),
                    "workers": {},
                }
            workers = previous.get("workers")
            if not isinstance(workers, dict):
                workers = {}
            current = workers.get(worker_key)
            if not isinstance(current, dict):
                current = {}
            is_new_snapshot = str(current.get("last_snapshot_id") or "") != str(snapshot_id)
            added_physical = (
                _nonnegative_int(stats.get("new_physical_bytes"))
                if is_new_snapshot
                else 0
            )
            worker = {
                "latest_logical_bytes": _nonnegative_int(stats.get("logical_size_bytes")),
                "cumulative_new_physical_bytes": (
                    _nonnegative_int(current.get("cumulative_new_physical_bytes"))
                    + added_physical
                ),
                "captured_file_count": _nonnegative_int(stats.get("captured_file_count")),
                "last_snapshot_id": str(snapshot_id),
                "last_stage_id": str(stage_id),
                "snapshot_kind": str(snapshot_kind),
                "sampled_at_utc": now,
            }
            workers[worker_key] = worker
            baseline_bytes = _nonnegative_int(previous.get("baseline_bytes"))
            if baseline_bytes:
                estimated_bytes = baseline_bytes + sum(
                    _nonnegative_int(value.get("cumulative_new_physical_bytes"))
                    for value in workers.values()
                    if isinstance(value, dict)
                )
            else:
                estimated_bytes = sum(
                    _nonnegative_int(value.get("latest_logical_bytes"))
                    + _nonnegative_int(value.get("cumulative_new_physical_bytes"))
                    for value in workers.values()
                    if isinstance(value, dict)
                )
            payload = {
                "schema_version": 1,
                "sampled_at_utc": now,
                "source": "snapshot_counters",
                "exact": False,
                "total_bytes": estimated_bytes,
                "baseline_bytes": baseline_bytes,
                "coverage": ["worker_workspaces", "snapshot_object_stores"],
                "excluded": ["datasets", "runtime_logs", "reports", "merge_outputs"],
                "workers": workers,
            }
            event = {
                "event": "storage_snapshot_sampled",
                "timestamp_utc": now,
                "worker_id": worker_key,
                "stage_id": str(stage_id),
                "snapshot_id": str(snapshot_id),
                "snapshot_kind": str(snapshot_kind),
                "logical_size_bytes": worker["latest_logical_bytes"],
                "new_physical_bytes": added_physical,
                "estimated_task_bytes": estimated_bytes,
            }
            _append_event(log_dir, event)
            _atomic_json_write(latest_path, payload)
        return True
    except OSError:
        return False


def _excluded_relative_paths(root: Path, paths: Iterable[str | Path]) -> tuple[Path, ...]:
    excluded: list[Path] = []
    resolved_root = root.resolve(strict=False)
    for raw in paths:
        if not str(raw).strip() or str(raw).strip().casefold() == "none":
            continue
        try:
            candidate = Path(raw).expanduser().resolve(strict=False)
            excluded.append(candidate.relative_to(resolved_root))
        except (OSError, ValueError):
            continue
    return tuple(excluded)


def _is_excluded_relative(path: Path, explicit: tuple[Path, ...]) -> bool:
    if path.name.casefold() in _DATASET_DIR_NAMES:
        return True
    return any(path == blocked or blocked in path.parents for blocked in explicit)


def measure_generated_storage(
    task_root_dir: str | Path,
    *,
    excluded_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Measure task-generated storage without entering datasets or symlink targets."""

    started = time.monotonic()
    root = Path(task_root_dir).resolve(strict=False)
    explicit = _excluded_relative_paths(root, excluded_paths)
    stack = [root]
    seen_inodes: set[tuple[int, int]] = set()
    apparent_bytes = 0
    allocated_bytes = 0
    file_count = 0
    directory_count = 0
    symlink_count = 0
    excluded_directory_count = 0
    error_count = 0
    while stack:
        path = stack.pop()
        try:
            info = path.stat(follow_symlinks=False)
        except OSError:
            error_count += 1
            continue
        inode = (int(info.st_dev), int(info.st_ino))
        if inode not in seen_inodes:
            seen_inodes.add(inode)
            apparent_bytes += max(0, int(info.st_size))
            blocks = getattr(info, "st_blocks", None)
            allocated_bytes += (
                max(0, int(info.st_size))
                if blocks is None
                else max(0, int(blocks)) * 512
            )
        mode = info.st_mode
        if stat.S_ISLNK(mode):
            symlink_count += 1
            continue
        if not stat.S_ISDIR(mode):
            file_count += 1
            continue
        directory_count += 1
        try:
            relative = path.relative_to(root)
            with os.scandir(path) as directory:
                entries = list(directory)
        except OSError:
            error_count += 1
            continue
        for entry in entries:
            child = Path(entry.path)
            child_relative = relative / entry.name
            if (
                len(child_relative.parts) == 2
                and child_relative.parts[0] == "task_logs"
                and entry.name in _TELEMETRY_FILE_NAMES
            ):
                continue
            if entry.is_dir(follow_symlinks=False) and _is_excluded_relative(
                child_relative, explicit
            ):
                excluded_directory_count += 1
                continue
            stack.append(child)
    return {
        "apparent_bytes": apparent_bytes,
        "allocated_bytes": allocated_bytes,
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "unique_inode_count": len(seen_inodes),
        "excluded_directory_count": excluded_directory_count,
        "error_count": error_count,
        "scan_duration_ms": round((time.monotonic() - started) * 1000, 3),
    }


def record_final_storage(
    *,
    task_root_dir: str | Path,
    excluded_paths: Iterable[str | Path] = (),
) -> bool:
    """Write one terminal physical-usage calibration for generated task data."""

    task_root = Path(task_root_dir)
    if not task_root.is_dir():
        return False
    log_dir = task_root / "task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    latest_path = log_dir / STORAGE_LATEST_FILENAME
    try:
        measured = measure_generated_storage(task_root, excluded_paths=excluded_paths)
        now = _timestamp_utc()
        with _locked(log_dir):
            previous = _read_json(latest_path)
            payload = {
                "schema_version": 1,
                "sampled_at_utc": now,
                "source": "final_generated_storage_scan",
                "exact": measured["error_count"] == 0,
                "total_bytes": measured["allocated_bytes"],
                "measurement": measured,
                "excluded": [
                    "dataset_directories",
                    "configured_input_data",
                    "symlink_targets",
                    "storage_telemetry_sidecars",
                ],
                "workers": previous.get("workers", {}),
            }
            event = {
                "event": "storage_final_sampled",
                "timestamp_utc": now,
                **measured,
            }
            _append_event(log_dir, event)
            _atomic_json_write(latest_path, payload)
        return True
    except OSError:
        return False


__all__ = [
    "STORAGE_EVENTS_FILENAME",
    "STORAGE_LATEST_FILENAME",
    "measure_generated_storage",
    "record_final_storage",
    "record_snapshot_storage",
]
