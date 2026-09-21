"""Read-only LNR monitor responsibility: artifact and process readers."""

from __future__ import annotations

import datetime as _dt
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scienceflow.interfaces.ui.monitor.lnr.rendering.formatting import _to_float


def _resume_prior_charged_elapsed_sec(final_state: dict[str, Any]) -> float:
    value = _to_float(final_state.get("resume_prior_charged_elapsed_sec"))
    if value is None:
        return 0.0
    return max(0.0, value)


def _load_current_resource_snapshot(task_root: Path) -> dict[str, Any]:
    """Return current lease/waiter counts without changing resource state."""
    resource_dir = Path(task_root) / "task_logs" / "resource"
    gpu_leases = _load_json_dict(resource_dir / "gpu_leases.json")
    resource_state = _load_json_dict(resource_dir / "resource_state.json")
    has_snapshot = bool(gpu_leases or resource_state)
    lease_count = max(
        _collection_len(gpu_leases.get("leases")),
        _collection_len(resource_state.get("leases")),
    )
    waiter_count = max(
        _collection_len(gpu_leases.get("waiters")),
        _collection_len(resource_state.get("waiters")),
    )
    waiter_workers = sorted(
        _worker_ids_from_collection(gpu_leases.get("waiters"))
        | _worker_ids_from_collection(resource_state.get("waiters"))
    )
    return {
        "has_snapshot": has_snapshot,
        "active_lease_count": lease_count,
        "active_waiter_count": waiter_count,
        "active_waiter_workers": waiter_workers,
    }

def _filter_stale_wait_states(worker_states: Any, current: dict[str, Any]) -> list[dict[str, Any]]:
    states = [s for s in worker_states if isinstance(s, dict)]
    if not current.get("has_snapshot"):
        return states
    waiter_count = int(current.get("active_waiter_count") or 0)
    if waiter_count <= 0:
        return [s for s in states if s.get("kind") != "wait"]
    waiter_workers = {
        str(w).upper()
        for w in current.get("active_waiter_workers") or []
        if str(w).strip()
    }
    if not waiter_workers:
        return states
    return [
        s
        for s in states
        if s.get("kind") != "wait" or str(s.get("worker_id") or "").upper() in waiter_workers
    ]

def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

def _collection_len(value: Any) -> int:
    if isinstance(value, (dict, list, tuple, set)):
        return len(value)
    return 0

def _worker_ids_from_collection(value: Any) -> set[str]:
    workers: set[str] = set()
    items: Iterable[Any]
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = value
    else:
        return workers
    for item in items:
        key: Any = ""
        payload: Any = item
        if isinstance(item, tuple) and len(item) == 2:
            key, payload = item
        candidates = [key]
        if isinstance(payload, dict):
            candidates.extend(
                [
                    payload.get("worker_id"),
                    payload.get("worker"),
                    payload.get("job_id"),
                    payload.get("command_id"),
                ]
            )
        for candidate in candidates:
            worker = _worker_id_from_text(candidate)
            if worker:
                workers.add(worker)
    return workers

def _worker_id_from_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith("W") and len(text) >= 3 and text[1:3].isdigit():
        return text[:3]
    for part in text.replace(":", " ").replace("/", " ").split():
        if part.startswith("W") and len(part) >= 3 and part[1:3].isdigit():
            return part[:3]
    return ""

def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""

def _live_process_summary_for_root(task_root: Path) -> dict[str, Any]:
    """Return live process count and oldest start time for *task_root*.

    Direct resume runs are launched from the project checkout, so their cwd is
    not inside the task workspace. They still carry ``--workspace <task_root>``
    in argv. Check both cwd and cmdline so the monitor does not display a
    historical terminal status while a resume process is active.  When a resume
    is active, progress should be relative to the new process lifetime, not the
    old task's completed state.
    """
    root_text = str(task_root)
    proc = Path("/proc")
    boot_time = _linux_boot_time()
    count = 0
    oldest_start: float | None = None
    try:
        entries = list(proc.iterdir())
    except OSError:
        return {"count": 0, "oldest_start_ts": None}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        matched = False
        try:
            cwd = (entry / "cwd").readlink()
        except OSError:
            cwd = None
        if cwd is not None and root_text in str(cwd):
            matched = True
        else:
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    "utf-8",
                    errors="ignore",
                )
            except OSError:
                cmdline = ""
            matched = root_text in cmdline
        if not matched:
            continue
        count += 1
        start_ts = _proc_start_timestamp(entry, boot_time=boot_time)
        if start_ts is not None and (oldest_start is None or start_ts < oldest_start):
            oldest_start = start_ts
    return {"count": count, "oldest_start_ts": oldest_start}

def _count_live_processes_for_root(task_root: Path) -> int:
    return int(_live_process_summary_for_root(task_root).get("count") or 0)

def _elapsed_from_live_process_summary(
    summary: dict[str, Any],
    final_state: dict[str, Any] | None = None,
) -> float | None:
    start_ts = _to_float(summary.get("oldest_start_ts"))
    if start_ts is None:
        return None
    segment_elapsed = max(0.0, _dt.datetime.now().timestamp() - start_ts)
    prior_elapsed = _resume_prior_charged_elapsed_sec(final_state or {})
    return prior_elapsed + segment_elapsed

def _linux_boot_time() -> float | None:
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, IndexError, ValueError):
        return None
    return _dt.datetime.now().timestamp() - uptime

def _proc_start_timestamp(proc_entry: Path, *, boot_time: float | None) -> float | None:
    if boot_time is None:
        return None
    try:
        text = (proc_entry / "stat").read_text(encoding="utf-8", errors="ignore")
        after_comm = text[text.rfind(")") + 2 :].split()
        start_ticks = float(after_comm[19])
        ticks_per_sec = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except (OSError, IndexError, KeyError, TypeError, ValueError):
        return None
    if ticks_per_sec <= 0:
        return None
    return boot_time + (start_ticks / ticks_per_sec)

__all__ = tuple(name for name in globals() if not name.startswith("__"))
