"""Share the TUI projection; read lineage only for the selected task."""

import csv
import json
import time
from pathlib import Path

from scienceflow.interfaces.ui.monitor_trace.presentation.builder import (
    build_task_trace,
)
from scienceflow.interfaces.ui.research.control.tasks.projection import TaskProjection
from scienceflow.interfaces.ui.research.control.tasks.records import TaskIndex
from scienceflow.interfaces.ui.research.resources import ResourceMonitor
from scienceflow.research.reporting import generate_task_report, read_report_manifest
from scienceflow.runtime.parallel.control import get_run, list_runs
from scienceflow.runtime.parallel.control.selection import workspace_aliases

from .projection import lineage, timestamped_points

_TERMINAL_STATES = frozenset({"completed", "failed", "stopped", "interrupted"})


class WorkspaceMonitor:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.projection = TaskProjection()
        self.resources = ResourceMonitor(self.workspace)
        self.rows = {}
        self.cache = {}
        self.terminal_details = {}
        self.registry_rows = {}
        self.registry_loaded = False
        self.index_entries = []
        self.index_signature = None

    def _task_index_signature(self):
        path = self.workspace / ".scienceflow/task-board/control.json"
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def _load_tasks(self, *, discover=False):
        signature = self._task_index_signature()
        reload_registry = (
            discover
            or not self.registry_loaded
            or signature != self.index_signature
        )
        if reload_registry:
            rows = list_runs()
            self.registry_rows = {row["run_id"]: row for row in rows}
            self.index_entries = TaskIndex(self.workspace).snapshot()
            self.index_signature = signature
            self.registry_loaded = True
            return rows, list(self.index_entries)
        for run_id, row in tuple(self.registry_rows.items()):
            if self.detail_is_active_run(row):
                try:
                    self.registry_rows[run_id] = get_run(run_id)
                except (OSError, ValueError, KeyError):
                    continue
        rows = sorted(
            self.registry_rows.values(),
            key=lambda row: row.get("created_at", 0),
            reverse=True,
        )
        return rows, list(self.index_entries)

    @staticmethod
    def detail_is_active_run(row):
        return bool(
            row.get("alive")
            and str(row.get("status") or "").lower() not in _TERMINAL_STATES
        )

    def detail_is_active(self, key):
        row = self.rows.get(key)
        return bool(
            row
            and row.get("alive")
            and str(row.get("status") or "").lower()
            not in _TERMINAL_STATES
        )

    def _terminal_detail_signature(self, key):
        row = self.rows.get(key)
        if not row or self.detail_is_active(key):
            return None
        status = str(row.get("status") or "").lower()
        if status not in _TERMINAL_STATES:
            return None
        return (
            row.get("run_id"),
            row.get("attempt", 0),
            status,
            row.get("finished_at"),
        )

    def detail_needs_refresh(self, key):
        if self.detail_is_active(key):
            return True
        signature = self._terminal_detail_signature(key)
        return signature is not None and self.terminal_details.get(key) != signature

    def task_root(self, key):
        row = self.rows.get(key)
        roots = row.get('task_roots') if row else None
        if not roots:
            return None
        root = Path(roots[0]).expanduser().resolve()
        return root if root.is_dir() else None

    def report(self, key):
        row = self.rows.get(key)
        root = self.task_root(key)
        if row is None or root is None:
            return {'available': False, 'status': 'missing'}
        manifest = read_report_manifest(root)
        if manifest is None:
            return {
                'available': False,
                'status': 'missing',
                'terminal': not self.detail_is_active(key),
            }
        return dict(manifest, available=True, terminal=True)

    def generate_report(self, key):
        row = self.rows.get(key)
        root = self.task_root(key)
        if row is None or root is None:
            raise ValueError('Task report workspace is unavailable.')
        if self.detail_is_active(key):
            raise ValueError('Final report is available after the task ends.')
        return dict(generate_task_report(row, root), available=True, terminal=True)

    def collect(self, *, discover=False):
        started = time.monotonic()
        rows, entries = self._load_tasks(discover=discover)
        known = {entry.get("run_id") for entry in entries}
        for row in reversed(rows):
            if row["run_id"] not in known and self.workspace in workspace_aliases(row):
                entries.append(
                    {
                        "number": max((e["number"] for e in entries), default=0) + 1,
                        "run_id": row["run_id"],
                        "name": (row.get("draft") or {}).get("exp_id", "Research"),
                        "status": row["status"],
                    }
                )
                known.add(row["run_id"])
        by_id = {row["run_id"]: row for row in rows}
        result = []
        self.rows = {}
        for entry in entries:
            row = by_id.get(entry.get("run_id"))
            key = str(entry["number"])
            try:
                item = self.projection.item(entry, row)
                data = item.get("monitor") or {
                    "name": entry["name"],
                    "status": entry["status"],
                }
                reports = row.get('final_reports') if row else None
                if isinstance(reports, list) and reports:
                    data['report'] = reports[0]
                result.append(dict(data, key=key))
                if row:
                    self.rows[key] = row
            except (OSError, ValueError, TypeError, KeyError) as exc:
                result.append(
                    {
                        "key": key,
                        "name": entry["name"],
                        "status": "unavailable",
                        "error": type(exc).__name__,
                    }
                )
        resources = self.resources.sample()
        return {
            "tasks": result,
            "resources": resources,
            "updated_at": time.time(),
            "scan_ms": round((time.monotonic() - started) * 1000, 1),
        }

    def detail(self, key):
        row = self.rows.get(key)
        if not row or not row.get("task_roots"):
            return {"points": [], "events": [], "lineage": []}
        cached = self.cache.get(key)
        terminal_signature = self._terminal_detail_signature(key)
        if (
            cached
            and terminal_signature is not None
            and self.terminal_details.get(key) == terminal_signature
        ):
            return cached[2]
        root = Path(row["task_roots"][0])
        # This expensive projection is requested only for the open task page.
        stamp = time.monotonic()
        sources = [
            root / "task_logs" / name
            for name in (
                "lhr_stage_performance.csv",
                "lhr_state.json",
                "lhr_events.jsonl",
                "resource/resource_events.jsonl",
            )
        ]
        signature = []
        for source in sources:
            try:
                stat = source.stat()
                signature.append((stat.st_size, stat.st_mtime_ns))
            except OSError:
                signature.append(None)
        identity = (row["run_id"], row.get("attempt"), signature)
        cached = self.cache.get(key)
        if cached and cached[0] == identity:
            if terminal_signature is not None:
                self.terminal_details[key] = terminal_signature
            return cached[2]
        trace = build_task_trace(
            run_id=row["run_id"],
            gpu_list="",
            monitor_state_path=root / "logs/monitor_state.json",
        )
        path = root / "task_logs/lhr_stage_performance.csv"
        rows = []
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        lineage_events = []
        events_path = root / "task_logs/lhr_events.jsonl"
        if events_path.is_file():
            with events_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if (
                        isinstance(event, dict)
                        and event.get("event")
                        in {
                            "estra_decision",
                            "estra_keep_current_compacted",
                            "estra_stage_switched",
                        }
                    ):
                        lineage_events.append(event)
        data = {
            "points": timestamped_points(trace.points, rows),
            "events": [event.to_dict() for event in trace.events],
            "lineage": lineage(rows, lineage_events),
            "started_at": row.get("started_at"),
            "metric": trace.metric_name,
            "lower_is_better": trace.lower_is_better,
        }
        self.cache[key] = (identity, stamp, data)
        if terminal_signature is not None:
            self.terminal_details[key] = terminal_signature
        else:
            self.terminal_details.pop(key, None)
        return data
