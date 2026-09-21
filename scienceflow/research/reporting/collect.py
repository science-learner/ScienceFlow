"""Collect final report evidence from the same sources as the live monitor."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scienceflow.interfaces.ui.monitor_trace.presentation.builder import (
    build_task_trace,
)
from scienceflow.interfaces.ui.research.control.tasks.projection import TaskProjection
from scienceflow.interfaces.ui.web_monitor.projection import lineage

from .models import ReportDocument

_SUMMARY_LIMIT = 3000
_DESCRIPTION_LIMIT = 12_000
_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)\bauthorization\s*[:=]\s*bearer\s+[^\s,;]+"),
        "Authorization: Bearer <redacted>",
    ),
    (re.compile(r"(?i)\bbearer\s+[^\s,;]+"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b(?:sk|key)-[a-z0-9_-]{8,}\b"), "<redacted-key>"),
    (
        re.compile(r"(?i)\b(api[_-]?key)(\s*[:=]\s*)[^\s,;]+"),
        r"\1=<redacted>",
    ),
    (re.compile(r"https?://\S+"), "<redacted-url>"),
)


def _redact(value: object) -> str:
    text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[str(key)] = _redact_mapping(item)
        elif isinstance(item, list):
            result[str(key)] = [
                _redact_mapping(entry) if isinstance(entry, dict) else _redact(entry)
                for entry in item
            ]
        elif isinstance(item, str):
            result[str(key)] = _redact(item)
        else:
            result[str(key)] = item
    return result


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        try:
            result[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _state(root: Path) -> dict[str, Any]:
    for path in (
        root / "task_logs/state.json",
        root / "logs/state.json",
        root / "state.json",
    ):
        value = _json(path)
        if value:
            return value
    return {}


def _result_summaries(root: Path) -> list[dict[str, str]]:
    values = []
    for path in sorted(root.glob("workers/*/workspace/result.md")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            values.append(
                {
                    "worker": path.parents[2].name,
                    "text": _redact(text[:_SUMMARY_LIMIT])
                    + ("…" if len(text) > _SUMMARY_LIMIT else ""),
                }
            )
    return values


def _task_description(root: Path) -> str:
    runtime = root / "task_runtime"
    paths = [runtime / "description.md", runtime / "description_lite.md"]
    if runtime.is_dir():
        paths.extend(sorted(runtime.glob("*/description.md")))
        paths.extend(sorted(runtime.glob("*/*/description.md")))
    paths.extend(
        [
            root / "description.md",
            root / "workers/w00/workspace/description.md",
        ]
    )
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            suffix = "…" if len(text) > _DESCRIPTION_LIMIT else ""
            return _redact(text[:_DESCRIPTION_LIMIT]) + suffix
    return ""


def _merge_summary(root: Path) -> dict[str, str]:
    try:
        text = (root / "merge/merge_report.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    allowed = {
        "status",
        "merge_mode",
        "agent_status",
        "packed_candidate_count",
        "required_final_count",
        "final_count",
        "valid_final_count",
        "requirement_met",
        "merge_budget_sec",
    }
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in allowed:
            values[key.strip()] = _redact(value.strip())
    return values


def _finalists(root: Path, *, lower_is_better: bool) -> list[dict[str, Any]]:
    values = []
    for path in sorted((root / "merge/finals").glob("final_*/eval_result.json")):
        result = _json(path)
        raw_metric = result.get("metric_value")
        try:
            metric = float(raw_metric)
        except (TypeError, ValueError):
            metric = None
        if metric is not None and not math.isfinite(metric):
            metric = None
        try:
            note = (path.parent / "merge_report.md").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            note = ""
        source = ""
        match = re.search(r"\((W\d+:[^)]+)\)", note)
        if match:
            source = match.group(1)
        values.append(
            {
                "candidate_id": str(result.get("candidate_id") or path.parent.name),
                "metric": metric,
                "valid": bool(result.get("validation_ok")),
                "eligible": bool(result.get("selection_eligible")),
                "evaluator": str(result.get("evaluator_status") or "unknown"),
                "artifact": str(result.get("artifact_path") or ""),
                "artifact_sha": str(result.get("artifact_sha") or "")[:12],
                "source": source,
                "note": _redact(note[:_SUMMARY_LIMIT]),
                "directory": str(path.parent),
            }
        )
    values.sort(
        key=lambda item: (
            item["metric"] is None,
            item["metric"] if lower_is_better else -(item["metric"] or 0),
        )
    )
    for rank, value in enumerate(values, 1):
        value["rank"] = rank
    return values


def _artifact_preview(root: Path, finalists: list[dict[str, Any]]) -> dict[str, Any]:
    if not finalists:
        return {}
    best = finalists[0]
    artifact = str(best.get("artifact") or "")
    if not artifact:
        return {}
    path = (Path(str(best["directory"])) / artifact).resolve()
    if not path.is_relative_to(root):
        return {}
    value = _json(path)
    circles = value.get("circles")
    if not isinstance(circles, list) or not circles or len(circles) > 500:
        return {}
    clean = []
    for circle in circles:
        if not isinstance(circle, list) or len(circle) != 3:
            return {}
        try:
            x, y, radius = (float(number) for number in circle)
        except (TypeError, ValueError):
            return {}
        if not all(math.isfinite(number) for number in (x, y, radius)):
            return {}
        clean.append([x, y, radius])
    return {"kind": "circle_packing", "circles": clean}


def _stage_lineage(root: Path) -> list[dict[str, Any]]:
    rows = []
    path = root / "task_logs/lhr_stage_performance.csv"
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError:
        pass
    events = []
    path = root / "task_logs/lhr_events.jsonl"
    try:
        with path.open(encoding="utf-8") as stream:
            for raw in stream:
                try:
                    event = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        pass
    return [_redact_mapping(node) for node in lineage(rows, events)]


def collect_report(record: dict[str, Any], root: str | Path) -> ReportDocument:
    root = Path(root).expanduser().resolve()
    state = _state(root)
    draft = record.get("draft") if isinstance(record.get("draft"), dict) else {}
    name = str(state.get("exp_id") or draft.get("exp_id") or root.name)
    run_id = str(record.get("run_id") or state.get("run_id") or root.name)
    status = str(record.get("status") or state.get("status") or "unknown")
    row = {
        **record,
        "run_id": run_id,
        "status": status,
        "alive": False,
        "task_roots": [str(root)],
        "draft": draft,
        "started_at": record.get("started_at") or state.get("run_started_at"),
        "finished_at": record.get("finished_at") or state.get("stopped_at"),
    }
    entry = {"number": 1, "name": name, "status": status}
    try:
        item = TaskProjection().item(entry, row)
        monitor = item.get("monitor") or {}
    except (OSError, ValueError, TypeError, KeyError):
        monitor = {}
    try:
        trace = build_task_trace(
            run_id=run_id,
            gpu_list=str(state.get("resolved_gpu") or draft.get("gpu_list") or ""),
            monitor_state_path=root / "logs/monitor_state.json",
        )
    except (OSError, ValueError, TypeError, KeyError):
        trace = SimpleNamespace(
            points=[],
            events=[],
            metric_name=str(draft.get("metric_name") or "metric"),
            lower_is_better=bool(draft.get("lower_is_better")),
        )
    points = [_redact_mapping(point.to_dict()) for point in trace.points]
    stages = _stage_lineage(root)
    if not stages:
        stages = [
            {
                "id": point.candidate_id,
                "candidate": point.candidate_id,
                "worker": point.worker_id,
                "stage": point.stage_id,
                "lineage": "",
                "parent": "",
                "restored": "",
                "metric": point.metric,
                "valid": point.valid_comparable,
                "kind": "evaluated_stage",
            }
            for point in trace.points
        ]
    started_at = row.get("started_at")
    finished_at = row.get("finished_at")
    elapsed = monitor.get("elapsed_sec")
    if elapsed is None:
        elapsed = state.get("charged_elapsed_sec", state.get("elapsed_sec", 0))
    lower_is_better = bool(monitor.get("lower_is_better", trace.lower_is_better))
    finalists = _finalists(root, lower_is_better=lower_is_better)
    artifact_preview = _artifact_preview(root, finalists)
    for finalist in finalists:
        finalist.pop("directory", None)
    return ReportDocument(
        schema_version=3,
        generated_at_utc=datetime.now(UTC).isoformat(),
        run_id=run_id,
        attempt=int(record.get("attempt") or 1),
        task_name=name,
        status=status,
        started_at=float(started_at) if isinstance(started_at, (int, float)) else None,
        finished_at=float(finished_at)
        if isinstance(finished_at, (int, float))
        else None,
        elapsed_sec=float(elapsed or 0),
        budget_sec=float(
            monitor.get("budget_sec")
            or draft.get("wall_clock_sec")
            or state.get("time_limit_sec")
            or 0
        ),
        metric_name=str(monitor.get("metric") or trace.metric_name or "metric"),
        lower_is_better=lower_is_better,
        best=str(monitor["best"]) if monitor.get("best") is not None else "—",
        evaluated=int(monitor.get("evaluated") or 0),
        valid=int(monitor.get("valid") or 0),
        workers=[_redact(value) for value in monitor.get("workers") or ()],
        worker_counts=_int_mapping(monitor.get("worker_counts")),
        estra=_int_mapping(monitor.get("estra")),
        eec=_int_mapping(monitor.get("eec")),
        eec_total=int(monitor.get("eec_total") or 0),
        gpu=_redact(monitor.get("gpu") or "—"),
        usage=_redact(monitor.get("usage") or ""),
        cost=_redact(monitor.get("cost") or "—"),
        failure_kind=_redact(state.get("failure_kind") or record.get("error") or ""),
        error=_redact(state.get("error") or record.get("error") or ""),
        resume_retriable=bool(state.get("resume_retriable")),
        stop_reason=_redact(state.get("stop_reason") or ""),
        points=points,
        events=[_redact_mapping(event.to_dict()) for event in trace.events],
        stages=[_redact_mapping(stage) for stage in stages],
        recent=[_redact(value) for value in monitor.get("recent") or ()],
        result_summaries=_result_summaries(root),
        task_description=_task_description(root),
        finalists=finalists,
        merge=_merge_summary(root),
        artifact_preview=artifact_preview,
    )
