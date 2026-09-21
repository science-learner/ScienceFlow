# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""LNR coordinator responsibility: workspace metric extraction and deterministic value projection.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    Path,
    _validation_leakage_reason,
    classify_metric_semantics,
    find_node_log_path,
    hashlib,
    json,
    normalize_shell_command,
    re,
)


def _count_memory_records(self) -> int:
    p = self.memory_dir / "ScienceAgent" / "short_term.json"
    if not p.is_file():
        return 0
    try:
        return sum(
            1
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        )
    except OSError:
        return 0


def _workspace_metric_audit_source_text(
    self, *, source_rel: str, max_files: int = 40, max_total_chars: int = 120_000
) -> str:
    """Collect bounded workspace Python source for metric-semantics audit."""
    parts: list[str] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        nonlocal parts
        if len(seen) >= max_files:
            return
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path
        if resolved in seen or not path.is_file():
            return
        try:
            rel = path.relative_to(self.workspace_dir)
        except ValueError:
            rel = path.name
        rel_text = str(rel).replace("\\", "/")
        if rel_text.startswith(
            (".venv/", "dataset/", "artifacts/", ".git/", ".logs/", "logs/", ".memory/")
        ):
            return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        remaining = max_total_chars - sum(len(x) for x in parts)
        if remaining <= 0:
            return
        seen.add(resolved)
        parts.append(f"\n# FILE: {rel_text}\n{text[:remaining]}")

    add(self.workspace_dir / source_rel)
    for path in sorted(self.workspace_dir.glob("*.py")):
        add(path)
    return "\n".join(parts)


def _compact_run_command(command: str, *, max_chars: int = 900) -> str:
    raw = str(command or "").strip()
    if not raw:
        return ""
    compact = normalize_shell_command(raw)
    compact = re.sub(r"\s+", " ", compact or raw).strip()
    if len(compact) > max_chars:
        return compact[: max(0, max_chars - 24)].rstrip() + " ... [truncated]"
    return compact


def _stage_run_signature(metric_event: dict[str, Any]) -> str:
    raw_command = re.sub(r"\s+", " ", str(metric_event.get("bash_cmd") or "")).strip()
    payload = {
        "solution_sha": str(metric_event.get("solution_sha") or ""),
        "submission_sha": str(metric_event.get("submission_sha") or ""),
        "artifact_sha": str(metric_event.get("artifact_sha") or ""),
        "metric_value": metric_event.get("metric_value"),
        "metric_name": str(metric_event.get("metric_name") or ""),
        "run_command": _compact_run_command(str(metric_event.get("bash_cmd") or "")),
        "raw_command": raw_command,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()


def _metric_event_from_workspace(self) -> dict[str, Any] | None:
    p = find_node_log_path(self.workspace_dir, "fullrun_tail_snapshot.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        metric_value = float(data.get("metric_value"))
    except (TypeError, ValueError):
        if bool(self.lhr.stage_commit_require_metric):
            return None
        metric_value = None
    if metric_value is not None and (metric_value != metric_value):
        return None
    direction = self._metric_lower_is_better_decision(
        {
            "declared_lower_is_better": data.get("lower_is_better"),
            "lower_is_better": data.get("lower_is_better"),
            "metric_name": data.get("metric_name") or "Final Validation Score",
            "val_score_type": data.get("val_score_type"),
            "metric_protocol": data.get("metric_protocol"),
            "metric_source_note": data.get("metric_source_note"),
        },
        fallback=data.get("lower_is_better"),
    )
    lower_is_better = bool(direction["lower_is_better"])
    validation_ok = data.get("validation_ok")
    execution_mode_hint = str(data.get("execution_mode") or "").strip()
    source_rel = (
        str(data.get("solution_path") or "").replace("\\", "/").strip().lstrip("/")
    )
    if not source_rel:
        source_rel = "" if execution_mode_hint == "inline_bash" else "solution.py"
    if source_rel and ".." in source_rel.split("/"):
        source_rel = "solution.py"
    validation_issue = ""
    audit_source_text = self._workspace_metric_audit_source_text(source_rel=source_rel)
    if validation_ok is not False and bool(
        getattr(self.lhr, "metric_validation_leakage_guard_enabled", True)
    ):
        if source_rel:
            try:
                source_text = (self.workspace_dir / source_rel).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                source_text = ""
        else:
            source_text = audit_source_text
        reason = _validation_leakage_reason(
            source_text, str(data.get("stdout_tail") or "")
        )
        if reason:
            validation_issue = reason
            validation_ok = False
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_metric_rejected",
                    "reason": reason,
                    "metric_value": metric_value,
                    "metric_name": str(
                        data.get("metric_name") or "Final Validation Score"
                    ),
                    "snapshot_path": str(p),
                },
            )
    semantics = classify_metric_semantics(
        metric_value=metric_value,
        data=data,
        workspace_source_text=audit_source_text,
        source_changed=data.get("source_changed"),
    )
    if not semantics.get("route_id") and data.get("solution_sha"):
        semantics["route_id"] = f"source:{str(data.get('solution_sha'))[:16]}"
    wall_sec = data.get("wall_sec")
    duration_sec = data.get("duration_sec")
    run_time_sec = wall_sec if wall_sec not in (None, "") else duration_sec
    bash_cmd = str(data.get("bash_cmd") or "")
    event = {
        "metric_value": metric_value,
        "metric_name": str(data.get("metric_name") or "Final Validation Score"),
        "lower_is_better": lower_is_better,
        "declared_lower_is_better": direction.get("declared_lower_is_better"),
        "metric_direction_source": direction.get("metric_direction_source"),
        "metric_direction_conflict": direction.get("metric_direction_conflict"),
        "solution_sha": str(data.get("solution_sha") or ""),
        "solution_path": source_rel,
        "submission_sha": str(data.get("submission_sha") or ""),
        "bash_cmd": bash_cmd,
        "validation_ok": validation_ok,
        "validation_issue": validation_issue,
        "submission_validation_ok": data.get("submission_validation_ok"),
        "submission_status": str(data.get("submission_status") or ""),
        "execution_mode": execution_mode_hint,
        "wall_sec": wall_sec,
        "duration_sec": duration_sec,
        "run_time_sec": run_time_sec,
        "snapshot_path": str(p),
        "stage_signal_kind": str(data.get("stage_signal_kind") or ""),
        "candidate_artifact_attached": data.get("candidate_artifact_attached"),
    }
    event.update(semantics)
    return event


def _fmt_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}" if value == value else ""
    return str(value)


def _csv_bool(value: Any, *, default: bool = True) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    if text in {"0", "false", "no", "n"}:
        return False
    if text in {"1", "true", "yes", "y"}:
        return True
    return default


def _csv_metric(value: Any) -> float | None:
    try:
        metric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if metric != metric:
        return None
    return metric
