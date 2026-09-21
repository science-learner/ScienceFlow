# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: wire events, workspace preparation, logs, and snapshot identity.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
    _stage_run_signature,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    LHR_EVENTS_JSONL,
    LHR_UNIFIED_ONLY_JSONL,
    Path,
    StageSnapshot,
    _deterministic_gate_timestamp,
    atomic_write,
    attach_stage_interaction_handlers,
    attach_workspace_interaction_logger,
    close_workspace_interaction_logger,
    ensure_stage_log_dir,
    ensure_workspace_source_git,
    json,
    logger,
    next_stage_id,
    normalize_workspace_git_track_globs,
    parse_stage_cards,
    read_ledger,
    reset_stage_log_dir,
    shutil,
    stage_log_dir,
    time,
)


def _jsonl_roots(self) -> tuple[Path, ...]:
    # LHR control-plane logs stay outside the agent workspace. The workspace
    # .logs/ tree is reserved for interaction/traj logs and run evidence.
    return (self.log_dir,)


def _jsonl(self, name: str, record: dict[str, Any]) -> None:
    record = {"timestamp": time.time(), **record}
    if name not in LHR_UNIFIED_ONLY_JSONL:
        for root in self._jsonl_roots():
            try:
                root.mkdir(parents=True, exist_ok=True)
                with (root / name).open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
            except OSError:
                logger.debug("[lnr] could not append %s", name, exc_info=True)
    self._mirror_state_event(name=name, record=record)


def _record_stage_lifecycle_trace(self, record: dict[str, Any]) -> None:
    self._jsonl("lhr_stage_lifecycle_events.jsonl", record)
    journal = getattr(self, "telemetry_journal", None)
    if journal is None:
        return
    try:
        journal.record(
            source="stage_lifecycle",
            event_type=str(record.get("event") or "stage_lifecycle_transition"),
            payload=record,
            run_id=str(
                getattr(self.cfg, "run_id", "") or getattr(self.cfg, "exp_id", "") or ""
            ),
            worker_id=self.worker_id or "W00",
            stage_id=str(record.get("machine_id") or ""),
            event_id=str(record.get("event_id") or ""),
            sequence=int(record.get("sequence") or 0),
        )
    except OSError:
        logger.debug("[lnr] stage correlation telemetry skipped", exc_info=True)


def _record_agent_factory_trace(self, record: dict[str, Any]) -> None:
    self._jsonl("lhr_agent_factory_events.jsonl", record)
    journal = getattr(self, "telemetry_journal", None)
    if journal is None:
        return
    try:
        correlation = (
            record.get("correlation")
            if isinstance(record.get("correlation"), dict)
            else {}
        )
        journal.record(
            source="agent_factory",
            event_type="agent_build",
            payload=record,
            run_id=str(
                getattr(self.cfg, "run_id", "") or getattr(self.cfg, "exp_id", "") or ""
            ),
            worker_id=str(correlation.get("worker_id") or self.worker_id or "W00"),
            event_id=str(record.get("build_id") or ""),
        )
    except OSError:
        logger.debug("[lnr] agent correlation telemetry skipped", exc_info=True)


@staticmethod
def _state_task_type_for_event(event: str) -> str:
    if event.startswith("stage_"):
        return "stage_commit"
    if event.endswith("_estra_check") or event in {
        "estra_decision",
        "estra_invalid",
        "estra_llm_error",
        "estra_main_context_invalid_fallback",
        "estra_fallback_llm_error",
        "estra_deterministic_fallback",
        "estra_no_valid_switch_candidate",
        "text_only_estra_noop",
    }:
        return "estra_decision"
    if event in {
        "estra_keep_current_compacted",
        "estra_stage_switched",
        "estra_failed",
        "state_packet_built",
    }:
        return "estra_restore"
    if event.startswith("worker_") or event.startswith("multi_worker"):
        return "coordinator"
    return "repl_search"


def _relativize_control_payload(self, value: Any) -> Any:
    if isinstance(value, str):
        text = value
        replacements = [
            (str(self.workspace_dir), "workspace"),
            (str(self.root_dir), "."),
            (str(self.task_root_dir), "."),
        ]
        replacements.sort(key=lambda item: len(item[0]), reverse=True)
        for raw, repl in replacements:
            if raw and raw in text:
                text = text.replace(raw, repl)
        return text
    if isinstance(value, dict):
        return {str(k): self._relativize_control_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [self._relativize_control_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(self._relativize_control_payload(v) for v in value)
    return value


def _mirror_state_event(self, *, name: str, record: dict[str, Any]) -> None:
    if name == LHR_EVENTS_JSONL:
        return
    event = str(record.get("event") or Path(name).stem or "event")
    task_type = self._state_task_type_for_event(event)
    if event.endswith("_estra_check") or event.startswith("estra_"):
        stage_id = str(
            record.get("latest_stage")
            or record.get("stage_id")
            or record.get("target_stage")
            or ""
        )
    else:
        stage_id = str(record.get("stage_id") or record.get("target_stage") or "")
    task_id = f"{task_type}:{self.worker_id or 'W00'}"
    if stage_id:
        task_id = f"{task_id}:{stage_id}"
    status = ""
    if event == "context_compact_started":
        status = "running"
    elif event == "context_compact_estra_check":
        status = "running"
    elif event == "context_compact_deferred":
        status = "succeeded"
    elif event == "context_compact_finished":
        compact_status = str(record.get("status") or "")
        status = "failed" if compact_status == "failed" else "succeeded"
    elif event == "context_compact_failed":
        status = "failed"
    elif event.endswith("_failed") or event.endswith("_error"):
        status = "failed"
    elif (
        event.endswith("_ok")
        or event.endswith("_captured")
        or event.endswith("_restored")
    ):
        status = "succeeded"
    try:
        payload = self._relativize_control_payload(
            {k: v for k, v in record.items() if k != "timestamp"}
        )
        self.state_machine.append_event(
            event,
            task_type=task_type,
            task_id=task_id,
            status=status,
            payload=payload,
        )
    except OSError:
        logger.debug("[lnr] state-machine event mirror failed", exc_info=True)


def _write_stage_map(self) -> None:
    archived = self._archived_stage_snapshot_index()
    payload = {
        "ledger_filename": self.ledger_filename,
        "stages": {
            sid: {
                "stage_id": snap.stage_id,
                "node_uid": self._snapshot_node_uid(snap),
                "lineage_id": str(
                    getattr(snap, "lineage_id", "")
                    or (
                        snap.source_event.get("lineage_id")
                        if isinstance(snap.source_event, dict)
                        else ""
                    )
                    or ""
                ),
                "snapshot_id": snap.snapshot_id,
                "snapshot_path": str(snap.snapshot_path),
                "metric_value": snap.metric_value,
                "metric_name": snap.metric_name,
                "lower_is_better": snap.lower_is_better,
                "memory_cut": snap.memory_cut,
                "candidate_ready": (
                    snap.source_event.get("candidate_ready")
                    if isinstance(snap.source_event, dict)
                    else None
                ),
                "validation_ok": (
                    snap.source_event.get("validation_ok")
                    if isinstance(snap.source_event, dict)
                    else None
                ),
                "validation_issue": (
                    snap.source_event.get("validation_issue")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "reported_val_score": (
                    snap.source_event.get("reported_val_score")
                    if isinstance(snap.source_event, dict)
                    else None
                ),
                "val_score_type": (
                    snap.source_event.get("val_score_type")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "selection_eligible": (
                    snap.source_event.get("selection_eligible")
                    if isinstance(snap.source_event, dict)
                    else None
                ),
                "selection_score": (
                    snap.source_event.get("selection_score")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "selection_note": (
                    snap.source_event.get("selection_note")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "metric_source_note": (
                    snap.source_event.get("metric_source_note")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "metric_protocol": (
                    snap.source_event.get("metric_protocol")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "train_data_used": (
                    snap.source_event.get("train_data_used")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "metric_eval_data": (
                    snap.source_event.get("metric_eval_data")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "execution_mode": (
                    snap.source_event.get("execution_mode")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "submission_status": (
                    snap.source_event.get("submission_status")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                "duplicate_submission_of_stage": (
                    snap.source_event.get("duplicate_submission_of_stage")
                    if isinstance(snap.source_event, dict)
                    else ""
                ),
                **{
                    key: snap.source_event.get(key)
                    for key in (
                        "artifact_path",
                        "artifact_sha",
                        "metric_validity",
                        "metric_validity_note",
                        "metric_validity_reason_code",
                        "gate_metric_validity",
                        "gate_policy",
                        "gate_policy_version",
                        "gate_action",
                        "gate_accepted",
                        "gate_reason_code",
                        "route_id",
                        "solution_sha",
                        "submission_sha",
                        "submission_snapshot",
                    )
                    if isinstance(snap.source_event, dict)
                },
            }
            for sid, snap in sorted(self.stage_snapshots.items())
        },
        "archive": {
            node_uid: {
                "stage_id": snap.stage_id,
                "node_uid": self._snapshot_node_uid(snap),
                "lineage_id": str(getattr(snap, "lineage_id", "") or ""),
                "snapshot_id": snap.snapshot_id,
                "snapshot_path": str(snap.snapshot_path),
                "metric_value": snap.metric_value,
                "metric_name": snap.metric_name,
                "lower_is_better": snap.lower_is_better,
            }
            for node_uid, snap in sorted(archived.items())
        },
    }
    for root in (self.log_dir,):
        try:
            root.mkdir(parents=True, exist_ok=True)
            atomic_write(
                root / "lhr_stage_map.json",
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )
        except OSError:
            logger.debug("[lnr] stage map write failed", exc_info=True)


def _load_existing_stage_snapshots(self) -> None:
    """Rehydrate committed stage snapshots for process-level resume."""

    # A Stage is durable only after both the ledger append and snapshot
    # capture complete. Repair a process crash between those writes before
    # deriving the active Stage set from the ledger.
    self._recover_stage_commit_transactions()
    discover_all = getattr(self.snapshot_store, "discover_all", None)
    if callable(discover_all):
        self.archived_stage_snapshots = dict(discover_all())
    else:
        self.archived_stage_snapshots = {
            self._archive_snapshot_key(snapshot): snapshot
            for snapshot in self.snapshot_store.discover().values()
        }
    cards = parse_stage_cards(read_ledger(self.ledger_path))
    active = {str(card.stage_id).strip().upper() for card in cards}
    if not active:
        return
    loaded: dict[str, StageSnapshot] = {}
    stage_map = self._read_json_file(self.log_dir / "lhr_stage_map.json")
    stages = (
        stage_map.get("stages") if isinstance(stage_map.get("stages"), dict) else {}
    )
    for stage_id, row in stages.items():
        sid = str(stage_id or "").strip().upper()
        if sid not in active or not isinstance(row, dict):
            continue
        snapshot_path = str(row.get("snapshot_path") or "").strip()
        if not snapshot_path:
            continue
        snapshot = self.snapshot_store.load(snapshot_path)
        if snapshot is not None and snapshot.stage_id == sid:
            loaded[sid] = snapshot
            self._index_archived_stage_snapshot(snapshot)
    for sid, snapshot in self.snapshot_store.discover().items():
        if sid in active and sid not in loaded:
            loaded[sid] = snapshot
            self._index_archived_stage_snapshot(snapshot)
    self.stage_snapshots = loaded
    if loaded:
        latest_sid = sorted(
            loaded,
            key=lambda value: (
                int(value[1:]) if value.startswith("S") and value[1:].isdigit() else -1
            ),
        )[-1]
        latest = loaded[latest_sid]
        source = latest.source_event if isinstance(latest.source_event, dict) else {}
        self.last_captured_solution_sha = str(source.get("solution_sha") or "")
        self.last_captured_run_signature = _stage_run_signature(source)
        lineage_numbers = [
            int(text[1:])
            for snapshot in loaded.values()
            if (text := str(snapshot.lineage_id or "").strip().upper()).startswith("L")
            and text[1:].isdigit()
        ]
        if lineage_numbers:
            self.current_lineage_no = max(lineage_numbers)
            self.current_lineage_id = f"L{self.current_lineage_no:02d}"
    missing = sorted(active - set(loaded))
    self._jsonl(
        "lhr_resume_events.jsonl",
        {
            "event": "stage_snapshots_rehydrated",
            "active_stage_count": len(active),
            "loaded_stage_count": len(loaded),
            "loaded_stages": sorted(loaded),
            "missing_snapshot_stages": missing,
        },
    )


def _prepare_workspace(self) -> None:
    self.workspace_dir.mkdir(parents=True, exist_ok=True)
    self.memory_dir.mkdir(parents=True, exist_ok=True)
    self._ensure_split_logs()
    self._prune_workspace_control_artifacts()
    init_applied = self._prepare_init_workspace()
    desc = self.workspace_dir / "description.md"
    if self.task_desc.strip() and (init_applied or not desc.exists()):
        desc.write_text(self.task_desc.strip() + "\n", encoding="utf-8")
    self._prepare_dataset_symlink()
    self._prepare_workspace_git()


def _init_workspace_path(self) -> str:
    init_cfg = getattr(self.lhr, "init_workspace", None)
    if init_cfg is None:
        return ""
    if isinstance(init_cfg, dict):
        return str(init_cfg.get("workspace_path") or "").strip()
    return str(getattr(init_cfg, "workspace_path", "") or "").strip()


def _prepare_init_workspace(self) -> bool:
    raw_path = self._init_workspace_path()
    self.initial_workspace_state = ""
    if not raw_path:
        return False
    if (self.memory_dir / "ScienceAgent").is_dir():
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "init_workspace_skipped",
                "reason": "existing_agent_memory",
                "source_workspace": raw_path,
            },
        )
        return False
    result = self.workspace_service.prepare_from(raw_path)
    self.initial_workspace_state = result.initial_workspace_state or (
        "This run starts from files copied from a previous workspace. "
        "Inspect the existing files before changing the route."
    )
    self._jsonl(
        "lhr_stage_events.jsonl",
        {
            "event": "init_workspace_applied",
            "source_workspace": result.source_workspace,
            "applied": result.applied,
            "copied_file_count": result.copied_file_count,
            "skipped_entry_count": result.skipped_entry_count,
            "initial_workspace_state_path": result.initial_workspace_state_path,
            "initial_workspace_state_chars": len(result.initial_workspace_state or ""),
        },
    )
    return bool(result.applied)


def _prepare_workspace_git(self) -> None:
    enabled = bool(getattr(self.lhr, "workspace_git_enabled", True))
    track_globs = normalize_workspace_git_track_globs(
        getattr(self.lhr, "workspace_git_track_globs", None),
    )
    try:
        self.lhr.workspace_git_track_globs = list(track_globs)
    except Exception:
        logger.debug("[lnr] could not normalize workspace git globs", exc_info=True)
    result = ensure_workspace_source_git(
        self.workspace_dir,
        enabled=enabled,
        track_globs=track_globs,
        initial_commit=bool(getattr(self.lhr, "workspace_git_initial_commit", True)),
        user_name="ScienceFlow NLR",
        user_email="scienceflow-lnr@local",
        commit_timestamp=_deterministic_gate_timestamp(),
    )
    self._jsonl(
        "lhr_stage_events.jsonl",
        {
            "event": "workspace_git_init",
            "enabled": result.enabled,
            "ready": result.ready,
            "initialized": result.initialized,
            "committed": result.committed,
            "message": result.message,
            "track_globs": list(track_globs),
        },
    )
    if result.enabled and not result.ready:
        self.lhr.workspace_git_enabled = False
        if result.message:
            logger.warning("[lnr] workspace git unavailable: %s", result.message)


def _interaction_log_node_dir(self) -> Path:
    # Kept as the stable logger identity root; actual LNR files live in self.log_dir.
    return self.root_dir


def _split_logs_dir(self, name: str) -> Path:
    return self.log_dir / name


def _close_lnr_interaction_loggers(self) -> None:
    color = bool(getattr(self.cfg, "scienceflow_interaction_log_color", True))
    seen: set[Path] = set()
    close_targets = (
        (self.workspace_dir, None),
        (self._interaction_log_node_dir(), self.log_dir),
    )
    for node_dir, direct_log_dir in close_targets:
        try:
            resolved = Path(node_dir).resolve()
        except OSError:
            resolved = Path(node_dir)
        key = (
            resolved
            if direct_log_dir is None
            else Path(direct_log_dir).resolve(strict=False)
        )
        if key in seen:
            continue
        seen.add(key)
        close_workspace_interaction_logger(
            resolved,
            color=color,
            layout="split",
            log_dir_override=direct_log_dir,
        )


def _attach_lnr_interaction_logger(self, agent: Any) -> None:
    self._close_lnr_interaction_loggers()
    node_dir = self._interaction_log_node_dir()
    try:
        ensure_stage_log_dir(self.workspace_dir)
        agent._ws_interaction_log = attach_workspace_interaction_logger(
            node_dir,
            color=bool(getattr(self.cfg, "scienceflow_interaction_log_color", True)),
            layout="split",
            log_dir_override=self.log_dir,
        )
        attach_stage_interaction_handlers(
            agent._ws_interaction_log,
            self.workspace_dir,
            color=bool(getattr(self.cfg, "scienceflow_interaction_log_color", True)),
        )
    except Exception:
        logger.debug("[lnr] interaction logger reattach failed", exc_info=True)
    try:
        agent._tool_output_artifacts = agent._make_tool_output_artifact_store(
            node_dir,
            log_dir_override=self.log_dir,
        )
        stage_dir = stage_log_dir(self.workspace_dir)
        setter_stage_dir = getattr(
            agent._tool_output_artifacts, "set_stage_log_dir", None
        )
        if callable(setter_stage_dir):
            setter_stage_dir(stage_dir)
        setter = getattr(agent._tool_output_artifacts, "set_mirror_raw_id_prefix", None)
        if callable(setter):
            setter(self._next_stage_id_for_logging())
    except Exception:
        logger.debug("[lnr] tool output store reset failed", exc_info=True)


def _worker_uid_prefix(self) -> str:
    raw = str(self.worker_id or "").strip()
    if raw:
        return raw
    return f"W{int(self.worker_index or 0):02d}"


def _lineage_uid_prefix(self) -> str:
    raw = str(getattr(self, "current_lineage_id", "") or "").strip()
    if raw:
        return raw
    return f"L{int(getattr(self, 'current_lineage_no', 1) or 1):02d}"


def _stage_node_uid(self, stage_id: str, *, lineage_id: str | None = None) -> str:
    lineage = str(lineage_id or self._lineage_uid_prefix()).strip()
    return f"{self._worker_uid_prefix()}:{lineage}:{str(stage_id or '').upper()}"


def _start_new_lineage(self) -> str:
    self.current_lineage_no = int(getattr(self, "current_lineage_no", 1) or 1) + 1
    self.current_lineage_id = f"L{self.current_lineage_no:02d}"
    return self.current_lineage_id


@staticmethod
def _snapshot_node_uid(snap: StageSnapshot | None) -> str:
    if snap is None:
        return ""
    raw = str(getattr(snap, "node_uid", "") or "").strip()
    if raw:
        return raw
    raw_source = getattr(snap, "source_event", {})
    source = raw_source if isinstance(raw_source, dict) else {}
    return str(source.get("node_uid") or "")


@classmethod
def _archive_snapshot_key(cls, snap: StageSnapshot) -> str:
    return cls._snapshot_node_uid(snap) or f"legacy:{snap.snapshot_id}"


def _archived_stage_snapshot_index(self) -> dict[str, StageSnapshot]:
    archived = getattr(self, "archived_stage_snapshots", None)
    if not isinstance(archived, dict):
        archived = {}
        self.archived_stage_snapshots = archived
    for snap in getattr(self, "stage_snapshots", {}).values():
        archived[self._archive_snapshot_key(snap)] = snap
    return archived


def _index_archived_stage_snapshot(self, snap: StageSnapshot) -> None:
    self._archived_stage_snapshot_index()[self._archive_snapshot_key(snap)] = snap


def _stage_snapshot_for_restore(
    self, *, stage_id: str, node_uid: str = ""
) -> StageSnapshot | None:
    sid = str(stage_id or "").strip().upper()
    expected_node_uid = str(node_uid or "").strip()
    active = getattr(self, "stage_snapshots", {}).get(sid)
    if not expected_node_uid:
        return active
    if active is not None and self._snapshot_node_uid(active) == expected_node_uid:
        return active
    archived = self._archived_stage_snapshot_index()
    snap = archived.get(expected_node_uid)
    if snap is None or str(snap.stage_id or "").strip().upper() != sid:
        return None
    return snap


def _active_node_uid_for_stage(self, stage_id: str) -> str:
    return self._snapshot_node_uid(
        self.stage_snapshots.get(str(stage_id or "").upper())
    )


def _next_active_stage_id(self, cards: list[Any] | None = None) -> str:
    if cards is None:
        cards = parse_stage_cards(read_ledger(self.ledger_path))
    return next_stage_id(cards)


def _next_stage_id_for_logging(self) -> str:
    return self._next_active_stage_id()


def _latest_stage_id_for_logging(self) -> str:
    cards = parse_stage_cards(read_ledger(self.ledger_path))
    if cards:
        return str(cards[-1].stage_id)
    seen: list[int] = []
    for sid in self.stage_snapshots:
        raw = str(sid or "").upper()
        if raw.startswith("S") and raw[1:].isdigit():
            seen.append(int(raw[1:]))
    if not seen:
        return ""
    return f"S{max(seen):02d}"


def _record_context_compact_event(self, *, phase: str, **payload: Any) -> None:
    phase_name = str(phase or "event").strip().lower() or "event"
    event = f"context_compact_{phase_name}"
    latest_stage = self._latest_stage_id_for_logging()
    next_stage = self._next_stage_id_for_logging()
    record = {
        "event": event,
        "stage_id": latest_stage,
        "latest_stage": latest_stage,
        "next_stage": next_stage,
        **payload,
    }
    self._jsonl("lhr_context_events.jsonl", record)
    stage_label = latest_stage or next_stage or "-"
    mode = str(payload.get("mode") or "")
    round_no = payload.get("round")
    if phase_name == "started":
        logger.info(
            "[context-compact] started mode=%s reason=%s stage=%s round=%s omitted_before=%s recent_keep=%s",
            mode,
            payload.get("reason") or "",
            stage_label,
            round_no,
            payload.get("omitted_before"),
            payload.get("recent_keep"),
        )
    elif phase_name == "finished":
        tokens_in = int(payload.get("tokens_in") or 0)
        tokens_cached = int(payload.get("tokens_cached") or 0)
        cache_rate = tokens_cached / tokens_in if tokens_in > 0 else None
        cache_text = f"{cache_rate:.3f}" if cache_rate is not None else "-"
        logger.info(
            "[context-compact] finished mode=%s status=%s stage=%s round=%s omitted_after=%s summary_chars=%s tokens_in=%s tokens_out=%s cache_rate=%s",
            mode,
            payload.get("status") or "",
            stage_label,
            round_no,
            payload.get("omitted_after"),
            payload.get("summary_chars"),
            tokens_in,
            payload.get("tokens_out"),
            cache_text,
        )
    elif phase_name == "failed":
        logger.warning(
            "[context-compact] failed mode=%s stage=%s round=%s reason=%s fallback=%s",
            mode,
            stage_label,
            round_no,
            payload.get("reason_detail") or "",
            payload.get("fallback") or "",
        )
    elif phase_name == "estra_check":
        logger.info(
            "[context-compact] estra_check mode=%s stage=%s round=%s omitted_before=%s",
            mode,
            stage_label,
            round_no,
            payload.get("omitted_before"),
        )
    elif phase_name == "deferred":
        logger.info(
            "[context-compact] deferred mode=%s stage=%s round=%s reason=%s",
            mode,
            stage_label,
            round_no,
            payload.get("reason") or "",
        )


def _ensure_split_logs(self) -> None:
    for rel in (
        ("interaction",),
        ("interaction", "tool_outputs"),
        ("traj_interaction",),
        ("traj_interaction", "tool_outputs"),
    ):
        self.log_dir.joinpath(*rel).mkdir(parents=True, exist_ok=True)
    (self.workspace_dir / "tmp").mkdir(parents=True, exist_ok=True)
    for rel in (("agentic_route",), ("submission_snapshots",)):
        self.log_dir.joinpath(*rel).mkdir(parents=True, exist_ok=True)
    ensure_stage_log_dir(self.workspace_dir)


def _prune_workspace_control_artifacts(self) -> None:
    """Keep LHR control-plane files out of the agent-visible workspace log tree."""
    for control_dir in (self.workspace_dir / "logs",):
        try:
            if control_dir.is_dir() and not control_dir.is_symlink():
                shutil.rmtree(control_dir)
            elif control_dir.exists() or control_dir.is_symlink():
                control_dir.unlink()
        except OSError:
            logger.debug(
                "[lnr] workspace control dir cleanup skipped: %s",
                control_dir,
                exc_info=True,
            )
    log_dir = self.workspace_dir / ".logs"
    for name in (
        "agentic_route_decisions.jsonl",
        "agentic_route_response.md",
        "full_run_stamp.json",
        "fullrun_tail_snapshot.json",
        "lhr_stage_map.json",
        "lhr_snapshot_meta.json",
    ):
        try:
            (log_dir / name).unlink(missing_ok=True)
        except OSError:
            logger.debug(
                "[lnr] workspace control artifact cleanup skipped", exc_info=True
            )
    try:
        for path in log_dir.glob("lhr_*.jsonl"):
            path.unlink(missing_ok=True)
    except OSError:
        logger.debug("[lnr] workspace jsonl cleanup skipped", exc_info=True)
    try:
        legacy_tmp = log_dir / "tmp"
        if legacy_tmp.is_dir():
            shutil.rmtree(legacy_tmp)
        elif legacy_tmp.exists() or legacy_tmp.is_symlink():
            legacy_tmp.unlink()
    except OSError:
        logger.debug("[lnr] workspace legacy tmp cleanup skipped", exc_info=True)
    for name in (
        ".memory",
        ".snapshots",
        "logs",
        "snapshots",
        "stage_memory",
        "submission_history",
        "submission_snapshots",
        "submissions",
    ):
        path = self.workspace_dir / name
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            logger.debug(
                "[lnr] workspace artifact cleanup skipped: %s", path, exc_info=True
            )
    ensure_stage_log_dir(self.workspace_dir)


def _reset_interaction_stage_files(self) -> None:
    inter_dir = self._split_logs_dir("interaction")
    try:
        if inter_dir.exists():
            shutil.rmtree(inter_dir)
        (inter_dir / "tool_outputs").mkdir(parents=True, exist_ok=True)
        (inter_dir / "interaction.log").touch(exist_ok=True)
    except OSError:
        logger.debug("[lnr] interaction stage reset skipped", exc_info=True)
    try:
        reset_stage_log_dir(self.workspace_dir)
    except OSError:
        logger.debug("[lnr] stage-local interaction reset skipped", exc_info=True)
    self._ensure_split_logs()


def _reset_agent_interaction_stage(self, agent: Any) -> None:
    self._reset_interaction_stage_files()
    self._attach_lnr_interaction_logger(agent)


@staticmethod
def _agent_memory_messages(agent: Any) -> list[Any]:
    try:
        records = agent.memory.chat_history_memory.retrieve(window_size=None)
    except Exception:
        return []
    out: list[Any] = []
    for rec in records or []:
        msg = getattr(getattr(rec, "memory_record", None), "message", None)
        if msg is not None:
            out.append(msg)
    return out
