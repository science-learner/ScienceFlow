# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: stage text protocol, transactions, and capture finalization.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    MemoryService,
    Message,
    Path,
    StageLifecycleCoordinator,
    StageLifecycleEvent,
    StageSnapshot,
    StageTransactionService,
    _deterministic_gate_timestamp,
    append_stage_event,
    atomic_write,
    auto_checkpoint_workspace_source,
    json,
    logger,
    normalize_stage_id,
    parse_stage_cards,
    read_ledger,
    stage_log_dir,
    time,
    validate_append_only_stage_commit,
)


def _append_stage_commit_from_judgment(
    self,
    *,
    agent: Any,
    stage_id: str,
    metric_event: dict[str, Any],
    judgment: dict[str, Any],
    block_text: str,
    source: str,
) -> tuple[bool, str]:
    ledger_before = read_ledger(self.ledger_path)
    entry = self._stage_commit_entry_from_metric_event(
        stage_id=stage_id,
        metric_event=metric_event,
        judgment=judgment,
        workspace_dir=self.workspace_dir,
    )
    sep = ""
    if ledger_before and not ledger_before.endswith("\n\n"):
        sep = "\n" if ledger_before.endswith("\n") else "\n\n"
    ledger_after = ledger_before + sep + entry
    ok, reason = validate_append_only_stage_commit(
        ledger_before, ledger_after, stage_id
    )
    if not ok:
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_append_rejected",
                "stage_id": stage_id,
                "reason": reason,
                "source": source,
            },
        )
        return False, reason
    try:
        self._prepare_stage_commit_transaction(
            stage_id=stage_id,
            ledger_before=ledger_before,
            ledger_after=ledger_after,
            metric_event=metric_event,
        )
    except OSError as exc:
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_transaction_prepare_error",
                "stage_id": stage_id,
                "error": str(exc),
                "source": source,
            },
        )
        return False, f"tool_error:{exc}"
    try:
        atomic_write(self.ledger_path, ledger_after)
    except OSError as exc:
        self._rollback_stage_commit_transaction(
            stage_id=stage_id, reason=f"ledger_write:{exc}"
        )
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_append_error",
                "stage_id": stage_id,
                "error": str(exc),
                "source": source,
            },
        )
        return False, f"tool_error:{exc}"
    self._jsonl(
        "lhr_stage_commit_events.jsonl",
        {
            "event": "stage_commit_appended_text_block",
            "stage_id": stage_id,
            "judgment_source": source,
        },
    )
    self._jsonl(
        "lhr_stage_commit_events.jsonl",
        {"event": "stage_commit_ok", "stage_id": stage_id, "turn": 1},
    )
    persist_agent_write = bool(
        getattr(self.lhr, "stage_commit_persist_agent_write_to_memory", True)
    ) or bool(getattr(self.lhr, "stage_commit_persist_to_memory", False))
    self._capture_s01_eda_prefix_end(stage_id=stage_id)
    if persist_agent_write:
        mem_text = self._stage_commit_text_memory_message(
            block_text=block_text,
            stage_id=stage_id,
            metric_event=metric_event,
            entry=entry,
        )
        try:
            mem_text = agent._sanitize_agent_visible_paths(mem_text)
        except Exception:
            pass
        memory = getattr(agent, "memory", None)
        if memory is not None and hasattr(memory, "add_message"):
            memory.add_message(Message.assistant_message(mem_text))
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_persisted_to_memory",
                    "stage_id": stage_id,
                    "turn": 1,
                    "agent_write_persisted": True,
                    "prompt_persisted": False,
                    "text_block_append": True,
                },
            )
    return True, ""


def _stage_transaction_root(self) -> Path:
    return self.log_dir / "stage_transactions"


def _stage_lifecycle(self) -> StageLifecycleCoordinator:
    coordinator = getattr(self, "stage_lifecycle_coordinator", None)
    if not isinstance(coordinator, StageLifecycleCoordinator):
        coordinator = StageLifecycleCoordinator()
        self.stage_lifecycle_coordinator = coordinator
    return coordinator


def _stage_transaction_service(self) -> StageTransactionService:
    workspace = getattr(self, "workspace_service", None)
    factory = getattr(workspace, "stage_transactions", None)
    if callable(factory):
        return factory(transaction_root=self._stage_transaction_root())
    # Focused legacy test doubles may not construct the composition graph.
    return StageTransactionService(
        ledger_path=self.ledger_path,
        transaction_root=self._stage_transaction_root(),
    )


def _prepare_stage_commit_transaction(
    self,
    *,
    stage_id: str,
    ledger_before: str,
    ledger_after: str,
    metric_event: dict[str, Any],
) -> None:
    transaction_id = f"{self._lineage_uid_prefix()}-{stage_id}".replace(":", "-")
    self.pending_stage_commit_transaction = self._stage_transaction_service().prepare(
        transaction_id=transaction_id,
        stage_id=stage_id,
        lineage_id=self._lineage_uid_prefix(),
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        metric_event=metric_event,
    )
    self._stage_lifecycle().advance_legacy(
        stage_id,
        StageLifecycleEvent.PREPARE,
        event_id=transaction_id,
    )


def _complete_stage_commit_transaction(
    self, *, stage_id: str, snapshot: StageSnapshot
) -> None:
    pending, completed = self._stage_transaction_service().complete(
        getattr(self, "pending_stage_commit_transaction", None),
        stage_id=stage_id,
        snapshot=snapshot,
    )
    self.pending_stage_commit_transaction = pending
    if completed:
        transaction_id = str(
            (getattr(snapshot, "source_event", {}) or {}).get("stage_transaction_id")
            or f"{self._lineage_uid_prefix()}-{stage_id}"
        )
        self._stage_lifecycle().advance_legacy(
            stage_id,
            StageLifecycleEvent.COMMIT_STAGE,
            event_id=transaction_id,
        )


def _rollback_stage_commit_transaction(self, *, stage_id: str, reason: str) -> bool:
    pending, rolled_back = self._stage_transaction_service().rollback(
        getattr(self, "pending_stage_commit_transaction", None),
        stage_id=stage_id,
        reason=reason,
    )
    self.pending_stage_commit_transaction = pending
    if rolled_back:
        self._stage_lifecycle().advance_legacy(
            stage_id,
            StageLifecycleEvent.ROLLBACK,
            event_id=f"rollback:{stage_id}:{reason}",
        )
    return rolled_back


def _recover_stage_commit_transactions(self) -> None:
    discover_all = getattr(self.snapshot_store, "discover_all", None)
    snapshots = (
        list(discover_all().values())
        if callable(discover_all)
        else list(self.snapshot_store.discover().values())
    )
    self._stage_transaction_service().recover(
        snapshots=snapshots,
        event_sink=lambda record: self._jsonl("lhr_resume_events.jsonl", record),
    )
    for manifest in sorted(self._stage_transaction_root().glob("*.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        stage_id = str(payload.get("stage_id") or "").strip().upper()
        transaction_id = str(payload.get("transaction_id") or manifest.stem)
        status = str(payload.get("status") or "")
        if stage_id and status == "committed":
            self._stage_lifecycle().advance_legacy(
                stage_id,
                StageLifecycleEvent.COMMIT_STAGE,
                event_id=f"recover:{transaction_id}",
            )
        elif stage_id and status == "rolled_back":
            self._stage_lifecycle().advance_legacy(
                stage_id,
                StageLifecycleEvent.ROLLBACK,
                event_id=f"recover:{transaction_id}",
            )


async def _finalize_stage_capture_after_commit(
    self,
    *,
    agent: Any,
    stage_id: str,
    metric_event: dict[str, Any],
    now: float,
    solution_sha: str,
    run_signature: str,
    allow_followup: bool = True,
) -> str | None:
    try:
        return await self._finalize_stage_capture_after_commit_impl(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            now=now,
            solution_sha=solution_sha,
            run_signature=run_signature,
            allow_followup=allow_followup,
        )
    except BaseException as exc:
        rolled_back = self._rollback_stage_commit_transaction(
            stage_id=stage_id,
            reason=f"finalize:{type(exc).__name__}:{exc}",
        )
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "stage_finalize_failed",
                "stage_id": stage_id,
                "ledger_rolled_back": rolled_back,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def _checkpoint_stage_workspace(
    self,
    *,
    stage_id: str,
    metric_event: dict[str, Any],
    now: float,
    solution_sha: str,
    run_signature: str,
) -> None:
    self.last_stage_commit_ts = now
    self.last_captured_solution_sha = solution_sha
    self.last_captured_run_signature = run_signature
    metric_value_raw = metric_event.get("metric_value")
    metric_value_override = (
        float(metric_value_raw) if isinstance(metric_value_raw, (int, float)) else None
    )
    checkpoint = auto_checkpoint_workspace_source(
        self.workspace_dir,
        enabled=bool(
            getattr(self.lhr, "workspace_git_enabled", True)
            and getattr(self.lhr, "workspace_git_auto_checkpoint", True)
        ),
        track_globs=getattr(self.lhr, "workspace_git_track_globs", None),
        tool_name=f"lhr_stage_{stage_id.lower()}",
        metric_value_override=metric_value_override,
        stage_id=stage_id,
        submission_snapshot_dir=self.log_dir / "submission_snapshots",
        checkpoint_dir=self.log_dir / "checkpoints",
        commit_timestamp=_deterministic_gate_timestamp(),
    )
    metric_event.update(
        {
            "source_commit_sha": checkpoint.commit_sha,
            "source_changed": checkpoint.source_changed,
            "workspace_git_stage_id": checkpoint.stage_id,
            "submission_snapshot": checkpoint.submission_snapshot,
            "submission_changed": checkpoint.submission_changed,
            "workspace_git_enabled": checkpoint.enabled,
            "workspace_git_ready": checkpoint.ready,
            "workspace_git_message": checkpoint.message,
            "workspace_git_ledger_path": checkpoint.ledger_path,
        }
    )
    self._jsonl(
        "lhr_stage_events.jsonl",
        {
            "event": "stage_workspace_git_checkpoint",
            "stage_id": stage_id,
            "enabled": checkpoint.enabled,
            "ready": checkpoint.ready,
            "committed": checkpoint.committed,
            "commit_sha": checkpoint.commit_sha,
            "workspace_git_stage_id": checkpoint.stage_id,
            "source_changed": checkpoint.source_changed,
            "submission_snapshot": checkpoint.submission_snapshot,
            "submission_changed": checkpoint.submission_changed,
            "ledger_path": checkpoint.ledger_path,
            "message": checkpoint.message,
        },
    )


async def _finalize_stage_capture_after_commit_impl(
    self,
    *,
    agent: Any,
    stage_id: str,
    metric_event: dict[str, Any],
    now: float,
    solution_sha: str,
    run_signature: str,
    allow_followup: bool = True,
) -> str | None:
    _checkpoint_stage_workspace(
        self,
        stage_id=stage_id,
        metric_event=metric_event,
        now=now,
        solution_sha=solution_sha,
        run_signature=run_signature,
    )
    self._prune_workspace_control_artifacts()
    await self._prepare_protected_eda_agent_summary(agent, stage_id=stage_id)
    self._mark_protected_eda_prefix(agent, stage_id=stage_id)
    cards_after = parse_stage_cards(read_ledger(self.ledger_path))
    await self._adjudicate_metric_validity_for_stage(
        agent=agent,
        stage_id=stage_id,
        metric_event=metric_event,
        cards_after=cards_after,
    )
    lineage_id = self._lineage_uid_prefix()
    node_uid = self._stage_node_uid(stage_id, lineage_id=lineage_id)
    metric_event_for_snapshot = {
        **dict(metric_event),
        "node_uid": node_uid,
        "lineage_id": lineage_id,
        "visible_stage_id": stage_id,
        "worker_id": self._worker_uid_prefix(),
    }
    if stage_id == "S01" and self._s01_eda_prefix_end_index is not None:
        metric_event_for_snapshot["protected_eda_end_index"] = int(
            self._s01_eda_prefix_end_index
        )
    append_stage_event(
        stage_log_dir(self.workspace_dir),
        "stage_capture_begin",
        stage_id=stage_id,
        node_uid=node_uid,
        metric_value=metric_event.get("metric_value"),
        metric_name=str(metric_event.get("metric_name") or ""),
    )
    snap = self.snapshot_store.capture(
        stage_id=stage_id,
        metric_value=metric_event.get("metric_value"),
        metric_name=str(metric_event.get("metric_name") or ""),
        lower_is_better=metric_event.get("lower_is_better")
        if isinstance(metric_event.get("lower_is_better"), bool)
        else None,
        memory_cut=self._count_memory_records(),
        source_event=metric_event_for_snapshot,
        node_uid=node_uid,
        lineage_id=lineage_id,
    )
    self.stage_snapshots[stage_id] = snap
    self._index_archived_stage_snapshot(snap)
    self._write_stage_map()
    self._jsonl(
        "lhr_stage_events.jsonl",
        {
            "event": "stage_captured",
            "stage_id": stage_id,
            "node_uid": node_uid,
            "lineage_id": lineage_id,
            "snapshot_id": snap.snapshot_id,
            "snapshot_path": str(snap.snapshot_path),
            "snapshot_mode": getattr(snap, "snapshot_mode", "full"),
            "snapshot_warnings": list(getattr(snap, "snapshot_warnings", ()) or ()),
            "metric_event": metric_event_for_snapshot,
            "preserved_logs": (snap.snapshot_path / "logs").is_dir(),
            "preserved_stage_logs": (snap.snapshot_path / ".logs").is_dir(),
            "preserved_memory": (snap.snapshot_path / "logs" / "memory").is_dir(),
        },
    )
    self._append_stage_performance_row(
        stage_id=stage_id,
        snap=snap,
        metric_event=metric_event_for_snapshot,
        cards_after=cards_after,
    )
    self._complete_stage_commit_transaction(stage_id=stage_id, snapshot=snap)
    if bool(getattr(self.lhr, "stage_memory_folding_enabled", True)):
        try:
            memory_service = getattr(self, "memory_service", None)
            if memory_service is None:
                memory_service = MemoryService(workspace_dir=self.workspace_dir)
                self.memory_service = memory_service
            memory_service.sync_current_segment(
                self._effective_stage_cards(cards_after)
            )
        except OSError:
            logger.debug(
                "[lnr] stage memory current segment sync failed", exc_info=True
            )
    self._stage_lifecycle().advance_legacy(
        stage_id,
        StageLifecycleEvent.PROJECT_MEMORY,
        event_id=f"stage:{stage_id}",
    )
    forced_estra_out = None
    hygiene_estra_out = None
    if allow_followup:
        forced_estra_out = await self._force_estra_after_stage_capture(
            agent=agent,
            cards_after=cards_after,
        )
        if not forced_estra_out:
            hygiene_estra_out = await self._context_hygiene_compact_after_stage_capture(
                agent=agent,
                cards_after=cards_after,
            )
    self._stage_lifecycle().advance_legacy(
        stage_id,
        StageLifecycleEvent.OBSERVE_ESTRA,
        event_id=f"stage:{stage_id}",
    )
    self._reset_agent_interaction_stage(agent)
    self._stage_lifecycle().advance_legacy(
        stage_id,
        StageLifecycleEvent.COMPLETE,
        event_id=f"stage:{stage_id}",
    )
    if forced_estra_out:
        return forced_estra_out
    if hygiene_estra_out:
        return hygiene_estra_out
    return None


def _resolve_stage_commit_text_judgment(
    self,
    *,
    agent: Any,
    pending: dict[str, Any],
    stage_id: str,
    metric_event: dict[str, Any],
    parsed: dict[str, Any],
    block_text: str,
    reason: str,
) -> tuple[dict[str, Any] | None, str, str, bool]:
    if reason:
        attempts = int(pending.get("attempts") or 0) + 1
        pending["attempts"] = attempts
        self.pending_text_stage_commit = pending
        setattr(
            agent,
            "_lnr_suppress_current_text_only_memory",
            f"stage_commit_parse_failed:{reason}",
        )
        if attempts <= 1:
            self._set_stage_commit_transient_prompt(
                agent, stage_id=stage_id, metric_event=metric_event, correction=reason
            )
            setattr(agent, "_lnr_stage_commit_text_handled", True)
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_text_parse_retry",
                    "stage_id": stage_id,
                    "reason": reason,
                    "attempts": attempts,
                },
            )
            return None, block_text, "", True
        judgment = self._stage_commit_fallback_judgment(metric_event)
        fallback_data = {
            "stage_id": stage_id,
            "metric": metric_event.get("metric_value", "unknown"),
            "metric_validity": metric_event.get("metric_validity") or "medium",
            "brief": "fallback stage commit after malformed output.",
            "why": (
                "parser issue "
                f"{self._stage_commit_one_line(reason, max_chars=120)}; "
                "metric event preserved; continue from preserved evidence."
            ),
            "files": judgment["files"],
        }
        if (
            str(getattr(self.lhr, "stage_commit_output_format", "text") or "text")
            .strip()
            .lower()
            == "json"
        ):
            block_text = (
                "```json\n"
                + json.dumps(fallback_data, ensure_ascii=False, indent=2)
                + "\n```"
            )
        else:
            block_text = (
                "STAGE_COMMIT_BEGIN\n"
                + "\n".join(f"{key}: {value}" for key, value in fallback_data.items())
                + "\nSTAGE_COMMIT_END"
            )
        source = "fallback_after_text_parse_failed"
    else:
        judgment = self._stage_commit_judgment_from_text_block(parsed)
        if parsed.get("metric_validity"):
            metric_event["metric_validity"] = (
                str(parsed.get("metric_validity") or "").strip().lower()
            )
        source = "main_agent_text_block"
    return judgment, block_text, source, False


async def _handle_pending_stage_commit_text(
    self,
    *,
    agent: Any,
    assistant_text: str,
) -> str | None:
    pending = getattr(self, "pending_text_stage_commit", None)
    if not isinstance(pending, dict):
        return None
    stage_id = str(pending.get("stage_id") or "").strip().upper()
    metric_event = (
        pending.get("metric_event")
        if isinstance(pending.get("metric_event"), dict)
        else {}
    )
    parsed, block_text, reason = self._parse_stage_commit_block(assistant_text)
    parsed_stage = normalize_stage_id(parsed.get("stage_id")) if parsed else ""
    if not reason and parsed_stage and parsed_stage != stage_id:
        reason = f"stage_id_mismatch:{parsed_stage}!={stage_id}"
    judgment, block_text, source, retry_pending = _resolve_stage_commit_text_judgment(
        self,
        agent=agent,
        pending=pending,
        stage_id=stage_id,
        metric_event=metric_event,
        parsed=parsed,
        block_text=block_text,
        reason=reason,
    )
    if retry_pending or judgment is None:
        return None
    await self._audit_stage_result_before_commit(
        agent=agent,
        stage_id=stage_id,
        metric_event=metric_event,
        judgment=judgment,
        block_text=block_text,
        source=source,
    )
    ok, append_reason = self._append_stage_commit_from_judgment(
        agent=agent,
        stage_id=stage_id,
        metric_event=metric_event,
        judgment=judgment,
        block_text=block_text,
        source=source,
    )
    if not ok:
        if "FILES" in str(append_reason or ""):
            attempts = int(pending.get("attempts") or 0) + 1
            pending["attempts"] = attempts
            self.pending_text_stage_commit = pending
            setattr(
                agent,
                "_lnr_suppress_current_text_only_memory",
                f"stage_commit_append_failed:{append_reason}",
            )
            if attempts <= 2:
                self._set_stage_commit_transient_prompt(
                    agent,
                    stage_id=stage_id,
                    metric_event=metric_event,
                    correction=append_reason,
                )
                setattr(agent, "_lnr_stage_commit_text_handled", True)
                self._jsonl(
                    "lhr_stage_commit_events.jsonl",
                    {
                        "event": "stage_commit_text_parse_retry",
                        "stage_id": stage_id,
                        "reason": append_reason,
                        "attempts": attempts,
                    },
                )
                return None
            fallback_judgment = self._stage_commit_fallback_judgment(metric_event)
            fallback_block = (
                "STAGE_COMMIT_BEGIN\n"
                f"stage_id: {stage_id}\n"
                f"metric: {metric_event.get('metric_value', 'unknown')}\n"
                f"metric_validity: {metric_event.get('metric_validity') or 'medium'}\n"
                "brief: deterministic fallback after invalid FILES metadata.\n"
                f"why: Gate evidence was accepted; bookkeeping fallback preserves it after {self._stage_commit_one_line(append_reason, max_chars=120)}.\n"
                f"files: {fallback_judgment['files']}\n"
                "STAGE_COMMIT_END"
            )
            fallback_source = "fallback_after_files_retry_exhausted"
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_text_fallback_applied",
                    "stage_id": stage_id,
                    "reason": append_reason,
                    "attempts": attempts,
                    "source": fallback_source,
                },
            )
            ok, append_reason = self._append_stage_commit_from_judgment(
                agent=agent,
                stage_id=stage_id,
                metric_event=metric_event,
                judgment=fallback_judgment,
                block_text=fallback_block,
                source=fallback_source,
            )
            if ok:
                block_text = fallback_block
        if not ok:
            self.pending_text_stage_commit = None
            self._clear_stage_commit_transient_prompt(agent)
            setattr(agent, "_lnr_stage_commit_text_handled", True)
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_capture_failed",
                    "stage_id": stage_id,
                    "reason": append_reason,
                    "metric_event": metric_event,
                },
            )
            return None
    stop_after_commit = bool(pending.get("stop_after_evaluator_query_budget"))
    self.pending_text_stage_commit = None
    self._clear_stage_commit_transient_prompt(agent)
    setattr(agent, "_lnr_stage_commit_text_handled", True)
    finalize_out = await self._finalize_stage_capture_after_commit(
        agent=agent,
        stage_id=stage_id,
        metric_event=metric_event,
        now=float(pending.get("now") or time.monotonic()),
        solution_sha=str(pending.get("solution_sha") or ""),
        run_signature=str(pending.get("run_signature") or ""),
        allow_followup=not stop_after_commit,
    )
    if stop_after_commit:
        return self._request_evaluator_graceful_stop(
            agent=agent,
            stage_id=stage_id,
        )
    return finalize_out


def _evaluator_stage_source_mode(self) -> str:
    cfg = getattr(self, "cfg", None)
    evaluator = getattr(cfg, "evaluator", None)
    mode = (
        str(getattr(evaluator, "stage_source_mode", "primary") or "primary")
        .strip()
        .lower()
    )
    return mode if mode in {"shadow", "adjudicate", "primary"} else "primary"
