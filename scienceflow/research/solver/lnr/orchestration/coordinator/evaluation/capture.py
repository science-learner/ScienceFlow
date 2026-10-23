# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Evaluation capture owner for artifacts, materialization, and callback admission.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.memory_projection import (
    _metric_value_float,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
    _csv_bool,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    Path,
    ToolResult,
    _deterministic_gate_timestamp,
    append_stage_event_summary,
    archive_workspace_candidate_artifact,
    auto_checkpoint_workspace_source,
    format_invalid_evaluator_feedback,
    hashlib,
    logger,
    normalize_stage_id,
)


class EvaluationCaptureOwner:
    """Own the pre-commit half of stage evaluation."""

    def _candidate_artifact_sha_from_workspace(
        self, metric_event: dict[str, Any]
    ) -> str:
        candidates: list[str] = []
        explicit = str(metric_event.get("artifact_path") or "").strip()
        if explicit:
            candidates.append(explicit)
        configured = self._evaluator_candidate_artifact()
        if configured and configured not in candidates:
            candidates.append(configured)
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                path = self.workspace_dir / path
            if not path.is_file():
                continue
            try:
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()
            except OSError:
                logger.debug(
                    "[lnr] candidate artifact sha failed: %s", path, exc_info=True
                )
        return ""

    def _evaluator_feedback_for_agent(self, metric_event: dict[str, Any]) -> str:
        return format_invalid_evaluator_feedback(
            metric_event,
            candidate_artifact=self._evaluator_candidate_artifact(),
        )

    def _evaluator_query_budget_terminal(self, metric_event: dict[str, Any]) -> bool:
        evaluator = getattr(getattr(self, "cfg", None), "evaluator", None)
        if not bool(getattr(evaluator, "stop_on_query_budget_exhausted", False)):
            return False
        extra = metric_event.get("extra")
        sources = [metric_event]
        if isinstance(extra, dict):
            sources.append(extra)
        reason = str(metric_event.get("metric_validity_reason_code") or "").strip()
        for source in sources:
            terminal = source.get("query_budget_exhausted") is True
            try:
                remaining = int(source.get("queries_remaining"))
                limit = int(source.get("query_limit"))
            except (TypeError, ValueError):
                if terminal or reason == "scientific_design_query_budget_exhausted":
                    return True
                continue
            if limit > 0 and remaining <= 0:
                return bool(
                    terminal
                    or reason == "scientific_design_query_budget_exhausted"
                    or _csv_bool(
                        metric_event.get("selection_eligible"),
                        default=False,
                    )
                )
        return False

    def _request_evaluator_graceful_stop(self, *, agent: Any, stage_id: str) -> str:
        self.evaluator_stop_requested = True
        self.evaluator_stop_reason = "evaluator_query_budget_exhausted"
        setattr(agent, "_lnr_graceful_stop_requested", True)
        self._jsonl(
            "lhr_coordinator_events.jsonl",
            {
                "event": "evaluator_graceful_stop_requested",
                "stage_id": str(stage_id or ""),
                "stop_reason": self.evaluator_stop_reason,
            },
        )
        return (
            "EVALUATOR_QUERY_BUDGET_EXHAUSTED\n"
            "The official query budget is exhausted. The current evaluated stage "
            "and the best prior stage are preserved for final selection."
        )

    def _archive_candidate_artifact_after_tool(
        self,
        *,
        agent: Any,
        tool_name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
    ) -> None:
        _ = args
        artifact_path = self._evaluator_candidate_artifact()
        is_submission = artifact_path.replace("\\", "/").strip() == "submission.csv"
        snapshot_dir = self.log_dir / (
            "submission_snapshots" if is_submission else "artifact_snapshots"
        )
        result = archive_workspace_candidate_artifact(
            self.workspace_dir,
            artifact_path=artifact_path,
            artifact_kind=self._evaluator_candidate_artifact_kind(),
            snapshot_dir=snapshot_dir,
            ledger_path=self.log_dir / "checkpoints" / "artifact_archive.jsonl",
            trigger=tool_name,
            tool_error=bool(tool_result.error),
        )
        current_sha = str(result.artifact_sha256 or "") if result.ready else ""
        previous_sha = str(
            getattr(agent, "_lnr_candidate_artifact_known_sha", "") or ""
        )
        changed = bool(current_sha and current_sha != previous_sha)
        setattr(agent, "_lnr_candidate_artifact_changed_after_tool", changed)
        setattr(
            agent,
            "_lnr_candidate_artifact_sha_after_tool",
            current_sha,
        )
        setattr(agent, "_lnr_candidate_artifact_known_sha", current_sha)
        if changed:
            setattr(agent, "_lnr_candidate_artifact_pending_sha", current_sha)
        elif not current_sha:
            setattr(agent, "_lnr_candidate_artifact_pending_sha", "")
        if not result.archived and result.ready:
            return
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "candidate_artifact_archived"
                if result.archived
                else "candidate_artifact_archive_error",
                "capture_type": "candidate_artifact_persisted",
                "trigger": tool_name,
                "tool_error": bool(tool_result.error),
                "artifact_path": result.artifact_path,
                "artifact_kind": result.artifact_kind,
                "artifact_sha": result.artifact_sha256,
                "artifact_snapshot": result.snapshot_path,
                "archive_ledger_path": result.ledger_path,
                "size_bytes": result.size_bytes,
                "selection_eligible": None,
                "metric_value": None,
                "message": result.message,
            },
        )

    def _materialized_existing_stage_target(
        self,
        *,
        cards_before: list[Any],
        metric_event: dict[str, Any],
        semantic_source_changed: bool,
    ) -> str:
        if semantic_source_changed:
            return ""
        if not _csv_bool(metric_event.get("candidate_ready"), default=False):
            return ""
        if not str(metric_event.get("submission_sha") or "").strip():
            return ""
        if not cards_before:
            return ""
        latest = cards_before[-1]
        target_stage = normalize_stage_id(getattr(latest, "stage_id", ""))
        if not target_stage:
            return ""
        snap = self.stage_snapshots.get(target_stage)
        source_event = (
            snap.source_event
            if snap is not None and isinstance(snap.source_event, dict)
            else {}
        )
        if _csv_bool(source_event.get("candidate_ready"), default=False):
            return ""
        previous_metric = _metric_value_float(getattr(latest, "metric", ""))
        current_metric = _metric_value_float(metric_event.get("metric_value"))
        if previous_metric is None or current_metric is None:
            return ""
        if abs(previous_metric - current_metric) > max(
            1e-9, abs(previous_metric) * 1e-6
        ):
            return ""
        previous_solution = str(source_event.get("solution_sha") or "").strip()
        current_solution = str(metric_event.get("solution_sha") or "").strip()
        if (
            previous_solution
            and current_solution
            and previous_solution != current_solution
        ):
            return ""
        return target_stage

    def _record_stage_materialization_event(
        self,
        *,
        target_stage: str,
        metric_event: dict[str, Any],
        now: float,
        solution_sha: str,
        run_signature: str,
    ) -> None:
        self.last_stage_commit_ts = now
        self.last_captured_solution_sha = solution_sha
        self.last_captured_run_signature = run_signature
        stage_id = normalize_stage_id(target_stage)
        summary = (
            "Ready submission generated from the same route; no new research stage."
        )
        append_stage_event_summary(
            self.ledger_path, target_stage=stage_id, summary=summary
        )
        metric_value_raw = metric_event.get("metric_value")
        metric_value_override = (
            float(metric_value_raw)
            if isinstance(metric_value_raw, (int, float))
            else None
        )
        checkpoint = auto_checkpoint_workspace_source(
            self.workspace_dir,
            enabled=bool(
                getattr(self.lhr, "workspace_git_enabled", True)
                and getattr(self.lhr, "workspace_git_auto_checkpoint", True)
            ),
            track_globs=getattr(self.lhr, "workspace_git_track_globs", None),
            tool_name=f"lhr_stage_event_{stage_id.lower()}",
            metric_value_override=metric_value_override,
            stage_id=f"{stage_id}-event",
            submission_snapshot_dir=self.log_dir / "submission_snapshots",
            checkpoint_dir=self.log_dir / "checkpoints",
            commit_timestamp=_deterministic_gate_timestamp(),
        )
        snap = self.stage_snapshots.get(stage_id)
        if snap is not None and isinstance(snap.source_event, dict):
            snap.source_event.update(
                {
                    "candidate_ready": True,
                    "submission_status": "ready",
                    "materialized_ready_submission": True,
                    "submission_sha": metric_event.get("submission_sha"),
                    "submission_snapshot": checkpoint.submission_snapshot,
                    "capture_type": "stage_event_materialization",
                }
            )
            self._write_stage_map()
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "stage_materialized_ready_submission",
                "target_stage": stage_id,
                "metric_value": metric_event.get("metric_value"),
                "metric_name": metric_event.get("metric_name"),
                "solution_sha": solution_sha,
                "submission_sha": metric_event.get("submission_sha"),
                "semantic_source_changed": metric_event.get("semantic_source_changed"),
                "candidate_ready": True,
                "stage_policy": "materialized_existing_stage_no_new_research_stage",
                "workspace_git_enabled": checkpoint.enabled,
                "workspace_git_ready": checkpoint.ready,
                "workspace_git_stage_id": checkpoint.stage_id,
                "source_commit_sha": checkpoint.commit_sha,
                "source_changed": checkpoint.source_changed,
                "submission_snapshot": checkpoint.submission_snapshot,
                "submission_changed": checkpoint.submission_changed,
                "workspace_git_message": checkpoint.message,
                "ledger_summary": summary,
            },
        )

    async def _stage_capture_callback(
        self,
        *,
        agent: Any,
        args: dict[str, Any],
        tool_result: ToolResult,
    ) -> str | None:
        """Legacy callback adapter; StageLifecycleCoordinator owns dispatch."""

        coordinator = self._stage_lifecycle()
        trigger = coordinator.new_legacy_trigger(
            worker_id=self.worker_id or "W00",
            tool_name=str(args.get("tool_name") or args.get("name") or "tool"),
            tool_status="error" if bool(getattr(tool_result, "error", None)) else "ok",
        )
        return await coordinator.dispatch_legacy(
            trigger,
            lambda: self._stage_capture_callback_impl(
                agent=agent,
                args=args,
                tool_result=tool_result,
            ),
        )

    def _apply_submission_capture_policy(
        self,
        *,
        metric_event,
        submission_invalid,
        now,
        solution_sha,
        run_signature,
        submission_sha,
        duplicate,
        artifact_sha,
        semantic_source_changed,
        generic_evaluator_capture,
    ) -> bool:
        if submission_invalid:
            metric_event.update(
                {
                    "candidate_ready": False,
                    "submission_status": "invalid_submission",
                    "capture_type": "invalid_submission_skipped",
                }
            )
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_invalid_submission_skipped",
                    "submission_sha": submission_sha,
                    "solution_sha": solution_sha,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "submission_status": "invalid_submission",
                    "candidate_ready": False,
                    "stage_policy": "skip_invalid_submission_not_a_stage",
                },
            )
            return False
        elif duplicate is not None:
            same_design_duplicate = self._duplicate_stage_same_design(
                metric_event, duplicate
            )
            force_capture_duplicate = bool(
                getattr(self.lhr, "force_estra_capture_duplicate_submissions", False)
                and int(getattr(self.lhr, "force_estra_after_stage_count", 0) or 0) > 0
            )
            metric_event.update(
                {
                    "candidate_ready": False,
                    "submission_status": "duplicate_or_stale_submission",
                    "duplicate_submission_of_stage": duplicate.stage_id,
                    "duplicate_submission_of_snapshot_id": duplicate.snapshot_id,
                    "capture_type": "duplicate_capture",
                }
            )
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            if same_design_duplicate:
                stage_policy = "duplicate_candidate_same_design"
            elif bool(semantic_source_changed):
                stage_policy = "duplicate_candidate_same_artifact_different_source"
            else:
                stage_policy = "duplicate_candidate_no_semantic_workspace_change"
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_duplicate_submission_skipped",
                    "duplicate_of_stage": duplicate.stage_id,
                    "duplicate_of_snapshot_id": duplicate.snapshot_id,
                    "artifact_sha": artifact_sha,
                    "force_capture_requested": bool(force_capture_duplicate),
                    "submission_sha": submission_sha,
                    "solution_sha": solution_sha,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "semantic_source_changed": bool(semantic_source_changed),
                    "same_design_duplicate": bool(same_design_duplicate),
                    "stage_policy": stage_policy,
                },
            )
            return False
        elif generic_evaluator_capture:
            metric_event.setdefault("candidate_ready", True)
            metric_event.setdefault(
                "submission_status", metric_event.get("evaluator_status") or "ready"
            )
        elif not submission_sha and (not artifact_sha):
            metric_event.update(
                {"candidate_ready": False, "submission_status": "missing_submission"}
            )
        else:
            metric_event.update({"candidate_ready": True, "submission_status": "ready"})
        return True

    def _stage_capture_cap_allows(
        self,
        *,
        stage_cap,
        cards_before,
        stop_after_evaluator_query,
        now,
        solution_sha,
        run_signature,
        metric_event,
    ) -> bool:
        if (
            stage_cap > 0
            and len(cards_before) >= stage_cap
            and (not stop_after_evaluator_query)
        ):
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_capture_cap_reached",
                    "observed_stage_count": len(cards_before),
                    "stage_capture_max_count": stage_cap,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "candidate_ready": metric_event.get("candidate_ready"),
                    "worker_id": self._worker_uid_prefix(),
                },
            )
            return False
        return True
