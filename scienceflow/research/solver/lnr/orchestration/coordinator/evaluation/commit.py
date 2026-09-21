# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Evaluation commit owner for gates, metric fallback, and stage persistence.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
    _csv_bool,
    _stage_run_signature,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    AgentFactory,
    Any,
    StageLifecycleEvent,
    ToolResult,
    copy,
    parse_stage_cards,
    read_ledger,
    time,
    workspace_source_changed,
)


class EvaluationCommitOwner:
    """Own the post-capture evaluation and commit sequence."""

    def _evaluate_missing_metric_stage(
        self, *, agent: Any, metric_event: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None, bool]:
        if self._evaluator_stage_source_mode() != "primary":
            return (metric_event, None, True)
        cards_before_for_eval = parse_stage_cards(read_ledger(self.ledger_path))
        stage_id_for_eval = self._next_active_stage_id(cards_before_for_eval)
        try:
            observed_artifact_sha = self._candidate_artifact_sha_from_workspace({})
        except Exception:
            observed_artifact_sha = ""
        prior_snapshot = self._stage_with_artifact_sha(
            self.stage_snapshots, observed_artifact_sha
        )
        if prior_snapshot is not None:
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "duplicate_candidate_pre_gate_skipped",
                    "stage_id": stage_id_for_eval,
                    "artifact_sha": observed_artifact_sha,
                    "duplicate_of_stage": prior_snapshot.stage_id,
                    "duplicate_of_snapshot_id": prior_snapshot.snapshot_id,
                    "stage_policy": "deduplicate_before_gate",
                },
            )
            return (metric_event, None, True)
        last_rejected_sha = str(
            getattr(self, "_last_rejected_gate_artifact_sha", "")
            or getattr(self, "_last_metric_missing_gate_artifact_sha", "")
            or ""
        )
        if observed_artifact_sha and observed_artifact_sha == last_rejected_sha:
            prior_reason = str(
                getattr(self, "_last_rejected_gate_reason_code", "") or ""
            )
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": (
                        "duplicate_pending_metric_pre_gate_skipped"
                        if prior_reason in {"", "metric_missing"}
                        else "duplicate_rejected_candidate_pre_gate_skipped"
                    ),
                    "stage_id": stage_id_for_eval,
                    "artifact_sha": observed_artifact_sha,
                    "prior_gate_reason_code": prior_reason,
                    "stage_policy": "deduplicate_rejected_artifact_before_gate",
                },
            )
            return (metric_event, None, True)
        metric_event = self._record_evaluator_stage_events(
            stage_id=stage_id_for_eval, metric_event={}
        )
        self._stage_lifecycle().advance_legacy(
            stage_id_for_eval,
            StageLifecycleEvent.ASSESS,
            event_id=f"assessment:{stage_id_for_eval}",
        )
        stop_after_evaluator_query = self._evaluator_query_budget_terminal(metric_event)
        if (
            metric_event.get("_gate_evaluated")
            and metric_event.get("gate_accepted") is not True
        ):
            self._stage_lifecycle().advance_legacy(
                stage_id_for_eval,
                StageLifecycleEvent.REJECT_GATE,
                event_id=f"gate:{stage_id_for_eval}:{metric_event.get('gate_reason_code') or 'rejected'}",
            )
            if observed_artifact_sha:
                reason_code = str(metric_event.get("gate_reason_code") or "")
                self._last_rejected_gate_artifact_sha = observed_artifact_sha
                self._last_rejected_gate_reason_code = reason_code
                if reason_code == "metric_missing":
                    self._last_metric_missing_gate_artifact_sha = observed_artifact_sha
            feedback = self._evaluator_feedback_for_agent(metric_event)
            if stop_after_evaluator_query:
                stop_feedback = self._request_evaluator_graceful_stop(
                    agent=agent, stage_id=stage_id_for_eval
                )
                return (
                    metric_event,
                    "\n\n".join((part for part in (feedback, stop_feedback) if part)),
                    True,
                )
            return (metric_event, feedback or None, True)
        if not metric_event or metric_event.get("metric_value") is None:
            feedback = self._evaluator_feedback_for_agent(metric_event)
            if stop_after_evaluator_query:
                stop_feedback = self._request_evaluator_graceful_stop(
                    agent=agent, stage_id=stage_id_for_eval
                )
                return (
                    metric_event,
                    "\n\n".join((part for part in (feedback, stop_feedback) if part)),
                    True,
                )
            if feedback:
                return (metric_event, feedback, True)
            return (metric_event, None, True)
        metric_event.setdefault("capture_type", "evaluator_primary")
        metric_event.setdefault("candidate_ready", True)
        metric_event.setdefault(
            "submission_status", metric_event.get("evaluator_status") or "ready"
        )
        metric_event.setdefault("semantic_source_changed", True)
        metric_event.setdefault(
            "artifact_sha", metric_event.get("submission_sha") or ""
        )
        metric_event["_evaluator_stage_facts_applied"] = True
        return metric_event, None, False

    def _reject_evaluated_stage_gate(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        now: float,
        solution_sha: str,
        run_signature: str,
        stop_after_evaluator_query: bool,
    ) -> str | None:
        self._stage_lifecycle().advance_legacy(
            stage_id,
            StageLifecycleEvent.REJECT_GATE,
            event_id=f"gate:{stage_id}:{metric_event.get('gate_reason_code') or 'rejected'}",
        )
        self.last_stage_commit_ts = now
        self.last_captured_solution_sha = solution_sha
        self.last_captured_run_signature = run_signature
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "stage_gate_rejected",
                "stage_id": stage_id,
                "gate_action": metric_event.get("gate_action"),
                "gate_reason_code": metric_event.get("gate_reason_code"),
                "gate_message": metric_event.get("gate_message"),
                "metric_value": metric_event.get("metric_value"),
                "metric_name": metric_event.get("metric_name"),
                "artifact_path": metric_event.get("artifact_path"),
                "artifact_sha": metric_event.get("artifact_sha"),
                "candidate_ready": metric_event.get("candidate_ready"),
                "selection_eligible": metric_event.get("selection_eligible"),
                "stage_policy": "gate_accept_required_for_stage",
            },
        )
        feedback = self._evaluator_feedback_for_agent(metric_event)
        if stop_after_evaluator_query:
            stop_feedback = self._request_evaluator_graceful_stop(
                agent=agent, stage_id=stage_id
            )
            return "\n\n".join((part for part in (feedback, stop_feedback) if part))
        return feedback or None

    def _mark_valid_stage_run(
        self, *, agent: Any, metric_event: dict[str, Any]
    ) -> None:
        mark_valid = getattr(agent, "_lnr_mark_valid_bare_run", None)
        if callable(mark_valid) and str(metric_event.get("solution_sha") or ""):
            mark_valid(
                bash_cmd=str(metric_event.get("bash_cmd") or ""),
                solution_rel_path=str(
                    metric_event.get("solution_path") or "solution.py"
                ),
            )

    async def _commit_evaluated_stage(
        self,
        *,
        agent: Any,
        cards_before: list[Any],
        stage_id: str,
        metric_event: dict[str, Any],
        semantic_source_changed: bool,
        now: float,
        solution_sha: str,
        run_signature: str,
        stop_after_evaluator_query: bool,
    ) -> str | None:
        self._mark_valid_stage_run(agent=agent, metric_event=metric_event)
        materialized_stage = self._materialized_existing_stage_target(
            cards_before=cards_before,
            metric_event=metric_event,
            semantic_source_changed=bool(semantic_source_changed),
        )
        if materialized_stage:
            self._record_stage_materialization_event(
                target_stage=materialized_stage,
                metric_event=metric_event,
                now=now,
                solution_sha=solution_sha,
                run_signature=run_signature,
            )
            if stop_after_evaluator_query:
                return self._request_evaluator_graceful_stop(
                    agent=agent, stage_id=materialized_stage
                )
            return None
        stage_cap = int(getattr(self.lhr, "stage_capture_max_count", 0) or 0)
        if not self._stage_capture_cap_allows(
            stage_cap=stage_cap,
            cards_before=cards_before,
            stop_after_evaluator_query=stop_after_evaluator_query,
            now=now,
            solution_sha=solution_sha,
            run_signature=run_signature,
            metric_event=metric_event,
        ):
            return None
        primary_evaluator_capture = (
            self._evaluator_stage_source_mode() == "primary"
            and bool(metric_event.get("evaluator_backend"))
        )
        if primary_evaluator_capture and (
            metric_event.get("validation_ok") is False
            or not _csv_bool(metric_event.get("candidate_ready"), default=False)
        ):
            self.last_stage_commit_ts = now
            self.last_captured_solution_sha = solution_sha
            self.last_captured_run_signature = run_signature
            metric_event["candidate_ready"] = False
            metric_event.setdefault(
                "submission_status",
                metric_event.get("evaluator_status") or "invalid_artifact",
            )
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_invalid_evaluator_candidate_skipped",
                    "stage_id": stage_id,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "artifact_path": metric_event.get("artifact_path"),
                    "evaluator_backend": metric_event.get("evaluator_backend"),
                    "evaluator_status": metric_event.get("evaluator_status"),
                    "submission_status": metric_event.get("submission_status"),
                    "candidate_ready": False,
                    "stage_policy": "skip_invalid_evaluator_candidate_not_a_stage",
                },
            )
            feedback = self._evaluator_feedback_for_agent(metric_event)
            if stop_after_evaluator_query:
                stop_feedback = self._request_evaluator_graceful_stop(
                    agent=agent, stage_id=stage_id
                )
                return "\n\n".join((part for part in (feedback, stop_feedback) if part))
            return feedback or None
        if primary_evaluator_capture and metric_event.get("metric_value") is None:
            feedback = self._evaluator_feedback_for_agent(metric_event)
            if stop_after_evaluator_query:
                stop_feedback = self._request_evaluator_graceful_stop(
                    agent=agent, stage_id=stage_id
                )
                return "\n\n".join((part for part in (feedback, stop_feedback) if part))
            if feedback:
                return feedback
            return None
        self._stage_lifecycle().advance_legacy(
            stage_id,
            StageLifecycleEvent.ACCEPT_GATE,
            event_id=f"gate:{stage_id}:{metric_event.get('gate_reason_code') or 'accepted'}",
        )
        if bool(getattr(self.lhr, "stage_commit_text_mode", True)):
            self.pending_text_stage_commit = {
                "stage_id": stage_id,
                "metric_event": copy.deepcopy(metric_event),
                "now": now,
                "solution_sha": solution_sha,
                "run_signature": run_signature,
                "attempts": 0,
                "stop_after_evaluator_query_budget": stop_after_evaluator_query,
            }
            self._set_stage_commit_transient_prompt(
                agent, stage_id=stage_id, metric_event=metric_event
            )
            self._jsonl(
                "lhr_stage_commit_events.jsonl",
                {
                    "event": "stage_commit_text_requested",
                    "stage_id": stage_id,
                    "metric_value": metric_event.get("metric_value"),
                    "metric_name": metric_event.get("metric_name"),
                    "candidate_ready": metric_event.get("candidate_ready"),
                },
            )
            return None
        (ok, reason) = await self._ephemeral_stage_commit(
            agent=agent, stage_id=stage_id, metric_event=metric_event
        )
        if not ok:
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "stage_capture_failed",
                    "stage_id": stage_id,
                    "reason": reason,
                    "metric_event": metric_event,
                },
            )
            return None
        finalize_out = await self._finalize_stage_capture_after_commit(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            now=now,
            solution_sha=solution_sha,
            run_signature=run_signature,
            allow_followup=not stop_after_evaluator_query,
        )
        if stop_after_evaluator_query:
            return self._request_evaluator_graceful_stop(agent=agent, stage_id=stage_id)
        return finalize_out

    async def _stage_capture_callback_impl(
        self, *, agent: Any, args: dict[str, Any], tool_result: ToolResult
    ) -> str | None:
        _ = (args, tool_result)
        if not bool(self.lhr.stage_capture_enabled):
            return None
        now = time.monotonic()
        min_gap = max(0.0, float(self.lhr.stage_commit_min_seconds_between or 0.0))
        if self.last_stage_commit_ts and now - self.last_stage_commit_ts < min_gap:
            return None
        pending_artifact_sha = str(
            getattr(agent, "_lnr_candidate_artifact_pending_sha", "") or ""
        ).strip()
        fresh_snapshot = bool(getattr(agent, "_lnr_snapshot_ok", False))
        # A newly materialized artifact without a fresh score must go through
        # its evaluator. Reusing the previous run's snapshot would pair two
        # unrelated events.
        metric_event = (
            None
            if pending_artifact_sha and not fresh_snapshot
            else self._metric_event_from_workspace()
        )
        if not metric_event:
            metric_event, early_response, finished = (
                self._evaluate_missing_metric_stage(
                    agent=agent, metric_event=metric_event
                )
            )
            if finished:
                return early_response
        metric_only = (
            str(metric_event.get("stage_signal_kind") or "").strip().lower()
            == "metric_only"
            or metric_event.get("candidate_artifact_attached") is False
        )
        if not metric_only and not str(metric_event.get("artifact_sha") or "").strip():
            artifact_sha_from_file = self._candidate_artifact_sha_from_workspace(
                metric_event
            )
            if artifact_sha_from_file:
                metric_event["artifact_sha"] = artifact_sha_from_file
                metric_event.setdefault(
                    "artifact_path", self._evaluator_candidate_artifact()
                )
        # The immutable metric event now owns the artifact association. A later
        # score-only run must not silently reuse this candidate.
        try:
            setattr(agent, "_lnr_candidate_artifact_pending_sha", "")
        except (AttributeError, TypeError):
            pass
        solution_sha = str(metric_event.get("solution_sha") or "")
        run_signature = _stage_run_signature(metric_event)
        if not run_signature and metric_event.get("artifact_sha"):
            run_signature = "evaluator:" + str(metric_event.get("artifact_sha") or "")
        if run_signature and run_signature == str(
            getattr(self, "last_captured_run_signature", "") or ""
        ):
            return None
        generic_evaluator_capture = (
            self._evaluator_stage_source_mode() == "primary"
            and bool(metric_event.get("evaluator_backend"))
        )
        submission_sha = str(metric_event.get("submission_sha") or "").strip()
        artifact_sha = str(metric_event.get("artifact_sha") or "").strip()
        submission_status_hint = str(
            metric_event.get("submission_status") or ""
        ).strip()
        submission_invalid = (
            metric_event.get("submission_validation_ok") is False
            or submission_status_hint == "invalid_submission"
        )
        duplicate = self._stage_with_submission_sha(
            self.stage_snapshots, submission_sha
        ) or self._stage_with_artifact_sha(self.stage_snapshots, artifact_sha)
        semantic_source_changed: bool | None = None
        if bool(getattr(self.lhr, "workspace_git_enabled", True)):
            try:
                semantic_source_changed = workspace_source_changed(
                    self.workspace_dir,
                    getattr(self.lhr, "workspace_git_track_globs", None),
                )
            except Exception as exc:
                metric_event["semantic_source_change_error"] = str(exc)
        if semantic_source_changed is None:
            raw_source_changed = metric_event.get(
                "semantic_source_changed", metric_event.get("source_changed")
            )
            semantic_source_changed = (
                raw_source_changed if isinstance(raw_source_changed, bool) else True
            )
        metric_event["semantic_source_changed"] = bool(semantic_source_changed)
        metric_event.setdefault("capture_type", "full_stage")
        if not self._apply_submission_capture_policy(
            metric_event=metric_event,
            submission_invalid=submission_invalid,
            now=now,
            solution_sha=solution_sha,
            run_signature=run_signature,
            submission_sha=submission_sha,
            duplicate=duplicate,
            artifact_sha=artifact_sha,
            semantic_source_changed=semantic_source_changed,
            generic_evaluator_capture=generic_evaluator_capture,
        ):
            return None
        cards_before = parse_stage_cards(read_ledger(self.ledger_path))
        stage_id = self._next_active_stage_id(cards_before)
        if not metric_event.pop("_evaluator_stage_facts_applied", False):
            metric_event = self._record_evaluator_stage_events(
                stage_id=stage_id, metric_event=metric_event
            )
        self._stage_lifecycle().advance_legacy(
            stage_id, StageLifecycleEvent.ASSESS, event_id=f"assessment:{stage_id}"
        )
        stop_after_evaluator_query = self._evaluator_query_budget_terminal(metric_event)
        gate_evaluated = bool(metric_event.get("_gate_evaluated"))
        if (
            self._evaluator_stage_source_mode() != "shadow"
            and gate_evaluated
            and (metric_event.get("gate_accepted") is not True)
        ):
            return self._reject_evaluated_stage_gate(
                agent=agent,
                stage_id=stage_id,
                metric_event=metric_event,
                now=now,
                solution_sha=solution_sha,
                run_signature=run_signature,
                stop_after_evaluator_query=stop_after_evaluator_query,
            )
        return await self._commit_evaluated_stage(
            agent=agent,
            cards_before=cards_before,
            stage_id=stage_id,
            metric_event=metric_event,
            semantic_source_changed=bool(semantic_source_changed),
            now=now,
            solution_sha=solution_sha,
            run_signature=run_signature,
            stop_after_evaluator_query=stop_after_evaluator_query,
        )

    def _agent_factory_service(self) -> AgentFactory:
        factory = getattr(self, "agent_factory", None)
        if isinstance(factory, AgentFactory):
            return factory

        def _trace(record: dict[str, Any]) -> None:
            append = getattr(self, "_record_agent_factory_trace", None)
            if callable(append):
                append(record)

        factory = AgentFactory(
            self.orchestrator.create_science_agent,
            trace_sink=(
                _trace if getattr(self, "telemetry_journal", None) is not None else None
            ),
        )
        self.agent_factory = factory
        return factory
