# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: metric interpretation, validity adjudication, and performance projection.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.control.ephemeral_agent_session import llm_correlation
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
    _fmt_csv_value,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.coordination import (
    _compact_event_text,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    LHR_STAGE_PERFORMANCE_COLUMNS,
    LHR_STAGE_PERFORMANCE_CSV,
    Any,
    Message,
    Path,
    StageSnapshot,
    adjudicate_metric_validity,
    build_metric_output_interpreter_prompt,
    build_metric_output_interpreter_system_prompt,
    build_metric_validity_adjudicator_prompt,
    build_metric_validity_adjudicator_system_prompt,
    build_metric_validity_fields,
    csv,
    infer_metric_validity,
    logger,
    parse_metric_output_interpretation_text,
    parse_metric_validity_judgment_text,
    parse_stage_cards,
    re,
    read_ledger,
    time,
)


class EvaluationAdapterOwner:
    """Own metric interpretation and agent-visible evaluation projection."""

    async def _metric_output_interpretation_callback(
        self,
        *,
        agent: Any,
        stdout: str,
        script_label: str,
    ) -> dict[str, Any] | None:
        if not bool(getattr(self.lhr, "metric_validity_adjudicator_enabled", True)):
            return None
        metric_llm = self._metric_feedback_llm_client()
        if metric_llm is None or not hasattr(metric_llm, "ask_tool_stream"):
            return None

        lines = str(stdout or "").splitlines()
        stdout_tail = "\n".join(lines[-160:])[-12000:]
        candidate_lines = [
            line
            for line in lines
            if re.search(
                r"(?<![A-Za-z0-9])"
                r"(?:best|final|val(?:idation)?|hold[ _-]?out|cv|oof|metric|score)"
                r"(?![A-Za-z0-9])",
                line,
                re.IGNORECASE,
            )
            and re.search(r"\d", line)
        ]
        metric_candidates = "\n".join(candidate_lines[-80:])[-8000:]
        facts = {
            "script_label": str(script_label or "solution.py")[:300],
            "task_metric_context_excerpt": self._task_metric_context_excerpt(),
            "metric_candidate_lines": metric_candidates,
            "stdout_tail": stdout_tail,
        }
        prompt = build_metric_output_interpreter_prompt(facts)
        stage_id = self._next_stage_id_for_logging()
        lineage_id = self._lineage_uid_prefix()
        try:
            text = await agent.run_ephemeral_agentic_route_prompt(
                prompt,
                trigger="metric_output_interpretation",
                base_messages=[],
                system_messages=[
                    Message.system_message(
                        build_metric_output_interpreter_system_prompt()
                    )
                ],
                timeout=max(
                    1.0,
                    float(
                        getattr(
                            self.lhr, "metric_validity_adjudicator_timeout_sec", 60.0
                        )
                        or 60.0
                    ),
                ),
                llm_override=metric_llm,
                llm_role="feedback",
                stage_id=stage_id,
                lineage_id=lineage_id,
                node_uid=self._stage_node_uid(stage_id, lineage_id=lineage_id),
            )
            self._accumulate_ephemeral_tokens(agent, "stage", llm=metric_llm)
            interpretation = parse_metric_output_interpretation_text(text)
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_output_interpreter_response",
                    "parsed": interpretation is not None,
                    "script_label": facts["script_label"],
                    "response": text[:1600],
                    "llm_correlation": llm_correlation(text),
                },
            )
            if interpretation is None:
                return None
            return {
                "metric_found": interpretation.metric_found,
                "metric_name": interpretation.metric_name,
                "metric_value": interpretation.metric_value,
                "split": interpretation.split,
                "is_final": interpretation.is_final,
                "evidence_line": interpretation.evidence_line,
                "confidence": interpretation.confidence,
                "reason": interpretation.reason,
            }
        except Exception as exc:
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_output_interpreter_error",
                    "script_label": facts["script_label"],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return None

    async def _metric_validity_feedback_judgment(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        card_fields: dict[str, Any],
        scope: str,
    ) -> Any | None:
        if (
            metric_event.get("metric_authoritative") is True
            and str(metric_event.get("evaluator_status") or "").strip().lower() == "ok"
            and metric_event.get("validation_ok") is True
        ):
            return None
        llm_enabled = bool(
            getattr(self.lhr, "metric_validity_adjudicator_enabled", True)
        )
        if not llm_enabled:
            return None
        metric_llm = self._metric_feedback_llm_client()
        if metric_llm is None or not hasattr(metric_llm, "ask_tool_stream"):
            return None
        prompt = build_metric_validity_adjudicator_prompt(
            self._metric_validity_fact_card(
                metric_event=metric_event, card_fields=card_fields
            )
        )
        try:
            text = await agent.run_ephemeral_agentic_route_prompt(
                prompt,
                trigger="metric_validity",
                base_messages=[],
                system_messages=[
                    Message.system_message(
                        build_metric_validity_adjudicator_system_prompt()
                    )
                ],
                timeout=max(
                    1.0,
                    float(
                        getattr(
                            self.lhr, "metric_validity_adjudicator_timeout_sec", 60.0
                        )
                        or 60.0
                    ),
                ),
                llm_override=metric_llm,
                llm_role="feedback",
                stage_id=stage_id,
                lineage_id=self._lineage_uid_prefix(),
                node_uid=self._stage_node_uid(stage_id),
            )
            self._accumulate_ephemeral_tokens(agent, "stage", llm=metric_llm)
            judgment = parse_metric_validity_judgment_text(text)
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_validity_adjudicator_response",
                    "stage_id": stage_id,
                    "scope": scope,
                    "parsed": bool(judgment),
                    "response": text[:1200],
                    "llm_correlation": llm_correlation(text),
                },
            )
            return judgment
        except BaseException as exc:
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_validity_adjudicator_error",
                    "stage_id": stage_id,
                    "scope": scope,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return None

    @staticmethod
    def _stage_commit_card_fields_from_judgment(
        judgment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data = judgment if isinstance(judgment, dict) else {}
        return {
            "metric_validity": data.get("metric_validity") or "",
            "brief": data.get("brief") or data.get("BRIEF") or "",
            "why": data.get("why") or data.get("WHY") or "",
            "route_evidence": data.get("route_evidence")
            or data.get("routeEvidence")
            or data.get("route")
            or "",
            "metric_note": data.get("metric_source") or data.get("metric_note") or "",
        }

    async def _audit_stage_result_before_commit(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        judgment: dict[str, Any] | None,
        block_text: str,
        source: str,
    ) -> None:
        before = {
            "lower_is_better": metric_event.get("lower_is_better"),
            "metric_validity": metric_event.get("metric_validity"),
            "selection_eligible": metric_event.get("selection_eligible"),
        }
        data = judgment if isinstance(judgment, dict) else {}
        if data.get("lower_is_better") not in (None, ""):
            metric_event["declared_stage_commit_lower_is_better"] = data.get(
                "lower_is_better"
            )
        self._apply_metric_direction_audit(metric_event)
        card_fields = self._stage_commit_card_fields_from_judgment(data)
        fields = build_metric_validity_fields(
            metric_event=metric_event, card_fields=card_fields
        )
        feedback = await self._metric_validity_feedback_judgment(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            card_fields=card_fields,
            scope="pre_commit",
        )
        if feedback is not None and feedback.expected_lower_is_better is not None:
            expected = bool(feedback.expected_lower_is_better)
            current = bool(metric_event.get("lower_is_better") is not False)
            source_name = str(metric_event.get("metric_direction_source") or "")
            if current != expected:
                metric_event["metric_direction_conflict"] = True
                if (
                    source_name in {"", "declared", "default_lower"}
                    and str(feedback.confidence or "").lower() == "high"
                ):
                    metric_event["lower_is_better"] = expected
                    metric_event["metric_direction_source"] = "feedback_llm"
            if feedback.metric_direction_reason:
                metric_event["metric_direction_feedback_reason"] = (
                    feedback.metric_direction_reason
                )
        adjudicated = adjudicate_metric_validity(fields, llm_judgment=feedback)
        metric_event.update(adjudicated)
        if not bool(adjudicated.get("selection_eligible")):
            metric_event["selection_eligible"] = False
        metric_event["stage_result_audited_before_commit"] = True
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_result_audited_before_commit",
                "stage_id": stage_id,
                "source": source,
                "before": before,
                "after": {
                    "lower_is_better": metric_event.get("lower_is_better"),
                    "metric_validity": metric_event.get("metric_validity"),
                    "selection_eligible": metric_event.get("selection_eligible"),
                    "metric_direction_source": metric_event.get(
                        "metric_direction_source"
                    ),
                    "metric_direction_conflict": metric_event.get(
                        "metric_direction_conflict"
                    ),
                    "metric_validity_source": metric_event.get(
                        "metric_validity_source"
                    ),
                    "metric_validity_reason_code": metric_event.get(
                        "metric_validity_reason_code"
                    ),
                },
                "declared_stage_commit_lower_is_better": metric_event.get(
                    "declared_stage_commit_lower_is_better"
                ),
                "block_text": _compact_event_text(block_text),
            },
        )

    async def _adjudicate_metric_validity_for_stage(
        self,
        *,
        agent: Any,
        stage_id: str,
        metric_event: dict[str, Any],
        cards_after: list[Any],
    ) -> None:
        if metric_event.get("stage_result_audited_before_commit"):
            self._jsonl(
                "lhr_stage_events.jsonl",
                {
                    "event": "metric_validity_adjudicated",
                    "stage_id": stage_id,
                    "metric_value": metric_event.get("metric_value"),
                    "scope": "pre_commit_reused",
                    "metric_validity": metric_event.get("metric_validity"),
                    "metric_validity_note": metric_event.get("metric_validity_note"),
                    "metric_validity_reason_code": metric_event.get(
                        "metric_validity_reason_code"
                    ),
                    "metric_validity_confidence": metric_event.get(
                        "metric_validity_confidence"
                    ),
                    "metric_validity_source": metric_event.get(
                        "metric_validity_source"
                    ),
                    "selection_eligible": metric_event.get("selection_eligible"),
                },
            )
            return
        card_fields = self._metric_validity_card_fields(cards_after, stage_id)
        fields = build_metric_validity_fields(
            metric_event=metric_event, card_fields=card_fields
        )
        judgment = await self._metric_validity_feedback_judgment(
            agent=agent,
            stage_id=stage_id,
            metric_event=metric_event,
            card_fields=card_fields,
            scope="post_commit",
        )
        adjudicated = adjudicate_metric_validity(fields, llm_judgment=judgment)
        metric_event.update(adjudicated)
        if not bool(adjudicated.get("selection_eligible")):
            metric_event["selection_eligible"] = False
        self._jsonl(
            "lhr_stage_events.jsonl",
            {
                "event": "metric_validity_adjudicated",
                "stage_id": stage_id,
                "metric_value": metric_event.get("metric_value"),
                **adjudicated,
            },
        )

    def _build_stage_performance_row(
        self,
        *,
        out: Path,
        stage_id: str,
        snap: StageSnapshot,
        metric_event: dict[str, Any],
        worker_id: str,
        node_uid: str,
        lineage_id: str,
        candidate_id: str,
        parent_node_uid: str,
        parent_stage: str,
        restored_node_uid: str,
        restored: str,
        lower_is_better: bool,
        metric_source_note: str,
        metric_validity: str,
        metric_validity_note: str,
        brief: str,
        why: str,
        route_evidence: str,
        estra_counts: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "row_order": self._next_global_stage_row_order(out),
            "candidate_id": candidate_id,
            "worker_id": worker_id,
            "worker_index": self.worker_index,
            "worker_stage_order": len(self.stage_snapshots),
            "stage_id": stage_id,
            "lineage_id": lineage_id,
            "node_uid": node_uid,
            "parent_candidate_id": parent_node_uid,
            "parent_stage_id": parent_stage,
            "restored_from_candidate_id": restored_node_uid,
            "restored_from_stage": restored,
            "restored_from_node_uid": restored_node_uid,
            "metric_value": metric_event.get("metric_value"),
            "metric_name": metric_event.get("metric_name") or snap.metric_name,
            "lower_is_better": lower_is_better,
            "validation_ok": metric_event.get("validation_ok"),
            "validation_issue": metric_event.get("validation_issue"),
            "reported_val_score": metric_event.get("reported_val_score"),
            "val_score_type": metric_event.get("val_score_type"),
            "selection_eligible": metric_event.get("selection_eligible"),
            "selection_score": metric_event.get("selection_score"),
            "selection_note": metric_event.get("selection_note"),
            "metric_source_note": metric_source_note,
            "metric_validity": metric_validity,
            "metric_validity_note": metric_validity_note,
            "metric_validity_reason_code": metric_event.get(
                "metric_validity_reason_code"
            ),
            "metric_validity_confidence": metric_event.get(
                "metric_validity_confidence"
            ),
            "metric_validity_source": metric_event.get("metric_validity_source"),
            "task_profile": metric_event.get("task_profile")
            or self._evaluator_task_profile(),
            "metric_protocol": metric_event.get("metric_protocol"),
            "train_data_used": metric_event.get("train_data_used"),
            "metric_eval_data": metric_event.get("metric_eval_data"),
            "artifacts_reused_from": metric_event.get("artifacts_reused_from"),
            "training_rows": metric_event.get("training_rows"),
            "validation_rows": metric_event.get("validation_rows"),
            "execution_mode": metric_event.get("execution_mode"),
            "route_id": metric_event.get("route_id"),
            "execution_scale": metric_event.get("execution_scale"),
            "is_best_so_far": self._best_stage_id_so_far() == stage_id,
            "brief": brief,
            "why": why,
            "route_evidence": route_evidence,
            "solution_sha": metric_event.get("solution_sha"),
            "submission_sha": metric_event.get("submission_sha"),
            "source_commit_sha": metric_event.get("source_commit_sha"),
            "source_changed": metric_event.get("source_changed"),
            "semantic_source_changed": metric_event.get("semantic_source_changed"),
            "capture_type": metric_event.get("capture_type"),
            "workspace_git_stage_id": metric_event.get("workspace_git_stage_id"),
            "submission_snapshot": metric_event.get("submission_snapshot"),
            "artifact_path": metric_event.get("artifact_path"),
            "artifact_sha": metric_event.get("artifact_sha"),
            "artifact_kind": metric_event.get("artifact_kind"),
            "evaluator_backend": metric_event.get("evaluator_backend"),
            "evaluator_status": metric_event.get("evaluator_status"),
            "gate_metric_validity": metric_event.get("gate_metric_validity"),
            "gate_policy": metric_event.get("gate_policy"),
            "gate_policy_version": metric_event.get("gate_policy_version"),
            "gate_action": metric_event.get("gate_action"),
            "gate_accepted": metric_event.get("gate_accepted"),
            "gate_reason_code": metric_event.get("gate_reason_code"),
            "submission_changed": metric_event.get("submission_changed"),
            "candidate_ready": metric_event.get("candidate_ready"),
            "submission_status": metric_event.get("submission_status"),
            "duplicate_submission_of_stage": metric_event.get(
                "duplicate_submission_of_stage"
            ),
            "duplicate_submission_of_snapshot_id": metric_event.get(
                "duplicate_submission_of_snapshot_id"
            ),
            "workspace_git_ready": metric_event.get("workspace_git_ready"),
            "workspace_git_message": metric_event.get("workspace_git_message"),
            "solution_run_sec": metric_event.get("wall_sec")
            or metric_event.get("duration_sec"),
            "elapsed_min": (time.time() - self.started_at) / 60.0,
            "main_llm_calls": self.main_llm_calls,
            "main_tokens_input": self.main_tokens_in,
            "main_tokens_output": self.main_tokens_out,
            "main_tokens_cached": self.main_tokens_cached,
            "main_cache_rate": self.main_tokens_cached / self.main_tokens_in
            if self.main_tokens_in
            else "",
            "stage_commit_llm_calls": self.stage_llm_calls,
            "estra_llm_calls": self.estra_llm_calls,
            "estra_decision_count_before": estra_counts["decisions"],
            "estra_continue_count_before": estra_counts["continue"],
            "estra_redirect_count_before": estra_counts["redirect"],
            "estra_switch_count_before": estra_counts["switch"],
            "estra_current_continue_count_before": estra_counts["current_continue"],
            "estra_current_redirect_count_before": estra_counts["current_redirect"],
            "estra_stage_continue_count_before": estra_counts["stage_continue"],
            "estra_stage_redirect_count_before": estra_counts["stage_redirect"],
            "estra_switch_stage_count_before": self._count_jsonl_events(
                "lhr_estras.jsonl", "estra_stage_switched"
            ),
            "snapshot_id": snap.snapshot_id,
            "snapshot_path": str(snap.snapshot_path),
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _append_stage_performance_row(
        self,
        *,
        stage_id: str,
        snap: StageSnapshot,
        metric_event: dict[str, Any],
        cards_after: list[Any],
    ) -> None:
        worker_id = self._worker_uid_prefix()
        node_uid = self._snapshot_node_uid(snap) or self._stage_node_uid(stage_id)
        lineage_id = str(
            getattr(snap, "lineage_id", "")
            or (
                snap.source_event.get("lineage_id")
                if isinstance(snap.source_event, dict)
                else ""
            )
            or self._lineage_uid_prefix()
        )
        candidate_id = node_uid
        parent_stage = ""
        for idx, card in enumerate(cards_after):
            if card.stage_id == stage_id and idx > 0:
                parent_stage = cards_after[idx - 1].stage_id
                break
        card_map = {c.stage_id: c for c in cards_after}
        card = card_map.get(stage_id)
        restored = self.current_restored_from_stage
        restored_node_uid = self.current_restored_from_node_uid
        parent_node_uid = (
            self._active_node_uid_for_stage(parent_stage) if parent_stage else ""
        )
        out = self.global_log_dir / LHR_STAGE_PERFORMANCE_CSV
        brief = getattr(card, "brief", "") if card is not None else ""
        why = getattr(card, "why", "") if card is not None else ""
        route_evidence = getattr(card, "route_evidence", "") if card is not None else ""
        card_metric_validity = (
            getattr(card, "metric_validity", "") if card is not None else ""
        )
        metric_source_note = str(metric_event.get("metric_source_note") or "").strip()
        if not metric_source_note:
            metric_source_note = self._metric_source_note(
                metric_event, brief=brief, why=why
            )
        lower_is_better = self._metric_lower_is_better_for_event(
            metric_event, fallback=snap.lower_is_better
        )
        if metric_event.get("metric_validity_source"):
            metric_validity = str(metric_event.get("metric_validity") or "medium")
            metric_validity_note = str(
                metric_event.get("metric_validity_note")
                or "metric_validity_adjudicated"
            )
        else:
            metric_validity, metric_validity_note = infer_metric_validity(
                {
                    **metric_event,
                    "metric_source_note": metric_source_note,
                    "metric_validity": card_metric_validity
                    or metric_event.get("metric_validity"),
                    "brief": brief,
                    "why": why,
                    "route_evidence": route_evidence,
                }
            )
        estra_counts = self._estra_decision_counts()
        row = self._build_stage_performance_row(
            out=out,
            stage_id=stage_id,
            snap=snap,
            metric_event=metric_event,
            worker_id=worker_id,
            node_uid=node_uid,
            lineage_id=lineage_id,
            candidate_id=candidate_id,
            parent_node_uid=parent_node_uid,
            parent_stage=parent_stage,
            restored_node_uid=restored_node_uid,
            restored=restored,
            lower_is_better=lower_is_better,
            metric_source_note=metric_source_note,
            metric_validity=metric_validity,
            metric_validity_note=metric_validity_note,
            brief=brief,
            why=why,
            route_evidence=route_evidence,
            estra_counts=estra_counts,
        )
        try:
            self.global_log_dir.mkdir(parents=True, exist_ok=True)
            file_exists = out.exists() and out.stat().st_size > 0
            with out.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LHR_STAGE_PERFORMANCE_COLUMNS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        k: _fmt_csv_value(row.get(k))
                        for k in LHR_STAGE_PERFORMANCE_COLUMNS
                    }
                )
        except OSError:
            logger.debug("[lnr] stage performance csv append failed", exc_info=True)
        self.current_restored_from_stage = ""
        self.current_restored_from_node_uid = ""

    def _accumulate_ephemeral_tokens(
        self, agent: Any, kind: str, *, llm: Any | None = None
    ) -> None:
        source_llm = llm or agent.llm
        try:
            ti = int(getattr(source_llm, "_last_call_input_tokens", 0) or 0)
            to = int(getattr(source_llm, "_last_call_output_tokens", 0) or 0)
            tc = int(getattr(source_llm, "_last_call_input_cached_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        if kind == "stage":
            self.stage_tokens_in += ti
            self.stage_tokens_out += to
            self.stage_tokens_cached += tc
            self.stage_llm_calls += 1
        else:
            self.estra_tokens_in += ti
            self.estra_tokens_out += to
            self.estra_tokens_cached += tc
            self.estra_llm_calls += 1

    def _sanitize_agent_visible_payload(self, agent: Any, value: Any) -> Any:
        if isinstance(value, str):
            return agent._sanitize_agent_visible_paths(value)
        if isinstance(value, dict):
            return {
                str(k): self._sanitize_agent_visible_payload(agent, v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize_agent_visible_payload(agent, v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_agent_visible_payload(agent, v) for v in value)
        return value

    def _sanitize_agent_prompt_surfaces(self, agent: Any) -> None:
        for attr in ("systemPrompt", "_system_prompt_core"):
            value = getattr(agent, attr, None)
            if isinstance(value, str) and value:
                try:
                    setattr(agent, attr, agent._sanitize_agent_visible_paths(value))
                except Exception:
                    logger.debug(
                        "[lnr] prompt path sanitize skipped for %s", attr, exc_info=True
                    )
        tools = getattr(agent, "_tools_with_thought", None)
        if tools:
            try:
                setattr(
                    agent,
                    "_tools_with_thought",
                    self._sanitize_agent_visible_payload(agent, tools),
                )
            except Exception:
                logger.debug("[lnr] tool schema path sanitize skipped", exc_info=True)
        sanitizer = getattr(agent, "sanitize_existing_memory_agent_visible_paths", None)
        if callable(sanitizer):
            try:
                sanitizer()
            except Exception:
                logger.debug("[lnr] loaded memory path sanitize skipped", exc_info=True)

    def _prune_active_stage_snapshots_to_ledger(self) -> None:
        cards = parse_stage_cards(read_ledger(self.ledger_path))
        active = {str(card.stage_id).upper() for card in cards}
        if not active:
            return
        removed = sorted(
            sid for sid in self.stage_snapshots if str(sid).upper() not in active
        )
        removed_node_uids = {
            sid: self._snapshot_node_uid(self.stage_snapshots.get(sid))
            for sid in removed
        }
        if not removed:
            return
        self.stage_snapshots = {
            sid: snap
            for sid, snap in self.stage_snapshots.items()
            if str(sid).upper() in active
        }
        self._write_stage_map()
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_active_stage_pruned",
                "active_stages": sorted(active),
                "removed_stages": removed,
                "removed_node_uids": removed_node_uids,
            },
        )

    def _sanitize_metric_event_for_agent_prompt(
        self,
        agent: Any,
        metric_event: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._sanitize_agent_visible_payload(agent, dict(metric_event or {}))
        if isinstance(out, dict) and out.get("snapshot_path"):
            out["snapshot_path"] = ".logs/fullrun_tail_snapshot.json"
        if isinstance(out, dict):
            out.pop("bash_cmd", None)
        return out if isinstance(out, dict) else {}
