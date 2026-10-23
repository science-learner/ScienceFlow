# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Resource observer responsibility: metric value assessment, intervention summaries, and proposals.

Function bodies are mechanically moved from the frozen V3 baseline. They
retain no host back-reference and do not duplicate resource-domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    ResourceJob,
    active_work_counterevidence,
    build_stage_performance_score_summary,
    classify_deliverable_validity,
    classify_route_viability,
    metric_lower_is_better_hint,
    normalize_route_key,
    normalize_validation_protocol,
    re,
    time,
    validation_protocols_comparable,
)


class ExecutionValueCallbacks:
    """Own this responsibility's state transitions and callbacks."""

    def _record_v7_resource_event(self, event_type: str, job: ResourceJob, payload: dict[str, Any]) -> None:
        if self.resource_runtime is None:
            return
        self.resource_runtime.record_resource_event(
            event_type,
            payload={"schema_version": 1, **dict(payload or {})},
            command_id=job.job_id,
            lease_id=job.job_id,
        )


    def _update_v7_route_profile(self, job: ResourceJob, signal: dict[str, Any], *, elapsed_sec: float) -> dict[str, Any]:
        raw_state = job.last_deliverable_completion_check.get("state") if isinstance(job.last_deliverable_completion_check, dict) else {}
        state = raw_state if isinstance(raw_state, dict) else {}
        validity = classify_deliverable_validity(state)
        job.deliverable_validity = validity
        route = classify_route_viability(
            elapsed_sec=elapsed_sec,
            resource_class=job.resource_class,
            progress_snapshot=self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed_sec),
            deliverable_validity=validity,
            thresholds=self._lease_suspect_thresholds(),
        )
        job.route_viability_state = dict(route)
        job.route_viability_proven = bool(route.get("route_viability_proven"))
        return route


    def _blocker_fact_card(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
        resource_confidence: str,
    ) -> dict[str, Any]:
        progress = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed_sec)
        return {
            "job_id": job.job_id,
            "worker_id": self.worker_id,
            "resource_class": job.resource_class,
            "gpu_ids": list(job.gpu_ids or []),
            "runtime_sec": float(elapsed_sec or 0.0),
            "progress_signal": progress.get("progress_signal"),
            "progress_confidence": progress.get("progress_confidence"),
            "progress_signal_windows": progress.get("progress_signal_windows"),
            "resource_confidence": resource_confidence,
            "route_viability": dict(job.route_viability_state or {}),
            "active_lease_suspect": dict(job.active_lease_suspect_state or {}),
            "process_liveness": self._process_liveness_for_job(job, signal),
            "last_gpu_util_sample": dict(job.last_gpu_util_sample or {}),
            "value_hint": dict(job.value_hint or {}),
            "recoverability": self._fresh_recoverability_for_job(job, elapsed_sec=elapsed_sec),
            "command_digest": job.command_digest,
            "command_preview": str(job.command or "")[:240],
        }


    def _contention_waiter_cards(
        self,
        job: ResourceJob,
        *,
        snapshot: dict[str, Any],
        now: float,
    ) -> list[dict[str, Any]]:
        leases = snapshot.get("leases") if isinstance(snapshot.get("leases"), dict) else {}
        raw_lease = leases.get(job.job_id) if isinstance(leases, dict) else None
        if not isinstance(raw_lease, dict):
            return []
        blocker_gpu_ids = {str(x) for x in (raw_lease.get("gpu_ids") or job.gpu_ids or []) if str(x).strip()}
        if not blocker_gpu_ids:
            return []
        raw_waiters = snapshot.get("waiters") if isinstance(snapshot.get("waiters"), dict) else {}
        rows: list[dict[str, Any]] = []
        for waiter_id, raw in raw_waiters.items():
            if str(waiter_id) == job.job_id or not isinstance(raw, dict):
                continue
            waiter_gpu_ids = self._waiter_gpu_ids(raw)
            if waiter_gpu_ids and not (blocker_gpu_ids & waiter_gpu_ids):
                continue
            submitted_at = self._nested_float(raw, "submitted_at", now)
            queue_age = max(0.0, now - submitted_at)
            if queue_age < self.arbiter_contention_min_waiter_age_sec:
                continue
            meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            value_hint = meta.get("value_hint") if isinstance(meta.get("value_hint"), dict) else {}
            priority = meta.get("admission_priority") if isinstance(meta.get("admission_priority"), dict) else {}
            priority_score = self._nested_float(raw, "priority_score", self._nested_float(priority, "admission_priority_score", 0.0))
            rows.append({
                "job_id": str(raw.get("job_id") or waiter_id),
                "worker_id": str(raw.get("worker_id") or ""),
                "resource_class": str(raw.get("resource_class") or ""),
                "gpu_ids": [str(x) for x in (raw.get("gpu_ids") or []) if str(x).strip()],
                "request_count": int(raw.get("request_count") or 1),
                "queue_age_sec": queue_age,
                "submitted_at": submitted_at,
                "priority_score": priority_score,
                "admission_priority_score": self._nested_float(priority, "admission_priority_score", priority_score),
                "expected_value_score": self._nested_float(value_hint, "expected_value_score", 0.0),
                "near_submission_score": self._nested_float(value_hint, "near_submission_score", 0.0),
                "value_hint": dict(value_hint),
            })
        rows.sort(key=lambda row: (-float(row.get("priority_score") or 0.0), float(row.get("submitted_at") or now), str(row.get("job_id") or "")))
        for index, row in enumerate(rows, start=1):
            row["queue_position"] = index
        return rows[:3]


    def _resource_metric_value_assessment(self, job: ResourceJob) -> dict[str, Any]:
        metric = self._current_structured_metric_for_job(job)
        if not metric:
            return {}
        live_assessment = self._live_metric_value_assessment(job, metric)
        score_context = self._score_context_fact_card()
        best = score_context.get("valid_best_score") if isinstance(score_context.get("valid_best_score"), dict) else {}
        best_value = self._float_or_none(best.get("value") if isinstance(best, dict) else None)
        if best_value is None:
            return live_assessment
        metric_protocol = str(metric.get("validation_protocol") or "").strip()
        metric_fold = str(metric.get("fold") or "").strip().lower()
        best_protocol = str(
            best.get("validation_protocol") or best.get("val_score_type") or ""
        ).strip()
        best_fold = str(best.get("fold") or "").strip().lower()
        incompatible_scope = bool(
            not validation_protocols_comparable(metric_protocol, best_protocol)
            or (bool(metric_fold or best_fold) and metric_fold != best_fold)
        )
        if incompatible_scope:
            return {**live_assessment, "global_comparison_skipped": "metric_scope_mismatch"}
        lower = bool(best.get("lower_is_better")) if isinstance(best, dict) else True
        current = float(metric["value"])
        delta_to_best = current - best_value if lower else best_value - current
        tolerance = max(1e-3, abs(best_value) * 0.003)
        if delta_to_best > tolerance:
            status = "below_observed_best"
            useful = False
        elif delta_to_best < -tolerance:
            status = "above_observed_best"
            useful = True
        else:
            status = "near_observed_best"
            useful = True
        return {
            **metric,
            "status": status,
            "useful": useful,
            "best_value": best_value,
            "best_stage": str(best.get("stage_id") or "") if isinstance(best, dict) else "",
            "best_worker": str(best.get("worker_id") or "") if isinstance(best, dict) else "",
            "best_metric_validity": str(best.get("metric_validity") or "") if isinstance(best, dict) else "",
            "lower_is_better": lower,
            "delta_to_best": delta_to_best,
            "relative_delta_to_best": delta_to_best / max(abs(best_value), 1e-12),
            "tolerance": tolerance,
            "comparison_source": "stage_performance_csv",
        }


    def _live_metric_value_assessment(self, job: ResourceJob, metric: dict[str, Any]) -> dict[str, Any]:
        name = str(metric.get("metric_name") or "").strip().lower()
        current = self._float_or_none(metric.get("value"))
        if not name or current is None:
            return {**metric, "status": "live_metric_unavailable", "useful": None}
        lower = metric_lower_is_better_hint(name)
        if lower is None:
            return {
                **metric,
                "status": "live_metric_direction_unknown",
                "useful": None,
                "comparison_source": "live_job_metric",
            }
        state_key = str(metric.get("metric_scope_key") or name)
        state = dict(job.live_metric_state.get(state_key) or {})
        signature = (
            round(float(current), 12),
            str(metric.get("progress_unit") or ""),
            self._float_or_none(metric.get("progress_current")),
            self._float_or_none(metric.get("progress_total")),
        )
        if tuple(state.get("signature") or ()) == signature and isinstance(state.get("assessment"), dict):
            return dict(state["assessment"])
        previous_best = self._float_or_none(state.get("best_value"))
        observation_count = int(state.get("observation_count") or 0) + 1
        if previous_best is None:
            best_value = float(current)
            assessment = {
                **metric,
                "status": "live_metric_baseline_established",
                "useful": None,
                "best_value": best_value,
                "lower_is_better": lower,
                "observation_count": observation_count,
                "comparison_source": "live_job_metric",
            }
        else:
            delta_to_best = float(current) - previous_best if lower else previous_best - float(current)
            tolerance = max(1e-3, abs(previous_best) * 0.003)
            improved = delta_to_best < -tolerance
            if improved:
                status = "above_live_best"
                best_value = float(current)
            elif delta_to_best > tolerance:
                status = "below_live_best"
                best_value = previous_best
            else:
                status = "live_metric_plateau"
                best_value = previous_best
            assessment = {
                **metric,
                "status": status,
                "useful": improved,
                "best_value": previous_best,
                "lower_is_better": lower,
                "delta_to_best": delta_to_best,
                "relative_delta_to_best": delta_to_best / max(abs(previous_best), 1e-12),
                "tolerance": tolerance,
                "observation_count": observation_count,
                "comparison_source": "live_job_metric",
            }
        job.live_metric_state[state_key] = {
            "signature": list(signature),
            "best_value": best_value,
            "observation_count": observation_count,
            "assessment": dict(assessment),
        }
        return assessment


    def _current_structured_metric_for_job(self, job: ResourceJob) -> dict[str, Any]:
        progress = job.last_progress if isinstance(job.last_progress, dict) else {}
        signals = progress.get("signals") if isinstance(progress.get("signals"), dict) else {}
        raw_metrics = signals.get("metrics") if isinstance(signals.get("metrics"), dict) else {}
        metric_evidence = (
            signals.get("metric_evidence")
            if isinstance(signals.get("metric_evidence"), dict)
            else {}
        )
        for raw_name, raw_value in raw_metrics.items():
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_name or "").strip().lower()).strip("_")
            if not name or name in {"loss", "train_loss", "training_loss"}:
                continue
            value = self._float_or_none(raw_value)
            if value is None:
                continue
            heartbeat = signals.get("heartbeat") if isinstance(signals.get("heartbeat"), dict) else {}
            phase = str(signals.get("phase") or heartbeat.get("phase") or progress.get("phase") or "").strip()
            route_id = str(
                heartbeat.get("route_id")
                or signals.get("route_id")
                or (job.value_hint or {}).get("route_id")
                or self._research_route_key_for_job(job)
                or "unknown"
            ).strip()
            fold = str(heartbeat.get("fold") or signals.get("fold") or "").strip()
            raw_validation_protocol = str(
                heartbeat.get("validation_protocol") or signals.get("validation_protocol") or ""
            ).strip()
            validation_protocol = (
                normalize_validation_protocol(raw_validation_protocol)
                or normalize_route_key(raw_validation_protocol).lower()
            )
            scope_values = (route_id, phase or "unknown", fold or "unknown", name, validation_protocol or "unknown")
            metric_scope_key = "|".join(re.sub(r"[^a-z0-9_.:-]+", "_", item.lower()).strip("_") or "unknown" for item in scope_values)
            return {
                "metric_name": name,
                "value": value,
                "phase": phase,
                "route_id": route_id,
                "fold": fold,
                "validation_protocol": validation_protocol,
                "metric_scope_key": metric_scope_key,
                "evidence_source": str(metric_evidence.get("source") or "unknown"),
                "evidence_trust": str(metric_evidence.get("trust") or "unknown"),
                "progress_unit": str((progress.get("structured_progress") or {}).get("unit") or ""),
                "progress_current": (progress.get("structured_progress") or {}).get("current"),
                "progress_total": (progress.get("structured_progress") or {}).get("total"),
            }
        return {}


    @staticmethod
    def _metric_value_useful_from_signal(signal: dict[str, Any]) -> bool | None:
        assessment = signal.get("resource_metric_value") if isinstance(signal.get("resource_metric_value"), dict) else {}
        useful = assessment.get("useful") if isinstance(assessment, dict) else None
        return useful if isinstance(useful, bool) else None


    def _attach_resource_metric_value_assessment(self, job: ResourceJob, signal: dict[str, Any]) -> None:
        assessment = self._resource_metric_value_assessment(job)
        if not assessment:
            return
        signal["resource_metric_value"] = assessment
        useful = assessment.get("useful")
        if isinstance(useful, bool):
            signal["metric_value_useful"] = useful


    def _update_execution_value_shadow(
        self,
        job: ResourceJob,
        signal: dict[str, Any],
        *,
        elapsed_sec: float,
    ) -> None:
        """Record an advisory value decision without changing resource control."""

        artifact = self._fresh_artifact_update_for_job(job, elapsed_sec=elapsed_sec)
        recoverability = self._fresh_recoverability_for_job(
            job, elapsed_sec=elapsed_sec
        )
        job.execution_value_shadow = self.execution_value_shadow.decide(
            command_id=job.job_id,
            elapsed_sec=elapsed_sec,
            signal=signal,
            artifact_progress=bool(artifact),
            recoverable_artifact=bool(
                recoverability.get("recoverable_artifact_on_disk")
            ),
        )


    def _score_context_fact_card(self) -> dict[str, Any]:
        if self.task_resource_dir is None:
            return {}
        path = self.task_resource_dir.parent / "lhr_stage_performance.csv"
        summary = build_stage_performance_score_summary(path, current_worker_id="")
        if not summary.get("record_count") and not summary.get("best_score"):
            return {}
        compact: dict[str, Any] = {
            "best_score": summary.get("best_score") or {},
            "valid_best_score": summary.get("valid_best_score") or {},
            "capture_gap": bool(summary.get("capture_gap")),
            "recommended_action": str(summary.get("recommended_action") or ""),
            "cheap_signal_best_score": summary.get("cheap_signal_best_score") or {},
            "record_count": summary.get("record_count") or 0,
            "valid_record_count": summary.get("valid_record_count") or 0,
        }
        for key in ("best_score", "valid_best_score", "cheap_signal_best_score"):
            value = compact.get(key)
            if isinstance(value, dict):
                value.pop("artifact_path", None)
                value.pop("snapshot_path", None)
        return compact


    def _trusted_valid_best_for_intervention(self) -> dict[str, Any]:
        score_context = self._score_context_fact_card()
        valid_best = score_context.get("valid_best_score") if isinstance(score_context.get("valid_best_score"), dict) else {}
        value = valid_best.get("value") if isinstance(valid_best, dict) else None
        if value is None or str(valid_best.get("validity") or "") != "valid_comparable":
            return {
                "current_valid_best": "unknown",
                "current_valid_best_status": "best_unreliable",
                "reason": "metric_direction_or_source_uncertain",
            }
        return {
            "current_valid_best": value,
            "current_valid_best_source": "lhr_stage_performance.csv",
            "current_valid_best_validity": "valid_comparable",
            "lower_is_better": bool(valid_best.get("lower_is_better")),
            "worker_id": str(valid_best.get("worker_id") or ""),
            "stage_id": str(valid_best.get("stage_id") or ""),
        }


    def _resource_intervention_summary_text(self, *, action: str, reason: str) -> str:
        best = self._trusted_valid_best_for_intervention()
        fields = [
            f"action={self._summary_token(action or 'resource_intervention')}",
            f"reason={self._summary_token(reason or 'resource_intervention')}",
        ]
        if best.get("current_valid_best_status") == "best_unreliable":
            fields.extend([
                "current_valid_best=unknown",
                "current_valid_best_status=best_unreliable",
                f"best_unreliable_reason={self._summary_token(best.get('reason') or 'unknown')}",
            ])
        else:
            fields.extend([
                f"current_valid_best={float(best.get('current_valid_best')):.6g}",
                f"current_valid_best_source={best.get('current_valid_best_source')}",
                f"current_valid_best_validity={best.get('current_valid_best_validity')}",
                f"lower_is_better={str(bool(best.get('lower_is_better'))).lower()}",
            ])
            if best.get("worker_id"):
                fields.append(f"best_worker={self._summary_token(best.get('worker_id'))}")
            if best.get("stage_id"):
                fields.append(f"best_stage={self._summary_token(best.get('stage_id'))}")
        return "RESOURCE_INTERVENTION_SUMMARY: " + "; ".join(fields) + ".\n"


    def _append_resource_intervention_summary(self, feedback: str, *, action: str, reason: str) -> str:
        text = str(feedback or "")
        if not text.strip() or "RESOURCE_INTERVENTION_SUMMARY:" in text:
            return text
        if "RESOURCE_FEEDBACK:" not in text:
            return text
        return text.rstrip() + "\n" + self._resource_intervention_summary_text(action=action, reason=reason)


    @staticmethod
    def _summary_token(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip()).strip("_") or "unknown"


    def _record_resource_review_proposal_event(
        self,
        job: ResourceJob,
        proposal: dict[str, Any],
        signal: dict[str, Any],
    ) -> None:
        if self.resource_runtime is None:
            return
        proposal = self._ensure_resource_proposal_facts(job, proposal, signal)
        if not isinstance(proposal.get("score_context"), dict):
            score_context = self._score_context_fact_card()
            if score_context:
                proposal["score_context"] = score_context
        if not isinstance(proposal.get("review_history"), dict):
            proposal.update(self._attach_review_history(proposal, now=time.time()))
        proposal = self._attach_budget_priority(proposal)
        proposal_id = str(proposal.get("proposal_id") or "")
        if not proposal_id:
            return
        trace_id = f"trace_{proposal_id}"
        self.resource_runtime.update_active_resource_proposal(proposal_id, proposal)
        self.resource_runtime.record_resource_event(
            "snapshot",
            payload={
                "resource_snapshot": proposal.get("resource_snapshot") or {},
                "progress_snapshot": proposal.get("progress_snapshot") or {},
                "blocker": proposal.get("blocker") or {},
                "waiters": proposal.get("waiters") or [],
                "execution_facts": proposal.get("execution_facts") or {},
                "research_cadence": proposal.get("research_cadence") or {},
            },
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        self.resource_runtime.record_resource_event(
            "resource_review_proposal",
            payload=proposal,
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        if proposal.get("requires_llm_decision"):
            self.resource_runtime.record_resource_event(
                "arbiter_input",
                payload={
                    "actor_id": "resource_arbiter",
                    "cache_namespace": "resource_arbiter",
                    "proposal_id": proposal_id,
                    "input_summary": {
                        "proposal_type": proposal.get("proposal_type"),
                        "reason": proposal.get("reason_code"),
                        "progress_confidence": (proposal.get("progress_snapshot") or {}).get("progress_confidence") if isinstance(proposal.get("progress_snapshot"), dict) else None,
                        "resource_confidence": (proposal.get("blocker") or {}).get("resource_confidence") if isinstance(proposal.get("blocker"), dict) else None,
                        "waiter_count": len(proposal.get("waiters") or []),
                        "intent_device_mismatch": (proposal.get("execution_facts") or {}).get("intent_device_mismatch") if isinstance(proposal.get("execution_facts"), dict) else None,
                        "observed_device": (proposal.get("execution_facts") or {}).get("observed_device") if isinstance(proposal.get("execution_facts"), dict) else None,
                    },
                },
                trace_id=trace_id,
                proposal_id=proposal_id,
                command_id=job.job_id,
                lease_id=job.job_id,
            )


    def _primary_kill_replan_candidate(self, job: ResourceJob, signal: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        share_blocking_reasons: list[str] = []
        progress = str(job.progress_signal or "unknown").strip().lower()
        progress_state = job.progress_signal_state if isinstance(job.progress_signal_state, dict) else {}
        windows = int(progress_state.get("progress_signal_windows") or job.progress_signal_windows or 0)
        if progress in {"stalled", "degraded"} and (windows >= self.arbiter_min_progress_windows or bool(progress_state.get("multi_window_low_progress"))):
            reason = f"progress_signal={progress}"
            reasons.append(reason)
            share_blocking_reasons.append(reason)
        if job.idle_gpu_lease_samples >= max(1, int(self.gpu_idle_lease_min_samples or 1)):
            reasons.append("idle_gpu_lease_samples_present")
            share_blocking_reasons.append("idle_gpu_lease_samples_present")
        if job.dataloader_bottleneck_samples >= max(1, int(self.gpu_dataloader_bottleneck_min_samples or 1)):
            progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=float(signal.get("elapsed_sec") or 0.0))
            active_work = active_work_counterevidence(progress_snapshot, thresholds=self._lease_suspect_thresholds())
            if active_work.get("active"):
                reasons.append("dataloader_bottleneck_samples_present")
            else:
                reasons.append("low_compute_no_active_work")
                share_blocking_reasons.append("low_compute_no_active_work")
        invalid_events = int(signal.get("invalid_metric_events") or 0)
        zero_events = int(signal.get("zero_score_events") or 0)
        if invalid_events >= self.metric_health_invalid_min_events:
            reasons.append("invalid_metric_events")
            share_blocking_reasons.append("invalid_metric_events")
        if zero_events >= self.metric_health_zero_score_min_events:
            reasons.append("zero_score_events")
            share_blocking_reasons.append("zero_score_events")
        share_blocking_set = set(share_blocking_reasons)
        return {
            "candidate": bool(reasons),
            "share_blocking_candidate": bool(share_blocking_reasons),
            "reasons": reasons,
            "share_blocking_reasons": share_blocking_reasons,
            "review_only_reasons": [reason for reason in reasons if reason not in share_blocking_set],
            "source": "v5_kill_efficiency_pipeline",
            "progress_signal": progress,
            "progress_signal_windows": windows,
            "idle_gpu_lease_samples": int(job.idle_gpu_lease_samples or 0),
            "dataloader_bottleneck_samples": int(job.dataloader_bottleneck_samples or 0),
            "invalid_metric_events": invalid_events,
            "zero_score_events": zero_events,
            "route_viability": dict(job.route_viability_state or {}),
        }


    def _record_gpu_share_event(self, event_type: str, job: ResourceJob, payload: dict[str, Any]) -> None:
        if self.resource_runtime is None:
            return
        waiter = payload.get("waiter") if isinstance(payload.get("waiter"), dict) else {}
        key = f"{event_type}:{job.job_id}:{waiter.get('job_id') or ''}"
        now = time.time()
        last = float(self._last_gpu_share_event_emit.get(key) or 0.0)
        cooldown = max(30.0, float(self.gpu_util_sample_interval_sec or 30.0))
        if last and now - last < cooldown:
            return
        self._last_gpu_share_event_emit[key] = now
        self.resource_runtime.record_resource_event(
            event_type,
            payload=payload,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
