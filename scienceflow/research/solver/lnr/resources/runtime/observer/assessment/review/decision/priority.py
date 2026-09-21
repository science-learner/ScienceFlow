# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Kill-boundary classification and waiter-aware proposal priority."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    PROGRESS_WINDOW,
    RESOURCE_PRESSURE,
    ROUTE_VALUE,
    ResourceJob,
    STALL,
    TIMEBOX_EXPIRED,
    build_resource_state_generation,
    hashlib,
    json,
    time,
)


class ReviewPriorityCallbacks:
    """Own ReviewPriorityCallbacks resource behavior without delegated forwarding."""

    @staticmethod
    def _int_bucket(value: Any, *, threshold: int) -> tuple[str, int]:
        try:
            number = int(value or 0)
        except (TypeError, ValueError):
            number = 0
        threshold = max(1, int(threshold or 1))
        if number <= 0:
            return "zero", 0
        if number < threshold:
            return "below_threshold", 1
        if number < threshold * 2:
            return "at_threshold", 2
        return "doubled_threshold", 3


    @staticmethod
    def _boundary_severity_rank(kind: str) -> int:
        value = str(kind or "").strip()
        if value in {STALL, TIMEBOX_EXPIRED}:
            return 4
        if value == RESOURCE_PRESSURE:
            return 3
        if value == ROUTE_VALUE:
            return 2
        if value == PROGRESS_WINDOW:
            return 1
        return 0


    def _kill_proposal_boundary_state(self, decision: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
        boundary = decision.get("resource_review_boundary") if isinstance(decision.get("resource_review_boundary"), dict) else {}
        if not boundary:
            boundary = signal.get("resource_review_boundary") if isinstance(signal.get("resource_review_boundary"), dict) else {}
        review_state = decision.get("resource_review_state") if isinstance(decision.get("resource_review_state"), dict) else {}
        if not review_state:
            review_state = signal.get("resource_review_state") if isinstance(signal.get("resource_review_state"), dict) else {}
        contention = decision.get("contention_context") if isinstance(decision.get("contention_context"), dict) else {}
        if not contention:
            contention = signal.get("contention_context") if isinstance(signal.get("contention_context"), dict) else {}
        no_useful = int(review_state.get("no_useful_progress_windows") or 0)
        no_useful_bucket, no_useful_rank = self._int_bucket(no_useful, threshold=int(self.review_config.value_windows or 1))
        blocked_workers = int(contention.get("blocked_worker_count") or review_state.get("blocked_worker_count") or 0)
        waiter_pressure = bool(contention.get("active_waiter_pressure") or review_state.get("active_waiter_pressure") or blocked_workers > 0)
        kind = str(boundary.get("kind") or "")
        return {
            "boundary_kind": kind,
            "boundary_reason": str(boundary.get("reason") or ""),
            "boundary_severity_rank": self._boundary_severity_rank(kind),
            "job_state_bucket": str(review_state.get("job_state_bucket") or ""),
            "no_useful_progress_windows": no_useful,
            "no_useful_progress_bucket": no_useful_bucket,
            "no_useful_progress_rank": no_useful_rank,
            "timebox_id": str(review_state.get("timebox_id") or ""),
            "timebox_windows": int(review_state.get("timebox_windows") or 0),
            "active_waiter_pressure": waiter_pressure,
            "blocked_worker_count": blocked_workers,
        }


    @staticmethod
    def _kill_proposal_boundary_escalated(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
        if not previous:
            return True
        if int(current.get("boundary_severity_rank") or 0) > int(previous.get("boundary_severity_rank") or 0):
            return True
        if (
            int(current.get("no_useful_progress_windows") or 0) > int(previous.get("no_useful_progress_windows") or 0)
            and int(current.get("no_useful_progress_rank") or 0) >= 2
        ):
            return True
        if str(current.get("boundary_kind") or "") == TIMEBOX_EXPIRED:
            return True
        if bool(current.get("active_waiter_pressure")) and not bool(previous.get("active_waiter_pressure")):
            return True
        if int(current.get("blocked_worker_count") or 0) > int(previous.get("blocked_worker_count") or 0):
            return True
        return False


    def _record_kill_proposal_event(
        self,
        job: ResourceJob,
        decision: dict[str, Any],
        signal: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any] | None:
        if self.resource_runtime is None or not decision.get("would_terminate"):
            return None
        now = time.time()
        reason = str(decision.get("reason") or "resource_guard")
        elapsed = float(signal.get("elapsed_sec") or 0.0)
        resource_snapshot = self._resource_snapshot_for_job(job, signal)
        progress_snapshot = self._progress_snapshot_for_job(job, signal, elapsed_sec=elapsed)
        execution_facts = self._execution_facts_for_job(
            job,
            resource_snapshot=resource_snapshot,
            progress_snapshot=progress_snapshot,
        )
        state_generation = build_resource_state_generation({
            "proposal_type": "kill_proposal",
            "reason_code": reason,
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "gpu_ids": list(job.gpu_ids or []),
            "execution_facts": execution_facts,
        })
        dedupe_key = f"{job.job_id}:{reason}:{state_generation.control_generation_key}"
        boundary_state = self._kill_proposal_boundary_state(decision, signal)
        previous_boundary_state = self._last_kill_proposal_boundary_by_dedupe.get(dedupe_key)
        escalated_boundary = self._kill_proposal_boundary_escalated(previous_boundary_state, boundary_state)
        last = float(self._last_kill_proposal_emit.get(dedupe_key) or 0.0)
        cooldown = max(1.0, float(self.kill_proposal_cooldown_sec or 600.0))
        cooldown_active = bool(last and now - last < cooldown)
        if cooldown_active and not escalated_boundary:
            suppressed_last = float(self._last_suppressed_kill_proposal_emit.get(dedupe_key) or 0.0)
            if now - suppressed_last >= min(60.0, cooldown):
                self._last_suppressed_kill_proposal_emit[dedupe_key] = now
                self.resource_runtime.record_resource_event(
                    "suppressed_proposal",
                    payload={
                        "reason": reason,
                        "dedupe_key": dedupe_key,
                        "control_generation_key": state_generation.control_generation_key,
                        "feedback_generation_key": state_generation.feedback_generation_key,
                        "state_generation": state_generation.to_json(),
                        "boundary_state": boundary_state,
                        "previous_boundary_state": previous_boundary_state or {},
                        "suppressed": True,
                        "suppressed_reason": "cooldown_same_unchanged_boundary",
                        "legacy_suppressed_reason": "proposal_cooldown_same_control_state",
                        "suppressed_by_proposal_id": self._last_kill_proposal_id_by_dedupe.get(dedupe_key, ""),
                        "would_trigger_arbiter": False,
                        "previous_same_kind_age_sec": now - last,
                        "cooldown_sec": cooldown,
                    },
                    command_id=job.job_id,
                    lease_id=job.job_id,
                )
            return None
        self._last_kill_proposal_emit[dedupe_key] = now
        self._last_kill_proposal_boundary_by_dedupe[dedupe_key] = dict(boundary_state)
        boundary_key = json.dumps(boundary_state, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
        proposal_id = f"rp_{job.job_id}_{hashlib.sha1((reason + state_generation.control_generation_key + boundary_key).encode('utf-8', errors='replace')).hexdigest()[:8]}"
        self._last_kill_proposal_id_by_dedupe[dedupe_key] = proposal_id
        trace_id = f"trace_{proposal_id}"
        decision["proposal_id"] = proposal_id
        decision["job_id"] = job.job_id
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": "kill_proposal",
            "severity": "red" if decision.get("terminate") else "yellow",
            "reason_code": reason,
            "dedupe_key": dedupe_key,
            "control_generation_key": state_generation.control_generation_key,
            "feedback_generation_key": state_generation.feedback_generation_key,
            "state_generation": state_generation.to_json(),
            "boundary_state": boundary_state,
            "previous_boundary_state": previous_boundary_state or {},
            "cooldown_bypassed_by_escalation": bool(cooldown_active and escalated_boundary),
            "proposal_cooldown_sec": cooldown,
            "suppressed": False,
            "suppressed_reason": "",
            "requires_llm_decision": bool(decision.get("requires_llm_decision")),
            "suggested_actions": ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"],
            "pending_recommendations": ["release_unused_lease"] if "idle_gpu_lease" in reason else [],
            "resource_snapshot": resource_snapshot,
            "progress_snapshot": progress_snapshot,
            "execution_facts": execution_facts,
            "resource_metric_value": dict(
                signal.get("resource_metric_value")
                or decision.get("resource_metric_value")
                or {}
            ),
            "decision_preview": dict(decision),
            "source": str(source or "resource_guard"),
            "command_id": job.job_id,
        }
        proposal = self._attach_budget_priority(proposal)
        self.resource_runtime.update_active_resource_proposal(proposal_id, proposal)
        self.resource_runtime.record_resource_event(
            "snapshot",
            payload={
                "resource_snapshot": proposal["resource_snapshot"],
                "progress_snapshot": proposal["progress_snapshot"],
                "execution_facts": proposal.get("execution_facts") or {},
            },
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        self.resource_runtime.record_resource_event(
            "kill_proposal",
            payload=proposal,
            trace_id=trace_id,
            proposal_id=proposal_id,
            command_id=job.job_id,
            lease_id=job.job_id,
        )
        if decision.get("requires_llm_decision"):
            self.resource_runtime.record_resource_event(
                "arbiter_input",
                payload={
                    "actor_id": "resource_arbiter",
                    "cache_namespace": "resource_arbiter",
                    "proposal_id": proposal_id,
                    "input_summary": {
                        "reason": reason,
                        "progress_confidence": proposal["progress_snapshot"].get("progress_confidence"),
                        "runtime_sec": elapsed,
                        "intent_device_mismatch": execution_facts.get("intent_device_mismatch"),
                        "observed_device": execution_facts.get("observed_device"),
                    },
                },
                trace_id=trace_id,
                proposal_id=proposal_id,
                command_id=job.job_id,
                lease_id=job.job_id,
            )
        decision["proposal"] = proposal
        return proposal


    @staticmethod
    def _budget_waiter_cards(proposal: dict[str, Any]) -> list[dict[str, Any]]:
        waiters = proposal.get("waiters")
        if isinstance(waiters, list):
            return [row for row in waiters if isinstance(row, dict)]
        candidate = proposal.get("candidate")
        if isinstance(candidate, dict) and candidate:
            return [candidate]
        return []


    def _budget_priority_for_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
        suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
        waiters = self._budget_waiter_cards(proposal)
        suggested_actions = {str(x).upper() for x in (proposal.get("suggested_actions") or [])}
        reason_code = str(proposal.get("reason_code") or "").strip().lower()
        proposal_type = str(proposal.get("proposal_type") or "").strip().lower()
        progress_signal = str(progress.get("progress_signal") or blocker.get("progress_signal") or "").strip().lower()
        progress_confidence = str(progress.get("progress_confidence") or blocker.get("progress_confidence") or "").strip().lower()
        efficiency = progress.get("resource_efficiency") if isinstance(progress.get("resource_efficiency"), dict) else {}
        queue_ages: list[float] = []
        for row in waiters:
            try:
                queue_ages.append(max(0.0, float(row.get("queue_age_sec") or 0.0)))
            except (TypeError, ValueError):
                continue
        max_queue_age = max(queue_ages) if queue_ages else 0.0
        score = 0
        reasons: list[str] = []
        job_key = self._proposal_job_id(proposal)
        metric_history_digest = self._proposal_metric_history_digest(proposal)
        metric_history_changed = bool(
            job_key
            and metric_history_digest
            and metric_history_digest != self._job_last_llm_metric_history_digest.get(job_key, "")
        )
        score += self._add_budget_signal(reasons, active=metric_history_changed, points=0, reason="new_metric_history_since_llm_review")
        score += self._add_budget_signal(reasons, active=bool(waiters), points=2, reason="waiter_blocked")
        score += self._add_budget_signal(reasons, active=max_queue_age >= max(1.0, float(self.arbiter_contention_min_waiter_age_sec or 0.0)), points=1, reason="waiter_age_exceeds_review_threshold")
        score += self._add_budget_signal(reasons, active=bool(progress.get("deadline_event")), points=2, reason="deadline_event")
        score += self._add_budget_signal(reasons, active=bool(suspect.get("resource_suspect")), points=2, reason="resource_suspect")
        score += self._add_budget_signal(reasons, active=bool(suspect.get("value_suspect")), points=2, reason="value_suspect")
        score += self._add_budget_signal(reasons, active=bool(suspect.get("unknown_progress_suspect")), points=1, reason="unknown_progress_suspect")
        progress_points, progress_reason = self._budget_progress_signal(progress_signal, progress_confidence)
        score += self._add_budget_signal(reasons, active=bool(progress_reason), points=progress_points, reason=progress_reason)
        reason_points, reason_signal = self._budget_reason_signal(reason_code)
        score += self._add_budget_signal(reasons, active=bool(reason_signal), points=reason_points, reason=reason_signal)
        score += self._add_budget_signal(reasons, active=proposal_type in {"resource_contention_review", "task_gpu_share_review", "shared_runtime_review"}, points=1, reason="resource_contention_or_share_review")
        actionable = bool(suggested_actions & {
            "KILL_AND_REPLAN",
            "GRANT_SHARED_GPU_LEASE",
            "STOP_SECONDARY_SHARED_JOB",
            "REVOKE_SHARED_LEASE",
        })
        score += self._add_budget_signal(reasons, active=actionable, points=1, reason="actionable_resource_decision")
        score += self._add_budget_signal(reasons, active=bool(efficiency.get("review_required") and efficiency.get("kill_authority") == "review_only"), points=3, reason="bounded_resource_efficiency_review")
        score += self._add_budget_signal(reasons, active=not waiters and progress_signal == "active" and progress_confidence == "high", points=-1, reason="active_high_confidence_no_waiter")
        level = self._budget_priority_level(score)
        if not reasons:
            reasons.append("no_opportunity_or_value_signal")
        return {
            "level": level,
            "score": score,
            "reasons": reasons[:8],
            "waiter_count": len(waiters),
            "max_queue_age_sec": max_queue_age,
            "progress_signal": progress_signal or "unknown",
            "progress_confidence": progress_confidence or "unknown",
            "source": "deterministic_budget_router",
        }

    @staticmethod
    def _add_budget_signal(reasons: list[str], *, active: bool, points: int, reason: str) -> int:
        if not active:
            return 0
        reasons.append(reason)
        return points

    @staticmethod
    def _budget_progress_signal(progress_signal: str, progress_confidence: str) -> tuple[int, str]:
        if progress_signal in {"stalled", "degraded"}:
            return 2, f"progress_{progress_signal}"
        if progress_signal == "unknown" and progress_confidence in {"low", "unknown", ""}:
            return 1, "progress_unknown_low_confidence"
        return 0, ""

    @staticmethod
    def _budget_reason_signal(reason_code: str) -> tuple[int, str]:
        if any(token in reason_code for token in ("stalled", "low_progress", "weak_metric", "no_viability", "invalid_metric")):
            return 2, "reason_indicates_low_value_or_stall"
        if "dataloader" in reason_code:
            return 1, "dataloader_bottleneck_review"
        return 0, ""

    @staticmethod
    def _budget_priority_level(score: int) -> str:
        if score >= 4:
            return "high"
        if score >= 2:
            return "medium"
        return "low"


    def _attach_budget_priority(self, proposal: dict[str, Any]) -> dict[str, Any]:
        out = dict(proposal or {})
        if not isinstance(out.get("budget_priority"), dict):
            out["budget_priority"] = self._budget_priority_for_proposal(out)
        return out
