# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Review cooldown, material history, confidence, and lease-suspect thresholds."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    LeaseSuspectThresholds,
    ResourceJob,
    hashlib,
    time,
)


class ReviewHistory:
    """Own ReviewHistory resource behavior without delegated forwarding."""

    @staticmethod
    def _waiter_gpu_ids(raw: dict[str, Any]) -> set[str]:
        return {str(x) for x in (raw.get("gpu_ids") or raw.get("candidate_gpu_ids") or []) if str(x).strip()}


    @staticmethod
    def _nested_float(raw: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(raw.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)
        return value if value == value else float(default)


    def _review_cooldown_active(self, job: ResourceJob, proposal_type: str, *, min_interval_sec: float, now: float) -> bool:
        key = f"{job.job_id}:{proposal_type}"
        observe_more_until = float(self._review_observe_more_until.get(key) or 0.0)
        if observe_more_until:
            if now < observe_more_until:
                return True
            self._review_observe_more_until.pop(key, None)
            return False
        job_last = float(self._last_review_proposal_emit.get(job.job_id) or 0.0)
        if job_last and self.arbiter_proposal_coalesce_window_sec > 0 and now - job_last < self.arbiter_proposal_coalesce_window_sec:
            return True
        last = float(self._last_review_proposal_emit.get(key) or 0.0)
        return bool(last and now - last < max(1.0, float(min_interval_sec or 1.0)))


    def _mark_review_proposal_emitted(self, job: ResourceJob, proposal_type: str, *, now: float) -> None:
        self._last_review_proposal_emit[job.job_id] = now
        self._last_review_proposal_emit[f"{job.job_id}:{proposal_type}"] = now


    def _review_history_key(self, job_id: str, proposal_type: str) -> str:
        return f"{job_id}:{proposal_type}"


    def _review_material_signature(self, proposal: dict[str, Any]) -> str:
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
        suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
        active_work = suspect.get("active_work_counterevidence") if isinstance(suspect.get("active_work_counterevidence"), dict) else {}
        route = progress.get("route_viability") if isinstance(progress.get("route_viability"), dict) else blocker.get("route_viability") if isinstance(blocker.get("route_viability"), dict) else {}
        cadence = proposal.get("research_cadence") if isinstance(proposal.get("research_cadence"), dict) else {}
        material = {
            "proposal_type": str(proposal.get("proposal_type") or ""),
            "reason_code": str(proposal.get("reason_code") or ""),
            "progress_signal": str(progress.get("progress_signal") or blocker.get("progress_signal") or ""),
            "progress_confidence": str(progress.get("progress_confidence") or blocker.get("progress_confidence") or ""),
            "route_viability_state": str(route.get("state") or ""),
            "resource_suspect": bool(suspect.get("resource_suspect")),
            "unknown_progress_suspect": bool(suspect.get("unknown_progress_suspect")),
            "value_suspect": bool(suspect.get("value_suspect")),
            "recent_useful_output": bool(suspect.get("recent_useful_output")),
            "active_work": bool(active_work.get("active")),
            "deadline_event": bool(progress.get("deadline_event")),
            "research_cadence_state": str(cadence.get("state") or ""),
            "route_metric_proven": bool(cadence.get("route_metric_proven")),
            "metric_budget_kind": str(cadence.get("metric_budget_kind") or ""),
        }
        return hashlib.sha1(repr(sorted(material.items())).encode("utf-8", errors="replace")).hexdigest()[:16]


    def _attach_review_history(self, proposal: dict[str, Any], *, now: float) -> dict[str, Any]:
        out = dict(proposal or {})
        job_id = str(out.get("command_id") or ((out.get("decision_preview") or {}).get("job_id") if isinstance(out.get("decision_preview"), dict) else "") or "")
        proposal_type = str(out.get("proposal_type") or "")
        if not job_id or not proposal_type:
            return out
        key = self._review_history_key(job_id, proposal_type)
        hist = dict(self._review_history.get(key) or {})
        signature = self._review_material_signature(out)
        unchanged = bool(hist.get("last_bad_signature") == signature and int(hist.get("observe_count") or 0) > 0)
        unchanged_windows = int(hist.get("unchanged_bad_fact_windows") or 0) if unchanged else 0
        repeated = bool(unchanged and int(hist.get("observe_count") or 0) >= 2 and unchanged_windows >= 2)
        waiters = out.get("waiters") if isinstance(out.get("waiters"), list) else []
        progress = out.get("progress_snapshot") if isinstance(out.get("progress_snapshot"), dict) else {}
        advisory = out.get("main_agent_advisory") if isinstance(out.get("main_agent_advisory"), dict) else {}
        advisory_preference = str(advisory.get("preference") or "").strip().lower()
        advisory_confidence = str(advisory.get("confidence") or "").strip().lower()
        advisory_safe_to_stop = advisory_preference in {"safe_to_stop", "kill_and_replan", "stop", "stop_and_replan"}
        progress_signal = str(progress.get("progress_signal") or "").strip().lower()
        reason_code = str(out.get("reason_code") or "").strip().lower()
        low_value_review = (
            str(out.get("proposal_type") or "").strip().lower() == "kill_proposal"
            and (
                progress_signal in {"stalled", "degraded", "unknown"}
                or "stalled" in reason_code
                or "low_progress" in reason_code
                or "dataloader" in reason_code
            )
        )
        advisory_opportunity_support = bool(
            repeated
            and low_value_review
            and advisory_safe_to_stop
            and advisory_confidence in {"low", "medium", "high"}
        )
        structured_support = bool(repeated and (waiters or progress.get("deadline_event") or advisory_opportunity_support))
        review_history = {
            "observe_count": int(hist.get("observe_count") or 0),
            "first_observe_at": hist.get("first_observe_at"),
            "last_action": str(hist.get("last_action") or ""),
            "last_decision_at": hist.get("last_decision_at"),
            "first_suspect_at": hist.get("first_suspect_at"),
            "unchanged_bad_fact_windows": unchanged_windows,
            "repeated_observe_support": repeated,
            "structured_opportunity_cost_support": structured_support,
            "advisory_opportunity_support": advisory_opportunity_support,
            "advisory_preference": advisory_preference,
            "advisory_confidence": advisory_confidence,
            "current_bad_signature": signature,
        }
        out["review_history"] = review_history
        if structured_support:
            out["structured_opportunity_cost_support"] = True
            if self.resource_runtime is not None:
                self.resource_runtime.record_resource_event(
                    "resource_review_observe_more_escalated",
                    payload={
                        "proposal_id": out.get("proposal_id"),
                        "proposal_type": proposal_type,
                        "job_id": job_id,
                        "review_history": review_history,
                    },
                    proposal_id=str(out.get("proposal_id") or ""),
                    command_id=job_id,
                    lease_id=job_id,
                )
        return out


    def _note_review_decision_history(self, proposal: dict[str, Any], decision: dict[str, Any]) -> None:
        job_id = str(proposal.get("command_id") or ((proposal.get("decision_preview") or {}).get("job_id") if isinstance(proposal.get("decision_preview"), dict) else "") or "")
        proposal_type = str(proposal.get("proposal_type") or "")
        if not job_id or not proposal_type:
            return
        key = self._review_history_key(job_id, proposal_type)
        now = time.time()
        action = str(decision.get("action") or "").upper()
        previous = dict(self._review_history.get(key) or {})
        signature = self._review_material_signature(proposal)
        is_observe = action in {"OBSERVE_MORE", "MARK_STALLED_NO_KILL"}
        if is_observe:
            same = previous.get("last_bad_signature") == signature
            observe_count = int(previous.get("observe_count") or 0) + 1
            unchanged_windows = int(previous.get("unchanged_bad_fact_windows") or 0) + 1 if same else 1
            first_observe_at = previous.get("first_observe_at") or now
            first_suspect_at = previous.get("first_suspect_at")
            blocker = proposal.get("blocker") if isinstance(proposal.get("blocker"), dict) else {}
            suspect = blocker.get("active_lease_suspect") if isinstance(blocker.get("active_lease_suspect"), dict) else {}
            if (suspect.get("resource_suspect") or suspect.get("unknown_progress_suspect") or suspect.get("value_suspect")) and not first_suspect_at:
                first_suspect_at = now
            self._review_history[key] = {
                "observe_count": observe_count,
                "first_observe_at": first_observe_at,
                "last_action": action,
                "last_decision_at": now,
                "first_suspect_at": first_suspect_at,
                "last_bad_signature": signature,
                "unchanged_bad_fact_windows": unchanged_windows,
            }
            return
        self._review_history[key] = {
            "observe_count": 0,
            "first_observe_at": None,
            "last_action": action,
            "last_decision_at": now,
            "first_suspect_at": previous.get("first_suspect_at"),
            "last_bad_signature": signature,
            "unchanged_bad_fact_windows": 0,
        }


    def _resource_confidence_for_review(self, job: ResourceJob, *, has_active_lease: bool, waiter_count: int = 0) -> str:
        if has_active_lease and waiter_count > 0 and job.last_gpu_util_sample:
            return "high"
        if has_active_lease and waiter_count > 0:
            return "medium"
        if has_active_lease:
            return "medium"
        return "low"


    def _lease_suspect_thresholds(self) -> LeaseSuspectThresholds:
        return LeaseSuspectThresholds(
            warmup_sec=max(0.0, float(self.gpu_idle_lease_warmup_sec or 0.0)),
            min_progress_windows=max(1, int(self.arbiter_min_progress_windows or 1)),
            low_gpu_util_pct=max(0.0, float(self.gpu_idle_lease_util_pct or 0.0)),
            min_gpu_mem_gb=max(0.0, float(self.gpu_idle_lease_mem_gb or 0.0)),
            route_viability_window_sec=max(
                float(self.low_progress_warmup_sec or 0.0),
                float(self.arbiter_periodic_min_runtime_sec or 0.0),
                float(self.arbiter_contention_min_runtime_sec or 0.0),
            ),
            unknown_progress_grace_sec=max(
                float(self.gpu_idle_lease_warmup_sec or 0.0),
                float(self.low_progress_warmup_sec or 0.0),
                float(self.arbiter_contention_min_runtime_sec or 0.0),
            ),
            unknown_progress_stale_output_sec=max(
                float(self.low_progress_no_heartbeat_sec or 0.0),
                float(self.low_progress_no_artifact_sec or 0.0),
                float(self.stalled_stdout_sec or 0.0),
                float(self.check_interval_sec or 1.0) * float(self.arbiter_min_progress_windows or 1),
            ),
        )
