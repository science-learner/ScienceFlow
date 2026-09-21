# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Proof-debt escalation, advisory commitments, and LLM review budgets."""

from __future__ import annotations

from scienceflow.research.solver.lnr.resources.runtime.observer.shared import (
    Any,
    PROGRESS_WINDOW,
    RESOURCE_REVIEW_KILL,
    RESOURCE_REVIEW_NO_ACTION,
    RESOURCE_REVIEW_TIMEBOX,
    ROUTE_VALUE,
    TERMINATING_ACTIONS,
    TIMEBOX_EXPIRED,
    hashlib,
    json,
    normalize_clear_on,
)


class ReviewBudget:
    """Own ReviewBudget resource behavior without delegated forwarding."""

    def _llm_budget_defer_reason(self, proposal: dict[str, Any], job_key: str, *, advisory: bool) -> str:
        if not job_key:
            return ""
        budget = proposal.get("budget_priority") if isinstance(proposal.get("budget_priority"), dict) else {}
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        efficiency = progress.get("resource_efficiency") if isinstance(progress.get("resource_efficiency"), dict) else {}
        if efficiency.get("review_required") and not efficiency.get("review_limit_reached"):
            return ""
        if self._proof_debt_review_has_priority(proposal, job_key):
            return ""
        metric_history_digest = self._proposal_metric_history_digest(proposal)
        last_metric_history = self._job_last_llm_metric_history_digest.get(job_key, "")
        if not advisory and metric_history_digest and metric_history_digest != last_metric_history:
            return ""
        if str(budget.get("level") or "").lower() != "low":
            return ""
        prior_calls = int(self._job_llm_call_count.get(job_key) or 0)
        prior_advisory = int(self._job_advisory_call_count.get(job_key) or 0)
        if advisory and prior_advisory <= 0:
            return ""
        if not advisory and prior_calls <= 0:
            return ""
        return "low_priority_unchanged_or_low_opportunity"


    @staticmethod
    def _value_review_proposal_type(proposal: dict[str, Any]) -> bool:
        proposal_type = str(proposal.get("proposal_type") or "").strip().lower()
        return proposal_type in {
            "kill_proposal",
            "periodic_efficiency_review",
            "quick_probe_review",
            "resource_contention_review",
        }


    @staticmethod
    def _proposal_boundary_kind(proposal: dict[str, Any]) -> str:
        boundary = proposal.get("resource_review_boundary")
        if not isinstance(boundary, dict):
            preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
            boundary = preview.get("resource_review_boundary") if isinstance(preview.get("resource_review_boundary"), dict) else {}
        return str(boundary.get("kind") or "").strip().lower()


    @staticmethod
    def _advisory_commitment_clear_on(advisory: dict[str, Any]) -> str:
        preference = str(advisory.get("preference") or "").strip().lower()
        if preference not in {"continue", "timebox_continue"}:
            return ""
        commitment = str(advisory.get("commitment") or "").strip()
        expected = str(advisory.get("expected_next_artifact") or "").strip()
        if not commitment and not expected:
            return ""
        text = f"{commitment} {expected}".lower()
        metric_tokens = (
            "metric",
            "score",
            "auc",
            "rmse",
            "rmsle",
            "map",
            "f1",
            "validation",
            "valid",
            "submission",
            "submit",
            "held-out",
            "heldout",
        )
        if any(token in text for token in metric_tokens):
            return "metric_update"
        artifact_tokens = ("artifact", "checkpoint", "submission", "log", "file", "path", "csv", "npy", "pth", "pkl")
        if expected or any(token in text for token in artifact_tokens):
            return "artifact_growth"
        return ""


    def _advisory_commitment_timebox_decision(self, proposal: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        if not self._value_review_proposal_type(proposal):
            return decision
        escalation = proposal.get("resource_budget_escalation") if isinstance(proposal.get("resource_budget_escalation"), dict) else {}
        if str(escalation.get("kind") or "").strip().lower() == "final_resource_review":
            return decision
        advisory = proposal.get("main_agent_advisory") if isinstance(proposal.get("main_agent_advisory"), dict) else {}
        clear_on = self._advisory_commitment_clear_on(advisory)
        if not clear_on:
            return decision
        job_key = self._proposal_job_id(proposal)
        current_state = self._review_states.get(job_key) if job_key else None
        if (
            current_state is not None
            and current_state.in_timebox
            and normalize_clear_on(current_state.timebox_success_condition) == clear_on
        ):
            out = dict(decision)
            original_reason = str(out.get("reason") or "").strip()
            out.update({
                "action": "OBSERVE_MORE",
                "canonical_outcome": RESOURCE_REVIEW_NO_ACTION,
                "confidence": str(out.get("confidence") or "medium").strip().lower() or "medium",
                "reason_code": "advisory_timebox_already_active",
                "reason": (
                    "A negotiated resource timebox is already active for the same clear condition; "
                    "do not restart the proof window before it clears or expires."
                    + (f" Original arbiter reason: {original_reason}" if original_reason else "")
                ),
                "advisory_commitment_timebox_active": True,
                "active_timebox_id": current_state.timebox_id,
                "active_timebox_clear_on": clear_on,
                "active_timebox_windows": int(current_state.timebox_windows or 0),
                "active_timebox_deadline_windows": int(current_state.timebox_deadline_windows or 0),
            })
            out.pop("clear_on", None)
            return out
        action = str(decision.get("action") or "").strip().upper()
        canonical = str(decision.get("canonical_outcome") or "").strip().upper()
        if action in TERMINATING_ACTIONS or canonical == RESOURCE_REVIEW_KILL:
            return decision
        confidence = str(advisory.get("confidence") or decision.get("confidence") or "medium").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        original_reason = str(decision.get("reason") or "").strip()
        out = dict(decision)
        out.update({
            "action": "OBSERVE_MORE",
            "canonical_outcome": RESOURCE_REVIEW_TIMEBOX,
            "clear_on": clear_on,
            "confidence": confidence,
            "reason_code": "main_agent_advisory_commitment_timebox",
            "reason": (
                "Main-agent advisory requested continued execution with a measurable commitment; "
                "start a negotiated timebox instead of plain defer."
                + (f" Original arbiter reason: {original_reason}" if original_reason else "")
            ),
            "advisory_commitment_timebox": True,
            "advisory_commitment": str(advisory.get("commitment") or "")[:300],
            "advisory_expected_next_artifact": str(advisory.get("expected_next_artifact") or "")[:300],
        })
        return out


    def _final_resource_review_escalation(self, proposal: dict[str, Any], job_key: str) -> dict[str, Any]:
        if not (job_key and self._value_review_proposal_type(proposal)):
            return {}
        state = self._review_states.get(job_key)
        review_state = state.to_json() if state is not None else {}
        if not review_state:
            preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
            review_state = preview.get("resource_review_state") if isinstance(preview.get("resource_review_state"), dict) else {}
        failed_count = int(review_state.get("failed_proof_window_count") or 0)
        max_windows = max(1, int(self.review_config.max_proof_windows or 2))
        if failed_count < max_windows:
            return {}
        boundary_kind = self._proposal_boundary_kind(proposal)
        failure_recorded = bool(review_state.get("timebox_failure_recorded"))
        if boundary_kind != TIMEBOX_EXPIRED and not failure_recorded:
            return {}
        return {
            "kind": "final_resource_review",
            "advisory_status": "exhausted",
            "failed_proof_window_count": failed_count,
            "max_proof_windows": max_windows,
            "proof_window_index": int(review_state.get("proof_window_index") or 0),
            "proof_window_source": str(review_state.get("proof_window_source") or ""),
            "timebox_success_condition": str(review_state.get("timebox_success_condition") or ""),
            "reason": "proof windows exhausted; resource arbiter must make the final resource-budget decision without another main-agent advisory",
        }


    def _with_final_resource_review_escalation(self, proposal: dict[str, Any], escalation: dict[str, Any]) -> dict[str, Any]:
        if not escalation:
            return proposal
        out = dict(proposal or {})
        out["resource_budget_escalation"] = dict(escalation)
        out["main_agent_advisory_status"] = "exhausted"
        out.setdefault("suggested_actions", ["DENY_KILL", "OBSERVE_MORE", "KILL_AND_REPLAN"])
        proposal_id = str(out.get("proposal_id") or "")
        if self.resource_runtime is not None and proposal_id:
            self.resource_runtime.update_active_resource_proposal(proposal_id, out)
            self.resource_runtime.record_resource_event(
                "resource_budget_escalation",
                payload={"proposal_id": proposal_id, **dict(escalation)},
                trace_id=f"trace_{proposal_id}",
                proposal_id=proposal_id,
                command_id=self._proposal_job_id(out),
                lease_id=self._proposal_job_id(out),
            )
        return out


    def _proof_window_override_raw_decision(self, proposal: dict[str, Any], job_key: str) -> dict[str, Any] | None:
        if not (job_key and self._value_review_proposal_type(proposal)):
            return None
        boundary_kind = self._proposal_boundary_kind(proposal)
        state = self._review_states.get(job_key)
        review_state = state.to_json() if state is not None else {}
        if not review_state:
            preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
            review_state = preview.get("resource_review_state") if isinstance(preview.get("resource_review_state"), dict) else {}
        failed_count = int(review_state.get("failed_proof_window_count") or 0)
        proof_source = str(review_state.get("proof_window_source") or "").strip().lower()
        failure_recorded = bool(review_state.get("timebox_failure_recorded"))
        max_windows = max(1, int(self.review_config.max_proof_windows or 2))
        if failed_count >= max_windows:
            return None
        if boundary_kind != TIMEBOX_EXPIRED or not failure_recorded or failed_count <= 0:
            return None
        clear_on = normalize_clear_on(review_state.get("timebox_success_condition")) or "metric_update"
        return {
            "outcome": RESOURCE_REVIEW_TIMEBOX,
            "reason_code": "proof_window_retry",
            "reason": (
                f"The command missed proof window {failed_count}/{max_windows}; "
                "start the next bounded proof window before final resource review."
            ),
            "confidence": "medium",
            "clear_on": clear_on,
            "proof_window_override": True,
            "proof_window_retry": True,
            "proof_window_source": "proof_retry",
            "failed_proof_window_count": failed_count,
            "proof_window_source_previous": proof_source,
            "max_proof_windows": max_windows,
        }


    def _proof_debt_review_has_priority(self, proposal: dict[str, Any], job_key: str) -> bool:
        if not (job_key and self._value_review_proposal_type(proposal)):
            return False
        state = self._review_states.get(job_key)
        if state is None or int(state.failed_proof_window_count or 0) <= 0:
            return False
        boundary = self._proposal_boundary_kind(proposal)
        return boundary in {PROGRESS_WINDOW, ROUTE_VALUE}


    def _record_llm_budget_deferred_event(self, proposal: dict[str, Any], reason: str, *, advisory: bool = False) -> None:
        if self.resource_runtime is None:
            return
        proposal_id = str(proposal.get("proposal_id") or "")
        job_key = self._proposal_job_id(proposal)
        budget = proposal.get("budget_priority") if isinstance(proposal.get("budget_priority"), dict) else {}
        self.resource_runtime.record_resource_event(
            "llm_budget_deferred",
            payload={
                "proposal_id": proposal_id,
                "job_id": job_key,
                "advisory": bool(advisory),
                "reason": reason or "llm_budget_deferred",
                "budget_priority": budget,
                "llm_call_count": int(self._job_llm_call_count.get(job_key) or 0),
                "advisory_call_count": int(self._job_advisory_call_count.get(job_key) or 0),
                "llm_token_count": int(self._job_llm_token_count.get(job_key) or 0),
            },
            trace_id=f"trace_{proposal_id}" if proposal_id else "",
            proposal_id=proposal_id,
            command_id=job_key,
            lease_id=job_key,
        )


    @staticmethod
    def _proposal_job_id(proposal: dict[str, Any], fallback: str | None = None) -> str:
        preview = proposal.get("decision_preview") if isinstance(proposal.get("decision_preview"), dict) else {}
        return str(proposal.get("command_id") or proposal.get("job_id") or preview.get("job_id") or fallback or "")


    @staticmethod
    def _estimate_llm_tokens(payload: dict[str, Any]) -> int:
        try:
            text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except Exception:
            text = str(payload or {})
        return max(1, int(len(text) / 4) + 1)


    def _llm_budget_blocked(self, job_key: str, *, estimated_tokens: int = 0, advisory: bool = False) -> dict[str, Any]:
        if not job_key:
            return {"blocked": False}
        llm_calls = int(self._job_llm_call_count.get(job_key) or 0)
        advisory_calls = int(self._job_advisory_call_count.get(job_key) or 0)
        tokens = int(self._job_llm_token_count.get(job_key) or 0)
        if self.arbiter_job_llm_call_cap > 0 and llm_calls >= self.arbiter_job_llm_call_cap:
            return {"blocked": True, "reason": "resource_arbiter_job_llm_call_cap", "llm_call_count": llm_calls}
        if advisory and self.arbiter_job_advisory_call_cap > 0 and advisory_calls >= self.arbiter_job_advisory_call_cap:
            return {"blocked": True, "reason": "resource_arbiter_job_advisory_call_cap", "advisory_call_count": advisory_calls}
        if self.arbiter_job_token_cap > 0 and tokens + max(0, int(estimated_tokens or 0)) > self.arbiter_job_token_cap:
            return {"blocked": True, "reason": "resource_arbiter_job_token_cap", "llm_token_count": tokens}
        return {"blocked": False}


    @staticmethod
    def _proposal_metric_history_digest(proposal: dict[str, Any]) -> str:
        generation = proposal.get("state_generation") if isinstance(proposal.get("state_generation"), dict) else {}
        control_fields = generation.get("control_fields") if isinstance(generation.get("control_fields"), dict) else {}
        digest = str(control_fields.get("metric_history_digest") or "").strip()
        if digest:
            return digest
        progress = proposal.get("progress_snapshot") if isinstance(proposal.get("progress_snapshot"), dict) else {}
        metric_history = str(progress.get("metric_history_text") or "").strip()
        if not metric_history:
            return ""
        return hashlib.sha1(metric_history.encode("utf-8", errors="replace")).hexdigest()[:16]


    def _note_llm_call(
        self,
        job_key: str,
        *,
        estimated_tokens: int = 0,
        advisory: bool = False,
        proposal: dict[str, Any] | None = None,
    ) -> None:
        if not job_key:
            return
        self._job_llm_call_count[job_key] = int(self._job_llm_call_count.get(job_key) or 0) + 1
        if advisory:
            self._job_advisory_call_count[job_key] = int(self._job_advisory_call_count.get(job_key) or 0) + 1
        self._job_llm_token_count[job_key] = int(self._job_llm_token_count.get(job_key) or 0) + max(0, int(estimated_tokens or 0))
        metric_history_digest = self._proposal_metric_history_digest(proposal or {})
        if metric_history_digest and not advisory:
            self._job_last_llm_metric_history_digest[job_key] = metric_history_digest


    def _record_llm_budget_event(self, proposal: dict[str, Any], block: dict[str, Any], *, advisory: bool = False) -> None:
        if self.resource_runtime is None:
            return
        proposal_id = str(proposal.get("proposal_id") or "")
        job_key = self._proposal_job_id(proposal)
        self.resource_runtime.record_resource_event(
            "llm_budget_exhausted",
            payload={
                "proposal_id": proposal_id,
                "job_id": job_key,
                "advisory": bool(advisory),
                "reason": block.get("reason") or "llm_budget_exhausted",
                "llm_call_count": int(self._job_llm_call_count.get(job_key) or 0),
                "advisory_call_count": int(self._job_advisory_call_count.get(job_key) or 0),
                "llm_token_count": int(self._job_llm_token_count.get(job_key) or 0),
                "llm_call_cap": self.arbiter_job_llm_call_cap,
                "advisory_call_cap": self.arbiter_job_advisory_call_cap,
                "token_cap": self.arbiter_job_token_cap,
            },
            trace_id=f"trace_{proposal_id}" if proposal_id else "",
            proposal_id=proposal_id,
            command_id=job_key,
            lease_id=job_key,
        )
