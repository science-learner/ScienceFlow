# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: context hygiene and text-only/context-limit continuation.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    Message,
    evaluate_context_hygiene_compact,
    large_code_file_touch_counts,
    large_tool_output_count,
    logger,
    parse_estra_decision,
    re,
    time,
)


def _note_context_hygiene_stage_delta(self, agent: Any) -> dict[str, Any]:
    tokens_in, tokens_cached = self._current_main_token_totals(agent)
    delta_in = max(
        0,
        tokens_in - int(getattr(self, "context_hygiene_last_stage_tokens_in", 0) or 0),
    )
    delta_cached = max(
        0,
        tokens_cached
        - int(getattr(self, "context_hygiene_last_stage_tokens_cached", 0) or 0),
    )
    rate = None
    if delta_in > 0:
        rate = max(0.0, min(1.0, delta_cached / delta_in))
        rates = list(getattr(self, "context_hygiene_cache_rates", []) or [])
        rates.append(rate)
        self.context_hygiene_cache_rates = rates[-100:]
    self.context_hygiene_last_stage_tokens_in = tokens_in
    self.context_hygiene_last_stage_tokens_cached = tokens_cached
    return {
        "tokens_in": tokens_in,
        "tokens_cached": tokens_cached,
        "delta_input_tokens": delta_in,
        "delta_cached_tokens": delta_cached,
        "incremental_cache_rate": rate,
    }


def _context_hygiene_active_high_risk_bash(self) -> bool:
    observer = getattr(self, "resource_observer", None)
    runtime = getattr(observer, "resource_runtime", None)
    jobs = getattr(observer, "_jobs", {}) if observer is not None else {}
    if not isinstance(jobs, dict) or runtime is None:
        return False
    for job in jobs.values():
        try:
            if not bool(getattr(job, "visible", False)):
                continue
            if getattr(job, "terminal_signal_seen", False) or getattr(
                job, "exit_code_seen", False
            ):
                continue
            if runtime.has_active_lease(job_id=getattr(job, "job_id", "")):
                return True
        except Exception:
            continue
    return False


def _context_hygiene_snapshot_can_preserve_current_state(
    self, *, cards_after: list[Any]
) -> tuple[bool, dict[str, Any]]:
    latest = str(cards_after[-1].stage_id if cards_after else "").upper()
    snap = self.stage_snapshots.get(latest)
    event = dict(getattr(snap, "source_event", {}) or {}) if snap is not None else {}
    fields = {
        "latest_stage": latest,
        "has_snapshot": snap is not None,
        "has_metric": event.get("metric_value") is not None,
        "has_submission_path": bool(
            event.get("submission_snapshot")
            or event.get("submission_sha")
            or event.get("submission_status") == "missing_submission"
        ),
        "has_code_ref": bool(
            event.get("source_commit_sha")
            or event.get("solution_sha")
            or getattr(snap, "snapshot_path", None)
        ),
        "has_constraints": True,
        "has_pending_high_value_marker": True,
    }
    ok = bool(fields["has_snapshot"] and fields["has_code_ref"])
    return ok, fields


def _evaluate_context_hygiene_after_stage(
    self, *, agent: Any, cards_after: list[Any]
) -> tuple[Any, list[Any], str, int, int, int, dict[str, Any], bool]:
    token_delta = self._note_context_hygiene_stage_delta(agent)
    effective_cards_after = self._effective_stage_cards(cards_after)
    latest = str(
        effective_cards_after[-1].stage_id if effective_cards_after else ""
    ).upper()
    stage_count = len(effective_cards_after)
    stage_count_since = max(
        0,
        stage_count
        - int(getattr(self, "context_hygiene_last_compact_stage_count", 0) or 0),
    )
    tokens_in, _tokens_cached = self._current_main_token_totals(agent)
    tokens_since = max(
        0,
        tokens_in
        - int(getattr(self, "context_hygiene_last_compact_tokens_in", 0) or 0),
    )
    messages = self._agent_memory_messages(agent)
    file_counts = large_code_file_touch_counts(messages)
    tool_outputs = large_tool_output_count(
        messages,
        min_chars=int(
            getattr(self.lhr, "context_hygiene_large_tool_output_chars", 12000) or 12000
        ),
    )
    snapshot_ok, snapshot_fields = (
        self._context_hygiene_snapshot_can_preserve_current_state(
            cards_after=effective_cards_after
        )
    )
    active_high_risk = self._context_hygiene_active_high_risk_bash()
    decision = evaluate_context_hygiene_compact(
        stage_count_since_last_compact=stage_count_since,
        total_input_tokens_since_last_compact=tokens_since,
        cache_rates=list(getattr(self, "context_hygiene_cache_rates", []) or []),
        large_file_touch_counts=file_counts,
        large_tool_output_count=tool_outputs,
        seconds_since_last_compact=time.time()
        - float(
            getattr(self, "context_hygiene_last_compact_ts", 0.0)
            or getattr(self, "started_at", time.time())
        ),
        active_high_risk_bash=active_high_risk,
        snapshot_can_preserve_current_state=snapshot_ok,
        max_stages_without_compact=int(
            getattr(self.lhr, "context_hygiene_max_stages_without_compact", 25) or 25
        ),
        code_churn_stage_threshold=int(
            getattr(self.lhr, "context_hygiene_code_churn_stage_threshold", 10) or 10
        ),
        large_file_repeat_threshold=int(
            getattr(self.lhr, "context_hygiene_large_file_repeat_threshold", 10) or 10
        ),
        low_incremental_cache_rate=float(
            getattr(self.lhr, "context_hygiene_low_incremental_cache_rate", 0.80)
            or 0.80
        ),
        low_cache_window=int(
            getattr(self.lhr, "context_hygiene_low_cache_window", 5) or 5
        ),
        min_tokens_since_compact=int(
            getattr(self.lhr, "context_hygiene_min_tokens_since_compact", 5_000_000)
            or 5_000_000
        ),
        tool_output_min_interval_sec=float(
            getattr(self.lhr, "context_hygiene_tool_output_min_interval_sec", 1800.0)
            or 1800.0
        ),
    )
    facts = {
        **decision.facts,
        "token_delta": token_delta,
        "snapshot_preservation": snapshot_fields,
        "latest_stage": latest,
    }
    self._jsonl(
        "lhr_estra_events.jsonl",
        {
            "event": "cache_hygiene_compact_check",
            "latest_stage": latest,
            "stage_count": stage_count,
            "reason": decision.reason,
            "should_compact": decision.should_compact,
            "facts": facts,
        },
    )
    return (
        decision,
        effective_cards_after,
        latest,
        stage_count,
        tokens_since,
        tokens_in,
        facts,
        active_high_risk,
    )


async def _context_hygiene_compact_after_stage_capture(
    self, *, agent: Any, cards_after: list[Any]
) -> str | None:
    if not bool(getattr(self.lhr, "context_hygiene_compact_enabled", False)):
        return None
    if getattr(self, "pending_estra", None):
        return None
    (
        decision,
        effective_cards_after,
        latest,
        stage_count,
        tokens_since,
        tokens_in,
        facts,
        active_high_risk,
    ) = _evaluate_context_hygiene_after_stage(
        self, agent=agent, cards_after=cards_after
    )
    if not decision.should_compact:
        if active_high_risk:
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "cache_hygiene_compact_delayed_active_bash",
                    "latest_stage": latest,
                    "stage_count": stage_count,
                    "facts": facts,
                },
            )
        return None

    candidates = [
        c.stage_id for c in effective_cards_after if c.stage_id in self.stage_snapshots
    ]
    candidate_node_uids = {
        sid: self._active_node_uid_for_stage(sid) for sid in candidates
    }
    switch_candidates = self._estra_switch_candidates(candidates, latest)
    can_call_estra = bool(
        getattr(self.lhr, "estra_enabled", True)
        and candidates
    )
    if can_call_estra:
        compact_decision = await self._ask_estra(
            agent, trigger_source="context_hygiene"
        )
        if "compact" not in compact_decision:
            compact_decision = {
                **compact_decision,
                "compact": self._derive_estra_compact(
                    action=str(compact_decision.get("action") or ""),
                    trigger_source="context_hygiene",
                ),
            }
    else:
        compact_decision = self._safe_estra_fallback_decision(
            cards=effective_cards_after,
            candidates=candidates,
            latest=latest,
            trigger_source="context_hygiene",
            reason="context hygiene compact; estra unavailable",
        )
        self._emit_estra_decision(
            action=str(compact_decision.get("action") or "keep_current"),
            target_stage=str(compact_decision.get("target_stage") or latest),
            latest_stage=latest,
            trigger_source="context_hygiene",
            compact=True,
            reason=str(compact_decision.get("reason") or ""),
            decision_mode="safe_fallback",
            candidates=candidates,
            switch_candidates=switch_candidates,
            candidate_node_uids=candidate_node_uids,
        )
    if compact_decision.get("action") not in {
        "keep_current",
        "keep_but_redirect",
        "switch_stage",
    }:
        compact_decision = self._safe_estra_fallback_decision(
            cards=effective_cards_after,
            candidates=candidates,
            latest=latest,
            trigger_source="context_hygiene",
            reason="context hygiene compact; invalid estra action",
        )
        self._emit_estra_decision(
            action="keep_current",
            target_stage=str(compact_decision.get("target_stage") or latest),
            latest_stage=latest,
            trigger_source="context_hygiene",
            compact=True,
            reason=str(compact_decision.get("reason") or ""),
            decision_mode="safe_fallback",
            candidates=candidates,
            switch_candidates=switch_candidates,
            candidate_node_uids=candidate_node_uids,
        )
    compact_decision = {
        **compact_decision,
        "compact": True,
        "trigger_source": "context_hygiene",
        "context_generation": f"stage_count={stage_count};tokens_since={tokens_since}",
    }
    target = str(compact_decision.get("target_stage") or latest).upper()
    await self._set_pending_estra_from_decision(
        agent=agent, target=target, decision=compact_decision
    )
    if not self.pending_estra:
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "cache_hygiene_compact_failed",
                "latest_stage": latest,
                "stage_count": stage_count,
                "reason": "pending_estra_not_created",
                "facts": facts,
            },
        )
        return None
    self.context_hygiene_last_compact_stage_count = stage_count
    self.context_hygiene_last_compact_tokens_in = tokens_in
    self.context_hygiene_last_compact_ts = time.time()
    self._jsonl(
        "lhr_estra_events.jsonl",
        {
            "event": "cache_hygiene_compact_triggered",
            "latest_stage": latest,
            "stage_count": stage_count,
            "target_stage": target,
            "action": str(compact_decision.get("action") or "keep_current"),
            "compact": True,
            "reason": str(
                compact_decision.get("reason") or "context_hygiene_low_cache_or_churn"
            ),
            "facts": facts,
        },
    )
    if compact_decision.get("action") == "switch_stage":
        return f"[lnr] estra switch_stage chosen from context hygiene: {target}"
    action = str(compact_decision.get("action") or "keep_current")
    return f"[lnr] estra {action} chosen from context hygiene; compact current stage: {target}"


async def _force_estra_after_stage_capture(
    self, *, agent: Any, cards_after: list[Any]
) -> str | None:
    latest = cards_after[-1].stage_id if cards_after else ""
    candidates = [c.stage_id for c in cards_after if c.stage_id in self.stage_snapshots]
    candidate_node_uids = {
        sid: self._active_node_uid_for_stage(sid) for sid in candidates
    }
    self._jsonl(
        "lhr_estra_events.jsonl",
        {
            "event": "force_stage_estra_check",
            "latest_stage": latest,
            "latest_node_uid": self._active_node_uid_for_stage(latest),
            "stage_count": len(cards_after),
            "candidate_count": len(candidates),
            "candidate_node_uids": candidate_node_uids,
            "last_estra_stage_count": self.last_estra_stage_count,
            "last_force_estra_observation_count": int(
                getattr(self, "last_force_estra_observation_count", 0) or 0
            ),
            "trigger_source": "force_stage_capture",
            "force_after_stage_count": int(
                getattr(self.lhr, "force_estra_after_stage_count", 0) or 0
            ),
            "observed_stage_count": len(cards_after),
            "estra_observation_key": self._estra_observation_key(
                cards=cards_after, latest=latest, candidates=candidates
            ),
        },
    )
    if not self._force_estra_trigger_allowed(
        cards=cards_after,
        candidates=candidates,
        observed_stage_count=len(cards_after),
    ):
        return None
    observation_key = self._estra_observation_key(
        cards=cards_after, latest=latest, candidates=candidates
    )
    self.last_estra_stage_count = max(self.last_estra_stage_count, len(cards_after))
    self.last_estra_observation_key = observation_key
    self.last_force_estra_observation_key = observation_key
    self.last_force_estra_observation_count = max(
        int(getattr(self, "last_force_estra_observation_count", 0) or 0),
        len(cards_after),
    )
    decision = await self._ask_estra(agent, trigger_source="force_stage_capture")
    if "compact" not in decision:
        decision = {
            **decision,
            "compact": self._derive_estra_compact(
                action=str(decision.get("action") or ""),
                trigger_source="force_stage_capture",
            ),
        }
    if not bool(decision.get("compact")):
        return None
    target = str(decision.get("target_stage") or latest).upper()
    await self._set_pending_estra_from_decision(
        agent=agent, target=target, decision=decision
    )
    if decision.get("action") == "switch_stage":
        return f"[lnr] estra switch_stage chosen from forced stage estra: {target}"
    action = str(decision.get("action") or "keep_current")
    return f"[lnr] estra {action} chosen from forced stage estra; compact current stage: {target}"


async def _text_only_estra_callback(
    self,
    *,
    agent: Any,
    assistant_text: str,
    round_idx: int,
    max_steps: int,
) -> str | None:
    if isinstance(getattr(self, "pending_text_stage_commit", None), dict):
        return await self._handle_pending_stage_commit_text(
            agent=agent, assistant_text=assistant_text
        )

    repeated_terminal_out = self._suppress_repeated_text_only_completion(
        agent=agent,
        assistant_text=assistant_text,
        round_idx=round_idx,
        max_steps=max_steps,
    )
    if repeated_terminal_out:
        return None
    cards, latest, candidates = self._estra_candidates_for_current_ledger()
    has_estra_context = bool(cards or candidates)
    self._jsonl(
        "lhr_estra_events.jsonl",
        {
            "event": "text_only_estra_check"
            if has_estra_context
            else "text_only_estra_noop",
            "noop_reason": "" if has_estra_context else "no_stage_candidates",
            "latest_stage": latest,
            "latest_node_uid": self._active_node_uid_for_stage(latest),
            "stage_count": len(cards),
            "candidate_count": len(candidates),
            "candidate_node_uids": {
                sid: self._active_node_uid_for_stage(sid) for sid in candidates
            },
            "last_estra_stage_count": self.last_estra_stage_count,
            "trigger_source": "text_only",
            "round": int(round_idx),
            "max_steps": int(max_steps),
            "assistant_text": _compact_event_text(assistant_text),
            "assistant_text_chars": len(str(assistant_text or "")),
            "estra_observation_key": self._estra_observation_key(
                cards=cards, latest=latest, candidates=candidates
            ),
        },
    )
    if not self._text_only_estra_trigger_allowed(cards=cards, candidates=candidates):
        return None
    self.last_estra_stage_count = len(cards)
    self.last_estra_observation_key = self._estra_observation_key(
        cards=cards, latest=latest, candidates=candidates
    )
    decision = await self._ask_estra(agent, trigger_source="text_only")
    if "compact" not in decision:
        decision = {
            **decision,
            "compact": self._derive_estra_compact(
                action=str(decision.get("action") or ""), trigger_source="text_only"
            ),
        }
    decision = {**decision, "trigger_source": "text_only"}
    if not bool(decision.get("compact")):
        return None
    target = str(decision.get("target_stage") or latest).upper()
    await self._set_pending_estra_from_decision(
        agent=agent, target=target, decision=decision
    )
    if decision.get("action") == "switch_stage":
        return f"[lnr] estra switch_stage chosen from text-only: {target}"
    action = str(decision.get("action") or "keep_current")
    return (
        f"[lnr] estra {action} chosen from text-only; compact current stage: {target}"
    )


async def _context_limit_estra_decision(
    self,
    *,
    agent: Any,
    cards: list[Any],
    latest: str,
    candidates: list[str],
) -> dict[str, Any]:
    self.last_estra_stage_count = max(self.last_estra_stage_count, len(cards))
    self.last_estra_observation_key = self._estra_observation_key(
        cards=cards, latest=latest, candidates=candidates
    )
    decision = await self._ask_estra(agent, trigger_source="context_limit")
    if "compact" not in decision:
        decision = {
            **decision,
            "compact": self._derive_estra_compact(
                action=str(decision.get("action") or ""),
                trigger_source="context_limit",
            ),
        }
    return decision


async def _context_limit_estra_callback(
    self,
    *,
    agent: Any,
    round_idx: int,
    max_steps: int,
    omitted: int,
) -> str | None:
    _ = max_steps
    cards, latest, candidates = self._estra_candidates_for_current_ledger()
    try:
        memory_records = self._count_memory_records()
    except Exception:
        memory_records = 0
    context_generation = f"round={int(round_idx) + 1};memory_records={memory_records};omitted={int(omitted or 0)}"
    self._jsonl(
        "lhr_estra_events.jsonl",
        {
            "event": "context_limit_estra_check",
            "latest_stage": latest,
            "latest_node_uid": self._active_node_uid_for_stage(latest),
            "stage_count": len(cards),
            "candidate_count": len(candidates),
            "candidate_node_uids": {
                sid: self._active_node_uid_for_stage(sid) for sid in candidates
            },
            "last_estra_stage_count": self.last_estra_stage_count,
            "trigger_source": "context_limit",
            "omitted_before": int(omitted or 0),
            "context_generation": context_generation,
            "estra_observation_key": self._estra_observation_key(
                cards=cards, latest=latest, candidates=candidates
            ),
        },
    )
    if (
        not bool(getattr(self.lhr, "estra_enabled", True))
        or not cards
        or not candidates
    ):
        return None
    if context_generation == str(
        getattr(self, "last_context_limit_estra_generation", "") or ""
    ):
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "context_limit_estra_restore_suppressed",
                "latest_stage": latest,
                "latest_node_uid": self._active_node_uid_for_stage(latest),
                "trigger_source": "context_limit",
                "context_generation": context_generation,
                "restore_key": str(
                    getattr(self, "last_context_limit_estra_restore_key", "") or ""
                ),
                "reason": "same context-limit generation already reviewed",
                "omitted_before": int(omitted or 0),
            },
        )
        return None
    decision = await _context_limit_estra_decision(
        self,
        agent=agent,
        cards=cards,
        latest=latest,
        candidates=candidates,
    )
    if decision.get("action") not in {
        "keep_current",
        "keep_but_redirect",
        "switch_stage",
    }:
        decision = self._safe_estra_fallback_decision(
            cards=cards,
            candidates=candidates,
            latest=latest,
            trigger_source="context_limit",
            reason=f"context limit estra returned {decision.get('action') or 'invalid'}; keeping current route",
        )
        self._emit_estra_decision(
            action="keep_current",
            target_stage=str(decision.get("target_stage") or latest),
            latest_stage=latest,
            trigger_source="context_limit",
            compact=True,
            reason=str(decision.get("reason") or ""),
            decision_mode="safe_fallback",
            candidates=candidates,
            switch_candidates=self._estra_switch_candidates(candidates, latest),
            candidate_node_uids={
                sid: self._active_node_uid_for_stage(sid) for sid in candidates
            },
        )
    target = str(decision.get("target_stage") or latest).upper()
    if not target:
        return None
    action = str(decision.get("action") or "keep_current").strip() or "keep_current"
    observation_key = self._estra_observation_key(
        cards=cards, latest=latest, candidates=candidates
    )
    restore_key = ";".join(
        [
            f"latest={latest}",
            f"target={target}",
            f"action={action}",
            f"observation={observation_key}",
            f"omitted={int(omitted or 0)}",
        ]
    )
    restore_keys = getattr(self, "context_limit_estra_restore_keys", None)
    if not isinstance(restore_keys, set):
        restore_keys = set()
        self.context_limit_estra_restore_keys = restore_keys
    if restore_key in restore_keys:
        self.last_context_limit_estra_generation = context_generation
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "context_limit_estra_restore_suppressed",
                "latest_stage": latest,
                "latest_node_uid": self._active_node_uid_for_stage(latest),
                "target_stage": target,
                "action": action,
                "trigger_source": "context_limit",
                "context_generation": context_generation,
                "restore_key": restore_key,
                "reason": "same context-limit estra restore already attempted; falling back to in-band compact",
                "omitted_before": int(omitted or 0),
            },
        )
        return None
    restore_keys.add(restore_key)
    self.last_context_limit_estra_generation = context_generation
    self.last_context_limit_estra_restore_key = restore_key
    decision = {
        **decision,
        "compact": True,
        "compact_strength": "strict_context_limit",
        "trigger_source": "context_limit",
        "context_generation": context_generation,
        "restore_key": restore_key,
    }
    await self._set_pending_estra_from_decision(
        agent=agent, target=target, decision=decision
    )
    if (self.pending_estra or {}).get("action") == "switch_stage":
        return f"[lnr] estra switch_stage chosen from context-limit: {target}"
    action = str((self.pending_estra or {}).get("action") or "keep_current")
    return f"[lnr] estra {action} chosen from context-limit; compact current stage: {target}"


@classmethod
def _parse_estra_decision(
    cls,
    text: str,
    *,
    switch_candidates: list[str],
) -> dict[str, Any]:
    return parse_estra_decision(text, switch_candidates=switch_candidates)


def _compact_event_text(text: str, *, max_chars: int = 900) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 1].rstrip() + "…"


def _text_only_completion_signature(text: str) -> str | None:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return None
    lower = raw.lower()
    terminal_markers = (
        "conversation has definitively concluded",
        "definitively concluded",
        "stop responding",
        "not respond further",
        "no further action",
        "no further actions",
        "all experiments concluded",
        "final status confirmed",
        "goodbye",
    )
    completion_markers = (
        "the task is complete",
        "task is complete",
        "final validation score",
        "final test score",
        "final score",
        *terminal_markers,
    )
    if not any(marker in lower for marker in completion_markers):
        return None
    parts: list[str] = []
    if "task is complete" in lower:
        parts.append("task_complete")
    if any(marker in lower for marker in terminal_markers):
        parts.append("terminal_text")
    score_re = re.compile(
        r"\b(final\s+(?:validation\s+|test\s+)?score)\s*[:=]\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
        flags=re.I,
    )
    for label, value in score_re.findall(raw):
        normalized_label = re.sub(r"\s+", "_", label.strip().lower())
        parts.append(f"{normalized_label}:{value.strip().lower()}")
    return "|".join(parts) if parts else "completion_status"


def _text_only_continue_search_prompt(
    self, *, signature: str, repeat_count: int
) -> str:
    remaining_sec = max(
        0, int(float(getattr(self, "deadline", 0.0) or 0.0) - time.monotonic())
    )
    remaining_min = max(0, int(round(remaining_sec / 60.0)))
    return (
        "[LNR_CONTINUE_SEARCH]\n"
        f"Wall-clock budget is still active (~{remaining_min} min remaining). "
        "A text-only final/goodbye message is not a valid stop condition for this worker; "
        "workers stop only when the time budget expires. Continue the ML search now with a concrete tool call in the current workspace. "
        "Do not reply with another final/goodbye/status-only message.\n"
        "Exploration state:\n"
        "This is a continued search state after a text-only terminal response. Prior summaries are completed evidence, not a stop signal and not something to repeat. "
        "Continue from the current workspace by testing a materially different idea, verifying the current route, or preserving/restoring the best known artifact. "
        "Do not overwrite the best submission without a validated candidate.\n"
        f"Suppressed repeated terminal text signature={signature}; repeat_count={int(repeat_count)}."
    )


def _suppress_repeated_text_only_completion(
    self,
    *,
    agent: Any,
    assistant_text: str,
    round_idx: int,
    max_steps: int,
) -> str | None:
    signature = _text_only_completion_signature(assistant_text)
    if not signature:
        self._last_text_only_completion_signature = ""
        self._last_text_only_completion_repeat_count = 0
        return None
    previous = str(getattr(self, "_last_text_only_completion_signature", "") or "")
    if signature != previous:
        self._last_text_only_completion_signature = signature
        self._last_text_only_completion_repeat_count = 1
        return None
    repeat_count = (
        int(getattr(self, "_last_text_only_completion_repeat_count", 1) or 1) + 1
    )
    self._last_text_only_completion_repeat_count = repeat_count
    reason = f"repeated_text_only_completion:{repeat_count}"
    continue_prompt = self._text_only_continue_search_prompt(
        signature=signature, repeat_count=repeat_count
    )
    prompt_injected = False
    try:
        setattr(agent, "_lnr_suppress_current_text_only_memory", reason)
    except Exception:
        logger.debug(
            "[lnr] could not mark repeated text-only completion for suppression",
            exc_info=True,
        )
    try:
        memory = getattr(agent, "memory", None)
        if memory is not None and hasattr(memory, "add_message"):
            memory.add_message(Message.user_message(continue_prompt))
            prompt_injected = True
    except Exception:
        logger.debug(
            "[lnr] could not inject repeated text-only continuation prompt",
            exc_info=True,
        )
    self._jsonl(
        "lhr_estra_events.jsonl",
        {
            "event": "text_only_completion_duplicate_suppressed",
            "trigger_source": "text_only",
            "round": int(round_idx),
            "max_steps": int(max_steps),
            "signature": signature,
            "repeat_count": repeat_count,
            "assistant_text": _compact_event_text(assistant_text),
            "assistant_text_chars": len(str(assistant_text or "")),
            "continue_prompt_injected": prompt_injected,
            "continue_prompt": _compact_event_text(continue_prompt),
        },
    )
    return "[lnr] repeated text-only terminal response suppressed; continue-search prompt injected"


def _parse_stage_commit_text_block(text: str) -> tuple[dict[str, Any], str, str]:
    raw = str(text or "")
    match = re.search(
        r"STAGE_COMMIT_BEGIN\s*(.*?)\s*STAGE_COMMIT_END", raw, flags=re.I | re.S
    )
    if not match:
        return {}, "", "missing_block"
    block_body = match.group(1).strip()
    block_text = "STAGE_COMMIT_BEGIN\n" + block_body + "\nSTAGE_COMMIT_END"
    data: dict[str, Any] = {}
    current_key = ""
    for line in block_body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key_part, sep, value_part = stripped.partition(":")
        if sep:
            key = re.sub(r"[^a-z0-9_]+", "_", key_part.strip().lower()).strip("_")
            if key:
                data[key] = value_part.strip()
                current_key = key
            continue
        if current_key:
            data[current_key] = (
                str(data.get(current_key) or "") + " " + stripped
            ).strip()
    required = ("stage_id", "metric", "metric_validity", "brief", "why", "files")
    missing = [k for k in required if not str(data.get(k) or "").strip()]
    if missing:
        return data, block_text, "missing=" + ",".join(missing)
    validity = str(data.get("metric_validity") or "").strip().lower()
    if validity not in {"high", "medium", "low"}:
        return data, block_text, "invalid_metric_validity"
    try:
        float(str(data.get("metric") or "").strip())
    except (TypeError, ValueError):
        return data, block_text, "invalid_metric"
    return data, block_text, ""
