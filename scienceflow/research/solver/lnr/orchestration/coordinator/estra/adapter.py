# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: EStra planning, decisions, archive summaries, and pending state.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.control.ephemeral_agent_session import llm_correlation
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.coordination import (
    _compact_event_text,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    EstraArchiveStore,
    EstraContext,
    EstraDecision,
    EstraPlanner,
    EstraPlanRequest,
    Message,
    build_estra_archive_summary_prompt,
    build_estra_prompt,
    derive_estra_compact,
    estra_axes_from_action,
    estra_decision_kind,
    hashlib,
    json,
    logger,
    parse_stage_cards,
    re,
    read_ledger,
    tail_summary_after_cards,
    tail_summary_from_cards,
    time,
)


@staticmethod
def _estra_switch_candidates(candidates: list[str], latest: str) -> list[str]:
    latest = str(latest or "").upper()
    return [
        str(sid or "").upper()
        for sid in candidates
        if str(sid or "").upper() and str(sid or "").upper() != latest
    ]


@staticmethod
def _is_keep_like_estra_action(action: str) -> bool:
    return str(action or "").strip() in {"keep_current", "keep_but_redirect"}


@staticmethod
def _estra_axes_from_action(action: str) -> tuple[str, str]:
    return estra_axes_from_action(action)


@staticmethod
def _estra_decision_kind(*, action: str, startpoint: str, intent: str) -> str:
    return estra_decision_kind(action=action, startpoint=startpoint, intent=intent)


@staticmethod
def _derive_estra_compact(*, action: str, trigger_source: str) -> bool:
    return derive_estra_compact(action=action, trigger_source=trigger_source)


def _estra_decision_fields(self, parsed: dict[str, Any]) -> dict[str, str]:
    field_names = (
        "startpoint",
        "intent",
        "exploration_summary",
        "bottleneck",
        "evidence",
        "missing_evidence",
        "decision_reason",
        "redirect_focus",
    )
    fields: dict[str, str] = {}
    for name in field_names:
        value = parsed.get(name)
        if value is None:
            continue
        compact = _compact_event_text(str(value), max_chars=240)
        if compact:
            fields[name] = compact
    return fields


@staticmethod
def _estra_observation_key(
    *, cards: list[Any], latest: str, candidates: list[str]
) -> str:
    ids = [str(getattr(card, "stage_id", "") or "").upper() for card in cards]
    candidate_text = ",".join(str(sid or "").upper() for sid in candidates)
    digest = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()[:12]
    return f"stages={len(ids)};latest={str(latest or '').upper()};candidates={digest}"


def _emit_estra_decision(
    self,
    *,
    action: str,
    target_stage: str,
    latest_stage: str,
    trigger_source: str,
    compact: bool,
    reason: str,
    decision_mode: str,
    candidates: list[str],
    switch_candidates: list[str],
    candidate_node_uids: dict[str, str],
    raw: str = "",
    estra_fields: dict[str, Any] | None = None,
    estra_context: dict[str, Any] | None = None,
) -> None:
    self.estra_decisions = int(getattr(self, "estra_decisions", 0) or 0) + 1
    target = str(target_stage or "").upper()
    latest = str(latest_stage or "").upper()
    fields = estra_fields or {}
    context = dict(estra_context or {})
    startpoint = str(
        fields.get("startpoint") or self._estra_axes_from_action(action)[0]
    )
    intent = str(fields.get("intent") or self._estra_axes_from_action(action)[1])
    decision_kind = self._estra_decision_kind(
        action=action, startpoint=startpoint, intent=intent
    )
    decision_text = " ".join(
        str(fields.get(name) or "")
        for name in ("decision_reason", "bottleneck", "evidence", "redirect_focus")
    ).lower()
    if context.get("peer_route_evidence_present"):
        peer_tokens = [
            str(context.get("peer_best_worker") or "").lower(),
            str(context.get("peer_best_stage") or "").lower(),
            "peer",
            "other worker",
        ]
        context["estra_addressed_peer_best"] = any(
            token and token in decision_text for token in peer_tokens
        )
    if context.get("backtrack_reflection_present"):
        backtrack_stage = str(context.get("backtrack_best_stage") or "").lower()
        context["estra_addressed_backtrack"] = (
            startpoint == "previous_stage"
            or "previous" in decision_text
            or "restore" in decision_text
            or "backtrack" in decision_text
            or bool(backtrack_stage and backtrack_stage in decision_text)
        )
    self._jsonl(
        "lhr_estra_events.jsonl",
        {
            "event": "estra_decision",
            "action": action,
            "target_stage": target,
            "target_node_uid": candidate_node_uids.get(target, ""),
            "latest_stage": latest,
            "latest_node_uid": candidate_node_uids.get(latest, ""),
            "trigger_source": trigger_source,
            "compact": bool(compact),
            "decision_mode": decision_mode,
            "candidates": candidates,
            "switch_candidates": switch_candidates,
            "candidate_node_uids": candidate_node_uids,
            "reason": reason,
            "startpoint": startpoint,
            "intent": intent,
            "decision_kind": decision_kind,
            "exploration_summary": str(fields.get("exploration_summary") or ""),
            "bottleneck": str(fields.get("bottleneck") or ""),
            "evidence": str(fields.get("evidence") or ""),
            "missing_evidence": str(fields.get("missing_evidence") or ""),
            "decision_reason": str(fields.get("decision_reason") or ""),
            "redirect_focus": str(fields.get("redirect_focus") or ""),
            **context,
            "raw": raw,
            "response_chars": len(str(raw or "")),
        },
    )
    archive = getattr(self, "estra_archive_store", None)
    if isinstance(archive, EstraArchiveStore):
        decision = EstraDecision(
            action=str(action or "keep_current"),
            startpoint=startpoint,
            intent=intent,
            target_stage=target,
            compact=bool(compact),
            reason=str(reason or ""),
            diagnostics={
                name: str(fields.get(name) or "")
                for name in (
                    "exploration_summary",
                    "bottleneck",
                    "evidence",
                    "missing_evidence",
                    "decision_reason",
                    "redirect_focus",
                )
                if fields.get(name) not in (None, "")
            },
        )
        correlation = hashlib.sha256(
            json.dumps(
                {
                    "worker_id": self.worker_id,
                    "latest_stage": latest,
                    "trigger_source": trigger_source,
                    "raw": raw,
                    "decision": decision.to_dict(),
                },
                ensure_ascii=True,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20]
        try:
            archive.record(
                request_id=f"estra:{self.worker_id or 'W00'}:{correlation}",
                context=EstraContext(
                    latest_stage=latest,
                    switch_candidates=tuple(switch_candidates),
                    trigger_source=trigger_source,
                    metadata={"candidates": tuple(candidates)},
                ),
                decision=decision,
                decision_mode=decision_mode,
                raw=raw,
                expected_stage=latest,
                metadata={"candidate_node_uids": candidate_node_uids},
            )
        except (OSError, TypeError, ValueError):
            logger.debug("[lnr] estra envelope archive failed", exc_info=True)


def _early_estra_decision(
    self,
    *,
    trigger_source: str,
    latest: str,
    candidates: list[str],
    switch_candidates: list[str],
    candidate_node_uids: dict[str, str],
) -> dict[str, Any] | None:
    forced_target = ""
    if str(trigger_source or "") == "force_stage_capture":
        forced_target = (
            str(getattr(self.lhr, "force_estra_target_stage", "") or "").strip().upper()
        )
    if forced_target:
        if forced_target == str(latest or "").upper():
            decision = {
                "action": "keep_current",
                "target_stage": latest,
                "compact": self._derive_estra_compact(
                    action="keep_current", trigger_source=trigger_source
                ),
                "reason": f"forced estra target {forced_target} is current stage",
            }
        elif forced_target in switch_candidates:
            decision = {
                "action": "switch_stage",
                "target_stage": forced_target,
                "compact": True,
                "reason": f"forced estra target {forced_target}",
            }
        else:
            decision = {
                "action": "keep_current",
                "target_stage": latest,
                "compact": self._derive_estra_compact(
                    action="keep_current", trigger_source=trigger_source
                ),
                "reason": f"forced estra target {forced_target} is not a valid switch candidate",
            }
        self._emit_estra_decision(
            action=decision["action"],
            target_stage=decision["target_stage"],
            latest_stage=latest,
            trigger_source=trigger_source,
            compact=bool(decision["compact"]),
            reason=str(decision.get("reason") or ""),
            decision_mode="forced_config",
            candidates=candidates,
            switch_candidates=switch_candidates,
            candidate_node_uids=candidate_node_uids,
        )
        return decision

    return None


async def _plan_estra_request(
    self,
    *,
    agent: Any,
    cards: list[Any],
    latest: str,
    candidates: list[str],
    switch_candidates: list[str],
    trigger_source: str,
) -> tuple[Any, dict[str, Any], str, bool]:
    best = self._best_candidate_stage_id(cards, candidates)
    stage_memory_text = self._stage_memory_view_for_prompt(
        cards,
        latest_stage=latest,
        best_stage=best,
    )
    peer_evidence = self._estra_peer_route_evidence()
    backtrack_text, backtrack_meta = self._estra_backtrack_reflection(
        switch_candidates,
        latest_stage=latest,
    )
    estra_context_audit = {**peer_evidence.audit_fields(), **backtrack_meta}
    prompt = build_estra_prompt(
        ledger_filename="Stage Memory View",
        ledger_text=agent._sanitize_agent_visible_paths(stage_memory_text),
        latest_stage=latest,
        switch_candidate_stages=switch_candidates,
        stage_checkpoint_context=self._estra_stage_checkpoint_context(candidates),
        peer_route_evidence=peer_evidence.text,
        backtrack_reflection=backtrack_text,
        resource_context=self._resource_context_for_prompt(),
    )
    if str(trigger_source or "") == "context_limit":
        prompt = (
            "ESTRA trigger: the main agent context is about to overflow before the next LLM turn. "
            "Choose a compacted next stage by setting startpoint and intent. Use previous_stage only if a historical completed stage is a better restart point.\n\n"
            + prompt
        )
    estra_system = (
        "You are a estra decision controller for one ML search run. "
        "Return exactly one JSON object and nothing else. "
        "Do not call tools, emit DSML/tool markup, write files, or use markdown."
    )
    use_main_context = (
        str(trigger_source or "") != "context_limit"
        and bool(getattr(self.lhr, "estra_use_main_agent_context", True))
        and hasattr(getattr(agent, "llm", None), "ask_tool_stream")
        and hasattr(agent, "_memory_ctx")
    )

    async def isolated_model(_request: EstraPlanRequest) -> str:
        output = await agent.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="estra",
            base_messages=[],
            system_messages=[Message.system_message(estra_system)],
            timeout=agent._llm_stream_timeout_sec,
        )
        estra_context_audit["llm_correlation"] = llm_correlation(output)
        self._accumulate_ephemeral_tokens(agent, "estra")
        return str(output or "")

    async def main_context_model(_request: EstraPlanRequest) -> str:
        output = await agent.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="estra",
            base_messages=agent._memory_ctx.build_messages_for_llm(),
            system_messages=agent._build_system_messages(),
            timeout=agent._llm_stream_timeout_sec,
        )
        estra_context_audit["llm_correlation"] = llm_correlation(output)
        self._accumulate_ephemeral_tokens(agent, "estra")
        return output

    planner = getattr(self, "estra_planner", None)
    if not isinstance(planner, EstraPlanner):
        planner = EstraPlanner(getattr(self, "estra_service", None))
        self.estra_planner = planner
    primary_mode = "main_agent_context" if use_main_context else "isolated"
    plan = await planner.plan(
        EstraPlanRequest(
            request_id=f"estra:{self.worker_id or 'W00'}:{latest}:{trigger_source}",
            context=EstraContext(
                latest_stage=latest,
                switch_candidates=tuple(switch_candidates),
                trigger_source=trigger_source,
            ),
            prompt=prompt,
            metadata={"candidates": tuple(candidates)},
        ),
        main_context_model if use_main_context else isolated_model,
        primary_mode=primary_mode,
        fallback=isolated_model if use_main_context else None,
    )
    return plan, estra_context_audit, primary_mode, use_main_context


def _normalize_estra_plan(
    self,
    *,
    plan: Any,
    use_main_context: bool,
    primary_mode: str,
    latest: str,
    trigger_source: str,
    candidates: list[str],
    switch_candidates: list[str],
    candidate_node_uids: dict[str, str],
    estra_context_audit: dict[str, Any],
) -> dict[str, Any]:
    first_attempt = plan.attempts[0]
    if first_attempt.outcome == "error":
        error = first_attempt.error.partition(": ")[2] or first_attempt.error
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "estra_llm_error",
                "error": error,
                "candidates": candidates,
                "switch_candidates": switch_candidates,
                "candidate_node_uids": candidate_node_uids,
                "latest_stage": latest,
                "trigger_source": trigger_source,
                "decision_mode": primary_mode,
            },
        )

    if use_main_context and first_attempt.outcome == "invalid":
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "estra_main_context_invalid_fallback",
                "latest_stage": latest,
                "trigger_source": trigger_source,
                "candidates": candidates,
                "switch_candidates": switch_candidates,
                "raw": first_attempt.raw,
                "response_chars": len(first_attempt.raw),
            },
        )
    if len(plan.attempts) > 1 and plan.attempts[-1].outcome == "error":
        fallback_error = plan.attempts[-1].error
        fallback_error = fallback_error.partition(": ")[2] or fallback_error
        self._jsonl(
            "lhr_estra_events.jsonl",
            {
                "event": "estra_fallback_llm_error",
                "error": fallback_error,
                "latest_stage": latest,
                "trigger_source": trigger_source,
            },
        )
    text = next(
        (attempt.raw for attempt in reversed(plan.attempts) if attempt.raw),
        "",
    )
    parsed = self._parse_estra_decision(text, switch_candidates=switch_candidates)
    has_valid_attempt = any(attempt.outcome == "valid" for attempt in plan.attempts)
    normalized = plan.decision.to_dict() if has_valid_attempt else {}
    decision_mode = plan.decision_mode
    if not normalized:
        if not any(attempt.outcome == "error" for attempt in plan.attempts):
            self._jsonl(
                "lhr_estra_events.jsonl",
                {
                    "event": "estra_invalid",
                    "latest_stage": latest,
                    "trigger_source": trigger_source,
                    "decision_mode": decision_mode,
                    "candidates": candidates,
                    "switch_candidates": switch_candidates,
                    "candidate_node_uids": candidate_node_uids,
                    "raw": text,
                    "response_chars": len(str(text or "")),
                },
            )
        error = next(
            (
                attempt.error.partition(": ")[2] or attempt.error
                for attempt in reversed(plan.attempts)
                if attempt.error
            ),
            "",
        )
        normalized = self._safe_estra_fallback_decision(
            cards=[],
            candidates=candidates,
            latest=latest,
            trigger_source=trigger_source,
            reason=f"estra llm error: {error}" if error else "invalid estra response",
        )
        decision_mode = "safe_fallback"
    reason = str(normalized.get("reason") or parsed.get("reason") or "")
    normalized["reason"] = reason
    self._emit_estra_decision(
        action=str(normalized.get("action") or "keep_current"),
        target_stage=str(normalized.get("target_stage") or latest),
        latest_stage=latest,
        trigger_source=trigger_source,
        compact=bool(normalized.get("compact")),
        reason=reason,
        decision_mode=decision_mode,
        candidates=candidates,
        switch_candidates=switch_candidates,
        candidate_node_uids=candidate_node_uids,
        raw=text,
        estra_fields=normalized,
        estra_context=estra_context_audit,
    )
    return normalized


async def _ask_estra(
    self, agent: Any, *, trigger_source: str = "manual"
) -> dict[str, Any]:
    cards = self._effective_stage_cards_from_ledger()
    latest = cards[-1].stage_id if cards else ""
    candidates = [c.stage_id for c in cards if c.stage_id in self.stage_snapshots]
    candidate_node_uids = {
        sid: self._active_node_uid_for_stage(sid) for sid in candidates
    }
    switch_candidates = self._estra_switch_candidates(candidates, latest)
    if not cards or not candidates:
        return {
            "action": "keep_current",
            "target_stage": latest,
            "compact": self._derive_estra_compact(
                action="keep_current", trigger_source=trigger_source
            ),
            "reason": "no durable stage candidate for estra",
        }
    early_decision = _early_estra_decision(
        self,
        trigger_source=trigger_source,
        latest=latest,
        candidates=candidates,
        switch_candidates=switch_candidates,
        candidate_node_uids=candidate_node_uids,
    )
    if early_decision is not None:
        return early_decision

    (
        plan,
        estra_context_audit,
        primary_mode,
        use_main_context,
    ) = await _plan_estra_request(
        self,
        agent=agent,
        cards=cards,
        latest=latest,
        candidates=candidates,
        switch_candidates=switch_candidates,
        trigger_source=trigger_source,
    )
    return _normalize_estra_plan(
        self,
        plan=plan,
        use_main_context=use_main_context,
        primary_mode=primary_mode,
        latest=latest,
        trigger_source=trigger_source,
        candidates=candidates,
        switch_candidates=switch_candidates,
        candidate_node_uids=candidate_node_uids,
        estra_context_audit=estra_context_audit,
    )


@staticmethod
def _normalize_estra_archive_summary_text(text: str, *, max_chars: int = 1000) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"^```(?:text|markdown)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())
    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        cleaned = re.sub(r"^[-*•]\s*", "", cleaned).strip()
        if cleaned:
            lines.append(f"- {cleaned}")
        if len(lines) >= 5:
            break
    summary = "\n".join(lines) if lines else re.sub(r"\s+", " ", raw).strip()
    if len(summary) > max_chars:
        summary = (
            summary[: max(0, max_chars - 32)].rstrip()
            + "\n... [archive summary truncated]"
        )
    return summary


async def _synthesize_estra_archive_summary(
    self,
    agent: Any,
    *,
    target_stage: str,
    terminal_stage: str,
    estra_reason: str,
    deterministic_summary: str,
) -> str:
    fallback = str(deterministic_summary or "").strip()
    if not fallback:
        return ""
    if not bool(getattr(self.lhr, "estra_archive_summary_llm_enabled", True)):
        return fallback
    llm = getattr(agent, "llm", None)
    if llm is None or not callable(getattr(llm, "ask", None)):
        return fallback
    prompt = build_estra_archive_summary_prompt(
        target_stage=target_stage,
        terminal_stage=terminal_stage,
        estra_reason=estra_reason,
        deterministic_summary=fallback,
    )
    system = (
        "You summarize estra-abandoned ML research tails. Return compact plain text only. "
        "Do not call tools, write files, or invent details."
    )
    try:
        text = await agent.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="estra_archive_summary",
            base_messages=[],
            system_messages=[Message.system_message(system)],
            timeout=float(getattr(agent, "_llm_stream_timeout_sec", 60) or 60),
        )
        self._accumulate_ephemeral_tokens(agent, "estra")
    except BaseException as exc:
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_archive_summary_llm_error",
                "target_stage": str(target_stage or "").upper(),
                "terminal_stage": str(terminal_stage or "").upper(),
                "error": str(exc),
                "fallback_chars": len(fallback),
            },
        )
        return fallback

    summary = self._normalize_estra_archive_summary_text(
        str(text or ""), max_chars=1000
    )
    if not summary:
        return fallback
    self._jsonl(
        "lhr_estras.jsonl",
        {
            "event": "estra_archive_summary_synthesized",
            "target_stage": str(target_stage or "").upper(),
            "terminal_stage": str(terminal_stage or "").upper(),
            "deterministic_summary_chars": len(fallback),
            "summary_chars": len(summary),
            "summary_excerpt": _compact_event_text(summary, max_chars=600),
            "llm_correlation": llm_correlation(text),
        },
    )
    return summary


async def _set_pending_estra_from_decision(
    self, *, agent: Any, target: str, decision: dict[str, Any]
) -> None:
    if not bool(decision.get("compact")):
        return
    current_ledger = read_ledger(self.ledger_path)
    cards = self._effective_stage_cards(parse_stage_cards(current_ledger))
    latest = cards[-1].stage_id if cards else ""
    action = str(decision.get("action") or "").strip()
    if self._is_keep_like_estra_action(action):
        target = latest
    elif action == "switch_stage":
        target = str(target or decision.get("target_stage") or "").upper()
    else:
        return
    if not target:
        return
    summary_max_chars = int(
        getattr(self.lhr, "estra_compact_max_chars", 0)
        or self.lhr.tail_summary_max_chars
        or 1200
    )
    if action == "switch_stage":
        deterministic_summary = tail_summary_after_cards(
            cards, target_stage=target, max_chars=summary_max_chars
        )
        summary = await self._synthesize_estra_archive_summary(
            agent,
            target_stage=target,
            terminal_stage=latest,
            estra_reason=str(decision.get("reason") or ""),
            deterministic_summary=deterministic_summary,
        )
    else:
        summary = tail_summary_from_cards(
            cards,
            start_stage="S02",
            target_stage=target,
            max_chars=summary_max_chars,
        )
        summary = summary.replace(
            "Abandoned compact trajectory", "Compact current trajectory"
        )
        summary = summary.replace(f" before estra to {target}", "")
        summary = summary.replace(" before estra", "")
    reason = str(decision.get("reason") or "")
    startpoint, intent = self._estra_axes_from_action(action)
    estra_fields = {
        "startpoint": str(decision.get("startpoint") or startpoint),
        "intent": str(decision.get("intent") or intent),
        **self._estra_decision_fields(decision),
    }
    state_packet = self._build_lhr_state_packet(
        action=action,
        target_stage=target,
        terminal_stage=latest,
        reason=reason,
        tail_summary=summary,
        estra_fields=estra_fields,
    )
    self.pending_estra = {
        "action": action,
        "target_stage": target,
        "target_node_uid": self._active_node_uid_for_stage(target),
        "terminal_stage": latest,
        "terminal_node_uid": self._active_node_uid_for_stage(latest),
        "tail_summary": summary,
        "terminal_ledger": current_ledger,
        "reason": reason,
        "state_packet": state_packet,
        "estra_fields": estra_fields,
        "compact_strength": str(decision.get("compact_strength") or ""),
        "trigger_source": str(decision.get("trigger_source") or ""),
        "context_generation": str(decision.get("context_generation") or ""),
        "restore_key": str(decision.get("restore_key") or ""),
    }


def _estra_candidates_for_current_ledger(self) -> tuple[list[Any], str, list[str]]:
    cards = self._effective_stage_cards_from_ledger()
    latest = cards[-1].stage_id if cards else ""
    candidates = [c.stage_id for c in cards if c.stage_id in self.stage_snapshots]
    return cards, latest, candidates


def _estra_trigger_allowed(self, *, cards: list[Any], candidates: list[str]) -> bool:
    if not bool(self.lhr.estra_enabled):
        return False
    trigger_after = int(self.lhr.estra_trigger_stage_count or 0)
    min_stages = trigger_after if trigger_after > 0 else 2
    if len(cards) < min_stages or not candidates:
        return False
    latest = cards[-1].stage_id if cards else ""
    observation_key = self._estra_observation_key(
        cards=cards, latest=latest, candidates=candidates
    )
    if observation_key == str(getattr(self, "last_estra_observation_key", "") or ""):
        return False
    if len(cards) <= self.last_estra_stage_count:
        return False
    return True


def _text_only_estra_trigger_allowed(
    self, *, cards: list[Any], candidates: list[str]
) -> bool:
    if not bool(getattr(self.lhr, "estra_enabled", True)):
        return False
    if not cards or not candidates:
        return False
    if getattr(self, "pending_estra", None):
        return False
    trigger_after = int(getattr(self.lhr, "estra_trigger_stage_count", 0) or 0)
    min_stages = trigger_after if trigger_after > 0 else 2
    if len(cards) < min_stages:
        return False
    observation_key = self._estra_observation_key(
        cards=cards,
        latest=cards[-1].stage_id,
        candidates=candidates,
    )
    if observation_key == str(getattr(self, "last_estra_observation_key", "") or ""):
        return False
    if len(cards) <= int(getattr(self, "last_estra_stage_count", 0) or 0):
        return False
    return time.monotonic() < float(getattr(self, "deadline", 0.0) or 0.0)


def _force_estra_trigger_allowed(
    self,
    *,
    cards: list[Any],
    candidates: list[str],
    observed_stage_count: int | None = None,
) -> bool:
    force_after = int(getattr(self.lhr, "force_estra_after_stage_count", 0) or 0)
    if force_after <= 0:
        return False
    if not bool(getattr(self.lhr, "estra_enabled", True)):
        return False
    observed_count = int(
        observed_stage_count if observed_stage_count is not None else len(cards)
    )
    if observed_count < force_after or not candidates:
        return False
    latest = cards[-1].stage_id if cards else ""
    last_observed = max(
        int(getattr(self, "last_force_estra_observation_count", 0) or 0),
        int(getattr(self, "last_estra_stage_count", 0) or 0),
    )
    if observed_count <= last_observed:
        return False
    observation_key = self._estra_observation_key(
        cards=cards, latest=latest, candidates=candidates
    )
    if observation_key == str(
        getattr(self, "last_force_estra_observation_key", "") or ""
    ):
        return False
    return True


def _current_main_token_totals(self, agent: Any) -> tuple[int, int]:
    run_in = int(
        getattr(agent, "_run_tokens_in", 0)
        or getattr(agent, "last_run_tokens_in", 0)
        or 0
    )
    run_cached = int(
        getattr(agent, "_run_tokens_cached", 0)
        or getattr(agent, "last_run_tokens_cached", 0)
        or 0
    )
    return int(getattr(self, "main_tokens_in", 0) or 0) + run_in, int(
        getattr(self, "main_tokens_cached", 0) or 0
    ) + run_cached
