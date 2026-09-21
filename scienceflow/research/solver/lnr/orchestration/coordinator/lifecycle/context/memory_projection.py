# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Extracted coordinator component with an explicit dependency surface."""

from __future__ import annotations

from scienceflow.research.control.ephemeral_agent_session import llm_correlation
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    MemoryService,
    Message,
    Path,
    ProtectedContextRequest,
    StageCard,
    logger,
    normalize_stage_id,
    parse_stage_cards,
    re,
    read_ledger,
    render_stage_cards,
    stage_cards_with_overrides,
)


@staticmethod
def _message_content_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _build_protected_eda_facts_summary(self, agent: Any, *, end_index: int) -> str:
    max_chars = max(
        1200, int(getattr(self.lhr, "protected_eda_facts_max_chars", 6000) or 6000)
    )
    messages = self._agent_memory_messages(agent)
    memory_service = getattr(self, "memory_service", None)
    if not isinstance(memory_service, MemoryService):
        memory_service = MemoryService(workspace_dir=self.workspace_dir)
        self.memory_service = memory_service
    return memory_service.protected_context.build_facts_summary(
        messages,
        end_index=end_index,
        max_chars=max_chars,
    )


def _mark_protected_eda_prefix(self, agent: Any, *, stage_id: str) -> dict[str, Any]:
    if str(stage_id or "").upper() != "S01":
        return {}
    if not bool(getattr(self.lhr, "preserve_prefix_and_eda", True)):
        return {}
    ctx = getattr(agent, "_memory_ctx", None)
    warn_chars = int(getattr(self.lhr, "protected_eda_warn_chars", 50_000) or 0)
    captured_end = getattr(self, "_s01_eda_prefix_end_index", None)
    end_index = (
        int(captured_end) if captured_end is not None else self._count_memory_records()
    )
    mode = (
        str(getattr(self.lhr, "protected_eda_mode", "facts") or "facts").strip().lower()
    )
    agent_summary = str(getattr(self, "_s01_agent_eda_summary", "") or "").strip()
    use_agent_summary = mode in {"agent", "agent_summary"} and bool(agent_summary)
    summary = ""
    request_mode = "raw"
    if mode not in {"raw", "verbatim"}:
        summary = (
            agent_summary
            if use_agent_summary
            else self._build_protected_eda_facts_summary(agent, end_index=end_index)
        )
        request_mode = (
            "agent"
            if use_agent_summary
            else ("facts_fallback" if mode in {"agent", "agent_summary"} else "facts")
        )
    memory_service = getattr(self, "memory_service", None)
    if not isinstance(memory_service, MemoryService):
        memory_service = MemoryService(workspace_dir=self.workspace_dir)
        self.memory_service = memory_service
    result = memory_service.protect_context(
        ctx,
        ProtectedContextRequest(
            end_index=end_index,
            mode=request_mode,
            warn_chars=warn_chars,
            summary=summary,
            captured_before_commit=captured_end is not None,
        ),
    )
    if not result.applied:
        logger.debug("[lnr] protected EDA prefix marker failed: %s", result.reason)
        return {}
    info = dict(result.info)
    effective_end_index = result.protected_end_index
    if captured_end is not None:
        self._s01_eda_prefix_end_index = effective_end_index
    self._jsonl(
        "lhr_context_events.jsonl",
        {
            "event": "protected_eda_prefix_set",
            "stage_id": "S01",
            **info,
        },
    )
    chars = int(info.get("chars") or 0)
    original_chars = int(info.get("original_chars") or 0)
    if original_chars:
        logger.info(
            "[lnr] protected EDA mode=%s fixed_chars=%d original_chars=%d original_messages=%s",
            info.get("mode"),
            chars,
            original_chars,
            info.get("original_message_count"),
        )
    if warn_chars > 0 and chars > warn_chars:
        logger.warning(
            "[lnr] protected EDA fixed prefix is %d chars, exceeding warning threshold %d; keeping it verbatim",
            chars,
            warn_chars,
        )
    return dict(info)


def _capture_s01_eda_prefix_end(self, *, stage_id: str) -> None:
    if str(stage_id or "").upper() != "S01":
        return
    if getattr(self, "_s01_eda_prefix_end_index", None) is None:
        self._s01_eda_prefix_end_index = self._count_memory_records()


def _build_protected_eda_agent_prompt(self, agent: Any, *, end_index: int) -> str:
    messages = self._agent_memory_messages(agent)
    end = max(1, min(int(end_index), len(messages)))
    deterministic_facts = self._build_protected_eda_facts_summary(
        agent,
        end_index=end,
    )
    task_text = self._message_content_text(messages[0]) if messages else self.task_desc
    task_text = task_text[:6000]

    excerpts: list[str] = []
    for msg in messages[1:end]:
        role = str(getattr(msg, "role", "") or "")
        if role not in {"assistant", "tool"}:
            continue
        text = self._message_content_text(msg).strip()
        if not text:
            continue
        cap = 900 if role == "tool" else 600
        if len(text) > cap:
            text = text[: cap - 16].rstrip() + "\n... [truncated]"
        excerpts.append(f"[{role}]\n{text}")

    history_cap = 36_000
    kept_rev: list[str] = []
    used = 0
    for excerpt in reversed(excerpts):
        size = len(excerpt) + 2
        if kept_rev and used + size > history_cap:
            break
        if not kept_rev and size > history_cap:
            excerpt = excerpt[-history_cap:]
            size = len(excerpt) + 2
        kept_rev.append(excerpt)
        used += size
    history = "\n\n".join(reversed(kept_rev)) or "(No usable EDA excerpts.)"
    s01_card = (
        read_ledger(self.ledger_path).strip()[:3500]
        or "(S01 ledger entry unavailable.)"
    )

    return (
        "Create a durable EDA and first-stage foundation summary for a long-running scientific "
        "modeling agent. Return markdown only and do not call tools. Separate observed facts from "
        "hypotheses. Use only the supplied evidence for numbers; do not invent fields, metrics, "
        "files, experiments, or conclusions. This summary is context assistance, not an evaluator "
        "or metric-validity authority.\n\n"
        "Use exactly these headings:\n"
        "## Data contract\n"
        "## Observed data facts\n"
        "## Source-candidate relationship\n"
        "## Modeling implications\n"
        "## Initial stage\n"
        "## Open questions and risks\n\n"
        "Keep concrete schema, scale, distribution, data-quality, validation, feature, model, and "
        "submission-construction details that matter for future stages. Preserve uncertainty explicitly.\n\n"
        f"[TASK CONTEXT]\n{task_text}\n\n"
        f"[DETERMINISTIC FACT EXTRACT]\n{deterministic_facts}\n\n"
        f"[AUTHORITATIVE S01 STAGE CARD]\n{s01_card}\n\n"
        f"[PRE-S01 EDA EXCERPTS]\n{history}"
    )


@staticmethod
def _normalize_protected_eda_agent_summary(text: str, *, max_chars: int) -> str:
    summary = str(text or "").strip()
    summary = re.sub(r"^```(?:markdown|md|text)?\s*", "", summary, flags=re.I)
    summary = re.sub(r"\s*```$", "", summary)
    if len(summary) > max_chars:
        summary = (
            summary[: max(0, max_chars - 36)].rstrip() + "\n... [EDA summary truncated]"
        )
    return summary


async def _prepare_protected_eda_agent_summary(
    self, agent: Any, *, stage_id: str
) -> None:
    if str(stage_id or "").upper() != "S01":
        return
    mode = (
        str(getattr(self.lhr, "protected_eda_mode", "facts") or "facts").strip().lower()
    )
    if mode not in {"agent", "agent_summary"}:
        return
    llm = getattr(agent, "llm", None)
    if llm is None or not callable(getattr(llm, "ask", None)):
        return
    try:
        captured_end = getattr(self, "_s01_eda_prefix_end_index", None)
        end_index = (
            int(captured_end)
            if captured_end is not None
            else self._count_memory_records()
        )
        prompt = self._build_protected_eda_agent_prompt(agent, end_index=end_index)
        max_chars = max(
            1200,
            int(getattr(self.lhr, "protected_eda_summary_max_chars", 8000) or 8000),
        )
        timeout = max(
            1.0,
            float(getattr(self.lhr, "stage_commit_llm_timeout_sec", 180.0) or 180.0),
        )
        text = await agent.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="protected_eda_summary",
            base_messages=[],
            system_messages=[
                Message.system_message(
                    "You summarize scientific EDA and the first evaluated modeling stage. "
                    "Return grounded markdown only. Never call tools or invent evidence.",
                ),
            ],
            timeout=timeout,
            stage_id=stage_id,
            lineage_id=self._lineage_uid_prefix(),
            node_uid=self._stage_node_uid(stage_id),
        )
        summary = self._normalize_protected_eda_agent_summary(text, max_chars=max_chars)
        if not summary:
            raise ValueError("empty agent EDA summary")
        self._s01_agent_eda_summary = summary
        self._accumulate_ephemeral_tokens(agent, "stage")
        self._jsonl(
            "lhr_context_events.jsonl",
            {
                "event": "protected_eda_agent_summary_ok",
                "stage_id": "S01",
                "prompt_chars": len(prompt),
                "summary_chars": len(summary),
                "llm_correlation": llm_correlation(text),
            },
        )
    except Exception as exc:
        self._s01_agent_eda_summary = ""
        self._jsonl(
            "lhr_context_events.jsonl",
            {
                "event": "protected_eda_agent_summary_error",
                "stage_id": "S01",
                "error": str(exc),
                "fallback": "facts",
            },
        )


def _restore_protected_eda_prefix_marker(self, agent: Any) -> None:
    if not bool(getattr(self.lhr, "preserve_prefix_and_eda", True)):
        return
    snap = self.stage_snapshots.get("S01")
    if snap is None or not int(getattr(snap, "memory_cut", 0) or 0):
        return
    raw_source = getattr(snap, "source_event", {})
    source = raw_source if isinstance(raw_source, dict) else {}
    captured_end = source.get("protected_eda_end_index")
    try:
        end_index = int(captured_end)
    except (TypeError, ValueError):
        end_index = int(snap.memory_cut)
    self._s01_eda_prefix_end_index = end_index
    ctx = getattr(agent, "_memory_ctx", None)
    warn_chars = int(getattr(self.lhr, "protected_eda_warn_chars", 50_000) or 0)
    memory_service = getattr(self, "memory_service", None)
    if not isinstance(memory_service, MemoryService):
        memory_service = MemoryService(workspace_dir=self.workspace_dir)
        self.memory_service = memory_service
    result = memory_service.protected_context.restore(
        ctx,
        end_index=end_index,
        warn_chars=warn_chars,
    )
    if not result.applied:
        logger.debug("[lnr] protected EDA prefix restore failed: %s", result.reason)


def _append_traj_summary(self, *, target_stage: str, summary: str) -> None:
    text = str(summary or "").strip()
    if not text:
        return
    path = self._split_logs_dir("traj_interaction") / "traj_interaction.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"\n[estra-summary] target={str(target_stage or '').upper()}\n{text}\n"
            )
    except OSError:
        logger.debug("[lnr] traj summary append failed", exc_info=True)


def _base_stage_id(self) -> str:
    cards = parse_stage_cards(read_ledger(self.ledger_path))
    snapshots = getattr(self, "stage_snapshots", {}) or {}
    for card in cards:
        sid = normalize_stage_id(getattr(card, "stage_id", ""))
        if sid and sid in snapshots:
            return sid
    seen = [normalize_stage_id(sid) for sid in snapshots]
    seen = [sid for sid in seen if sid]
    if not seen:
        return ""
    return sorted(seen, key=lambda sid: int(sid[1:]))[0]


def _base_stage_agent_memory_dir(self) -> Path | None:
    base_stage = self._base_stage_id()
    if not base_stage:
        return None
    base = self.stage_snapshots.get(base_stage)
    if base is None:
        return None
    snapshot_path = getattr(base, "snapshot_path", None)
    if snapshot_path is None:
        return None
    for candidate in (
        Path(snapshot_path) / "logs" / "memory" / "ScienceAgent",
        Path(snapshot_path) / ".memory" / "ScienceAgent",
    ):
        if candidate.is_dir():
            return candidate
    return None


def _metric_value_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


@staticmethod
def _stage_card_override_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return str(value or "").strip()


def _stage_card_effective_overrides(
    self, cards: list[Any]
) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for card in cards:
        sid = normalize_stage_id(getattr(card, "stage_id", ""))
        if not sid:
            continue
        snap = self.stage_snapshots.get(sid)
        raw_source = getattr(snap, "source_event", {}) if snap is not None else {}
        source = raw_source if isinstance(raw_source, dict) else {}
        if not source:
            continue
        fields: dict[str, Any] = {}
        metric = source.get("metric_value")
        if metric not in (None, ""):
            fields["metric"] = self._stage_card_override_text(metric)
        lower = source.get("lower_is_better")
        if isinstance(lower, bool):
            fields["lower_is_better"] = "true" if lower else "false"
        metric_type = source.get("val_score_type") or source.get("metric_protocol")
        if metric_type not in (None, ""):
            fields["metric_type"] = self._stage_card_override_text(metric_type)
        note = (
            source.get("selection_note")
            or source.get("metric_validity_note")
            or source.get("validation_issue")
        )
        if note not in (None, ""):
            fields["metric_note"] = self._stage_card_override_text(note)
        validity = source.get("metric_validity")
        if validity not in (None, ""):
            fields["metric_validity"] = self._stage_card_override_text(validity).lower()
        eligible = source.get("selection_eligible")
        if eligible not in (None, ""):
            fields["selection_eligible"] = self._stage_card_override_text(
                eligible
            ).lower()
        reason_code = source.get("metric_validity_reason_code") or source.get(
            "metric_reason_code"
        )
        if reason_code not in (None, ""):
            fields["metric_validity_reason_code"] = self._stage_card_override_text(
                reason_code
            )
        if fields:
            overrides[sid] = fields
    return overrides


def _effective_stage_cards(self, cards: list[Any]) -> list[Any]:
    return stage_cards_with_overrides(
        cards, self._stage_card_effective_overrides(cards)
    )


def _effective_stage_cards_from_ledger(self) -> list[Any]:
    return self._effective_stage_cards(parse_stage_cards(read_ledger(self.ledger_path)))


def _stage_cards_from_snapshots(self) -> list[StageCard]:
    cards: list[StageCard] = []
    for sid in sorted(
        self.stage_snapshots, key=lambda x: normalize_stage_id(x) or str(x)
    ):
        stage_id = normalize_stage_id(sid)
        if not stage_id:
            continue
        snap = self.stage_snapshots.get(stage_id)
        raw_source = getattr(snap, "source_event", {}) if snap is not None else {}
        source = raw_source if isinstance(raw_source, dict) else {}
        metric = source.get("metric_value", getattr(snap, "metric_value", ""))
        lower = source.get("lower_is_better", getattr(snap, "lower_is_better", ""))
        lower_text = (
            self._stage_card_override_text(lower).lower()
            if lower not in (None, "")
            else ""
        )
        cards.append(
            StageCard(
                stage_id=stage_id,
                body="",
                metric=self._stage_card_override_text(metric)
                if metric not in (None, "")
                else "",
                lower_is_better=lower_text,
                metric_validity=self._stage_card_override_text(
                    source.get("metric_validity")
                ).lower(),
                selection_eligible=self._stage_card_override_text(
                    source.get("selection_eligible")
                ).lower(),
                metric_validity_reason_code=self._stage_card_override_text(
                    source.get("metric_validity_reason_code")
                    or source.get("metric_reason_code")
                ),
                brief="restorable stage snapshot",
                why="ledger card unavailable; using checkpoint metadata",
            )
        )
    return cards


def _stage_memory_view_for_prompt(
    self,
    cards: list[Any],
    *,
    target_stage: str = "",
    latest_stage: str = "",
    best_stage: str = "",
) -> str:
    if not bool(getattr(self.lhr, "stage_memory_folding_enabled", True)):
        return render_stage_cards(cards)
    budget = int(getattr(self.lhr, "stage_memory_context_budget_chars", 24000) or 24000)
    memory_service = getattr(self, "memory_service", None)
    if memory_service is None:
        memory_service = MemoryService(workspace_dir=self.workspace_dir)
        self.memory_service = memory_service
    view = memory_service.build_stage_view(
        cards,
        context_budget_chars=budget,
        rebuild_on_stale=bool(getattr(self.lhr, "stage_memory_rebuild_on_stale", True)),
        target_stage=target_stage,
        latest_stage=latest_stage,
        best_stage=best_stage,
    )
    if view.folded_stage_count or view.summary_ids:
        self._jsonl(
            "lhr_stage_memory_events.jsonl",
            {
                "event": "stage_memory_view_built",
                "raw_chars": view.raw_chars,
                "view_chars": view.view_chars,
                "folded_stage_count": view.folded_stage_count,
                "summary_ids": list(view.summary_ids),
                "verification_stage_ids": list(view.verification_stage_ids),
                "reused_summary_count": view.reused_summary_count,
                "created_summary_count": view.created_summary_count,
            },
        )
    return view.text
