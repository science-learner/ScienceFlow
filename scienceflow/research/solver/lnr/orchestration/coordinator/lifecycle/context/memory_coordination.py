# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: protected context, stage memory views, and memory rebuilds.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.memory_projection import (
    _metric_value_float,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.coordination import _compact_event_text
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    MemoryCompactionRequest,
    MemoryService,
    Path,
    build_estra_resume_prompt,
    build_keep_current_compact_prompt,
    logger,
    re,
)


@staticmethod
def _stage_card_one_line(text: str, *, max_chars: int) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max(0, max_chars - 24)].rstrip() + " ... [truncated]"


def _best_candidate_stage_id(self, cards: list[Any], candidates: list[str]) -> str:
    allowed = {str(sid).upper() for sid in candidates}
    best_stage = ""
    best_metric: float | None = None
    best_lower = True
    for card in cards:
        sid = str(getattr(card, "stage_id", "") or "").upper()
        if sid not in allowed:
            continue
        metric = _metric_value_float(getattr(card, "metric", ""))
        if metric is None:
            continue
        lower_raw = str(getattr(card, "lower_is_better", "") or "").strip().lower()
        lower = lower_raw not in {"false", "0", "no"}
        if best_metric is None:
            best_stage = sid
            best_metric = metric
            best_lower = lower
            continue
        if (lower and metric < best_metric) or ((not lower) and metric > best_metric):
            best_stage = sid
            best_metric = metric
            best_lower = lower
    _ = best_lower
    return best_stage


def _safe_estra_fallback_decision(
    self,
    *,
    cards: list[Any],
    candidates: list[str],
    latest: str,
    trigger_source: str,
    reason: str,
) -> dict[str, Any]:
    latest = str(latest or "").upper()
    candidates = [str(sid or "").upper() for sid in candidates]
    _ = cards
    action = "keep_current"
    target = latest or (candidates[-1] if candidates else "")
    startpoint, intent = self._estra_axes_from_action(action)
    return {
        "action": action,
        "startpoint": startpoint,
        "intent": intent,
        "target_stage": target,
        "compact": self._derive_estra_compact(
            action=action, trigger_source=trigger_source
        ),
        "reason": reason,
    }


def _build_lhr_state_packet(
    self,
    *,
    action: str,
    target_stage: str,
    terminal_stage: str,
    reason: str,
    tail_summary: str = "",
    estra_fields: dict[str, Any] | None = None,
) -> str:
    cards = self._effective_stage_cards_from_ledger()
    if not cards:
        cards = self._stage_cards_from_snapshots()
    candidates = [c.stage_id for c in cards if c.stage_id in self.stage_snapshots]
    latest = str(terminal_stage or (cards[-1].stage_id if cards else "")).upper()
    target = str(target_stage or "").upper()
    best = self._best_candidate_stage_id(cards, candidates)
    max_chars = int(getattr(self.lhr, "state_packet_max_chars", 12000) or 12000)
    max_branch_chars = int(
        getattr(self.lhr, "state_packet_archived_branch_max_chars", 1500) or 1500
    )
    summary = str(tail_summary or "").strip()
    estra_fields = estra_fields or {}
    startpoint = str(
        estra_fields.get("startpoint") or self._estra_axes_from_action(action)[0]
    )
    intent = str(estra_fields.get("intent") or self._estra_axes_from_action(action)[1])
    if len(summary) > max_branch_chars:
        summary = (
            summary[: max(0, max_branch_chars - 40)].rstrip()
            + "\n... [branch summary truncated]"
        )

    def render(target_cards: list[Any], *, include_why: bool = True) -> str:
        lines = [
            "## LHR State Packet v2",
            f"ESTRA action: {str(action or '').strip() or 'keep_current'}",
            f"ESTRA startpoint: {startpoint or '(unknown)'}",
            f"ESTRA intent: {intent or '(unknown)'}",
            f"ESTRA target stage: {target or '(none)'}",
            f"Terminal stage before estra: {latest or '(none)'}",
            f"Best known restorable stage: {best or '(unknown)'}",
            f"ESTRA reason: {_compact_event_text(reason, max_chars=360)}",
        ]
        for label, key in (
            ("Exploration summary", "exploration_summary"),
            ("ESTRA bottleneck", "bottleneck"),
            ("ESTRA evidence", "evidence"),
            ("Missing evidence", "missing_evidence"),
            ("Redirect focus", "redirect_focus"),
        ):
            value = _compact_event_text(
                str(estra_fields.get(key) or ""), max_chars=240
            )
            if value:
                lines.append(f"{label}: {value}")
        if include_why:
            checkpoint_context = self._estra_stage_checkpoint_context(candidates)
            if checkpoint_context:
                lines.extend(["", "## Stage Checkpoint Metadata", checkpoint_context])
            lines.append(
                self._stage_memory_view_for_prompt(
                    target_cards,
                    target_stage=target,
                    latest_stage=latest,
                    best_stage=best,
                )
            )
        else:
            lines.extend(["", "## Active Stage Cards (shrunk)"])
            for card in target_cards:
                sid = str(getattr(card, "stage_id", "") or "").upper()
                brief = self._stage_card_one_line(
                    getattr(card, "brief", ""), max_chars=160
                )
                lines.append(
                    f"- {sid} | metric={getattr(card, 'metric', '') or 'unknown'} | BRIEF={brief}"
                )
        if summary:
            branch_label = (
                "## Historical Exploration Summary (abandoned branch evidence)"
                if str(action or "") == "switch_stage"
                else "## Current Branch Compact Summary"
            )
            lines.extend(["", branch_label, summary])
        lines.extend(
            [
                "",
                "## Workspace State",
                "Workspace files are the source of truth. Inspect files before editing.",
                "Tracked source globs: *.py, *.md",
                "Reusable layout preference: train.py / predict.py / util.py when useful.",
                self._workspace_state_deliverable_line(),
            ]
        )
        return "\n".join(lines).strip()

    packet = render(cards)
    if len(packet) <= max_chars:
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "state_packet_built",
                "action": action,
                "target_stage": target,
                "terminal_stage": latest,
                "best_stage": best,
                "stage_count": len(cards),
                "candidate_count": len(candidates),
                "chars": len(packet),
                "shrink_level": 0,
            },
        )
        return packet

    base_stage = self._base_stage_id()
    keep_ids = {sid for sid in (base_stage, target, latest, best) if sid}
    for card in cards[-3:]:
        keep_ids.add(str(getattr(card, "stage_id", "") or "").upper())
    shrunk_cards = [
        card
        for card in cards
        if str(getattr(card, "stage_id", "") or "").upper() in keep_ids
    ]
    packet = render(shrunk_cards)
    shrink_level = 1
    if len(packet) > max_chars:
        packet = render(shrunk_cards, include_why=False)
        shrink_level = 2
    if len(packet) > max_chars:
        packet = (
            packet[: max(0, max_chars - 44)].rstrip()
            + "\n... [state packet truncated to fit budget]"
        )
        shrink_level = 3
    self._jsonl(
        "lhr_estras.jsonl",
        {
            "event": "state_packet_built",
            "action": action,
            "target_stage": target,
            "terminal_stage": latest,
            "best_stage": best,
            "stage_count": len(cards),
            "candidate_count": len(candidates),
            "chars": len(packet),
            "shrink_level": shrink_level,
        },
    )
    return packet


def _rebuild_memory_after_estra(
    self,
    *,
    target_stage: str,
    summary: str,
    state_packet: str = "",
    strict_context_limit: bool = False,
) -> tuple[bool, int]:
    if not bool(getattr(self.lhr, "estra_compact_enabled", True)):
        return False, 0
    base_stage = self._base_stage_id()
    base_dir = self._base_stage_agent_memory_dir()
    if base_dir is None:
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_memory_compact_skipped",
                "reason": f"missing_{(base_stage or 'base_stage').lower()}_memory",
                "base_stage": base_stage,
            },
        )
        return False, 0
    next_stage = self._next_stage_id_for_logging()
    prompt = build_estra_resume_prompt(
        target_stage=str(target_stage or "").upper(),
        next_stage=next_stage,
        ledger_filename=self.ledger_filename,
        base_stage=base_stage,
        compact_summary=summary,
        state_packet=state_packet,
        parallel_worker_snapshot=self._parallel_worker_snapshot_for_prompt(),
        resource_context=self._resource_context_for_prompt(),
    )
    dest = self.memory_dir / "ScienceAgent"
    memory_service = getattr(self, "memory_service", None)
    if not isinstance(memory_service, MemoryService):
        memory_service = MemoryService(workspace_dir=self.workspace_dir)
        self.memory_service = memory_service
    result = memory_service.compact(
        MemoryCompactionRequest(
            source_dir=str(base_dir),
            destination_dir=str(dest),
            prompt=prompt,
            max_messages=int(getattr(self.cfg, "max_messages", 100) or 100),
            strict_context_limit=strict_context_limit,
            metadata={"mode": "switch_stage", "base_stage": base_stage},
        )
    )
    if not result.applied:
        if result.reason == "empty_source_memory":
            self._jsonl(
                "lhr_estras.jsonl",
                {
                    "event": "estra_memory_compact_skipped",
                    "reason": f"empty_{(base_stage or 'base_stage').lower()}_memory",
                    "base_stage": base_stage,
                },
            )
        else:
            logger.debug("[lnr] estra memory compact failed: %s", result.reason)
        return False, 0
    record_count = result.record_count
    context_preview = {
        "event": "estra_context_prepared",
        "target_stage": str(target_stage or "").upper(),
        "next_stage": next_stage,
        "base_stage": base_stage,
        "memory_records": record_count,
        "compact_strength": "strict_context_limit"
        if strict_context_limit
        else "normal",
        "context_shape": [
            (
                f"{base_stage or 'base-stage'} minimal task prefix records"
                if strict_context_limit
                else f"{base_stage or 'base-stage'} prefix-safe memory records"
            ),
            "deterministic LHR state packet",
            "one estra resume user prompt",
            "compact tail summary from skipped stages",
        ],
        "resume_prompt_excerpt": _compact_event_text(prompt, max_chars=900),
        "tail_summary_excerpt": _compact_event_text(summary, max_chars=900),
        "summary_chars": len(str(summary or "")),
        "state_packet_chars": len(str(state_packet or "")),
    }
    self._jsonl("lhr_estras.jsonl", context_preview)
    self._jsonl(
        "lhr_estras.jsonl",
        {
            "event": "estra_memory_compacted",
            "target_stage": str(target_stage or "").upper(),
            "base_stage": base_stage,
            "records": record_count,
            "compact_strength": "strict_context_limit"
            if strict_context_limit
            else "normal",
            "summary_chars": len(str(summary or "")),
            "state_packet_chars": len(str(state_packet or "")),
            "next_stage": next_stage,
        },
    )
    return True, record_count


def _rebuild_memory_after_keep_current(
    self,
    *,
    terminal_stage: str,
    summary: str,
    state_packet: str = "",
    strict_context_limit: bool = False,
    continuation_action: str = "keep_current",
    estra_fields: dict[str, Any] | None = None,
) -> tuple[bool, int]:
    if not bool(getattr(self.lhr, "estra_compact_enabled", True)):
        return False, 0
    base_stage = self._base_stage_id()
    base_dir = self._base_stage_agent_memory_dir()
    if base_dir is None:
        self._jsonl(
            "lhr_estras.jsonl",
            {
                "event": "estra_keep_current_memory_compact_skipped",
                "reason": f"missing_{(base_stage or 'base_stage').lower()}_memory",
                "base_stage": base_stage,
            },
        )
        return False, 0
    next_stage = self._next_stage_id_for_logging()
    fields = estra_fields or {}
    prompt = build_keep_current_compact_prompt(
        terminal_stage=str(terminal_stage or "").upper(),
        next_stage=next_stage,
        compact_summary=summary,
        base_stage=base_stage,
        state_packet=state_packet,
        parallel_worker_snapshot=self._parallel_worker_snapshot_for_prompt(),
        resource_context=self._resource_context_for_prompt(),
        continuation_action=continuation_action,
        exploration_summary=str(fields.get("exploration_summary") or ""),
        bottleneck=str(fields.get("bottleneck") or ""),
        evidence=str(fields.get("evidence") or ""),
        missing_evidence=str(fields.get("missing_evidence") or ""),
        decision_reason=str(fields.get("decision_reason") or ""),
        redirect_focus=str(fields.get("redirect_focus") or ""),
    )
    dest = self.memory_dir / "ScienceAgent"
    memory_service = getattr(self, "memory_service", None)
    if not isinstance(memory_service, MemoryService):
        memory_service = MemoryService(workspace_dir=self.workspace_dir)
        self.memory_service = memory_service
    result = memory_service.compact(
        MemoryCompactionRequest(
            source_dir=str(base_dir),
            destination_dir=str(dest),
            prompt=prompt,
            max_messages=int(getattr(self.cfg, "max_messages", 100) or 100),
            strict_context_limit=strict_context_limit,
            metadata={"mode": continuation_action, "base_stage": base_stage},
        )
    )
    if not result.applied:
        if result.reason == "empty_source_memory":
            self._jsonl(
                "lhr_estras.jsonl",
                {
                    "event": "estra_keep_current_memory_compact_skipped",
                    "reason": f"empty_{(base_stage or 'base_stage').lower()}_memory",
                    "base_stage": base_stage,
                },
            )
        else:
            logger.debug(
                "[lnr] terminal continue memory compact failed: %s",
                result.reason,
            )
        return False, 0
    record_count = result.record_count
    self._jsonl(
        "lhr_estras.jsonl",
        {
            "event": "estra_keep_current_context_prepared",
            "action": continuation_action,
            "terminal_stage": str(terminal_stage or "").upper(),
            "next_stage": next_stage,
            "base_stage": base_stage,
            "memory_records": record_count,
            "compact_strength": "strict_context_limit"
            if strict_context_limit
            else "normal",
            "context_shape": [
                (
                    f"{base_stage or 'base-stage'} minimal task prefix records"
                    if strict_context_limit
                    else f"{base_stage or 'base-stage'} prefix-safe memory records"
                ),
                "deterministic LHR state packet",
                "one terminal-continue user prompt",
                "compact current trajectory summary",
            ],
            "resume_prompt_excerpt": _compact_event_text(prompt, max_chars=900),
            "tail_summary_excerpt": _compact_event_text(summary, max_chars=900),
            "summary_chars": len(str(summary or "")),
            "state_packet_chars": len(str(state_packet or "")),
        },
    )
    self._jsonl(
        "lhr_estras.jsonl",
        {
            "event": "estra_keep_current_memory_compacted",
            "action": continuation_action,
            "terminal_stage": str(terminal_stage or "").upper(),
            "base_stage": base_stage,
            "records": record_count,
            "compact_strength": "strict_context_limit"
            if strict_context_limit
            else "normal",
            "summary_chars": len(str(summary or "")),
            "state_packet_chars": len(str(state_packet or "")),
            "next_stage": next_stage,
        },
    )
    return True, record_count


def _prepare_dataset_symlink(self) -> None:
    inp = Path(self.cfg.input_data_dir).expanduser().resolve(strict=False)
    if not str(inp).strip() or not inp.exists():
        return
    ws_dataset = self.workspace_dir / "dataset"
    if ws_dataset.exists() or ws_dataset.is_symlink():
        return
    from scienceflow.research.solver.lnr.lifecycle.workspace.prep_fs import (
        prepare_workspace_dataset_flat,
        resolve_workspace_dataset_source,
    )

    source, _layout = resolve_workspace_dataset_source(inp)
    prepare_workspace_dataset_flat(source, ws_dataset)
    resolved = inp.resolve()
    roots = list(getattr(self.cfg, "path_guard_extra_roots", None) or [])
    if resolved not in {Path(p).expanduser().resolve(strict=False) for p in roots}:
        roots.append(resolved)
        self.cfg.path_guard_extra_roots = roots
