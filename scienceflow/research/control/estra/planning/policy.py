# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Normalize EStra decisions without mutating runtime state."""

from __future__ import annotations

import re
from typing import Any

from scienceflow.foundation.contracts import EstraContext, EstraDecision


def axes_from_action(action: str) -> tuple[str, str]:
    normalized = str(action or "").strip()
    if normalized == "switch_stage":
        return "previous_stage", "continue"
    if normalized == "keep_but_redirect":
        return "current_workspace", "redirect"
    return "current_workspace", "continue"


def action_from_axes(startpoint: str, intent: str) -> str:
    if str(startpoint or "").strip() == "previous_stage":
        return "switch_stage"
    if str(intent or "").strip() == "redirect":
        return "keep_but_redirect"
    return "keep_current"


def decision_kind(*, action: str, startpoint: str, intent: str) -> str:
    if action == "switch_stage" or startpoint == "previous_stage":
        return "switch"
    if action == "keep_but_redirect" or intent == "redirect":
        return "redirect"
    if action == "keep_current":
        return "continue"
    return "invalid"


def normalize_startpoint(value: Any, *, action: str = "") -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {
        "previous_stage",
        "historical_stage",
        "switch_stage",
        "restore_stage",
        "stage",
    }:
        return "previous_stage"
    if raw in {"current_workspace", "current", "latest", "keep_current", "workspace"}:
        return "current_workspace"
    return axes_from_action(action)[0]


def normalize_intent(value: Any, *, action: str = "") -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"redirect", "refocus", "change_tactic", "change_focus"}:
        return "redirect"
    if raw in {"continue", "keep", "deepen", "resume"}:
        return "continue"
    return axes_from_action(action)[1]


def derive_compact(*, action: str, trigger_source: str) -> bool:
    if str(action or "").strip() in {
        "keep_current",
        "keep_but_redirect",
        "switch_stage",
    }:
        return True
    return str(trigger_source or "").strip() in {
        "context_limit",
        "text_only",
        "context_hygiene",
    }


def compact_event_text(text: str, *, max_chars: int) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 1].rstrip() + "…"


def decision_fields(parsed: dict[str, Any]) -> dict[str, str]:
    names = (
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
    for name in names:
        value = parsed.get(name)
        if value is None:
            continue
        compact = compact_event_text(str(value), max_chars=240)
        if compact:
            fields[name] = compact
    return fields


def normalize_decision(
    parsed: dict[str, Any], *, context: EstraContext
) -> EstraDecision | None:
    latest = str(context.latest_stage or "").upper()
    legacy_action = str(parsed.get("action") or "").strip()
    if not legacy_action and "startpoint" not in parsed and "intent" not in parsed:
        return None
    startpoint = normalize_startpoint(parsed.get("startpoint"), action=legacy_action)
    intent = normalize_intent(parsed.get("intent"), action=legacy_action)
    action = action_from_axes(startpoint, intent)
    target = str(
        parsed.get("target_stage") or parsed.get("target_step") or ""
    ).strip().upper()
    if action == "switch_stage":
        if target not in context.switch_candidates:
            return None
    elif action in {"keep_current", "keep_but_redirect"}:
        target = latest
    else:
        return None
    fields = decision_fields({**parsed, "startpoint": startpoint, "intent": intent})
    reason = fields.get("decision_reason") or str(parsed.get("reason") or "").strip()
    parts = [reason] if reason else []
    if fields.get("bottleneck"):
        parts.append(f"bottleneck: {fields['bottleneck']}")
    if fields.get("evidence"):
        parts.append(f"evidence: {fields['evidence']}")
    return EstraDecision(
        action=action,
        startpoint=startpoint,
        intent=intent,
        target_stage=target,
        compact=derive_compact(
            action=action, trigger_source=context.trigger_source
        ),
        reason=compact_event_text(" | ".join(parts), max_chars=360),
        diagnostics=fields,
    )


__all__ = [
    "action_from_axes",
    "axes_from_action",
    "decision_fields",
    "decision_kind",
    "derive_compact",
    "normalize_decision",
    "normalize_intent",
    "normalize_startpoint",
]
