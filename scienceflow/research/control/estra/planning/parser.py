# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Parse structured and legacy free-text EStra responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def compact_reason(text: str, *, max_chars: int = 240) -> str:
    raw = re.sub(r"<[^>]+>", " ", text or "")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:max_chars]


def parse_estra_decision(
    text: str, *, switch_candidates: list[str]
) -> dict[str, Any]:
    parsed = parse_json_object(text)
    allowed = {str(candidate).upper() for candidate in switch_candidates}
    if parsed:
        action = str(parsed.get("action") or "").strip()
        target = str(
            parsed.get("target_stage") or parsed.get("target_step") or ""
        ).strip().upper()
        if action in {"keep_current", "keep_but_redirect"}:
            return {**parsed, "action": action}
        if action == "switch_stage" and target in allowed:
            return {**parsed, "action": "switch_stage", "target_stage": target}
        return parsed
    raw = (text or "").strip()
    if not raw:
        return {}
    reason = compact_reason(raw)
    switch_patterns = (
        r"(?:switch|estra|return|restore|rollback|go\s+back|back)\s+(?:to\s+)?(S\d{1,4})",
        r"(?:切到|切换到|回到|恢复到|转向)\s*(S\d{1,4})",
    )
    for pattern in switch_patterns:
        for match in re.finditer(pattern, raw, flags=re.I):
            target = match.group(1).upper()
            if target in allowed:
                return {
                    "action": "switch_stage",
                    "target_stage": target,
                    "reason": reason,
                }
    if re.search(r"\b(redirect|refocus)\b", raw, flags=re.I) or re.search(
        r"重新聚焦|调整方向|换个打法", raw
    ):
        return {
            "action": "keep_but_redirect",
            "reason": reason,
            "bottleneck": reason,
        }
    if re.search(
        r"\b(keep|continue|stay)\b.*\b(current|latest|trajectory|stage)\b",
        raw,
        flags=re.I,
    ) or re.search(r"保持|继续当前|不切换", raw):
        return {"action": "keep_current", "reason": reason}
    return {}


__all__ = ["compact_reason", "parse_estra_decision", "parse_json_object"]
