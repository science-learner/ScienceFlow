# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Narrow adapter for protected InquiryCraft context-prefix capabilities."""

from __future__ import annotations

import json
import re
from typing import Any

from scienceflow.research.state.knowledge.memory.core.contracts import (
    ProtectedContextRequest,
    ProtectedContextResult,
)


_EDA_FACT_RE = re.compile(
    r"(train|test|valid|validation|sample|submission|shape|rows|columns|dtype|"
    r"missing|null|nan|unique|target|spacegroup|metric|rmsle|id,|percent_|"
    r"lattice_|formation_energy|bandgap_energy|min|max|mean|std|corr|skew)",
    flags=re.IGNORECASE,
)
_EXPERIMENT_NOISE_RE = re.compile(
    r"(Final Validation Score|Validation RMSLE|Individual model|Total models|"
    r"RandomForest|ExtraTrees|XGBoost|LightGBM|CatBoost|MLPRegressor|Ridge|"
    r"n_estimators|max_depth|num_leaves|learning_rate|avg=|form=|band=)",
    flags=re.IGNORECASE,
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class ProtectedContextAdapter:
    """Apply a prepared protection projection without knowing solver state."""

    def apply(
        self, context: Any, request: ProtectedContextRequest
    ) -> ProtectedContextResult:
        mode = str(request.mode or "facts").strip().lower()
        end_index = max(0, int(request.end_index or 0))
        if mode in {"raw", "verbatim"}:
            setter = getattr(context, "set_protected_raw_prefix", None)
            if not callable(setter):
                return ProtectedContextResult(False, "capability_unavailable")
            label = "LHR protected EDA fixed prefix"
            summary_mode = "raw"
            try:
                raw_info = setter(
                    end_index,
                    warn_chars=int(request.warn_chars or 0),
                    label=label,
                )
            except Exception as exc:
                return ProtectedContextResult(
                    False, f"apply_failed:{type(exc).__name__}"
                )
        else:
            setter = getattr(
                context, "replace_protected_raw_prefix_with_summary", None
            )
            if not callable(setter):
                return ProtectedContextResult(False, "capability_unavailable")
            summary_mode = mode if mode in {"agent", "facts_fallback"} else "facts"
            label = (
                "LHR protected EDA agent summary"
                if summary_mode == "agent"
                else "LHR protected EDA facts"
            )
            try:
                raw_info = setter(
                    end_index,
                    str(request.summary or ""),
                    warn_chars=int(request.warn_chars or 0),
                    label=label,
                )
            except Exception as exc:
                return ProtectedContextResult(
                    False, f"apply_failed:{type(exc).__name__}"
                )
        info = {
            **dict(raw_info or {}),
            "mode": summary_mode,
        }
        if summary_mode != "raw":
            info["summary_chars"] = len(str(request.summary or ""))
        try:
            effective_end = int(info.get("end_index", end_index))
        except (TypeError, ValueError):
            effective_end = end_index
        info.update(
            {
                "protected_end_index": effective_end,
                "boundary_source": (
                    "pre_stage_commit"
                    if request.captured_before_commit
                    else "current_memory"
                ),
            }
        )
        return ProtectedContextResult(
            True,
            "applied",
            protected_end_index=effective_end,
            info=info,
        )

    def restore(
        self, context: Any, *, end_index: int, warn_chars: int
    ) -> ProtectedContextResult:
        return self.apply(
            context,
            ProtectedContextRequest(
                end_index=end_index,
                mode="raw",
                warn_chars=warn_chars,
                captured_before_commit=True,
            ),
        )

    @classmethod
    def build_facts_summary(
        cls,
        messages: list[Any],
        *,
        end_index: int,
        max_chars: int,
    ) -> str:
        limit = max(1200, int(max_chars or 1200))
        candidates = messages[1 : max(1, min(int(end_index), len(messages)))]
        facts: list[str] = []
        seen: set[str] = set()
        scratch_tool_call_chars = 0
        scratch_tool_call_count = 0
        for message in candidates:
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                try:
                    chars = len(json.dumps(tool_calls, ensure_ascii=False))
                except (TypeError, ValueError):
                    chars = len(str(tool_calls))
                scratch_tool_call_chars += chars
                scratch_tool_call_count += 1
            if str(getattr(message, "role", "") or "") != "tool":
                continue
            content = getattr(message, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            for raw_line in text.splitlines():
                line = _ANSI_RE.sub("", str(raw_line or ""))
                line = re.sub(r"\s+", " ", line).strip()
                if len(line) > 220:
                    line = line[:217].rstrip() + "..."
                if not cls._looks_like_fact(line):
                    continue
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                facts.append(line)
                if sum(len(item) + 3 for item in facts) >= limit - 700 or len(facts) >= 36:
                    break
            if sum(len(item) + 3 for item in facts) >= limit - 700 or len(facts) >= 36:
                break
        lines = [
            "Fixed EDA facts retained from the first successful stage.",
            "Scratch commands, temporary code, full heredocs, and model-probe tool-call payloads were removed from the fixed prefix; raw audit remains in `.logs/interaction/`, `.logs/traj_interaction/`, and stage snapshots.",
        ]
        if scratch_tool_call_count:
            lines.append(
                "Removed raw pre-S01 scratch payloads: "
                f"{scratch_tool_call_count} assistant tool-call message(s), "
                f"about {scratch_tool_call_chars} serialized chars."
            )
        if facts:
            lines.append("EDA facts:")
            lines.extend(f"- {fact}" for fact in facts)
        else:
            lines.append(
                "EDA facts: no deterministic fact lines were confidently extracted; "
                "inspect `dataset/` or `.logs/` if a specific early detail is needed."
            )
        summary = "\n".join(lines).strip()
        if len(summary) > limit:
            summary = summary[: limit - 38].rstrip() + "\n... [EDA facts truncated]"
        return summary

    @staticmethod
    def _looks_like_fact(line: str) -> bool:
        if not line or len(line) < 4:
            return False
        if line.startswith("[exit=") or line.startswith("[bash:") or line.startswith(
            "[stream"
        ):
            return False
        if _EXPERIMENT_NOISE_RE.search(line):
            return False
        return bool(_EDA_FACT_RE.search(line))


__all__ = ["ProtectedContextAdapter"]
