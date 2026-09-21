# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Typed candidate collection and stage experiment-state projection."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.memory_projection import (
    _metric_value_float,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import _csv_bool
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    LHR_STAGE_PERFORMANCE_CSV,
    logger,
    normalize_stage_id,
    parse_stage_cards,
    read_ledger,
)


def _stage_commit_one_line(value: Any, *, max_chars: int | None = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("###", "stage")
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[: max(0, max_chars - 3)].rstrip() + "..."
    return text


def _stage_commit_query_state(metric_event: dict[str, Any]) -> dict[str, Any]:
    """Return only structured query counters already present in the event."""

    state: dict[str, Any] = {}
    extra = metric_event.get("extra")
    sources = [metric_event, extra] if isinstance(extra, dict) else [metric_event]
    aliases = {
        "queries_remaining": ("queries_remaining", "query_remaining"),
        "queries_used": ("queries_used", "query_used"),
        "query_limit": ("query_limit", "max_queries"),
    }
    for output_key, input_keys in aliases.items():
        for source in sources:
            for input_key in input_keys:
                value = source.get(input_key)
                if value not in (None, ""):
                    state[output_key] = value
                    break
            if output_key in state:
                break
    return state


@dataclass(slots=True)
class CandidateMetrics:
    """Deduplicated eligible metrics plus rejected global candidate IDs."""

    values: list[tuple[str, float]] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    rejected: set[str] = field(default_factory=set)

    def add(self, label: str, metric: Any) -> None:
        value = _metric_value_float(metric)
        normalized = str(label or "").strip()
        if value is None or not normalized or normalized in self.seen:
            return
        self.seen.add(normalized)
        self.values.append((normalized, value))

    def replace(self, label: str, metric: Any) -> None:
        self.values = [item for item in self.values if item[0] != label]
        self.seen.discard(label)
        self.add(label, metric)


def _history_candidate_label(row: dict[str, Any], worker_id: str) -> str:
    return str(
        row.get("candidate_id")
        or row.get("node_uid")
        or ":".join(
            part
            for part in (
                worker_id,
                str(row.get("lineage_id") or "").strip(),
                str(row.get("stage_id") or "").strip(),
            )
            if part
        )
        or ""
    ).strip()


def _history_row_eligible(row: dict[str, Any]) -> bool:
    return (
        _csv_bool(row.get("validation_ok"), default=True) is not False
        and str(row.get("metric_validity") or "").strip().lower() != "low"
        and _csv_bool(row.get("selection_eligible"), default=True) is not False
    )


def _collect_history_candidates(owner: Any, candidates: CandidateMetrics) -> None:
    global_log_dir = Path(getattr(owner, "global_log_dir", owner.log_dir))
    path = global_log_dir / LHR_STAGE_PERFORMANCE_CSV
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        logger.debug("[lnr] experiment state could not read stage history", exc_info=True)
        return
    current_worker = owner._worker_uid_prefix()
    for row in rows:
        worker_id = str(row.get("worker_id") or "W00").strip() or "W00"
        if worker_id != current_worker:
            continue
        label = _history_candidate_label(row, worker_id)
        if not _history_row_eligible(row):
            candidates.rejected.add(label)
            continue
        candidates.add(label, row.get("metric_value"))


def _card_eligible(card: Any, source: dict[str, Any]) -> bool:
    return (
        _metric_value_float(card.metric) is not None
        and str(card.metric_validity or "").strip().lower() != "low"
        and _csv_bool(getattr(card, "selection_eligible", ""), default=True)
        is not False
        and _csv_bool(source.get("validation_ok"), default=True) is not False
        and str(source.get("metric_validity") or "").strip().lower() != "low"
        and _csv_bool(source.get("selection_eligible"), default=True) is not False
    )


def _collect_card_candidates(
    owner: Any,
    cards: list[Any],
    candidates: CandidateMetrics,
) -> None:
    snapshots = getattr(owner, "stage_snapshots", {})
    for card in cards:
        snapshot = snapshots.get(card.stage_id)
        source = (
            snapshot.source_event
            if snapshot is not None and isinstance(snapshot.source_event, dict)
            else {}
        )
        if not _card_eligible(card, source):
            continue
        label = owner._snapshot_node_uid(snapshot) or card.stage_id
        if label not in candidates.rejected:
            candidates.add(label, card.metric)


def _best_candidate(
    candidates: CandidateMetrics,
    *,
    lower_is_better: bool,
) -> tuple[str, str]:
    if not candidates.values:
        return "unknown", "unknown"
    chooser = min if lower_is_better else max
    stage, metric = chooser(candidates.values, key=lambda item: item[1])
    return stage, f"{metric:.12g}"


def _build_stage_commit_experiment_state(
    self: Any,
    *,
    stage_id: str,
    metric_event: dict[str, Any],
) -> str:
    target = normalize_stage_id(stage_id) or str(stage_id or "").strip().upper()
    cards = parse_stage_cards(read_ledger(self.ledger_path))
    current_metric = _metric_value_float(metric_event.get("metric_value"))
    lower_is_better = metric_event.get("lower_is_better")
    if not isinstance(lower_is_better, bool):
        lower_is_better = getattr(self, "_task_metric_lower_is_better", False)

    candidates = CandidateMetrics()
    _collect_history_candidates(self, candidates)
    _collect_card_candidates(self, cards, candidates)
    current_validity = str(metric_event.get("metric_validity") or "").strip().lower()
    current_eligible = metric_event.get("selection_eligible") is not False
    if current_metric is not None and current_validity != "low" and current_eligible:
        candidates.replace(self._stage_node_uid(target), current_metric)
    best_stage, best_metric_text = _best_candidate(
        candidates,
        lower_is_better=bool(lower_is_better),
    )
    latest_metric_text = (
        f"{current_metric:.12g}" if current_metric is not None else "unknown"
    )
    lines = [
        "EXPERIMENT_STATE",
        f"latest_stage: {target}",
        f"latest_metric: {latest_metric_text}",
        f"global_best_stage: {best_stage}",
        f"global_best_metric: {best_metric_text}",
    ]
    lines.extend(
        f"{key}: {value}"
        for key, value in _stage_commit_query_state(metric_event).items()
    )
    lines.append("recent_route_results:")
    for card in cards[-3:]:
        method = _stage_commit_one_line(
            card.brief or card.why or "method not recorded",
            max_chars=180,
        )
        lines.append(f"- {card.stage_id}: metric={card.metric or 'unknown'}; {method}")
    if not any(card.stage_id == target for card in cards):
        lines.append(
            f"- {target}: metric={latest_metric_text}; current stage pending summary"
        )
    return "\n".join(lines)


__all__ = [
    "_build_stage_commit_experiment_state",
    "_stage_commit_one_line",
    "_stage_commit_query_state",
]
