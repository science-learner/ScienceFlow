# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: dataset setup, metric extraction, score/resource context, and route evidence.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.metric_projection import (
    _csv_bool,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    LHR_STAGE_PERFORMANCE_CSV,
    Path,
    PeerRouteEvidence,
    StageSnapshot,
    _build_llm,
    _deterministic_gate_enabled,
    build_peer_route_evidence_from_csv,
    csv,
    json,
    metric_lower_is_better_hint,
    re,
)


from scienceflow.research.solver.lnr.orchestration.coordinator.resource.peer_snapshot import (
    _parallel_worker_snapshot_for_prompt as _parallel_worker_snapshot_for_prompt,
)

def _estra_peer_route_evidence(self) -> PeerRouteEvidence:
    if not bool(getattr(self.lhr, "estra_reflection_prompt_enabled", True)):
        return PeerRouteEvidence(text="")
    if not bool(getattr(self.lhr, "estra_peer_evidence_enabled", True)):
        return PeerRouteEvidence(text="")
    worker_count = max(1, int(getattr(self, "worker_count", 1) or 1))
    if worker_count <= 1:
        return PeerRouteEvidence(text="")
    path = (
        Path(getattr(self, "global_log_dir", getattr(self, "log_dir", Path("."))))
        / LHR_STAGE_PERFORMANCE_CSV
    )
    return build_peer_route_evidence_from_csv(
        path,
        current_worker_id=self._worker_uid_prefix(),
        max_chars=max(
            400, int(getattr(self.lhr, "estra_peer_evidence_max_chars", 1400) or 1400)
        ),
        min_delta_ratio=float(
            getattr(self.lhr, "estra_peer_evidence_min_delta_ratio", 0.0) or 0.0
        ),
    )


def _estra_backtrack_reflection(
    self,
    switch_candidates: list[str],
    *,
    latest_stage: str,
) -> tuple[str, dict[str, Any]]:
    if not bool(getattr(self.lhr, "estra_reflection_prompt_enabled", True)):
        return "", {
            "backtrack_reflection_present": False,
            "backtrack_candidate_count": 0,
        }
    if not bool(getattr(self.lhr, "estra_backtrack_reflection_enabled", True)):
        return "", {
            "backtrack_reflection_present": False,
            "backtrack_candidate_count": 0,
        }
    candidates = [
        str(sid or "").strip().upper()
        for sid in switch_candidates
        if str(sid or "").strip()
    ]
    ranked: list[tuple[float, str, StageSnapshot, str]] = []
    latest_snap = self.stage_snapshots.get(str(latest_stage or "").strip().upper())
    lower = True
    latest_lower = (
        getattr(latest_snap, "lower_is_better", None)
        if latest_snap is not None
        else None
    )
    if latest_lower is not None:
        lower = bool(latest_lower)
    for sid in candidates:
        snap = self.stage_snapshots.get(sid)
        metric_value = getattr(snap, "metric_value", None) if snap is not None else None
        if snap is None or metric_value is None:
            continue
        raw_source = getattr(snap, "source_event", {})
        source = raw_source if isinstance(raw_source, dict) else {}
        if _csv_bool(source.get("validation_ok"), default=True) is False:
            continue
        if str(source.get("metric_validity") or "").strip().lower() == "low":
            continue
        selection_eligible = source.get("selection_eligible")
        if (
            selection_eligible not in (None, "")
            and _csv_bool(selection_eligible, default=True) is False
        ):
            continue
        snap_lower = getattr(snap, "lower_is_better", None)
        if snap_lower is not None:
            lower = bool(snap_lower)
        score = float(metric_value) if lower else -float(metric_value)
        status_bits = []
        metric_validity = str(source.get("metric_validity") or "").strip()
        if metric_validity:
            status_bits.append(f"metric_validity={metric_validity}")
        if selection_eligible not in (None, ""):
            status_bits.append(
                f"selection_eligible={self._stage_card_override_text(selection_eligible)}"
            )
        if source.get("submission_sha"):
            status_bits.append("submission_ready=True")
        if source.get("source_commit_sha"):
            status_bits.append(f"commit={str(source.get('source_commit_sha'))[:12]}")
        ranked.append((score, sid, snap, ",".join(status_bits) or "metric-backed"))
    ranked.sort(key=lambda item: item[0])
    max_chars = max(
        400,
        int(getattr(self.lhr, "estra_backtrack_reflection_max_chars", 1000) or 1000),
    )
    meta: dict[str, Any] = {
        "backtrack_reflection_present": False,
        "backtrack_candidate_count": len(ranked),
        "backtrack_best_stage": ranked[0][1] if ranked else "",
        "backtrack_best_metric": ranked[0][2].metric_value if ranked else None,
    }
    if not ranked:
        return "", meta
    lines = [
        "- Previous-stage restore is a normal research action when an older stage is a cleaner base.",
    ]
    if latest_snap is not None and latest_snap.metric_value is not None:
        lines.append(
            f"- Current/latest {latest_stage}: metric={latest_snap.metric_value}."
        )
    for _score, sid, snap, status in ranked[:3]:
        source = snap.source_event if isinstance(snap.source_event, dict) else {}
        method = self._compact_peer_method(
            str(source.get("brief") or source.get("why") or ""), max_chars=120
        )
        lines.append(
            f"- Candidate {sid}: metric={snap.metric_value}; status={status}; method={method}"
        )
    lines.append(
        "- Compare current continue, current redirect, and previous_stage redirect before choosing."
    )
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = (
            text[: max_chars - 39].rstrip() + "\n... [backtrack reflection truncated]"
        )
    meta["backtrack_reflection_present"] = bool(text)
    return text, meta


def _next_global_stage_row_order(self, path: Path) -> int:
    if not path.is_file():
        return 1
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return sum(1 for _ in csv.DictReader(f)) + 1
    except OSError:
        return 1


def _best_stage_id_so_far(self) -> str:
    best_stage = ""
    best_metric: float | None = None
    lower = True
    for sid, snap in self.stage_snapshots.items():
        if snap.metric_value is None:
            continue
        if best_metric is None:
            best_metric = float(snap.metric_value)
            best_stage = sid
            lower = bool(snap.lower_is_better is not False)
            continue
        value = float(snap.metric_value)
        if lower and value < best_metric:
            best_metric = value
            best_stage = sid
        elif not lower and value > best_metric:
            best_metric = value
            best_stage = sid
    return best_stage


@staticmethod
def _stage_with_submission_sha(
    stage_snapshots: dict[str, StageSnapshot],
    submission_sha: str,
) -> StageSnapshot | None:
    needle = str(submission_sha or "").strip()
    if not needle:
        return None
    for _sid, snap in sorted(stage_snapshots.items()):
        source = snap.source_event if isinstance(snap.source_event, dict) else {}
        if str(source.get("submission_sha") or "").strip() == needle:
            return snap
    return None


@staticmethod
def _stage_with_artifact_sha(
    stage_snapshots: dict[str, StageSnapshot],
    artifact_sha: str,
) -> StageSnapshot | None:
    needle = str(artifact_sha or "").strip()
    if not needle:
        return None
    for _sid, snap in sorted(stage_snapshots.items()):
        source = snap.source_event if isinstance(snap.source_event, dict) else {}
        if (
            str(
                source.get("artifact_sha") or source.get("submission_sha") or ""
            ).strip()
            == needle
        ):
            return snap
    return None


@staticmethod
def _duplicate_stage_same_design(
    metric_event: dict[str, Any], duplicate: StageSnapshot
) -> bool:
    source = duplicate.source_event if isinstance(duplicate.source_event, dict) else {}
    current_solution = str(metric_event.get("solution_sha") or "").strip()
    prior_solution = str(source.get("solution_sha") or "").strip()
    current_artifact = str(
        metric_event.get("artifact_sha") or metric_event.get("submission_sha") or ""
    ).strip()
    prior_artifact = str(
        source.get("artifact_sha") or source.get("submission_sha") or ""
    ).strip()
    if current_solution and prior_solution and current_solution != prior_solution:
        return False
    if current_artifact and prior_artifact and current_artifact != prior_artifact:
        return False
    return bool(
        current_solution and prior_solution and current_artifact and prior_artifact
    )


def _metric_lower_is_better_decision(
    self, metric_event: dict[str, Any], *, fallback: Any = None
) -> dict[str, Any]:
    declared_raw = metric_event.get("declared_lower_is_better")
    if declared_raw in (None, ""):
        declared_raw = metric_event.get("lower_is_better")
    declared = _csv_bool(declared_raw, default=None)
    fallback_declared = _csv_bool(fallback, default=None)
    if declared is None:
        declared = fallback_declared
    hint_text = " ".join(
        str(metric_event.get(key) or "")
        for key in (
            "metric_name",
            "val_score_type",
            "metric_protocol",
            "metric_source_note",
        )
    )
    event_hint = metric_lower_is_better_hint(hint_text)
    task_hint = getattr(self, "_task_metric_lower_is_better", None)
    if metric_event.get("metric_authoritative") is True and declared is not None:
        lower = bool(declared)
        source = "authoritative_evaluator"
    elif task_hint is not None:
        lower = bool(task_hint)
        source = "task_description"
    elif event_hint is not None:
        lower = bool(event_hint)
        source = "metric_text_hint"
    elif declared is not None:
        lower = bool(declared)
        source = "declared"
    else:
        lower = True
        source = "default_lower"
    conflict = bool(declared is not None and bool(declared) != lower)
    if event_hint is not None and bool(event_hint) != lower:
        conflict = True
    return {
        "lower_is_better": lower,
        "metric_direction_source": source,
        "declared_lower_is_better": declared if declared is not None else "",
        "metric_direction_conflict": conflict,
    }


def _apply_metric_direction_audit(
    self, metric_event: dict[str, Any], *, fallback: Any = None
) -> dict[str, Any]:
    decision = self._metric_lower_is_better_decision(metric_event, fallback=fallback)
    metric_event.update(decision)
    return decision


def _metric_lower_is_better_for_event(
    self, metric_event: dict[str, Any], *, fallback: Any = None
) -> bool:
    return bool(
        self._metric_lower_is_better_decision(metric_event, fallback=fallback)[
            "lower_is_better"
        ]
    )


@staticmethod
def _metric_source_note(
    metric_event: dict[str, Any], *, brief: str = "", why: str = ""
) -> str:
    parts: list[str] = []
    for key, label in (
        ("val_score_type", "type"),
        ("metric_protocol", "protocol"),
        ("metric_eval_data", "eval"),
        ("train_data_used", "train_data"),
        ("execution_mode", "execution"),
    ):
        value = str(metric_event.get(key) or "").strip()
        if value:
            parts.append(f"{label}={value}")
    for key, label in (
        ("selection_note", "note"),
        ("validation_issue", "validation_issue"),
    ):
        value = str(metric_event.get(key) or "").strip()
        if value:
            parts.append(f"{label}={value[:160]}")
    risk_text = " ".join(
        str(x or "")
        for x in (
            metric_event.get("selection_note"),
            metric_event.get("validation_issue"),
            brief,
            why,
        )
    ).lower()
    risk_tags: list[str] = []
    if any(
        term in risk_text
        for term in (
            "target leak",
            "target-leaking",
            "leaky",
            "leakage risk",
            "data leakage",
        )
    ):
        risk_tags.append("leakage_suspected")
    if any(
        term in risk_text
        for term in (
            "overfit",
            "overfitting",
            "unreliable",
            "not generalizable",
            "inflated",
        )
    ):
        risk_tags.append("overfit_suspected")
    if any(
        term in risk_text
        for term in (
            "benign_malignant",
            "diagnosis",
            "pat_malig",
            "pat_mal",
            "mal_ratio",
            "pos_rate",
            "positive rate",
            "target-derived",
            "patient positive",
        )
    ):
        risk_tags.append("target_derived_feature_suspected")
    if any(
        term in risk_text
        for term in (
            "trained and evaluated on the same",
            "evaluated on the same",
            "same validation set",
            "meta-learner overfit",
        )
    ):
        risk_tags.append("same_validation_overfit_suspected")
    if any(
        term in risk_text
        for term in ("honest estimate is", "honest score would", "real honest score")
    ):
        risk_tags.append("reported_metric_differs_from_honest_estimate")
    if risk_tags:
        parts.append("risk=" + ",".join(dict.fromkeys(risk_tags)))
    return "; ".join(parts)[:500]


def _metric_validity_card_fields(
    self, cards_after: list[Any], stage_id: str
) -> dict[str, Any]:
    for card in cards_after:
        if getattr(card, "stage_id", "") == stage_id:
            return {
                "metric_validity": getattr(card, "metric_validity", ""),
                "brief": getattr(card, "brief", ""),
                "why": getattr(card, "why", ""),
                "route_evidence": getattr(card, "route_evidence", ""),
            }
    return {}


def _task_metric_context_excerpt(self, *, max_chars: int = 1200) -> str:
    task = str(getattr(self, "task_desc", "") or "")
    if not task:
        return ""
    lines = []
    for line in task.splitlines():
        lowered = line.lower()
        if any(
            token in lowered
            for token in (
                "metric",
                "evaluation",
                "score",
                "higher",
                "lower",
                "kendall",
                "auc",
                "rmse",
                "dice",
            )
        ):
            compact = re.sub(r"\s+", " ", line).strip()
            if compact:
                lines.append(compact)
    excerpt = "\n".join(lines) if lines else task[:max_chars]
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max(0, max_chars - 24)].rstrip() + " ... [truncated]"
    return excerpt


@staticmethod
def _metric_validity_snapshot_tail(metric_event: dict[str, Any]) -> dict[str, str]:
    path = Path(str(metric_event.get("snapshot_path") or ""))
    if not path.is_file():
        return {"stdout_tail": "", "stderr_tail": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {"stdout_tail": "", "stderr_tail": ""}
    return {
        "stdout_tail": str(data.get("stdout_tail") or "")[-5000:],
        "stderr_tail": str(data.get("stderr_tail") or "")[-2000:],
    }


def _metric_validity_fact_card(
    self,
    *,
    metric_event: dict[str, Any],
    card_fields: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "metric_value",
        "metric_name",
        "lower_is_better",
        "declared_lower_is_better",
        "declared_stage_commit_lower_is_better",
        "metric_direction_source",
        "metric_direction_conflict",
        "validation_ok",
        "validation_issue",
        "submission_validation_ok",
        "submission_status",
        "val_score_type",
        "selection_eligible",
        "selection_score",
        "selection_note",
        "metric_source_note",
        "metric_validity",
        "metric_validity_note",
        "task_profile",
        "metric_protocol",
        "train_data_used",
        "metric_eval_data",
        "execution_mode",
        "candidate_ready",
        "metric_authoritative",
    )
    facts = {key: metric_event.get(key) for key in keys if key in metric_event}
    facts["stage_card"] = {
        key: str(card_fields.get(key) or "")[:1200]
        for key in ("metric_validity", "brief", "why", "route_evidence")
    }
    facts.update(self._metric_validity_snapshot_tail(metric_event))
    task_hint = getattr(self, "_task_metric_lower_is_better", None)
    if task_hint is not None:
        facts["task_metric_lower_is_better"] = bool(task_hint)
        facts["task_metric_direction_source"] = "task_description"
    excerpt = self._task_metric_context_excerpt()
    if excerpt:
        facts["task_metric_context_excerpt"] = excerpt
    if facts.get("stdout_tail"):
        facts["stdout_tail"] = str(facts["stdout_tail"])[-5000:]
    if facts.get("stderr_tail"):
        facts["stderr_tail"] = str(facts["stderr_tail"])[-2000:]
    return facts


def _metric_feedback_llm_client(self) -> Any | None:
    metric_llm = getattr(self, "_metric_validity_feedback_llm", None)
    if metric_llm is not None:
        return metric_llm
    cfg = getattr(self, "cfg", None)
    feedback_cfg = getattr(getattr(cfg, "agent", None), "feedback", None)
    if _deterministic_gate_enabled():
        feedback_cfg = self._worker_llm_stage_override("feedback") or feedback_cfg
    if feedback_cfg is None:
        return None
    metric_llm = _build_llm(feedback_cfg)
    self._metric_validity_feedback_llm = metric_llm
    return metric_llm
