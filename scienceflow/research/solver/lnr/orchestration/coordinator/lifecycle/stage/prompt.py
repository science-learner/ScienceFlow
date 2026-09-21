# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Extracted coordinator component with an explicit dependency surface."""

from __future__ import annotations

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.coordination import (
    _parse_stage_commit_text_block,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    json,
    normalize_stage_id,
    re,
    time,
)


def _parse_stage_commit_json_block(text: str) -> tuple[dict[str, Any], str, str]:
    raw = str(text or "").strip()
    match = re.fullmatch(r"```json\s*(\{.*\})\s*```", raw, flags=re.I | re.S)
    if not match:
        return {}, "", "missing_json_block"
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}, raw, "invalid_json"
    if not isinstance(parsed, dict):
        return {}, raw, "json_not_object"
    required = (
        "stage_id",
        "metric",
        "metric_validity",
        "metric_source",
        "lower_is_better",
        "run_time_sec",
        "brief",
        "why",
        "files",
    )
    missing = [key for key in required if parsed.get(key) in (None, "")]
    if missing:
        return parsed, raw, "missing=" + ",".join(missing)
    validity = str(parsed.get("metric_validity") or "").strip().lower()
    if validity not in {"high", "medium", "low"}:
        return parsed, raw, "invalid_metric_validity"
    try:
        float(parsed.get("metric"))
    except (TypeError, ValueError):
        return parsed, raw, "invalid_metric"
    if not isinstance(parsed.get("lower_is_better"), bool):
        return parsed, raw, "invalid_lower_is_better"
    return parsed, raw, ""


def _parse_stage_commit_block(self, text: str) -> tuple[dict[str, Any], str, str]:
    output_format = (
        str(
            getattr(getattr(self, "lhr", None), "stage_commit_output_format", "text")
            or "text"
        )
        .strip()
        .lower()
    )
    if output_format == "json":
        return _parse_stage_commit_json_block(text)
    if output_format != "text":
        return {}, "", f"unsupported_output_format:{output_format}"
    return _parse_stage_commit_text_block(text)


@classmethod
def _stage_commit_judgment_from_text_block(
    cls, parsed: dict[str, Any]
) -> dict[str, Any]:
    return {
        "brief": parsed.get("brief") or "",
        "why": parsed.get("why") or "",
        "route_evidence": parsed.get("route_evidence") or "",
        "metric_validity": parsed.get("metric_validity") or "",
        "metric_source": parsed.get("metric_source") or "",
        "lower_is_better": parsed.get("lower_is_better") or "",
        "files": parsed.get("files")
        or parsed.get("stage_files")
        or parsed.get("artifacts")
        or "",
    }


def _build_compact_stage_commit_text_prompt(
    self,
    *,
    stage_id: str,
    metric_event: dict[str, Any],
    correction: str,
    one: Any,
) -> str:
    lines = [
        "[LNR_STAGE_COMMIT_REQUEST]",
        "Emit exactly one text-only STAGE_COMMIT block now. Do not call tools in this turn.",
        "Use only the observed metric/artifact facts below. Keep brief/why as compact judgments, not a log summary.",
        "The request is transient; only your STAGE_COMMIT block and the append confirmation will remain in memory.",
    ]
    if correction:
        lines.append(
            f"Previous block parse issue: {self._stage_commit_one_line(correction, max_chars=220)}"
        )
    lines.extend(
        [
            "",
            "FACTS:",
            f"stage_id: {normalize_stage_id(stage_id) or stage_id}",
            f"metric: {metric_event.get('metric_value', metric_event.get('reported_val_score', 'unknown'))}",
            f"metric_name: {one('metric_name', 'Final Validation Score')}",
            f"metric_validity: {one('metric_validity', 'medium') or 'medium'}",
            f"metric_source: {one('metric_source_note') or one('val_score_type') or one('metric_protocol') or 'metric source not specified'}",
            f"lower_is_better: {self._stage_commit_bool_text(metric_event.get('lower_is_better'))}",
            f"run_time_sec: {metric_event.get('run_time_sec') or metric_event.get('wall_sec') or metric_event.get('duration_sec') or 'unknown'}",
            f"solution_path: {one('solution_path') or 'unknown'}",
            f"submission_status: {one('submission_status') or 'unknown'}",
            f"candidate_ready: {metric_event.get('candidate_ready')}",
            f"validation_issue: {one('validation_issue') or 'none'}",
            "",
            "Required output shape:",
            "STAGE_COMMIT_BEGIN",
            f"stage_id: {normalize_stage_id(stage_id) or stage_id}",
            "metric: <numeric metric>",
            "metric_validity: high|medium|low",
            "metric_source: <one short source phrase>",
            "lower_is_better: true|false",
            "run_time_sec: <seconds|unknown>",
            "brief: <one compact judgment sentence>",
            "why: <one compact reason this stage matters, including any route lesson or avoid-repeat evidence>",
            "files: <code=core.py,helper.py weights=model.ckpt; write none only when no core code/weights exist>",
            "STAGE_COMMIT_END",
        ]
    )
    return "\n".join(lines).strip()


def _build_stage_commit_text_prompt(
    self, *, stage_id: str, metric_event: dict[str, Any], correction: str = ""
) -> str:
    def one(key: str, default: str = "") -> str:
        return self._stage_commit_one_line(
            metric_event.get(key) if isinstance(metric_event, dict) else default,
            max_chars=180,
        )

    experiment_state_enabled = bool(
        getattr(
            getattr(self, "lhr", None), "stage_commit_experiment_state_enabled", False
        )
    )
    brief_shape = (
        "<model + key features + data processing + submission construction>"
        if experiment_state_enabled
        else "<one compact judgment sentence>"
    )
    why_shape = (
        "<compare with latest and global best; state the route lesson and what to retain or avoid>"
        if experiment_state_enabled
        else "<one compact reason this stage matters, including any route lesson or avoid-repeat evidence>"
    )
    output_format = (
        str(
            getattr(getattr(self, "lhr", None), "stage_commit_output_format", "text")
            or "text"
        )
        .strip()
        .lower()
    )
    if output_format not in {"text", "json"}:
        raise ValueError(
            f"unsupported lnr.stage_commit_output_format: {output_format!r}"
        )
    json_output = output_format == "json"
    if not json_output and not experiment_state_enabled:
        return _build_compact_stage_commit_text_prompt(
            self,
            stage_id=stage_id,
            metric_event=metric_event,
            correction=correction,
            one=one,
        )
    lines = [
        "[LNR_STAGE_COMMIT_REQUEST]",
        (
            "Return exactly one fenced JSON object and no additional text. Do not call tools in this turn."
            if json_output
            else "Emit exactly one text-only STAGE_COMMIT block now. Do not call tools in this turn."
        ),
        "Use only the observed metric/artifact facts below. Keep brief/why as compact judgments, not a log summary.",
        "The request is transient; only your bookkeeping output and the append confirmation will remain in memory.",
    ]
    if correction:
        lines.append(
            f"Previous block parse issue: {self._stage_commit_one_line(correction, max_chars=220)}"
        )
    lines.extend(
        [
            "",
            "FACTS:",
            f"stage_id: {normalize_stage_id(stage_id) or stage_id}",
            f"metric: {metric_event.get('metric_value', metric_event.get('reported_val_score', 'unknown'))}",
            f"metric_name: {one('metric_name', 'Final Validation Score')}",
            f"metric_validity: {one('metric_validity', 'medium') or 'medium'}",
            f"metric_source: {one('metric_source_note') or one('val_score_type') or one('metric_protocol') or 'metric source not specified'}",
            f"lower_is_better: {self._stage_commit_bool_text(metric_event.get('lower_is_better'))}",
            f"run_time_sec: {metric_event.get('run_time_sec') or metric_event.get('wall_sec') or metric_event.get('duration_sec') or 'unknown'}",
            f"solution_path: {one('solution_path') or 'unknown'}",
            f"submission_status: {one('submission_status') or 'unknown'}",
            f"candidate_ready: {metric_event.get('candidate_ready')}",
            f"validation_issue: {one('validation_issue') or 'none'}",
        ]
    )
    if experiment_state_enabled:
        lines.extend(
            [
                "",
                "Use the deterministic experiment state below for comparisons; do not replace its metrics with recollection:",
                self._build_stage_commit_experiment_state(
                    stage_id=stage_id,
                    metric_event=metric_event,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "FILES guidance: filenames in examples are illustrative only. List only actual existing persistent core code and model-weight files used by this stage. Do not list submission artifacts or temporary/cache files. Write none when no such files exist.",
            "",
            "Required output shape:",
        ]
    )
    if json_output:
        metric_value = metric_event.get(
            "metric_value",
            metric_event.get("reported_val_score", "unknown"),
        )
        run_time = (
            metric_event.get("run_time_sec")
            or metric_event.get("wall_sec")
            or metric_event.get("duration_sec")
            or "unknown"
        )
        example = {
            "stage_id": normalize_stage_id(stage_id) or stage_id,
            "metric": metric_value,
            "metric_validity": one("metric_validity", "medium") or "medium",
            "metric_source": (
                one("metric_source_note")
                or one("val_score_type")
                or one("metric_protocol")
                or "metric source not specified"
            ),
            "lower_is_better": self._stage_commit_bool_text(
                metric_event.get("lower_is_better"),
            )
            == "true",
            "run_time_sec": run_time,
            "brief": brief_shape,
            "why": why_shape,
            "files": "<actual persistent core files or none>",
        }
        lines.extend(
            ["```json", json.dumps(example, ensure_ascii=False, indent=2), "```"]
        )
    else:
        lines.extend(
            [
                "STAGE_COMMIT_BEGIN",
                f"stage_id: {normalize_stage_id(stage_id) or stage_id}",
                "metric: <numeric metric>",
                "metric_validity: high|medium|low",
                "metric_source: <one short source phrase>",
                "lower_is_better: true|false",
                "run_time_sec: <seconds|unknown>",
                f"brief: {brief_shape}",
                f"why: {why_shape}",
                "files: <actual existing core code and weight files used by this stage; write none only when no such files exist>",
                "STAGE_COMMIT_END",
            ]
        )
    return "\n".join(lines).strip()


def _extend_stage_commit_text_policy_deadline(self, agent: Any) -> None:
    try:
        timeout = float(
            getattr(getattr(self, "lhr", None), "stage_commit_llm_timeout_sec", 180.0)
            or 180.0
        )
    except (TypeError, ValueError):
        timeout = 180.0
    grace_deadline = time.monotonic() + max(30.0, min(300.0, timeout + 30.0))
    policy = getattr(agent, "_run_policy", None)
    if policy is not None and hasattr(policy, "deadline_monotonic"):
        try:
            policy.deadline_monotonic = max(
                float(policy.deadline_monotonic), grace_deadline
            )
        except (TypeError, ValueError):
            policy.deadline_monotonic = grace_deadline


def _set_stage_commit_transient_prompt(
    self,
    agent: Any,
    *,
    stage_id: str,
    metric_event: dict[str, Any],
    correction: str = "",
) -> None:
    tool_choice = (
        str(
            getattr(getattr(self, "lhr", None), "stage_commit_tool_choice", "none")
            or "none"
        )
        .strip()
        .lower()
    )
    if tool_choice not in {"auto", "none"}:
        raise ValueError(f"unsupported lnr.stage_commit_tool_choice: {tool_choice!r}")
    context_mode = (
        str(
            getattr(getattr(self, "lhr", None), "stage_commit_context_mode", "compact")
            or "compact"
        )
        .strip()
        .lower()
    )
    if context_mode not in {"compact", "inherit"}:
        raise ValueError(f"unsupported lnr.stage_commit_context_mode: {context_mode!r}")
    prompt = self._build_stage_commit_text_prompt(
        stage_id=stage_id, metric_event=metric_event, correction=correction
    )
    setattr(agent, "_lnr_transient_user_prompt", prompt)
    # Most profiles disable tools at the API layer. Cache-sensitive profiles
    # may retain tool_choice=auto; the run-loop guard still rejects tool calls
    # during this bookkeeping turn.
    setattr(agent, "_lnr_transient_tool_choice_none", tool_choice == "none")
    setattr(
        agent,
        "_lnr_transient_context_mode",
        "" if context_mode == "inherit" else "stage_commit_compact",
    )
    setattr(agent, "_lnr_stage_commit_text_pending", True)
    setattr(agent, "_lnr_stage_commit_text_handled", False)
    self._extend_stage_commit_text_policy_deadline(agent)


def _clear_stage_commit_transient_prompt(self, agent: Any) -> None:
    for name in (
        "_lnr_transient_user_prompt",
        "_lnr_transient_tool_choice_none",
        "_lnr_transient_context_mode",
        "_lnr_transient_user_prompt_active",
        "_lnr_stage_commit_text_pending",
        "_lnr_suppress_current_text_only_memory",
        "_lnr_stage_commit_text_handled",
    ):
        try:
            if name in {
                "_lnr_transient_tool_choice_none",
                "_lnr_transient_user_prompt_active",
                "_lnr_stage_commit_text_pending",
                "_lnr_stage_commit_text_handled",
            }:
                setattr(agent, name, False)
            else:
                setattr(agent, name, "")
        except Exception:
            pass


def _pending_stage_commit_text_active(self) -> bool:
    return isinstance(getattr(self, "pending_text_stage_commit", None), dict)


def _abandon_pending_stage_commit_text(self, agent: Any) -> None:
    self.pending_text_stage_commit = None
    self._clear_stage_commit_transient_prompt(agent)


def _stage_commit_text_memory_message(
    self, *, block_text: str, stage_id: str, metric_event: dict[str, Any], entry: str
) -> str:
    metric = self._stage_commit_one_line(metric_event.get("metric_value"), max_chars=80)
    validity = self._stage_commit_one_line(
        metric_event.get("metric_validity") or "medium", max_chars=40
    ).lower()
    checked_text = entry.strip() or block_text.strip()
    text = (
        f"{checked_text}\n\n"
        "bash/edit output:\n"
        "[stage append-only write]\n"
        "path: .memory/stage_ledger.md\n"
        "status: ok\n"
        f"stage_id: {normalize_stage_id(stage_id) or stage_id}\n"
        f"metric: {metric}\n"
        f"metric_validity: {validity}\n"
        "appended: true"
    )
    if bool(
        getattr(
            getattr(self, "lhr", None), "stage_commit_experiment_state_enabled", False
        )
    ):
        text += "\n\n" + self._build_stage_commit_experiment_state(
            stage_id=stage_id,
            metric_event=metric_event,
        )
    return text
