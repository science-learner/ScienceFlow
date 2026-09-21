# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""LNR coordinator responsibility: stage-commit facts, judgment, and ephemeral commit coordination.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.control.ephemeral_agent_session import llm_correlation
from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.stage import (
    experiment_state,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    Any,
    Message,
    Path,
    atomic_write,
    build_stage_commit_judgment_prompt,
    format_stage_files,
    normalize_stage_id,
    parse_arbiter_decision_text,
    read_ledger,
    validate_append_only_stage_commit,
)

_build_stage_commit_experiment_state = (
    experiment_state._build_stage_commit_experiment_state
)
_stage_commit_one_line = staticmethod(experiment_state._stage_commit_one_line)
_stage_commit_query_state = staticmethod(experiment_state._stage_commit_query_state)


@staticmethod
def _stage_commit_bool_text(value: Any, *, default: bool = False) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "lower"}:
        return "true"
    if text in {"false", "0", "no", "n", "higher"}:
        return "false"
    return "true" if default else "false"


@staticmethod
def _stage_commit_explicit_no_files(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "none",
        "no files",
        "no core files",
        "n/a",
        "na",
    }


@classmethod
def _stage_commit_default_judgment(cls, metric_event: dict[str, Any]) -> dict[str, str]:
    solution = cls._stage_commit_one_line(
        metric_event.get("solution_path")
        or metric_event.get("bash_kind")
        or "experiment",
        max_chars=80,
    )
    status = cls._stage_commit_one_line(
        metric_event.get("submission_status") or "metric captured", max_chars=80
    )
    validity = cls._stage_commit_one_line(
        metric_event.get("metric_validity") or "unknown", max_chars=40
    )
    return {
        "brief": f"Recorded metric-backed {solution} stage with {status}.",
        "why": f"Preserves {validity} validation evidence for estra and merge review; compare against current best.",
    }


def _stage_commit_fallback_judgment(
    self, metric_event: dict[str, Any]
) -> dict[str, str]:
    """Build a deterministic, ledger-valid judgment for an accepted Gate result."""
    judgment = self._stage_commit_default_judgment(metric_event)
    inferred_files = format_stage_files(
        None,
        metric_event=metric_event,
        workspace_dir=self.workspace_dir,
    )
    # The snapshot and metric event retain the evaluated artifact path/SHA.
    # FILES is limited to reusable source/weight files, so an artifact-only
    # task can legitimately have no eligible FILES entry.
    judgment["files"] = inferred_files or "none"
    return judgment


@classmethod
def _stage_commit_metric_note(cls, metric_event: dict[str, Any]) -> str:
    for key in ("selection_note", "metric_validity_note", "validation_issue"):
        text = cls._stage_commit_one_line(metric_event.get(key), max_chars=180)
        if text:
            return text
    metric_type = cls._stage_commit_one_line(
        metric_event.get("val_score_type") or metric_event.get("metric_protocol"),
        max_chars=60,
    )
    validity = cls._stage_commit_one_line(
        metric_event.get("metric_validity"), max_chars=40
    )
    if metric_type and metric_type != "unknown":
        return f"{metric_type} metric evidence; validity={validity or 'unknown'}."
    return f"Metric semantics unknown; validity={validity or 'unknown'}."


@classmethod
def _stage_commit_entry_from_metric_event(
    cls,
    *,
    stage_id: str,
    metric_event: dict[str, Any],
    judgment: dict[str, Any] | None,
    workspace_dir: Path | str | None = None,
) -> str:
    data = judgment if isinstance(judgment, dict) else {}
    fallback = cls._stage_commit_default_judgment(metric_event)
    metric_raw = metric_event.get(
        "metric_value", metric_event.get("reported_val_score", "unknown")
    )
    if isinstance(metric_raw, (int, float)):
        metric = f"{float(metric_raw):.12g}"
    else:
        metric = cls._stage_commit_one_line(metric_raw or "unknown", max_chars=80)
    run_time = metric_event.get("run_time_sec")
    if run_time in (None, ""):
        run_time = metric_event.get("wall_sec")
    if run_time in (None, ""):
        run_time = metric_event.get("duration_sec")
    if isinstance(run_time, (int, float)):
        run_time_text = f"{float(run_time):.3f}".rstrip("0").rstrip(".")
    else:
        run_time_text = cls._stage_commit_one_line(run_time or "unknown", max_chars=50)
    metric_type = cls._stage_commit_one_line(
        metric_event.get("val_score_type") or "unknown", max_chars=80
    )
    metric_validity = cls._stage_commit_one_line(
        metric_event.get("metric_validity") or "medium", max_chars=40
    ).lower()
    if metric_validity not in {"high", "medium", "low"}:
        metric_validity = "medium"
    brief = cls._stage_commit_one_line(
        data.get("brief") or data.get("BRIEF") or fallback["brief"], max_chars=None
    )
    why = cls._stage_commit_one_line(
        data.get("why") or data.get("WHY") or fallback["why"], max_chars=None
    )
    route_raw = (
        data.get("route_evidence")
        or data.get("routeEvidence")
        or data.get("route")
        or ""
    )
    if isinstance(route_raw, dict):
        route_raw = "; ".join(
            f"{k}={v}" for k, v in route_raw.items() if v not in (None, "")
        )
    route_evidence = cls._stage_commit_one_line(
        route_raw or fallback.get("route_evidence", ""), max_chars=None
    )
    if route_evidence and route_evidence not in why:
        why = cls._stage_commit_one_line(f"{why}; {route_evidence}", max_chars=None)
    files_raw = (
        data.get("files")
        or data.get("FILES")
        or data.get("stage_files")
        or data.get("stageFiles")
        or data.get("artifacts")
        or ""
    )
    files = format_stage_files(
        files_raw,
        metric_event=metric_event,
        workspace_dir=workspace_dir,
    )
    lines = [
        f"### {normalize_stage_id(stage_id) or stage_id}",
        f"metric: {metric}",
        f"lower_is_better: {cls._stage_commit_bool_text(metric_event.get('lower_is_better'))}",
        f"run_time_sec: {run_time_text}",
        f"metric_type: {metric_type or 'unknown'}",
        f"metric_note: {cls._stage_commit_metric_note(metric_event)}",
        f"metric_validity: {metric_validity}",
        f"BRIEF: {brief or fallback['brief']}",
        f"WHY: {why or fallback['why']}",
    ]
    if files:
        lines.append(f"FILES: {files}")
    elif cls._stage_commit_explicit_no_files(files_raw):
        lines.append("FILES: none")
    return "\n".join(lines).rstrip() + "\n"


@staticmethod
def _stage_commit_parse_judgment(text: str) -> dict[str, Any]:
    data = parse_arbiter_decision_text(text)
    return data if isinstance(data, dict) else {}


async def _fork_stage_commit_judgment(
    self,
    *,
    agent: Any,
    stage_id: str,
    metric_event: dict[str, Any],
    prompt: str,
    timeout_sec: float,
) -> dict[str, Any]:
    if getattr(agent, "llm", None) is None:
        return {}
    stage_system = (
        "You are a forked bookkeeping reviewer. Return JSON only. "
        "Do not call tools, inspect files, write files, continue experiments, or use markdown."
    )
    try:
        messages = (
            self._drop_dangling_tool_call_tail(
                agent._memory_ctx.build_messages_for_llm()
            )
            if hasattr(agent, "_memory_ctx")
            else []
        )
        system_msgs = (
            list(agent._build_system_messages())
            if hasattr(agent, "_build_system_messages")
            else []
        )
        system_msgs.append(Message.system_message(stage_system))
        text = await agent.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="stage_commit",
            base_messages=messages,
            system_messages=system_msgs,
            timeout=timeout_sec,
            stage_id=stage_id,
            lineage_id=self._lineage_uid_prefix(),
            node_uid=self._stage_node_uid(stage_id),
        )
        self._accumulate_ephemeral_tokens(agent, "stage")
        parsed = self._stage_commit_parse_judgment(text)
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_fork_judgment_ok",
                "stage_id": stage_id,
                "response_chars": len(str(text or "")),
                "has_brief": bool(parsed.get("brief") or parsed.get("BRIEF")),
                "has_why": bool(parsed.get("why") or parsed.get("WHY")),
                "llm_correlation": llm_correlation(text),
            },
        )
        return parsed
    except BaseException as exc:
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_fork_judgment_error",
                "stage_id": stage_id,
                "error": str(exc),
            },
        )
        return {}


async def _ephemeral_stage_commit(
    self,
    *,
    agent: Any,
    stage_id: str,
    metric_event: dict[str, Any],
) -> tuple[bool, str]:
    prompt = build_stage_commit_judgment_prompt(
        stage_id=stage_id,
        ledger_filename=self.ledger_filename,
        metric_event=self._sanitize_metric_event_for_agent_prompt(agent, metric_event),
        existing_ledger=agent._sanitize_agent_visible_paths(
            read_ledger(self.ledger_path)
        ),
    )
    if bool(getattr(self.lhr, "stage_commit_experiment_state_enabled", False)):
        prompt += (
            "\n\nUse this deterministic experiment state when comparing the current route "
            "with the latest and global-best stages:\n"
            + self._build_stage_commit_experiment_state(
                stage_id=stage_id,
                metric_event=metric_event,
            )
        )
    legacy_persist = bool(getattr(self.lhr, "stage_commit_persist_to_memory", False))
    persist_agent_write = (
        bool(getattr(self.lhr, "stage_commit_persist_agent_write_to_memory", True))
        or legacy_persist
    )
    persist_stage_prompt = (
        bool(getattr(self.lhr, "stage_commit_persist_prompt_to_memory", False))
        or legacy_persist
    )
    stage_user_msg = Message.user_message(prompt)
    stage_timeout = max(
        1.0,
        float(getattr(self.lhr, "stage_commit_llm_timeout_sec", 180.0) or 180.0),
    )
    judgment = await self._fork_stage_commit_judgment(
        agent=agent,
        stage_id=stage_id,
        metric_event=metric_event,
        prompt=prompt,
        timeout_sec=stage_timeout,
    )
    await self._audit_stage_result_before_commit(
        agent=agent,
        stage_id=stage_id,
        metric_event=metric_event,
        judgment=judgment,
        block_text="",
        source="fork_context" if judgment else "fallback",
    )
    ledger_before = read_ledger(self.ledger_path)
    entry = self._stage_commit_entry_from_metric_event(
        stage_id=stage_id,
        metric_event=metric_event,
        judgment=judgment,
        workspace_dir=self.workspace_dir,
    )
    sep = ""
    if ledger_before and not ledger_before.endswith("\n\n"):
        sep = "\n" if ledger_before.endswith("\n") else "\n\n"
    ledger_after = ledger_before + sep + entry
    ok, reason = validate_append_only_stage_commit(
        ledger_before, ledger_after, stage_id
    )
    if not ok:
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_deterministic_append_rejected",
                "stage_id": stage_id,
                "reason": reason,
            },
        )
        return False, reason
    try:
        self._prepare_stage_commit_transaction(
            stage_id=stage_id,
            ledger_before=ledger_before,
            ledger_after=ledger_after,
            metric_event=metric_event,
        )
    except OSError as exc:
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_transaction_prepare_error",
                "stage_id": stage_id,
                "error": str(exc),
            },
        )
        return False, f"tool_error:{exc}"
    try:
        atomic_write(self.ledger_path, ledger_after)
    except OSError as exc:
        self._rollback_stage_commit_transaction(
            stage_id=stage_id, reason=f"ledger_write:{exc}"
        )
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_deterministic_append_error",
                "stage_id": stage_id,
                "error": str(exc),
            },
        )
        return False, f"tool_error:{exc}"
    self._jsonl(
        "lhr_stage_commit_events.jsonl",
        {
            "event": "stage_commit_appended_deterministic",
            "stage_id": stage_id,
            "judgment_source": "fork_context" if judgment else "fallback",
        },
    )
    self._jsonl(
        "lhr_stage_commit_events.jsonl",
        {"event": "stage_commit_ok", "stage_id": stage_id, "turn": 1},
    )
    self._capture_s01_eda_prefix_end(stage_id=stage_id)
    if persist_agent_write:
        if persist_stage_prompt:
            agent.memory.add_message(stage_user_msg)
        stage_memory_text = entry
        if bool(getattr(self.lhr, "stage_commit_experiment_state_enabled", False)):
            stage_memory_text += "\n" + self._build_stage_commit_experiment_state(
                stage_id=stage_id,
                metric_event=metric_event,
            )
        stage_msg = Message.assistant_message(
            agent._sanitize_agent_visible_paths(stage_memory_text),
        )
        agent.memory.add_message(stage_msg)
        self._jsonl(
            "lhr_stage_commit_events.jsonl",
            {
                "event": "stage_commit_persisted_to_memory",
                "stage_id": stage_id,
                "turn": 1,
                "agent_write_persisted": True,
                "prompt_persisted": persist_stage_prompt,
                "deterministic_append": True,
            },
        )
    return True, ""


def _estra_stage_checkpoint_context(self, candidate_stages: list[str]) -> str:
    lines: list[str] = []
    for sid in candidate_stages:
        snap = self.stage_snapshots.get(sid)
        if snap is None:
            continue
        raw_source = getattr(snap, "source_event", {})
        source = raw_source if isinstance(raw_source, dict) else {}
        metric = source.get("metric_value", getattr(snap, "metric_value", ""))
        commit = str(source.get("source_commit_sha") or "")[:12]
        validation_ok = source.get("validation_ok")
        metric_validity = source.get("metric_validity")
        selection_eligible = source.get("selection_eligible")
        reason_code = source.get("metric_validity_reason_code") or source.get(
            "metric_reason_code"
        )
        parts = [f"- {sid}: metric={metric}"]
        if commit:
            parts.append(f"source_commit={commit}")
        if validation_ok is not None:
            parts.append(f"validation_ok={validation_ok}")
        if metric_validity not in (None, ""):
            parts.append(f"metric_validity={metric_validity}")
        if selection_eligible not in (None, ""):
            parts.append(
                f"selection_eligible={self._stage_card_override_text(selection_eligible)}"
            )
        if reason_code not in (None, ""):
            parts.append(f"metric_validity_reason={reason_code}")
        lines.append("; ".join(parts))
    return "\n".join(lines)
