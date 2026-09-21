# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Score contracts, artifact snapshots, and run-control memory effects."""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from inquirycraft.memory import Message
from inquirycraft.tools import ToolResult

from scienceflow.agent.core.ports.callback_ports import resolve_agent_callback
from scienceflow.research.quality.gate import legacy_score_contract_enabled
from scienceflow.runtime.safety.policy.agent_policies.bash_utils import (
    _looks_like_bare_solution_run,
    _python_script_run_rel_path,
    _quick_test_output_has_traceback,
)
from scienceflow.runtime.safety.policy.execution_policy import (
    EnsureFullRunResult,
    _extract_metric_from_stdout,
    _write_full_run_stamp,
    compute_fullrun_text_tail,
    final_validation_score_contract_error,
    format_score_contract_repair_feedback,
    looks_like_score_source_template,
    resolve_lower_is_better_for_bare_snapshot,
    validation_metric_value_error,
    write_fullrun_tail_snapshot,
)
from scienceflow.research.control.execution_value.evidence.quick_test import (
    parse_bash_tool_elapsed_sec,
    strip_bash_tool_prefix,
)

def _agent_gate_policy_context(agent: Any) -> tuple[str, str, str]:
    profile = str(getattr(agent, "_scienceflow_task_profile", "auto") or "auto")
    backend = str(getattr(agent, "_scienceflow_evaluator_backend", "") or "")
    artifact = str(getattr(agent, "_lnr_candidate_artifact_rel", "") or "")
    return profile, backend, artifact


def _legacy_score_gate_enabled_for_agent(agent: Any) -> bool:
    profile, backend, artifact = _agent_gate_policy_context(agent)
    return legacy_score_contract_enabled(
        task_profile=profile,
        evaluator_backend=backend,
        candidate_artifact=artifact,
    )


def _score_contract_user_message(
    detail: str, *, script_label: str = "solution.py"
) -> str:
    return format_score_contract_repair_feedback(detail, script_label=script_label)


def _score_contract_errors(stdout: str) -> tuple[str | None, str | None]:
    """Return ``(hard_error, soft_warning)`` for bare-run score parsing.

    Missing, unparseable, or non-finite ``Final Validation Score`` is a hard gate:
    the node has no reliable metric. A parseable score that is not the last
    stdout line is only a formatting issue; forcing another full training run for
    that case is too expensive and does not improve metric trust once submission
    validation has already passed.
    """
    hard_error = final_validation_score_contract_error(
        stdout,
        require_present=True,
        require_final_line=False,
    )
    if hard_error:
        return hard_error, None
    soft_warning = final_validation_score_contract_error(
        stdout,
        require_present=True,
        require_final_line=True,
    )
    return None, soft_warning


_MISSING_FINAL_SCORE_ERROR = (
    "missing required `Final Validation Score: <finite_float>` line"
)
_INTERPRETED_METRIC_SPLITS = {"validation", "val", "holdout", "heldout", "cv", "oof"}
_INTERPRETED_METRIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_NON_VALIDATION_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:private|public[ _-]+leaderboard|leaderboard|test(?:ing)?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_VALIDATION_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:val(?:idation)?|hold[ _-]?out|cv|oof)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_FINAL_OBJECTIVE_RE = re.compile(
    r"^\s*FINAL\s+(?P<name>[A-Za-z][A-Za-z0-9_.-]*)\s*=\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.IGNORECASE,
)


def _extract_final_objective_metric(stdout: str) -> tuple[float, str] | None:
    """Parse the explicit final-objective contract used by artifact tasks."""
    for raw_line in reversed(str(stdout or "").splitlines()):
        match = _FINAL_OBJECTIVE_RE.fullmatch(raw_line)
        if match is None or looks_like_score_source_template(raw_line):
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if math.isfinite(value):
            return value, match.group("name")[:160]
    return None


def _validate_llm_metric_interpretation(
    stdout: str,
    interpretation: Any,
) -> tuple[float | None, str, str | None]:
    """Verify that an LLM interpretation is directly grounded in stdout."""
    if not isinstance(interpretation, dict):
        return None, "", "metric interpreter returned no structured result"
    if interpretation.get("metric_found") is not True:
        return None, "", "metric interpreter did not find a final validation metric"
    if interpretation.get("is_final") is not True:
        return None, "", "interpreted metric is not explicitly final or best"
    if str(interpretation.get("confidence") or "").strip().lower() != "high":
        return None, "", "metric interpreter confidence is not high"

    split = str(interpretation.get("split") or "").strip().lower().replace("-", "")
    if split not in _INTERPRETED_METRIC_SPLITS:
        return (
            None,
            "",
            f"interpreted metric split is not validation-like: {split or 'missing'}",
        )

    evidence = str(interpretation.get("evidence_line") or "")
    if not evidence or "\n" in evidence or "\r" in evidence:
        return None, "", "metric evidence must be one non-empty stdout line"
    if evidence not in (stdout or ""):
        return None, "", "metric evidence line is not present verbatim in stdout"
    if looks_like_score_source_template(evidence):
        return None, "", "metric evidence is a source-code template, not runtime output"
    if _NON_VALIDATION_EVIDENCE_RE.search(evidence):
        return None, "", "metric evidence refers to test or leaderboard data"
    if re.search(
        r"(?<![A-Za-z0-9])train(?:ing)?(?![A-Za-z0-9])",
        evidence,
        re.IGNORECASE,
    ) and not _VALIDATION_EVIDENCE_RE.search(evidence):
        return None, "", "metric evidence is training-only"

    metric_name = str(interpretation.get("metric_name") or "").strip()
    if not metric_name:
        return None, "", "interpreted metric name is missing"
    try:
        value = float(interpretation.get("metric_value"))
    except (TypeError, ValueError):
        return None, "", "interpreted metric value is not numeric"
    value_error = validation_metric_value_error(stdout, value)
    if value_error:
        return None, "", value_error

    evidence_values: list[float] = []
    for match in _INTERPRETED_METRIC_NUMBER_RE.finditer(evidence):
        try:
            candidate = float(match.group(0))
        except ValueError:
            continue
        if math.isfinite(candidate):
            evidence_values.append(candidate)
    if not any(
        math.isclose(value, candidate, rel_tol=1e-9, abs_tol=1e-12)
        for candidate in evidence_values
    ):
        return None, "", "interpreted metric value does not occur in the evidence line"
    return value, metric_name[:160], None


_METRIC_STAGE_PYTHON_RUN_RE = re.compile(
    r"(?is)(?:^|[;&|\n(]\s*)"
    r"(?:(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;&|()]+\s+)*)"
    r"(?:timeout\s+(?:--[^\s]+\s+)*\d+(?:\.\d+)?[smhd]?\s+)?"
    r"(?:env\s+(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;&|()]+\s+)*)?"
    r"(?:uv\s+run\s+)?"
    r"(?:python(?:\d+(?:\.\d+)?)?|/[^ \t\n;&|()]*python(?:\d+(?:\.\d+)?)?)"
    r"(?:\s+-(?:B|E|I|O|OO|P|q|s|S|u))*"
    r"\s+(?:-c\b|-?\s*<<|[^\s;&|()]+\.py\b)"
)


def _looks_like_metric_stage_python_run(cmd: str) -> bool:
    """True when bash appears to execute Python code that can emit a fresh metric."""
    s = (cmd or "").strip()
    if not s:
        return False
    if _python_script_run_rel_path(s):
        return True
    return bool(_METRIC_STAGE_PYTHON_RUN_RE.search(s))


def _count_nonempty_data_lines_csv(path: Path) -> int:
    """Return body row count for a CSV (exclude one header line)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - 1)


def _expected_test_rows_from_dataset(workspace_dir: Path) -> int | None:
    """Prefer ``dataset/test.csv``, else ``dataset/sample_submission.csv`` row counts."""
    ds = Path(workspace_dir) / "dataset"
    for name in ("test.csv", "sample_submission.csv"):
        p = ds / name
        if p.is_file():
            n = _count_nonempty_data_lines_csv(p)
            if n > 0:
                return n
    return None


def submission_rows_plausible_for_progressive_bare(
    workspace_dir: Path,
) -> tuple[bool, str]:
    """If submission is much shorter than test/sample_submission, do not stamp progressive bare run."""
    ws = Path(workspace_dir)
    sub = ws / "submission.csv"
    if not sub.is_file():
        return False, "missing submission.csv"
    n_sub = _count_nonempty_data_lines_csv(sub)
    if n_sub <= 0:
        return False, "submission has no data rows"
    expected = _expected_test_rows_from_dataset(ws)
    if expected is None or expected < 1:
        return True, "no dataset test rows to compare"
    ratio = float(n_sub) / float(expected)
    if ratio < 0.95:
        return (
            False,
            f"submission rows={n_sub} expected≈{expected} (ratio={ratio:.3f})",
        )
    return True, ""


def _safe_metric_label(metric: Any) -> str:
    try:
        value = float(metric)
    except (TypeError, ValueError):
        return "metric_null"
    if value != value or value in (float("inf"), float("-inf")):
        return "metric_null"
    return f"metric_{value:.6g}".replace("-", "neg_").replace(".", "p")


def _sha256_file(path: Path) -> str | None:
    try:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def archive_submission_history_snapshot(
    workspace_dir: Path,
    result: EnsureFullRunResult,
    *,
    bash_cmd: str,
    validation_ok: bool,
) -> Path | None:
    """Persist run-control-ready REPL artifacts without adding them to LLM context."""
    ws = Path(workspace_dir)
    submission = ws / "submission.csv"
    if not submission.is_file():
        return None

    history_dir = ws / "submission_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in history_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
    ]
    seq = len(existing) + 1
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    metric_label = _safe_metric_label(result.metric_value)
    stem = f"run_{seq:04d}_{timestamp}_{metric_label}"
    snap_dir = history_dir / stem
    while snap_dir.exists():
        seq += 1
        stem = f"run_{seq:04d}_{timestamp}_{metric_label}"
        snap_dir = history_dir / stem
    snap_dir.mkdir(parents=False, exist_ok=False)

    copied: dict[str, dict[str, Any]] = {}
    for rel in ("solution.py", "submission.csv"):
        src = ws / rel
        if not src.is_file():
            continue
        dst = snap_dir / rel
        shutil.copy2(src, dst)
        copied[rel] = {
            "bytes": dst.stat().st_size,
            "sha256": _sha256_file(dst),
        }

    metadata = {
        "created_at_utc": timestamp,
        "metric_value": result.metric_value,
        "metric_name": result.metric_name,
        "lower_is_better": result.lower_is_better,
        "exit_code": result.exit_code,
        "wall_sec": result.wall_sec,
        "validation_ok": validation_ok,
        "bash_cmd": bash_cmd,
        "stdout_tail": compute_fullrun_text_tail(
            result.stdout, line_limit=50, max_chars=4000
        ),
        "stderr_tail": compute_fullrun_text_tail(
            result.stderr, line_limit=20, max_chars=2000
        ),
        "copied_files": copied,
    }
    (snap_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snap_dir


def _node_exec_budget_sec_for_cost_report(workspace_dir: Path) -> int | None:
    """Resolve per-node wall budget for cost reports (L2 ``node_exec_budget_sec``).

    Order: ``SCIENCEFLOW_NODE_EXEC_BUDGET_SEC`` env, then hidden/root context from the LNR solver.
    """
    env = os.environ.get("SCIENCEFLOW_NODE_EXEC_BUDGET_SEC", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    from scienceflow.runtime.core.support.node_paths import find_node_context_path

    ctx_path = find_node_context_path(Path(workspace_dir))
    if not ctx_path.is_file():
        return None
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    v = ctx.get("node_exec_budget_sec")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Module-level helpers for JSONL last-line rewriting
# ---------------------------------------------------------------------------


def _read_last_jsonl_tool_content(path: Path | None) -> str | None:
    """Return the ``message.content`` of the last JSONL line in *path* if its role is 'tool'.

    Returns None if the file is absent, the last line is not a valid tool record,
    or any I/O / parse error occurs.
    """
    if path is None or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        rec = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    msg = rec.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return None
    return str(msg.get("content") or "")


def _rewrite_last_jsonl_tool_content(path: Path, new_content: str) -> bool:
    """Overwrite the ``message.content`` of the last JSONL line in *path*.

    Only rewrites when the last record's role is 'tool'.  Returns True on success.
    Silently returns False on any error or role mismatch.
    """
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    try:
        rec = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False
    msg = rec.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return False
    msg["content"] = new_content
    lines[-1] = json.dumps(rec, ensure_ascii=False)
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def _load_node_context(workspace: Path) -> dict[str, Any]:
    """Load the optional node context without leaking storage failures upstream."""

    from scienceflow.runtime.core.support.node_paths import find_node_context_path

    path = find_node_context_path(workspace)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


async def _interpret_missing_snapshot_metric(
    agent: Any,
    stdout: str,
    script_label: str,
) -> tuple[tuple[float, str] | None, str | None]:
    """Resolve a missing canonical metric through the optional grounded callback."""

    callback = resolve_agent_callback(agent, "metric_interpretation")
    if not callable(callback):
        return None, None
    try:
        interpretation = callback(
            agent=agent,
            stdout=stdout,
            script_label=script_label,
        )
        if inspect.isawaitable(interpretation):
            interpretation = await interpretation
        value, name, error = _validate_llm_metric_interpretation(
            stdout,
            interpretation,
        )
        if error is None and value is not None:
            return (value, name), None
        agent._log_info(
            "[run-control] metric interpretation rejected: %s",
            error or "unknown reason",
        )
    except Exception as exc:
        agent._log_warning(
            "[run-control] metric interpretation failed: %s",
            type(exc).__name__,
        )
    return None, None


def _reject_snapshot_score(
    agent: Any,
    error: str,
    *,
    gate_candidate: bool,
    script_label: str,
) -> None:
    """Apply the score-contract repair or exhaustion effect."""

    if not gate_candidate:
        agent._lnr_snapshot_reason = "score_contract_invalid"
        return
    max_fix = int(getattr(agent, "_lnr_run_control_max_fix_rounds", 5) or 0)
    if max_fix <= 0:
        max_fix = 1_000_000
    injections = int(getattr(agent, "_run_control_user_injections", 0) or 0)
    if injections < max_fix:
        agent._inject_run_control_user_message(
            _score_contract_user_message(error, script_label=script_label)
        )
        agent._log_info("[run-control] score contract invalid: %s", error)
        agent._lnr_snapshot_reason = "score_contract_invalid"
        return
    agent._log_warning(
        "[run-control] score contract invalid and max fix rounds exceeded; no snapshot",
    )
    agent._lnr_snapshot_reason = "score_contract_exhausted"


def _bare_run_result(
    workspace: Path,
    raw_output: str,
    stdout: str,
    interpreted_metric: tuple[float, str] | None,
) -> EnsureFullRunResult:
    if interpreted_metric is None:
        metric_value, metric_name, _ = _extract_metric_from_stdout(stdout)
    else:
        metric_value, metric_name = interpreted_metric
    lower_is_better = resolve_lower_is_better_for_bare_snapshot(
        workspace=workspace,
        context_json=_load_node_context(workspace),
        stdout=stdout,
    )
    wall_sec = 0.0
    try:
        parsed_elapsed = parse_bash_tool_elapsed_sec(raw_output)
        if parsed_elapsed is not None:
            wall_sec = float(parsed_elapsed)
    except (TypeError, ValueError):
        wall_sec = 0.0
    return EnsureFullRunResult(
        executed=True,
        skipped=False,
        reason="gate_passing_bare_agent_run",
        exit_code=0,
        wall_sec=wall_sec,
        metric_value=metric_value,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        stdout=stdout,
        stderr="",
    )


def _write_bare_run_snapshot(
    agent: Any,
    result: EnsureFullRunResult,
    *,
    command: str,
    script_rel_path: str,
    metric_protocol: str = "",
) -> None:
    workspace = agent._workspace_dir
    snapshot_args = {
        "stdout_tail_lines": int(
            getattr(agent, "_fullrun_output_tail_stdout_lines", 50)
        ),
        "stderr_tail_lines": int(
            getattr(agent, "_fullrun_output_tail_stderr_lines", 0)
        ),
        "max_chars": int(getattr(agent, "_fullrun_output_tail_max_chars", 4000)),
        "bash_cmd": command,
        "validation_ok": True,
        "solution_path": script_rel_path,
        "execution_mode": "python_script" if script_rel_path else "inline_bash",
    }
    pending_artifact_sha = getattr(
        agent, "_lnr_candidate_artifact_pending_sha", None
    )
    artifact_changed = getattr(agent, "_lnr_candidate_artifact_changed_after_tool", None)
    if pending_artifact_sha is not None:
        metric_only = not bool(str(pending_artifact_sha or "").strip())
    elif isinstance(artifact_changed, bool):
        metric_only = not artifact_changed
    else:
        configured_artifact = str(
            getattr(agent, "_lnr_candidate_artifact_rel", "submission.csv")
            or "submission.csv"
        )
        metric_only = not (workspace / configured_artifact).is_file()
    if metric_only:
        write_fullrun_tail_snapshot(
            workspace,
            result,
            submission_status="missing_submission",
            metric_only=True,
            metric_protocol=metric_protocol,
            **snapshot_args,
        )
        agent._lnr_snapshot_ok = True
        agent._lnr_snapshot_reason = "metric_only_tail_snapshot"
        agent._log_info(
            "[run-control] metric-only tail snapshot written: no submission.csv metric=%s",
            "null" if result.metric_value is None else str(result.metric_value),
        )
        return

    write_fullrun_tail_snapshot(
        workspace,
        result,
        submission_status="pending_evaluator",
        metric_protocol=metric_protocol,
        **snapshot_args,
    )
    if (workspace / "submission.csv").is_file():
        _archive_bare_submission(agent, result, command)
    if result.metric_value is not None:
        _write_full_run_stamp(
            workspace,
            "agent",
            metric_value=result.metric_value,
            metric_name=result.metric_name,
            lower_is_better=result.lower_is_better,
        )
    agent._lnr_snapshot_ok = True
    agent._lnr_snapshot_reason = "candidate_tail_snapshot"
    try:
        import hashlib

        command_hash = hashlib.sha256(command.encode()).hexdigest()[:8]
    except Exception:
        command_hash = "?"
    agent._log_info(
        "[candidate-trigger] tail snapshot written bash_cmd_hash=%s metric=%s",
        command_hash,
        "null" if result.metric_value is None else str(result.metric_value),
    )


def _archive_bare_submission(
    agent: Any,
    result: EnsureFullRunResult,
    command: str,
) -> None:
    if not bool(getattr(agent, "_submission_history_archive_enabled", True)):
        return
    try:
        history_path = archive_submission_history_snapshot(
            agent._workspace_dir,
            result,
            bash_cmd=command,
            validation_ok=True,
        )
        if history_path is not None:
            agent._log_info(
                "[run-control] submission_history archived path=%s metric=%s",
                str(history_path.relative_to(agent._workspace_dir)),
                "null" if result.metric_value is None else str(result.metric_value),
            )
    except Exception as exc:
        agent._log_warning("[run-control] submission_history archive failed: %s", exc)


async def _maybe_write_bare_run_tail_snapshot(
    self,
    args: dict[str, Any],
    tool_result: ToolResult,
) -> None:
    """When embedded full-run is off, write a candidate metric snapshot."""
    self._lnr_snapshot_ok = False
    self._lnr_snapshot_reason = ""
    if self._embedded_full_run_enabled:
        self._lnr_snapshot_reason = "embedded_path"
        return
    cmd = (args.get("command") or "").strip()
    script_rel_path = _python_script_run_rel_path(cmd)
    lhr_any_script = bool(getattr(self, "_lnr_allow_any_stage_script", False))
    gate_candidate = _looks_like_bare_solution_run(cmd) or bool(
        lhr_any_script and script_rel_path
    )
    metric_stage_candidate = gate_candidate or _looks_like_metric_stage_python_run(cmd)
    if not metric_stage_candidate or tool_result.error:
        self._lnr_snapshot_reason = "not_bare_or_error"
        return
    script_rel_path = script_rel_path or ("solution.py" if gate_candidate else "")
    raw_out = str(tool_result.output or "")
    stripped = strip_bash_tool_prefix(raw_out)
    if _quick_test_output_has_traceback(stripped):
        self._lnr_snapshot_reason = "traceback_in_output"
        return
    metric_protocol = ""
    interpreted_metric: tuple[float, str] | None = None
    if not _legacy_score_gate_enabled_for_agent(self):
        profile, backend, artifact = _agent_gate_policy_context(self)
        interpreted_metric = _extract_final_objective_metric(stripped)
        if interpreted_metric is None:
            self._lnr_snapshot_reason = "artifact_evaluator_primary"
            self._log_info(
                "[run-control] no final objective signal profile=%s backend=%s artifact=%s",
                profile,
                backend or "default",
                artifact,
            )
            return
        score_warning = None
        metric_protocol = "benchmark"
    else:
        score_err, score_warning = _score_contract_errors(stripped)
        if score_err == _MISSING_FINAL_SCORE_ERROR:
            interpreted_metric, _ = await _interpret_missing_snapshot_metric(
                self,
                stripped,
                script_rel_path or "solution.py",
            )
            if interpreted_metric is not None:
                score_err = None
                score_warning = (
                    "accepted a high-confidence non-canonical validation metric "
                    "grounded in stdout"
                )
        if score_err:
            _reject_snapshot_score(
                self,
                score_err,
                gate_candidate=gate_candidate,
                script_label=script_rel_path or "solution.py",
            )
            return
    if interpreted_metric is not None and metric_protocol == "benchmark":
        self._log_info(
            "[run-control] accepted explicit final objective metric=%s value=%s",
            interpreted_metric[1],
            interpreted_metric[0],
        )
    if score_warning:
        self._log_info("[run-control] score contract soft warning: %s", score_warning)
    result = _bare_run_result(
        self._workspace_dir,
        raw_out,
        stripped,
        interpreted_metric,
    )
    _write_bare_run_snapshot(
        self,
        result,
        command=cmd,
        script_rel_path=script_rel_path,
        metric_protocol=metric_protocol,
    )


def _inject_run_control_user_message(self, msg: str) -> None:
    add_after_bundle = getattr(self, "_add_message_after_current_tool_bundle", None)
    if callable(add_after_bundle):
        add_after_bundle(Message.user_message(msg))
    else:
        self.memory.add_message(Message.user_message(msg))
    self._run_control_user_injections = (
        int(getattr(self, "_run_control_user_injections", 0)) + 1
    )


def _append_to_last_tool_message(self, text: str) -> None:
    """Persist *text* to the last tool record in both short_term.json and long_term.jsonl.

    Memory.messages is a property that re-reads from disk on every access, so
    mutating the returned Message object has no effect.  We instead rewrite the
    last line of each JSONL backing file directly.
    """
    snippet = (text or "").strip()
    if not snippet:
        return
    # Resolve short_term.json path via chat_history_memory.storage.json_path
    storage = getattr(
        getattr(self.memory, "chat_history_memory", None), "storage", None
    )
    sp = getattr(storage, "json_path", None) if storage is not None else None
    short_path = Path(sp) if sp else None
    # Resolve long_term.jsonl path via memory.long_term_log
    lp = getattr(self.memory, "long_term_log", None)
    long_path = Path(lp) if lp else None
    if short_path is None and long_path is None:
        return
    # Read current content from whichever file exists (prefer long_term)
    src = long_path if (long_path and long_path.is_file()) else short_path
    prev_content = _read_last_jsonl_tool_content(src)
    if prev_content is None:
        return
    new_content = prev_content.rstrip() + "\n\n" + snippet
    if short_path is not None:
        _rewrite_last_jsonl_tool_content(short_path, new_content)
    if long_path is not None:
        _rewrite_last_jsonl_tool_content(long_path, new_content)


def _inject_run_control_ok_message(
    self,
    *,
    validator: str,
    submission_csv: Path,
    wall_sec: float,
    context_json: dict,
) -> None:
    """Deprecated no-op: run-control success must not mutate LLM-visible tool memory."""
    _ = validator, submission_csv, wall_sec, context_json
    return




__all__ = [
    "_agent_gate_policy_context",
    "_append_to_last_tool_message",
    "_inject_run_control_ok_message",
    "_inject_run_control_user_message",
    "_legacy_score_gate_enabled_for_agent",
    "_load_node_context",
    "_maybe_write_bare_run_tail_snapshot",
    "_node_exec_budget_sec_for_cost_report",
    "_read_last_jsonl_tool_content",
    "_rewrite_last_jsonl_tool_content",
    "_score_contract_errors",
    "_score_contract_user_message",
    "_validate_llm_metric_interpretation",
    "archive_submission_history_snapshot",
    "submission_rows_plausible_for_progressive_bare",
]
