# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Embedded full-run eligibility, execution effects, and result projection."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from inquirycraft.memory import Message
from inquirycraft.tools import ToolResult

from scienceflow.foundation.contracts import EvalContext, EvaluationRequest
from scienceflow.runtime.observability.agent_io.interaction_log import truncate_for_interaction_log
from scienceflow.runtime.observability.interaction_log import (
    FullRunLightGBMStreamDeduper,
    write_raw_to_interaction_log,
)
from scienceflow.runtime.safety.policy.agent_policies.bash_utils import (
    _embedded_failure_user_message,
    _looks_like_bare_solution_run,
    _quick_test_output_has_traceback,
)
from scienceflow.runtime.safety.policy.execution_policy import (
    EnsureFullRunResult,
    _extract_metric_from_stdout,
    _write_full_run_stamp,
    compute_fullrun_text_tail,
    ensure_full_execution,
    final_validation_score_contract_error,
    format_full_run_log_body,
    resolve_lower_is_better_for_bare_snapshot,
    write_embedded_full_run_result,
    write_fullrun_tail_snapshot,
)
from scienceflow.research.control.execution_value.evidence.cost_tracker import (
    format_fullrun_cost_report,
    parse_fullrun_cost_summary,
)
from scienceflow.research.control.execution_value.evidence.quick_test import (
    estimate_full_run_seconds,
    estimate_full_run_seconds_from_wall,
    format_extrapolation_block,
    load_dataset_total_rows_from_context,
    parse_bash_tool_elapsed_sec,
    parse_quick_test_stdout,
    strip_bash_tool_prefix,
)
from scienceflow.research.quality.fullrun.snapshot import (
    _agent_gate_policy_context,
    _legacy_score_gate_enabled_for_agent,
    _load_node_context,
    _node_exec_budget_sec_for_cost_report,
    _score_contract_user_message,
    archive_submission_history_snapshot,
    submission_rows_plausible_for_progressive_bare,
)

logger = logging.getLogger("scienceflow")

def _evaluate_embedded_candidate(
    agent: Any, safety_result: EnsureFullRunResult
) -> Any | None:
    """Evaluate an embedded full-run through the framework service when attached."""

    service = getattr(agent, "_scienceflow_assessment_pipeline", None) or getattr(
        agent, "_scienceflow_evaluation_service", None
    )
    if service is None:
        return None
    cfg = getattr(agent, "_scienceflow_evaluation_cfg", None)
    task_id = str(
        getattr(cfg, "exp_id", "") or getattr(agent, "_mlebench_exp_id", "") or ""
    )
    task_root = Path(
        getattr(agent, "_scienceflow_task_root", "") or agent._workspace_dir
    )
    ctx = EvalContext(
        task_profile=str(getattr(agent, "_scienceflow_task_profile", "") or ""),
        task_id=task_id,
        task_root=task_root,
        workspace=agent._workspace_dir,
        worker_id=str(getattr(agent, "_scienceflow_worker_id", "") or "AGENT"),
        stage_id="embedded_final",
        cfg=cfg,
        metadata={
            "metric_event": {
                "metric_value": safety_result.metric_value,
                "metric_name": safety_result.metric_name,
                "lower_is_better": safety_result.lower_is_better,
                "validation_ok": safety_result.exit_code == 0,
            }
        },
    )
    try:
        request = EvaluationRequest(context=ctx, trigger="finalize")
        assess = getattr(service, "assess", None)
        outcomes = assess(request) if callable(assess) else service.evaluate(request)
    except Exception:
        logger.debug(
            "[embedded-full-run] unified candidate evaluation failed",
            exc_info=True,
        )
        return None
    return outcomes[0] if outcomes else None


def _mirror_embedded_full_run_to_interaction_log(
    self, safety_result: EnsureFullRunResult
) -> None:
    """Append safety-policy full-run lines to interaction.log (same shape as orchestrator)."""
    ws_log = self._ws_interaction_log
    if ws_log is None:
        return
    ws_log.info("[full-run] python3 solution.py (no QUICK_TEST_ROWS)")
    content = format_full_run_log_body(
        safety_result.stdout or "",
        safety_result.stderr or "",
    )
    ec = safety_result.exit_code if safety_result.exit_code is not None else -1
    ws_log.info(
        "[full-run] exit=%s [%.1fs]\n%s",
        ec,
        safety_result.wall_sec,
        content,
    )


def _append_embedded_full_run_to_result_md(
    self, safety_result: EnsureFullRunResult
) -> None:
    """Append a ``## Full run (system)`` section after embedded full-run."""
    from scienceflow.runtime.observability.formatting import _fmt_num

    path = self._workspace_dir / "result.md"
    mv = safety_result.metric_value
    mv_s = "null" if mv is None else _fmt_num(mv)
    ec = safety_result.exit_code
    wall_s = _fmt_num(safety_result.wall_sec)
    block = (
        "\n\n## Full run (system)\n\n"
        "Appended automatically after a successful **quick-test** bash run "
        "(``python3 solution.py`` under the configured quick-test mode). Full stdout/stderr: "
        "see ``[full-run]`` lines in ``logs/interaction.log``.\n\n"
        f"- **exit_code**: {ec}\n"
        f"- **wall_sec**: {wall_s}\n"
        f"- **metric_name**: {safety_result.metric_name}\n"
        f"- **metric_value**: {mv_s}\n"
        f"- **lower_is_better**: {safety_result.lower_is_better}\n"
        "\n"
        f"metric_value: {mv_s}\n"
        f"exec_time: {wall_s}\n"
        f"lower_is_better: {str(safety_result.lower_is_better).lower()}\n"
    )
    if mv is not None:
        block += f"METRIC: {mv}\n"

    # Append stdout/stderr tail for downstream clone nodes to see sub-metric details.
    _tail_stdout = int(getattr(self, "_fullrun_output_tail_stdout_lines", 50))
    _tail_stderr = int(getattr(self, "_fullrun_output_tail_stderr_lines", 0))
    _tail_max = int(getattr(self, "_fullrun_output_tail_max_chars", 4000) or 4000)
    if (_tail_stdout > 0 or _tail_stderr > 0) and (_tail_stdout + _tail_stderr > 0):
        stdout_tail = compute_fullrun_text_tail(
            safety_result.stdout or "",
            line_limit=_tail_stdout,
            max_chars=_tail_max,
        )
        stderr_tail = compute_fullrun_text_tail(
            safety_result.stderr or "",
            line_limit=_tail_stderr,
            max_chars=_tail_max,
        )
        if stdout_tail or stderr_tail:
            tail_section = "\n\n### Full run — output tail (system)\n\n"
            if stdout_tail:
                tail_section += f"<details><summary>stdout tail (last {_tail_stdout} lines)</summary>\n\n```text\n{stdout_tail}\n```\n\n</details>\n"
            if stderr_tail:
                tail_section += f"<details><summary>stderr tail (last {_tail_stderr} lines)</summary>\n\n```text\n{stderr_tail}\n```\n\n</details>\n"
            block += tail_section

    try:
        if path.is_file():
            prev = path.read_text(encoding="utf-8", errors="replace")
            if "## Full run (system)" in prev:
                return
            path.write_text(prev + block, encoding="utf-8")
        else:
            path.write_text(
                "# Results\n" + block,
                encoding="utf-8",
            )
    except OSError as exc:
        self._log_warning("[embedded-full-run] could not update result.md: %s", exc)


def _append_mlebench_validation_to_result_md(
    self,
    exp_id: str,
    ok_call: bool,
    is_valid: bool,
    result_text: str,
) -> None:
    """Append ``## MLE-bench submission validation`` after embedded full-run (if enabled)."""
    path = self._workspace_dir / "result.md"
    block = (
        "\n\n## MLE-bench submission validation (system)\n\n"
        f"- **exp_id**: `{exp_id}`\n"
        f"- **sdk_call_ok**: {ok_call}\n"
        f"- **is_valid**: {is_valid}\n\n"
        f"{result_text}\n"
    )
    try:
        if path.is_file():
            prev = path.read_text(encoding="utf-8", errors="replace")
            if "## MLE-bench submission validation (system)" in prev:
                return
            path.write_text(prev + block, encoding="utf-8")
        else:
            path.write_text("# Results\n" + block, encoding="utf-8")
    except OSError as exc:
        self._log_warning("[mlebench-validate] could not update result.md: %s", exc)


def _run_mlebench_validation_after_embedded_full_run(
    self,
) -> tuple[bool | None, str, tuple[str, bool, str] | None]:
    """After full-run success: local mlebench ``validate_submission``; log only (result.md in caller).

    Returns ``(status, text, append_payload)``:
    - ``(None, "", None)`` — validation disabled or skipped
    - ``(False, error_detail, (exp_id, ok_call, result_text))`` — invalid; caller writes
      full-run + failed validation blocks to ``result.md``
    - ``(True, assistant_suffix, (exp_id, ok_call, result_text))`` — valid; caller appends result.md
    """
    if not getattr(self, "_mlebench_validate_after_embedded_full_run", False):
        return None, "", None
    mdir = getattr(self, "_mlebench_data_dir", None) or ""
    if not str(mdir).strip():
        notice = (
            "\n\n[MLE-bench] validation skipped: `mlebench_data_root_dir` is not set "
            "(set `mlebench_data_root_dir` in YAML or `MLEBENCH_DATA_ROOT_DIR`).\n"
        )
        self._log_info(
            "[mlebench-validate] skipped: mlebench_data_root_dir empty — no local validation run",
        )
        return None, notice, None
    from scienceflow.research.quality.evaluator.providers.mlebench import (
        resolve_mlebench_exp_id,
        validate_submission_local,
    )

    exp_id = resolve_mlebench_exp_id(
        self._workspace_dir,
        cfg_exp_id=getattr(self, "_mlebench_exp_id", None) or "",
    )
    sub = self._workspace_dir / "submission.csv"
    if not exp_id:
        notice = (
            "\n\n[MLE-bench] validation skipped: could not resolve competition `exp_id` "
            "(set `exp_id` in config or use LNR layout `…/<slug>/wsp/<node>/`).\n"
        )
        self._log_info(
            "[mlebench-validate] skipped: could not resolve exp_id — no local validation run",
        )
        return None, notice, None
    if not sub.is_file():
        notice = "\n\n[MLE-bench] validation skipped: `submission.csv` not found in workspace.\n"
        self._log_info(
            "[mlebench-validate] skipped: submission.csv missing — no local validation run",
        )
        return None, notice, None
    ok_call, payload = validate_submission_local(exp_id, sub, mdir)
    is_valid = bool(payload.get("is_valid"))
    result_text = str(payload.get("result", ""))
    ws_log = self._ws_interaction_log
    if ws_log is not None:
        ws_log.info(
            "[mlebench-validate] exp_id=%s call_ok=%s is_valid=%s\n%s",
            exp_id,
            ok_call,
            is_valid,
            result_text[:8000],
        )
    self._log_info(
        "[mlebench-validate] ran exp_id=%s sdk_call_ok=%s is_valid=%s",
        exp_id,
        ok_call,
        is_valid,
    )
    if not is_valid:
        err_detail = f"exp_id={exp_id!r} sdk_call_ok={ok_call}\n{result_text}"
        return False, err_detail, (exp_id, ok_call, result_text)
    suffix = (
        f"\n\n[MLE-bench submission validation] exp_id={exp_id!r} "
        f"is_valid={is_valid} sdk_call_ok={ok_call}\n{result_text}"
    )
    return True, suffix, (exp_id, ok_call, result_text)


def _prepare_embedded_quick_test(
    agent: Any,
    args: dict[str, Any],
    tool_result: ToolResult,
) -> tuple[str, float | None] | None:
    """Validate the quick-test trigger and project its bounded output facts."""

    if not agent._embedded_full_run_enabled or agent._embedded_full_run_done:
        agent._log_info(
            "[embedded-full-run] skip: embedded disabled=%s or already done=%s",
            not agent._embedded_full_run_enabled,
            agent._embedded_full_run_done,
        )
        return None
    if not _legacy_score_gate_enabled_for_agent(agent):
        profile, backend, artifact = _agent_gate_policy_context(agent)
        agent._log_info(
            "[embedded-full-run] skip legacy score contract profile=%s backend=%s artifact=%s",
            profile,
            backend or "default",
            artifact,
        )
        return None

    command = (args.get("command") or "").strip()
    triggers_embedded = _looks_like_bare_solution_run(command)
    if tool_result.error or not triggers_embedded:
        agent._log_info(
            "[embedded-full-run] skip: tool_error=%s trigger=%s cmd_prefix=%r",
            bool(tool_result.error),
            triggers_embedded,
            command[:160],
        )
        return None

    raw_output = str(tool_result.output or "")
    stdout = strip_bash_tool_prefix(raw_output)
    if _quick_test_output_has_traceback(stdout):
        agent._log_info(
            "[embedded-full-run] skip: quick-test output contains traceback "
            "(pipe may have masked non-zero exit code)",
        )
        tail = stdout[-2000:] if len(stdout) > 2000 else stdout
        agent.memory.add_message(
            Message.user_message(
                "[Guard] Your quick-test appears to have crashed — the output contains "
                "a Python traceback, but the bash exit code was 0 (likely masked by a "
                "pipe like `| head`). Fix `solution.py` and re-run the quick-test.\n\n"
                "=== output (tail) ===\n" + tail
            )
        )
        return None
    if not (agent._workspace_dir / "submission.csv").is_file():
        agent._log_info(
            "[embedded-full-run] skip: submission.csv missing under %s",
            agent._workspace_dir,
        )
        return None
    return stdout, parse_bash_tool_elapsed_sec(raw_output)


def _block_expensive_embedded_run(
    agent: Any,
    stdout: str,
    quick_test_wall_sec: float | None,
) -> bool:
    """Apply the quick-test extrapolation guard and report whether it blocked."""

    parsed = parse_quick_test_stdout(stdout)
    total_rows = load_dataset_total_rows_from_context(agent._workspace_dir)
    if total_rows is None:
        total_rows = parsed.total_train_rows
    budget = agent._quick_test_extrapolation_budget_sec
    if budget is None:
        budget = float(
            agent._embedded_full_run_timeout_sec
            if agent._embedded_full_run_timeout_sec is not None
            else agent._bash_timeout_sec,
        )
    quick_rows = int(agent._quick_test_extrapolation_rows)
    if parsed.quick_test_rows is not None and parsed.quick_test_rows > 0:
        quick_rows = min(quick_rows, parsed.quick_test_rows)
    if parsed.data_rows is not None and parsed.data_rows > 0:
        quick_rows = min(quick_rows, parsed.data_rows)
    if not agent._quick_test_extrapolation_enabled:
        return False

    estimate = estimate_full_run_seconds(
        parsed,
        quick_test_rows=quick_rows,
        total_rows=total_rows,
    )
    if estimate is None and quick_test_wall_sec is not None:
        estimate = estimate_full_run_seconds_from_wall(
            quick_test_wall_sec=quick_test_wall_sec,
            quick_test_rows=quick_rows,
            total_rows=total_rows,
        )
    safety_factor = float(
        getattr(agent, "_quick_test_extrapolation_safety_factor", 5.0)
    )
    if estimate is None or estimate <= float(budget) * safety_factor:
        return False
    message = format_extrapolation_block(
        est_sec=estimate,
        budget_sec=float(budget),
        total_rows=int(total_rows or 0),
        quick_test_rows=quick_rows,
        parsed=parsed,
    )
    agent.memory.add_message(Message.user_message(message))
    agent._log_info(
        "[quick-test-extrapolation] blocked full-run est=%.0fs budget=%.0fs safety=%.1fx",
        estimate,
        budget,
        safety_factor,
    )
    return True


def _progressive_bare_result(
    agent: Any,
    stdout: str,
    quick_test_wall_sec: float | None,
) -> EnsureFullRunResult | None:
    """Reuse a proven full progressive run instead of executing it twice."""

    if not bool(
        getattr(agent, "_skip_embedded_duplicate_when_progressive_bare_success", True)
    ):
        return None
    score_error = final_validation_score_contract_error(
        stdout,
        require_present=True,
        require_final_line=True,
    )
    metric_value, metric_name, _ = _extract_metric_from_stdout(stdout)
    if score_error:
        agent._log_info(
            "[embedded-full-run] progressive bare run ignored: %s",
            score_error,
        )
        metric_value = None
    lower_is_better = resolve_lower_is_better_for_bare_snapshot(
        workspace=agent._workspace_dir,
        context_json=_load_node_context(agent._workspace_dir),
        stdout=stdout,
    )
    wall_sec = float(quick_test_wall_sec) if quick_test_wall_sec is not None else 0.0
    if metric_value is not None:
        rows_ok, row_reason = submission_rows_plausible_for_progressive_bare(
            agent._workspace_dir
        )
        if not rows_ok:
            agent._log_info(
                "[embedded-full-run] skip progressive bare stamp: submission row check: %s",
                row_reason,
            )
            metric_value = None
    if metric_value is None:
        return None

    result = EnsureFullRunResult(
        executed=True,
        skipped=False,
        reason="progressive_bare_agent_run",
        exit_code=0,
        wall_sec=wall_sec,
        metric_value=metric_value,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
        stdout=stdout,
        stderr="",
    )
    agent._embedded_full_run_done = True
    _write_full_run_stamp(
        agent._workspace_dir,
        "agent",
        metric_value=metric_value,
        metric_name=metric_name,
        lower_is_better=lower_is_better,
    )
    agent._log_info(
        "[embedded-full-run] progressive bare run complete; skipping duplicate "
        "ensure_full_execution (metric=%s wall=%.1fs)",
        metric_value,
        wall_sec,
    )
    return result


def _fullrun_stream_callback(agent: Any) -> tuple[Any, Any]:
    interaction_log = agent._ws_interaction_log
    if interaction_log is None:
        return None, None

    def emit(message: str) -> None:
        write_raw_to_interaction_log(
            interaction_log,
            message if message.endswith("\n") else message + "\n",
        )

    deduper = FullRunLightGBMStreamDeduper(
        emit=emit,
        format_line=lambda stream, line: f"[full-run-stream] {stream}: {line}",
    )

    def on_stream(stream: str, line: str) -> None:
        deduper(stream, line)

    return deduper, on_stream


async def _execute_embedded_full_run(agent: Any) -> EnsureFullRunResult | None:
    """Run the full execution effect and convert exceptions to repair feedback."""

    agent._embedded_full_run_done = True
    agent._log_info(
        "[handoff] quick-test passed; starting in-agent full-run (no QUICK_TEST_ROWS)",
    )
    timeout = (
        float(agent._embedded_full_run_timeout_sec)
        if agent._embedded_full_run_timeout_sec is not None
        else agent._bash_timeout_sec
    )
    watchdog_budget = agent._fullrun_epoch_watchdog_budget_sec
    if watchdog_budget is None:
        watchdog_budget = float(timeout)
    deduper, on_stream = _fullrun_stream_callback(agent)
    try:
        return await ensure_full_execution(
            agent._workspace_dir,
            timeout_sec=timeout,
            extra_env=agent._extra_env,
            on_stream_line=on_stream,
            fullrun_watchdog_enabled=agent._fullrun_epoch_watchdog_enabled,
            fullrun_watchdog_budget_sec=watchdog_budget,
            bash_output_dedup_enabled=agent._bash_output_dedup_enabled,
            bash_output_dedup_min_repeat=agent._bash_output_dedup_min_repeat,
            bash_output_dedup_summary_prefix=agent._bash_output_dedup_summary_prefix,
        )
    except Exception as exc:
        logger.warning(
            "[embedded-full-run] ensure_full_execution failed: %s",
            exc,
            exc_info=True,
        )
        agent._embedded_full_run_done = False
        error_result = EnsureFullRunResult(
            executed=False,
            skipped=False,
            reason=f"embedded_exception:{type(exc).__name__}",
            exit_code=-1,
            wall_sec=0.0,
            metric_value=None,
            metric_name="unknown",
            lower_is_better=True,
            stdout="",
            stderr=str(exc),
        )
        write_embedded_full_run_result(agent._workspace_dir, error_result)
        cost_note = (
            "[Full-run cost report] status=failed (exception — process did not start)\n"
            "No epoch data available."
        )
        message = _embedded_failure_user_message(str(exc), -1) + f"\n\n{cost_note}"
        agent.memory.add_message(Message.user_message(message))
        agent._log_info(
            "[embedded-full-run] exception; continuing session for fixes: %s",
            type(exc).__name__,
        )
        return None
    finally:
        if deduper is not None:
            deduper.flush()


def _archive_embedded_submission(agent: Any, result: EnsureFullRunResult) -> None:
    if not bool(getattr(agent, "_submission_history_archive_enabled", True)):
        return
    try:
        history_path = archive_submission_history_snapshot(
            agent._workspace_dir,
            result,
            bash_cmd="python3 solution.py",
            validation_ok=True,
        )
        if history_path is not None:
            agent._log_info(
                "[embedded-full-run] submission_history archived path=%s metric=%s",
                str(history_path.relative_to(agent._workspace_dir)),
                str(result.metric_value),
            )
    except Exception as exc:
        agent._log_warning(
            "[embedded-full-run] submission_history archive failed: %s",
            exc,
        )


def _record_embedded_full_run(agent: Any, result: EnsureFullRunResult) -> str:
    """Persist full-run facts and return the user-facing cost projection."""

    agent._mirror_embedded_full_run_to_interaction_log(result)
    write_embedded_full_run_result(agent._workspace_dir, result)
    validation_ok = (
        result.exit_code == 0
        and result.metric_value is not None
        and not result.score_contract_error
    )
    write_fullrun_tail_snapshot(
        agent._workspace_dir,
        result,
        stdout_tail_lines=int(
            getattr(agent, "_fullrun_output_tail_stdout_lines", 50)
        ),
        stderr_tail_lines=int(
            getattr(agent, "_fullrun_output_tail_stderr_lines", 20)
        ),
        max_chars=int(getattr(agent, "_fullrun_output_tail_max_chars", 4000)),
        bash_cmd="python3 solution.py",
        validation_ok=validation_ok,
    )
    if validation_ok:
        _archive_embedded_submission(agent, result)
    summary = parse_fullrun_cost_summary(
        result.stdout or "",
        wall_sec=result.wall_sec,
        exit_code=result.exit_code,
        metric_value=result.metric_value,
        metric_name=result.metric_name,
        lower_is_better=result.lower_is_better,
        node_exec_budget_sec=_node_exec_budget_sec_for_cost_report(
            agent._workspace_dir
        ),
    )
    return format_fullrun_cost_report(summary)


async def _maybe_embedded_full_run_after_quick_test(
    self,
    args: dict[str, Any],
    tool_result: ToolResult,
) -> str | None:
    """If quick-test passed, run full ``python3 solution.py``.

    On **success** (exit 0): optionally run MLE-bench ``validate_submission``; if invalid,
    still append full-run + validation (``is_valid: false``) to ``result.md`` when enabled,
    then inject a user message, reset the embedded flag, and return ``None`` so the agent can fix.
    If valid or validation skipped, append ``result.md`` and end the session (return message).

    On **failure** or **exception**, write a snapshot for the outer safety policy, inject a user
    message with stderr, reset the embedded flag, and return ``None`` so the agent can fix.
    """
    prepared = _prepare_embedded_quick_test(self, args, tool_result)
    if prepared is None:
        return None
    stripped, quick_test_wall_sec = prepared

    if _block_expensive_embedded_run(self, stripped, quick_test_wall_sec):
        return None
    safety_result = _progressive_bare_result(
        self,
        stripped,
        quick_test_wall_sec,
    )
    if safety_result is None:
        safety_result = await _execute_embedded_full_run(self)
    if safety_result is None:
        return None
    _cost_report = _record_embedded_full_run(self, safety_result)

    ec = safety_result.exit_code
    if ec is not None and ec == 0:
        service_outcome = _evaluate_embedded_candidate(self, safety_result)
        if service_outcome is None:
            score_err = (
                safety_result.score_contract_error
                or final_validation_score_contract_error(
                    safety_result.stdout or "",
                    require_present=True,
                    require_final_line=True,
                )
            )
            if score_err or safety_result.metric_value is None:
                detail = score_err or "missing finite Final Validation Score metric"
                self._embedded_full_run_done = False
                um = _score_contract_user_message(detail) + f"\n\n{_cost_report}"
                self.memory.add_message(Message.user_message(um))
                self._log_info(
                    "[embedded-full-run] score contract invalid; continuing session for fix: %s",
                    detail,
                )
                return None
            val_status, val_text, val_append = (
                self._run_mlebench_validation_after_embedded_full_run()
            )
        else:
            decision = service_outcome.decision
            event = service_outcome.event
            if not decision.accepted:
                detail = decision.message or event.metric_note or decision.reason_code
                self._embedded_full_run_done = False
                um = (
                    f"[Guard] Candidate evaluation rejected ({decision.reason_code}).\n"
                    f"{detail}\n\n{_cost_report}"
                )
                self.memory.add_message(Message.user_message(um))
                self._log_info(
                    "[embedded-full-run] unified evaluation rejected candidate: %s",
                    decision.reason_code,
                )
                return None
            val_status = True
            val_text = (
                f"\n\n[Unified evaluator] backend={event.evaluator_backend} "
                f"status={event.evaluator_status} gate={decision.reason_code}"
            )
            val_append = None
        if val_status is False:
            if self._embedded_full_run_update_result_md:
                self._append_embedded_full_run_to_result_md(safety_result)
            if val_append is not None:
                eid, ocall, rtxt = val_append
                self._append_mlebench_validation_to_result_md(
                    eid,
                    ocall,
                    False,
                    rtxt,
                )
            self._embedded_full_run_done = False
            um = (
                "[Guard] MLE-bench submission validation FAILED after full-run.\n"
                "Fix `solution.py` so `submission.csv` passes format checks, "
                "then run quick-test again.\n\n"
                "=== Validation error ===\n"
                f"{val_text}\n\n"
                f"{_cost_report}"
            )
            self.memory.add_message(Message.user_message(um))
            self._log_info(
                "[mlebench-validate] invalid — continuing session for fix",
            )
            return None

        if self._embedded_full_run_update_result_md:
            self._append_embedded_full_run_to_result_md(safety_result)
        if val_status is True and val_append is not None:
            eid, ocall, rtxt = val_append
            self._append_mlebench_validation_to_result_md(eid, ocall, True, rtxt)

        val_suffix = val_text if val_text else ""
        mv = safety_result.metric_value
        msg = (
            f"[ScienceAgent] Embedded full-run finished: exit={ec} "
            f"wall={safety_result.wall_sec:.1f}s metric={mv} "
            f"({safety_result.metric_name}). See [full-run] in logs/interaction.log."
            f"\n\n{_cost_report}"
        )
        full_msg = msg + val_suffix
        self.memory.add_message(Message.assistant_message(full_msg))
        self._log_info("[assistant] %s", truncate_for_interaction_log(full_msg))
        if self._ui:
            self._ui.render_agent_reply(full_msg)
        else:
            print(full_msg, file=sys.stdout)
        return full_msg

    self._embedded_full_run_done = False
    stderr = safety_result.stderr or ""
    um = _embedded_failure_user_message(stderr, ec) + f"\n\n{_cost_report}"
    self.memory.add_message(Message.user_message(um))
    self._log_info(
        "[embedded-full-run] exit=%s; continuing session so the model can fix solution.py",
        ec,
    )
    return None



__all__ = [
    "_append_embedded_full_run_to_result_md",
    "_append_mlebench_validation_to_result_md",
    "_evaluate_embedded_candidate",
    "_maybe_embedded_full_run_after_quick_test",
    "_mirror_embedded_full_run_to_interaction_log",
    "_run_mlebench_validation_after_embedded_full_run",
]
