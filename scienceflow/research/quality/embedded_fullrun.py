# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Compatibility surface for componentized embedded full-run owners."""

from scienceflow.research.quality.fullrun.execution import (
    _append_embedded_full_run_to_result_md as _append_embedded_full_run_to_result_md,
    _append_mlebench_validation_to_result_md as _append_mlebench_validation_to_result_md,
    _evaluate_embedded_candidate as _evaluate_embedded_candidate,
    _maybe_embedded_full_run_after_quick_test as _maybe_embedded_full_run_after_quick_test,
    _mirror_embedded_full_run_to_interaction_log as _mirror_embedded_full_run_to_interaction_log,
    _run_mlebench_validation_after_embedded_full_run as _run_mlebench_validation_after_embedded_full_run,
)
from scienceflow.research.quality.fullrun.snapshot import (
    _append_to_last_tool_message as _append_to_last_tool_message,
    _inject_run_control_ok_message as _inject_run_control_ok_message,
    _inject_run_control_user_message as _inject_run_control_user_message,
    _maybe_write_bare_run_tail_snapshot as _maybe_write_bare_run_tail_snapshot,
    _read_last_jsonl_tool_content as _read_last_jsonl_tool_content,
    _rewrite_last_jsonl_tool_content as _rewrite_last_jsonl_tool_content,
    _score_contract_errors as _score_contract_errors,
    _score_contract_user_message as _score_contract_user_message,
    _validate_llm_metric_interpretation as _validate_llm_metric_interpretation,
    archive_submission_history_snapshot as archive_submission_history_snapshot,
    submission_rows_plausible_for_progressive_bare as submission_rows_plausible_for_progressive_bare,
)

__all__ = [
    "_append_embedded_full_run_to_result_md",
    "_append_mlebench_validation_to_result_md",
    "_append_to_last_tool_message",
    "_evaluate_embedded_candidate",
    "_inject_run_control_ok_message",
    "_inject_run_control_user_message",
    "_maybe_embedded_full_run_after_quick_test",
    "_maybe_write_bare_run_tail_snapshot",
    "_mirror_embedded_full_run_to_interaction_log",
    "_run_mlebench_validation_after_embedded_full_run",
]
