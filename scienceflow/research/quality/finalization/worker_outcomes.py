# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Worker outcome classification owned by Finalization."""

from __future__ import annotations

from typing import Any


def worker_error_kind(error: str) -> str:
    text = str(error or "")
    lower = text.lower()
    if not text.strip():
        return ""
    if (
        "llm_quota_error" in lower
        or "insufficient balance" in lower
        or "error code: 402" in lower
    ):
        return "llm_quota_error"
    if (
        "context_compact_failed" in lower
        or "compact did not fit context" in lower
        or "omitted-history main-agent request" in lower
    ):
        return "context_compact_failed"
    if "llm_api_error" in lower or "apistatuserror" in lower:
        return "llm_api_error"
    if (
        "separator is found, but chunk is longer than limit" in lower
        or "limitoverrunerror" in lower
        or "search output line exceeded" in lower
    ):
        return "tool_output_limit"
    if (
        "readerror" in lower
        or "apiconnectionerror" in lower
        or "remoteprotocolerror" in lower
        or "connection reset" in lower
        or "server disconnected" in lower
    ):
        return "llm_transport_error"
    if "timeout" in lower:
        return "worker_timeout"
    return "worker_error"


def multi_worker_failure_kind(
    worker_results: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    if not worker_results:
        return "no_workers", []
    kinds = sorted(
        {
            kind
            for result in worker_results
            for kind in [worker_error_kind(str(result.get("error") or ""))]
            if kind
        }
    )
    if not kinds:
        statuses = {
            str(result.get("status") or "").strip() for result in worker_results
        }
        if statuses and statuses <= {"no_candidate"}:
            return "no_stage_candidate", []
        if statuses and statuses <= {"failed"}:
            return "all_workers_failed", []
        return "no_viable_worker_result", []
    if len(kinds) == 1:
        return kinds[0], kinds
    return "worker_failed_mixed", kinds


def multi_worker_stop_reason(
    *, run_succeeded: bool, worker_results: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    if run_succeeded:
        success_reasons = {
            str(result.get("stop_reason") or "").strip()
            for result in worker_results
            if str(result.get("status") or "").strip() == "success"
            and str(result.get("stop_reason") or "").strip()
        }
        if success_reasons == {"evaluator_query_budget_exhausted"}:
            return "evaluator_query_budget_exhausted", []
        return "budget_expired", []
    failure_kind, kinds = multi_worker_failure_kind(worker_results)
    return failure_kind, kinds


__all__ = [
    "multi_worker_failure_kind",
    "multi_worker_stop_reason",
    "worker_error_kind",
]
