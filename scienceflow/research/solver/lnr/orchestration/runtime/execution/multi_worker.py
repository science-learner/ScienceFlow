# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Multi-worker lifecycle, reduction, and coordinator finalization."""

from __future__ import annotations

import asyncio
from typing import Any

from scienceflow.research.quality.finalization import materialize_best_stage_final
from scienceflow.research.quality.finalization.artifacts.submission_links import (
    canonicalize_worker_workspace_artifacts,
)
from scienceflow.research.solver.lnr.orchestration.runtime.services.worker_services import (
    CoordinatorServices,
    LegacyRunStatus,
    WorkerRequest,
)


async def run_multi_worker(services: CoordinatorServices) -> dict[str, Any]:
    """Coordinate workers using only declared coordinator capabilities."""

    spec = services.spec
    n_workers = spec.worker_count
    final_artifact_mode = spec.final_artifact_mode
    merge_enabled = spec.merge_enabled
    spec.root_dir.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    services.status_projection.write(
        LegacyRunStatus.RUNNING,
        {"num_workers": n_workers, "mode": "multi_worker"},
    )
    services.append_event(
        "lhr_coordinator_events.jsonl",
        {"event": "multi_worker_start", "num_workers": n_workers},
    )
    tasks = [
        asyncio.create_task(
            services.run_one_worker(
                WorkerRequest(worker_index=index, worker_count=n_workers)
            )
        )
        for index in range(n_workers)
    ]
    live_aggregation_task = asyncio.create_task(
        services.aggregate_worker_state_periodically(
            n_workers,
            LegacyRunStatus.RUNNING,
        )
    )
    try:
        worker_results = await asyncio.gather(*tasks)
    except BaseException:
        await services.close_merge_owner_agent()
        raise
    finally:
        live_aggregation_task.cancel()
        try:
            await live_aggregation_task
        except asyncio.CancelledError:
            pass
    services.aggregate_worker_state(n_workers, LegacyRunStatus.REDUCING)
    try:
        candidates: list[dict[str, Any]] = []
        for result in worker_results:
            candidates.extend(
                services.load_worker_candidates(
                    result,
                    (not merge_enabled and final_artifact_mode == "best_stage"),
                )
            )
        collection_dir = services.write_stage_collection_outputs(
            worker_results,
            candidates,
        )
    except BaseException:
        await services.close_merge_owner_agent()
        raise
    merge_manifest: dict[str, Any] = {}
    final_artifact_manifest: dict[str, Any] = {}
    try:
        if merge_enabled:
            merge_manifest = await services.write_merge_outputs(
                worker_results,
                candidates,
            )
        elif final_artifact_mode == "best_stage":
            final_artifact_manifest = materialize_best_stage_final(
                merge_dir=services.merge_dir(),
                candidates=candidates,
                artifact_path=services.evaluator_candidate_artifact(),
                ledger_filename=spec.ledger_filename,
            )
            services.append_event(
                "lhr_coordinator_events.jsonl",
                {"event": "best_stage_finalized", **final_artifact_manifest},
            )
        else:
            services.append_event(
                "lhr_coordinator_events.jsonl",
                {
                    "event": "global_merge_skipped",
                    "reason": "merge_enabled=false",
                    "candidate_count": len(candidates),
                    "stage_collection_dir": str(collection_dir),
                },
            )
    finally:
        try:
            services.refresh_merge_owner_result(worker_results)
        finally:
            await services.close_merge_owner_agent()
    canonicalized = canonicalize_worker_workspace_artifacts(
        worker_roots=[services.worker_root(index) for index in range(n_workers)],
        candidates=candidates,
        artifact_path=services.evaluator_candidate_artifact(),
    )
    services.append_event(
        "lhr_coordinator_events.jsonl",
        {
            "event": "worker_workspace_artifacts_canonicalized",
            "artifact_count": len(canonicalized),
            "artifacts": canonicalized,
        },
    )
    services.refresh_submission_links(
        n_workers,
        merge_enabled or bool(final_artifact_manifest),
    )
    services.write_global_time_trace(worker_results)
    services.cleanup_coordinator_workspace_shell()

    main_in = sum(int(item.get("main_tokens_input", 0) or 0) for item in worker_results)
    main_cached = sum(
        int(item.get("main_tokens_cached", 0) or 0) for item in worker_results
    )
    main_out = sum(
        int(item.get("main_tokens_output", 0) or 0) for item in worker_results
    )
    main_calls = sum(int(item.get("main_llm_calls", 0) or 0) for item in worker_results)
    stage_in = sum(
        int(item.get("stage_tokens_input", 0) or 0) for item in worker_results
    )
    stage_cached = sum(
        int(item.get("stage_tokens_cached", 0) or 0) for item in worker_results
    )
    stage_calls = sum(
        int(item.get("stage_llm_calls", 0) or 0) for item in worker_results
    )
    estra_in = sum(
        int(item.get("estra_tokens_input", 0) or 0) for item in worker_results
    )
    estra_cached = sum(
        int(item.get("estra_tokens_cached", 0) or 0) for item in worker_results
    )
    estra_calls = sum(
        int(item.get("estra_llm_calls", 0) or 0) for item in worker_results
    )
    estra_switches = sum(
        int(item.get("estra_switch_stage_count", 0) or 0) for item in worker_results
    )
    estra_compacts = sum(
        int(item.get("estra_compact_count", 0) or 0) for item in worker_results
    )
    estra_decisions = sum(
        int(item.get("estra_decisions", 0) or 0) for item in worker_results
    )
    estra_fallbacks = sum(
        int(item.get("estra_fallback_count", 0) or 0) for item in worker_results
    )
    estra_continue = sum(
        int(item.get("estra_continue_count", 0) or 0) for item in worker_results
    )
    estra_redirect = sum(
        int(item.get("estra_redirect_count", 0) or 0) for item in worker_results
    )
    estra_switch_decisions = sum(
        int(item.get("estra_switch_count", 0) or 0) for item in worker_results
    )
    estra_current_continue = sum(
        int(item.get("estra_current_continue_count", 0) or 0) for item in worker_results
    )
    estra_current_redirect = sum(
        int(item.get("estra_current_redirect_count", 0) or 0) for item in worker_results
    )
    estra_stage_continue = sum(
        int(item.get("estra_stage_continue_count", 0) or 0) for item in worker_results
    )
    estra_stage_redirect = sum(
        int(item.get("estra_stage_redirect_count", 0) or 0) for item in worker_results
    )
    worker_run_succeeded = any(
        str(item.get("status") or "") == "success" for item in worker_results
    )
    partial_merge_available = bool(candidates) or bool(
        merge_manifest.get("final_count")
    )
    finalization_ready = bool(candidates) and (
        bool(merge_manifest.get("final_count"))
        if merge_enabled
        else bool(final_artifact_manifest.get("final_count"))
        if final_artifact_mode == "best_stage"
        else True
    )
    run_succeeded = worker_run_succeeded and finalization_ready
    failure_kind, worker_error_kinds = (
        ("", [])
        if run_succeeded
        else services.multi_worker_failure_kind(worker_results)
    )
    stop_reason, _ = services.multi_worker_stop_reason(
        run_succeeded,
        worker_results,
    )
    result = {
        "solver": spec.solver_name,
        "status": "success" if run_succeeded else "failed",
        "stop_reason": stop_reason,
        "failure_kind": failure_kind,
        "worker_error_kinds": worker_error_kinds,
        "worker_run_succeeded": worker_run_succeeded,
        "partial_merge_available": partial_merge_available,
        "outcome": "finalize_candidate" if run_succeeded else "stop_no_candidate",
        "finalization_ready": finalization_ready,
        "num_workers": n_workers,
        "merge_enabled": merge_enabled,
        "merge_status": str(merge_manifest.get("status") or ""),
        "merge_final_count": int(merge_manifest.get("final_count") or 0),
        "final_artifact_mode": final_artifact_mode,
        "final_artifact_status": str(final_artifact_manifest.get("status") or ""),
        "final_artifact_count": int(final_artifact_manifest.get("final_count") or 0),
        "final_artifact_manifest": (
            str(services.merge_dir() / "best_stage_manifest.json")
            if final_artifact_manifest
            else ""
        ),
        "stage_count": len(candidates),
        "global_candidate_count": len(candidates),
        "estra_switch_stage_count": estra_switches,
        "estra_compact_count": estra_compacts,
        "estra_decisions": estra_decisions,
        "estra_fallback_count": estra_fallbacks,
        "estra_continue_count": estra_continue,
        "estra_redirect_count": estra_redirect,
        "estra_switch_count": estra_switch_decisions,
        "estra_current_continue_count": estra_current_continue,
        "estra_current_redirect_count": estra_current_redirect,
        "estra_stage_continue_count": estra_stage_continue,
        "estra_stage_redirect_count": estra_stage_redirect,
        "main_cache_rate": main_cached / main_in if main_in else 0.0,
        "main_tokens_input": main_in,
        "main_tokens_cached": main_cached,
        "main_tokens_output": main_out,
        "main_llm_calls": main_calls,
        "stage_tokens_input": stage_in,
        "stage_tokens_cached": stage_cached,
        "stage_llm_calls": stage_calls,
        "estra_tokens_input": estra_in,
        "estra_tokens_cached": estra_cached,
        "estra_llm_calls": estra_calls,
        "worker_results": worker_results,
        "stage_collection_dir": str(collection_dir),
        "merge_dir": str(services.merge_dir()),
        "workspace_dir": str(services.merge_dir()),
        "ledger": str(collection_dir / "stage_collection_report.md"),
    }
    services.append_event(
        "lhr_coordinator_events.jsonl", {"event": "multi_worker_done", **result}
    )
    terminal_status = (
        LegacyRunStatus.FINISHED if run_succeeded else LegacyRunStatus.FAILED
    )
    services.status_projection.write(
        terminal_status,
        {
            "num_workers": n_workers,
            "best_metric": result.get("best_metric"),
            "stop_reason": stop_reason,
            "failure_kind": failure_kind,
            "worker_error_kinds": worker_error_kinds,
            "partial_merge_available": partial_merge_available,
        },
    )
    services.aggregate_worker_state(n_workers, terminal_status)
    return result


__all__ = ["run_multi_worker"]
