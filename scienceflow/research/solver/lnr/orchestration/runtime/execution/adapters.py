# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Compatibility adapters kept at the edge of the modular runtime."""

from __future__ import annotations

from typing import Any

from scienceflow.runtime.observability.monitoring import (
    HookTraceObservationSink,
    MonitorService,
    RuntimeObservationHook,
)
from scienceflow.research.state.knowledge.prompt import PromptContextBuilder
from scienceflow.runtime.core.kernel.hooks import (
    HookDispatcher,
    HookFailureMode,
    HookIdempotencyScope,
    HookPoint,
)
from scienceflow.runtime.observability.telemetry import CorrelationJournal, HookTraceCorrelationSink
from scienceflow.research.solver.lnr.orchestration.runtime.services.models import RunSpec
from scienceflow.research.solver.lnr.orchestration.runtime.services.facade import RuntimeServices
from scienceflow.research.solver.lnr.orchestration.runtime.services.worker_services import (
    CoordinatorServices,
    CoordinatorSpec,
    LegacyStatusProjection,
    WorkerLifecycleController,
    WorkerResult,
    WorkerRuntimeServices,
    WorkerRuntimeSpec,
)


def _legacy_status_projection(host: Any) -> LegacyStatusProjection:
    return LegacyStatusProjection(
        write=lambda status, payload: host.state_machine.mark_run_status(
            status.value,
            payload=payload,
        )
    )


def worker_runtime_services_from_legacy_host(host: Any) -> WorkerRuntimeServices:
    """Adapt the legacy solver to the capabilities used by one worker loop."""

    worker_id = str(host.worker_id or "W00")
    spec = WorkerRuntimeSpec(
        worker_id=worker_id,
        task_description=str(host.task_desc),
        wall_clock_budget_sec=int(host.lhr.wall_clock_budget_sec or 0),
        seed=int(getattr(host.lhr, "seed", 0) or 0),
        max_steps=max(1, int(host.lhr.max_steps or 1)),
        memory_dir=host.memory_dir,
    )
    lifecycle = WorkerLifecycleController(
        worker_id=worker_id,
        projection=_legacy_status_projection(host),
    )
    prompt_context_builder = getattr(host, "prompt_context_builder", None)
    if not isinstance(prompt_context_builder, PromptContextBuilder):
        prompt_context_builder = PromptContextBuilder()
    return WorkerRuntimeServices(
        spec=spec,
        lifecycle=lifecycle,
        prepare_workspace=lambda: host._prepare_workspace(),
        load_existing_stage_snapshots=lambda: host._load_existing_stage_snapshots(),
        initialize_snapshot_baseline=lambda: host.snapshot_store.initialize_baseline(),
        deadline=lambda: float(host.deadline),
        initial_workspace_state=lambda: host.initial_workspace_state,
        parallel_worker_snapshot_for_prompt=(
            lambda: host._parallel_worker_snapshot_for_prompt()
        ),
        resource_context_for_prompt=lambda: host._resource_context_for_prompt(),
        skill_hint=lambda: host._lnr_skill_hint(),
        evaluator_prompt_contract=lambda: host._evaluator_prompt_contract(),
        evaluator_task_profile=lambda: host._evaluator_task_profile(),
        task_runtime_prompt_contract=lambda: host._task_runtime_prompt_contract(),
        build_prompt_context=prompt_context_builder.project,
        make_agent=lambda load_existing: host._make_agent(
            load_existing_memory=load_existing
        ),
        append_event=lambda filename, payload: host._jsonl(filename, payload),
        pending_stage_commit_active=(
            lambda: host._pending_stage_commit_text_active()
        ),
        pending_stage_commit=lambda: getattr(
            host,
            "pending_text_stage_commit",
            {},
        ),
        extend_stage_commit_deadline=(
            lambda agent: host._extend_stage_commit_text_policy_deadline(agent)
        ),
        accumulate_main_run_tokens=lambda agent: host._accumulate_main_run_tokens(
            agent
        ),
        evaluator_stop_requested=lambda: bool(host.evaluator_stop_requested),
        evaluator_stop_reason=lambda: str(host.evaluator_stop_reason or ""),
        pending_estra=lambda: bool(host.pending_estra),
        restore_pending_estra=lambda: host._restore_pending_estra(),
        build_result=lambda stop_reason: host._result(stop_reason=stop_reason),
        retain_live_agent=lambda agent: setattr(host, "_live_agent", agent),
        abandon_pending_stage_commit=(
            lambda agent: host._abandon_pending_stage_commit_text(agent)
        ),
        metric_feedback_llm=lambda: getattr(
            host,
            "_metric_validity_feedback_llm",
            None,
        ),
    )


def coordinator_services_from_legacy_host(host: Any) -> CoordinatorServices:
    """Adapt the legacy solver to coordinator-only capabilities."""

    spec = CoordinatorSpec(
        worker_count=max(1, int(host.lhr.num_workers or 1)),
        final_artifact_mode=str(host._final_artifact_mode()),
        merge_enabled=bool(getattr(host.lhr, "merge_enabled", True)),
        root_dir=host.root_dir,
        log_dir=host.log_dir,
        solver_name=str(host.solver_name),
        ledger_filename=str(host.ledger_filename),
    )

    def refresh_merge_owner_result(worker_results: list[WorkerResult]) -> None:
        owner = getattr(host, "_merge_owner_solver", None)
        if owner is None:
            return
        refreshed = owner._result(
            stop_reason=(owner.evaluator_stop_reason or "budget_expired")
        )
        for worker_result in worker_results:
            if worker_result.get("worker_id") == owner.worker_id:
                worker_result.update(refreshed)
                break

    return CoordinatorServices(
        spec=spec,
        status_projection=_legacy_status_projection(host),
        append_event=lambda filename, payload: host._jsonl(filename, payload),
        run_one_worker=lambda request: host._run_one_worker(
            request.worker_index,
            request.worker_count,
        ),
        aggregate_worker_state_periodically=lambda count, status: (
            host._aggregate_worker_state_periodically(
                n_workers=count,
                run_status=status.value,
            )
        ),
        aggregate_worker_state=lambda count, status: host._aggregate_worker_state(
            n_workers=count,
            run_status=status.value,
        ),
        close_merge_owner_agent=lambda: host._close_merge_owner_agent(),
        load_worker_candidates=lambda result, include_archived: (
            host._load_worker_candidates(
                result,
                include_archived=include_archived,
            )
        ),
        write_stage_collection_outputs=lambda results, candidates: (
            host._write_stage_collection_outputs(
                worker_results=results,
                candidates=candidates,
            )
        ),
        write_merge_outputs=lambda results, candidates: host._write_merge_outputs(
            worker_results=results,
            candidates=candidates,
        ),
        merge_dir=lambda: host._merge_dir(),
        evaluator_candidate_artifact=lambda: host._evaluator_candidate_artifact(),
        refresh_merge_owner_result=refresh_merge_owner_result,
        worker_root=lambda index: host._worker_root(index),
        refresh_submission_links=lambda count, include_merge: (
            host._refresh_submission_links(
                n_workers=count,
                include_merge=include_merge,
            )
        ),
        write_global_time_trace=lambda results: host._write_global_time_trace(results),
        cleanup_coordinator_workspace_shell=(
            lambda: host._cleanup_coordinator_workspace_shell()
        ),
        multi_worker_failure_kind=lambda results: host._multi_worker_failure_kind(
            results
        ),
        multi_worker_stop_reason=lambda succeeded, results: (
            host._multi_worker_stop_reason(
                run_succeeded=succeeded,
                worker_results=results,
            )
        ),
    )


def runtime_services_from_legacy_host(host: Any) -> RuntimeServices:
    """Capture only the capabilities required by the runtime.

    The legacy object remains outside the kernel. Dynamic method lookup inside
    the closures preserves test doubles and downstream overrides.
    """

    async def run_single() -> dict[str, Any]:
        return await host._run_single()

    async def run_multi() -> dict[str, Any]:
        return await host._run_multi_worker()

    hooks = getattr(host, "runtime_hooks", None)
    if not isinstance(hooks, HookDispatcher):
        hooks = HookDispatcher()
        host.runtime_hooks = hooks
    monitor = getattr(host, "runtime_monitor", None)
    if not isinstance(monitor, MonitorService):
        monitor = MonitorService(source="scienceflow.lnr.runtime")
        host.runtime_monitor = monitor
        observation_hook = RuntimeObservationHook(monitor)
        for point in HookPoint:
            hooks.register(
                point,
                observation_hook,
                name="runtime_monitor",
                failure_mode=HookFailureMode.SOFT,
                idempotency_scope=HookIdempotencyScope.EVENT,
                owner_component="monitoring",
            )
    trace_sink = getattr(host, "runtime_hook_trace_sink", None)
    if not isinstance(trace_sink, HookTraceObservationSink) or trace_sink.monitor is not monitor:
        trace_sink = HookTraceObservationSink(monitor)
        host.runtime_hook_trace_sink = trace_sink
    hooks.subscribe_trace(trace_sink)
    telemetry_journal = getattr(host, "telemetry_journal", None)
    if isinstance(telemetry_journal, CorrelationJournal):
        correlation_sink = getattr(host, "runtime_hook_correlation_sink", None)
        if (
            not isinstance(correlation_sink, HookTraceCorrelationSink)
            or correlation_sink.journal is not telemetry_journal
        ):
            correlation_sink = HookTraceCorrelationSink(telemetry_journal)
            host.runtime_hook_correlation_sink = correlation_sink
        hooks.subscribe_trace(correlation_sink)

    return RuntimeServices(
        spec_factory=lambda: RunSpec.from_coordinator(host),
        run_single=run_single,
        run_multi=run_multi,
        hooks=hooks,
    )


__all__ = [
    "coordinator_services_from_legacy_host",
    "runtime_services_from_legacy_host",
    "worker_runtime_services_from_legacy_host",
]
