# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Contract tests for the extracted LNR runtime boundary."""

from __future__ import annotations

import ast
import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

from scienceflow.research.solver.lnr.orchestration.runtime import (
    LnrRuntime,
    RunMode,
    RunSpec,
    RuntimeServices,
)
from scienceflow.research.solver.lnr.orchestration.runtime.services.deadline import (
    global_merge_reserve_sec,
    worker_wall_clock_budget_sec,
)
from scienceflow.research.solver.lnr.orchestration.runtime.execution.adapters import (
    worker_runtime_services_from_legacy_host,
)
from scienceflow.research.quality.finalization.worker_outcomes import (
    multi_worker_failure_kind,
    worker_error_kind,
)
from scienceflow.research.solver.lnr.orchestration.runtime.execution.worker_environment import (
    build_worker_environment,
    format_cpu_ids,
    slice_cpu_ids,
)
from scienceflow.research.solver.lnr.orchestration.runtime.services.worker_services import WorkerRequest
from scienceflow.research.solver.lnr.orchestration.runtime.execution.single_worker import run_single_worker
from scienceflow.runtime.core.kernel import WorkerLifecycleState
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.adapter import (
    EvaluationAdapterOwner,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.capture import (
    EvaluationCaptureOwner,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.commit import EvaluationCommitOwner
from scienceflow.research.solver.lnr.orchestration.coordinator.evaluation.profile import (
    EvaluationProfileOwner,
)
from scienceflow.research.solver.lnr.orchestration.solver import LnrSolver


def test_evaluation_owners_are_inherited_without_solver_descriptor_copies() -> None:
    assert issubclass(LnrSolver, EvaluationProfileOwner)
    assert issubclass(LnrSolver, EvaluationAdapterOwner)
    assert issubclass(LnrSolver, EvaluationCaptureOwner)
    assert issubclass(LnrSolver, EvaluationCommitOwner)

    for method_name in (
        "_evaluator_task_profile",
        "_metric_validity_feedback_judgment",
        "_stage_capture_callback",
        "_commit_evaluated_stage",
    ):
        assert method_name not in LnrSolver.__dict__
        assert callable(getattr(LnrSolver, method_name))


def test_runtime_routes_coordinator_and_worker_without_owning_solver_state() -> None:
    async def exercise() -> None:
        calls: list[str] = []

        async def run_single() -> dict[str, str]:
            calls.append("single")
            return {"mode": "single"}

        async def run_multi() -> dict[str, str]:
            calls.append("multi")
            return {"mode": "multi"}

        runtime = LnrRuntime(
            RuntimeServices(
                spec_factory=lambda: RunSpec(
                    worker_id="",
                    worker_count=2,
                    mode=RunMode.MULTI_WORKER,
                ),
                run_single=run_single,
                run_multi=run_multi,
            )
        )
        assert await runtime.run() == {"mode": "multi"}
        assert await runtime.run(
            RunSpec(worker_id="W00", worker_count=2, mode=RunMode.SINGLE_WORKER)
        ) == {"mode": "single"}
        assert calls == ["multi", "single"]
        assert not hasattr(runtime, "host")

    asyncio.run(exercise())


def test_worker_environment_preserves_cpu_seed_and_thread_partitioning() -> None:
    cfg = SimpleNamespace(exec=SimpleNamespace(cpu_list=""))
    lhr = SimpleNamespace(omp_threads_cap=8, seed=2222)
    env = build_worker_environment(
        cfg=cfg,
        lhr=lhr,
        worker_index=1,
        worker_count=2,
        environ={"SCIENCEFLOW_TASK_CPU_LIST": "0-19"},
    )
    assert (
        format_cpu_ids(slice_cpu_ids(list(range(20)), worker_index=1, worker_count=2))
        == "10-19"
    )
    assert env["SCIENCEFLOW_TASK_CPU_LIST"] == "0-19"
    assert env["SCIENCEFLOW_WORKER_CPU_LIST"] == "10-19"
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["SCIENCEFLOW_RANDOM_SEED"] == "2223"
    assert env["PYTHONHASHSEED"] == "2223"


def test_worker_request_has_stable_coordinator_identity() -> None:
    request = WorkerRequest(worker_index=1, worker_count=2)

    assert request.worker_id == "W01"
    assert (request.worker_index, request.worker_count) == (1, 2)


def test_single_worker_legacy_adapter_preserves_status_and_result(tmp_path) -> None:
    async def exercise() -> None:
        statuses: list[tuple[str, dict[str, object]]] = []
        abandoned: list[object] = []
        agent = SimpleNamespace(llm=None)
        host = SimpleNamespace(
            worker_id="W00",
            task_desc="Solve the fixture task.",
            lhr=SimpleNamespace(
                wall_clock_budget_sec=60,
                seed=2222,
                max_steps=2,
            ),
            memory_dir=tmp_path / "memory",
            deadline=0.0,
            initial_workspace_state="fixture-state",
            evaluator_stop_requested=False,
            evaluator_stop_reason="",
            pending_estra=False,
            state_machine=SimpleNamespace(
                mark_run_status=lambda status, payload=None: statuses.append(
                    (status, dict(payload or {}))
                )
            ),
            snapshot_store=SimpleNamespace(initialize_baseline=lambda: True),
            _prepare_workspace=lambda: None,
            _load_existing_stage_snapshots=lambda: None,
            _parallel_worker_snapshot_for_prompt=lambda: "",
            _resource_context_for_prompt=lambda: "",
            _lnr_skill_hint=lambda: "",
            _evaluator_prompt_contract=lambda: "",
            _evaluator_task_profile=lambda: "mlebench",
            _task_runtime_prompt_contract=lambda: "",
            _make_agent=lambda **_kwargs: agent,
            _jsonl=lambda _filename, _payload: None,
            _pending_stage_commit_text_active=lambda: False,
            _extend_stage_commit_text_policy_deadline=lambda _agent: None,
            _accumulate_main_run_tokens=lambda _agent: None,
            _restore_pending_estra=lambda: asyncio.sleep(0),
            _result=lambda **_kwargs: {"status": "success", "best_metric": 0.5},
            _abandon_pending_stage_commit_text=abandoned.append,
            _metric_validity_feedback_llm=None,
        )
        services = worker_runtime_services_from_legacy_host(host)

        result = await run_single_worker(services, keep_agent_open=True)

        assert result == {"status": "success", "best_metric": 0.5}
        assert [status for status, _ in statuses] == ["running", "finished"]
        assert services.lifecycle.machine.state is WorkerLifecycleState.SUCCEEDED
        assert abandoned == [agent]
        assert host._live_agent is agent

        failed_statuses: list[tuple[str, dict[str, object]]] = []
        host.state_machine = SimpleNamespace(
            mark_run_status=lambda status, payload=None: failed_statuses.append(
                (status, dict(payload or {}))
            )
        )
        host._result = lambda **_kwargs: {
            "status": "no_candidate",
            "outcome": "stop_no_candidate",
            "stage_count": 0,
        }
        failed_services = worker_runtime_services_from_legacy_host(host)

        failed = await run_single_worker(failed_services)

        assert failed["status"] == "no_candidate"
        assert [status for status, _ in failed_statuses] == ["running", "failed"]
        assert failed_services.lifecycle.machine.state is WorkerLifecycleState.FAILED

    asyncio.run(exercise())


def test_runtime_deadline_and_failure_classification_are_stable() -> None:
    lhr = SimpleNamespace(
        merge_enabled=True,
        wall_clock_budget_sec=3600,
        global_merge_wall_clock_sec=900,
    )
    assert global_merge_reserve_sec(lhr) == 540
    assert worker_wall_clock_budget_sec(lhr) == 3060
    assert worker_error_kind("ReadError: stream disconnected") == "llm_transport_error"
    assert multi_worker_failure_kind(
        [
            {"status": "failed", "error": "Error code: 402"},
            {"status": "failed", "error": "context_compact_failed"},
        ]
    ) == ("worker_failed_mixed", ["context_compact_failed", "llm_quota_error"])


def test_runtime_package_does_not_import_coordinator_implementation() -> None:
    root = (
        Path(__file__).parents[1]
        / "scienceflow"
        / "research"
        / "solver"
        / "lnr"
        / "orchestration"
        / "runtime"
    )
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "scienceflow.research.solver.lnr.orchestration.solver", path
            elif isinstance(node, ast.Import):
                assert all(
                    alias.name != "scienceflow.research.solver.lnr.orchestration.solver"
                    for alias in node.names
                ), path


def test_worker_runners_depend_on_capability_services_not_legacy_host() -> None:
    root = (
        Path(__file__).parents[1]
        / "scienceflow"
        / "research"
        / "solver"
        / "lnr"
        / "orchestration"
        / "runtime"
    )
    for filename, function_name in (
        ("execution/single_worker.py", "run_single_worker"),
        ("execution/multi_worker.py", "run_multi_worker"),
    ):
        path = root / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name
        )
        assert all(argument.arg != "host" for argument in function.args.args), path
        assert all(
            not (
                isinstance(node, ast.Name)
                and node.id == "host"
                and isinstance(node.ctx, ast.Load)
            )
            for node in ast.walk(function)
        ), path


def test_public_implementations_keep_stable_module_identity() -> None:
    solver = importlib.import_module("scienceflow.research.solver.lnr.orchestration.solver")
    observer = importlib.import_module(
        "scienceflow.research.solver.lnr.resources.runtime.observer.controller"
    )

    assert solver.LnrSolver.__module__ == "scienceflow.research.solver.lnr.orchestration.solver"
    assert (
        observer.LHRResourceObserver.__module__
        == "scienceflow.research.solver.lnr.resources.runtime.observer.controller"
    )
