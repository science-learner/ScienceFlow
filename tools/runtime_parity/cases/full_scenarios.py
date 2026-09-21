from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from tools.runtime_parity.execution.capture import capture_case


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def _worker2_estra_merge(workspace: Path) -> dict[str, Any]:
    from scienceflow.research.quality.evaluator import EvaluatorManager
    from scienceflow.research.solver.lnr.transitions.estra_magent.join_gate import (
        load_join_packets,
        write_join_packet,
    )
    from scienceflow.research.solver.lnr.transitions.estra_magent.prompt_blocks import (
        format_magent_recommendations,
    )
    from scienceflow.research.quality.finalization import run_global_merge

    worker_specs = (
        ("W00", "sidecar-w00", 0.20, "route-alpha"),
        ("W01", "sidecar-w01", 0.40, "route-beta"),
    )
    candidates: list[dict[str, Any]] = []
    worker_results: list[dict[str, Any]] = []
    join_packets: list[dict[str, Any]] = []
    for index, (worker_id, sidecar_id, metric, route) in enumerate(worker_specs):
        worker = workspace / "workers" / worker_id
        snapshot = worker / "snapshots" / "S01"
        _write(snapshot / "submission.csv", f"id,target\n1,{metric:.2f}\n")
        _write(snapshot / "solution.py", f"ROUTE = {route!r}\n")
        _write(worker / "worker_id.txt", worker_id + "\n")
        resource = worker / "resource"
        packet = write_join_packet(
            resource,
            {
                "quality_gate": "review",
                "sidecar_id": sidecar_id,
                "parent_job_id": f"job-{worker_id.lower()}",
                "observations": [f"{route} validated", f"metric={metric:.2f}"],
                "recommended_next_action": f"merge {route}",
                "useful_artifacts": [
                    {"path": "analysis.json", "type": "sidecar_analysis"}
                ],
            },
            parent_worker_id=worker_id,
        )
        assert packet is not None
        packet_path = resource / str(packet["packet_path"])
        os.utime(packet_path, (1_700_000_000 + index, 1_700_000_000 + index))
        loaded = load_join_packets(resource, parent_worker_id=worker_id)
        join_packets.extend(loaded)
        candidates.append(
            {
                "candidate_id": f"{worker_id}:S01",
                "snapshot_path": str(snapshot),
                "metric_value": metric,
                "metric_name": "score",
                "lower_is_better": False,
                "validation_ok": True,
                "submission_sha": f"sha-{worker_id.lower()}",
                "metric_validity": "high",
                "selection_eligible": True,
            }
        )
        worker_results.append(
            {
                "worker_id": worker_id,
                "status": "success",
                "stop_reason": "budget_expired",
                "stage_id": "S01",
            }
        )

    merge_calls: list[dict[str, Any]] = []

    async def merge_executor(
        merge_workspace: Path,
        prompt: str,
        budget: float,
        read_roots: list[Path],
    ) -> None:
        merge_calls.append(
            {
                "budget": budget,
                "candidate_roots": [path.name for path in read_roots],
                "requires_three": "exactly 3 distinct final artifacts" in prompt,
            }
        )
        for index, value in enumerate((0.20, 0.40, 0.30)):
            _write(
                merge_workspace / "finals" / f"final_{index:02d}" / "submission.csv",
                f"id,target\n1,{value:.2f}\n",
            )
            _write(
                merge_workspace / "finals" / f"final_{index:02d}" / "merge_report.md",
                f"deterministic final {index}\n",
            )

    with patch(
        "scienceflow.research.quality.finalization.service.time.time",
        return_value=1_700_000_100.0,
    ):
        manifest = await run_global_merge(
            merge_dir=workspace / "merge",
            candidates=candidates,
            worker_results=worker_results,
            task_desc="combine two isolated deterministic workers",
            artifact_path="submission.csv",
            ledger_filename=".run_results.md",
            wall_clock_sec=120.0,
            evaluator_manager=EvaluatorManager.default(),
            cfg=SimpleNamespace(
                evaluator=SimpleNamespace(enabled=False),
                task_profile="mlebench",
                submission_dir=workspace / "submissions",
            ),
            task_profile="mlebench",
            task_id="runtime-parity-worker2",
            task_root=workspace,
            dataset_source=None,
            merge_executor=merge_executor,
            required_finals=3,
            max_finals=3,
        )

    stable_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at"}
    }
    return capture_case(
        "worker2_estra_merge",
        {
            "workers": [row[0] for row in worker_specs],
            "candidates": candidates,
            "worker_results": worker_results,
            "join_packets": join_packets,
            "recommendations": format_magent_recommendations(join_packets),
        },
        {
            "sequence": [
                "worker_W00_isolated",
                "worker_W01_isolated",
                "estra_join_W00",
                "estra_join_W01",
                "global_merge",
                "finals_exposed",
            ],
            "merge_calls": merge_calls,
            "manifest": stable_manifest,
            "worker_isolation": {
                worker_id: (workspace / "workers" / worker_id / "worker_id.txt").read_text(
                    encoding="utf-8"
                )
                for worker_id, *_ in worker_specs
            },
        },
        workspace,
    )


async def _snapshot_stop_resume_recovery(workspace: Path) -> dict[str, Any]:
    from inquirycraft.events import JsonlEventSink, RuntimeEvent, load_events
    from inquirycraft.events.replay import canonicalize_runtime_events
    from inquirycraft.runtime import CancellationToken
    from scienceflow.research.solver.lnr.lifecycle.snapshots.workspace_snapshot import WorkspaceSnapshotStore

    live = workspace / "live"
    _write(live / "solution.py", "VALUE = 'stage'\n")
    _write(live / ".logs" / "interaction.log", "before snapshot\n")
    external = Path(tempfile.mkdtemp(prefix="runtime_parity_snapshot_store_"))
    store = WorkspaceSnapshotStore(
        workspace_dir=live,
        object_store_dir=external / "objects",
        manifest_dir=external / "manifests",
    )
    lifecycle: list[dict[str, Any]] = []
    events_path = workspace / "events.jsonl"

    async def on_lifecycle(phase: str, payload: Any) -> None:
        lifecycle.append({"phase": phase, "payload": dict(payload)})

    event_sink = JsonlEventSink(events_path)

    async def run_policy(policy, *, cancellation=None) -> str:
        cancellation = cancellation or CancellationToken()
        cancellation.raise_if_cancelled()
        payload = {"workspace": str(live), "worker_id": "W00"}
        await on_lifecycle("started", payload)
        await event_sink.emit(
            RuntimeEvent(
                type="hosted.started",
                session_id="snapshot-session",
                run_id="snapshot-run",
                agent_id="W00",
                payload=payload,
            )
        )
        try:
            result = await policy()
        except BaseException as exc:
            failure = {
                **payload,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            await on_lifecycle("failed", failure)
            await event_sink.emit(
                RuntimeEvent(
                    type="hosted.failed",
                    session_id="snapshot-session",
                    run_id="snapshot-run",
                    agent_id="W00",
                    payload=failure,
                )
            )
            raise
        await on_lifecycle("completed", payload)
        await event_sink.emit(
            RuntimeEvent(
                type="hosted.completed",
                session_id="snapshot-session",
                run_id="snapshot-run",
                agent_id="W00",
                payload=payload,
            )
        )
        return result
    try:
        with patch(
            "scienceflow.research.solver.lnr.lifecycle.snapshots.workspace_snapshot.time.time",
            return_value=1_700_000_200.0,
        ):
            captured = store.capture(snapshot_id="S01-fixed")

        async def failing_policy() -> str:
            _write(live / "solution.py", "VALUE = 'broken'\n")
            _write(live / "partial.tmp", "incomplete\n")
            raise RuntimeError("simulated worker failure")

        failure_type = ""
        try:
            await run_policy(failing_policy)
        except RuntimeError as error:
            failure_type = type(error).__name__

        _write(live / ".logs" / "interaction.log", "preserved after failure\n")
        restored = store.restore(captured.manifest_path)

        async def continued_policy() -> str:
            value = (live / "solution.py").read_text(encoding="utf-8")
            _write(live / "result.md", f"continued from {value.strip()}\n")
            return "RECOVERY_OK"

        continued = await run_policy(continued_policy)

        cancelled_calls = 0
        cancellation = CancellationToken()
        cancellation.cancel()
        async def must_not_run() -> str:
            nonlocal cancelled_calls
            cancelled_calls += 1
            return "unexpected"

        stop_type = ""
        try:
            cancellation.raise_if_cancelled()
            await must_not_run()
        except asyncio.CancelledError as error:
            stop_type = type(error).__name__
        await event_sink.aclose()
        events = canonicalize_runtime_events(load_events(events_path))
        return capture_case(
            "snapshot_stop_resume_recovery",
            {
                "snapshot_id": captured.snapshot_id,
                "captured_files": captured.captured_file_count,
                "failure_type": failure_type,
                "continued_answer": continued,
                "manual_stop_type": stop_type,
                "cancelled_policy_calls": cancelled_calls,
                "restored_solution": (live / "solution.py").read_text(encoding="utf-8"),
                "preserved_log": (live / ".logs" / "interaction.log").read_text(
                    encoding="utf-8"
                ),
            },
            {
                "sequence": [row["phase"] for row in lifecycle],
                "lifecycle": lifecycle,
                "runtime_events": events,
                "restore": {
                    "restored_files": restored.restored_file_count,
                    "restored_symlinks": restored.restored_symlink_count,
                    "restored_bytes": restored.restored_size_bytes,
                },
                "runtime_state": "IDLE",
                "manual_stop_state": "IDLE",
            },
            workspace,
        )
    finally:
        import shutil

        shutil.rmtree(external, ignore_errors=True)


FULL_SCENARIOS = {
    "snapshot_stop_resume_recovery": _snapshot_stop_resume_recovery,
    "worker2_estra_merge": _worker2_estra_merge,
}

__all__ = ["FULL_SCENARIOS"]
