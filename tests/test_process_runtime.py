from __future__ import annotations

import asyncio
import ast
import inspect
import os
import signal
import sys
import textwrap
from pathlib import Path

import pytest

from inquirycraft.runtime import terminate_process_tree
from scienceflow.runtime.observability.monitoring import MonitorService, ProcessObservationPublisher
from scienceflow.runtime.core.process import (
    InquiryCraftProcessAdapter,
    ProcessExecutionSession,
    ProcessLifecycleState,
    ProcessObservation,
    ProcessRequest,
    ProcessStatus,
    ProcessStreamChunk,
    render_combined_output,
)
from scienceflow.runtime.safety.tooling.bash import BashTool


def _request(
    tmp_path: Path,
    code: str,
    *,
    timeout: float = 5.0,
    session_id: str = "fixture",
    raw_output: bool = False,
    stream_limit: int = 1024,
) -> ProcessRequest:
    return ProcessRequest(
        argv=(sys.executable, "-u", "-c", code),
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout_sec=timeout,
        stream_limit_bytes=stream_limit,
        session_id=session_id,
        raw_output_dir=tmp_path / "raw" if raw_output else None,
    )


@pytest.mark.asyncio
async def test_process_session_normal_and_nonzero_exit_have_typed_lifecycle(
    tmp_path: Path,
) -> None:
    adapter = InquiryCraftProcessAdapter()
    success = ProcessExecutionSession(
        _request(tmp_path, "print('out'); print('err', file=__import__('sys').stderr)"),
        adapter=adapter,
    )
    success_outcome = await success.run()

    assert success_outcome.status is ProcessStatus.SUCCEEDED
    assert success_outcome.returncode == 0
    assert success_outcome.output.stdout == "out\n"
    assert success_outcome.output.stderr == "err\n"
    assert success.lifecycle.state is ProcessLifecycleState.SUCCEEDED
    assert success_outcome.process_group_id not in adapter.live_process_group_ids()

    failed = ProcessExecutionSession(
        _request(tmp_path, "raise SystemExit(7)", session_id="failed"),
        adapter=adapter,
    )
    failed_outcome = await failed.run()

    assert failed_outcome.status is ProcessStatus.FAILED
    assert failed_outcome.returncode == 7
    assert failed.lifecycle.state is ProcessLifecycleState.FAILED
    assert failed_outcome.process_group_id not in adapter.live_process_group_ids()


@pytest.mark.asyncio
async def test_process_session_timeout_terminates_and_unregisters_process_group(
    tmp_path: Path,
) -> None:
    adapter = InquiryCraftProcessAdapter(
        terminate_fn=lambda process, **_: terminate_process_tree(process, grace=0.1)
    )
    session = ProcessExecutionSession(
        _request(tmp_path, "import time; print('ready', flush=True); time.sleep(30)", timeout=0.1),
        adapter=adapter,
    )

    outcome = await session.run()

    assert outcome.status is ProcessStatus.TIMED_OUT
    assert outcome.output.stdout == "ready\n"
    assert session.lifecycle.state is ProcessLifecycleState.TIMED_OUT
    assert outcome.process_group_id not in adapter.live_process_group_ids()


@pytest.mark.asyncio
async def test_process_session_escalates_ignored_sigterm_to_sigkill(tmp_path: Path) -> None:
    ready = asyncio.Event()

    def on_chunk(chunk: ProcessStreamChunk) -> None:
        if "READY" in chunk.text:
            ready.set()

    adapter = InquiryCraftProcessAdapter(
        terminate_fn=lambda process, **_: terminate_process_tree(process, grace=0.05)
    )
    session = ProcessExecutionSession(
        _request(
            tmp_path,
            "import signal,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "print('READY',flush=True);time.sleep(30)",
            timeout=5.0,
            session_id="sigkill-escalation",
        ),
        adapter=adapter,
        on_chunk=on_chunk,
    )
    task = asyncio.create_task(session.run())
    await asyncio.wait_for(ready.wait(), timeout=2.0)

    await session.terminate(sigterm_grace=0.05)
    outcome = await task

    assert outcome.status is ProcessStatus.TERMINATED
    assert outcome.returncode == -signal.SIGKILL
    assert outcome.process_group_id not in adapter.live_process_group_ids()


@pytest.mark.asyncio
async def test_process_session_cancellation_reaps_process_and_preserves_outcome(
    tmp_path: Path,
) -> None:
    adapter = InquiryCraftProcessAdapter(
        terminate_fn=lambda process, **_: terminate_process_tree(process, grace=0.1)
    )
    session = ProcessExecutionSession(
        _request(tmp_path, "import time; time.sleep(30)"),
        adapter=adapter,
    )
    task = asyncio.create_task(session.run())
    while session.process is None:
        await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.lifecycle.state is ProcessLifecycleState.CANCELLED
    assert session.last_outcome is not None
    assert session.last_outcome.status is ProcessStatus.CANCELLED
    assert session.process_group_id not in adapter.live_process_group_ids()


@pytest.mark.asyncio
async def test_process_session_recoverable_stop_observes_durable_marker(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "interrupted.json"
    ready = asyncio.Event()

    def on_chunk(chunk: ProcessStreamChunk) -> None:
        if "READY" in chunk.text:
            ready.set()

    code = (
        "import pathlib,signal,sys,time;"
        f"p=pathlib.Path({str(marker)!r});"
        "signal.signal(signal.SIGUSR1,lambda *_:(p.write_text('interrupted'),sys.exit(0)));"
        "print('READY',flush=True);"
        "time.sleep(30)"
    )
    adapter = InquiryCraftProcessAdapter()
    session = ProcessExecutionSession(
        _request(tmp_path, code, timeout=10.0, session_id="recoverable"),
        adapter=adapter,
        on_chunk=on_chunk,
    )
    task = asyncio.create_task(session.run())
    await asyncio.wait_for(ready.wait(), timeout=2.0)

    await session.terminate(
        recoverable=True,
        marker_check=marker.is_file,
        sigusr1_grace=1.0,
        marker_exit_grace=1.0,
        sigterm_grace=0.1,
    )
    outcome = await task

    assert marker.read_text() == "interrupted"
    assert outcome.status is ProcessStatus.TERMINATED
    assert session.lifecycle.state is ProcessLifecycleState.TERMINATED
    assert outcome.process_group_id not in adapter.live_process_group_ids()


@pytest.mark.asyncio
async def test_stream_pump_preserves_oversized_raw_output_and_artifact_reference(
    tmp_path: Path,
) -> None:
    payload = "x" * 100_000
    session = ProcessExecutionSession(
        _request(
            tmp_path,
            f"import sys;sys.stdout.write({payload!r});sys.stderr.write('problem')",
            raw_output=True,
            stream_limit=64,
        ),
        adapter=InquiryCraftProcessAdapter(),
    )

    outcome = await session.run()
    combined = render_combined_output(outcome.output)

    assert outcome.status is ProcessStatus.SUCCEEDED
    assert outcome.output.stdout == payload
    assert outcome.output.stderr == "problem"
    assert len(combined[:100]) < len(combined)
    assert outcome.raw_output_ref is not None
    assert outcome.raw_output_ref.stdout_path.read_bytes() == payload.encode()
    assert outcome.raw_output_ref.stderr_path.read_bytes() == b"problem"
    assert outcome.raw_output_ref.manifest_path.is_file()


@pytest.mark.asyncio
async def test_process_observer_failure_is_soft_and_event_order_is_total(
    tmp_path: Path,
) -> None:
    observations: list[ProcessObservation] = []

    def broken(_observation: ProcessObservation) -> None:
        raise RuntimeError("monitor failed")

    session = ProcessExecutionSession(
        _request(tmp_path, "print('one');print('two')"),
        adapter=InquiryCraftProcessAdapter(),
        observers=(observations.append, broken),
    )

    outcome = await session.run()

    assert outcome.status is ProcessStatus.SUCCEEDED
    assert [item.sequence for item in observations] == list(
        range(1, len(observations) + 1)
    )
    assert observations[0].kind == "process.starting"
    assert observations[1].kind == "process.started"
    assert observations[-1].kind == "process.exited"
    assert outcome.observer_failures


@pytest.mark.asyncio
async def test_process_observations_publish_to_read_only_monitor_plane(
    tmp_path: Path,
) -> None:
    monitor = MonitorService(source="test.process")
    publisher = ProcessObservationPublisher(
        monitor,
        run_id="run-1",
        worker_id="worker-0",
    )
    session = ProcessExecutionSession(
        _request(tmp_path, "print('observed')", session_id="monitored"),
        adapter=InquiryCraftProcessAdapter(),
        observers=(publisher,),
    )

    outcome = await session.run()

    assert outcome.status is ProcessStatus.SUCCEEDED
    assert [item.kind for item in monitor.observations] == [
        "process.starting",
        "process.started",
        "process.stream",
        "process.exited",
    ]
    assert all(item.run_id == "run-1" for item in monitor.observations)
    assert all(item.worker_id == "worker-0" for item in monitor.observations)
    assert monitor.observations[2].payload["byte_count"] == len(b"observed\n")


@pytest.mark.asyncio
async def test_bash_visible_reduction_keeps_full_raw_output_out_of_band(
    tmp_path: Path,
) -> None:
    payload = "raw-evidence-" * 1000
    tool = BashTool(
        workspace_dir=tmp_path,
        max_output_chars=200,
        bash_timeout_sec=5.0,
        bash_timeout_slow_sec=5.0,
    )

    result = await tool.execute(f"python3 -c \"print({payload!r})\"")

    assert result.error is None
    assert len(result.output or "") < len(payload)
    assert payload in (result.system or "")


def test_legacy_stream_monitor_entry_point_is_only_a_compatibility_delegate() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(BashTool._execute_with_stream_monitoring))
    )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert len(calls) == 1
    assert isinstance(calls[0].func, ast.Attribute)
    assert calls[0].func.attr == "_execute_process_pipeline"
