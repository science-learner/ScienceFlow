"""Report recovery survives failed launches and loss of its supervisor."""

import asyncio
import os
import select
import signal
import subprocess
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import psutil
import pytest

from scienceflow.research.reporting import service as reports
from scienceflow.runtime.parallel.control import resource_claims, service
from scienceflow.runtime.parallel.control.registry import (
    read_record,
    run_dir,
    write_record,
)


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_launch_failure_has_report_without_existing_task_directory(tmp_path, status):
    root = tmp_path / "new" / "task"
    result = reports.generate_run_reports(
        dict(
            run_id="test",
            attempt=1,
            status=status,
            error="LaunchError",
            task_roots=[str(root)],
        ),
        enrich=False,
    )
    assert result[0]["status"] == "ready"
    assert result[0]["task_status"] == status
    assert "LaunchError" in (root / "report.md").read_text()
    assert (root / "report.html").is_file()


def test_enriched_archive_matches_current_and_preserves_previous_attempt(
    tmp_path, monkeypatch
):
    record = dict(run_id="test", attempt=1, status="completed")
    reports.generate_task_report(record, tmp_path, enrich=False)
    older = tmp_path / "task_logs/reports/attempt-001/report.md"
    old_bytes = older.read_bytes()
    record["attempt"] = 2
    reports.generate_task_report(record, tmp_path, enrich=False)
    monkeypatch.setattr(
        reports,
        "enrich_report",
        lambda record, report, **kwargs: replace(
            report,
            report_agent={"status": "ready", "model": "test-model"},
            research_analysis={"interpretation": "ENRICHED_ANALYSIS"},
        ),
    )
    reports.generate_task_report(record, tmp_path)
    archive = tmp_path / "task_logs/reports/attempt-002"
    assert "ENRICHED_ANALYSIS" in (archive / "report.md").read_text()
    for path in (tmp_path / "report_assets").glob("*.svg"):
        assert path.read_bytes() == (archive / "report_assets" / path.name).read_bytes()
    for name in ("report.md", "report.html", "report.json"):
        assert (tmp_path / name).read_bytes() == (archive / name).read_bytes()
    assert older.read_bytes() == old_bytes


@pytest.mark.parametrize(
    "status,remaining,mode",
    [
        ("failed", 0, "report_only"),
        ("failed", 20, "report_only"),
        ("stopped", 0, "report_only"),
        ("stopped", 20, None),
    ],
)
def test_report_recovery_keeps_normal_stop_resume(
    tmp_path, monkeypatch, status, remaining, mode
):
    from scienceflow.runtime.parallel.execution import runner

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    record = dict(
        run_id="test",
        status=status,
        research_status=status,
        research_finished_at=10,
        report_status="generation_failed",
        phase="finished",
        created_at=1,
        task_roots=[],
        manifest="unused",
        resource_requests=[],
    )
    write_record(run_dir("test"), record)
    monkeypatch.setattr(service, "_launch", lambda directory, row, **kwargs: row)
    monkeypatch.setattr(service, "validate_record_location", lambda _: None)
    monkeypatch.setattr(resource_claims, "task_requests", lambda _: [])
    reservations = []
    monkeypatch.setattr(
        resource_claims, "apply_reservation", lambda *args: reservations.append(True)
    )
    monkeypatch.setattr(
        runner,
        "ParallelRunner",
        lambda _: SimpleNamespace(
            _tasks=[SimpleNamespace(execution_plan=None)],
            _check_resume=lambda _: remaining,
        ),
    )
    resumed = service.resume_run("test")
    assert resumed.get("recovery_mode") == mode
    assert bool(reservations) == (mode is None)


@pytest.mark.asyncio
async def test_failed_report_retry_preserves_research_error(tmp_path, monkeypatch):
    from scienceflow.runtime.parallel.control import reporting, worker

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(
        asyncio.get_running_loop(), "add_signal_handler", lambda *_: None
    )
    write_record(
        run_dir("test"),
        dict(
            run_id="test",
            status="starting",
            launch_token="token",
            pid=os.getpid(),
            research_status="failed",
            research_error="LaunchError",
            research_finished_at=10,
            recovery_mode="report_only",
            manifest_payload={},
            task_roots=[],
        ),
    )

    async def finalize(record):
        assert record["research_error"] == "LaunchError"
        return [{"status": "ready"}], False

    async def forbidden(*args):
        raise AssertionError("Report retry started workers")

    monkeypatch.setattr(reporting, "finalize_reports", finalize)
    monkeypatch.setattr(worker, "run_manifest", forbidden)
    await worker.supervise("test", "token")
    row = read_record(run_dir("test"))
    assert row["status"] == "failed"
    assert row["report_status"] == "ready"
    assert row["error"] == "LaunchError"


_GUARDED_CHILD = """
import os,sys,time
from scienceflow.runtime.parallel.control.reporting.guard import report_guard
with report_guard(int(sys.argv[1]),float(sys.argv[2])):
    print(os.getpid(),flush=True)
    time.sleep(60)
"""


def test_child_enforces_deadline_without_supervisor_timer():
    process = subprocess.Popen(
        [sys.executable, "-c", _GUARDED_CHILD, str(os.getpid()), ".15"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0]
        assert int(process.stdout.readline()) == process.pid
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_supervisor_sigkill_does_not_leave_report_child():
    script = """
import os,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),'60'],start_new_session=True)
time.sleep(60)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", script, _GUARDED_CHILD],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child = None
    try:
        assert select.select([parent.stdout], [], [], 10)[0]
        child = psutil.Process(int(parent.stdout.readline()))
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while (
            time.monotonic() < deadline
            and child.is_running()
            and child.status() != psutil.STATUS_ZOMBIE
        ):
            time.sleep(0.02)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            os.killpg(child.pid, signal.SIGKILL)
