"""A run cannot finish before its separate report phase has usable files."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scienceflow.runtime.parallel.control import reporting, worker
from scienceflow.runtime.parallel.control.registry import (
    read_record,
    run_dir,
    write_record,
)


@pytest.mark.parametrize("value", [0, -1, True, "nan", "inf", None, "bad"])
def test_invalid_report_budget(value):
    with pytest.raises(ValueError, match="report_budget_sec"):
        reporting.report_budget({"report_budget_sec": value})


def test_default_report_budget():
    assert reporting.report_budget({}) == 300


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [True, False])
@pytest.mark.parametrize("report_only", [True, False])
async def test_completion_waits_for_report(tmp_path, monkeypatch, ready, report_only):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr(
        asyncio.get_running_loop(), "add_signal_handler", lambda *_: None
    )
    record = dict(
        run_id="test",
        launch_token="token",
        pid=os.getpid(),
        status="starting",
        manifest="unused",
        manifest_payload={},
        task_roots=[],
        attempt=1,
    )
    if report_only:
        record.update(
            recovery_mode="report_only",
            research_status="completed",
            research_finished_at=1,
        )
    directory = run_dir("test")
    write_record(directory, record)

    async def run(_):
        assert not report_only, "Report retry must not restart research"
        return SimpleNamespace(text="done", results=[])

    async def finalize(current):
        persisted = read_record(directory)
        assert persisted["status"] == "running"
        assert persisted["phase"] == "reporting"
        assert "finished_at" not in persisted
        assert persisted["research_status"] == "completed"
        return [{"status": "ready" if ready else "generation_failed"}], False

    monkeypatch.setattr(worker, "run_manifest", run)
    monkeypatch.setattr(reporting, "finalize_reports", finalize)
    from scienceflow.research.onboarding.library import export

    monkeypatch.setattr(export, "export_success", lambda _: {"status": "done"})
    await worker.supervise("test", "token")
    final = read_record(directory)
    assert final["status"] == ("completed" if ready else "failed")
    assert final["report_status"] == ("ready" if ready else "generation_failed")
    assert final["finished_at"] >= final["research_finished_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_report_deadline_reaps_process_and_keeps_evidence(
    tmp_path, monkeypatch, cancel
):
    pidfile = tmp_path / "pid"
    original = asyncio.create_subprocess_exec
    children = []

    async def launch(*_, **kwargs):
        p = await original(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        children.append(p)
        pidfile.write_text(str(p.pid))
        return p

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    (tmp_path / "report.md").write_text("# Evidence")
    (tmp_path / "report.html").write_text("<article>Evidence</article>")
    (tmp_path / "report.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                run_id="test",
                attempt=1,
                status="ready",
                formats={"report.md": True, "report.html": True},
            )
        )
    )
    record = dict(
        run_id="test",
        attempt=1,
        launch_token="token",
        task_roots=[str(tmp_path)],
        manifest_payload={"report_budget_sec": 0.1},
    )
    pending = asyncio.create_task(reporting.finalize_reports(record))
    if cancel:
        while not children:
            await asyncio.sleep(0.001)
        pending.cancel()
    reports, timed_out = await pending
    assert timed_out
    assert reports[0]["status"] == "ready"
    assert children[0].returncode is not None
    assert not Path("/proc", pidfile.read_text()).exists()
    record["attempt"] = 2
    reports, _ = await reporting.finalize_reports(record)
    assert reports[0]["status"] == "generation_failed"


def test_report_retry_does_not_allocate_workers(tmp_path, monkeypatch):
    from scienceflow.runtime.parallel.control import service

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    record = dict(
        run_id="test",
        status="failed",
        report_status="generation_failed",
        research_finished_at=1,
        research_status="completed",
        manifest_payload={},
        attempt=1,
        task_roots=[],
        created_at=1,
    )
    write_record(run_dir("test"), record)
    monkeypatch.setattr(service, "_launch", lambda directory, row, **_: row)
    resumed = service.resume_run("test")
    assert resumed["recovery_mode"] == "report_only"
    assert resumed["resource_claims"] == {"cpu": [], "gpu": []}


def test_unscored_final_produces_markdown_and_html_without_network(
    tmp_path, monkeypatch
):
    from scienceflow.research.reporting import service as reports

    final = tmp_path / "merge/finals/final_00"
    final.mkdir(parents=True)
    (final / "eval_result.json").write_text(
        json.dumps(
            dict(
                candidate_id="final_00",
                metric_value=None,
                validation_ok=True,
                selection_eligible=False,
                evaluator_status="validated_no_metric",
            )
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Evidence report must not call the model or network")

    monkeypatch.setattr(reports, "enrich_report", forbidden)
    result = reports.generate_task_report(
        dict(run_id="test", attempt=1, status="completed"), tmp_path, enrich=False
    )
    assert result["status"] == "ready"
    assert all(result["formats"].values())
    text = (tmp_path / "report.md").read_text()
    assert "no independent numerical score" in text
    assert "ranked first" not in text


@pytest.mark.asyncio
async def test_real_report_child_generates_both_formats(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SCIENCEFLOW_REPORT_AGENT", "0")
    root = tmp_path / "task"
    root.mkdir()
    record = dict(
        run_id="test",
        launch_token="token",
        attempt=1,
        research_status="completed",
        research_finished_at=10,
        manifest_payload={"report_budget_sec": 15},
        task_roots=[str(root)],
    )
    write_record(run_dir("test"), record)
    results, timed_out = await reporting.finalize_reports(record)
    assert not timed_out
    assert results[0]["status"] == "ready"
    assert (root / "report.md").stat().st_size > 100
    assert (root / "report.html").stat().st_size > 100


def test_task_panel_shows_report_budget_and_freezes_research_time(
    tmp_path, monkeypatch
):
    from scienceflow.interfaces.ui.research.control.tasks import projection

    monkeypatch.setattr(projection.time, "time", lambda: 150)
    view = projection.TaskProjection()
    row = dict(
        run_id="test",
        attempt=1,
        alive=True,
        status="running",
        phase="reporting",
        started_at=10,
        research_finished_at=110,
        report_started_at=140,
        report_budget_sec=300,
        task_roots=[str(tmp_path)],
        draft={"wall_clock_sec": 100},
    )
    item = view.item({"number": 1, "name": "Nomad", "workspace": str(tmp_path)}, row)
    assert "Report · 10/300s" in item["detail"]
    assert "report " in item["progress_tail"]
    assert view.monitors[("test", 1)].final_elapsed == 100
    assert not view.monitors[("test", 1)].active
