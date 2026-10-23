"""Report recovery keeps workspace boundaries and publishes whole versions."""

import json
import select
import subprocess
import sys
from dataclasses import replace

import pytest

from scienceflow.research.reporting import service as reports
from scienceflow.research.reporting.storage import publication
from scienceflow.runtime.parallel.control import service
from scienceflow.runtime.parallel.control.registry import run_dir, write_record


@pytest.mark.parametrize("location", [True, False])
def test_report_resume_and_writer_reject_replaced_workspace(
    tmp_path, monkeypatch, location
):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    allocated = tmp_path / "allocated"
    other = tmp_path / "other"
    other.mkdir()
    allocated.symlink_to(other, target_is_directory=True)
    record = dict(
        run_id="test",
        created_at=1,
        status="failed",
        research_status="failed",
        research_finished_at=10,
        report_status="generation_failed",
        phase="finished",
        task_roots=[str(allocated / "runs/task")],
        control_workspace=str(allocated),
        manifest_payload={},
        attempt=1,
    )
    if location:
        record["execution_location"] = dict(
            backend="local", task_workspace=str(allocated)
        )
    write_record(run_dir("test"), record)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe workspace launched a report process")

    monkeypatch.setattr(service, "_launch", forbidden)
    with pytest.raises(ValueError, match="symlink|moved"):
        service.resume_run("test")
    result = reports.generate_run_reports(record, enrich=False)
    assert result[0]["status"] == "generation_failed"
    assert not list(other.iterdir())


def test_missing_leaf_within_valid_workspace_can_generate_report(tmp_path):
    record = dict(
        run_id="test",
        attempt=1,
        status="failed",
        control_workspace=str(tmp_path),
        execution_location=dict(backend="local", task_workspace=str(tmp_path)),
        task_roots=[str(tmp_path / "runs/new-task")],
    )
    result = reports.generate_run_reports(record, enrich=False)
    assert result[0]["status"] == "ready"
    assert (tmp_path / "runs/new-task/report.html").is_file()


def _record(root):
    return dict(run_id="test", attempt=1, status="completed", task_roots=[str(root)])


def _files(root):
    paths = [root / name for name in ("report.md", "report.html", "report.json")]
    paths.extend((root / "report_assets").glob("*.svg"))
    return {str(path.relative_to(root)): path.read_bytes() for path in paths}


@pytest.mark.parametrize("failure", ["copy", "switch"])
def test_interrupted_enrichment_preserves_complete_old_version(
    tmp_path, monkeypatch, failure
):
    record = _record(tmp_path)
    reports.generate_task_report(record, tmp_path, enrich=False)
    before = _files(tmp_path)
    monkeypatch.setattr(
        reports,
        "enrich_report",
        lambda record, report, **kw: replace(
            report,
            research_analysis={"interpretation": "NEW_ANALYSIS"},
            report_agent={"status": "ready", "model": "test"},
        ),
    )
    with monkeypatch.context() as fault:
        if failure == "copy":
            original = publication.shutil.copyfile

            def broken(source, target, *args, **kwargs):
                if target.name == "report.html" and target.is_relative_to(
                    tmp_path / ".report-versions"
                ):
                    raise OSError("injected copy failure")
                return original(source, target, *args, **kwargs)

            fault.setattr(publication.shutil, "copyfile", broken)
        else:
            original = publication._switch

            def broken(root, version):
                if root == tmp_path:
                    raise OSError("injected commit failure")
                return original(root, version)

            fault.setattr(publication, "_switch", broken)
        assert reports.generate_run_reports(record)[0]["status"] == "generation_failed"
    assert _files(tmp_path) == before
    assert reports.read_report_manifest(tmp_path)["status"] == "ready"
    reports.generate_task_report(record, tmp_path)
    assert "NEW_ANALYSIS" in (tmp_path / "report.md").read_text()
    assert "NEW_ANALYSIS" in (tmp_path / "report.html").read_text()


def test_legacy_regular_report_migrates_without_changing_old_bytes_on_failure(
    tmp_path, monkeypatch
):
    legacy = tmp_path / "legacy"
    assets = legacy / "report_assets"
    assets.mkdir(parents=True)
    (legacy / "report.md").write_text("# Original")
    (legacy / "report.html").write_text("<h1>Original</h1>")
    (legacy / "report.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                status="ready",
                formats={"report.md": True, "report.html": True},
            )
        )
    )
    (assets / "metric-history.svg").write_text("<svg/>")
    before = _files(legacy)
    original = publication._switch
    calls = []

    def interrupted(root, version):
        if root == legacy:
            calls.append(version)
            if len(calls) == 2:
                raise OSError("interrupted after legacy bootstrap")
        return original(root, version)

    monkeypatch.setattr(publication, "_switch", interrupted)
    assert (
        reports.generate_run_reports(_record(legacy), enrich=False)[0]["status"]
        == "generation_failed"
    )
    for name, content in before.items():
        assert (legacy / name).read_bytes() == content


def test_sigkill_before_version_switch_preserves_old_report(tmp_path):
    record = _record(tmp_path)
    reports.generate_task_report(record, tmp_path, enrich=False)
    before = _files(tmp_path)
    script = """
import sys,time
from pathlib import Path
from scienceflow.research.reporting import service
from scienceflow.research.reporting.storage import publication
root=Path(sys.argv[1])
original=publication._switch
def paused(target,version):
    if target==root:
        print('READY',flush=True)
        time.sleep(60)
    original(target,version)
publication._switch=paused
service.generate_task_report(dict(run_id='test',attempt=1,status='failed',error='NEW'),root,enrich=False)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert select.select([process.stdout], [], [], 15)[0]
        assert process.stdout.readline().strip() == "READY"
        process.kill()
        process.wait(timeout=5)
        assert _files(tmp_path) == before
        assert reports.read_report_manifest(tmp_path)["status"] == "ready"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_bad_task_root_does_not_suppress_safe_sibling_report(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    bad = tmp_path / "bad"
    bad.symlink_to(external, target_is_directory=True)
    good = tmp_path / "good"
    record = dict(
        run_id="test", attempt=1, status="failed", task_roots=[str(bad), str(good)]
    )
    result = reports.generate_run_reports(record, enrich=False)
    assert [row["status"] for row in result] == ["generation_failed", "ready"]
    assert not list(external.iterdir())


def test_report_reader_rejects_redirected_root_and_external_format(tmp_path):
    root = tmp_path / "task"
    reports.generate_task_report(_record(root), root, enrich=False)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert reports.read_report_manifest(alias) is None
    external = tmp_path / "outside.html"
    external.write_text("not a report")
    (root / "report.html").unlink()
    (root / "report.html").symlink_to(external)
    manifest = reports.read_report_manifest(root)
    assert manifest["status"] == "partial"
    assert not manifest["formats"]["report.html"]


@pytest.mark.asyncio
async def test_report_completion_rechecks_replaced_workspace(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from scienceflow.runtime.parallel.control import reporting

    root = tmp_path / "task"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()

    async def finished():
        root.rmdir()
        root.symlink_to(other, target_is_directory=True)
        return 0

    async def spawn(*args, **kwargs):
        return SimpleNamespace(returncode=0, wait=finished)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(ValueError, match="symlink|moved"):
        await reporting.finalize_reports(
            dict(
                run_id="test",
                launch_token="token",
                attempt=1,
                task_roots=[str(root)],
                manifest_payload={},
            )
        )
