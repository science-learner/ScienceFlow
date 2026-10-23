"""Managed entry points share validation, reservations and registry freshness."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from scienceflow.interfaces.cli import main
from scienceflow.interfaces.ui.research.control.tasks import manager
from scienceflow.runtime.parallel.control import registry, resource_claims


@pytest.mark.parametrize(
    "option,value",
    [
        ("--workers", "4"),
        ("--duration", "2h"),
        ("--cpu", "0-7"),
        ("--gpu", "cpu"),
        ("--config", "custom.yaml"),
        ("--input-data-dir", "data"),
    ],
)
def test_manifest_rejects_silently_ignored_overrides(
    tmp_path, monkeypatch, option, value
):
    from scienceflow.interfaces.cli.commands.run import control

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("tasks: []\n")
    launch = Mock()
    monkeypatch.setattr(control, "managed_run", launch)
    result = CliRunner().invoke(
        main, ["run", "--manifest", str(manifest), option, value]
    )
    assert result.exit_code == 2
    assert "Cannot combine --manifest" in result.output
    assert option in result.output
    launch.assert_not_called()


def test_batch_reserves_all_resolved_task_pools(monkeypatch):
    monkeypatch.setattr(resource_claims.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    tasks = [
        SimpleNamespace(cpu_list=cpu, gpu_list_raw="cpu", lnr_patch={})
        for cpu in ("0-1", "2-3")
    ]
    record = {
        "resource_requests": resource_claims.task_requests(tasks),
        "manifest_payload": {"tasks": [{}, {}]},
    }
    resource_claims.apply_reservation(record, [])
    assert record["resource_claims"] == {"cpu": [0, 1, 2, 3], "gpu": []}
    with pytest.raises(resource_claims.ResourceBusy):
        resource_claims.reserve_resources({"cpu_list": "3"}, [dict(record, alive=True)])


def test_registry_change_discovers_external_resume_without_rescanning_history(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    row = {"run_id": "demo", "status": "stopped", "alive": False, "created_at": 1}
    listing = Mock(side_effect=lambda: [dict(row)])
    reading = Mock(side_effect=lambda _: dict(row))
    monkeypatch.setattr(manager, "list_runs", listing)
    monkeypatch.setattr(manager, "get_run", reading)
    registry.write_record(registry.run_dir("demo"), row)
    assert tasks._load_rows()[0]["status"] == "stopped"
    tasks._load_rows()
    assert listing.call_count == 1
    reading.assert_not_called()

    row.update(status="running", alive=True, attempt=2)
    registry.write_record(registry.run_dir("demo"), row)
    assert tasks._load_rows()[0]["status"] == "running"
    assert listing.call_count == 2
    tasks._load_rows()
    assert reading.call_count == 1


def test_session_cleanup_preserves_other_process_workspaces(tmp_path, monkeypatch):
    from tests import conftest

    own = tmp_path / "session-own"
    other = tmp_path / "session-other"
    own.mkdir()
    other.mkdir()
    (other / "models.json").write_text("{}")
    monkeypatch.setattr(conftest, "TEST_WORKSPACE_ROOT", own)
    monkeypatch.delenv("SCIENCEFLOW_TEST_KEEP_TMP", raising=False)
    monkeypatch.delenv("SCIENCEFLOW_TEST_CLEAN_REPO_ROOT", raising=False)
    conftest.pytest_sessionfinish(None, 0)
    assert not own.exists()
    assert (other / "models.json").read_text() == "{}"


@pytest.mark.asyncio
async def test_idle_monitor_notices_external_registry_changes(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(manager, "_POLL_INTERVAL_SEC", 0.01)
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    tasks._registry_loaded = True
    tasks._registry_revision = registry.registry_revision()
    tasks.catalog_at = 0
    refreshed = asyncio.Event()

    async def refresh(_host):
        tasks._registry_revision = registry.registry_revision()
        refreshed.set()

    monkeypatch.setattr(tasks, "refresh", refresh)
    watcher = asyncio.create_task(tasks.watch(SimpleNamespace()))
    try:
        await asyncio.sleep(0.03)
        assert not refreshed.is_set()
        registry.write_record(registry.run_dir("external"), {"status": "starting"})
        await asyncio.wait_for(refreshed.wait(), timeout=5)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


def test_auto_gpu_reservation_can_move_on_resume(monkeypatch):
    monkeypatch.setattr(
        resource_claims,
        "query_gpu_devices",
        lambda: [
            SimpleNamespace(index=0),
            SimpleNamespace(index=1),
        ],
    )
    request = {"cpu_list": "", "gpu_list": "auto:1", "workers": 1}
    record = {
        "draft": dict(request),
        "resource_requests": [dict(request)],
        "manifest_payload": {"tasks": [{"gpu_list": "auto:1"}]},
    }
    resource_claims.apply_reservation(record, [])
    assert record["resource_claims"]["gpu"] == ["0"]
    occupied = {"alive": True, "resource_claims": {"cpu": [], "gpu": ["0"]}}
    resource_claims.apply_reservation(record, [occupied])
    assert record["resource_claims"]["gpu"] == ["1"]
    assert record["manifest_payload"]["tasks"][0]["gpu_list"] == "1"
    assert record["draft"]["gpu_list"] == "1"
    assert record["resource_requests"][0]["gpu_list"] == "auto:1"
