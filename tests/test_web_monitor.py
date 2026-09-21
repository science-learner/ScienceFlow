"""Read-only Web projection, ancestry and independent HTTP lifecycle."""

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from scienceflow.interfaces.ui.web_monitor.data import WorkspaceMonitor, lineage
from scienceflow.interfaces.ui.web_monitor.server import (
    DEFAULT_REFRESH_SEC,
    MonitorServer,
)


def test_page_bundles_packaged_assets_and_real_version():
    from importlib.metadata import PackageNotFoundError, version

    from scienceflow.interfaces.ui.web_monitor.page import PAGE

    try:
        expected = version("scienceflow")
    except PackageNotFoundError:
        expected = "dev"
    assert f"v{expected}</div>" in PAGE
    assert "function layoutLineage" in PAGE
    assert "function createLineageViewport" in PAGE
    assert "__SCIENCEFLOW_" not in PAGE
    assert "<script src=" not in PAGE
    assert "initialRefresh = true" in PAGE
    assert "Math.max(10000" in PAGE
    assert "function currentTaskRoute" in PAGE
    assert "Run monitor" in PAGE and "Research report" in PAGE
    assert "Download PDF" in PAGE
    assert DEFAULT_REFRESH_SEC == 10


def test_timestamp_projection_is_finite_and_handles_worker_case():
    from scienceflow.interfaces.ui.web_monitor.projection import timestamped_points

    point = SimpleNamespace(
        to_dict=lambda: {"worker_id": "W00", "stage_id": "S01", "candidate_id": ""}
    )
    assert (
        timestamped_points(
            [point], [{"worker_id": "w00", "stage_id": "S01", "timestamp_utc": 10}]
        )[0]["timestamp"]
        == 10
    )
    nodes = lineage(
        [
            {"worker_id": "w00", "stage_id": "S01", "timestamp_utc": float("nan")},
            {"worker_id": "w00", "stage_id": "S01", "timestamp_utc": float("inf")},
        ],
        [{"event": "estra_decision", "timestamp_utc": 1, "payload": "incomplete"}],
    )
    assert nodes[0]["id"] != nodes[1]["id"]
    assert all(node["timestamp"] is None for node in nodes)
    json.dumps(nodes, allow_nan=False)


def fetch(server, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(
        f"http://127.0.0.1:{server.server_port}/{path}", timeout=2
    ) as response:
        return response.read()


def test_lineage_preserves_explicit_parent_and_restore():
    nodes = lineage(
        [
            {"candidate_id": "a", "stage_id": "s1", "worker_id": "w00"},
            {"candidate_id": "b", "stage_id": "s2", "parent_candidate_id": "a"},
            {"candidate_id": "c", "stage_id": "s3", "restored_from_candidate_id": "a"},
            {"candidate_id": "d", "stage_id": "s4"},
        ]
    )
    assert nodes[1]["parent"] == "a"
    assert nodes[2]["restored"] == "a"
    assert nodes[2]["parent"] == nodes[3]["parent"] == ""


def test_lineage_resolves_parent_stage_within_worker():
    nodes = lineage(
        [
            {"node_uid": "W00:L01:S01", "worker_id": "W00", "stage_id": "S01"},
            {
                "node_uid": "W00:L01:S02",
                "worker_id": "W00",
                "stage_id": "S02",
                "parent_stage_id": "S01",
            },
        ]
    )

    assert nodes[1]["parent"] == "W00:L01:S01"


def test_lineage_estra_phases_are_worker_local_and_timestamped():
    rows = [
        {"worker_id": worker, "timestamp_utc": stamp, "stage_id": str(i)}
        for i, (worker, stamp) in enumerate([("w00", 10), ("w00", 30), ("w01", 30)])
    ]
    event = {"event": "estra_decision", "worker_id": "W00", "timestamp_utc": 20}
    nodes = lineage(rows, [event, event])
    assert [node["estra_phase"] for node in nodes] == [0, 1, None]
    assert [node["timestamp"] for node in nodes] == [10, 30, 30]
    assert lineage([{"worker_id": "w00"}], [event])[0]["estra_phase"] is None


def test_lineage_registers_lineage_starts_without_fake_metrics():
    rows = [
        {
            "candidate_id": "candidate-a",
            "node_uid": "W00:L01:S01",
            "worker_id": "W00",
            "stage_id": "S01",
            "lineage_id": "L01",
            "metric_value": "0.7",
            "timestamp_utc": 10,
        },
        {
            "node_uid": "W00:L02:S02",
            "worker_id": "W00",
            "stage_id": "S02",
            "lineage_id": "L02",
            "timestamp_utc": 25,
        },
        {
            "node_uid": "W00:L03:S03",
            "worker_id": "W00",
            "stage_id": "S03",
            "lineage_id": "L03",
            "timestamp_utc": 35,
        },
    ]
    events = [
        {
            "event": "estra_keep_current_compacted",
            "worker_id": "W00",
            "timestamp_utc": 20,
            "payload": {
                "previous_lineage_id": "L01",
                "new_lineage_id": "L02",
                "target_node_uid": "W00:L01:S01",
            },
        },
        {
            "event": "estra_stage_switched",
            "worker_id": "W00",
            "timestamp_utc": 30,
            "payload": {
                "previous_lineage_id": "L02",
                "new_lineage_id": "L03",
                "target_node_uid": "W00:L01:S01",
            },
        },
    ]

    nodes = lineage(rows, events)

    assert [node["lineage"] for node in nodes] == ["L01", "L02", "L02", "L03", "L03"]
    assert nodes[1] == {
        "id": "W00:L02:phase",
        "candidate": "",
        "worker": "W00",
        "stage": "",
        "lineage": "L02",
        "parent": "W00:L01:S01",
        "parent_stage": "",
        "restored": "",
        "metric": "",
        "valid": "",
        "timestamp": 20,
        "estra_phase": None,
        "kind": "lineage_start",
        "previous_lineage": "L01",
        "transition": "redirect",
    }
    assert nodes[3]["transition"] == "restore"
    assert nodes[3]["restored"] == "W00:L01:S01"
    assert nodes[2]["parent"] == "W00:L02:phase"
    assert nodes[2]["restored"] == ""
    assert nodes[4]["parent"] == "W00:L03:phase"


def test_lineage_deduplicates_replayed_transition_events():
    event = {
        "event": "estra_keep_current_compacted",
        "worker_id": "W00",
        "timestamp_utc": 20,
        "payload": {"new_lineage_id": "L02"},
    }

    nodes = lineage(
        [{"worker_id": "W00", "stage_id": "S02", "lineage_id": "L02"}],
        [event, event],
    )

    starts = [node for node in nodes if node["kind"] == "lineage_start"]
    assert len(starts) == 1
    assert starts[0]["id"] == "W00:L02:phase"


def test_lineage_omits_transitions_without_a_formal_stage():
    rows = [
        {
            "worker_id": "W00",
            "stage_id": "S01",
            "lineage_id": "L01",
            "node_uid": "W00:L01:S01",
        }
    ]
    event = {
        "event": "estra_keep_current_compacted",
        "worker_id": "W00",
        "payload": {"new_lineage_id": "L02"},
    }

    nodes = lineage(rows, [event])

    assert [node["id"] for node in nodes] == ["W00:L01:S01"]


def test_current_event_timestamp_and_resource_category():
    from scienceflow.interfaces.ui.monitor_trace.presentation.builder import (
        _event_from_json,
    )

    event = _event_from_json(
        {
            "event": "resource_gpu_lease_acquired",
            "timestamp_utc": "2026-09-12T12:00:00Z",
            "payload": {},
        },
        x_mode="wall_clock",
    )
    assert event.x_value is not None
    assert event.kind == "resource"


def test_trace_events_merge_research_and_resource_sources_by_time(tmp_path):
    from scienceflow.interfaces.ui.monitor_trace.presentation.builder import (
        _load_trace_events,
    )

    logs = tmp_path / "task_logs"
    resources = logs / "resource"
    resources.mkdir(parents=True)
    (logs / "lhr_events.jsonl").write_text(
        json.dumps(
            {
                "event": "estra_decision",
                "timestamp_utc": "2026-09-12T12:00:20Z",
            }
        )
        + "\n"
    )
    (resources / "resource_events.jsonl").write_text(
        json.dumps(
            {
                "event": "resource_gpu_lease_acquired",
                "timestamp_utc": "2026-09-12T12:00:10Z",
            }
        )
        + "\n"
    )

    events = _load_trace_events(tmp_path, x_mode="wall_clock")

    assert [event.kind for event in events] == ["resource", "estra"]
    assert [event.x_value for event in events] == sorted(
        event.x_value for event in events
    )


def test_service_reuses_process_and_stop_isolated(tmp_path):
    from scienceflow.interfaces.ui.web_monitor.service import read_service, start, stop

    first = start(tmp_path, port=0)
    try:
        second = start(tmp_path, port=0)
        assert second["pid"] == first["pid"] and second["reused"]
        assert first["port"] > 0
    finally:
        stop(tmp_path)
    assert read_service(tmp_path) is None


def test_port_conflict_selects_another_port(tmp_path):
    from scienceflow.interfaces.ui.web_monitor.service import read_service, start, stop

    first_workspace, second_workspace = tmp_path / "one", tmp_path / "two"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first = start(first_workspace, port=0)
    try:
        second = start(second_workspace, port=first["port"])
        try:
            assert first["port"] != second["port"]
            stop(first_workspace)
            assert read_service(second_workspace)["pid"] == second["pid"]
        finally:
            stop(second_workspace)
    finally:
        if read_service(first_workspace):
            stop(first_workspace)


@pytest.mark.asyncio
async def test_tui_web_bypasses_research_preparation(tmp_path, monkeypatch):
    from scienceflow.interfaces.ui.long_research import LongResearchInteraction
    from scienceflow.interfaces.ui.web_monitor import service

    messages = []

    class Host:
        async def notice(self, text):
            messages.append(text)

        async def echo_user(self, text):
            pass

    monkeypatch.setattr(
        service,
        "start",
        lambda path: {"port": 9876, "token": "test", "host": "127.0.0.1"},
    )
    interaction = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    assert await interaction.try_handle("/web", Host())
    assert interaction.session is None
    assert "9876" in messages[-1] and "ssh -N -L" in messages[-1]

    class RichHost(Host):
        async def notice_rich(self, content):
            messages.append(content)

    assert await interaction.try_handle("/web", RichHost())
    assert "http://127.0.0.1:9876/test/" in messages[-1].plain
    assert len(messages[-1].plain.splitlines()) == 4
    assert "ssh -N -L 9876:127.0.0.1:9876 user@server" in messages[-1].plain
    assert any(
        "http://127.0.0.1:9876/test/" in str(span.style) for span in messages[-1].spans
    )


def test_workspace_read_does_not_write_task_index(tmp_path, monkeypatch):
    from scienceflow.interfaces.ui.web_monitor import data

    monkeypatch.setattr(data, "list_runs", list)
    monitor = WorkspaceMonitor(tmp_path)
    monkeypatch.setattr(monitor.resources, "sample", lambda: "Host test")
    result = monitor.collect()
    assert result["tasks"] == []
    assert result["resources"] == "Host test"
    assert not (tmp_path / ".scienceflow").exists()


def test_workspace_snapshot_loads_history_once_and_polls_only_active_runs(
    tmp_path, monkeypatch
):
    from scienceflow.interfaces.ui.research.control.tasks.records import TaskIndex
    from scienceflow.interfaces.ui.web_monitor import data

    index = TaskIndex(tmp_path)
    first = index.reserve("history")
    second = index.reserve("active")
    index.update(first["number"], run_id="done", status="completed")
    index.update(second["number"], run_id="live", status="running")
    done = {"run_id": "done", "status": "completed", "alive": False, "created_at": 1}
    live = {"run_id": "live", "status": "running", "alive": True, "created_at": 2}
    finished = dict(live, status="completed", alive=False)
    registry_calls = []
    active_calls = []
    monkeypatch.setattr(
        data,
        "list_runs",
        lambda: registry_calls.append(True) or [live, done],
    )
    monkeypatch.setattr(
        data,
        "get_run",
        lambda run_id: active_calls.append(run_id) or finished,
    )
    monitor = WorkspaceMonitor(tmp_path)
    monkeypatch.setattr(monitor.resources, "sample", lambda: "Host test")
    monkeypatch.setattr(
        monitor.projection,
        "item",
        lambda entry, row: {
            "monitor": {"name": entry["name"], "status": row["status"]}
        },
    )

    monitor.collect()
    monitor.collect()
    monitor.collect()

    assert len(registry_calls) == 1
    assert active_calls == ["live"]


def test_open_task_detail_rebuilds_as_soon_as_trace_sources_change(
    tmp_path, monkeypatch
):
    from scienceflow.interfaces.ui.web_monitor import data

    root = tmp_path / "task"
    source = root / "task_logs/lhr_state.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    monitor = WorkspaceMonitor(tmp_path)
    monitor.rows["1"] = {
        "run_id": "run",
        "attempt": 1,
        "alive": True,
        "status": "running",
        "task_roots": [str(root)],
        "started_at": 1,
    }
    calls = []

    def trace(**_):
        calls.append(len(calls) + 1)
        return SimpleNamespace(
            points=[],
            events=[],
            metric_name=f"metric-{calls[-1]}",
            lower_is_better=False,
        )

    monkeypatch.setattr(data, "build_task_trace", trace)
    first = monitor.detail("1")
    source.write_text('{"changed": true}')
    second = monitor.detail("1")

    assert first["metric"] == "metric-1"
    assert second["metric"] == "metric-2"
    assert len(calls) == 2


def test_open_task_lineage_refreshes_when_a_formal_stage_is_registered(
    tmp_path, monkeypatch
):
    from scienceflow.interfaces.ui.web_monitor import data

    root = tmp_path / "task"
    logs = root / "task_logs"
    logs.mkdir(parents=True)
    stages = logs / "lhr_stage_performance.csv"
    stages.write_text(
        "worker_id,stage_id,lineage_id,node_uid,timestamp_utc\n"
        "W00,S01,L01,W00:L01:S01,10\n"
    )
    (logs / "lhr_events.jsonl").write_text(
        json.dumps(
            {
                "event": "estra_keep_current_compacted",
                "worker_id": "W00",
                "timestamp_utc": 20,
                "payload": {
                    "new_lineage_id": "L02",
                    "target_node_uid": "W00:L01:S01",
                },
            }
        )
        + "\n"
    )
    monitor = WorkspaceMonitor(tmp_path)
    monitor.rows["1"] = {
        "run_id": "run",
        "attempt": 1,
        "alive": True,
        "status": "running",
        "task_roots": [str(root)],
        "started_at": 1,
    }
    monkeypatch.setattr(
        data,
        "build_task_trace",
        lambda **_: SimpleNamespace(
            points=[], events=[], metric_name="radius", lower_is_better=False
        ),
    )

    first = monitor.detail("1")
    with stages.open("a") as stream:
        stream.write("W00,S02,L02,W00:L02:S02,25\n")
    second = monitor.detail("1")

    assert [node["lineage"] for node in first["lineage"]] == ["L01"]
    assert [node["lineage"] for node in second["lineage"]] == [
        "L01",
        "L02",
        "L02",
    ]


def test_terminal_task_detail_reuses_final_projection_without_source_stats(
    tmp_path,
    monkeypatch,
):
    from scienceflow.interfaces.ui.web_monitor import data

    root = tmp_path / "task"
    source = root / "task_logs/lhr_state.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    monitor = WorkspaceMonitor(tmp_path)
    monitor.rows["1"] = {
        "run_id": "run",
        "attempt": 1,
        "alive": False,
        "status": "completed",
        "task_roots": [str(root)],
        "started_at": 1,
    }
    calls = []

    def trace(**_):
        calls.append(True)
        return SimpleNamespace(
            points=[],
            events=[],
            metric_name="metric",
            lower_is_better=False,
        )

    monkeypatch.setattr(data, "build_task_trace", trace)
    first = monitor.detail("1")
    source.unlink()
    second = monitor.detail("1")

    assert second is first
    assert len(calls) == 1
    assert monitor.detail_needs_refresh("1") is False


def test_running_detail_gets_one_final_refresh_after_task_stops(tmp_path, monkeypatch):
    from scienceflow.interfaces.ui.web_monitor import data

    root = tmp_path / "task"
    source = root / "task_logs/lhr_state.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    monitor = WorkspaceMonitor(tmp_path)
    row = {
        "run_id": "run",
        "attempt": 1,
        "alive": True,
        "status": "running",
        "task_roots": [str(root)],
        "started_at": 1,
    }
    monitor.rows["1"] = row
    calls = []

    def trace(**_):
        calls.append(True)
        return SimpleNamespace(
            points=[],
            events=[],
            metric_name=f"metric-{len(calls)}",
            lower_is_better=False,
        )

    monkeypatch.setattr(data, "build_task_trace", trace)
    monitor.detail("1")
    row.update(alive=False, status="completed", finished_at=2)
    source.write_text('{"final": true}')

    assert monitor.detail_needs_refresh("1") is True
    final = monitor.detail("1")
    assert final["metric"] == "metric-2"
    assert monitor.detail_needs_refresh("1") is False
    assert monitor.detail("1") is final
    assert len(calls) == 2


def test_http_uses_cached_snapshot_and_rejects_files(tmp_path):
    server = MonitorServer(("127.0.0.1", 0), tmp_path, "test-token")
    server.snapshot = {"tasks": [{"key": "1", "name": "demo"}]}
    server.details = {"1": {"loading": True}}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert json.loads(fetch(server, "test-token/api")) == server.snapshot
        assert json.loads(fetch(server, "test-token/task/1")) == {"loading": True}
        assert "1" in server.requested
        for path in ("api", "test-token/../../.env", "wrong/api"):
            with pytest.raises(urllib.error.HTTPError) as error:
                fetch(server, path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_serves_only_generated_report_formats(tmp_path):
    from scienceflow.research.reporting import generate_task_report

    root = tmp_path / "task"
    root.mkdir()
    record = {
        "run_id": "run",
        "attempt": 1,
        "alive": False,
        "status": "completed",
        "task_roots": [str(root)],
        "draft": {"exp_id": "demo", "wall_clock_sec": 60},
    }
    generate_task_report(record, root)
    server = MonitorServer(("127.0.0.1", 0), tmp_path, "test-token")
    server.snapshot = {"tasks": [{"key": "1", "name": "demo"}]}
    server.monitor.rows["1"] = record
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        metadata = json.loads(fetch(server, "test-token/task/1/report"))
        assert metadata["available"] is True
        assert fetch(server, "test-token/task/1/report.pdf").startswith(b"%PDF-")
        with opener.open(
            f"http://127.0.0.1:{server.server_port}/test-token/", timeout=2
        ) as response:
            assert (
                "script-src 'unsafe-inline'"
                in response.headers["Content-Security-Policy"]
            )
        with opener.open(
            f"http://127.0.0.1:{server.server_port}/test-token/task/1/report.html",
            timeout=2,
        ) as response:
            assert "script-src" not in response.headers["Content-Security-Policy"]
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/test-token/task/1/report.md"
        )
        with opener.open(request, timeout=2) as response:
            assert response.headers["Content-Disposition"].startswith("attachment;")
        for path in (
            "test-token/task/1/report.json",
            "test-token/task/1/report.pdf/../../.env",
        ):
            with pytest.raises(urllib.error.HTTPError) as error:
                fetch(server, path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_report_endpoint_rejects_active_task_and_generates_history(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    row = {
        "run_id": "run",
        "attempt": 1,
        "alive": True,
        "status": "running",
        "task_roots": [str(root)],
        "draft": {"exp_id": "demo", "wall_clock_sec": 60},
    }
    server = MonitorServer(("127.0.0.1", 0), tmp_path, "test-token")
    server.snapshot = {"tasks": [{"key": "1", "name": "demo"}]}
    server.monitor.rows["1"] = row
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/test-token/task/1/report"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(urllib.request.Request(url, method="POST"), timeout=5)
        assert error.value.code == 409

        row.update(alive=False, status="failed", finished_at=time.time())
        with opener.open(
            urllib.request.Request(url, method="POST"), timeout=10
        ) as response:
            result = json.loads(response.read())
        assert result["available"] is True
        assert result["task_status"] == "failed"
        assert (root / "report.md").is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_terminal_open_task_is_loaded_once_then_never_rescanned(tmp_path):
    server = MonitorServer(("127.0.0.1", 0), tmp_path, "test-token")
    server.snapshot = {"tasks": [{"key": "1", "name": "done"}]}
    server.monitor.rows["1"] = {"alive": False, "status": "completed"}
    calls = []

    def detail(key):
        calls.append(key)
        return {"metric": "final"}

    server.monitor.detail = detail
    server.monitor.detail_needs_refresh = lambda key: not calls
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first = json.loads(fetch(server, "test-token/task/1"))
        second = json.loads(fetch(server, "test-token/task/1?fresh=1"))
        assert first == second == {"metric": "final"}
        assert calls == ["1"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_active_open_task_rescans_on_explicit_browser_reload(tmp_path):
    server = MonitorServer(("127.0.0.1", 0), tmp_path, "test-token")
    server.snapshot = {"tasks": [{"key": "1", "name": "running"}]}
    server.monitor.rows["1"] = {"alive": True, "status": "running"}
    calls = []

    def detail(key):
        calls.append(key)
        return {"revision": len(calls)}

    server.monitor.detail = detail
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first = json.loads(fetch(server, "test-token/task/1?fresh=1"))
        second = json.loads(fetch(server, "test-token/task/1?fresh=1"))
        assert first == {"revision": 1}
        assert second == {"revision": 2}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_background_detail_refreshes_only_requested_nonterminal_tasks(
    tmp_path,
    monkeypatch,
):
    from scienceflow.interfaces.ui.web_monitor import server as server_module

    server = MonitorServer(("127.0.0.1", 0), tmp_path, "test-token")
    now = time.monotonic()
    server.requested = {"1": now, "2": now}
    server.monitor.rows = {
        "1": {"alive": True, "status": "running"},
        "2": {
            "run_id": "done",
            "attempt": 1,
            "alive": False,
            "status": "completed",
            "finished_at": 10,
        },
        "3": {"alive": True, "status": "running"},
    }
    server.monitor.terminal_details["2"] = server.monitor._terminal_detail_signature(
        "2"
    )
    server.monitor.collect = lambda: {"tasks": [], "updated_at": time.time()}
    calls = []
    server.monitor.detail = lambda key: calls.append(key) or {"key": key}

    async def finish(_):
        server.stopping.set()

    monkeypatch.setattr(server_module.asyncio, "sleep", finish)
    try:
        await server.refresh()
        assert calls == ["1"]
    finally:
        server.server_close()


def test_browser_reload_synchronously_refreshes_summary_and_open_task(tmp_path):
    server = MonitorServer(("127.0.0.1", 0), tmp_path, "test-token")
    server.snapshot = {"tasks": [{"key": "1", "name": "old"}]}
    calls = []

    def collect():
        calls.append("summary")
        return {
            "tasks": [{"key": "1", "name": "fresh"}],
            "updated_at": time.time(),
        }

    def detail(key):
        calls.append("detail:" + key)
        return {"metric": "fresh"}

    server.monitor.collect = collect
    server.monitor.detail = detail
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        summary = json.loads(fetch(server, "test-token/api?fresh=1"))
        opened = json.loads(fetch(server, "test-token/task/1?fresh=1"))
        assert summary["tasks"][0]["name"] == "fresh"
        assert summary["refresh_sec"] >= DEFAULT_REFRESH_SEC
        assert opened == {"metric": "fresh"}
        assert calls == ["summary", "detail:1"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_slow_scan_does_not_block_async_loop_or_http(tmp_path):
    server = MonitorServer(("127.0.0.1", 0), tmp_path, "token")
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(timeout=3)
        return {"tasks": [], "updated_at": time.time()}

    server.monitor.collect = slow
    http = threading.Thread(target=server.serve_forever, daemon=True)
    http.start()
    task = asyncio.create_task(server.refresh())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        before = time.monotonic()
        result = await asyncio.to_thread(fetch, server, "token/api")
        assert json.loads(result)["loading"]
        assert time.monotonic() - before < 1
    finally:
        release.set()
        server.stopping.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(server.shutdown)
        server.server_close()


def test_shared_tui_projection_values(tmp_path, monkeypatch):
    from scienceflow.interfaces.ui.research.control.tasks.projection import (
        TaskProjection,
    )
    from scienceflow.interfaces.ui.web_monitor import data

    root = tmp_path / "run"
    root.mkdir()
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "demo": {
                        "model": "demo",
                        "pricing": {
                            "input_usd_per_1m": 0.15,
                            "cached_input_usd_per_1m": 0.003,
                            "output_usd_per_1m": 0.60,
                        },
                        "endpoints": [{"url": "https://example.test/v1", "key": "x"}],
                    }
                },
                "defaults": {"code_models": ["demo"]},
            }
        ),
        encoding="utf-8",
    )
    logs = root / "workers/w00/logs"
    logs.mkdir(parents=True)
    (logs / "lhr_state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "global_best": {"validation_ok": True, "metric_value": 2.5},
            }
        )
    )
    events = [
        {"event": "estra_decision", "payload": {}},
        {"event": "resource_job_started", "payload": {"job_id": "job"}},
        {
            "event": "evaluator_metric_event",
            "payload": {
                "candidate_id": "a",
                "artifact_sha": "hash",
                "evaluator_status": "ok",
                "selection_eligible": True,
                "metric_value": 2.5,
            },
        },
    ]
    (logs / "lhr_events.jsonl").write_text(
        "".join(
            json.dumps(
                {**event, "worker_id": "w00", "timestamp_utc": "2026-09-12T02:00:00Z"}
            )
            + "\n"
            for event in events
        )
    )
    audit = logs / "agent_runtime_audit/agent_provider_calls.jsonl"
    audit.parent.mkdir()
    audit.write_text(
        json.dumps(
            {
                "phase": "completed",
                "worker_id": "w00",
                "call_seq": 1,
                "model": "demo",
                "timestamp": 0,
                "tokens_input": 1000,
                "tokens_output": 100,
                "tokens_cached": 500,
                "llm_cost_usd": 0.25,
            }
        )
        + "\n"
    )
    row = {
        "run_id": "run",
        "task_roots": [str(root)],
        "control_workspace": str(tmp_path),
        "draft": {
            "workspace_base": str(tmp_path),
            "exp_id": "demo",
            "run_id": "run",
            "model_config_path": str(model_config),
            "workers": 2,
            "wall_clock_sec": 60,
            "metric_name": "radius",
            "lower_is_better": False,
        },
        "attempt": 1,
        "alive": False,
        "status": "completed",
        "created_at": 100,
        "started_at": 100,
        "finished_at": 160,
    }
    entry = {"number": 1, "run_id": "run", "name": "demo", "status": "completed"}
    monkeypatch.setattr(data, "list_runs", lambda: [row])
    monitor = WorkspaceMonitor(tmp_path)
    monkeypatch.setattr(monitor.resources, "sample", lambda: "Host test")
    web = monitor.collect()["tasks"][0]
    tui = TaskProjection().item(entry, row)["monitor"]
    assert web["best"] == "2.5" and web["evaluated"] == web["valid"] == 1
    assert web["eec_total"] == web["estra"]["estra_decision"] == 1
    assert "Cache 50%" in web["usage"] and web["cost"].startswith("~$")
    for key in (
        "status",
        "elapsed_sec",
        "best",
        "evaluated",
        "valid",
        "estra",
        "eec_total",
        "usage",
        "cost",
    ):
        assert web[key] == tui[key]
