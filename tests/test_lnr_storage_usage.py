"""Storage telemetry must stay cheap while a run is active and exact at exit."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scienceflow.research.solver.lnr.lifecycle.records import storage_usage
from scienceflow.research.solver.lnr.lifecycle.snapshots.snapshot_store import SnapshotStore


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_snapshot_samples_reuse_existing_counters_without_scanning(tmp_path, monkeypatch):
    def reject_scan(*_args, **_kwargs):
        raise AssertionError("snapshot telemetry must not scan directories")

    monkeypatch.setattr(storage_usage.os, "scandir", reject_scan)
    stats = {
        "logical_size_bytes": 3 * 1024**2,
        "new_physical_bytes": 1024**2,
        "captured_file_count": 4,
    }
    for worker in ("w00", "w01"):
        assert storage_usage.record_snapshot_storage(
            task_root_dir=tmp_path,
            worker_id=worker,
            stage_id="S01",
            snapshot_id=f"{worker}-s01",
            stats=stats,
        )
    # Replaying the same sample must not double count new object bytes.
    assert storage_usage.record_snapshot_storage(
        task_root_dir=tmp_path,
        worker_id="w00",
        stage_id="S01",
        snapshot_id="w00-s01",
        stats=stats,
    )

    latest = _read(tmp_path / "task_logs" / storage_usage.STORAGE_LATEST_FILENAME)
    assert latest["exact"] is False
    assert latest["total_bytes"] == 8 * 1024**2
    assert latest["workers"]["w00"]["cumulative_new_physical_bytes"] == 1024**2


def test_final_measurement_skips_dataset_explicit_inputs_and_symlink_targets(
    tmp_path, monkeypatch
):
    root = tmp_path / "task"
    generated = root / "workers" / "w00" / "workspace"
    generated.mkdir(parents=True)
    artifact = generated / "solution.bin"
    artifact.write_bytes(b"x" * 8192)
    os.link(artifact, generated / "same-object.bin")

    dataset = generated / "dataset"
    dataset.mkdir()
    (dataset / "large-input.bin").write_bytes(b"d" * 1024**2)
    configured_input = root / "raw_inputs"
    configured_input.mkdir()
    (configured_input / "large-input.bin").write_bytes(b"i" * 1024**2)
    external = tmp_path / "external"
    external.mkdir()
    (external / "large-input.bin").write_bytes(b"e" * 1024**2)
    (generated / "external-data").symlink_to(external, target_is_directory=True)

    real_scandir = storage_usage.os.scandir

    def guarded_scandir(path):
        assert Path(path).name not in {"dataset", "raw_inputs", "external"}
        return real_scandir(path)

    monkeypatch.setattr(storage_usage.os, "scandir", guarded_scandir)
    result = storage_usage.measure_generated_storage(
        root,
        excluded_paths=(configured_input,),
    )

    assert result["error_count"] == 0
    assert result["excluded_directory_count"] == 2
    assert result["symlink_count"] == 1
    assert result["file_count"] == 2  # one inode exposed through two hard links
    assert result["apparent_bytes"] < 256 * 1024


def test_snapshot_store_projects_real_workspace_snapshot_counters(tmp_path):
    task_root = tmp_path / "task"
    worker_root = task_root / "workers" / "w00"
    workspace = worker_root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=workspace,
        snapshot_dirname="snapshots",
        archive_dirname="snapshots/archives",
        control_log_dir=worker_root / "logs",
        strict_layout=True,
        workspace_snapshot_enabled=True,
        task_root_dir=task_root,
        worker_id="w00",
    )

    store.capture(
        stage_id="S01",
        metric_value=1.0,
        metric_name="score",
        lower_is_better=False,
        memory_cut=0,
        source_event={"event_type": "stage_completed"},
    )

    latest = _read(task_root / "task_logs" / storage_usage.STORAGE_LATEST_FILENAME)
    assert latest["source"] == "snapshot_counters"
    assert latest["workers"]["w00"]["captured_file_count"] == 1
    assert latest["total_bytes"] > 0


def test_final_sample_records_allocated_generated_storage(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    (root / "result.bin").write_bytes(b"r" * 8192)
    assert storage_usage.record_final_storage(task_root_dir=root)

    latest = _read(root / "task_logs" / storage_usage.STORAGE_LATEST_FILENAME)
    assert latest["source"] == "final_generated_storage_scan"
    assert latest["exact"] is True
    assert latest["total_bytes"] == latest["measurement"]["allocated_bytes"]
    assert latest["measurement"]["file_count"] >= 1
    events = (root / "task_logs" / storage_usage.STORAGE_EVENTS_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "storage_final_sampled" in events


def test_managed_worker_records_terminal_storage_after_outputs(tmp_path, monkeypatch):
    from scienceflow.runtime.parallel.control import worker

    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(worker, "record_final_storage", record)
    record_data = {
        "task_roots": [str(tmp_path / "task")],
        "draft": {"input_data_dir": str(tmp_path / "large-input")},
        "manifest_payload": {
            "defaults": {"input_data_dir": str(tmp_path / "default-input")},
            "tasks": [{"data_dir": str(tmp_path / "task-input")}],
        },
    }
    worker._record_terminal_storage(record_data)

    assert calls == [{
        "task_root_dir": str(tmp_path / "task"),
        "excluded_paths": (
            str(tmp_path / "large-input"),
            str(tmp_path / "default-input"),
            str(tmp_path / "task-input"),
        ),
    }]
