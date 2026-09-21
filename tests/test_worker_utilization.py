"""Worker process attribution and bounded multi-GPU presentation."""

from types import SimpleNamespace

from scienceflow.interfaces.ui.research.telemetry.formatting import (
    gpu_summary,
    worker_resources,
)
from scienceflow.interfaces.ui.research.telemetry.sampler import UtilizationSampler


def test_worker_cpu_descendants_and_shared_gpu(monkeypatch, tmp_path):
    sampler = UtilizationSampler()
    monkeypatch.setattr(sampler, "refresh", lambda: None)
    root = tmp_path / "w00"
    sampler.processes = {
        10: {
            "ppid": 1,
            "cwd": str(root / "workspace"),
            "percent": 200,
            "create_time": 1,
        },
        11: {"ppid": 10, "cwd": "/tmp", "percent": 100, "create_time": 1},
        12: {"ppid": 1, "cwd": str(tmp_path / "w01"), "percent": 300, "create_time": 1},
    }
    monkeypatch.setattr(
        "scienceflow.interfaces.ui.research.telemetry.sampler.psutil.Process",
        lambda pid: SimpleNamespace(
            create_time=lambda: 1, cpu_affinity=lambda: [0, 1, 2, 3]
        ),
    )
    sampler.gpus = {
        "0": {"uuid": "GPU-a", "util": 80, "used": 1024, "total": 4096},
        "2": {"uuid": "GPU-b", "util": 20, "used": 512, "total": 4096},
    }
    sampler.gpu_apps = [(10, "GPU-a", 100), (12, "GPU-a", 100), (11, "GPU-b", 100)]
    sample = sampler.worker(root)
    assert sample["cpu"] == 75 and sample["cores"] == 4
    assert [gpu["id"] for gpu in sample["gpus"]] == ["0", "2"]
    assert sample["gpus"][0]["shared"] is True
    assert sample["gpus"][1]["shared"] is False
    assert "shared" in worker_resources(sample)
    assert sampler.worker(root) is sample


def test_first_sample_and_stopped_worker_do_not_invent_utilization(
    monkeypatch, tmp_path
):
    sampler = UtilizationSampler()
    monkeypatch.setattr(sampler, "refresh", lambda: None)
    sample = sampler.worker(tmp_path, assigned=["4"])
    assert sample["cpu"] is None
    assert sample["gpus"][0]["id"] == "4"
    assert "4:—" in worker_resources(sample)
    monkeypatch.setattr(
        sampler,
        "refresh",
        lambda: (_ for _ in ()).throw(AssertionError("must not sample")),
    )
    assert sampler.worker(tmp_path, active=False)["active"] is False


def test_many_gpus_are_compact_but_details_preserve_every_id():
    devices = [
        {"id": str(i), "util": 50, "used": 1024, "total": 2048} for i in range(32)
    ]
    summary = gpu_summary(devices)
    assert "[0–31]" in summary and "avg 50%" in summary
    assert len(summary) < 75 and "\n" not in summary
    assert "31:50%" in gpu_summary(devices, compact=False)


def test_host_resources_use_one_aggregate_line(monkeypatch, tmp_path):
    from scienceflow.interfaces.ui.research import resources

    sampler = SimpleNamespace(
        refresh=lambda **_kwargs: None,
        host_cpu=20,
        gpus={str(i): {"util": 50, "used": 1024, "total": 2048} for i in range(32)},
    )
    monkeypatch.setattr(resources, "SAMPLER", sampler)
    text = resources.ResourceMonitor(tmp_path).sample()
    assert "GPU ×32 avg 50%" in text and "VRAM" not in text
    assert "\n" not in text and "GPU0" not in text


def test_sampling_is_cached_and_host_cpu_uses_time_deltas(monkeypatch):
    from collections import namedtuple

    from scienceflow.interfaces.ui.research.telemetry import sampler as module

    clock = [0]
    cpu = namedtuple("CPU", "user system idle iowait")
    times = iter([cpu(10, 5, 85, 0), cpu(15, 5, 90, 0)])
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.psutil, "process_iter", lambda attrs: [])
    monkeypatch.setattr(module.psutil, "cpu_times", lambda: next(times))
    sampler = UtilizationSampler()
    queries = []
    monkeypatch.setattr(sampler, "_query", lambda *args: queries.append(args) or [])
    sampler.refresh()
    assert sampler.host_cpu is None and len(queries) == 2
    clock[0] = 4
    sampler.refresh()
    assert len(queries) == 2
    clock[0] = 6
    sampler.refresh()
    assert sampler.host_cpu == 50 and len(queries) == 2


def test_host_only_sampling_skips_process_inventory(monkeypatch):
    from collections import namedtuple

    from scienceflow.interfaces.ui.research.telemetry import sampler as module

    cpu = namedtuple("CPU", "user system idle iowait")
    monkeypatch.setattr(
        module.psutil,
        "process_iter",
        lambda _attrs: (_ for _ in ()).throw(AssertionError("process scan")),
    )
    monkeypatch.setattr(module.psutil, "cpu_times", lambda: cpu(10, 5, 85, 0))
    sampler = UtilizationSampler()
    monkeypatch.setattr(sampler, "_query", lambda *_args: [])

    sampler.refresh(include_processes=False)

    assert sampler.at is None
    assert sampler.host_times is not None
