from __future__ import annotations

import subprocess

from scienceflow.runtime.core.support import system_resources


def test_query_gpu_devices_exposes_identity_and_memory(monkeypatch) -> None:
    def fake_check_output(*args, **kwargs):
        return "0, GPU-aaaa, 81920, 73728\nmalformed row\n1, GPU-bbbb, 81920, 69632\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    devices = system_resources.query_gpu_devices()

    assert [device.index for device in devices] == [0, 1]
    assert devices[0].uuid == "GPU-aaaa"
    assert devices[0].memory_total_mib == 81920
    assert devices[0].memory_free_mib == 73728
    assert system_resources.query_gpu_free_memory() == {0: 73728, 1: 69632}


def test_query_gpu_devices_fails_closed_when_nvidia_smi_is_missing(monkeypatch) -> None:
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", raise_missing)

    assert system_resources.query_gpu_devices() == ()
    assert system_resources.query_gpu_free_memory() == {}
