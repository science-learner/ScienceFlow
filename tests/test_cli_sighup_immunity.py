# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

"""Regression tests for launcher-provided SIGHUP immunity."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scienceflow.interfaces import cli


def _restore(sig: int, disposition: object) -> None:
    try:
        signal.signal(sig, disposition)  # type: ignore[arg-type]
    except (OSError, ValueError):
        pass


@pytest.fixture()
def pristine_signal_dispositions():
    saved = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    yield
    for sig, disposition in saved.items():
        _restore(sig, disposition)


def test_sighup_ignored_at_startup_is_not_overwritten(
    pristine_signal_dispositions: None,
) -> None:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    cli._install_cleanup_signals()

    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    assert callable(signal.getsignal(signal.SIGINT))
    assert callable(signal.getsignal(signal.SIGTERM))


def test_sighup_default_still_gets_cleanup_handler(
    pristine_signal_dispositions: None,
) -> None:
    signal.signal(signal.SIGHUP, signal.SIG_DFL)

    cli._install_cleanup_signals()

    assert callable(signal.getsignal(signal.SIGHUP))


def test_nohup_like_child_survives_sighup_end_to_end() -> None:
    repo_root = Path(cli.__file__).resolve().parents[3]
    script = (
        "import signal, sys, time\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
        "from scienceflow.interfaces.cli import _install_cleanup_signals\n"
        "_install_cleanup_signals()\n"
        "signal.raise_signal(signal.SIGHUP)\n"
        "time.sleep(0.2)\n"
        "print('SURVIVED')\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    assert "SURVIVED" in proc.stdout
