# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Launcher SIGHUP immunity preserved by _install_cleanup_signals.

The CLI cleanup-signal setup used to overwrite a launcher-installed SIG_IGN
disposition on SIGHUP (nohup / ``trap '' HUP``), silently disarming the
protection: a session-teardown SIGHUP then ran the child-sweep handler and
killed a long orchestrator run with no error signature. These tests pin the
fixed behavior, including an end-to-end subprocess proof.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scienceflow import cli


def _restore(sig: int, disposition: object) -> None:
    try:
        signal.signal(sig, disposition)  # type: ignore[arg-type]
    except (OSError, ValueError):
        pass


@pytest.fixture()
def pristine_signal_dispositions():
    """Restore SIGINT/SIGTERM/SIGHUP dispositions after each in-process test."""
    saved = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    yield
    for sig, disp in saved.items():
        _restore(sig, disp)


def test_sighup_ignored_at_startup_is_not_overwritten(
    pristine_signal_dispositions: None,
) -> None:
    """With SIGHUP already SIG_IGN (nohup), the installer must leave it alone."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    cli._install_cleanup_signals()

    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    # SIGINT / SIGTERM still get the sweep handler (they are not ignorable
    # through this mechanism — a launcher ignoring them is not the nohup case).
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_DFL
    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL


def test_sighup_default_still_gets_handler(
    pristine_signal_dispositions: None,
) -> None:
    """Without launcher immunity (SIGHUP at SIG_DFL), the sweep handler installs."""
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    cli._install_cleanup_signals()

    handler = signal.getsignal(signal.SIGHUP)
    assert callable(handler)
    assert handler is not signal.SIG_IGN


def test_nohup_like_child_survives_sighup_end_to_end() -> None:
    """End-to-end: a child that sets SIG_IGN (nohup semantics), runs the real
    installer, then receives SIGHUP — it must survive and report OK.

    Without the fix, _install_cleanup_signals replaces SIG_IGN with the sweep
    handler, which re-raises the signal after restoring SIG_DFL, killing the
    process (exit by SIGHUP, returncode -1 / 128+1).
    """
    repo_root = Path(cli.__file__).resolve().parents[1]
    script = (
        "import signal, sys, time\n"
        "sys.path.insert(0, %r)\n"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN)  # nohup semantics\n"
        "from scienceflow.cli import _install_cleanup_signals\n"
        "_install_cleanup_signals()\n"
        "signal.raise_signal(signal.SIGHUP)  # session teardown arrives\n"
        "time.sleep(0.2)\n"
        "print('SURVIVED')\n"
    % str(repo_root)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SURVIVED" in proc.stdout


def test_nohup_like_child_still_sweeps_children_on_sigterm() -> None:
    """The fix must not weaken SIGTERM: the handler still sweeps live PGIDs.

    A child process registers a live PGID, then SIGTERM arrives; the sweep
    handler should terminate the registered group and re-raise, so the parent
    exits by SIGTERM (negative returncode) and the grandchild is gone.
    """
    repo_root = Path(cli.__file__).resolve().parents[1]
    script = (
        "import os, signal, subprocess, sys, time\n"
        "sys.path.insert(0, %r)\n"
        "from scienceflow.cli import _install_cleanup_signals\n"
        "from scienceflow.core.subprocess_utils import _LIVE_PGIDS\n"
        "_install_cleanup_signals()\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "                        start_new_session=True)\n"
        "_LIVE_PGIDS.add(child.pid)\n"
        "print(child.pid, flush=True)\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
        "time.sleep(30)\n"
    % str(repo_root)
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    pid_line = proc.stdout.readline().strip()
    assert pid_line.isdigit(), f"child pid line missing: {pid_line!r} / {proc.stderr.read()}"
    grandchild_pid = int(pid_line)
    try:
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    # Parent died via SIGTERM (handler re-raised after the sweep).
    assert proc.returncode == -int(signal.SIGTERM)
    # Grandchild was swept away (killpg SIGTERM -> dead within the 5s window).
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        pytest.fail(f"grandchild {grandchild_pid} survived the SIGTERM sweep")
    # Reap to avoid a zombie in the test runner.
    try:
        os.waitpid(grandchild_pid, os.WNOHANG)
    except ChildProcessError:
        pass
