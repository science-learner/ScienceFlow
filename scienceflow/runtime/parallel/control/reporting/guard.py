"""Child-owned deadline and parent-death protection before report imports."""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager


def _terminate() -> None:
    # This entry point owns a dedicated session; never kill a caller's group.
    if os.getpgrp() == os.getpid():
        os.killpg(os.getpid(), signal.SIGKILL)
    os._exit(124)


@contextmanager
def report_guard(parent_pid: int, budget: float):
    """Also enforce liveness when the supervisor cannot execute its finally."""
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        # Linux delivers this even while Python is blocked in a native call.
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot arm report parent-death signal")
    if os.getppid() != parent_pid:
        _terminate()
    # OS timer also terminates a child blocked in native code with the GIL held.
    previous_alarm = signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, budget)
    deadline = time.monotonic() + budget
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(min(0.1, max(0, deadline - time.monotonic()))):
            if os.getppid() != parent_pid or time.monotonic() >= deadline:
                _terminate()

    thread = threading.Thread(target=watch, name="report-deadline", daemon=True)
    thread.start()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        stopped.set()
        thread.join()
