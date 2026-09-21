"""One separately supervised monitor process per workspace."""

import json
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

from scienceflow.runtime.parallel.control.registry import locked


def directory(workspace):
    return Path(workspace).resolve() / ".scienceflow/web"


def read_service(workspace):
    try:
        row = json.loads((directory(workspace) / "service.json").read_text())
        process = psutil.Process(row["pid"])
        if (
            abs(process.create_time() - row["created"]) < 0.01
            and process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            return row
    except (OSError, ValueError, KeyError, psutil.Error):
        pass
    return None


def start(workspace, *, host="127.0.0.1", port=8765):
    if host not in {"127.0.0.1", "0.0.0.0"}:
        raise ValueError("Host must be 127.0.0.1 or 0.0.0.0")
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError("Workspace does not exist")
    target = directory(workspace)
    with locked(target):
        row = read_service(workspace)
        if row:
            return dict(row, reused=True)
        token = secrets.token_urlsafe(24)
        # Secret is passed through a private file, not the process command line.
        config = target / "launch.json"
        config.touch(mode=0o600, exist_ok=True)
        config.write_text(json.dumps({"host": host, "port": port, "token": token}))
        with (target / "server.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scienceflow.interfaces.ui.web_monitor.server",
                    str(workspace),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        for _ in range(100):
            row = read_service(workspace)
            if row and row["pid"] == process.pid:
                return dict(row, reused=False)
            if process.poll() is not None:
                raise ValueError(
                    f"Web service could not start; check {target / 'server.log'}"
                )
            time.sleep(0.1)
        process.terminate()
        process.wait(timeout=5)
        raise ValueError("Web service startup timed out")


def stop(workspace):
    with locked(directory(workspace)):
        row = read_service(workspace)
        if not row:
            return "No Web monitor is running for this workspace."
        url = f"http://127.0.0.1:{row['port']}/{row['token']}/stop"
        request = urllib.request.Request(url, data=b"", method="POST")
        # No proxy: the monitor is always on this host.
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=3
        ).close()
        for _ in range(30):
            if not read_service(workspace):
                break
            time.sleep(0.1)
        return "Web monitor stopping. Research tasks continue running."


def instructions(row, workspace):
    url = f"http://127.0.0.1:{row['port']}/{row['token']}/"
    port = row["port"]
    return (
        f"Web · {url}\n"
        "Local: open the link · Stop: /web stop\n"
        f"Remote: run locally: ssh -N -L {port}:127.0.0.1:{port} user@server\n"
        f"Replace user@server with your SSH login; or forward port {port} in VS Code Ports, then open the link."
    )
