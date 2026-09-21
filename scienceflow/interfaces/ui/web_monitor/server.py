"""HTTP requests serve cached snapshots; asynchronous scans run outside the TUI."""

import asyncio
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import psutil

from .data import WorkspaceMonitor
from .page import PAGE
from .service import directory

DEFAULT_REFRESH_SEC = 10.0


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, workspace, token):
        super().__init__(address, Handler)
        self.token = token
        self.monitor = WorkspaceMonitor(workspace)
        self.snapshot = {"tasks": [], "updated_at": None, "loading": True}
        self.details = {}
        self.requested = {}
        self.request_lock = threading.Lock()
        self.scan_lock = threading.Lock()
        self.report_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.stopping = threading.Event()

    def collect_snapshot(self, *, discover=False):
        """Serialize projections so a browser refresh can safely request fresh data."""
        with self.scan_lock:
            if discover:
                self.monitor.registry_loaded = False
            return self.monitor.collect()

    def collect_detail(self, key):
        with self.scan_lock:
            try:
                detail = self.monitor.detail(key)
            except (OSError, ValueError, TypeError, KeyError):
                detail = {"error": "Task detail unavailable; retrying"}
            with self.state_lock:
                self.details[key] = detail
            return detail

    def detail_needs_refresh(self, key):
        with self.scan_lock:
            return self.monitor.detail_needs_refresh(key)

    def publish_snapshot(self, snapshot, *, started):
        elapsed = time.monotonic() - started
        refresh_sec = max(DEFAULT_REFRESH_SEC, elapsed * 3)
        candidate = dict(
            snapshot,
            refresh_sec=refresh_sec,
            cycle_ms=round(elapsed * 1000, 1),
        )
        with self.state_lock:
            candidate_stamp = candidate.get("updated_at")
            current_stamp = self.snapshot.get("updated_at")
            if not (
                isinstance(candidate_stamp, (int, float))
                and isinstance(current_stamp, (int, float))
                and candidate_stamp < current_stamp
            ):
                self.snapshot = candidate
        return refresh_sec

    async def refresh(self):
        while not self.stopping.is_set():
            started = time.monotonic()
            refresh_sec = DEFAULT_REFRESH_SEC
            try:
                snapshot = await asyncio.to_thread(self.collect_snapshot)
                self.publish_snapshot(snapshot, started=started)
                with self.request_lock:
                    keys = [
                        key
                        for key, stamp in self.requested.items()
                        if time.monotonic() - stamp < 30
                    ]
                    self.requested = {key: self.requested[key] for key in keys}
                keys = [key for key in keys if self.detail_needs_refresh(key)]
                for key in keys:
                    await asyncio.to_thread(self.collect_detail, key)
                with self.request_lock:
                    active_keys = {
                        key
                        for key, stamp in self.requested.items()
                        if time.monotonic() - stamp < 30
                    }
                    self.requested = {
                        key: self.requested[key] for key in active_keys
                    }
                with self.scan_lock, self.state_lock:
                    self.details = {
                        key: value
                        for key, value in self.details.items()
                        if key in active_keys
                    }
                    self.monitor.cache = {
                        key: value
                        for key, value in self.monitor.cache.items()
                        if key in active_keys
                    }
                    self.monitor.terminal_details = {
                        key: value
                        for key, value in self.monitor.terminal_details.items()
                        if key in active_keys
                    }
            except (OSError, ValueError, TypeError, KeyError, psutil.Error) as exc:
                with self.state_lock:
                    self.snapshot = dict(self.snapshot, error=type(exc).__name__)
            with self.state_lock:
                current = self.snapshot
            refresh_sec = self.publish_snapshot(current, started=started)
            # Single producer, no overlapping scans; slow disks automatically reduce polling.
            await asyncio.sleep(refresh_sec)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Do not log capability URLs.

    def reply(
        self,
        status,
        body,
        content_type="application/json",
        *,
        disposition=None,
        content_security_policy=None,
    ):
        if isinstance(body, bytes):
            data = body
        elif isinstance(body, str):
            data = body.encode()
        else:
            data = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        suffix = "; charset=utf-8" if content_type.startswith(('text/', 'application/json')) else ''
        self.send_header("Content-Type", content_type + suffix)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        if content_security_policy:
            self.send_header('Content-Security-Policy', content_security_policy)
        if disposition:
            self.send_header('Content-Disposition', disposition)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def route(self):
        prefix = "/" + self.server.token + "/"
        path = urlsplit(self.path).path
        return path[len(prefix) :] if path.startswith(prefix) else None

    def fresh(self):
        values = parse_qs(urlsplit(self.path).query).get("fresh", ())
        return any(value == "1" for value in values)

    def do_GET(self):
        route = self.route()
        if route == "":
            self.reply(
                200,
                PAGE,
                "text/html",
                content_security_policy=(
                    "default-src 'none'; style-src 'unsafe-inline'; "
                    "script-src 'unsafe-inline'; img-src data:; "
                    "connect-src 'self'; frame-src 'self'"
                ),
            )
        elif route == "api":
            if self.fresh():
                started = time.monotonic()
                try:
                    snapshot = self.server.collect_snapshot(discover=True)
                    self.server.publish_snapshot(snapshot, started=started)
                except (OSError, ValueError, TypeError, KeyError, psutil.Error) as exc:
                    with self.server.state_lock:
                        self.server.snapshot = dict(
                            self.server.snapshot,
                            error=type(exc).__name__,
                        )
            with self.server.state_lock:
                snapshot = self.server.snapshot
            self.reply(200, snapshot)
        elif route and (match := re.fullmatch(r'task/(\d+)(?:/(report(?:\.(?:md|html|pdf))?))?', route)):
            key, report_route = match.groups()
            with self.server.state_lock:
                known = any(
                    task["key"] == key for task in self.server.snapshot["tasks"]
                )
            if not known:
                self.reply(404, {"error": "Unknown task"})
                return
            if report_route:
                report = self.server.monitor.report(key)
                if report_route == 'report':
                    self.reply(200, report)
                    return
                name = report_route
                if not report.get('available') or not report.get('formats', {}).get(name):
                    self.reply(404, {'error': 'Report format unavailable'})
                    return
                root = self.server.monitor.task_root(key)
                if root is None:
                    self.reply(404, {'error': 'Task workspace unavailable'})
                    return
                content_types = {
                    'report.md': 'text/markdown',
                    'report.html': 'text/html',
                    'report.pdf': 'application/pdf',
                }
                try:
                    body = (root / name).read_bytes()
                except OSError:
                    self.reply(404, {'error': 'Report format unavailable'})
                    return
                self.reply(
                    200,
                    body,
                    content_types[name],
                    disposition=(f'attachment; filename="{name}"' if name != 'report.html' else None),
                    content_security_policy=(
                        "default-src 'none'; style-src 'unsafe-inline'; img-src data:"
                        if name == 'report.html'
                        else None
                    ),
                )
                return
            with self.server.request_lock:
                self.server.requested[key] = time.monotonic()
            with self.server.state_lock:
                missing = key not in self.server.details
            if missing or (self.fresh() and self.server.detail_needs_refresh(key)):
                self.server.collect_detail(key)
            with self.server.state_lock:
                detail = self.server.details.get(key, {"loading": True})
            self.reply(200, detail)
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        route = self.route()
        if route == 'stop':
            self.reply(200, {"stopping": True})
            self.server.stopping.set()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        match = re.fullmatch(r'task/(\d+)/report', route or '')
        if not match:
            self.reply(404, {"error": "Not found"})
            return
        key = match[1]
        with self.server.state_lock:
            known = any(task['key'] == key for task in self.server.snapshot['tasks'])
        if not known:
            self.reply(404, {'error': 'Unknown task'})
            return
        try:
            with self.server.report_lock:
                report = self.server.monitor.generate_report(key)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.reply(409, {'error': str(exc)})
            return
        self.reply(200, report)


def main(workspace):
    target = directory(workspace)
    config = json.loads((target / "launch.json").read_text())
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        server = MonitorServer(
            (config["host"], config["port"]), workspace, config["token"]
        )
    except OSError as exc:
        import errno

        if exc.errno != errno.EADDRINUSE or not config["port"]:
            raise
        server = MonitorServer((config["host"], 0), workspace, config["token"])
    record = {
        "pid": os.getpid(),
        "created": psutil.Process().create_time(),
        "host": config["host"],
        "port": server.server_port,
        "token": config["token"],
    }
    temp = target / "service.tmp"
    temp.touch(mode=0o600, exist_ok=True)
    temp.write_text(json.dumps(record))
    temp.replace(target / "service.json")
    threading.Thread(target=lambda: asyncio.run(server.refresh()), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.stopping.set()
        server.server_close()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
