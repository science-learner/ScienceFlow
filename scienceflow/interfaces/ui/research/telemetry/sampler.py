"""One process/device scan per interval, reused by all worker projections."""

import csv
import io
import subprocess
import threading
import time
from pathlib import Path

import psutil


class UtilizationSampler:
    def __init__(self):
        self.lock = threading.Lock()
        self.at = None
        self.host_at = None
        self.gpu_at = None
        self.processes = {}
        self.previous = {}
        self.gpus = {}
        self.gpu_apps = None
        self.host_cpu = None
        self.host_times = None
        self.worker_cache = {}

    @staticmethod
    def _query(fields, kind="gpu"):
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))

    def refresh(self, *, include_processes=True):
        with self.lock:
            now = time.monotonic()
            if include_processes and (self.at is None or now - self.at >= 5):
                interval = now - self.at if self.at is not None else None
                current = {}
                previous = {}
                for proc in psutil.process_iter(["pid", "ppid", "cwd", "cpu_times"]):
                    try:
                        info = proc.info
                        times = info["cpu_times"]
                        if times is None:
                            continue
                        identity = info["pid"]
                        total = times.user + times.system
                        old = self.previous.get(identity)
                        percent = (
                            (total - old) * 100 / interval
                            if old is not None and interval and total >= old
                            else None
                        )
                        previous[identity] = total
                        current[identity] = dict(
                            info,
                            process=proc,
                            percent=percent,
                        )
                    except (psutil.Error, OSError, TypeError):
                        continue
                self.previous, self.processes, self.at = previous, current, now
                self.worker_cache = {}
            if self.host_at is None or now - self.host_at >= 5:
                times = psutil.cpu_times()
                total = (
                    sum(times)
                    - getattr(times, "guest", 0)
                    - getattr(times, "guest_nice", 0)
                )
                idle = times.idle + getattr(times, "iowait", 0)
                if self.host_times is not None and total > self.host_times[0]:
                    self.host_cpu = max(
                        0,
                        min(
                            100,
                            100
                            * (
                                1
                                - (idle - self.host_times[1])
                                / (total - self.host_times[0])
                            ),
                        ),
                    )
                self.host_times = (total, idle)
                self.host_at = now
            if self.gpu_at is None or now - self.gpu_at >= 10:
                self.gpu_at = now
                try:
                    devices = self._query(
                        "index,uuid,utilization.gpu,memory.used,memory.total"
                    )
                    self.gpus = {
                        index: {
                            "uuid": uuid,
                            "util": self._number(util),
                            "used": self._number(used),
                            "total": self._number(total),
                        }
                        for index, uuid, util, used, total in devices
                    }
                except (OSError, ValueError, subprocess.SubprocessError):
                    self.gpus = {}
                try:
                    self.gpu_apps = [
                        (int(pid), uuid, self._number(memory))
                        for pid, uuid, memory in self._query(
                            "pid,gpu_uuid,used_gpu_memory", "compute-apps"
                        )
                    ]
                except (OSError, ValueError, subprocess.SubprocessError):
                    self.gpu_apps = None

    @staticmethod
    def _number(value):
        try:
            number = float(value)
            return number if 0 <= number < float("inf") else None
        except (ValueError, TypeError):
            return None

    def worker(self, root, *, assigned=(), active=True):
        if not active:
            return {"cpu": None, "cores": None, "gpus": [], "active": False}
        self.refresh()
        key = (str(root), tuple(sorted(map(str, assigned))))
        with self.lock:
            if key not in self.worker_cache:
                self.worker_cache[key] = self._worker(root, assigned=assigned)
            return self.worker_cache[key]

    def _worker(self, root, *, assigned):
        root = Path(root).resolve()
        owned = set()
        for pid, info in self.processes.items():
            cwd = info.get("cwd")
            if cwd and Path(cwd).is_relative_to(root):
                owned.add(pid)
        # Include descendants that changed directory; do not count shared coordinator threads.
        children = {}
        for pid, info in self.processes.items():
            children.setdefault(info["ppid"], []).append(pid)
        queue = list(owned)
        while queue:
            for pid in children.get(queue.pop(), ()):
                if pid not in owned:
                    owned.add(pid)
                    queue.append(pid)
        affinities = set()
        for pid in owned:
            try:
                process = self.processes[pid].get("process") or psutil.Process(pid)
                affinities.update(process.cpu_affinity())
            except (psutil.Error, OSError, AttributeError):
                pass
        values = [self.processes[pid]["percent"] for pid in owned]
        cpu = (
            min(100, sum(values) / len(affinities))
            if values and all(v is not None for v in values) and affinities
            else None
        )
        ids = {str(value) for value in assigned}
        apps = self.gpu_apps or []
        ids.update(
            index
            for index, gpu in self.gpus.items()
            if any(pid in owned and uuid == gpu["uuid"] for pid, uuid, _ in apps)
        )
        devices = []
        for index in sorted(
            ids,
            key=lambda value: (
                not value.isdigit(),
                int(value) if value.isdigit() else value,
            ),
        ):
            gpu = self.gpus.get(index, {})
            shared = (
                any(
                    pid not in owned and uuid == gpu.get("uuid")
                    for pid, uuid, _ in apps
                )
                if self.gpu_apps is not None
                else None
            )
            devices.append(dict(gpu, id=index, shared=shared))
        return {
            "cpu": cpu,
            "cores": len(affinities) or None,
            "gpus": devices,
            "active": True,
        }


SAMPLER = UtilizationSampler()
