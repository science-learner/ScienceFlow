"""Low-frequency host resource sampling; no recursive workspace scans."""

import asyncio
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass

import psutil

from .telemetry.formatting import percentage
from .telemetry.sampler import SAMPLER

_VISIBLE_TASK_STATES = frozenset({'starting', 'running'})


def _draft_mapping(value) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return {}


def _claims(record: Mapping[str, object]) -> dict[str, list[object]]:
    claims = record.get('resource_claims')
    if isinstance(claims, Mapping):
        return {
            'cpu': list(claims.get('cpu') or ()),
            'gpu': list(claims.get('gpu') or ()),
        }
    from scienceflow.runtime.parallel.control.resource_claims import claims_for

    return claims_for(_draft_mapping(record.get('draft')))


def _compact_ids(values: Sequence[object]) -> str:
    try:
        ids = sorted({int(value) for value in values})
    except (TypeError, ValueError):
        return ','.join(str(value) for value in values) or '—'
    if not ids:
        return '—'
    ranges: list[str] = []
    start = previous = ids[0]
    for current in ids[1:]:
        if current == previous + 1:
            previous = current
            continue
        ranges.append(str(start) if start == previous else f'{start}-{previous}')
        start = previous = current
    ranges.append(str(start) if start == previous else f'{start}-{previous}')
    return ','.join(ranges)


def task_allocation_line(
    label: str,
    name: str,
    record: Mapping[str, object],
) -> str:
    """Render one logical line without changing the task-board layout."""

    draft = _draft_mapping(record.get('draft'))
    claims = _claims(record)
    raw_gpu = str(draft.get('gpu_list') or '').strip().lower()
    if '*' in claims['gpu']:
        gpu = raw_gpu or 'auto'
    elif claims['gpu']:
        gpu = _compact_ids(claims['gpu'])
    elif raw_gpu in {'cpu', 'none', 'off', 'disabled', '-1'}:
        gpu = 'CPU-only'
    else:
        gpu = '—'
    workers = draft.get('workers')
    worker_text = str(workers) if workers is not None else '—'
    compact_name = ' '.join(str(name or 'Research').split())[:48]
    return (
        f'{label} · {compact_name} · CPU {_compact_ids(claims["cpu"])} · '
        f'GPU {gpu} · {worker_text} workers'
    )


def allocation_summary(
    host_summary: str,
    entries: Sequence[Mapping[str, object]],
    rows: Mapping[str, Mapping[str, object]],
    *,
    cpu_ids: Sequence[int] | None = None,
    gpu_ids: Sequence[int] | None = None,
) -> str:
    """Show current tasks one per line and compute availability conservatively."""

    lines = [host_summary]
    for entry in sorted(entries, key=lambda item: int(item.get('number') or 0)):
        row = rows.get(str(entry.get('run_id') or ''))
        if (
            row is None
            or not row.get('alive')
            or row.get('status') not in _VISIBLE_TASK_STATES
        ):
            continue
        lines.append(
            task_allocation_line(
                f'Task {entry.get("number")}',
                str(entry.get('name') or ''),
                row,
            )
        )

    occupied_cpu: set[int] = set()
    occupied_gpu: set[int] = set()
    all_gpus_reserved = False
    for row in rows.values():
        if not row.get('alive'):
            continue
        claims = _claims(row)
        occupied_cpu.update(int(value) for value in claims['cpu'])
        if '*' in claims['gpu']:
            all_gpus_reserved = True
        else:
            occupied_gpu.update(int(value) for value in claims['gpu'])

    if cpu_ids is None:
        cpu_ids = (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, 'sched_getaffinity')
            else list(range(os.cpu_count() or 1))
        )
    if gpu_ids is None:
        gpu_ids = [int(value) for value in SAMPLER.gpus]
    available_cpu = [value for value in cpu_ids if value not in occupied_cpu]
    available_gpu = [] if all_gpus_reserved else [
        value for value in gpu_ids if value not in occupied_gpu
    ]
    lines.append(
        f'Available · CPU {_compact_ids(available_cpu)} · '
        f'GPU {_compact_ids(available_gpu)}'
    )
    return '\n'.join(lines)


def allocation_preview(
    entries: Sequence[Mapping[str, object]],
    rows: Mapping[str, Mapping[str, object]],
    preparing: Mapping[str, object] | None,
    draft,
) -> str:
    """Describe visible running allocations plus the proposed task."""

    lines: list[str] = []
    for entry in sorted(entries, key=lambda item: int(item.get('number') or 0)):
        row = rows.get(str(entry.get('run_id') or ''))
        if (
            row is None
            or not row.get('alive')
            or row.get('status') not in _VISIBLE_TASK_STATES
        ):
            continue
        lines.append(
            task_allocation_line(
                f'Task {entry.get("number")}',
                str(entry.get('name') or ''),
                row,
            )
        )
    if preparing is not None:
        proposed = task_allocation_line(
            f'Proposed Task {preparing.get("number")}',
            str(preparing.get('name') or ''),
            {'draft': draft},
        )
        duration = _draft_mapping(draft).get('wall_clock_sec')
        if duration is not None:
            proposed += f' · {int(duration) / 60:g} min'
        lines.append(proposed)
    return '\n'.join(lines)


class ResourceMonitor:
    def __init__(self, workspace):
        self.workspace = workspace
        self.latest = "Host resources loading…"

    def sample(self):
        # The footer needs aggregate host/device data only. Per-process scans
        # are reserved for active worker cards.
        SAMPLER.refresh(include_processes=False)
        ram = psutil.virtual_memory()
        disk = shutil.disk_usage(self.workspace)
        devices = list(SAMPLER.gpus.values())
        utilization = [gpu["util"] for gpu in devices]
        avg = (
            sum(utilization) / len(devices)
            if devices and all(v is not None for v in utilization)
            else None
        )
        gpu = (
            f"GPU ×{len(devices)} avg {percentage(avg)}"
            if devices
            else "GPU —"
        )
        free = disk.free / 2**30
        storage = f"{free / 1024:.1f}T" if free >= 1024 else f"{free:.0f}G"
        return (
            f"Host · CPU {percentage(SAMPLER.host_cpu)} · RAM {ram.used / 2**30:.0f}/{ram.total / 2**30:.0f}G"
            f" · {gpu} · Disk free {storage}"
        )

    async def watch(self, host):
        while True:
            try:
                text = await asyncio.to_thread(self.sample)
            except (OSError, psutil.Error):
                text = "Host resources unavailable"
            if text != self.latest:
                self.latest = text
                host.set_resource_status(text)
            await asyncio.sleep(5)
