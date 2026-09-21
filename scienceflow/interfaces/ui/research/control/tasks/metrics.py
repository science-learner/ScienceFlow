"""Read-only resource facts from executed events, never inferred from proposals."""

import hashlib
import json
from collections import Counter, deque
from datetime import datetime


class ResourceFacts:
    def __init__(self):
        self.seen = set()
        self.counts = Counter()
        self.leases = {}
        self.intervals = []
        self.killed = set()
        self.recent = deque(maxlen=3)
        self._recent = []
        self.observed = False
        self.incomplete = False

    def observe(self, event, worker):
        name = event.get('event', '')
        if name not in {'resource_gpu_lease_acquired', 'resource_gpu_lease_released',
                        'resource_guard_action', 'estra_stage_switched', 'resource_job_started',
                        'resource_job_finished', 'resource_gpu_queue_wait_started'}:
            return
        payload = event.get('payload') or {}
        identity = hashlib.sha256((worker + json.dumps(event, sort_keys=True)).encode()).digest()
        if identity in self.seen:
            return
        self.seen.add(identity)
        fields = {'resource_job_started': 'Started', 'resource_job_finished': 'Finished',
                  'resource_gpu_queue_wait_started': 'Waits', 'resource_gpu_lease_acquired': 'Acquired',
                  'resource_gpu_lease_released': 'Released'}
        executed = payload.get('executed') is True or payload.get('execution_status') == 'executed'
        action = str(payload.get('execution_outcome') or payload.get('action') or '').lower()
        if name in fields:
            self.counts[fields[name]] += 1
        elif name == 'resource_guard_action' and executed and action not in {'', 'no_action', 'none', 'noop'}:
            self.counts['Interventions'] += 1
            if action == 'kill' and payload.get('job_id'):
                self.killed.add((worker, str(payload['job_id'])))
        stamp = event.get('timestamp_utc')
        try:
            now = datetime.fromisoformat(str(stamp).replace('Z', '+00:00')).timestamp()
        except (ValueError, TypeError, OverflowError):
            self.incomplete = True
            return
        key = (worker, str(payload.get('job_id') or ''))
        label = ''
        if name == 'resource_gpu_lease_acquired':
            ids = payload.get('assigned_physical_gpus') or payload.get('gpu_ids')
            if not key[1] or not isinstance(ids, list) or not ids:
                self.incomplete = True
                return
            self.observed = True
            if key not in self.leases:
                self.leases[key] = (now, tuple(str(gpu) for gpu in ids))
            label = 'GPU acquired ' + ','.join(map(str, ids))
        elif name == 'resource_gpu_lease_released':
            self.observed = True
            lease = self.leases.pop(key, None)
            if lease is None or now < lease[0]:
                self.incomplete = True
            else:
                self.intervals.append((lease[0], now, lease[1]))
            label = 'GPU released'
        elif name == 'resource_guard_action':
            executed = payload.get('executed') is True or payload.get('execution_status') == 'executed'
            action = str(payload.get('execution_outcome') or payload.get('action') or '').lower()
            if not executed or action in {'', 'no_action', 'none', 'noop'} or not key[1]:
                return
            label = 'EEC ' + action + ' · ' + str(payload.get('reason') or 'reason unavailable')[:90]
        elif name == 'estra_stage_switched':
            label = 'ESTRA switched stage'
        elif name in fields:
            label = 'EEC ' + fields[name].lower()
        if label:
            self._recent.append((now, f'{str(stamp)[11:19]} {worker} · {label}'))
            self._recent.sort(key=lambda item: item[0])
            self._recent = self._recent[-3:]
            self.recent = deque((text for _, text in self._recent), maxlen=3)

    @property
    def event_total(self):
        return sum(self.counts.values())

    def summary(self):
        return (f'EEC · Events {self.event_total} · ' + ' · '.join(
            f'{name} {self.counts[name]}' for name in ('Started', 'Finished', 'Waits', 'Acquired', 'Released', 'Interventions'))
            + f' · Kill {len(self.killed)}')

    def gpu_summary(self, now, *, alive, cpu_only=False):
        if cpu_only:
            return '0 · 0.0m'
        if not self.observed or self.incomplete or (not alive and self.leases):
            return '—'
        intervals = list(self.intervals) + [(start, now, ids) for start, ids in self.leases.values()]
        # Shared leases on the same physical GPU count wall time once.
        by_gpu = {}
        for start, end, ids in intervals:
            for gpu in ids:
                by_gpu.setdefault(gpu, []).append((start, max(start, end)))
        seconds = 0.0
        for spans in by_gpu.values():
            left, right = sorted(spans)[0]
            for start, end in sorted(spans)[1:]:
                if start > right:
                    seconds += right - left
                    left, right = start, end
                else:
                    right = max(right, end)
            seconds += right - left
        current = len({gpu for _, ids in self.leases.values() for gpu in ids}) if alive else 0
        return f'{current} · {seconds / 60:.1f}m'
