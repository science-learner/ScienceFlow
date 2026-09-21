"""Conservative cross-run resource reservations under the managed registry lock."""

import os

from scienceflow.runtime.core.support.system_resources import (
    parse_cpu_list,
    parse_gpu_list_auto,
    query_gpu_devices,
)


class ResourceBusy(ValueError):
    pass


def claims_for(draft):
    cpu = set(parse_cpu_list(str(draft.get('cpu_list') or '')))
    raw = str(draft.get('gpu_list') or '').lower()
    gpu = set() if raw in {'cpu', 'none', 'off', 'disabled', '-1', ''} else (
        {'*'} if raw.startswith('auto') else {str(i) for i in parse_cpu_list(raw)})
    return {'cpu': sorted(cpu), 'gpu': sorted(gpu)}


def reserve_resources(draft, rows):
    """Choose concrete GPUs for auto requests and reject overlapping pools."""
    claims = claims_for(draft)
    occupied = [(r, r.get('resource_claims') or claims_for(r.get('draft') or {}))
                for r in rows if r['alive']]
    if '*' in claims['gpu']:
        raw = str(draft.get('gpu_list') or 'auto')
        _, count, candidates = parse_gpu_list_auto(raw)
        if raw.strip().lower() == 'auto':
            count = max(count, int(draft.get('workers') or 2))
        if count < 1:
            raise ResourceBusy('GPU count must be positive.')
        busy = {gpu for _, c in occupied for gpu in c['gpu']}
        free = [] if '*' in busy else [str(g.index) for g in query_gpu_devices()
                                      if str(g.index) not in busy and (not candidates or g.index in candidates)]
        if len(free) < count:
            raise ResourceBusy('Not enough unreserved GPUs. Choose another GPU pool or retry after a task finishes.')
        claims['gpu'] = free[:count]
    affinity = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    if affinity is not None and not set(claims['cpu']).issubset(affinity):
        raise ResourceBusy('The requested CPU pool is outside the current process affinity.')
    for row, other in occupied:
        cpu = set(claims['cpu']) & set(other['cpu'])
        gpu = set(claims['gpu']) & set(other['gpu'])
        if claims['gpu'] and '*' in other['gpu']:
            gpu = set(claims['gpu'])
        if cpu or gpu:
            path = row.get('control_workspace') or row.get('cwd', '')
            raise ResourceBusy(f'Resources reserved by {path}: CPU {sorted(cpu)}; GPU {sorted(gpu)}. '
                               'Change this task’s pool or retry after the other task stops.')
    return claims


def apply_reservation(record, rows):
    claims = reserve_resources(record.get('draft') or {}, rows)
    record['resource_claims'] = claims
    draft = record.get('draft') or {}
    if str(draft.get('gpu_list', '')).startswith('auto'):
        draft['gpu_list'] = ','.join(claims['gpu'])
        for task in record['manifest_payload'].get('tasks', []):
            task['gpu_list'] = draft['gpu_list']
