"""Project independent run logs into compact task rows and optional details."""

import json
import math
import time
from dataclasses import fields
from pathlib import Path

from inquirycraft.tui.formatting import duration
from rich.text import Text

from scienceflow.research.onboarding import LongResearchDraft

from ...long_research_progress import LongResearchProgress

_TERMINAL_STATES = frozenset({'completed', 'failed', 'stopped', 'interrupted'})
_TASK_NAME_WIDTH = 0  # The renderer derives K from the remaining row width.
_BEST_WIDTH = 9
_ESTRA_WIDTH = 4
_EEC_WIDTH = 4


def friendly_status(status: str, *, has_progress: bool = False) -> str:
    """Use an action-oriented label in the TUI while keeping raw state intact."""
    if status == 'failed':
        return 'partial' if has_progress else 'needs review'
    if status == 'interrupted':
        return 'paused'
    return status


def _fit_cell(value: object, width: int) -> str:
    """Fit a task field by terminal cells so CJK names do not shift columns."""
    text = Text(str(value).replace('\n', ' '))
    text.truncate(width, overflow='ellipsis')
    return text.plain + ' ' * max(0, width - text.cell_len)


def _cost_cell(amount: float, priced: int) -> str:
    """Eight terminal cells, keeping the estimate marker and explicit units."""
    if not priced or not math.isfinite(amount) or amount < 0:
        return '—'.rjust(8)
    for scale, unit in ((1, ''), (1e3, 'K'), (1e6, 'M'), (1e9, 'B'), (1e12, 'T')):
        value = amount / scale
        decimals = 1 if unit and round(value, 2) >= 100 else 2
        if round(value, decimals) >= 1000:
            continue
        return f'~${value:.{decimals}f}{unit}'.rjust(8)
    return f'~${amount:.0e}'.rjust(8)


def _storage_cell(size_bytes: int | None, *, exact: bool = True) -> str:
    """Compact storage value; row layout pads before the whole metric block."""
    if size_bytes is None or size_bytes < 0:
        return '—'
    mib = size_bytes / (1024 ** 2)
    marker = '' if exact else '~'
    if mib < 1000:
        rendered = f'{marker}{mib:.1f}M' if mib < 10 else f'{marker}{mib:.0f}M'
    else:
        gib = mib / 1024
        rendered = f'{marker}{gib:.1f}G' if gib < 100 else f'{marker}{gib:.0f}G'
    return rendered


def _storage_sample(entry, row) -> tuple[int | None, bool]:
    """Read the LNR-owned projection; the UI never traverses task directories."""
    candidates = [
        *((row or {}).get('task_roots') or ()),
        (row or {}).get('control_workspace'),
        entry.get('workspace'),
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value) / 'task_logs' / 'storage_latest.json'
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            size = int(payload.get('total_bytes'))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if size >= 0:
            return size, payload.get('exact') is True
    return None, False


def _time_left_text(elapsed: float, budget: float, *, running: bool) -> str:
    """Return a compact live budget label without implying negative time."""
    if not running:
        return ''
    remaining = budget - elapsed
    return f'{duration(remaining, relative=True)} left' if remaining > 0 else 'finalizing'


class TaskProjection:
    def __init__(self):
        self.monitors = {}
        self.terminal_items = {}

    @staticmethod
    def _terminal_signature(entry, row, *, historical):
        status = str(row.get('status') or '').lower()
        if row.get('alive') or status not in _TERMINAL_STATES:
            return None
        return (
            row.get('run_id'),
            row.get('attempt', 0),
            status,
            row.get('finished_at'),
            tuple(row.get('task_roots') or ()),
            entry.get('name'),
            entry.get('status'),
            historical,
        )

    def item(self, entry, row, *, historical=False):
        number = str(entry['number'])
        canonical_name = entry['name'].replace('\n', ' ')
        name = canonical_name + (' · H' if historical else '')
        full_name = canonical_name + (' · history' if historical else '')
        if row is None:
            display = friendly_status(entry['status'])
            cost = _cost_cell(0, 0)
            storage_bytes, storage_exact = _storage_sample(entry, row)
            storage = _storage_cell(storage_bytes, exact=storage_exact)
            narrow_suffix = (f"B {_fit_cell('—', 7)} Es {_fit_cell('—', 3)} "
                             f"Ec {_fit_cell('—', 3)} D {storage} {cost}")
            compact_suffix = (f"Best {_fit_cell('—', 7)} ESTRA {_fit_cell('—', 3)} "
                              f"EEC {_fit_cell('—', 3)} D {storage} {cost}")
            return dict(key=number, text=f"{number:>3} {_fit_cell(name, 18)} {_fit_cell(display, 16)}", detail='',
                        compact=(f"{number:>3} {_fit_cell(name, 10)} Best {_fit_cell('—', _BEST_WIDTH)} "
                                 f"ESTRA {_fit_cell('—', 7)} "
                                 f"EEC {_fit_cell('—', 5)} Disk {storage} Cost {cost}"),
                        row_lead=f'{number} ', row_name=name,
                        row_full_name=full_name,
                        row_name_max_width=_TASK_NAME_WIDTH,
                        row_suffix=(f"{_fit_cell(display, 16)} Best {_fit_cell('—', _BEST_WIDTH)} "
                                    f"ESTRA {_fit_cell('—', 7)} · EEC {_fit_cell('—', 7)} · "
                                    f"Disk {storage} · Cost {cost}"),
                        row_compact_suffix=compact_suffix,
                        row_narrow_suffix=narrow_suffix,
                        state=entry['status'], display_state=display)
        item_key = (row['run_id'], historical)
        terminal_signature = self._terminal_signature(
            entry,
            row,
            historical=historical,
        )
        cached = self.terminal_items.get(item_key)
        if terminal_signature is not None and cached and cached[0] == terminal_signature:
            return cached[1]
        if terminal_signature is None:
            self.terminal_items.pop(item_key, None)
        raw = row.get('draft') or {}
        cache_key = (row['run_id'], row.get('attempt', 0))
        progress = self.monitors.get(cache_key)
        if progress is None:
            allowed = {field.name for field in fields(LongResearchDraft)}
            progress = LongResearchProgress(LongResearchDraft(**{k: v for k, v in raw.items() if k in allowed}))
            if row.get('task_roots'):
                progress.root = Path(row['task_roots'][0])
            self.monitors[cache_key] = progress
        now = time.time()
        active = row['alive'] and row['status'] not in {'completed', 'failed', 'stopped', 'interrupted'}
        finish = now if active else row.get('finished_at')
        elapsed = max(0, (finish or row.get('started_at', now)) - row.get('started_at', now))
        progress.budget = max(1, row.get('display_budget_sec') or raw.get('wall_clock_sec') or 1)
        progress.started = time.monotonic() - elapsed
        progress.active = active
        progress.final_elapsed = None if progress.active else elapsed
        progress.final_state = row['status']
        progress.snapshot()
        display_state = friendly_status(row['status'], has_progress=progress.best_text not in {'—', '-', ''})
        fraction = min(1, elapsed / progress.budget)
        bar = ''.join('■' if i < int(fraction * 10) else '□' for i in range(10))
        counts = progress.events.counts
        facts = progress.events.resources
        gpu = facts.gpu_summary(finish or row.get('started_at', now), alive=active, cpu_only=raw.get('gpu_list') in {'cpu', 'none', '-1'})
        state = 'running' if active and row['status'] != 'stopping' else row['status']
        timer = display_state
        time_left = _time_left_text(elapsed, progress.budget, running=state == 'running')
        eec = str(facts.event_total) if progress.events.offsets else '—'
        estra = str(counts['estra_decision']) if progress.events.offsets else '—'
        amount, priced, _ = progress.usage.cost_totals()
        cost = _cost_cell(amount, priced)
        storage_bytes, storage_exact = _storage_sample(entry, row)
        storage = _storage_cell(storage_bytes, exact=storage_exact)
        suffix = (f"Best {_fit_cell(progress.best_text, _BEST_WIDTH)} "
                  f"ESTRA {_fit_cell(estra, _ESTRA_WIDTH)} "
                  f"EEC {_fit_cell(eec, _EEC_WIDTH)} "
                  f"Disk {storage}")
        narrow_suffix = (f"B {_fit_cell(progress.best_text, 7)} "
                         f"Es {_fit_cell(estra, 3)} Ec {_fit_cell(eec, 3)} "
                         f"D {storage}")
        compact_suffix = (f"Best {_fit_cell(progress.best_text, 7)} "
                          f"ESTRA {_fit_cell(estra, 3)} EEC {_fit_cell(eec, 3)} "
                          f"D {storage}")
        text = (f"{number:>3} {_fit_cell(name, 18)} {_fit_cell(timer, 16)} "
                + suffix)
        progress.summary = progress.summary.replace("EEC · ", f"EEC · GPU {gpu.replace(' · ', '/')} · ", 1)
        compact = (f"{number:>3} {_fit_cell(name, 10)} Best {_fit_cell(progress.best_text, _BEST_WIDTH)} "
                   f"ESTRA {_fit_cell(estra, _ESTRA_WIDTH)} "
                   f"EEC {_fit_cell(eec, _EEC_WIDTH)} Disk {storage} Cost {cost}")
        detail = (progress.summary + ('\n\n' + '\n'.join('- ' + line for line in progress.rows[:3]) if progress.rows else '')
                  + (f'\n+{len(progress.rows) - 3} workers · /status {number}' if len(progress.rows) > 3 else '')
                  + ('\n' + '\n'.join(facts.recent) if facts.recent else ''))
        wide_tail = f'[{bar}] {fraction:4.0%}'
        compact_bar = ''.join(
            '■' if i < int(fraction * 6) else '□'
            for i in range(6)
        )
        compact_tail = f'[{compact_bar}] {fraction:4.0%}'
        if time_left:
            wide_tail += f' · {time_left}'
            compact_tail += f' · {time_left}'
            narrow_tail = f'{fraction:4.0%} · {time_left}'
        else:
            narrow_tail = f'{fraction:4.0%}'
        result = dict(key=number, text=text, compact=compact, detail=detail, state=state,
                      row_lead=f'{number} ', row_name=name,
                      row_full_name=full_name,
                      row_name_max_width=_TASK_NAME_WIDTH,
                      row_suffix=suffix,
                      row_compact_suffix=compact_suffix,
                      row_narrow_suffix=narrow_suffix,
                      progress_tail=f'{cost} {wide_tail}',
                      progress_tail_compact=f'{cost} {compact_tail}',
                      progress_tail_narrow=narrow_tail,
                      display_state=display_state, cost=progress.usage.cost_totals(),
                      storage_bytes=storage_bytes,
                      storage_exact=storage_exact,
                      worker_details=dict(progress.worker_details),
                      monitor=dict(name=canonical_name, status=display_state,
                                   elapsed_sec=elapsed, budget_sec=progress.budget,
                                   fraction=fraction, best=progress.best_text,
                                   metric=raw.get('metric_name', 'metric'),
                                   lower_is_better=getattr(progress, 'lower_is_better', None),
                                   evaluated=len(progress.events.candidates),
                                   valid=sum(value[0] for value in progress.events.candidates.values()),
                                   worker_counts=dict(progress.worker_counts), workers=list(progress.rows),
                                   worker_resources=dict(progress.resource_samples),
                                   estra={key: counts[key] for key in (
                                       'estra_decision', 'estra_stage_switched',
                                       'estra_keep_current_compacted', 'estra_invalid')},
                                   eec=dict(facts.counts), eec_total=facts.event_total,
                                   kills=len(facts.killed), gpu=gpu,
                                   recent=list(facts.recent), usage=progress.usage.summary(),
                                   cost=progress.usage.cost_text(), storage_bytes=storage_bytes,
                                   storage_exact=storage_exact))
        if terminal_signature is not None:
            self.terminal_items[item_key] = (terminal_signature, result)
        return result
