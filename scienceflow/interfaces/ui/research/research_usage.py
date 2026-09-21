"""Incremental, replay-safe research accounting; no calls to billing endpoints."""

import csv
import hashlib
import json
import math
from datetime import datetime

from scienceflow.runtime.observability.telemetry.agent.llm_cost import (
    benchmark_cost,
    extract_model_from_trace_detail,
    format_usd,
    load_model_config_prices,
)


def _tokens(value: int) -> str:
    if value >= 1024 ** 2:
        return f'{value / 1024 ** 2:.1f}M'
    return f'{value / 1024:.1f}K'


def _timestamp(value):
    try:
        if isinstance(value, (float, int)):
            return float(value)
        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


class ResearchUsage:
    def __init__(self, model_config_path=None):
        self.ledgers = {}
        self.sources = {}
        try:
            self.prices = load_model_config_prices(model_config_path)
        except ValueError:
            self.prices = {}

    def read_task(self, root):
        trace = root / 'task_logs/scienceflow_time_trace.csv'
        if not trace.is_file():
            return False
        self.sources = {'task': trace}
        self._read(trace, trace=True)
        return True

    def read_worker(self, log_dir):
        trace = log_dir / 'scienceflow_time_trace.csv'
        path = trace if trace.is_file() else log_dir / 'agent_runtime_audit/agent_provider_calls.jsonl'
        # When the unified trace appears, replace the fallback, never add both.
        self.sources[str(log_dir)] = path
        self._read(path, trace=path == trace)

    def read(self, path):
        """Legacy provider-audit entry point remains supported."""
        self.sources[str(path)] = path
        self._read(path, trace=False)

    def _read(self, path, *, trace):
        ledger = self.ledgers.setdefault(
            path,
            {
                'offset': 0,
                'seen': set(),
                'totals': {},
                'cost': 0.0,
                'unknown': 0,
                'priced': 0,
                'missing_cache': False,
                'cache_input': 0,
                'cache_hits': 0,
                'header': None,
                'pending': False,
            },
        )
        try:
            with path.open('rb') as stream:
                offset = ledger['offset']
                if path.stat().st_size < offset:
                    offset = 0
                stream.seek(offset)
                for _ in range(250):
                    raw = stream.readline(1_000_000)
                    if not raw or not raw.endswith(b'\n'):
                        break
                    ledger['offset'] = stream.tell()
                    try:
                        if trace:
                            cells = next(csv.reader([raw.decode()]))
                            if {'timestamp', 'category', 'tokens_input'}.issubset(cells):
                                ledger['header'] = cells
                                continue
                            row = dict(zip(ledger['header'] or (), cells))
                            if row.get('category') != 'llm_api':
                                continue
                            identity = hashlib.sha256(raw).hexdigest()
                            model = extract_model_from_trace_detail(row.get('detail')) or '?'
                        else:
                            row = json.loads(raw)
                            if row.get('phase') != 'completed':
                                continue
                            identity = tuple(str(row.get(k, '')) for k in
                                             ('worker_id', 'session_id', 'run_id', 'turn_id', 'call_seq', 'attempt'))
                            model = str(row.get('model') or '?')
                        if identity in ledger['seen']:
                            continue
                        ledger['seen'].add(identity)
                        self._accept(ledger, row, model)
                    except (ValueError, TypeError, KeyError, AttributeError, csv.Error):
                        ledger['unknown'] += 1
                        ledger['missing_cache'] = True
                ledger['pending'] = ledger['offset'] < path.stat().st_size
        except OSError:
            pass

    def _accept(self, ledger, row, model):
        values = [int(row[k]) if row.get(k) not in (None, '') else None
                  for k in ('tokens_input', 'tokens_output', 'tokens_cached')]
        incoming, outgoing, cached = values
        if any(v is not None and v < 0 for v in values) or (incoming is not None and cached is not None and cached > incoming):
            raise ValueError('Invalid provider token counts')
        if incoming is None or cached is None:
            ledger["missing_cache"] = True
        else:
            ledger["cache_input"] += incoming
            ledger["cache_hits"] += cached
        key = (str(row.get('worker_id') or row.get('node_id') or '?'), model)
        total = ledger['totals'].setdefault(key, [0, 0, 0])
        for index, value in enumerate(values):
            total[index] += value or 0
        timestamp = _timestamp(row.get('timestamp', row.get('timestamp_utc')))
        if timestamp is not None and row.get('duration_hrs'):
            timestamp -= max(0, float(row['duration_hrs'])) * 3600
        cost = benchmark_cost(
            incoming, outgoing, cached, timestamp,
            model=model, prices=self.prices)
        saved = row.get('llm_cost_usd')
        if (
            cost is None
            and saved not in (None, '')
            and 'pricing=models.json' in str(row.get('detail') or '')
        ):
            cost = float(saved)
        if cost is None or not math.isfinite(cost) or cost < 0:
            ledger['unknown'] += 1
        else:
            ledger['cost'] += cost
            ledger['priced'] += 1

    def _active(self):
        return [self.ledgers[path] for path in set(self.sources.values()) if path in self.ledgers]

    @property
    def totals(self):
        result = {}
        for ledger in self._active():
            for model, values in ledger['totals'].items():
                total = result.setdefault(model, [0, 0, 0])
                for index, value in enumerate(values):
                    total[index] += value
        return result

    def cost_totals(self):
        ledgers = self._active()
        return (sum(row['cost'] for row in ledgers), sum(row['priced'] for row in ledgers),
                sum(row['unknown'] + int(row['pending']) for row in ledgers))

    def cost_text(self):
        cost, priced, unknown = self.cost_totals()
        if not priced:
            return '— (incomplete)' if unknown else '—'
        return '~' + format_usd(cost)

    def summary(self):
        totals = self.totals
        if not totals:
            return 'Research usage · awaiting provider reports · Cost ' + self.cost_text()
        incoming, outgoing, _cached = (
            sum(t[i] for t in totals.values()) for i in range(3)
        )
        incomplete = any(row['missing_cache'] for row in self._active())
        cache_input = sum(row['cache_input'] for row in self._active())
        cache_hits = sum(row['cache_hits'] for row in self._active())
        cache = (f'{cache_hits/cache_input:.0%}' + (' (partial)' if incomplete else '')) if cache_input else '—'
        return f'Research usage · in {_tokens(incoming)} / out {_tokens(outgoing)} · Cache {cache} · Cost {self.cost_text()}'

    def details(self):
        return self.summary() + '\n' + '\n'.join(
            f'{worker} · {model} · input {_tokens(values[0])} · output {_tokens(values[1])} · cached {_tokens(values[2])}'
            for (worker, model), values in self.totals.items()) + (
                '\nPricing: models.json; USD estimate, not a bill.'
                '\nUnified traces include code and feedback calls; provider-only fallback may omit feedback.'
                '\nMissing usage or request time is marked incomplete; chat is accounted separately.')
