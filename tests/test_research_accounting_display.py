"""Cost evidence, lifecycle rendering, and EEC aggregation regressions."""

import csv
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from scienceflow.interfaces.ui.research.control.tasks.metrics import ResourceFacts
from scienceflow.interfaces.ui.research.control.tasks.projection import (
    TaskProjection,
    _time_left_text,
)
from scienceflow.interfaces.ui.research.long_research_progress import (
    LongResearchProgress,
)
from scienceflow.interfaces.ui.research.research_usage import ResearchUsage
from scienceflow.runtime.observability.telemetry.agent.llm_cost import (
    LLMPrice,
    benchmark_cost,
)


def stamp(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def model_config(path, model='custom'):
    path.write_text(json.dumps({
        'version': 1,
        'models': {
            'priced': {
                'model': model,
                'pricing': {
                    'input_usd_per_1m': .15,
                    'cached_input_usd_per_1m': .003,
                    'output_usd_per_1m': .60,
                },
                'endpoints': [{'url': 'https://example.test/v1', 'key': 'x'}],
            },
        },
        'defaults': {'code_models': ['priced']},
    }))
    return path


@pytest.mark.parametrize('amount,expected', [
    (12.34, '~$12.34'), (999.99, '~$999.99'), (1000, '~$1.00K'),
    (12345, '~$12.35K'), (99999, '~$100.0K'), (999999, '~$1.00M'),
    (1000000, '~$1.00M'), (1e20, '~$1e+20'),
])
def test_task_cost_cell_has_stable_width(amount, expected):
    from scienceflow.interfaces.ui.research.control.tasks.projection import _cost_cell
    value = _cost_cell(amount, 1)
    assert len(value) == 8
    assert value.strip() == expected
    assert _cost_cell(amount, 0) == '       —'


def test_task_time_left_is_compact_and_never_negative():
    assert _time_left_text(360, 1200, running=True) == '14m left'
    assert _time_left_text(1191, 1200, running=True) == '9s left'
    assert _time_left_text(1200, 1200, running=True) == 'finalizing'
    assert _time_left_text(1300, 1200, running=True) == 'finalizing'
    assert _time_left_text(360, 1200, running=False) == ''


def test_running_task_projection_prioritizes_time_left_in_every_width(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        'scienceflow.interfaces.ui.research.control.tasks.projection.time.time',
        lambda: 1000.0,
    )
    row = {
        'run_id': 'run',
        'alive': True,
        'status': 'running',
        'started_at': 640.0,
        'display_budget_sec': 1200,
        'task_roots': [str(tmp_path / 'task')],
        'draft': {
            'workspace_base': str(tmp_path),
            'workers': 0,
            'wall_clock_sec': 1200,
        },
    }

    item = TaskProjection().item(
        {'number': 1, 'name': 'Demo', 'status': 'running'},
        row,
    )

    assert '30% · 14m left' in item['progress_tail']
    assert '30% · 14m left' in item['progress_tail_compact']
    assert item['progress_tail'].index('—') < item['progress_tail'].index('[')
    assert item['progress_tail_compact'].index('—') < item['progress_tail_compact'].index('[')
    assert item['progress_tail_narrow'].strip() == '30% · 14m left'
    assert '$' not in item['progress_tail_narrow']


@pytest.mark.parametrize('size,expected', [
    (0, '0.0M'), (512 * 1024 * 1024, '512M'),
    (1024 * 1024 * 1024, '1.0G'), (int(2.25 * 1024 ** 3), '2.2G'),
])
def test_task_storage_cell_uses_m_and_g_units(size, expected):
    from scienceflow.interfaces.ui.research.control.tasks.projection import _storage_cell

    value = _storage_cell(size)
    assert value.strip() == expected
    assert value == expected
    assert _storage_cell(size, exact=False).strip() == f'~{expected}'
    assert _storage_cell(None) == '—'


def test_task_overview_limits_names_and_keeps_best_in_compact_rows():
    from scienceflow.interfaces.ui.research.control.tasks.projection import _fit_cell

    item = TaskProjection().item(
        {'number': 4, 'name': 'nomad2018-predict-transparent-conductors', 'status': 'preparing'},
        None,
    )

    assert item['row_name_max_width'] == 0
    assert item['row_lead'] == '4 '
    assert item['row_compact_suffix'].startswith('Best ')
    assert 'ESTRA —' in item['row_compact_suffix']
    assert 'D —' in item['row_compact_suffix']
    assert 'Disk —' in item['row_suffix']
    assert _fit_cell('123456789012345', 9) == '12345678…'


def test_task_overview_shows_decisions_only_and_reads_lnr_storage_log(tmp_path):
    root = tmp_path / 'task'
    log = root / 'workers' / 'w00' / 'logs' / 'lhr_events.jsonl'
    log.parent.mkdir(parents=True)
    log.write_text(
        '\n'.join(json.dumps({
            'event': event,
            'timestamp_utc': f'2026-09-12T00:00:0{index}Z',
            'payload': {},
        }) for index, event in enumerate(
            ('estra_decision', 'estra_decision', 'estra_stage_switched')
        )) + '\n'
    )
    storage_log = root / 'task_logs' / 'storage_latest.json'
    storage_log.parent.mkdir(parents=True)
    storage_log.write_text(json.dumps({
        'total_bytes': 2 * 1024 * 1024,
        'exact': False,
    }))
    row = dict(
        run_id='run', alive=True, status='running', started_at=0,
        task_roots=[str(root)], control_workspace=str(root),
        draft=dict(workspace_base=str(tmp_path), workers=1, wall_clock_sec=120),
    )

    item = TaskProjection().item(dict(number=1, name='Demo', status='running'), row)

    assert item['row_lead'] == '1 '
    assert 'ESTRA 2' in item['row_compact_suffix']
    assert 'ESTRA 2/' not in item['row_compact_suffix']
    assert 'D ~2.0M' in item['row_compact_suffix']
    assert 'Cost' not in item['row_compact_suffix']
    assert item['progress_tail'].lstrip().startswith('— [')
    storage_log.write_text(json.dumps({
        'total_bytes': 2 * 1024 ** 3,
        'exact': True,
    }))
    exact = TaskProjection().item(dict(number=1, name='Demo', status='running'), row)
    storage_log.unlink()
    missing = TaskProjection().item(dict(number=1, name='Demo', status='running'), row)
    assert 'D 2.0G' in exact['row_compact_suffix']
    assert 'D —' in missing['row_compact_suffix']
    assert item['storage_bytes'] == 2 * 1024 * 1024
    assert item['storage_exact'] is False


def test_only_explicit_model_prices_are_used():
    assert benchmark_cost(1_000_000, 1_000_000, 500_000, stamp('2026-09-12T02:00:00Z'), prices={}) is None
    assert benchmark_cost(1, 1, None, 0, prices={}) is None
    assert benchmark_cost(1, 1, 0, None, prices={}) is None
    assert benchmark_cost(1_000_000, 0, 0, None, model='custom', prices={'custom': LLMPrice(2, 1, 4)}) == 2


def write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=('timestamp', 'category', 'node_id', 'tokens_input', 'tokens_output', 'tokens_cached', 'llm_cost_usd', 'detail'))
        writer.writeheader()
        writer.writerows(rows)


def test_unified_trace_replaces_audits_includes_feedback_and_survives_replay(tmp_path):
    usage = ResearchUsage(model_config(tmp_path / 'models.json'))
    logs = tmp_path / 'workers/w00/logs'
    audit = logs / 'agent_runtime_audit/agent_provider_calls.jsonl'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps(dict(phase='completed', worker_id='w00', call_seq=1, timestamp=0,
                                    tokens_input=100, tokens_output=10, tokens_cached=0)) + '\n')
    usage.read_worker(logs)
    trace = tmp_path / 'task_logs/scienceflow_time_trace.csv'
    rows = [dict(timestamp='2026-09-12T02:00:00+00:00', category='llm_api', node_id='lnr:w00',
                 tokens_input=1_000_000, tokens_output=100_000, tokens_cached=500_000,
                 llm_cost_usd=cost, detail=f'llm_role={role};model=custom;call_seq={index}')
            for index, (role, cost) in enumerate([('code', ''), ('feedback', '2.0')])]
    write_trace(trace, rows)
    assert usage.read_task(tmp_path)
    assert sum(v[0] for v in usage.totals.values()) == 2_000_000
    assert usage.cost_totals()[0] == pytest.approx(2 * (.075 + .0015 + .06))
    assert 'Cost ~$0.27' in usage.summary()
    usage.read_task(tmp_path)
    trace.write_text('')
    usage.read_task(tmp_path)
    write_trace(trace, rows)
    usage.read_task(tmp_path)
    assert sum(v[0] for v in usage.totals.values()) == 2_000_000
    assert usage.cost_totals()[1:] == (2, 0)


def test_missing_usage_is_not_free_or_a_zero_cache_hit(tmp_path):
    path = tmp_path / 'calls.jsonl'
    path.write_text(json.dumps(dict(phase='completed', tokens_input=1000, tokens_output=20, tokens_cached=None)) + '\n')
    usage = ResearchUsage()
    usage.read(path)
    assert 'in 1.0K' in usage.summary()
    assert 'Cache —' in usage.summary()
    assert 'Cost — (incomplete)' in usage.summary()


def test_eec_counts_executed_interventions_not_proposals():
    facts = ResourceFacts()
    names = ['resource_job_started', 'resource_gpu_queue_wait_started', 'resource_gpu_lease_acquired',
             'resource_gpu_lease_released', 'resource_job_finished']
    events = [dict(event=name, timestamp_utc='2026-09-12T00:00:00Z', payload={'job_id': 'a', 'gpu_ids': [0]}) for name in names]
    events += [dict(event='resource_guard_action', timestamp_utc='2026-09-12T00:00:01Z',
                    payload={'job_id': 'a', 'action': action, 'executed': executed})
               for action, executed in [('KILL', False), ('KILL', True), ('EXTEND', True)]]
    for event in events * 2:
        facts.observe(event, 'w00')
    assert facts.event_total == 7
    assert facts.counts['Interventions'] == 2
    assert len(facts.killed) == 1
    assert 'EEC · Events 7' in facts.summary()


def test_stopped_task_freezes_time_and_hides_worker_age(tmp_path, monkeypatch):
    import scienceflow.interfaces.ui.research.control.tasks.projection as projection

    now = [10000.]
    monkeypatch.setattr(projection.time, 'time', lambda: now[0])
    root = tmp_path / 'task'
    path = root / 'workers/w00/logs/lhr_state.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'running', 'generated_at': 100, 'progress_last': {'phase': 'running'}}))
    row = dict(run_id='run', alive=False, status='stopped', started_at=0, finished_at=7200,
               task_roots=[str(root)], draft=dict(workspace_base=str(tmp_path), workers=1, wall_clock_sec=10800))
    entry = dict(number=1, name='Demo', status='stopped')
    board = TaskProjection()
    first = board.item(entry, row)
    now[0] += 3600
    second = board.item(entry, row)
    assert first is second
    assert first['detail'] == second['detail']
    assert 'ago' not in first['detail']
    assert '2h 00m/3h 00m' in first['detail']
    assert '- w00: stopped' in first['detail']
    assert 'EEC' in first['text'] and 'Kill' not in first['text']
    row['attempt'] = 2
    assert board.item(entry, row) is not first


def test_terminal_worker_hides_age_while_other_worker_remains_active(tmp_path):
    draft = SimpleNamespace(workspace_base=str(tmp_path), run_id='run', exp_id='task', wall_clock_sec=120, workers=2)
    progress = LongResearchProgress(draft)
    for i, status in enumerate(('completed', 'running')):
        path = progress.root / f'workers/w0{i}/logs/lhr_state.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'status': status, 'generated_at': 1, 'progress_last': {'phase': 'running'}}))
    progress.snapshot()
    assert 'ago' not in progress.rows[0]
    assert 'h ago' in progress.rows[1]


def test_task_global_best_is_attributed_to_its_worker_and_live_candidates_can_advance_it(
    tmp_path,
):
    draft = SimpleNamespace(
        workspace_base=str(tmp_path),
        run_id='run',
        exp_id='task',
        metric_name='radius',
        lower_is_better=False,
        wall_clock_sec=120,
        workers=2,
    )
    progress = LongResearchProgress(draft)
    task_logs = progress.root / 'task_logs'
    task_logs.mkdir(parents=True)
    (task_logs / 'lhr_state.json').write_text(json.dumps({
        'global_best': {
            'worker_id': 'W01',
            'metric_name': 'radius',
            'metric_value': 3.0,
            'lower_is_better': False,
            'validation_ok': True,
        },
    }))
    for worker, value in (('w00', 2.0), ('w01', None)):
        logs = progress.root / f'workers/{worker}/logs'
        logs.mkdir(parents=True)
        state = {'progress_last': {'phase': 'running'}}
        if value is not None:
            state['global_best'] = {
                'metric_name': 'radius',
                'metric_value': value,
                'lower_is_better': False,
                'validation_ok': True,
            }
        (logs / 'lhr_state.json').write_text(json.dumps(state))
    event = {
        'event': 'evaluator_metric_event',
        'worker_id': 'w00',
        'payload': {
            'candidate_id': 'new',
            'evaluator_status': 'ok',
            'selection_eligible': True,
            'validation_ok': True,
            'metric_name': 'radius',
            'metric_value': 3.5,
            'lower_is_better': False,
        },
    }
    (progress.root / 'workers/w00/logs/lhr_events.jsonl').write_text(
        json.dumps(event) + '\n'
    )

    progress.snapshot()

    assert progress.best_text == '3.5'
    assert any(row.startswith('w00:') and 'best 3.5' in row for row in progress.rows)
    assert any(row.startswith('w01:') and 'best 3' in row for row in progress.rows)


def test_unknown_call_keeps_known_cache_average_marked_partial(tmp_path):
    path = tmp_path / 'calls.jsonl'
    rows = [dict(phase='completed', call_seq=1, tokens_input=100, tokens_output=10, tokens_cached=50, timestamp=0),
            dict(phase='completed', call_seq=2, tokens_input=100, tokens_output=10, tokens_cached=None, timestamp=0)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    usage = ResearchUsage(model_config(tmp_path / 'models.json', model='?'))
    usage.read(path)
    assert 'Cache 50% (partial)' in usage.summary()
    assert usage.cost_text().startswith('~$')
    assert '(partial)' not in usage.cost_text()
    assert usage.cost_totals()[2] == 1


def test_merged_worker_trace_header_and_replay(tmp_path):
    trace = tmp_path / 'task_logs/scienceflow_time_trace.csv'
    trace.parent.mkdir()
    with trace.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['worker_id', 'worker_index', 'timestamp', 'category', 'tokens_input', 'tokens_output', 'tokens_cached', 'detail'])
        writer.writerow(['w00', '0', '2026-09-12T02:00:00Z', 'llm_api', 1000000, 1000000, 500000, 'model=demo'])
        writer.writerow(['w01', '1', '2026-09-12T02:00:00Z', 'llm_api', 1000000, 1000000, 500000, 'model=demo'])
    usage = ResearchUsage(model_config(tmp_path / 'models.json', model='demo'))
    usage.read_task(tmp_path)
    usage.read_task(tmp_path)
    assert usage.cost_totals() == pytest.approx((1.353, 2, 0))
    assert 'Cost ~$1.35' in usage.summary()
    assert 'Cache 50%' in usage.summary()
