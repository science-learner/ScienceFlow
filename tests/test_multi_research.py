"""Multi-task isolation, stable addressing and evidence-based resource summaries."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scienceflow.interfaces.ui.long_research import LongResearchInteraction
from scienceflow.interfaces.ui.research.control.tasks import manager
from scienceflow.interfaces.ui.research.control.tasks.metrics import ResourceFacts
from scienceflow.interfaces.ui.research.control.tasks.records import TaskIndex
from scienceflow.interfaces.ui.research.research_events import ResearchEvents
from scienceflow.interfaces.ui.research.resources import (
    allocation_preview,
    allocation_summary,
)
from scienceflow.research.onboarding import (
    LongResearchDraft,
    LongResearchSession,
    OnboardingState,
)
from scienceflow.runtime.parallel.control.resource_claims import (
    ResourceBusy,
    reserve_resources,
)


class Host:
    busy = False

    def __init__(self):
        self.notices = []
        self.items = ()

    async def notice(self, text):
        self.notices.append(text)

    async def echo_user(self, text):
        pass

    def set_host_status(self, text):
        pass

    def conversation_snapshot(self):
        return ()

    def set_host_tasks(self, title, items):
        self.title = title
        self.items = items


@pytest.mark.asyncio
async def test_header_counts_terminal_tasks_instead_of_elapsed_time(tmp_path, monkeypatch):
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    items = tuple(dict(state=state, display_state=display) for state, display in (
        ('completed', 'completed'), ('failed', 'partial'), ('stopped', 'stopped'),
        ('running', 'running'), ('interrupted', 'paused'), ('preparing', 'preparing'),
    ))
    monkeypatch.setattr(tasks, 'collect', lambda: ([], {}, items))
    host = Host()
    await tasks.refresh(host)
    assert '4/6 done' in host.title
    assert '1 running' in host.title
    assert ' stopped' not in host.title and ' paused' not in host.title
    assert ' pending' not in host.title
    assert '[' not in host.title
    assert 'Time' not in host.title
    items = (dict(state='running'), dict(state='stopping'))
    await tasks.refresh(host)
    assert '0/2 done · 1 running' in host.title
    items = ()
    await tasks.refresh(host)
    assert '0/0 done' in host.title


@pytest.mark.asyncio
async def test_refresh_summarizes_task_library_registrations(tmp_path, monkeypatch):
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    package = tmp_path / 'task_library' / 'circle-packing-2619707c5880'
    entries = [
        {'number': 7, 'run_id': 'run-7'},
        {'number': 8, 'run_id': 'run-8'},
    ]
    rows = {
        entry['run_id']: {
            'run_id': entry['run_id'],
            'task_registration': {'status': 'registered', 'path': str(package)},
        }
        for entry in entries
    }
    items = ({'state': 'completed'}, {'state': 'completed'})
    monkeypatch.setattr(tasks, 'collect', lambda: (entries, rows, items))
    host = Host()

    await tasks.refresh(host)
    await tasks.refresh(host)

    assert host.notices == [
        'Task library · Tasks 7, 8 registered · circle-packing-2619707c5880'
    ]
    assert str(tmp_path) not in host.notices[0]


@pytest.mark.asyncio
async def test_refresh_summarizes_deferred_task_registrations(tmp_path, monkeypatch):
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    entries = [
        {'number': 2, 'run_id': 'run-2'},
        {'number': 3, 'run_id': 'run-3'},
    ]
    rows = {
        entry['run_id']: {
            'run_id': entry['run_id'],
            'task_registration': {
                'status': 'deferred',
                'reason': 'final artifact is not ready',
            },
        }
        for entry in entries
    }
    items = ({'state': 'completed'}, {'state': 'completed'})
    monkeypatch.setattr(tasks, 'collect', lambda: (entries, rows, items))
    host = Host()

    await tasks.refresh(host)

    assert host.notices == [
        (
            'Task library · Tasks 2, 3 saved · registration deferred · '
            'final artifact is not ready'
        )
    ]


def test_persistent_numbers_and_concurrent_isolated_directories(tmp_path):
    index = TaskIndex(tmp_path)
    with ThreadPoolExecutor(4) as pool:
        entries = list(pool.map(index.reserve, ['circle'] * 8))
    assert sorted(e['number'] for e in entries) == list(range(1, 9))
    assert len({e['workspace'] for e in entries}) == 8
    row = dict(run_id='run-a', control_workspace=entries[0]['workspace'], status='running')
    reloaded = TaskIndex(tmp_path).sync([row])
    synced = next(e for e in reloaded if e['run_id'])
    assert synced['number'] == entries[0]['number']
    row['status'] = 'failed'
    resynced = TaskIndex(tmp_path).sync([row])
    assert next(e for e in resynced if e['number'] == synced['number'])['status'] == 'failed'
    assert TaskIndex(tmp_path).reserve('another')['number'] == 9


def test_task_package_path_uses_metadata_name(tmp_path):
    package = tmp_path / 'task_library' / 'circle-packing-versioned'
    package.mkdir(parents=True)
    (package / 'task.yaml').write_text(
        'id: circle-packing\nname: Circle Packing\n', encoding='utf-8'
    )

    entry = TaskIndex(tmp_path).reserve(str(package))

    assert entry['name'] == 'Circle Packing'
    assert str(package) not in entry['name']


def test_inline_runtime_options_are_removed_from_task_name(tmp_path):
    index = TaskIndex(tmp_path)

    entry = index.reserve('circle-packing cpu=8 workers=2 duration=2h')

    assert entry['name'] == 'circle-packing'


def test_sync_repairs_legacy_inline_options_from_run_metadata(tmp_path):
    index = TaskIndex(tmp_path)
    entry = index.reserve('circle-packing')
    index.update(
        entry['number'],
        name='circle-packing cpu=8 workers=2 duration=2h',
    )
    row = {
        'run_id': 'circle-run',
        'control_workspace': entry['workspace'],
        'status': 'completed',
        'created_at': 1,
        'draft': {'exp_id': 'circle-packing'},
    }

    synced = index.sync([row])

    assert synced[0]['name'] == 'circle-packing'


def test_sync_repairs_legacy_path_name_from_run_metadata(tmp_path):
    index = TaskIndex(tmp_path)
    entry = index.reserve('/old/task_library/circle-packing-versioned')
    row = {
        'run_id': 'circle-run',
        'control_workspace': entry['workspace'],
        'status': 'completed',
        'created_at': 1,
        'draft': {'exp_id': 'circle-packing'},
    }

    synced = index.sync([row])
    repaired = next(item for item in synced if item['number'] == entry['number'])

    assert repaired['name'] == 'circle-packing'
    assert json.loads((tmp_path / '.scienceflow/task-board/control.json').read_text())[
        'tasks'
    ][0]['name'] == 'circle-packing'


@pytest.mark.asyncio
async def test_history_badge_distinguishes_restored_tasks_from_current_session(tmp_path, monkeypatch):
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    tasks.index.reserve('restored-task')
    await tasks.new_workspace('current-task', Host())
    monkeypatch.setattr(manager, 'list_runs', list)
    tasks.catalog_at = manager.time.monotonic()

    _, _, items = tasks.collect()
    by_key = {item['key']: item for item in items}

    assert by_key['1']['row_name'] == 'restored-task · H'
    assert by_key['1']['row_full_name'] == 'restored-task · history'
    assert by_key['2']['row_name'] == 'current-task'


def test_symlink_outside_workspace_cannot_receive_task_files(tmp_path):
    root, outside = tmp_path / 'root', tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    (root / 'tasks').symlink_to(outside)
    with pytest.raises(ValueError, match='inside'):
        TaskIndex(root).reserve('circle')
    assert list(outside.iterdir()) == []


def test_resource_pools_reject_overlap_and_auto_chooses_unreserved_gpu(monkeypatch):
    from scienceflow.runtime.parallel.control import resource_claims
    monkeypatch.setattr(resource_claims.os, 'sched_getaffinity', lambda _: {0, 1, 2, 3})
    monkeypatch.setattr(resource_claims, 'query_gpu_devices', lambda: [SimpleNamespace(index=0), SimpleNamespace(index=1)])
    rows = [dict(alive=True, control_workspace='/other', resource_claims={'cpu': [0, 1], 'gpu': ['0']})]
    with pytest.raises(ResourceBusy, match='reserved'):
        reserve_resources({'cpu_list': '0', 'gpu_list': 'cpu'}, rows)
    assert reserve_resources({'cpu_list': '2-3', 'gpu_list': 'auto:1'}, rows) == {'cpu': [2, 3], 'gpu': ['1']}
    with pytest.raises(ResourceBusy, match='GPUs'):
        reserve_resources({'cpu_list': '2-3', 'gpu_list': 'auto:2'}, rows)


def test_resource_summary_shows_running_tasks_one_per_line_and_hides_stopping():
    entries = [
        {'number': 1, 'name': 'circle-packing', 'run_id': 'running'},
        {'number': 2, 'name': 'image-model', 'run_id': 'stopping'},
        {'number': 3, 'name': 'old-task', 'run_id': 'completed'},
    ]
    rows = {
        'running': {
            'alive': True,
            'status': 'running',
            'draft': {'workers': 2, 'gpu_list': '0'},
            'resource_claims': {'cpu': [0, 1, 2, 3], 'gpu': ['0']},
        },
        'stopping': {
            'alive': True,
            'status': 'stopping',
            'draft': {'workers': 1, 'gpu_list': '1'},
            'resource_claims': {'cpu': [4, 5, 6, 7], 'gpu': ['1']},
        },
        'completed': {
            'alive': False,
            'status': 'completed',
            'draft': {'workers': 1, 'gpu_list': '2'},
            'resource_claims': {'cpu': [8], 'gpu': ['2']},
        },
    }

    lines = allocation_summary(
        'Host · CPU 20%',
        entries,
        rows,
        cpu_ids=range(10),
        gpu_ids=range(3),
    ).splitlines()

    assert lines == [
        'Host · CPU 20%',
        'Task 1 · circle-packing · CPU 0-3 · GPU 0 · 2 workers',
        'Available · CPU 8-9 · GPU 2',
    ]


def test_resource_preview_keeps_each_running_and_proposed_task_on_one_line():
    entries = [
        {'number': 1, 'name': 'circle-packing', 'run_id': 'running'},
        {'number': 2, 'name': 'hidden', 'run_id': 'stopping'},
    ]
    rows = {
        'running': {
            'alive': True,
            'status': 'running',
            'draft': {'workers': 2, 'gpu_list': 'cpu'},
            'resource_claims': {'cpu': [0, 1, 2, 3], 'gpu': []},
        },
        'stopping': {
            'alive': True,
            'status': 'stopping',
            'draft': {'workers': 1, 'gpu_list': '0'},
            'resource_claims': {'cpu': [4, 5], 'gpu': ['0']},
        },
    }
    draft = LongResearchDraft(
        workers=2,
        cpu_list='6-13',
        gpu_list='auto:1',
        wall_clock_sec=1200,
    )

    lines = allocation_preview(
        entries,
        rows,
        {'number': 3, 'name': 'new-task'},
        draft,
    ).splitlines()

    assert lines == [
        'Task 1 · circle-packing · CPU 0-3 · GPU CPU-only · 2 workers',
        'Proposed Task 3 · new-task · CPU 6-13 · GPU auto:1 · 2 workers · 20 min',
    ]


def event(name, seconds, **payload):
    return {'event': name, 'worker_id': 'w00', 'timestamp_utc': datetime.fromtimestamp(seconds, timezone.utc).isoformat(),
            'payload': payload}


def test_gpu_minutes_union_shared_leases_and_executed_kills_only():
    facts = ResourceFacts()
    records = [event('resource_gpu_lease_acquired', 100, job_id='a', gpu_ids=['0']),
               event('resource_gpu_lease_acquired', 130, job_id='b', gpu_ids=['0']),
               event('resource_guard_action', 140, job_id='a', action='KILL', execution_status='pending'),
               event('resource_guard_action', 145, job_id='a', action='KILL', executed=False),
               event('resource_guard_action', 150, job_id='a', action='KILL', executed=True, reason='idle'),
               event('resource_gpu_lease_released', 160, job_id='a'),
               event('resource_gpu_lease_released', 190, job_id='b')]
    for record in records + records:
        facts.observe(record, 'w00')
    assert facts.gpu_summary(190, alive=False) == '0 · 1.5m'
    assert len(facts.killed) == 1
    assert len(facts.recent) == 3
    unknown = ResourceFacts()
    unknown.observe(event('resource_gpu_lease_released', 160, job_id='a'), 'w00')
    assert unknown.gpu_summary(190, alive=True) == '—'


def test_control_events_do_not_double_count_after_log_replay(tmp_path):
    path = tmp_path / 'events.jsonl'
    record = event('estra_decision', 100, action='keep_current')
    line = json.dumps(record) + '\n'
    path.write_text(line + line)
    projection = ResearchEvents()
    projection.read(path)
    path.write_text(line)  # Truncation/replay.
    projection.read(path)
    assert projection.counts['estra_decision'] == 1


def test_control_summary_shows_safe_and_legacy_estra_fallbacks(tmp_path):
    path = tmp_path / 'events.jsonl'
    records = [
        event('estra_decision', 100, action='keep_current', decision_mode='safe_fallback'),
        event('estra_decision', 101, action='keep_current', decision_mode='isolated'),
        event('estra_deterministic_fallback', 102, estra_action='keep_current'),
    ]
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))

    projection = ResearchEvents()
    projection.read(path)

    assert projection.control_summary().startswith(
        'ESTRA · Decisions 3 (fallback 2) · Switches 0'
    )


@pytest.mark.asyncio
async def test_two_launches_release_conversation_and_stop_targets_only_one(tmp_path, monkeypatch):
    owner = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    host = Host()
    rows, stopped = {}, []
    monkeypatch.setattr(manager, 'list_runs', lambda: list(rows.values()))
    monkeypatch.setattr(manager, 'get_run', lambda key: rows[key])

    def start(path, *, draft, check_resources, library_workspace):
        assert check_resources
        key = str(len(rows) + 1)
        rows[key] = dict(run_id=key, alive=True, status='running', draft={}, task_roots=[],
                         control_workspace=str(owner.task_workspace), started_at=100, log='/tmp/log')
        return rows[key]

    def stop(key, *, timeout):
        assert timeout == 0
        stopped.append(key)
        rows[key].update(status='stopping')
        return rows[key]

    monkeypatch.setattr(manager, 'start_run', start)
    monkeypatch.setattr(manager, 'stop_run', stop)
    try:
        for n in range(2):
            owner._task_workspace = await owner._managed.new_workspace('circle', host)
            owner.session = LongResearchSession.start(workspace=owner.task_workspace)
            owner.files = SimpleNamespace(manifest_path=tmp_path / 'unused')
            await owner._managed.launch(host)
            assert owner.session is None
            assert not owner.running
        assert len(host.items) == 2
        assert not await owner.try_handle('hello', host)
        assert await owner.try_handle('/stop', host)
        assert stopped == []
        assert 'Multiple tasks' in host.notices[-1]
        assert await owner.try_handle('/stop 2', host)
        assert stopped == ['2'] and rows['1']['alive']
        assert host.notices[-1] == 'Stop requested · Task 2 will preserve completed work.'
        before = {e['run_id']: e['number'] for e in owner._managed.entries}
        await owner.aclose()
        owner = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
        await owner.on_mount(host)
        assert {e['run_id']: e['number'] for e in owner._managed.entries} == before
        assert rows['1']['alive']  # Reopening/closing a TUI never stops background tasks.
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_failed_launch_retains_preparation_for_retry(tmp_path, monkeypatch):
    owner = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    host = Host()
    monkeypatch.setattr(manager, 'list_runs', list)
    monkeypatch.setattr(manager, 'start_run', lambda *a, **k: (_ for _ in ()).throw(ResourceBusy('CPU reserved')))
    try:
        owner._task_workspace = await owner._managed.new_workspace('circle', host)
        session = owner.session = LongResearchSession.start(workspace=owner.task_workspace)
        owner.files = SimpleNamespace(manifest_path=tmp_path / 'unused')
        await owner._managed.launch(host)
        assert owner.session is session and session.state == OnboardingState.CONFIRM
        assert owner.files is not None
        assert 'CPU reserved' in host.notices[-1]
        await owner._cancel(host)
        assert owner._managed.preparing is None
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_run_routing_prepares_next_task_while_previous_is_active(tmp_path, monkeypatch):
    from dataclasses import asdict

    owner = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    model_config = tmp_path / 'models.json'
    model_config.write_text(json.dumps({
        'models': {
            'test-model': {
                'model': 'test-model',
                'endpoints': [{'url': 'https://example.test/v1', 'key': 'test'}],
            },
        },
        'defaults': {
            'code_models': ['test-model'],
            'feedback_models': ['test-model'],
            'selection': 'auto',
        },
    }))
    owner.configure_model_registry(model_config, ())
    host = Host()
    rows = {}
    monkeypatch.setattr(manager, 'list_runs', lambda: list(rows.values()))

    def start(path, *, draft, check_resources, library_workspace):
        key = str(len(rows) + 1)
        row = dict(run_id=key, alive=True, status='running', draft=asdict(draft), task_roots=[],
                   control_workspace=str(owner.task_workspace), started_at=100, log='/tmp/log')
        rows[key] = row
        return row

    async def preflight(host):
        owner.session.apply_preflight(SimpleNamespace(status='exploratory'))
        owner.files = SimpleNamespace(manifest_path=tmp_path / 'unused')

    monkeypatch.setattr(manager, 'start_run', start)
    monkeypatch.setattr(owner, '_preflight', preflight)
    try:
        for index in range(2):
            assert await owner.try_handle('/long-research circle-packing data=none gpu=cpu cpu=0-1 workers=2 duration=10m', host)
            assert owner.session.state == OnboardingState.CONFIRM
            assert str(owner.task_workspace).endswith(f'task-{index + 1:04d}')
            assert await owner.try_handle('run', host)
            task = owner._run_task
            assert await owner.try_handle('run', host)  # No duplicate launch while scheduling.
            await task
            assert len(rows) == index + 1
            assert owner.session is None
        assert len({row['control_workspace'] for row in rows.values()}) == 2
        assert all(row['alive'] for row in rows.values())
    finally:
        await owner.aclose()


def test_auto_gpu_is_pinned_in_both_record_and_execution_manifest(monkeypatch):
    from scienceflow.runtime.parallel.control import resource_claims

    monkeypatch.setattr(resource_claims, 'query_gpu_devices', lambda: [SimpleNamespace(index=3)])
    record = dict(draft={'gpu_list': 'auto:1', 'cpu_list': ''},
                  manifest_payload={'tasks': [{'gpu_list': 'auto:1'}]})
    resource_claims.apply_reservation(record, [])
    assert record['resource_claims']['gpu'] == ['3']
    assert record['draft']['gpu_list'] == '3'
    assert record['manifest_payload']['tasks'][0]['gpu_list'] == '3'


def test_recent_events_merge_workers_by_timestamp():
    facts = ResourceFacts()
    for stamp in (300, 400, 500):
        facts.observe(event('estra_stage_switched', stamp), 'w00')
    facts.observe(event('estra_stage_switched', 100), 'w01')
    assert all('w00' in line for line in facts.recent)
    facts.observe(event('estra_stage_switched', 600), 'w01')
    assert 'w01' in facts.recent[-1]


def test_resource_selection_preserves_legacy_auto_and_cpu_aliases(monkeypatch):
    from scienceflow.runtime.parallel.control import resource_claims

    monkeypatch.setattr(resource_claims, 'query_gpu_devices', lambda: [SimpleNamespace(index=i) for i in range(4)])
    assert reserve_resources({'gpu_list': 'auto', 'workers': 3}, [])['gpu'] == ['0', '1', '2']
    assert reserve_resources({'gpu_list': 'auto:2:1-3'}, [])['gpu'] == ['1', '2']
    assert reserve_resources({'gpu_list': 'disabled'}, [])['gpu'] == []


def test_explicit_attach_preserves_existing_external_run_addressing(tmp_path, monkeypatch):
    owner = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    row = dict(run_id='external-run', control_workspace='/external/workspace', status='running', alive=True)
    monkeypatch.setattr(manager, 'resolve_run', lambda target, cwd: row)
    entry = owner._managed.select('attach', 'external-run')
    assert entry['number'] == 1
    assert entry['workspace'] == '/external/workspace'
    assert TaskIndex(tmp_path).sync([row])[0]['run_id'] == 'external-run'
    assert owner._managed.select('stop', '1')['run_id'] == 'external-run'


def test_task_display_is_newest_first_without_renumbering(tmp_path, monkeypatch):
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    for name in ('first', 'second', 'third'):
        tasks.index.reserve(name)
    monkeypatch.setattr(manager, 'list_runs', list)
    tasks.catalog_at = manager.time.monotonic()
    entries, _, items = tasks.collect()
    assert [entry['number'] for entry in entries] == [1, 2, 3]
    assert [item['key'] for item in items] == ['3', '2', '1']
    tasks.index.update(1, status='completed')
    assert [item['key'] for item in tasks.collect()[2]] == ['3', '2', '1']


def test_task_monitor_loads_history_once_and_polls_only_active_runs(
    tmp_path, monkeypatch
):
    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    running = {
        'run_id': 'active-run',
        'status': 'running',
        'alive': True,
        'created_at': 1,
    }
    completed = dict(running, status='completed', alive=False)
    registry_calls = []
    active_rows = iter((running, completed))

    monkeypatch.setattr(
        manager, 'list_runs', lambda: registry_calls.append(True) or [running]
    )
    monkeypatch.setattr(manager, 'get_run', lambda _: next(active_rows))
    monkeypatch.setattr(tasks.index, 'sync', lambda _: [])
    tasks.catalog_at = manager.time.monotonic()

    tasks.collect()
    tasks.collect()
    tasks.collect()
    tasks.collect()

    assert len(registry_calls) == 1
    assert tasks._registry_rows['active-run']['status'] == 'completed'


def test_task_catalog_is_cached_until_the_library_changes(tmp_path, monkeypatch):
    from scienceflow.research.onboarding.library import catalog

    tasks = manager.MultiResearch(SimpleNamespace(workspace=tmp_path))
    tasks._registry_loaded = True
    monkeypatch.setattr(tasks.index, 'sync', lambda _rows: [])
    calls = []
    monkeypatch.setattr(
        catalog,
        'task_choices',
        lambda _workspace: calls.append(True) or (('demo', 'Demo'),),
    )

    tasks.collect()
    tasks.collect()
    assert len(calls) == 1

    tasks.catalog_at = None
    tasks.collect()
    assert len(calls) == 2
