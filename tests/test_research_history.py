import asyncio
import json
from dataclasses import asdict
from time import monotonic
from types import SimpleNamespace

import pytest

from scienceflow.interfaces.ui import long_research, long_research_progress
from scienceflow.interfaces.ui.long_research import LongResearchInteraction
from scienceflow.interfaces.ui.research.control import attachment
from scienceflow.interfaces.ui.research.control.tasks import manager
from scienceflow.interfaces.ui.research.history import ResearchHistory
from scienceflow.research.onboarding import LongResearchDraft, LongResearchSession
from scienceflow.runtime.parallel.service import ParallelRunSummary


pytestmark = pytest.mark.usefixtures("research_model_registry")


class HistoryHost:
    busy = False

    def __init__(self):
        self.events = []
        self.fold_error = False

    async def begin_transcript_section(self):
        section = object()
        self.events.append(('begin', section))
        return section

    async def fold_transcript_section(self, section, summary):
        if self.fold_error:
            raise RuntimeError('view cleared')
        self.events.append(('fold', section, summary))
        return True

    async def notice(self, message):
        self.events.append(('notice', message))

    async def echo_user(self, message):
        self.events.append(('user', message))

    def conversation_snapshot(self):
        return ({'role': 'user', 'content': 'earlier chat'},)

    def set_host_status(self, text):
        pass


def draft():
    return LongResearchDraft(
        task_text='circle-packing', workers=2, cpu_list='0-15', gpu_list='cpu',
        wall_clock_sec=3600, code_models=['chosen'], feedback_models=['chosen'],
    )


@pytest.mark.asyncio
async def test_session_activity_replays_without_changing_chat_or_recording_queries(tmp_path):
    host = HistoryHost()
    messages = [SimpleNamespace(role='user', content='first chat')]
    host.runtime = SimpleNamespace(
        options=SimpleNamespace(session_log=tmp_path / 'session.jsonl'),
        conversation=SimpleNamespace(messages=messages),
    )
    history = ResearchHistory()
    await history.record(host, 'Task 1 · preparation started')
    messages.append(SimpleNamespace(role='assistant', content='answer'))
    await history.record(host, 'Task 1 · running · run abc')
    saved = (tmp_path / 'scienceflow_activity.jsonl').read_text()
    assert '/status' not in saved and len(messages) == 2
    assert (tmp_path / 'scienceflow_activity.jsonl').stat().st_mode & 0o777 == 0o600

    restored = ResearchHistory()
    await restored.load(host)
    assert restored.history_before(0) == ()
    assert restored.history_before(1) == ()
    assert restored.history_before(2) == ('Task 1 · running',)
    assert restored.history_before(2) == ()

    legacy = ResearchHistory()
    await legacy.load(host)
    await legacy.show_legacy_history(host)
    assert 'Task 1 · running' in host.events[-1][1]
    assert 'preparation started' not in host.events[-1][1]
    assert legacy.history_before(2) == ()


@pytest.mark.asyncio
async def test_restored_activity_keeps_one_latest_line_per_task(tmp_path):
    path = tmp_path / 'scienceflow_activity.jsonl'
    events = (
        (1, 'Task 2 · preparation started'),
        (1, 'Long research · preparation cancelled'),
        (2, 'Task 3 · preparation started'),
        (3, 'Task 3 · preflight verified'),
        (4, 'Task 3 · circle-packing · starting · run circle-packing-20261006T220241999247Z-517c13d5'),
    )
    path.write_text(''.join(
        json.dumps({'message_index': position, 'summary': summary}) + '\n'
        for position, summary in events
    ), encoding='utf-8')
    original = path.read_bytes()
    host = HistoryHost()
    host.runtime = SimpleNamespace(options=SimpleNamespace(session_log=tmp_path / 'session.jsonl'))

    history = ResearchHistory()
    await history.load(host)

    assert history.history_before(1) == ('Task 2 · preparation cancelled',)
    assert history.history_before(2) == ()
    assert history.history_before(3) == ()
    assert history.history_before(4) == ('Task 3 · circle-packing · starting',)
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_display_history_is_optional_and_does_not_modify_draft():
    history = ResearchHistory()
    original = draft()
    snapshot = asdict(original)
    await history.begin(object())
    await history.fold(object(), original, 'Task 1')
    host = HistoryHost()
    await history.begin(host)
    section = history.section
    await history.fold(host, original, 'Task 1 · circle-packing')
    assert host.events[-1] == (
        'fold', section,
        'Task 1 · circle-packing · Configuration · 2 workers · CPU 0-15 · GPU cpu · 60 min · chosen',
    )
    assert asdict(original) == snapshot
    count = len(host.events)
    await history.fold(host, original, 'again')
    assert len(host.events) == count


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['running', 'starting', 'failed', 'interrupted', 'exception', 'view_error'])
async def test_multi_research_shows_one_task_line_after_success(tmp_path, monkeypatch, outcome):
    owner = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    owner._model_config_path = tmp_path / 'missing.json'
    host = HistoryHost()
    monkeypatch.setattr(owner._managed, 'ensure_poll', lambda host: None)

    async def refresh(host):
        pass

    monkeypatch.setattr(owner._managed, 'refresh', refresh)
    await owner.try_handle('/long-research circle-packing', host)
    assert host.events[0][0] == 'begin' and host.events[1] == ('user', '/long-research circle-packing')
    owner.session.draft = original = draft()
    snapshot = asdict(original)
    owner.files = SimpleNamespace(manifest_path=tmp_path / 'unused.yaml')
    assert not any(event[0] == 'fold' for event in host.events)

    def start(*args, **kwargs):
        if outcome == 'exception':
            raise ValueError('launch failed')
        status = 'running' if outcome == 'view_error' else outcome
        return {'run_id': 'run-1', 'status': status, 'alive': status in {'running', 'starting'},
                'draft': snapshot, 'resource_claims': {'cpu': list(range(16)), 'gpu': []}}

    host.fold_error = outcome == 'view_error'
    monkeypatch.setattr(manager, 'start_run', start)
    try:
        launch_offset = len(host.events)
        await owner._managed.launch(host)
        launch_events = host.events[launch_offset:]
        folds = [event for event in host.events if event[0] == 'fold']
        assert len(folds) == int(outcome in {'running', 'starting'})
        assert asdict(original) == snapshot
        if folds:
            assert folds[0][2] == (
                'Task 1 · circle-packing · CPU 0-15 · GPU CPU-only · 2 workers · 60 min'
            )
            assert not any(event[0] == 'notice' for event in launch_events)
            assert owner.session is None
        if outcome == 'exception':
            assert owner.session is not None
            assert any('launch failed' in event[1] for event in host.events if event[0] == 'notice')
        if outcome == 'view_error':
            assert owner.session is None
            notices = [event[1] for event in launch_events if event[0] == 'notice']
            assert notices == [
                'Task 1 · circle-packing · CPU 0-15 · GPU CPU-only · 2 workers · 60 min'
            ]
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_cancelled_preparation_folds_after_busy_request_was_ignored(tmp_path):
    owner = LongResearchInteraction(tmp_path)
    owner._model_config_path = tmp_path / 'missing.json'
    host = HistoryHost()
    host.busy = True
    await owner.try_handle('/long-research', host)
    assert not any(event[0] == 'begin' for event in host.events)
    host.busy = False
    await owner.try_handle('/long-research circle-packing', host)
    assert not any(event[0] == 'fold' for event in host.events)
    await owner.try_handle('/cancel', host)
    assert owner.session is None
    folds = [event for event in host.events if event[0] == 'fold']
    assert len(folds) == 1
    assert folds[0][2] == 'Long research · Preparation cancelled'
    assert host.events.index(folds[0]) > next(
        index for index, event in enumerate(host.events)
        if event[0] == 'notice' and 'cancelled' in event[1]
    )
    await owner.aclose()


@pytest.mark.asyncio
async def test_cancelled_task_selection_folds_without_a_draft(tmp_path):
    owner = LongResearchInteraction(tmp_path)
    host = HistoryHost()

    await owner.try_handle('/long-research', host)
    assert owner._awaiting_task_target
    await owner.try_handle('/cancel', host)

    assert not owner._awaiting_task_target
    assert [event[2] for event in host.events if event[0] == 'fold'] == [
        'Long research · Task selection cancelled'
    ]
    await owner.aclose()


@pytest.mark.asyncio
async def test_cancelled_managed_preparation_names_task_in_fold(tmp_path, monkeypatch):
    owner = LongResearchInteraction(tmp_path, managed=True, multi_task=True)
    host = HistoryHost()
    await owner._history.begin(host)
    owner.session = LongResearchSession(draft())
    owner._managed.preparing = {'number': 3}

    async def cancel_preparation():
        owner._managed.preparing = None

    monkeypatch.setattr(owner._managed, 'cancel_preparation', cancel_preparation)
    await owner.try_handle('/cancel', host)

    assert owner.session is None
    assert [event[2] for event in host.events if event[0] == 'fold'] == [
        'Task 3 · Preparation cancelled'
    ]
    await owner.aclose()


@pytest.mark.asyncio
async def test_single_managed_research_folds_before_attached_notice(tmp_path, monkeypatch):
    owner = LongResearchInteraction(tmp_path, managed=True)
    owner.session = LongResearchSession(draft())
    owner.files = SimpleNamespace(manifest_path=tmp_path / 'unused.yaml')
    host = HistoryHost()
    await owner._history.begin(host)
    monkeypatch.setattr(attachment, 'start_run', lambda *a, **k: {'alive': True, 'status': 'running'})

    async def watch(row, host):
        assert host.events[-1][0] == 'fold'
        await host.notice('Attached')

    monkeypatch.setattr(owner._managed, '_watch', watch)
    await owner._managed.launch(host)
    assert host.events[-1] == ('notice', 'Attached')
    await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'failed_before_start', 'failed_after_start'])
async def test_direct_run_uses_started_callback_not_run_confirmation(tmp_path, monkeypatch, outcome):
    owner = LongResearchInteraction(tmp_path)
    original = draft()
    owner.session = LongResearchSession(original)
    owner.files = SimpleNamespace(manifest_path=tmp_path / 'unused.yaml')
    host = HistoryHost()
    await owner._history.begin(host)

    async def watch(host):
        await asyncio.Event().wait()

    progress = SimpleNamespace(
        draft=original, budget=3600, root=tmp_path, started=monotonic(),
        watch=watch, snapshot=lambda: None,
    )
    monkeypatch.setattr(long_research_progress, 'LongResearchProgress', lambda draft: progress)

    async def run(manifest, *, on_started):
        assert not any(event[0] == 'fold' for event in host.events)
        if outcome == 'failed_before_start':
            raise ValueError('runner could not start')
        on_started(1, 1)
        if outcome == 'failed_after_start':
            raise ValueError('worker failed after start')
        return ParallelRunSummary(results=(), text='done', task_count=1, max_concurrent=1)

    monkeypatch.setattr(long_research, 'run_manifest', run)
    await owner._run(host)
    folds = [event for event in host.events if event[0] == 'fold']
    assert len(folds) == int(outcome != 'failed_before_start')
    if outcome != 'failed_before_start':
        index = host.events.index(folds[0])
        assert host.events[index + 1][1].startswith('Running in worker subprocesses')
    if outcome != 'success':
        assert host.events[-1][1].startswith('Long research failed')
    await owner.aclose()
