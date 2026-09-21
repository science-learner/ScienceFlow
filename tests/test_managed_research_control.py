from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import psutil
import pytest
from click.testing import CliRunner

from scienceflow.interfaces.cli import main
from scienceflow.interfaces.ui.long_research import LongResearchInteraction
from scienceflow.research.onboarding import LongResearchSession, write_onboarding_files
from scienceflow.runtime.parallel.control import (
    get_run,
    resume_run,
    service,
    start_run,
    stop_run,
)
from scienceflow.runtime.parallel.control.registry import alive


@pytest.fixture
def controlled_run(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'problem.json').write_text('{}')
    session = LongResearchSession.start(workspace=tmp_path, constraints=(
        f'circle-packing data={data} gpu=cpu cpu=0-1 workers=2 duration=10m'))
    manifest = write_onboarding_files(session.draft, tmp_path).manifest_path
    original = subprocess.Popen
    # Exercise the actual supervisor lifecycle with two harmless child processes.
    script = '''
import asyncio,json,sys
from pathlib import Path
from types import SimpleNamespace
from scienceflow.runtime.parallel.control import worker
async def synthetic(manifest):
    children=[]
    try:
        for _ in range(2):
            children.append(await asyncio.create_subprocess_exec(sys.executable,'-c','import time; time.sleep(120)'))
        Path(manifest).with_suffix('.children.json').write_text(json.dumps([p.pid for p in children]))
        await asyncio.sleep(120)
        return SimpleNamespace(text='done',results=[])
    finally:
        for child in children:
            if child.returncode is None: child.terminate()
        await asyncio.gather(*(p.wait() for p in children))
worker.run_manifest=synthetic
asyncio.run(worker.supervise(sys.argv[1],sys.argv[2]))
'''

    def launch(args, **kwargs):
        return original([args[0], '-c', script, *args[-2:]], **kwargs)

    monkeypatch.setattr(service.subprocess, 'Popen', launch)
    row = start_run(manifest, draft=session.draft)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if Path(row['manifest']).with_suffix('.children.json').exists():
                break
            time.sleep(.05)
        else:
            pytest.fail('Synthetic supervisor did not start.')
        yield row, manifest, session
    finally:
        stop_run(row['run_id'], timeout=10)


def test_detached_supervisor_stop_resume_and_duplicate_guard(controlled_run):
    row, manifest, session = controlled_run
    run_id = row['run_id']
    assert get_run(run_id)['alive']
    with pytest.raises(ValueError, match='already running'):
        start_run(manifest, draft=session.draft)
    pids = json.loads(Path(row['manifest']).with_suffix('.children.json').read_text())
    stopped = stop_run(run_id, timeout=10)
    assert stopped['status'] == 'stopped'
    assert not stopped['alive']
    assert all(not psutil.pid_exists(pid) for pid in pids)
    checkpoint = Path(row['task_roots'][0]) / 'task_logs' / 'state.json'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({'status': 'stopped_by_user', 'charged_elapsed_sec': 30}))
    resumed = resume_run(run_id)
    assert resumed['display_budget_sec'] == 570
    assert resumed['run_id'] == run_id
    assert resumed['attempt'] == 2
    import yaml
    payload = yaml.safe_load(Path(resumed['manifest']).read_text())
    assert payload['resume'] is True
    assert payload['resume_budget_policy'] == 'remaining'
    assert resumed['task_roots'] == row['task_roots']
    with pytest.raises(ValueError, match='already active'):
        resume_run(run_id)


@pytest.mark.asyncio
async def test_tui_close_detaches_without_stopping_supervisor(controlled_run, tmp_path):
    row, _, _ = controlled_run

    class Host:
        busy = False
        def set_host_status(self, value): pass
        async def notice(self, value): pass

    interaction = LongResearchInteraction(tmp_path, managed=True)
    await interaction._managed.attach(get_run(row['run_id']), Host())
    await asyncio.sleep(.1)
    await interaction.aclose()
    assert get_run(row['run_id'])['alive']


def test_pid_reuse_identity_does_not_match():
    process = psutil.Process()
    assert not alive({'pid': process.pid, 'process_created': process.create_time() - 5})


def test_cli_positional_target_and_tui_flag_dispatch_to_shared_control(monkeypatch, tmp_path):
    from scienceflow.interfaces.cli.commands.run import control
    captured = {}
    monkeypatch.setattr(control, 'managed_run', lambda target, **kwargs: captured.update(target=target, **kwargs))
    result = CliRunner().invoke(main, ['run', 'circle-packing', '--tui', '--workers', '2', '--duration', '20m', '--workspace', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert captured['target'] == 'circle-packing'
    assert captured['with_tui'] is True
    assert captured['workers'] == 2


def test_status_does_not_expose_manifest_credentials(controlled_run):
    row, _, _ = controlled_run
    assert 'manifest_payload' not in get_run(row['run_id'])
    assert 'launch_token' not in get_run(row['run_id'])
    result = CliRunner().invoke(main, ['status', row['run_id']])
    assert result.exit_code == 0, result.output
    assert row['run_id'] in result.output
    assert 'manifest_payload' not in result.output


@pytest.mark.asyncio
async def test_stop_then_resume_can_be_sent_immediately(controlled_run, tmp_path):
    row, _, _ = controlled_run
    class Host:
        busy = False
        def set_host_status(self, value): pass
        async def notice(self, value): pass
    host = Host()
    interaction = LongResearchInteraction(tmp_path, managed=True)
    await interaction._managed.attach(get_run(row['run_id']), host)
    await asyncio.sleep(.05)
    await interaction.try_handle('/stop', host)
    assert interaction.session is None
    await interaction.try_handle('/resume ' + row['run_id'], host)
    await asyncio.sleep(.05)
    assert get_run(row['run_id'])['attempt'] == 2
    await interaction.aclose()


def test_legacy_worker_does_not_relaunch_from_inherited_management_env(monkeypatch, tmp_path):
    from scienceflow.interfaces.cli.commands.run import task
    manifest = tmp_path / 'tasks.yaml'
    manifest.write_text('tasks: []')
    class LegacyReached(Exception):
        pass
    def legacy(config):
        raise LegacyReached()
    monkeypatch.setattr(task, 'load_cfg', legacy)
    result = CliRunner().invoke(main, ['run', '--task', 'worker objective'], env={
        'SCIENCEFLOW_MANIFEST': str(manifest), 'SCIENCEFLOW_RUN_TUI': 'true',
        'SCIENCEFLOW_RUN_WORKERS': '2'})
    assert isinstance(result.exception, LegacyReached)


def test_directory_resolution_current_dir_symlink_and_ambiguity(controlled_run, monkeypatch, tmp_path):
    from scienceflow.runtime.parallel.control import AmbiguousRun, resolve_run
    from scienceflow.runtime.parallel.control.registry import (
        read_record,
        run_dir,
        write_record,
    )
    row, _, _ = controlled_run
    folder = Path(row['control_workspace'])
    assert resolve_run(str(folder))['run_id'] == row['run_id']
    assert resolve_run(cwd=folder)['run_id'] == row['run_id']
    link = tmp_path / 'alias'
    link.symlink_to(folder, target_is_directory=True)
    assert resolve_run(str(link))['run_id'] == row['run_id']
    with pytest.raises(ValueError, match='No managed run'):
        resolve_run(cwd=folder.parent)
    copied = read_record(run_dir(row['run_id']))
    copied.update(run_id='another-run', pid=0, status='stopped')
    write_record(run_dir('another-run'), copied)
    with pytest.raises(AmbiguousRun):
        resolve_run(str(folder))
    result = CliRunner().invoke(main, ['stop', str(folder)])
    assert result.exit_code == 0, result.output
    assert not get_run(row['run_id'])['alive']
    assert get_run('another-run')['status'] == 'stopped'


def test_concurrent_resume_starts_exactly_one_process(controlled_run):
    from concurrent.futures import ThreadPoolExecutor
    row, _, _ = controlled_run
    stop_run(row['run_id'], timeout=10)
    def attempt():
        try:
            return resume_run(row['run_id'])['attempt']
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results, key=lambda v: v or 0) == [None, 2]
    assert get_run(row['run_id'])['attempt'] == 2


def test_same_creation_folder_cannot_start_a_different_run(controlled_run):
    row, _, session = controlled_run
    session.draft.run_id = 'different-run-same-folder'
    manifest = write_onboarding_files(session.draft, Path(row['control_workspace'])).manifest_path
    with pytest.raises(ValueError, match='already running'):
        start_run(manifest, draft=session.draft)


def test_completed_and_exhausted_runs_do_not_resume(controlled_run):
    from scienceflow.runtime.parallel.control.registry import (
        read_record,
        run_dir,
        write_record,
    )
    row, _, _ = controlled_run
    stop_run(row['run_id'], timeout=10)
    directory = run_dir(row['run_id'])
    record = read_record(directory)
    record['status'] = 'completed'
    write_record(directory, record)
    with pytest.raises(ValueError, match='already completed'):
        resume_run(row['run_id'])
    record['status'] = 'stopped'
    write_record(directory, record)
    checkpoint = Path(row['task_roots'][0]) / 'task_logs/state.json'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({'status': 'stopped_by_user', 'charged_elapsed_sec': 600}))
    with pytest.raises(ValueError, match='No remaining budget'):
        resume_run(row['run_id'])
    assert get_run(row['run_id'])['attempt'] == 1


def test_cli_stop_without_id_uses_current_directory(controlled_run, monkeypatch):
    row, _, _ = controlled_run
    monkeypatch.chdir(row['control_workspace'])
    result = CliRunner().invoke(main, ['stop'])
    assert result.exit_code == 0, result.output
    assert get_run(row['run_id'])['status'] == 'stopped'


def test_repeated_stop_does_not_signal_cleanup_twice(controlled_run, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    row, _, _ = controlled_run
    signals = []
    original = service.os.kill
    def signal_once(pid, sig):
        if pid == row['pid']:
            signals.append(sig)
        return original(pid, sig)
    monkeypatch.setattr(service.os, 'kill', signal_once)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: stop_run(row['run_id'], timeout=10), range(2)))
    assert len(signals) == 1
    assert all(not result['alive'] for result in results)


@pytest.mark.asyncio
async def test_duplicate_tui_launch_attaches_existing_run_without_applying_new_draft(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scienceflow.interfaces.ui.research.control import attachment
    from scienceflow.research.onboarding import LongResearchDraft
    from scienceflow.runtime.parallel.control.service import WorkspaceBusy

    owner = SimpleNamespace(files=SimpleNamespace(manifest_path=tmp_path / 'manifest.yaml'),
                            session=SimpleNamespace(draft=LongResearchDraft(workers=8)))
    control = attachment.ManagedResearch(owner)
    notices, watched = [], []

    async def notice(text):
        notices.append(text)

    def start(*args, **kwargs):
        raise WorkspaceBusy('existing')

    async def watch(row, host):
        watched.append((row['run_id'], owner.session.draft.workers))

    monkeypatch.setattr(attachment, 'start_run', start)
    monkeypatch.setattr(attachment, 'get_run', lambda run_id: {
        'run_id': run_id, 'alive': True, 'draft': {'workers': 4}})
    monkeypatch.setattr(control, '_watch', watch)
    await control.launch(SimpleNamespace(notice=notice, set_host_status=lambda text: None))
    assert watched == [('existing', 4)]
    assert any('新配置未启动' in text for text in notices)
    assert not any('Could not launch' in text for text in notices)


def test_workspace_selection_prefers_active_and_never_other_workspace(tmp_path, monkeypatch):
    from scienceflow.runtime.parallel.control import selection
    local = tmp_path / 'local'
    rows = [dict(run_id='old', control_workspace=str(local), alive=False, status='stopped'),
            dict(run_id='active', control_workspace=str(local), alive=True, status='running'),
            dict(run_id='elsewhere', control_workspace=str(tmp_path / 'other'), alive=True, status='running')]
    monkeypatch.setattr(selection, 'list_runs', lambda: rows)
    assert selection.select_workspace_run('resume', cwd=local)['run_id'] == 'active'
    rows[1]['alive'] = False
    rows[1]['status'] = 'stopped'
    with pytest.raises(selection.AmbiguousRun):
        selection.select_workspace_run('resume', cwd=local)
    with pytest.raises(ValueError, match='No active'):
        selection.select_workspace_run('stop', cwd=local)
    assert len(selection.workspace_runs(local)) == 2


@pytest.mark.asyncio
async def test_mount_attaches_active_but_does_not_resume_stopped(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from scienceflow.interfaces.ui.research.control import attachment
    owner = SimpleNamespace(workspace=tmp_path)
    managed = attachment.ManagedResearch(owner)
    managed.attach = AsyncMock()
    host = SimpleNamespace(notice=AsyncMock())
    row = dict(run_id='saved', alive=False, status='stopped')
    monkeypatch.setattr(attachment, 'workspace_runs', lambda _: [row])
    await managed.on_mount(host)
    managed.attach.assert_not_awaited()
    assert '/resume' in host.notice.call_args.args[0]
    row['alive'] = True
    await managed.on_mount(host)
    managed.attach.assert_awaited_once_with(row, host)


def test_cross_task_resource_reservations_are_atomic(controlled_run, tmp_path):
    import os
    from concurrent.futures import ThreadPoolExecutor

    from scienceflow.runtime.parallel.control.resource_claims import ResourceBusy

    free = sorted(set(os.sched_getaffinity(0)) - {0, 1})
    if len(free) < 2:
        pytest.skip('Requires two CPU cores outside the fixture reservation')
    manifests = []
    for name in ('task-a', 'task-b'):
        workspace = tmp_path / name
        session = LongResearchSession.start(workspace=workspace, constraints=(
            f'circle-packing data=none gpu=cpu cpu={free[0]},{free[1]} workers=2 duration=10m'))
        files = write_onboarding_files(session.draft, workspace)
        manifests.append((files.manifest_path, session.draft))

    def launch(spec):
        try:
            return start_run(spec[0], draft=spec[1], check_resources=True)
        except ResourceBusy:
            return None

    started = []
    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(launch, manifests))
        started = [row for row in results if row]
        assert len(started) == 1
        assert get_run(controlled_run[0]['run_id'])['alive']
        stopped = stop_run(started[0]['run_id'], timeout=10)
        assert not stopped['alive']
        other = manifests[results.index(None)]
        second = start_run(other[0], draft=other[1], check_resources=True)
        started.append(second)
        with pytest.raises(ResourceBusy):
            resume_run(stopped['run_id'], check_resources=True)
        assert get_run(second['run_id'])['alive']
    finally:
        for row in started:
            stop_run(row['run_id'], timeout=10)
