"""Portable task folders, success eligibility, and context-aware completion."""

import asyncio
import json
import shlex
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from scienceflow.interfaces.ui.research.control.tasks.completion import task_completions
from scienceflow.research.onboarding.library.catalog import resolve_local, task_choices
from scienceflow.research.onboarding.library.export import export_success
from scienceflow.research.onboarding.session import LongResearchSession
from scienceflow.research.onboarding.support.preparation.validation import (
    validate_evaluator,
)
from scienceflow.runtime.task_package import load_task_package, prepare_task_runtime


def experiment(tmp_path):
    workspace = tmp_path / 'experiment'
    workspace.mkdir()
    script = workspace / 'evaluate.py'
    script.write_text('import json, sys\nx = json.load(open(sys.argv[1]))\nassert x > 0\nprint(json.dumps({"metric": x}))\n')
    valid, invalid = workspace / 'valid.json', workspace / 'invalid.json'
    valid.write_text('2')
    invalid.write_text('-1')
    command = f'{shlex.quote(sys.executable)} {shlex.quote(str(script))} {{artifact_abs_path}}'
    asyncio.run(validate_evaluator(command, {'valid_artifact': str(valid), 'invalid_artifact': str(invalid)}, workspace))
    root = workspace / 'runs' / 'demo'
    state = root / 'workers/w00/logs/lhr_state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'global_best': {'validation_ok': True, 'metric_value': 2}}))
    return {'run_id': 'run-one', 'status': 'completed', 'library_workspace': str(tmp_path), 'task_roots': [str(root)],
            'draft': {'workspace_base': str(workspace / 'runs'), 'task_text': 'Maximize a positive number.',
                      'exp_id': 'positive-number', 'metric_name': 'score', 'lower_is_better': False,
                      'artifact_path': 'solution.json', 'artifact_command': command, 'input_data_dir': 'none'}}


def test_export_copied_folder_runs_without_original_workspace(tmp_path):
    record = experiment(tmp_path)
    result = export_success(record)
    assert result['status'] == 'registered', result
    folder = Path(result['path'])
    spec = load_task_package(folder)
    assert spec.config['inputs'] == {'required': False}
    assert 'workers' not in spec.config and 'api_keys' not in spec.config
    copied = tmp_path / 'elsewhere' / 'task with spaces'
    shutil.copytree(folder, copied)
    shutil.rmtree(tmp_path / 'experiment')
    session = LongResearchSession.start(workspace=tmp_path, constraints=shlex.quote(str(copied)), natural_language=True)
    assert session.draft.registered_task
    assert session.draft.input_data_dir == 'none'
    assert session.draft.workers is None
    samples = {key: str(copied / value) for key, value in spec.config['validation'].items()}
    evidence = asyncio.run(validate_evaluator(session.draft.artifact_command, samples, tmp_path / 'checks'))
    assert evidence['valid_passed'] and evidence['invalid_rejected']


def test_repeat_and_reused_package_update_history_and_best(tmp_path):
    record = experiment(tmp_path)
    first = export_success(record)
    assert first['status'] == 'registered', first
    assert export_success(record)['path'] == first['path']
    folder = Path(first['path'])
    session = LongResearchSession.start(workspace=tmp_path, constraints=shlex.quote(str(folder)), natural_language=True)
    record['draft'] = asdict(session.draft)
    record['run_id'] = 'run-two'
    state = Path(record['task_roots'][0]) / 'workers/w00/logs/lhr_state.json'
    state.write_text(json.dumps({'global_best': {'validation_ok': True, 'metric_value': 3}}))
    second = export_success(record)
    assert second['status'] == 'registered', second
    assert second['path'] == first['path']
    provenance = json.loads((folder / 'provenance.json').read_text())
    assert len(provenance['runs']) == 2 and provenance['best_metric'] == 3


def test_contract_change_creates_separate_version(tmp_path):
    record = experiment(tmp_path)
    first = export_success(record)
    record['draft']['task_text'] = 'A different scientific objective.'
    record['run_id'] = 'run-two'
    second = export_success(record)
    assert first['status'] == second['status'] == 'registered'
    assert first['path'] != second['path']


def test_failed_invalid_or_nonportable_results_are_not_registered(tmp_path):
    record = experiment(tmp_path)
    record['status'] = 'failed'
    assert export_success(record)['status'] == 'skipped'
    record['status'] = 'completed'
    state = Path(record['task_roots'][0]) / 'workers/w00/logs/lhr_state.json'
    state.write_text(json.dumps({'global_best': {'validation_ok': False, 'metric_value': 9}}))
    assert export_success(record)['status'] == 'deferred'
    state.write_text(json.dumps({'global_best': {'validation_ok': True, 'metric_value': 2}}))
    record['draft']['task_text'] = 'api_key="private-credential"'
    assert export_success(record)['status'] == 'deferred'
    assert not list((tmp_path / 'task_library').glob('*/task.yaml'))


def test_local_catalog_handles_bad_folders_and_long_natural_descriptions(tmp_path):
    record = experiment(tmp_path)
    result = export_success(record)
    bad = tmp_path / 'task_library' / 'broken'
    bad.mkdir()
    (bad / 'task.yaml').write_text('[')
    assert resolve_local('positive-number', tmp_path).source_dir == Path(result['path'])
    assert resolve_local('a' * 1000, tmp_path) is None
    assert any('positive-number' in description and 'Local' in description for _, description in task_choices(tmp_path))


def test_explicit_entrypoint_package_runtime_is_isolated(tmp_path):
    source = tmp_path / 'portable'
    source.mkdir()
    (source / 'task.yaml').write_text('id: demo\nevaluator:\n  entrypoint: evaluate.py:evaluate\n')
    (source / 'evaluate.py').write_text('def evaluate(context):\n    return {"metric": 1}\n')
    runtime = prepare_task_runtime('demo', task_root=tmp_path / 'run', source_dir=source)
    assert runtime.entrypoint_path.is_file()
    assert runtime.entrypoint_path.is_relative_to(tmp_path / 'run')
    shutil.rmtree(source)
    assert runtime.entrypoint_path.is_file()


def test_control_completion_filters_targets_without_executing():
    manager = SimpleNamespace(entries=[{'number': i, 'name': f'Task {i}', 'run_id': str(i)} for i in range(1, 4)],
                              rows={'1': {'alive': True, 'status': 'running'}, '2': {'alive': False, 'status': 'stopped'},
                                    '3': {'alive': False, 'status': 'completed'}},
                              task_catalog=(("'/tasks/a b'", 'Demo · Local'),))
    assert [value for value, _ in task_completions(manager, '/stop ')] == ['/stop 1']
    assert [value for value, _ in task_completions(manager, '/resume ')] == ['/resume 2']
    assert len(task_completions(manager, '/status ')) == 3
    assert task_completions(manager, '/long-research Demo')[0][0] == "/long-research '/tasks/a b' "
    assert task_completions(manager, "/long-research '/tasks/a b' " ) is None
    assert task_completions(manager, "/long-research '/tasks/a b' wor") is None
    assert task_completions(manager, '/long-research workers=') is None
    assert task_completions(manager, '/long-research new unknown task') == []


def test_portable_package_manifest_passes_existing_preflight(tmp_path):
    from scienceflow.research.onboarding import run_preflight, write_onboarding_files

    result = export_success(experiment(tmp_path))
    folder = Path(result['path'])
    session = LongResearchSession.start(workspace=tmp_path, constraints=(
        f'{shlex.quote(str(folder))} gpu=cpu workers=1 cpu=0 duration=10m'))
    files = write_onboarding_files(session.draft, tmp_path)
    report = run_preflight(files.manifest_path)
    assert report.ok, report


def test_supervisor_registers_completed_task_without_tui(tmp_path, monkeypatch):
    import os

    from scienceflow.runtime.parallel.control import worker
    from scienceflow.runtime.parallel.control.registry import read_record, write_record

    record = experiment(tmp_path)
    record.update(launch_token='test-token', pid=os.getpid(), manifest='synthetic')
    directory = tmp_path / 'registry'
    write_record(directory, record)
    monkeypatch.setattr(worker, 'run_dir', lambda run_id: directory)

    async def synthetic(manifest):
        return SimpleNamespace(text='done', results=[])

    async def run():
        monkeypatch.setattr(asyncio.get_running_loop(), 'add_signal_handler', lambda *args: None)
        await worker.supervise(record['run_id'], 'test-token')

    monkeypatch.setattr(worker, 'run_manifest', synthetic)
    asyncio.run(run())
    saved = read_record(directory)
    assert saved['status'] == 'completed'
    assert saved['task_registration']['status'] == 'registered'
    assert saved['final_reports'][0]['status'] == 'ready'
    task_root = Path(record['task_roots'][0])
    assert (task_root / 'report.md').is_file()
    assert (task_root / 'report.html').is_file()
    assert (task_root / 'report.pdf').is_file()


def test_supervisor_generates_report_for_failed_task(tmp_path, monkeypatch):
    import os

    from scienceflow.runtime.parallel.control import worker
    from scienceflow.runtime.parallel.control.registry import read_record, write_record

    record = experiment(tmp_path)
    record.update(launch_token='test-token', pid=os.getpid(), manifest='synthetic')
    directory = tmp_path / 'registry'
    write_record(directory, record)
    monkeypatch.setattr(worker, 'run_dir', lambda run_id: directory)

    async def synthetic(manifest):
        return SimpleNamespace(
            text='failed',
            results=[SimpleNamespace(status='failed')],
        )

    async def run():
        monkeypatch.setattr(
            asyncio.get_running_loop(),
            'add_signal_handler',
            lambda *args: None,
        )
        await worker.supervise(record['run_id'], 'test-token')

    monkeypatch.setattr(worker, 'run_manifest', synthetic)
    asyncio.run(run())
    saved = read_record(directory)
    assert saved['status'] == 'failed'
    assert 'task_registration' not in saved
    assert saved['final_reports'][0]['task_status'] == 'failed'
    assert Path(record['task_roots'][0], 'report.md').is_file()


def test_export_rejects_credential_fields_in_serialized_config():
    import pytest

    from scienceflow.research.onboarding.library.portable import clean_text

    for text in ('api_keys: [private]', '{"api_keys": ["private"]}', 'API_KEY="private"'):
        with pytest.raises(ValueError, match='Credential'):
            clean_text(text)
