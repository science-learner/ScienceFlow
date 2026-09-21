from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from scienceflow.interfaces.ui.long_research import LongResearchInteraction
from scienceflow.research.onboarding import LongResearchSession, OnboardingState
from scienceflow.research.onboarding.support.preparation import parse_preparation
from scienceflow.research.onboarding.support.preparation.agent import preparation_prompt
from scienceflow.research.onboarding.support.preparation.registered import (
    discover_registered_input,
    registered_missing_prompt,
    registered_preparation_fields,
)
from scienceflow.research.onboarding.support.preparation.validation import (
    validate_evaluator,
)
from scienceflow.runtime.parallel.config.manifest import _resolve_parallel_task_text


def test_registered_description_survives_tui_manifest_and_keeps_extra_request(tmp_path):
    session = LongResearchSession.start(workspace=tmp_path, constraints='leaf-classification')
    canonical = _resolve_parallel_task_text('leaf-classification', None).strip()
    assert canonical in session.draft.task_text
    assert len(session.draft.task_text) > 1000
    extra = LongResearchSession.start(workspace=tmp_path, constraints='leaf-classification use a small model')
    assert canonical in extra.draft.task_text
    assert 'use a small model' in extra.draft.task_text


def test_registered_task_alias_and_natural_resource_options_are_deterministic(tmp_path):
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints='circle packing workers2，cpu8，20min',
        natural_language=True,
    )
    assert session.draft.registered_task
    assert session.draft.input_data_dir == 'none'

    fields = registered_preparation_fields(
        session.task_package,
        'circle packing workers2，cpu8，20min',
        workspace=tmp_path,
    )

    assert fields['workers'] == 2
    assert fields['wall_clock_sec'] == 1200
    assert fields['gpu_list'] == 'cpu'
    assert len(_cpu_ids(fields['cpu_list'])) == 8


@pytest.mark.parametrize('answer', ('20min。', 'duration=20min。'))
def test_registered_duration_accepts_sentence_punctuation(tmp_path, answer):
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints='circle-packing',
    )

    fields = registered_preparation_fields(
        session.task_package,
        answer,
        workspace=tmp_path,
    )

    assert fields['wall_clock_sec'] == 1200


@pytest.mark.parametrize('answer', ('duration=2', 'duration=9min。'))
def test_registered_duration_rejects_values_below_ten_minutes(tmp_path, answer):
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints='circle-packing',
    )

    with pytest.raises(ValueError, match='at least 10 minutes'):
        registered_preparation_fields(
            session.task_package,
            answer,
            workspace=tmp_path,
        )


def test_registered_runtime_prompt_does_not_attach_punctuation_to_example(tmp_path):
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints='circle-packing',
    )

    prompt = registered_missing_prompt(session.draft)

    assert prompt == (
        '请补充运行配置：workers=2 cpu=8 gpu=cpu duration=20min '
        '可在一行内填写'
    )


def test_registered_mlebench_input_discovery_uses_configured_root(tmp_path, monkeypatch):
    root = tmp_path / 'mlebench'
    data = root / 'nomad2018-predict-transparent-conductors/prepared/dataset_split/Deep'
    data.mkdir(parents=True)
    monkeypatch.setenv('MLEBENCH_DATA_ROOT_DIR', str(root))
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints='nomad2018-predict-transparent-conductors',
    )

    assert discover_registered_input(session.task_package, workspace=tmp_path) == data


def test_registered_explicit_dataset_is_remembered_per_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv('MLEBENCH_DATA_ROOT_DIR', raising=False)
    data = tmp_path / 'external-datasets' / 'nomad-deep'
    data.mkdir(parents=True)
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints='nomad2018-predict-transparent-conductors',
    )

    fields = registered_preparation_fields(
        session.task_package,
        f'data={data} workers=2 cpu=8 duration=20min',
        workspace=tmp_path,
    )

    assert fields['input_data_dir'] == str(data)
    registry = json.loads((tmp_path / '.scienceflow/datasets.json').read_text())
    assert registry['tasks'][session.draft.exp_id] == 'external-datasets/nomad-deep'
    assert discover_registered_input(session.task_package, workspace=tmp_path) == data

    data.rmdir()
    assert discover_registered_input(session.task_package, workspace=tmp_path) is None


def test_registered_cpu_count_skips_other_task_reservations(tmp_path, monkeypatch):
    session = LongResearchSession.start(workspace=tmp_path, constraints='circle-packing')
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: set(range(12)))

    fields = registered_preparation_fields(
        session.task_package,
        'workers=2 cpu=8 duration=20min',
        workspace=tmp_path,
        reserved_cpu_ids=frozenset(range(4)),
    )

    assert fields['cpu_list'] == '4-11'
    with pytest.raises(ValueError, match='already reserved'):
        registered_preparation_fields(
            session.task_package,
            'workers=2 cpu=0-7 duration=20min',
            workspace=tmp_path,
            reserved_cpu_ids=frozenset(range(4)),
        )


def _cpu_ids(value):
    ids = []
    for part in str(value).split(','):
        bounds = [int(item) for item in part.split('-', 1)]
        ids.extend(range(bounds[0], bounds[-1] + 1))
    return ids


def test_registered_preparation_uses_loaded_facts_without_source_path(tmp_path):
    session = LongResearchSession.start(workspace=tmp_path, constraints='circle-packing')
    source = session.draft.description_sources[0]
    session.draft.model_config_path = '/home/example/.config/scienceflow/models.json'

    prompt = preparation_prompt(session.draft, '', '', tmp_path)

    assert source not in prompt
    assert str(Path(source).parent) not in prompt
    assert 'Circle packing opt_solver task' in prompt
    assert 'Maximize the sum of all radii' in prompt
    assert session.draft.model_config_path not in prompt
    assert 'Do not reread package source files' in prompt


def test_directory_discovery_reads_description_without_executing_it(tmp_path):
    root = tmp_path / 'new task'
    root.mkdir()
    (root / 'instances').mkdir()
    (root / 'description.md').write_text('Minimize schedule makespan. Never allow late jobs.')
    (root / 'evaluate.py').write_text('raise RuntimeError("must not execute during discovery")')
    session = LongResearchSession.start(workspace=tmp_path, constraints=shlex.quote(str(root)))
    assert 'Minimize schedule' in session.draft.task_text
    assert session.draft.input_data_dir == str(root / 'instances')
    assert 'evaluate.py' in session.draft.discovered_files
    assert session.draft.exploratory
    assert session.next_question().field == 'metric_name'
    with pytest.raises(ValueError, match='does not exist'):
        LongResearchSession.start(workspace=tmp_path, constraints='./missing')


def test_preparation_fields_preserve_registered_contract_and_validate_paths(tmp_path):
    session = LongResearchSession.start(workspace=tmp_path, constraints='leaf-classification')
    original = session.draft.task_text
    fields, _, _, _ = parse_preparation(json.dumps({'fields': {
        'task_text': 'wrong', 'metric_name': 'wrong', 'workers': 2, 'wall_clock_sec': 1200,
        'input_data_dir': str(tmp_path)}}), session.draft, tmp_path)
    session.revise(fields)
    assert session.draft.task_text == original
    assert session.draft.workers == 2
    with pytest.raises(ValueError):
        session.revise({'workers': 3, 'artifact_path': '../bad'})
    assert session.draft.workers == 2
    with pytest.raises(ValueError, match='does not exist'):
        parse_preparation(json.dumps({'fields': {'input_data_dir': str(tmp_path/'missing')}}), session.draft, tmp_path)


def evaluator_samples(tmp_path):
    script = tmp_path / 'evaluate.py'
    script.write_text('import json,sys\nv=json.load(open(sys.argv[1]))\n'
                      'if v["late"]: sys.exit(2)\nprint(json.dumps({"metric":v["makespan"]}))\n')
    valid, invalid = tmp_path / 'valid.json', tmp_path / 'invalid.json'
    valid.write_text('{"late":false,"makespan":10}')
    invalid.write_text('{"late":true,"makespan":9}')
    return f'{shlex.quote(sys.executable)} {shlex.quote(str(script))} {{artifact_abs_path}}', {
        'valid_artifact': str(valid), 'invalid_artifact': str(invalid)}


@pytest.mark.asyncio
async def test_evaluator_validation_exercises_domain_constraint_from_other_cwd(tmp_path):
    command, samples = evaluator_samples(tmp_path)
    result = await validate_evaluator(command, samples, tmp_path)
    assert result['metric'] == 10
    assert result['invalid_rejected']
    (tmp_path / 'invalid.json').write_text('{"late":false,"makespan":9}')
    with pytest.raises(ValueError, match='validation failed'):
        await validate_evaluator(command, samples, tmp_path)


class PreparationHost:
    busy = False

    def __init__(self, responses):
        self.responses = iter(responses)
        self.notices = []
        self.prompts = []

    def conversation_snapshot(self):
        return ()

    def set_host_status(self, value):
        self.status = value

    async def notice(self, value):
        self.notices.append(value)

    async def run_host_task(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(next(self.responses))


@pytest.mark.asyncio
async def test_new_task_clarification_then_validation_then_preflight(tmp_path, monkeypatch):
    (tmp_path / 'description.md').write_text('Optimize schedule. Clarify whether late jobs are invalid.')
    command, samples = evaluator_samples(tmp_path)
    host = PreparationHost([
        {'fields': {'metric_name': 'makespan', 'lower_is_better': True}, 'question': '迟到是否直接判为无效？'},
        {'fields': {'input_data_dir': str(tmp_path), 'artifact_path': 'schedule.json',
                    'artifact_command': command, 'gpu_list': 'cpu', 'cpu_list': '0-3',
                    'workers': 2, 'wall_clock_sec': 1200}, 'validation': samples,
         'summary': 'Read description.md; checked lateness constraints.'},
    ])
    interaction = LongResearchInteraction(tmp_path)
    calls = []

    async def preflight(host):
        calls.append(interaction.session.draft)

    monkeypatch.setattr(interaction, '_preflight', preflight)
    await interaction.try_handle('/long-research .', host)
    await interaction._preparation.task
    assert not calls
    assert interaction.session.state == OnboardingState.ASKING
    assert host.notices[-1] == '迟到是否直接判为无效？'
    await interaction.try_handle('run', host)
    assert not calls
    await interaction.try_handle('是的。20分钟，2个worker，用CPU', host)
    await interaction._preparation.task
    assert len(calls) == 1
    assert calls[0].evaluator['metric']['json_path'] == 'metric'
    assert calls[0].workers == 2
    assert '20分钟' in host.prompts[-1]
    assert (tmp_path / '.scienceflow/preparation/validation.json').exists()
    await interaction.aclose()


@pytest.mark.asyncio
async def test_cancel_preparation_does_not_launch_or_restore_session(tmp_path):
    started, stopped = asyncio.Event(), asyncio.Event()
    host = PreparationHost([])

    async def slow(prompt):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    host.run_host_task = slow
    interaction = LongResearchInteraction(tmp_path)
    await interaction.try_handle('/long-research new optimization task', host)
    await started.wait()
    await interaction.try_handle('/cancel', host)
    assert stopped.is_set()
    assert interaction.session is None
    assert not interaction.running


@pytest.mark.asyncio
async def test_failed_preparation_edit_invalidates_previous_preflight(tmp_path, monkeypatch):
    host = PreparationHost([{'fields': {'workers': 0}}])
    interaction = LongResearchInteraction(tmp_path)
    interaction.session = LongResearchSession.start(workspace=tmp_path, constraints='leaf-classification')
    interaction.session.state = OnboardingState.CONFIRM
    interaction.files = object()
    interaction.preflight = object()
    await interaction.try_handle('Change workers', host)
    await interaction._preparation.task
    assert interaction.files is None
    assert interaction.preflight is None
    assert interaction.session.state == OnboardingState.ASKING
    await interaction.try_handle('run', host)
    assert not interaction.running
    await interaction.aclose()


@pytest.mark.asyncio
async def test_yes_answers_preparation_question_without_confirming_run(tmp_path):
    host = PreparationHost([{'fields': {}, 'question': 'How long should it run?'}])
    interaction = LongResearchInteraction(tmp_path)
    interaction.session = LongResearchSession.start(workspace=tmp_path, constraints='a new task')
    interaction._preparation.question = 'Should late jobs be invalid?'
    await interaction.try_handle('yes', host)
    await interaction._preparation.task
    assert 'yes' in host.prompts[-1]
    assert host.notices[-1] == 'How long should it run?'
    assert not interaction.session.confirmed
    await interaction.aclose()


def test_plain_text_refusal_reports_contract_error_without_mutating_draft(tmp_path):
    session = LongResearchSession.start(workspace=tmp_path, constraints='circle-packing')
    original = session.draft.workers
    with pytest.raises(ValueError, match='required preparation JSON'):
        parse_preparation('抱歉，我只能协助处理 OpenAI 相关的任务。', session.draft, tmp_path)
    assert session.draft.workers == original


def test_no_dataset_answer_survives_preparation_and_materializes_empty_input(tmp_path):
    from scienceflow.research.onboarding.manifest import write_onboarding_files

    session = LongResearchSession.start(workspace=tmp_path, constraints='circle-packing')
    fields, _, _, _ = parse_preparation(
        json.dumps({'fields': {'input_data_dir': 'none'}}), session.draft, tmp_path)
    session.revise(fields)
    assert session.draft.input_data_dir == 'none'
    assert session.next_question().field != 'input_data_dir'
    session.revise({'gpu_list': 'cpu', 'cpu_list': '0-7', 'workers': 2, 'wall_clock_sec': 1200})
    files = write_onboarding_files(session.draft, tmp_path)
    from pathlib import Path

    import yaml

    manifest = yaml.safe_load(files.manifest_path.read_text())
    data = Path(manifest['tasks'][0]['input_data_dir'])
    assert data.is_relative_to(tmp_path / '.scienceflow/empty_inputs')
    assert data.is_dir() and not list(data.iterdir())
    assert session.draft.input_data_dir == 'none'
    initial = LongResearchSession.start(workspace=tmp_path, constraints='task=circle-packing data=none')
    assert initial.draft.input_data_dir == 'none'


@pytest.mark.asyncio
async def test_registered_tui_requests_missing_resources_without_agent_retry(tmp_path):
    host = PreparationHost([])
    interaction = LongResearchInteraction(tmp_path)
    request = 'circle packing workers=两个，cpu8，20min task="unfinished'
    await interaction.try_handle('/long-research ' + request, host)
    assert interaction.session is not None
    assert interaction.session.draft.workers is None
    await interaction._preparation.task
    assert not host.prompts
    assert any('workers=2' in text for text in host.notices)
    await interaction.aclose()


@pytest.mark.asyncio
async def test_invalid_historical_dataset_is_rejected_then_repaired_by_model(tmp_path):
    old_path = str(tmp_path / 'old-run/workers/w00/workspace/dataset')
    host = PreparationHost([
        {'fields': {'input_data_dir': old_path}},
        {'fields': {'input_data_dir': 'none'}, 'question': '运行多久？'},
    ])
    interaction = LongResearchInteraction(tmp_path)
    interaction.session = LongResearchSession.start(workspace=tmp_path, constraints='a new task')
    await interaction._preparation.begin(host, '不需要dataset')
    await interaction._preparation.task
    assert len(host.prompts) == 2
    assert 'Dataset directory does not exist' in host.prompts[1]
    assert old_path in host.prompts[1]
    assert '不需要dataset' in host.prompts[1]
    assert interaction.session.draft.input_data_dir == 'none'
    assert not (tmp_path / 'old-run').exists()
    assert not any('Preparation could not complete' in text for text in host.notices)
    await interaction.aclose()


@pytest.mark.asyncio
async def test_repeated_invalid_dataset_stops_retry_and_keeps_feedback_for_next_answer(tmp_path):
    old_path = str(tmp_path / 'missing')
    host = PreparationHost([
        {'fields': {'input_data_dir': old_path}},
        {'fields': {'input_data_dir': old_path}},
        {'fields': {'input_data_dir': 'none'}, 'question': '运行多久？'},
    ])
    interaction = LongResearchInteraction(tmp_path)
    interaction.session = LongResearchSession.start(workspace=tmp_path, constraints='a new task')
    await interaction._preparation.begin(host, '准备任务')
    await interaction._preparation.task
    assert len(host.prompts) == 2
    assert interaction.session.draft.input_data_dir == ''
    await interaction._preparation.begin(host, '不需要外部数据')
    await interaction._preparation.task
    assert 'Dataset directory does not exist' in host.prompts[2]
    assert '不需要外部数据' in host.prompts[2]
    assert interaction.session.draft.input_data_dir == 'none'
    await interaction.aclose()


@pytest.mark.asyncio
async def test_registered_unknown_direction_answer_reaches_preflight(tmp_path, monkeypatch):
    data_root = tmp_path / 'mlebench'
    (data_root / 'nomad2018-predict-transparent-conductors/prepared/dataset_split/Deep').mkdir(parents=True)
    monkeypatch.setenv('MLEBENCH_DATA_ROOT_DIR', str(data_root))
    host = PreparationHost([])
    interaction = LongResearchInteraction(tmp_path)
    await interaction.try_handle(
        '/long-research nomad2018-predict-transparent-conductors workers=2 cpu=8 duration=20min',
        host,
    )
    await interaction._preparation.task
    assert interaction.session.draft.lower_is_better is None
    assert interaction.session.state == OnboardingState.ASKING
    assert 'Default · Code' in host.notices[-1]
    await interaction.try_handle('defaults', host)
    assert interaction.session.state == OnboardingState.CONFIRM, interaction.preflight
    assert interaction.preflight.ok
    assert not host.prompts
    await interaction.aclose()


@pytest.mark.asyncio
async def test_registered_followup_resources_skip_host_agent(tmp_path, monkeypatch):
    data_root = tmp_path / 'mlebench'
    data = data_root / 'nomad2018-predict-transparent-conductors/prepared/dataset_split/Deep'
    data.mkdir(parents=True)
    monkeypatch.setenv('MLEBENCH_DATA_ROOT_DIR', str(data_root))
    host = PreparationHost([])
    interaction = LongResearchInteraction(tmp_path)

    await interaction.try_handle('/long-research nomad2018-predict-transparent-conductors', host)
    await interaction._preparation.task
    assert interaction.session.state == OnboardingState.ASKING
    assert 'workers=2' in host.notices[-1]

    await interaction.try_handle('workers=2，cpu=16，duration=1h', host)
    await interaction._preparation.task
    assert interaction.session.state == OnboardingState.ASKING
    assert 'Default · Code' in host.notices[-1]
    await interaction.try_handle('defaults', host)
    assert interaction.session.state == OnboardingState.CONFIRM
    assert interaction.session.draft.input_data_dir == str(data)
    assert interaction.session.draft.workers == 2
    assert interaction.session.draft.wall_clock_sec == 3600
    assert len(_cpu_ids(interaction.session.draft.cpu_list)) == 16
    assert not host.prompts
    await interaction.aclose()


def test_registered_direction_keeps_explicit_input_and_protects_known_value(tmp_path):
    session = LongResearchSession.start(workspace=tmp_path, constraints=(
        'nomad2018-predict-transparent-conductors direction=minimized'))
    assert session.draft.lower_is_better is True
    assert session.draft.evaluator['metric']['lower_is_better'] is True
    fields, *_ = parse_preparation(json.dumps({'fields': {'lower_is_better': False}}), session.draft, tmp_path)
    assert 'lower_is_better' not in fields
    known = LongResearchSession.start(workspace=tmp_path, constraints='circle-packing direction=minimized')
    assert known.draft.lower_is_better is False
