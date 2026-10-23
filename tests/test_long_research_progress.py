import json
from types import SimpleNamespace

from scienceflow.interfaces.ui import long_research_progress as progress


def test_time_budget_bar_reports_worker_state_and_cleanup(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    draft = SimpleNamespace(workspace_base=str(tmp_path), run_id="run", exp_id="task", wall_clock_sec=1200)
    monitor = progress.LongResearchProgress(draft)
    status, detail = monitor.snapshot()
    assert "0%" in status
    assert "Preparing" in detail
    worker = tmp_path / "run/task/workers/w00/logs/lhr_state.json"
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps({"progress_last": {"phase": "artifact_update"}, "stage_count": 2}))
    clock[0] = 700
    status, detail = monitor.snapshot()
    assert "50%" in status
    assert "remaining 10m 00s" in status
    assert "w00: updating candidate artifacts" in detail
    assert "Stages 2" in monitor.rows[0]
    clock[0] = 1400
    status, detail = monitor.snapshot()
    assert "100%" in status
    assert "remaining 0s" in status
    assert "waiting for evaluation and cleanup" in detail
    worker.write_text("{")
    assert "waiting for progress report" in monitor.snapshot()[1]


def test_evaluator_events_are_incremental_and_deduplicated(tmp_path):
    from scienceflow.interfaces.ui.research.research_events import ResearchEvents

    path = tmp_path / 'events.jsonl'
    event = {'event': 'evaluator_metric_event', 'worker_id': 'W00', 'payload': {
        'candidate_id': 'S01', 'artifact_sha': 'abc', 'evaluator_status': 'ok',
        'selection_eligible': True, 'metric_value': 2.4}}
    line = json.dumps(event) + '\n'
    path.write_text(line + line + '{')
    reader = ResearchEvents()
    reader.read(path)
    reader.read(path)
    assert len(reader.candidates) == 1
    assert len(reader.new_events) == 1
    assert next(iter(reader.candidates.values())) == (True, 2.4)


def test_worker_stage_count_uses_distinct_capture_events_when_state_lags(tmp_path):
    draft = SimpleNamespace(
        workspace_base=str(tmp_path),
        run_id="run",
        exp_id="task",
        wall_clock_sec=100,
        workers=2,
    )
    monitor = progress.LongResearchProgress(draft)
    logs = monitor.root / "workers/w00/logs"
    logs.mkdir(parents=True)
    (logs / "lhr_state.json").write_text(json.dumps({"stage_count": 1}))
    (logs / "lhr_events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "event": "stage_captured",
                    "worker_id": "W00",
                    "payload": {"stage_id": stage_id, "node_uid": f"W00:L01:{stage_id}"},
                }
            )
            + "\n"
            for stage_id in ("S01", "S02", "S02")
        )
    )

    monitor.snapshot()

    assert "Stages 2" in monitor.rows[0]
    assert "Stages 0" in monitor.rows[1]


def test_worker_usage_is_aggregated_once_and_kept_separate_from_chat(tmp_path):
    from scienceflow.interfaces.ui.research.research_usage import ResearchUsage

    usage = ResearchUsage()
    for worker, incoming, cached in [('W00', 100, 100), ('W01', 900, 0)]:
        path = tmp_path / (worker + '.jsonl')
        row = {'phase': 'completed', 'worker_id': worker, 'session_id': 'session',
               'run_id': 'run', 'turn_id': 1, 'call_seq': 1, 'attempt': 1, 'model': 'model',
               'tokens_input': incoming, 'tokens_output': 10, 'tokens_cached': cached}
        path.write_text(json.dumps(row) + '\n')
        usage.read(path)
        usage.read(path)
    assert 'in 1.0K / out 0.0K' in usage.summary()
    assert 'Cache 10%' in usage.summary()
    assert 'Cost —' in usage.summary()



def test_hundred_workers_have_counts_bounded_details_and_cell_timer(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    draft = SimpleNamespace(workspace_base=str(tmp_path), run_id="run", exp_id="task", wall_clock_sec=100, workers=100)
    monitor = progress.LongResearchProgress(draft)
    for index in range(100):
        path = monitor.root / 'workers' / f'w{index:02d}' / 'logs/lhr_state.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'status': 'failed' if index == 3 else 'running', 'progress_last': {'phase': 'running'}}))
    clock[0] = 150
    status, detail = monitor.snapshot()
    assert status.count('■') == 5
    assert status.count('□') == 5
    assert '50%' in status
    assert len(monitor.rows) == 100
    assert 'Workers 100' in monitor.summary
    assert 'issues 1' in monitor.summary
    assert any('needs review' in row for row in monitor.rows)
    assert 'running 99' in monitor.summary
    assert len(detail) < 1000
    assert '+95 workers' in detail


def test_control_event_counts_and_summary_order(tmp_path):
    draft = SimpleNamespace(workspace_base=str(tmp_path), run_id="run", exp_id="task", wall_clock_sec=100)
    monitor = progress.LongResearchProgress(draft)
    path = monitor.root / 'workers/w00/logs/lhr_state.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    names = ['estra_decision', 'estra_stage_switched', 'estra_keep_current_compacted',
             'resource_job_started', 'resource_gpu_queue_wait_started', 'resource_gpu_lease_acquired']
    path.with_name('lhr_events.jsonl').write_text(''.join(
        json.dumps({'event': name, 'payload': {}}) + '\n' for name in names))
    monitor.snapshot()
    monitor.snapshot()
    lines = monitor.summary.splitlines()
    assert len(lines) == 6
    assert lines[0].startswith('Time ')
    assert 'Workers 1' in lines[1]
    assert 'Δfirst' not in lines[2]
    assert 'Decisions 1 · Switches 1 · Compactions 1 · Invalid 0' in lines[3]
    assert 'EEC · Events 3 · Started 1 · Finished 0 · Waits 1 · Acquired 1 · Released 0 · Interventions 0 · Kill 0' in lines[4]

    assert lines[5].startswith('Research usage')


def test_live_best_uses_evaluations_before_stage_commit_and_survives_failure(tmp_path):
    draft = SimpleNamespace(workspace_base=str(tmp_path), run_id='run', exp_id='task',
                            wall_clock_sec=100, metric_name='radii_sum', lower_is_better=False)
    monitor = progress.LongResearchProgress(draft)
    logs = monitor.root / 'workers/w00/logs'
    logs.mkdir(parents=True)
    state = logs / 'lhr_state.json'
    state.write_text(json.dumps({'global_best': {}, 'stage_count': 0, 'status': 'running'}))
    events = logs / 'lhr_events.jsonl'

    def evaluation(candidate, value, **changes):
        payload = {
            'candidate_id': candidate,
            'artifact_sha': candidate,
            'evaluator_status': 'ok',
            'validation_ok': True,
            'selection_eligible': True,
            'metric_name': 'radii_sum',
            'lower_is_better': False,
            'metric_value': value,
        }
        payload.update(changes)
        with events.open('a') as stream:
            stream.write(
                json.dumps(
                    {
                        'event': 'evaluator_metric_event',
                        'worker_id': 'W00',
                        'payload': payload,
                    }
                )
                + '\n'
            )

    evaluation('first', 2.4)
    monitor.snapshot()
    assert monitor.best_text == '2.4'
    evaluation('better', 2.63)
    evaluation('invalid', 9, validation_ok=False)
    evaluation('ineligible', 10, selection_eligible=False)
    evaluation('different', 11, metric_name='another_metric')
    evaluation('direction', 12, lower_is_better=True)
    evaluation('infinite', float('inf'))
    state.write_text(json.dumps({'global_best': {'metric_value': 2.4, 'validation_ok': True}, 'status': 'failed'}))
    monitor.snapshot()
    assert monitor.best_text == '2.63'
    assert 'best 2.63' in monitor.rows[0]
    monitor.active = False
    monitor.snapshot()
    assert monitor.best_text == '2.63'
    assert json.loads(state.read_text())['global_best']['metric_value'] == 2.4


def test_live_best_minimizes_and_reads_events_without_state_file(tmp_path):
    draft = SimpleNamespace(workspace_base=str(tmp_path), run_id='run', exp_id='task',
                            wall_clock_sec=100, metric_name='loss', lower_is_better=True)
    monitor = progress.LongResearchProgress(draft)
    for worker, value in [('w00', 0.7), ('w01', 0.3)]:
        logs = monitor.root / 'workers' / worker / 'logs'
        logs.mkdir(parents=True)
        (logs / 'lhr_events.jsonl').write_text(json.dumps({
            'event': 'evaluator_metric_event', 'worker_id': worker.upper(), 'payload': {
                'candidate_id': 'first', 'evaluator_status': 'ok', 'selection_eligible': True,
                'validation_ok': True, 'metric_name': 'loss', 'lower_is_better': True,
                'metric_value': value}}) + '\n')
    monitor.snapshot()
    assert monitor.best_text == '0.3'
    assert 'best 0.7' in monitor.rows[0]
    assert 'best 0.3' in monitor.rows[1]


def test_deferred_direction_is_filled_from_runtime_evaluator_events(tmp_path):
    draft = SimpleNamespace(
        workspace_base=str(tmp_path),
        run_id='run',
        exp_id='task',
        wall_clock_sec=100,
        metric_name='rmse',
        lower_is_better=None,
    )
    monitor = progress.LongResearchProgress(draft)
    logs = monitor.root / 'workers/w00/logs'
    logs.mkdir(parents=True)
    (logs / 'lhr_events.jsonl').write_text('\n'.join([
        json.dumps({
            'event': 'evaluator_metric_event',
            'worker_id': 'W00',
            'payload': {
                'candidate_id': candidate,
                'evaluator_status': 'ok',
                'selection_eligible': True,
                'validation_ok': True,
                'metric_name': 'rmse',
                'lower_is_better': True,
                'metric_value': value,
            },
        })
        for candidate, value in [('first', 0.7), ('better', 0.3)]
    ]) + '\n')

    monitor.snapshot()

    assert monitor.lower_is_better is True
    assert monitor.best_text == '0.3'
    assert 'Result  rmse ↓' in monitor.summary
