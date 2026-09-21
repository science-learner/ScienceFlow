"""Long Horizon Repl contracts: resume."""

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403
from scienceflow.research.solver.lnr.orchestration.coordinator.resource import advisory, prompt_projection
from scienceflow.research.solver.lnr.orchestration.coordinator.run import coordination, event_projection


def test_lhr_agent_metric_only_snapshot_without_submission(tmp_path: Path) -> None:
    ws = tmp_path / "metric_only_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)
    interpretation_cb = MagicMock()
    object.__setattr__(agent, "_lnr_metric_output_interpretation_callback", interpretation_cb)

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(output="[exit=0, 1.0s]\nok\nFinal Validation Score: 0.42\n", error=""),
        )
    )

    assert getattr(agent, "_lnr_snapshot_ok", False) is True
    assert getattr(agent, "_lnr_snapshot_reason", "") == "metric_only_tail_snapshot"
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.42)
    assert data["metric_name"] == "Final Validation Score"
    assert data["bash_cmd"] == "python3 solution.py"
    assert data["submission_status"] == "missing_submission"
    assert "submission_sha" not in data
    assert data["solution_path"] == "solution.py"
    assert data["solution_sha"]
    interpretation_cb.assert_not_called()


def test_lhr_metric_only_snapshot_does_not_attach_stale_submission(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "metric_only_with_stale_submission"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "submission.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)
    object.__setattr__(agent, "_lnr_candidate_artifact_changed_after_tool", False)

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(
                output="[exit=0, 1.0s]\nFinal Validation Score: 0.31\n",
                error="",
            ),
        )
    )

    data = json.loads(
        (ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8")
    )
    assert data["metric_value"] == pytest.approx(0.31)
    assert data["submission_status"] == "missing_submission"
    assert data["stage_signal_kind"] == "metric_only"
    assert data["candidate_artifact_attached"] is False
    assert "submission_sha" not in data

def test_lhr_agent_metric_only_snapshot_accepts_inline_python_without_solution(tmp_path: Path) -> None:
    ws = tmp_path / "inline_metric_ws"
    ws.mkdir()
    agent = _make_metric_snapshot_agent(ws)
    cmd = "python3 <<'PY'\nprint('Final Validation Score: 0.33')\nPY"

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": cmd},
            ToolResult(output="[exit=0, 0.5s]\nFinal Validation Score: 0.33\n", error=""),
        )
    )

    assert getattr(agent, "_lnr_snapshot_ok", False) is True
    assert getattr(agent, "_lnr_snapshot_reason", "") == "metric_only_tail_snapshot"
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.33)
    assert data["submission_status"] == "missing_submission"
    assert data["execution_mode"] == "inline_bash"
    assert data["solution_path"] == ""
    assert "solution_sha" not in data

def test_lhr_agent_submission_snapshot_waits_for_unified_evaluator(tmp_path: Path) -> None:
    ws = tmp_path / "invalid_submission_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "submission.csv").write_text("bad\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)

    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(output="[exit=0, 2.0s]\nFinal Validation Score: 0.51\n", error=""),
        )
    )

    assert getattr(agent, "_lnr_snapshot_ok", False) is True
    assert getattr(agent, "_lnr_snapshot_reason", "") == "candidate_tail_snapshot"
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.51)
    assert "submission_validation_ok" not in data
    assert data["submission_status"] == "pending_evaluator"
    assert data["submission_sha"]
    agent._inject_run_control_user_message.assert_not_called()

def test_lhr_resume_prompts_use_dynamic_base_stage() -> None:
    switched = build_estra_resume_prompt(
        target_stage="S02",
        next_stage="S04",
        ledger_filename=".run_results.md",
        base_stage="S02",
        compact_summary="S03 was abandoned",
    )
    kept = build_keep_current_compact_prompt(
        terminal_stage="S03",
        next_stage="S04",
        base_stage="S02",
        compact_summary="recent branch compacted",
    )

    assert "original S02 base stage" in switched
    assert "Historical exploration after the restored stage" in switched
    assert "abandoned-branch evidence" in switched
    assert "original S01 base stage" not in switched
    assert "original S02 base stage" in kept
    for prompt in (switched, kept):
        assert "Exploration state:" in prompt
        assert "completed evidence, not a stop signal" in prompt
        assert "not something to repeat" in prompt
        assert "Do not overwrite the best submission without a validated candidate" in prompt

def test_fullrun_tail_snapshot_records_actual_entrypoint_sha(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    source = ws / "train.py"
    source.write_text("print('train')\n")
    (ws / "submission.csv").write_text("id,target\n1,0\n")
    result = EnsureFullRunResult(
        executed=True,
        skipped=False,
        reason="ok",
        exit_code=0,
        wall_sec=1.0,
        metric_value=0.123,
        metric_name="Final Validation Score",
        lower_is_better=True,
        stdout="Final Validation Score: 0.123\n",
        stderr="",
    )
    write_fullrun_tail_snapshot(
        ws,
        result,
        bash_cmd="python3 train.py 2>&1",
        validation_ok=True,
        solution_path="train.py",
    )
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text())
    assert data["solution_path"] == "train.py"
    assert data["solution_sha"] == hashlib.sha256(source.read_bytes()).hexdigest()

def test_snapshot_store_preserves_logs_and_memory(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / ".logs").mkdir()
    (ws / ".logs" / "fullrun_tail_snapshot.json").write_text("{}\n")
    (ws / ".memory" / "ScienceAgent").mkdir(parents=True)
    (ws / ".memory" / "ScienceAgent" / "short_term.json").write_text("{}\n")
    (ws / "solution.py").write_text("print('ok')\n")
    store = SnapshotStore(
        root_dir=tmp_path,
        workspace_dir=ws,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    snap = store.capture(
        stage_id="S01",
        metric_value=0.1,
        metric_name="score",
        lower_is_better=True,
        memory_cut=1,
        source_event={"metric_value": 0.1},
    )
    assert (snap.snapshot_path / ".logs" / "fullrun_tail_snapshot.json").is_file()
    assert (snap.snapshot_path / ".memory" / "ScienceAgent" / "short_term.json").is_file()
    (ws / "solution.py").write_text("print('changed')\n")
    archive = store.restore(snap)
    assert archive.is_dir()
    archive_meta = json.loads((archive / "terminal_archive_meta.json").read_text())
    archive_manifest = json.loads(Path(archive_meta["workspace_snapshot_manifest_path"]).read_text())
    assert archive_manifest["entries"]["solution.py"]["size"] == len("print('changed')\n")
    assert not (archive / "solution.py").exists()
    assert (ws / "solution.py").read_text() == "print('ok')\n"
    assert (ws / ".logs").is_dir()
    assert (ws / ".memory").is_dir()

def test_snapshot_store_falls_back_to_workspace_root_when_stage_root_blocked(tmp_path: Path, monkeypatch) -> None:
    worker_root = tmp_path / "worker"
    ws = worker_root / "workspace"
    ws.mkdir(parents=True)
    (ws / "solution.py").write_text("print('ok')\n")
    blocked_root = worker_root / "snaps"
    original_mkdir = Path.mkdir

    def guarded_mkdir(self: Path, *args, **kwargs):
        if self == blocked_root:
            raise PermissionError("blocked snapshot root")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    store = SnapshotStore(
        root_dir=worker_root,
        workspace_dir=ws,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )

    snap = store.capture(
        stage_id="S01",
        metric_value=0.1,
        metric_name="score",
        lower_is_better=True,
        memory_cut=0,
        source_event={"metric_value": 0.1},
    )

    assert snap.snapshot_path.parent == ws / "snaps"
    assert (snap.snapshot_path / "solution.py").read_text() == "print('ok')\n"
    assert not (snap.snapshot_path / "snaps").exists()

def test_load_worker_candidates_includes_exact_archived_lineages(tmp_path: Path) -> None:
    worker_root = tmp_path / "workers" / "w00"
    worker_log = worker_root / "logs"
    global_log = tmp_path / "task_logs"
    global_log.mkdir(parents=True)

    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def make_snapshot(name: str, node_uid: str, metric: float) -> tuple[Path, str]:
        snapshot = worker_root / "snapshots" / name
        artifact = snapshot / "artifacts" / "submission.json"
        write(artifact, json.dumps({"node_uid": node_uid}) + "\n")
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        write(
            snapshot / "logs" / "lhr_snapshot_meta.json",
            json.dumps(
                {
                    "source_event": {
                        "node_uid": node_uid,
                        "lineage_id": node_uid.split(":")[1],
                        "metric_value": metric,
                        "artifact_path": "artifacts/submission.json",
                        "artifact_sha": artifact_sha,
                        "candidate_ready": True,
                        "validation_ok": True,
                        "selection_eligible": True,
                        "metric_validity": "high",
                    }
                }
            ),
        )
        return snapshot, artifact_sha

    abandoned, _ = make_snapshot("lineage-one", "W00:L01:S01", 0.9)
    active, _ = make_snapshot("lineage-two", "W00:L02:S01", 0.6)
    write(
        worker_log / "lhr_stage_map.json",
        json.dumps(
            {
                "stages": {
                    "S01": {
                        "stage_id": "S01",
                        "node_uid": "W00:L02:S01",
                        "lineage_id": "L02",
                        "snapshot_path": str(active),
                        "metric_value": 0.6,
                        "metric_name": "score",
                        "lower_is_better": False,
                    }
                },
                "archive": {
                    "W00:L01:S01": {
                        "stage_id": "S01",
                        "node_uid": "W00:L01:S01",
                        "lineage_id": "L01",
                        "snapshot_path": str(abandoned),
                        "metric_value": 0.9,
                        "metric_name": "score",
                        "lower_is_better": False,
                    },
                    "W00:L02:S01": {
                        "stage_id": "S01",
                        "node_uid": "W00:L02:S01",
                        "lineage_id": "L02",
                        "snapshot_path": str(active),
                        "metric_value": 0.6,
                        "metric_name": "score",
                        "lower_is_better": False,
                    },
                },
            }
        ),
    )
    owner = SimpleNamespace(
        log_dir=global_log,
        _worker_root=lambda _index: worker_root,
        _worker_log_dir=lambda _index: worker_log,
        _evaluator_candidate_artifact=lambda: "artifacts/submission.json",
        _read_json_file=coordination._read_json_file,
    )
    owner._candidate_from_stage = lambda **kwargs: coordination._candidate_from_stage(
        owner, **kwargs
    )

    active_candidates = coordination._load_worker_candidates(
        owner,
        {"worker_id": "W00", "worker_index": 0},
    )
    assert {candidate["candidate_id"] for candidate in active_candidates} == {
        "W00:L02:S01",
    }

    candidates = coordination._load_worker_candidates(
        owner,
        {"worker_id": "W00", "worker_index": 0},
        include_archived=True,
    )

    assert {candidate["candidate_id"] for candidate in candidates} == {
        "W00:L01:S01",
        "W00:L02:S01",
    }
    assert {candidate["lineage_id"] for candidate in candidates} == {"L01", "L02"}
    assert all(candidate["candidate_ready"] is True for candidate in candidates)

def test_metric_event_uses_task_direction_over_declared_snapshot_value(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.task_desc = """
    ## Target metric (evaluation)
    Kendall tau over all notebooks (higher better).
    """
    solver._task_metric_lower_is_better = False
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "score_validation.py").write_text("print('Final Validation Score: 0.742407')\n", encoding="utf-8")
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.742407,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": "Final Validation Score: 0.742407\n",
                "stderr_tail": "",
                "bash_cmd": "python3 score_validation.py",
                "solution_path": "score_validation.py",
                "validation_ok": True,
                "val_score_type": "holdout",
                "selection_eligible": True,
                "candidate_ready": True,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["lower_is_better"] is False
    assert event["declared_lower_is_better"] is True
    assert event["metric_direction_source"] == "task_description"
    assert event["metric_direction_conflict"] is True

def test_snapshot_store_does_not_create_roots_until_used(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = SnapshotStore(
        root_dir=tmp_path,
        workspace_dir=ws,
        snapshot_dirname="snaps",
        archive_dirname="archives",
        workspace_snapshot_enabled=False,
    )
    assert not (tmp_path / "snaps").exists()
    assert not (tmp_path / "archives").exists()
    snap = store.capture(
        stage_id="S01",
        metric_value=0.1,
        metric_name="score",
        lower_is_better=True,
        memory_cut=0,
        source_event={"metric_value": 0.1},
    )
    assert snap.snapshot_path.is_dir()
    assert (tmp_path / "snaps").is_dir()
    assert not (tmp_path / "archives").exists()

def test_lhr_node_uid_uses_worker_lineage_and_visible_stage() -> None:
    owner = SimpleNamespace(
        worker_id="W01",
        worker_index=1,
        current_lineage_id="L03",
        current_lineage_no=3,
    )
    owner._lineage_uid_prefix = lambda: event_projection._lineage_uid_prefix(owner)
    owner._worker_uid_prefix = lambda: event_projection._worker_uid_prefix(owner)

    assert event_projection._stage_node_uid(owner, "S02") == "W01:L03:S02"

def test_lhr_parallel_worker_snapshot_uses_global_best_per_worker(tmp_path: Path) -> None:
    owner = SimpleNamespace(
        lhr=SimpleNamespace(
            worker_peer_summary_enabled=True,
            worker_peer_summary_max_chars=2200,
        ),
        worker_id="W01",
        worker_index=1,
        worker_count=3,
        global_log_dir=tmp_path / "logs",
        _compact_peer_method=prompt_projection._compact_peer_method,
    )
    owner._worker_uid_prefix = lambda: event_projection._worker_uid_prefix(owner)
    owner.global_log_dir.mkdir()
    perf = owner.global_log_dir / "lhr_stage_performance.csv"
    with perf.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LHR_STAGE_PERFORMANCE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "row_order": "1",
                "candidate_id": "W00:L01:S01",
                "worker_id": "W00",
                "worker_index": "0",
                "worker_stage_order": "1",
                "stage_id": "S01",
                "lineage_id": "L01",
                "node_uid": "W00:L01:S01",
                "metric_value": "0.090000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "brief": "baseline features",
                "why": "first try",
                "elapsed_min": "2.0",
            }
        )
        writer.writerow(
            {
                "row_order": "2",
                "candidate_id": "W00:L01:S02",
                "worker_id": "W00",
                "worker_index": "0",
                "worker_stage_order": "2",
                "stage_id": "S02",
                "lineage_id": "L01",
                "node_uid": "W00:L01:S02",
                "metric_value": "0.070000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "brief": "lightgbm tuned",
                "why": "better",
                "elapsed_min": "12.3",
            }
        )
        writer.writerow(
            {
                "row_order": "3",
                "candidate_id": "W02:L01:S01",
                "worker_id": "W02",
                "worker_index": "2",
                "worker_stage_order": "1",
                "stage_id": "S01",
                "lineage_id": "L01",
                "node_uid": "W02:L01:S01",
                "metric_value": "0.010000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "0",
                "validation_issue": "train_validation_concat_before_reported_metric",
                "brief": "invalid leak",
                "why": "skip",
                "elapsed_min": "5.5",
            }
        )
        writer.writerow(
            {
                "row_order": "4",
                "candidate_id": "W02:L01:S02",
                "worker_id": "W02",
                "worker_index": "2",
                "worker_stage_order": "2",
                "stage_id": "S02",
                "lineage_id": "L01",
                "node_uid": "W02:L01:S02",
                "metric_value": "0.080000",
                "metric_name": "Final Validation Score",
                "lower_is_better": "1",
                "validation_ok": "1",
                "brief": "safe ridge baseline",
                "why": "valid fallback",
                "elapsed_min": "8.0",
            }
        )

    snapshot = advisory._parallel_worker_snapshot_for_prompt(owner)

    assert "There are 3 workers" in snapshot
    assert "- W00: best_metric=0.070000 (ok) at L01:S02; elapsed=12.3m; method=lightgbm tuned" in snapshot
    assert "- W01 (this worker): no metric-backed stage recorded yet." in snapshot
    assert "- W02: best_metric=0.010000 at L01:S01; elapsed=5.5m; status=suspicious" in snapshot
    assert "ok_best_metric=0.080000 at L01:S02; ok_elapsed=8.0m; ok_method=safe ridge baseline" in snapshot
    assert "train_validation_concat_before_reported_metric" in snapshot

def test_lhr_protected_eda_prefix_resume_uses_pre_commit_boundary(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        preserve_prefix_and_eda=True,
        protected_eda_warn_chars=50_000,
    )
    solver._s01_eda_prefix_end_index = None
    solver.stage_snapshots = {
        "S01": SimpleNamespace(
            memory_cut=2,
            source_event={"protected_eda_end_index": 1},
        )
    }

    class _MemoryContext:
        end_index = None

        def set_protected_raw_prefix(
            self,
            end_index: int,
            **_kwargs: object,
        ) -> None:
            self.end_index = end_index

    ctx = _MemoryContext()
    solver._restore_protected_eda_prefix_marker(SimpleNamespace(_memory_ctx=ctx))

    assert ctx.end_index == 1
    assert solver._s01_eda_prefix_end_index == 1

def test_lhr_process_resume_rehydrates_stage_snapshots(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True, exist_ok=True)
    solver.ledger_path.write_text(
        "### S01\nmetric: 0.5\nlower_is_better: false\nBRIEF: accepted\nWHY: evaluator accepted\n",
        encoding="utf-8",
    )
    store = SnapshotStore(
        root_dir=tmp_path,
        workspace_dir=solver.workspace_dir,
        snapshot_dirname="snapshots",
        archive_dirname="snapshots/archives",
        control_log_dir=solver.log_dir,
        metadata_dirname="logs",
        strict_layout=True,
        workspace_snapshot_enabled=False,
    )
    (solver.workspace_dir / "solution.py").write_text("print(1)\n", encoding="utf-8")
    snapshot = store.capture(
        stage_id="S01",
        metric_value=0.5,
        metric_name="score",
        lower_is_better=False,
        memory_cut=2,
        source_event={
            "solution_sha": "solution-one",
            "artifact_sha": "artifact-one",
            "gate_accepted": True,
            "_gate_evaluated": True,
            "lineage_id": "L03",
        },
        node_uid="W00:L03:S01",
        lineage_id="L03",
    )
    solver.snapshot_store = store
    solver.stage_snapshots = {"S01": snapshot}
    solver._write_stage_map()

    solver.stage_snapshots = {}
    solver.last_captured_solution_sha = ""
    solver.last_captured_run_signature = ""
    solver.current_lineage_no = 1
    solver.current_lineage_id = "L01"
    solver._load_existing_stage_snapshots()

    assert list(solver.stage_snapshots) == ["S01"]
    assert solver.stage_snapshots["S01"].snapshot_id == snapshot.snapshot_id
    assert solver.archived_stage_snapshots["W00:L03:S01"].snapshot_id == snapshot.snapshot_id
    assert solver.last_captured_solution_sha == "solution-one"
    assert solver.current_lineage_id == "L03"
    rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
    event = [row for row in rows if row["event"] == "stage_snapshots_rehydrated"][-1]
    assert event["payload"]["loaded_stages"] == ["S01"]

def test_lhr_failed_post_restore_step_rolls_back_terminal_archive(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        snapshot = StageSnapshot(
            "S01",
            "snapshot-one",
            tmp_path / "snapshots" / "S01",
            0.5,
            "score",
            False,
            1,
            {},
            node_uid="W00:L01:S01",
            lineage_id="L01",
        )
        solver.stage_snapshots = {"S01": snapshot}
        solver.archived_stage_snapshots = {"W00:L01:S01": snapshot}
        solver.pending_estra = {
            "action": "switch_stage",
            "target_stage": "S01",
            "target_node_uid": "W00:L01:S01",
        }
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        archive = tmp_path / "terminal-archive"
        restored_archives: list[Path] = []
        solver.snapshot_store = SimpleNamespace(
            restore=lambda _snap: archive,
            restore_terminal_archive=lambda path: restored_archives.append(Path(path)),
        )
        solver._close_lnr_interaction_loggers = lambda: None
        solver._prepare_dataset_symlink = lambda: (_ for _ in ()).throw(
            OSError("injected post-restore failure")
        )

        await solver._restore_pending_estra()

        assert restored_archives == [archive]
        assert solver.current_lineage_no == 1
        assert solver.current_lineage_id == "L01"
        assert solver.stage_snapshots == {"S01": snapshot}
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        failed = [row for row in rows if row["event"] == "estra_failed"][-1]
        assert failed["payload"]["rollback_attempted"] is True
        assert failed["payload"]["rollback_succeeded"] is True

    asyncio.run(_run())

def test_lhr_resume_rolls_back_ledger_when_stage_snapshot_is_missing(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    ledger_before = solver.ledger_path.read_text(encoding="utf-8")
    ledger_after = ledger_before + "### S03\nmetric: 0.09\nlower_is_better: true\nBRIEF: pending\nWHY: pending\n"
    metric_event = {"artifact_sha": "new-artifact", "gate_accepted": True}
    solver.snapshot_store = SimpleNamespace(discover=lambda: {})

    solver._prepare_stage_commit_transaction(
        stage_id="S03",
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        metric_event=metric_event,
    )
    solver.ledger_path.write_text(ledger_after, encoding="utf-8")
    solver.pending_stage_commit_transaction = None  # simulate process restart
    solver._recover_stage_commit_transactions()

    assert solver.ledger_path.read_text(encoding="utf-8") == ledger_before
    manifest = json.loads((solver.log_dir / "stage_transactions" / "L01-S03.json").read_text())
    assert manifest["status"] == "rolled_back"
    assert manifest["rollback_reason"] == "resume_missing_snapshot"

def test_lhr_resume_completes_only_the_matching_stage_transaction_snapshot(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    ledger_before = solver.ledger_path.read_text(encoding="utf-8")
    ledger_after = ledger_before + "### S03\nmetric: 0.09\nlower_is_better: true\nBRIEF: pending\nWHY: pending\n"
    metric_event = {"artifact_sha": "new-artifact", "gate_accepted": True}
    solver._prepare_stage_commit_transaction(
        stage_id="S03",
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        metric_event=metric_event,
    )
    snapshot = StageSnapshot(
        stage_id="S03",
        snapshot_id="snapshot-three",
        snapshot_path=tmp_path / "snapshots" / "S03",
        metric_value=0.09,
        metric_name="score",
        lower_is_better=True,
        memory_cut=3,
        source_event=dict(metric_event),
        node_uid="W00:L01:S03",
        lineage_id="L01",
    )
    solver.snapshot_store = SimpleNamespace(discover=lambda: {"S03": snapshot})
    solver.ledger_path.write_text(ledger_after, encoding="utf-8")
    solver.pending_stage_commit_transaction = None  # simulate process restart
    solver._recover_stage_commit_transactions()

    assert solver.ledger_path.read_text(encoding="utf-8") == ledger_after
    manifest = json.loads((solver.log_dir / "stage_transactions" / "L01-S03.json").read_text())
    assert manifest["status"] == "committed"
    assert manifest["recovered_after_restart"] is True

def test_lhr_resume_does_not_match_an_older_snapshot_with_same_stage_id(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    ledger_before = solver.ledger_path.read_text(encoding="utf-8")
    ledger_after = ledger_before + "### S03\nmetric: 0.09\nlower_is_better: true\nBRIEF: pending\nWHY: pending\n"
    metric_event = {"artifact_sha": "same-artifact", "gate_accepted": True}
    solver._prepare_stage_commit_transaction(
        stage_id="S03",
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        metric_event=metric_event,
    )
    old_snapshot = StageSnapshot(
        stage_id="S03",
        snapshot_id="old-snapshot",
        snapshot_path=tmp_path / "snapshots" / "old-S03",
        metric_value=0.09,
        metric_name="score",
        lower_is_better=True,
        memory_cut=3,
        source_event={"artifact_sha": "same-artifact"},
        node_uid="W00:L01:S03",
        lineage_id="L01",
    )
    solver.snapshot_store = SimpleNamespace(discover=lambda: {"S03": old_snapshot})
    solver.ledger_path.write_text(ledger_after, encoding="utf-8")
    solver.pending_stage_commit_transaction = None
    solver._recover_stage_commit_transactions()

    assert solver.ledger_path.read_text(encoding="utf-8") == ledger_before
    manifest = json.loads((solver.log_dir / "stage_transactions" / "L01-S03.json").read_text())
    assert manifest["status"] == "rolled_back"

def test_lhr_context_limit_suppresses_repeated_restore_key(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            estra_enabled=True,
            estra_max_decisions=0,
            estra_trigger_stage_count=2,
            estra_decision_only=True,
            estra_compact_max_chars=1200,
            tail_summary_max_chars=1200,
            state_packet_max_chars=12000,
            state_packet_stage_card_max_chars=420,
            state_packet_archived_branch_max_chars=1500,
        )
        solver.stage_snapshots = {"S01": object(), "S02": object()}
        solver.last_estra_stage_count = 2
        solver.estra_decisions = 0
        solver.pending_estra = None

        calls = []

        async def fake_ask_estra(agent, *, trigger_source: str):
            calls.append(trigger_source)
            return {
                "action": "keep_current",
                "target_stage": "S02",
                "reason": "compact current route",
            }

        solver._ask_estra = fake_ask_estra
        first = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=10,
            max_steps=100,
            omitted=3,
        )
        assert first == "[lnr] estra keep_current chosen from context-limit; compact current stage: S02"
        assert solver.pending_estra["compact_strength"] == "strict_context_limit"
        first_key = solver.pending_estra["restore_key"]
        solver.pending_estra = None

        second = await LnrSolver._context_limit_estra_callback(
            solver,
            agent=object(),
            round_idx=10,
            max_steps=100,
            omitted=3,
        )

        assert second is None
        assert calls == ["context_limit"]
        assert solver.pending_estra is None
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        suppressed = [row for row in rows if row["event"] == "context_limit_estra_restore_suppressed"]
        assert suppressed
        assert suppressed[-1]["payload"]["restore_key"] == first_key

    asyncio.run(_run())

def test_lhr_stage_commit_experiment_state_keeps_abandoned_lineage_global_best(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.global_log_dir = tmp_path / "task_logs"
    solver.global_log_dir.mkdir()
    solver._task_metric_lower_is_better = False
    solver.lhr = SimpleNamespace(stage_commit_experiment_state_enabled=True)
    history_path = solver.global_log_dir / lnr_solver_module.LHR_STAGE_PERFORMANCE_CSV
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "worker_id",
                "candidate_id",
                "stage_id",
                "lineage_id",
                "metric_value",
                "validation_ok",
                "metric_validity",
                "selection_eligible",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "worker_id": "W00",
                "candidate_id": "W00:L01:S04",
                "stage_id": "S04",
                "lineage_id": "L01",
                "metric_value": 0.91,
                "validation_ok": "true",
                "metric_validity": "high",
                "selection_eligible": "true",
            }
        )

    state = solver._build_stage_commit_experiment_state(
        stage_id="S03",
        metric_event={
            "metric_value": 0.80,
            "metric_validity": "high",
            "lower_is_better": False,
            "selection_eligible": True,
        },
    )

    assert "global_best_stage: W00:L01:S04" in state
    assert "global_best_metric: 0.91" in state
