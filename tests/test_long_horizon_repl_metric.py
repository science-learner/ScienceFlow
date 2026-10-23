"""Long Horizon Repl contracts: metric."""

from __future__ import annotations

from tests._long_horizon_repl_support import *  # noqa: F401,F403

from scienceflow.research.solver.lnr.orchestration.coordinator.lifecycle.context.coordination import (
    _parse_stage_commit_text_block,
)


def test_lhr_agent_uses_grounded_llm_metric_fallback(tmp_path: Path) -> None:
    ws = tmp_path / "interpreted_metric_ws"
    ws.mkdir()
    (ws / "solution.py").write_text("print('metric')\n", encoding="utf-8")
    (ws / "context.json").write_text("{}", encoding="utf-8")
    agent = _make_metric_snapshot_agent(ws)

    async def interpret(**_: object) -> dict[str, object]:
        return {
            "metric_found": True,
            "metric_name": "weighted_auc",
            "metric_value": 0.641136,
            "split": "validation",
            "is_final": True,
            "evidence_line": "Best Validation wAUC: 0.641136",
            "confidence": "high",
        }

    object.__setattr__(agent, "_lnr_metric_output_interpretation_callback", interpret)
    asyncio.run(
        agent._maybe_write_bare_run_tail_snapshot(
            {"command": "python3 solution.py"},
            ToolResult(output="Best Validation wAUC: 0.641136\n", error=""),
        )
    )

    assert agent._lnr_snapshot_ok is True
    data = json.loads((ws / ".logs" / "fullrun_tail_snapshot.json").read_text(encoding="utf-8"))
    assert data["metric_value"] == pytest.approx(0.641136)
    assert data["metric_name"] == "weighted_auc"

def test_lhr_stage_commit_prompt_includes_metric_semantics_fields() -> None:
    text = build_stage_commit_prompt(
        stage_id="S02",
        ledger_filename=".run_results.md",
        metric_event={
            "metric_value": 0.061,
            "lower_is_better": True,
            "run_time_sec": 12.3,
            "val_score_type": "holdout",
            "selection_note": "",
        },
        existing_ledger="### S01\nmetric: 0.07\nlower_is_better: true\nBRIEF: base\nWHY: ok\n",
    )

    assert "Append exactly one new stage entry" in text
    assert "Preserve all existing entries unchanged" in text
    assert "metric_type:" in text
    assert "metric_note:" in text
    assert "run_time_sec:" in text
    assert "copy `run_time_sec`" in text
    assert "copy `val_score_type`" in text
    assert "FILES:" in text
    assert "exclude outputs" in text

def test_lhr_state_packet_uses_adjudicated_metric_validity(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(
        state_packet_max_chars=12000,
        state_packet_stage_card_max_chars=420,
        state_packet_archived_branch_max_chars=1500,
    )
    solver.ledger_path.write_text(
        "### S01\n"
        "metric: 0.91\n"
        "lower_is_better: false\n"
        "metric_validity: high\n"
        "selection_eligible: true\n"
        "BRIEF: raw ledger claim\n"
        "WHY: raw\n\n"
        "### S02\n"
        "metric: 0.90\n"
        "lower_is_better: false\n"
        "metric_validity: high\n"
        "selection_eligible: true\n"
        "BRIEF: second\n"
        "WHY: second\n",
        encoding="utf-8",
    )
    solver.stage_snapshots = {
        "S01": StageSnapshot(
            stage_id="S01",
            snapshot_id="snap-s01",
            snapshot_path=tmp_path / "snap-s01",
            metric_value=0.91,
            metric_name="Final Validation Score",
            lower_is_better=False,
            memory_cut=1,
            source_event={
                "metric_value": 0.91,
                "lower_is_better": False,
                "metric_validity": "medium",
                "selection_eligible": False,
                "metric_validity_reason_code": "unknown_protocol",
                "source_commit_sha": "abcdef1234567890",
            },
            node_uid="W00:L01:S01",
            lineage_id="L01",
        ),
        "S02": StageSnapshot(
            stage_id="S02",
            snapshot_id="snap-s02",
            snapshot_path=tmp_path / "snap-s02",
            metric_value=0.90,
            metric_name="Final Validation Score",
            lower_is_better=False,
            memory_cut=2,
            source_event={"source_commit_sha": "123456abcdef9999"},
            node_uid="W00:L01:S02",
            lineage_id="L01",
        ),
    }

    packet = solver._build_lhr_state_packet(
        action="keep_current",
        target_stage="S02",
        terminal_stage="S02",
        reason="compact",
    )

    assert "metric_validity=medium" in packet
    assert "selection_eligible=false" in packet
    assert "metric_validity_reason=unknown_protocol" in packet
    s01_line = packet.split("- S01", 1)[1].split("- S02", 1)[0]
    assert "metric_validity=high" not in s01_line

def test_validation_leakage_guard_flags_train_val_metric_reuse() -> None:
    source = """
import pandas as pd
train = pd.read_csv('dataset/train.csv')
val = pd.read_csv('dataset/validation.csv')
full_train = pd.concat([train, val], axis=0)
print(f"Final Validation Score: {0.029353:.6f}")
"""
    stdout = """
=== Training final models ===
=== Optimizing ensemble weights on validation ===
=== Final Validation Score: 0.029353 ===
"""
    assert _validation_leakage_reason(source, stdout)

def test_validation_leakage_guard_allows_post_metric_final_retrain() -> None:
    source = """
print(f"Final Validation Score: {0.061:.6f}")
full_train = pd.concat([train, val], axis=0)
"""
    assert _validation_leakage_reason(source, "Final Validation Score: 0.061") == ""

def test_metric_semantics_detects_multifile_fulltrain_replay(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "train.py").write_text(
        """
import pandas as pd
train = pd.read_csv('dataset/train.csv')
val = pd.read_csv('dataset/validation.csv')
X_full = pd.concat([train, val], axis=0)
""",
        encoding="utf-8",
    )
    (solver.workspace_dir / "predict.py").write_text(
        "print('predict using saved full-train artifacts')\n",
        encoding="utf-8",
    )
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.0004834944942880695,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": (
                    "Model trained on train+val (2160 samples)\n"
                    "Loading validation features for score computation...\n"
                    "Final Validation Score: 0.0004834944942880695\n"
                ),
                "stderr_tail": "",
                "bash_cmd": "python3 predict.py 2>&1",
                "solution_path": "predict.py",
                "validation_ok": True,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["reported_val_score"] == 0.0004834944942880695
    assert event["val_score_type"] == "post_fulltrain_replay"
    assert event["selection_eligible"] is False
    assert event["selection_score"] == ""
    assert event["selection_note"] == "fulltrain_replay_metric_kept_but_not_used_for_selection"
    assert event["metric_validity"] == "low"
    assert event["execution_mode"] == "predict_only_reuse_artifacts"
    assert event["validation_ok"] is True

def test_authoritative_evaluator_direction_overrides_task_text_hint(
    tmp_path: Path,
) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver._task_metric_lower_is_better = True
    event = {
        "metric_name": "global_ndcg",
        "lower_is_better": False,
        "metric_authoritative": True,
    }

    decision = solver._metric_lower_is_better_decision(event)

    assert decision["lower_is_better"] is False
    assert decision["metric_direction_source"] == "authoritative_evaluator"
    assert decision["metric_direction_conflict"] is False

def test_metric_semantics_system_overrides_declared_holdout_replay(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "predict.py").write_text("print('predict')\n", encoding="utf-8")
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.0123,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": (
                    "Metric Protocol: holdout\n"
                    "Model trained on train+val (2160 samples)\n"
                    "Loading validation features for score computation...\n"
                    "Final Validation Score: 0.0123\n"
                ),
                "stderr_tail": "",
                "bash_cmd": "python3 predict.py",
                "solution_path": "predict.py",
                "validation_ok": True,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["metric_protocol"] == "holdout"
    assert event["val_score_type"] == "post_fulltrain_replay"
    assert event["selection_eligible"] is False
    assert event["selection_note"] == "agent_declared_holdout_but_system_detected_fulltrain_replay"
    assert event["metric_validity"] == "low"

def test_metric_semantics_allows_holdout_then_full_retrain(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.lhr = SimpleNamespace(stage_commit_require_metric=True, metric_validation_leakage_guard_enabled=True)
    solver.workspace_dir.mkdir()
    logs_dir = solver.workspace_dir / ".logs"
    logs_dir.mkdir()
    (solver.workspace_dir / "train.py").write_text(
        """
print('Final Validation Score: 0.061')
full_train = pd.concat([train, val], axis=0)
print('Retraining on full train+val for submission')
""",
        encoding="utf-8",
    )
    (logs_dir / "fullrun_tail_snapshot.json").write_text(
        json.dumps(
            {
                "metric_value": 0.061,
                "metric_name": "Final Validation Score",
                "lower_is_better": True,
                "stdout_tail": (
                    "Validation RMSLE: 0.061\n"
                    "Retraining on full train+val for submission\n"
                    "Final Validation Score: 0.061\n"
                ),
                "stderr_tail": "",
                "bash_cmd": "python3 train.py",
                "solution_path": "train.py",
                "validation_ok": True,
                "wall_sec": 8.75,
            }
        ),
        encoding="utf-8",
    )

    event = solver._metric_event_from_workspace()

    assert event is not None
    assert event["val_score_type"] == "holdout"
    assert event["selection_eligible"] is True
    assert event["selection_score"] == 0.061
    assert event["metric_validity"] == "high"
    assert event["run_time_sec"] == 8.75

def test_lhr_primary_gate_fails_closed_when_evaluator_emits_no_outcome(tmp_path: Path) -> None:
    solver = _minimal_lhr_solver(tmp_path)
    solver.workspace_dir.mkdir(parents=True, exist_ok=True)
    solver.cfg = SimpleNamespace(
        exp_id="test-task",
        evaluator=SimpleNamespace(
            task_profile="mlebench",
            backend="task_package",
            stage_source_mode="primary",
            event_log="evaluator_events.jsonl",
        ),
    )
    solver.deadline = float("inf")
    solver.gate_service = SimpleNamespace(evaluate=lambda _request: [])

    facts = solver._record_evaluator_stage_events(
        stage_id="S03",
        metric_event={
            "metric_value": 0.8,
            "metric_name": "score",
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
        },
    )

    assert facts["metric_value"] == 0.8
    assert facts["validation_ok"] is False
    assert facts["candidate_ready"] is False
    assert facts["gate_accepted"] is False
    assert facts["gate_reason_code"] == "evaluator_no_outcome"
    assert facts["_gate_evaluated"] is True

def test_lhr_primary_evaluator_error_is_returned_to_agent(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
        )
        solver.last_stage_commit_ts = 0.0
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._evaluator_candidate_artifact = lambda: "artifacts/best_solution.json"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            "metric_value": None,
            "evaluator_backend": "artifact_command",
            "evaluator_status": "evaluator_error",
            "artifact_path": "artifacts/best_solution.json",
            "metric_source_note": "evaluator command exited with 2",
            "evaluator_stderr_tail": "kttsp_eval error: leg 6 tof 0.0 is below min_tof 0.001",
        }

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is not None
        assert out.startswith("EVALUATOR_INVALID_ARTIFACT")
        assert "artifacts/best_solution.json" in out
        assert "evaluator_error" in out
        assert "leg 6 tof 0.0" in out
        assert "satisfies the task evaluator" in out

    asyncio.run(_run())

def test_lhr_primary_evaluator_ready_candidate_requires_text_commit(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.deadline = 10.0
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            stage_commit_llm_timeout_sec=60,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            "metric_value": 0.42,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "artifact_path": "submission.csv",
            "artifact_sha": "ready-sha",
            "submission_sha": "ready-sha",
            "submission_status": "ok",
            "evaluator_backend": "task_package",
            "evaluator_status": "ok",
            "gate_action": "accept",
            "gate_accepted": True,
            "gate_reason_code": "eligible",
            "_gate_evaluated": True,
            "_evaluator_stage_facts_applied": True,
        }

        async def fake_stage_commit(**_kwargs):  # pragma: no cover - must not run
            raise AssertionError("primary evaluator ready candidate must wait for text commit")

        solver._ephemeral_stage_commit = fake_stage_commit
        agent = SimpleNamespace(_run_policy=SimpleNamespace(deadline_monotonic=10.0))

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=agent,
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert solver.pending_text_stage_commit["stage_id"] == "S03"
        assert solver.pending_text_stage_commit["metric_event"]["artifact_path"] == "submission.csv"
        assert "### S03" not in solver.ledger_path.read_text(encoding="utf-8")
        assert "STAGE_COMMIT_BEGIN" in agent._lnr_transient_user_prompt
        assert agent._lnr_stage_commit_text_pending is True
        assert agent._run_policy.deadline_monotonic > 10.0

    asyncio.run(_run())


def test_lhr_metric_only_stage_does_not_inherit_workspace_artifact(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        artifact = solver.workspace_dir / "artifacts" / "best_solution.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"score": 1.0}\n', encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        solver.deadline = 10.0
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            stage_commit_llm_timeout_sec=60,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {
            "S01": StageSnapshot(
                "S01",
                "snap1",
                tmp_path / "snap1",
                1.0,
                "radii_sum",
                False,
                1,
                {
                    "artifact_sha": artifact_sha,
                    "artifact_path": "artifacts/best_solution.json",
                    "candidate_ready": True,
                },
                node_uid="W00:L01:S01",
                lineage_id="L01",
            )
        }
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 1.1,
            "metric_name": "radii_sum",
            "lower_is_better": False,
            "validation_ok": True,
            "selection_eligible": True,
            "metric_validity": "high",
            "val_score_type": "benchmark",
            "solution_sha": "new-score-source",
            "submission_status": "missing_submission",
            "stage_signal_kind": "metric_only",
            "candidate_artifact_attached": False,
        }

        def unexpected_artifact_lookup(_event):
            raise AssertionError("metric-only Stage must not inherit a stale artifact")

        solver._candidate_artifact_sha_from_workspace = unexpected_artifact_lookup
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            **metric_event,
            "candidate_ready": True,
            "evaluator_backend": "result_signal",
            "evaluator_status": "valid_full_run_result",
            "gate_action": "accept",
            "gate_accepted": True,
            "gate_reason_code": "eligible",
            "_gate_evaluated": True,
        }
        agent = SimpleNamespace(_run_policy=SimpleNamespace(deadline_monotonic=10.0))

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=agent,
            args={},
            tool_result=ToolResult(output="FINAL radii_sum=1.1"),
        )

        assert out is None
        assert solver.pending_text_stage_commit["stage_id"] == "S03"
        event = solver.pending_text_stage_commit["metric_event"]
        assert event["metric_value"] == pytest.approx(1.1)
        assert not event.get("artifact_sha")

    asyncio.run(_run())


def test_lhr_new_artifact_without_fresh_score_ignores_stale_snapshot(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
        )
        solver.last_stage_commit_ts = 0.0

        def stale_snapshot_lookup():
            raise AssertionError("stale metric snapshot must not be read")

        solver._metric_event_from_workspace = stale_snapshot_lookup
        solver._evaluate_missing_metric_stage = lambda **_kwargs: (
            {},
            "evaluated-current-artifact",
            True,
        )
        agent = SimpleNamespace(
            _lnr_candidate_artifact_pending_sha="new-artifact-sha",
            _lnr_snapshot_ok=False,
        )

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=agent,
            args={},
            tool_result=ToolResult(output="artifact ready"),
        )

        assert out == "evaluated-current-artifact"

    asyncio.run(_run())

def test_lhr_metric_missing_artifact_is_not_re_evaluated_while_metric_absent(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
        )
        solver.last_stage_commit_ts = 0.0
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._candidate_artifact_sha_from_workspace = lambda _event: "pending-artifact"
        solver._evaluator_feedback_for_agent = lambda _event: "metric missing"
        evaluations = 0

        def reject_missing_metric(**_kwargs):
            nonlocal evaluations
            evaluations += 1
            return {
                "_gate_evaluated": True,
                "gate_accepted": False,
                "gate_reason_code": "metric_missing",
            }

        solver._record_evaluator_stage_events = reject_missing_metric
        first = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="artifact written"),
        )
        second = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="unrelated tool"),
        )

        assert first == "metric missing"
        assert second is None
        assert evaluations == 1
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        skipped = [
            row
            for row in rows
            if row["event"] == "duplicate_pending_metric_pre_gate_skipped"
        ]
        assert skipped[-1]["payload"]["artifact_sha"] == "pending-artifact"

    asyncio.run(_run())


def test_lhr_rejected_artifact_is_not_re_evaluated_after_session_rollover(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
        )
        solver.last_stage_commit_ts = 0.0
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: None
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._candidate_artifact_sha_from_workspace = lambda _event: "invalid-artifact"
        solver._evaluator_feedback_for_agent = lambda _event: "invalid candidate"
        evaluations = 0

        def reject_invalid_candidate(**_kwargs):
            nonlocal evaluations
            evaluations += 1
            return {
                "_gate_evaluated": True,
                "gate_accepted": False,
                "gate_reason_code": "validation_failed",
            }

        solver._record_evaluator_stage_events = reject_invalid_candidate
        first = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="invalid artifact written"),
        )
        # A resumed IQ session invokes the same host callback with the same artifact.
        second = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="unrelated tool after rollover"),
        )

        assert first == "invalid candidate"
        assert second is None
        assert evaluations == 1
        rows = [
            json.loads(line)
            for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()
        ]
        skipped = [
            row
            for row in rows
            if row["event"] == "duplicate_rejected_candidate_pre_gate_skipped"
        ]
        assert skipped[-1]["payload"]["artifact_sha"] == "invalid-artifact"
        assert skipped[-1]["payload"]["prior_gate_reason_code"] == "validation_failed"

    asyncio.run(_run())

def test_lhr_gate_retry_with_valid_metric_does_not_create_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.52,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "solution_sha": "src",
            "submission_sha": "candidate-sha",
        }
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            **metric_event,
            "validation_ok": True,
            "candidate_ready": True,
            "selection_eligible": False,
            "metric_validity": "medium",
            "artifact_path": "submission.csv",
            "artifact_sha": "candidate-sha",
            "evaluator_backend": "task_package",
            "evaluator_status": "ok",
            "gate_action": "retry",
            "gate_accepted": False,
            "gate_reason_code": "metric_validity_below_gate",
            "gate_message": "metric validity is below required high",
            "_gate_evaluated": True,
        }
        solver._evaluator_feedback_for_agent = lambda _event: "gate feedback"

        async def fake_stage_commit(**_kwargs):  # pragma: no cover - must not run
            raise AssertionError("a rejected gate outcome must not request a stage commit")

        solver._ephemeral_stage_commit = fake_stage_commit

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out == "gate feedback"
        assert not hasattr(solver, "pending_text_stage_commit") or solver.pending_text_stage_commit is None
        assert "### S03" not in solver.ledger_path.read_text(encoding="utf-8")
        assert "S03" not in solver.stage_snapshots
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        rejected = [row for row in rows if row["event"] == "stage_gate_rejected"]
        assert rejected[-1]["payload"]["gate_reason_code"] == "metric_validity_below_gate"

    asyncio.run(_run())

def test_lhr_primary_evaluator_invalid_candidate_is_skipped(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_text_mode=True,
            workspace_git_enabled=False,
        )
        solver.last_stage_commit_ts = 0.0
        solver.last_captured_run_signature = ""
        solver.stage_snapshots = {}
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.52,
            "metric_name": "Final Validation Score",
            "lower_is_better": False,
            "solution_sha": "src",
            "submission_sha": "bad-sha",
        }
        solver._evaluator_stage_source_mode = lambda: "primary"
        solver._record_evaluator_stage_events = lambda *, stage_id, metric_event: {
            **metric_event,
            "validation_ok": False,
            "candidate_ready": False,
            "selection_eligible": False,
            "artifact_path": "submission.csv",
            "artifact_sha": "bad-sha",
            "evaluator_backend": "task_package",
            "evaluator_status": "invalid_submission",
        }
        solver._evaluator_feedback_for_agent = lambda _event: "feedback"

        async def fake_stage_commit(**_kwargs):  # pragma: no cover - must not run
            raise AssertionError("invalid evaluator candidate must not commit")

        solver._ephemeral_stage_commit = fake_stage_commit

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out == "feedback"
        assert "S03" not in solver.stage_snapshots
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "stage_invalid_evaluator_candidate_skipped" in [row["event"] for row in rows]

    asyncio.run(_run())

def test_lhr_metric_stage_without_submission_marks_missing_submission(tmp_path: Path) -> None:
    async def _run() -> None:
        solver = _minimal_lhr_solver(tmp_path)
        solver.workspace_dir.mkdir(parents=True, exist_ok=True)
        solver.memory_dir = tmp_path / "memory"
        solver.memory_dir.mkdir(parents=True, exist_ok=True)
        solver.global_log_dir = solver.log_dir
        solver.lhr = SimpleNamespace(
            stage_capture_enabled=True,
            stage_commit_min_seconds_between=0,
            stage_commit_require_metric=True,
            stage_commit_text_mode=False,
            metric_validity_adjudicator_enabled=False,
            force_estra_capture_duplicate_submissions=False,
            force_estra_after_stage_count=0,
            workspace_git_enabled=False,
            workspace_git_auto_checkpoint=False,
            workspace_git_track_globs=["*.py", "*.md"],
        )
        solver.stage_snapshots = {}
        solver.last_captured_solution_sha = ""
        solver.last_captured_run_signature = ""
        solver.last_stage_commit_ts = 0.0
        solver.current_lineage_no = 1
        solver.current_lineage_id = "L01"
        solver.current_restored_from_stage = ""
        solver.current_restored_from_node_uid = ""
        solver.main_llm_calls = 0
        solver.main_tokens_in = 0
        solver.main_tokens_out = 0
        solver.main_tokens_cached = 0
        solver.stage_llm_calls = 0
        solver.estra_llm_calls = 0
        solver._metric_event_from_workspace = lambda: {
            "metric_value": 0.061,
            "metric_name": "Final Validation Score",
            "lower_is_better": True,
            "solution_sha": "train-only-source",
            "solution_path": "train.py",
            "submission_sha": "",
            "validation_ok": True,
            "submission_status": "missing_submission",
            "selection_eligible": True,
            "selection_score": 0.061,
        }

        async def fake_stage_commit(*, agent, stage_id, metric_event):
            assert stage_id == "S03"
            assert metric_event["candidate_ready"] is False
            assert metric_event["submission_status"] == "missing_submission"
            with solver.ledger_path.open("a", encoding="utf-8") as f:
                f.write("\n### S03\nmetric: 0.061\nlower_is_better: true\nBRIEF: validation-only train stage\nWHY: no submission required for stage\n")
            return True, ""

        class FakeSnapshotStore:
            def capture(self, **kwargs):
                return StageSnapshot(
                    kwargs["stage_id"],
                    "snap1",
                    tmp_path / "snap1",
                    kwargs.get("metric_value"),
                    kwargs.get("metric_name", ""),
                    kwargs.get("lower_is_better"),
                    kwargs.get("memory_cut", 0),
                    kwargs.get("source_event", {}),
                    node_uid=kwargs.get("node_uid", ""),
                    lineage_id=kwargs.get("lineage_id", ""),
                )

        solver._ephemeral_stage_commit = fake_stage_commit
        solver.snapshot_store = FakeSnapshotStore()
        solver._append_stage_performance_row = lambda **kwargs: None
        solver._mark_protected_eda_prefix = lambda agent, stage_id: {}
        solver._reset_agent_interaction_stage = lambda agent: None

        async def fake_force_estra_after_stage_capture(**kwargs):
            return None

        solver._force_estra_after_stage_capture = fake_force_estra_after_stage_capture

        out = await LnrSolver._stage_capture_callback(
            solver,
            agent=object(),
            args={},
            tool_result=ToolResult(output="ok"),
        )

        assert out is None
        assert "S03" in solver.stage_snapshots
        assert solver.stage_snapshots["S03"].source_event["candidate_ready"] is False
        assert solver.stage_snapshots["S03"].source_event["submission_status"] == "missing_submission"
        rows = [json.loads(line) for line in (solver.log_dir / "lhr_events.jsonl").read_text().splitlines()]
        assert "stage_captured" in [row["event"] for row in rows]

    asyncio.run(_run())

def test_lhr_stage_commit_text_block_parser_accepts_negative_metric() -> None:
    text = """
    STAGE_COMMIT_BEGIN
    stage_id: S04
    metric: -4.835054
    metric_validity: medium
    brief: valid negative metric stage.
    why: lower-is-better metric improved relative to prior run.
    route_evidence: verdict=continue; reason=metric is comparable; next=refine calibration
    files: code=train.py weights=models/best.pt
    STAGE_COMMIT_END
    """

    parsed, block_text, reason = _parse_stage_commit_text_block(text)

    assert reason == ""
    assert parsed["stage_id"] == "S04"
    assert float(parsed["metric"]) == pytest.approx(-4.835054)
    assert parsed["metric_validity"] == "medium"
    assert parsed["files"] == "code=train.py weights=models/best.pt"
    assert block_text.startswith("STAGE_COMMIT_BEGIN")
