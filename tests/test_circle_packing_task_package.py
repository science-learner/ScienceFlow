"""The built-in circle-packing task is self-contained and executable."""

from __future__ import annotations

import json
import venv

import yaml

from scienceflow.foundation.contracts import EvalContext
from scienceflow.research.onboarding import (
    LongResearchSession,
    PreflightStatus,
    run_preflight,
    write_onboarding_files,
)
from scienceflow.research.quality.evaluator import EvaluatorManager
from scienceflow.runtime.task_package import find_task_package


def test_circle_packing_preflight_runs_real_samples_without_dataset(tmp_path):
    clean_env = tmp_path / "clean-env"
    venv.EnvBuilder(with_pip=False).create(clean_env)
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints="circle-packing data=none gpu=cpu cpu=0-1 workers=1 duration=10m",
    )
    files = write_onboarding_files(session.draft, tmp_path)
    manifest = yaml.safe_load(files.manifest_path.read_text(encoding="utf-8"))
    task = manifest["tasks"][0]
    task["evaluator"]["command"] = {
        "python_executable": str(clean_env / "bin/python")
    }
    files.manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    report = run_preflight(files.manifest_path)

    checks = {check.name: check for check in report.checks}
    assert report.status == PreflightStatus.VERIFIED
    assert checks["dataset_scan"].ok
    assert checks["evaluator_execution"].ok
    assert "accepted the valid sample" in checks["evaluator_execution"].detail


def test_circle_packing_backend_scores_without_problem_json_or_numpy(tmp_path):
    spec = find_task_package("circle-packing")
    assert spec is not None
    valid = spec.source_dir / "validation/valid_solution.json"
    artifact = tmp_path / spec.artifact_path
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(valid.read_bytes())
    ctx = EvalContext(
        task_profile="opt_solver",
        task_id="circle-packing",
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W00",
        stage_id="S01",
        cfg={},
    )
    backend = EvaluatorManager.default().get("task_package")
    assert backend is not None

    candidates = backend.detect_candidates(ctx)
    result = backend.evaluate(ctx, candidates[0])

    assert not (tmp_path / "dataset/problem.json").exists()
    assert result.ok and result.selection_eligible
    assert result.metric_name == "radii_sum"
    assert result.metric_value == 0.26
    assert "numpy" not in (spec.source_dir / "evaluator.py").read_text(encoding="utf-8")


def test_circle_packing_backend_rejects_invalid_artifact(tmp_path):
    spec = find_task_package("circle-packing")
    assert spec is not None
    artifact = tmp_path / spec.artifact_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"circles": [[0.5, 0.5, 0.1]] * 26}))
    ctx = EvalContext(
        task_profile="opt_solver",
        task_id="circle-packing",
        task_root=tmp_path,
        workspace=tmp_path,
        worker_id="W00",
        stage_id="S01",
        cfg={},
    )
    backend = EvaluatorManager.default().get("task_package")
    assert backend is not None

    result = backend.evaluate(ctx, backend.detect_candidates(ctx)[0])

    assert not result.ok
    assert not result.selection_eligible
    assert "overlap" in result.stderr_tail
