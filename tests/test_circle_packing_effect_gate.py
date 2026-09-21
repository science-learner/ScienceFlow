from __future__ import annotations

import json
from pathlib import Path

import yaml

from tools.benchmarks.circle_packing_effect_gate import compare, discover_runs


def _write_run(
    root: Path,
    *,
    seed: int,
    metric: float,
    elapsed: float,
    lower: bool = False,
) -> None:
    task = root / f"seed-{seed}" / "circle-packing"
    (task / "task_logs").mkdir(parents=True)
    final = task / "merge" / "finals" / "final_00"
    final.mkdir(parents=True)
    (task / "resolved_config.yaml").write_text(
        yaml.safe_dump({"lnr": {"seed": seed}}),
        encoding="utf-8",
    )
    (task / "task_logs" / "state.json").write_text(
        json.dumps(
            {
                "run_id": f"seed-{seed}",
                "status": "completed",
                "elapsed_sec": elapsed,
            }
        ),
        encoding="utf-8",
    )
    (final / "eval_result.json").write_text(
        json.dumps(
            {
                "metric_name": "radii_sum",
                "metric_value": metric,
                "lower_is_better": lower,
                "validation_ok": True,
                "selection_eligible": True,
            }
        ),
        encoding="utf-8",
    )


def test_three_seed_gate_passes_only_when_each_pair_and_aggregate_hold(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    for index, seed in enumerate((2222, 3333, 4444)):
        _write_run(reference_root, seed=seed, metric=2.0 + index * 0.1, elapsed=100 + index)
        _write_run(candidate_root, seed=seed, metric=2.01 + index * 0.1, elapsed=105 + index)

    report = compare(
        discover_runs(reference_root),
        discover_runs(candidate_root),
        seeds=[2222, 3333, 4444],
        score_tolerance_pct=0.0,
        max_runtime_regression_pct=10.0,
        ignore_runtime=False,
    )

    assert report["passed"] is True
    assert report["aggregate"]["median_quality_pass"] is True
    assert report["aggregate"]["worst_quality_pass"] is True


def test_three_seed_gate_fails_on_one_regressed_seed(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    for seed in (2222, 3333, 4444):
        _write_run(reference_root, seed=seed, metric=2.0, elapsed=100)
        _write_run(candidate_root, seed=seed, metric=1.9 if seed == 3333 else 2.1, elapsed=100)

    report = compare(
        discover_runs(reference_root),
        discover_runs(candidate_root),
        seeds=[2222, 3333, 4444],
        score_tolerance_pct=0.0,
        max_runtime_regression_pct=10.0,
        ignore_runtime=False,
    )

    assert report["passed"] is False
    assert any("seed 3333" in failure for failure in report["failures"])
