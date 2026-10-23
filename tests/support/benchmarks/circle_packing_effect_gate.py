#!/usr/bin/env python3
"""Compare paired three-seed Circle Packing reference/candidate runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunResult:
    seed: int
    run_id: str
    status: str
    elapsed_sec: float
    metric_name: str
    metric_value: float
    lower_is_better: bool
    validation_ok: bool
    selection_eligible: bool
    task_dir: str


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def discover_runs(root: Path) -> dict[int, RunResult]:
    runs: dict[int, RunResult] = {}
    for config_path in sorted(root.rglob("resolved_config.yaml")):
        task_dir = config_path.parent
        state_path = task_dir / "task_logs" / "state.json"
        eval_path = task_dir / "merge" / "finals" / "final_00" / "eval_result.json"
        if not state_path.is_file() or not eval_path.is_file():
            continue
        config = _load_yaml(config_path)
        lnr = config.get("lnr") if isinstance(config.get("lnr"), dict) else {}
        seed = int(lnr.get("seed"))
        if seed in runs:
            raise ValueError(f"duplicate seed {seed} below {root}")
        state = _load_json(state_path)
        evaluation = _load_json(eval_path)
        runs[seed] = RunResult(
            seed=seed,
            run_id=str(state.get("run_id") or ""),
            status=str(state.get("status") or ""),
            elapsed_sec=float(state.get("elapsed_sec") or 0.0),
            metric_name=str(evaluation.get("metric_name") or ""),
            metric_value=float(evaluation.get("metric_value")),
            lower_is_better=bool(evaluation.get("lower_is_better", False)),
            validation_ok=bool(evaluation.get("validation_ok", False)),
            selection_eligible=bool(evaluation.get("selection_eligible", False)),
            task_dir=str(task_dir),
        )
    return runs


def _quality_ok(candidate: float, reference: float, *, lower: bool, tolerance_pct: float) -> bool:
    allowance = abs(reference) * tolerance_pct / 100.0
    if lower:
        return candidate <= reference + allowance
    return candidate >= reference - allowance


def _quality_worst(values: list[float], *, lower: bool) -> float:
    return max(values) if lower else min(values)


def compare(
    reference: dict[int, RunResult],
    candidate: dict[int, RunResult],
    *,
    seeds: list[int],
    score_tolerance_pct: float,
    max_runtime_regression_pct: float,
    ignore_runtime: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    missing_reference = [seed for seed in seeds if seed not in reference]
    missing_candidate = [seed for seed in seeds if seed not in candidate]
    extra_reference = sorted(set(reference) - set(seeds))
    extra_candidate = sorted(set(candidate) - set(seeds))
    if missing_reference:
        failures.append(f"missing reference seeds: {missing_reference}")
    if missing_candidate:
        failures.append(f"missing candidate seeds: {missing_candidate}")
    if extra_reference:
        failures.append(f"unexpected reference seeds: {extra_reference}")
    if extra_candidate:
        failures.append(f"unexpected candidate seeds: {extra_candidate}")

    paired = [seed for seed in seeds if seed in reference and seed in candidate]
    for seed in paired:
        ref = reference[seed]
        cand = candidate[seed]
        if ref.metric_name != cand.metric_name:
            failures.append(f"seed {seed}: metric name differs ({ref.metric_name!r} vs {cand.metric_name!r})")
        if ref.lower_is_better != cand.lower_is_better:
            failures.append(f"seed {seed}: metric direction differs")
        for label, run in (("reference", ref), ("candidate", cand)):
            if run.status != "completed":
                failures.append(f"seed {seed}: {label} status is {run.status!r}")
            if not run.validation_ok or not run.selection_eligible:
                failures.append(f"seed {seed}: {label} evaluation is not eligible")
            if not math.isfinite(run.metric_value):
                failures.append(f"seed {seed}: {label} metric is not finite")
        quality_pass = _quality_ok(
            cand.metric_value,
            ref.metric_value,
            lower=ref.lower_is_better,
            tolerance_pct=score_tolerance_pct,
        )
        runtime_limit = ref.elapsed_sec * (1.0 + max_runtime_regression_pct / 100.0)
        runtime_pass = ignore_runtime or cand.elapsed_sec <= runtime_limit
        if not quality_pass:
            failures.append(f"seed {seed}: candidate quality regressed")
        if not runtime_pass:
            failures.append(f"seed {seed}: candidate runtime regressed")
        rows.append(
            {
                "seed": seed,
                "reference": asdict(ref),
                "candidate": asdict(cand),
                "metric_delta": cand.metric_value - ref.metric_value,
                "elapsed_delta_sec": cand.elapsed_sec - ref.elapsed_sec,
                "quality_pass": quality_pass,
                "runtime_pass": runtime_pass,
            }
        )

    aggregate: dict[str, Any] = {}
    if len(paired) == len(seeds) and paired:
        lower = reference[paired[0]].lower_is_better
        metric_name = reference[paired[0]].metric_name
        if any(reference[seed].lower_is_better != lower for seed in paired):
            failures.append("reference metric direction differs across seeds")
        if any(candidate[seed].lower_is_better != lower for seed in paired):
            failures.append("candidate metric direction differs across seeds")
        if any(reference[seed].metric_name != metric_name for seed in paired):
            failures.append("reference metric name differs across seeds")
        if any(candidate[seed].metric_name != metric_name for seed in paired):
            failures.append("candidate metric name differs across seeds")
        ref_scores = [reference[seed].metric_value for seed in paired]
        cand_scores = [candidate[seed].metric_value for seed in paired]
        ref_times = [reference[seed].elapsed_sec for seed in paired]
        cand_times = [candidate[seed].elapsed_sec for seed in paired]
        ref_median = statistics.median(ref_scores)
        cand_median = statistics.median(cand_scores)
        ref_worst = _quality_worst(ref_scores, lower=lower)
        cand_worst = _quality_worst(cand_scores, lower=lower)
        median_quality_pass = _quality_ok(
            cand_median,
            ref_median,
            lower=lower,
            tolerance_pct=score_tolerance_pct,
        )
        worst_quality_pass = _quality_ok(
            cand_worst,
            ref_worst,
            lower=lower,
            tolerance_pct=score_tolerance_pct,
        )
        ref_runtime_median = statistics.median(ref_times)
        cand_runtime_median = statistics.median(cand_times)
        runtime_median_pass = ignore_runtime or cand_runtime_median <= (
            ref_runtime_median * (1.0 + max_runtime_regression_pct / 100.0)
        )
        if not median_quality_pass:
            failures.append("aggregate median quality regressed")
        if not worst_quality_pass:
            failures.append("aggregate worst-seed quality regressed")
        if not runtime_median_pass:
            failures.append("aggregate median runtime regressed")
        aggregate = {
            "reference_metric_median": ref_median,
            "candidate_metric_median": cand_median,
            "reference_metric_worst": ref_worst,
            "candidate_metric_worst": cand_worst,
            "reference_elapsed_median_sec": ref_runtime_median,
            "candidate_elapsed_median_sec": cand_runtime_median,
            "median_quality_pass": median_quality_pass,
            "worst_quality_pass": worst_quality_pass,
            "runtime_median_pass": runtime_median_pass,
        }

    return {
        "gate": "circle_packing_three_seed_effect",
        "passed": not failures,
        "seeds": seeds,
        "policy": {
            "score_tolerance_pct": score_tolerance_pct,
            "max_runtime_regression_pct": max_runtime_regression_pct,
            "ignore_runtime": ignore_runtime,
            "requirements": [
                "all paired runs completed with valid eligible evaluations",
                "every paired seed preserves metric quality",
                "aggregate median and worst-seed metric preserve quality",
                "every paired seed and aggregate median preserve runtime unless explicitly ignored",
            ],
        },
        "rows": rows,
        "aggregate": aggregate,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--seeds", default="2222,3333,4444")
    parser.add_argument("--score-tolerance-pct", type=float, default=0.0)
    parser.add_argument("--max-runtime-regression-pct", type=float, default=10.0)
    parser.add_argument("--ignore-runtime", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        parser.error("--seeds must contain unique integers")
    report = compare(
        discover_runs(args.reference_root),
        discover_runs(args.candidate_root),
        seeds=seeds,
        score_tolerance_pct=max(0.0, args.score_tolerance_pct),
        max_runtime_regression_pct=max(0.0, args.max_runtime_regression_pct),
        ignore_runtime=args.ignore_runtime,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
