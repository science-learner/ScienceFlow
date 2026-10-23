from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import random
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from tests.support.runtime_parity.comparison.compare import (
    compare_capture,
    compare_performance_capture,
)
from tests.support.runtime_parity.cases.scenarios import SCENARIOS, run_scenario

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "runtime_parity"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.yaml"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def load_manifest() -> dict[str, Any]:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or int(manifest.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported runtime parity manifest: {MANIFEST_PATH}")
    return manifest


def _run_worker(case_id: str, work_root: str) -> dict[str, Any]:
    random.seed(0)
    os.environ["TZ"] = "UTC"
    if hasattr(__import__("time"), "tzset"):
        __import__("time").tzset()
    workspace = Path(work_root) / case_id
    capture = run_scenario(case_id, workspace)
    return {"case_id": case_id, "capture": capture}


def _baseline_path(manifest: dict[str, Any], case_id: str) -> Path:
    version = str(manifest["baseline_version"])
    return FIXTURE_ROOT / version / "captures" / f"{case_id}.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _execute_cases(
    case_ids: list[str], *, jobs: int, work_root: Path
) -> list[dict[str, Any]]:
    unknown = sorted(set(case_ids) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Manifest references unknown cases: {', '.join(unknown)}")
    results: list[dict[str, Any]] = []
    if jobs <= 1:
        for case_id in case_ids:
            results.append(_run_worker(case_id, str(work_root)))
        return results
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TZ", "UTC")
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
        futures = {
            executor.submit(_run_worker, case_id, str(work_root)): case_id
            for case_id in case_ids
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(
                    {
                        "case_id": case_id,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
    return sorted(results, key=lambda row: str(row["case_id"]))


def run_benchmark(
    *,
    suite: str = "quick",
    jobs: int = 4,
    output: Path | None = None,
    record_baseline: bool = False,
    allow_dirty: bool = False,
    keep_workspaces: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest()
    suites = manifest.get("suites") or {}
    if suite not in suites:
        raise ValueError(f"Unknown runtime parity suite: {suite}")
    case_ids = [str(value) for value in suites[suite]]
    candidate_commit = _git("rev-parse", "HEAD")
    if record_baseline and candidate_commit != str(manifest["baseline_commit"]):
        raise RuntimeError(
            "Baseline commit mismatch: "
            f"manifest={manifest['baseline_commit']} candidate={candidate_commit}"
        )
    if record_baseline and not allow_dirty and _git("status", "--porcelain"):
        raise RuntimeError(
            "Baseline recording requires a clean worktree; use --allow-dirty only to bootstrap"
        )
    work_root = Path(tempfile.mkdtemp(prefix=f"runtime_parity_{suite}_"))
    try:
        raw_results = _execute_cases(case_ids, jobs=max(1, jobs), work_root=work_root)
        cases: list[dict[str, Any]] = []
        for raw in raw_results:
            case_id = str(raw["case_id"])
            if "error" in raw:
                cases.append(
                    {"case_id": case_id, "compatible": False, "error": raw["error"]}
                )
                continue
            capture = raw["capture"]
            baseline_path = _baseline_path(manifest, case_id)
            if record_baseline:
                _write_json(baseline_path, capture)
                comparison = {"compatible": True, "diff_count": 0, "sections": {}}
            elif not baseline_path.is_file():
                comparison = {
                    "compatible": False,
                    "diff_count": 1,
                    "sections": {},
                    "error": f"missing baseline: {baseline_path.relative_to(REPO_ROOT)}",
                }
            else:
                expected = json.loads(baseline_path.read_text(encoding="utf-8"))
                if suite == "performance":
                    thresholds = manifest["performance_thresholds"]
                    comparison = compare_performance_capture(
                        expected,
                        capture,
                        regression_ratio=float(thresholds["regression_ratio"]),
                        median_floor_ms=float(thresholds["median_floor_ms"]),
                        p95_floor_ms=float(thresholds["p95_floor_ms"]),
                    )
                else:
                    comparison = compare_capture(expected, capture)
            cases.append({"case_id": case_id, **comparison})
        failed = sum(not bool(case.get("compatible")) for case in cases)
        totals = {
            "context_diff_count": sum(
                int(case.get("sections", {}).get("context", {}).get("diff_count", 0))
                for case in cases
            ),
            "mechanism_diff_count": sum(
                int(case.get("sections", {}).get("mechanism", {}).get("diff_count", 0))
                for case in cases
            ),
            "workspace_unregistered_diff_count": sum(
                int(case.get("sections", {}).get("workspace", {}).get("diff_count", 0))
                for case in cases
            ),
            "failed_cases": failed,
        }
        report = {
            "schema_version": 1,
            "baseline_version": manifest["baseline_version"],
            "baseline_commit": manifest["baseline_commit"],
            "candidate_commit": candidate_commit,
            "environment": {
                "python": platform.python_version(),
                "dependency_lock_sha256": hashlib.sha256(
                    (REPO_ROOT / "uv.lock").read_bytes()
                ).hexdigest(),
            },
            "suite": suite,
            "jobs": max(1, jobs),
            "cases": cases,
            "totals": totals,
            "verdict": "pass" if failed == 0 else "fail",
        }
        if output is not None:
            _write_json(output, report)
        return report
    finally:
        if keep_workspaces:
            print(f"runtime parity workspaces: {work_root}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic runtime parity cases"
    )
    parser.add_argument("--suite", default="quick")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/runtime_parity_report.json")
    )
    parser.add_argument("--record-baseline", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--keep-workspaces", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(
        suite=args.suite,
        jobs=args.jobs,
        output=args.output,
        record_baseline=args.record_baseline,
        allow_dirty=args.allow_dirty,
        keep_workspaces=args.keep_workspaces,
    )
    print(
        json.dumps(
            {
                "suite": report["suite"],
                "jobs": report["jobs"],
                "totals": report["totals"],
                "verdict": report["verdict"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == "pass" else 1


__all__ = ["FIXTURE_ROOT", "MANIFEST_PATH", "load_manifest", "main", "run_benchmark"]
