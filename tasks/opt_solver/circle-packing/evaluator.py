# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Evaluate circle-packing JSON artifacts for opt_solver tasks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT_TOL = 1e-6
DEFAULT_PROBLEM = {
    "benchmark_radii_sum": 2.635,
    "container": "unit_square",
    "num_circles": 26,
    "tolerance": DEFAULT_TOL,
}
Circle = tuple[float, float, float]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON: {path}: {exc}") from exc


def _load_problem(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"problem file not found: {path}")
    data = _load_json(path)
    if not isinstance(data, dict):
        raise TypeError("problem JSON must be an object")
    return data


def _triple(value: Any, *, label: str) -> Circle:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be [x, y, radius]")
    if any(isinstance(item, bool) for item in value):
        raise ValueError(f"{label} values must be numbers")
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be numbers") from exc


def _load_circles(path: Path) -> list[Circle]:
    if not path.is_file():
        raise FileNotFoundError(f"solution file not found: {path}")
    data = _load_json(path)
    if isinstance(data, dict):
        raw = data.get("circles")
        if raw is None and "centers" in data and "radii" in data:
            centers, radii = data["centers"], data["radii"]
            if not isinstance(centers, list) or any(
                not isinstance(center, (list, tuple)) or len(center) != 2
                for center in centers
            ):
                raise ValueError("centers must be a list of [x, y] pairs")
            if not isinstance(radii, list) or len(radii) != len(centers):
                raise ValueError("radii must have shape (n,) and match centers")
            raw = [[*center, radius] for center, radius in zip(centers, radii)]
    else:
        raw = data
    if raw is None:
        raise ValueError("solution JSON must be a list of circles or contain key 'circles'")
    if not isinstance(raw, list):
        raise TypeError("circles must be a list")
    return [_triple(circle, label=f"circle {index}") for index, circle in enumerate(raw)]


def _validate_shape(circles: list[Circle], expected_n: int) -> None:
    if len(circles) != expected_n:
        raise ValueError(f"expected exactly {expected_n} circles, got {len(circles)}")
    for index, circle in enumerate(circles):
        if not all(math.isfinite(value) for value in circle):
            raise ValueError(f"circle {index} contains a non-finite value")
        if circle[2] < 0:
            raise ValueError(f"circle {index} has negative radius {circle[2]}")


def _validate_overlaps(circles: list[Circle], tol: float) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.dist(circles[i][:2], circles[j][:2])
            required = circles[i][2] + circles[j][2]
            if dist < required - tol:
                raise ValueError(
                    f"circles {i} and {j} overlap: dist={dist:.12g}, radii_sum={required:.12g}"
                )


def _validate_unit_square(circles: list[Circle], tol: float) -> None:
    for idx, (x, y, r) in enumerate(circles):
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            raise ValueError(f"circle {idx} is outside the unit square")


def _validate_rectangle_perimeter4(circles: list[Circle], tol: float) -> tuple[float, float]:
    min_x = min(x - radius for x, _, radius in circles)
    max_x = max(x + radius for x, _, radius in circles)
    min_y = min(y - radius for _, y, radius in circles)
    max_y = max(y + radius for _, y, radius in circles)
    width = max_x - min_x
    height = max_y - min_y
    if width + height > 2.0 + tol:
        raise ValueError(
            f"minimum enclosing rectangle has perimeter {2 * (width + height):.12g}, exceeding 4"
        )
    return width, height


def evaluate(
    *,
    artifact_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    dataset_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    _ = workspace_dir, task_dir, dataset_dir
    task = config.get("task") if isinstance(config.get("task"), dict) else {}
    problem = task.get("problem") if isinstance(task.get("problem"), dict) else DEFAULT_PROBLEM
    return _evaluate_solution(problem, artifact_path)


def _evaluate_problem_solution(problem_path: Path, solution_path: Path) -> dict[str, Any]:
    return _evaluate_solution(_load_problem(problem_path), solution_path)


def _evaluate_solution(problem: dict[str, Any], solution_path: Path) -> dict[str, Any]:
    expected_n = int(problem.get("num_circles", 26))
    container = str(problem.get("container", "unit_square"))
    tol = float(problem.get("tolerance", DEFAULT_TOL))
    benchmark = problem.get("benchmark_radii_sum")

    circles = _load_circles(solution_path)
    _validate_shape(circles, expected_n)
    _validate_overlaps(circles, tol)

    metadata: dict[str, Any] = {"container": container, "num_circles": expected_n}
    if container == "unit_square":
        _validate_unit_square(circles, tol)
    elif container == "rectangle_perimeter4":
        width, height = _validate_rectangle_perimeter4(circles, tol)
        metadata.update({"rectangle_width": width, "rectangle_height": height})
    else:
        raise ValueError(f"unsupported circle packing container: {container}")

    radii_sum = math.fsum(radius for _, _, radius in circles)
    metadata["radii_sum"] = radii_sum
    if benchmark is not None:
        metadata["benchmark_ratio"] = radii_sum / float(benchmark)
    return {
        "metric": {"name": "radii_sum", "value": radii_sum},
        "valid": True,
        **metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem")
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()
    try:
        problem = _load_problem(Path(args.problem)) if args.problem else DEFAULT_PROBLEM
        result = _evaluate_solution(problem, Path(args.solution))
    except (OSError, TypeError, ValueError) as exc:
        print(f"circle_packing_eval error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
