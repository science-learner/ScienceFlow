from __future__ import annotations

from typing import Any

from .normalize import normalize_value


def strict_differences(
    expected: Any, actual: Any, *, limit: int = 100
) -> list[dict[str, Any]]:
    """Return deterministic structural differences without silently ignoring fields."""
    differences: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, path: str) -> None:
        if len(differences) >= limit:
            return
        if type(left) is not type(right):
            differences.append(
                {
                    "path": path,
                    "kind": "type",
                    "expected": type(left).__name__,
                    "actual": type(right).__name__,
                }
            )
            return
        if isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                child_path = f"{path}.{key}"
                if key not in left:
                    differences.append({"path": child_path, "kind": "additional"})
                elif key not in right:
                    differences.append({"path": child_path, "kind": "missing"})
                else:
                    visit(left[key], right[key], child_path)
                if len(differences) >= limit:
                    return
            return
        if isinstance(left, list):
            if len(left) != len(right):
                differences.append(
                    {
                        "path": path,
                        "kind": "length",
                        "expected": len(left),
                        "actual": len(right),
                    }
                )
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
                if len(differences) >= limit:
                    return
            return
        if left != right:
            differences.append(
                {"path": path, "kind": "value", "expected": left, "actual": right}
            )

    visit(expected, actual, "$")
    return differences


def compare_capture(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    # Baselines are immutable captures. Reapplying the current, explicitly approved
    # normalization registry keeps old null timing values and newly observed timing
    # values equivalent without rewriting the frozen evidence.
    expected = normalize_value(expected)
    actual = normalize_value(actual)
    sections: dict[str, Any] = {}
    total = 0
    for name in ("context", "mechanism", "workspace"):
        differences = strict_differences(expected.get(name), actual.get(name))
        sections[name] = {"diff_count": len(differences), "differences": differences}
        total += len(differences)
    return {"compatible": total == 0, "diff_count": total, "sections": sections}


def compare_performance_capture(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    regression_ratio: float,
    median_floor_ms: float,
    p95_floor_ms: float,
) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    total = 0
    for name in ("context", "workspace"):
        differences = strict_differences(expected.get(name), actual.get(name))
        sections[name] = {"diff_count": len(differences), "differences": differences}
        total += len(differences)
    differences: list[dict[str, Any]] = []
    expected_metrics = expected.get("mechanism", {}).get("metrics", {})
    actual_metrics = actual.get("mechanism", {}).get("metrics", {})
    if set(expected_metrics) != set(actual_metrics):
        differences.extend(strict_differences(expected_metrics, actual_metrics))
    else:
        for metric_name in sorted(expected_metrics):
            for field, floor in (
                ("median_ms", median_floor_ms),
                ("p95_ms", p95_floor_ms),
            ):
                baseline = float(expected_metrics[metric_name][field])
                candidate = float(actual_metrics[metric_name][field])
                limit = max(floor, baseline * (1.0 + regression_ratio))
                if candidate > limit:
                    differences.append(
                        {
                            "path": f"$.metrics.{metric_name}.{field}",
                            "kind": "performance_regression",
                            "expected": baseline,
                            "actual": candidate,
                            "limit": limit,
                        }
                    )
    sections["mechanism"] = {
        "diff_count": len(differences),
        "differences": differences,
        "metrics": actual_metrics,
    }
    total += len(differences)
    return {"compatible": total == 0, "diff_count": total, "sections": sections}


__all__ = ["compare_capture", "compare_performance_capture", "strict_differences"]
