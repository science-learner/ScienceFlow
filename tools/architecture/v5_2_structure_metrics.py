#!/usr/bin/env python3
"""Measure V5.2 structure debt and emit the binding owner matrix."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class FacadeSpec:
    path: str
    class_name: str
    owner_map: dict[str, str]


FACADE_SPECS = (
    FacadeSpec(
        "scienceflow/research/solver/lnr/orchestration/coordinator/solver.py",
        "LnrSolver",
        {
            "construction": "composition",
            "agent_coordination": "composition.agent",
            "run_coordination": "run",
            "stage_adapter": "stage.capture",
            "stage_commit": "stage.commit",
            "stage_prompt": "stage.prompt",
            "evaluation_adapter": "evaluation.adjudication",
            "evaluation_profile": "evaluation.profile",
            "evaluation_runtime": "evaluation.runtime",
            "metric_projection": "evaluation.metrics",
            "context_coordination": "context",
            "memory_coordination": "context.memory",
            "memory_projection": "context.projection",
            "estra_adapter": "estra",
            "resource_advisory": "resource.advisory",
            "resource_construction": "resource.composition",
            "resource_observer_construction": "resource.composition",
            "resource_prompt_projection": "resource.projection",
            "event_projection": "split_required.event_workspace_lineage",
        },
    ),
    FacadeSpec(
        "scienceflow/research/solver/lnr/resources/runtime/observer/controller.py",
        "LHRResourceObserver",
        {
            "construction": "observer.composition",
            "construction_runtime": "observer.composition",
            "admission_bridge": "admission",
            "preflight": "admission.preflight",
            "lease_bridge": "lease.callbacks",
            "job_adapter": "observer.callbacks.job",
            "cleanup": "observer.callbacks.lifecycle",
            "monitor_bridge": "observation.monitor",
            "observation_bridge": "observation.projection",
            "execution_value_bridge": "observation.execution_value",
            "review_bridge": "review.machine",
            "arbiter_bridge": "review.arbiter",
            "safety_priority": "review.priority",
            "intervention_bridge": "review.intervention",
            "safety_bridge": "review.safety",
            "sharing_bridge": "sharing",
            "feedback_bridge": "observer.projection.feedback",
        },
    ),
    FacadeSpec(
        "scienceflow/research/solver/lnr/resources/runtime/execution/facade.py",
        "ResourceRuntime",
        {
            "state_policy": "runtime.state",
            "admission_coordination": "admission",
            "wait_coordination": "admission.wait",
            "queue_coordination": "admission.queue",
            "pressure_coordination": "sharing.pressure",
            "lease_coordination": "lease",
            "monitoring": "observation.monitor",
        },
    ),
    FacadeSpec(
        "scienceflow/agent/core/runtime/agent.py",
        "ScienceAgent",
        {
            "construction": "agent.factory.builder",
            "path_hygiene": "state.workspace",
            "workspace_session": "state.workspace",
            "tool_feedback": "observability.agent_feedback",
            "runtime_state": "control.agent_session",
            "edit_guard": "safety.agent_runtime",
            "compaction": "state.memory",
            "embedded_fullrun": "quality.fullrun",
            "routing": "agent.routing",
            "system_prompt": "agent.prompt",
            "write_coaching": "agent.prompt",
            "agent_hooks": "solver.lnr.agent_hooks",
        },
    ),
)

PACKAGE_ROOTS = (
    "scienceflow/research/solver/lnr/orchestration/coordinator",
    "scienceflow/research/solver/lnr/resources/runtime/observer",
    "scienceflow/research/solver/lnr/resources/runtime/runtime_components",
    "scienceflow/agent",
    "scienceflow/research/quality/finalization",
    "scienceflow/research/quality/fullrun",
    "scienceflow/runtime/safety/tooling/bash",
    "scienceflow/research/state/dataset",
)

COMPLEXITY_TARGETS = PACKAGE_ROOTS + (
    "scienceflow/research/quality/embedded_fullrun.py",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"class not found: {class_name} in {path}")


def _binding_rows(spec: FacadeSpec) -> list[dict[str, Any]]:
    path = REPO_ROOT / spec.path
    node = _class_node(path, spec.class_name)
    rows: list[dict[str, Any]] = []
    for statement in node.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not isinstance(statement.value, (ast.Attribute, ast.Name)):
            continue
        targets = [target.id for target in statement.targets if isinstance(target, ast.Name)]
        if not targets:
            continue
        implementation = ast.unparse(statement.value)
        source_alias = implementation.split(".", 1)[0]
        target_owner = spec.owner_map.get(source_alias, "public_or_local_entry")
        for target in targets:
            rows.append(
                {
                    "facade": spec.class_name,
                    "facade_path": spec.path,
                    "symbol": target,
                    "implementation": implementation,
                    "target_owner": target_owner,
                    "visibility": "private" if target.startswith("_") else "public",
                    "action": "retain_public_entry" if target_owner == "public_or_local_entry" else "migrate_then_delete_binding",
                }
            )
    return rows


def _is_sys_modules_proxy(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            value = target.value
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "sys"
                and value.attr == "modules"
            ):
                return True
    return False


def _package_metrics(relative: str) -> dict[str, Any]:
    root = REPO_ROOT / relative
    direct = sorted(path for path in root.glob("*.py") if path.name != "__init__.py")
    recursive = sorted(root.rglob("*.py"))
    top_level_functions = 0
    top_level_classes = 0
    proxy_modules: list[str] = []
    for path in recursive:
        tree = _tree(path)
        top_level_functions += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body
        )
        top_level_classes += sum(isinstance(node, ast.ClassDef) for node in tree.body)
        if _is_sys_modules_proxy(tree):
            proxy_modules.append(str(path.relative_to(REPO_ROOT)))
    return {
        "path": relative,
        "direct_business_modules": len(direct),
        "python_files_recursive": len(recursive),
        "lines_recursive": sum(len(path.read_text(encoding="utf-8").splitlines()) for path in recursive),
        "top_level_functions": top_level_functions,
        "top_level_classes": top_level_classes,
        "shared_modules": [
            str(path.relative_to(REPO_ROOT))
            for path in recursive
            if path.name in {"shared.py", "common.py", "utils.py"}
        ],
        "sys_modules_proxies": proxy_modules,
    }


def _test_seams() -> dict[str, int]:
    tests_root = REPO_ROOT / "tests"
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tests_root.rglob("*.py")
    )
    return {
        "object_new_lnr_solver": len(re.findall(r"object\.__new__\(LnrSolver\)", text)),
        "solver_private_attribute_access": len(re.findall(r"\bsolver\._[A-Za-z0-9_]+", text)),
        "observer_private_attribute_access": len(re.findall(r"\bobserver\._[A-Za-z0-9_]+", text)),
    }


def _ruff_complexity() -> dict[str, Any]:
    executable = shutil.which("ruff")
    sibling = Path(sys.executable).with_name("ruff")
    if executable is None and sibling.is_file():
        executable = str(sibling)
    if executable is None:
        return {"available": False, "violations": []}
    completed = subprocess.run(
        [
            executable,
            "check",
            "--select",
            "C901",
            "--output-format",
            "json",
            *COMPLEXITY_TARGETS,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        violations = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return {
            "available": True,
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "violations": [],
        }
    return {
        "available": True,
        "count": len(violations),
        "violations": [
            {
                "path": _relative_display_path(item.get("filename")),
                "line": (item.get("location") or {}).get("row"),
                "message": item.get("message"),
            }
            for item in violations
        ],
    }


def _relative_display_path(raw: object) -> str:
    path = Path(str(raw or ""))
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_report(*, include_complexity: bool) -> dict[str, Any]:
    bindings = [row for spec in FACADE_SPECS for row in _binding_rows(spec)]
    by_facade = {
        spec.class_name: sum(row["facade"] == spec.class_name for row in bindings)
        for spec in FACADE_SPECS
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "packages": [_package_metrics(relative) for relative in PACKAGE_ROOTS],
        "callable_descriptor_bindings": by_facade,
        "test_seams": _test_seams(),
    }
    if include_complexity:
        report["ruff_c901"] = _ruff_complexity()
    return report


def write_owner_matrix(path: Path) -> None:
    rows = [row for spec in FACADE_SPECS for row in _binding_rows(spec)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complexity", action="store_true", help="include Ruff C901 output")
    parser.add_argument("--owner-matrix", type=Path, help="write facade binding owner CSV")
    parser.add_argument("--output", type=Path, help="write JSON report instead of stdout")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    report = build_report(include_complexity=args.complexity)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.owner_matrix:
        write_owner_matrix(args.owner_matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
