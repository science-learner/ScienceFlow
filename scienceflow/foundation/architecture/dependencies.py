# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""AST-based import boundary checks for the modular ScienceFlow core."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class DependencyRule:
    name: str
    source: str
    forbidden: tuple[str, ...]
    exclude_parts: tuple[str, ...] = ("adapters",)


@dataclass(frozen=True, slots=True)
class DependencyViolation:
    rule: str
    path: Path
    line: int
    imported: str

    def format(self, root: Path) -> str:
        try:
            display = self.path.relative_to(root)
        except ValueError:
            display = self.path
        return f"{display}:{self.line}: [{self.rule}] forbidden import {self.imported}"


_DOMAIN_MODULES = (
    "scienceflow.research.control.admission",
    "scienceflow.research.quality.assessment",
    "scienceflow.research.quality.evaluator",
    "scienceflow.research.control.execution_value",
    "scienceflow.research.quality.gate",
    "scienceflow.research.state.knowledge.memory",
    "scienceflow.research.control.resources",
    "scienceflow.research.control.estra",
    "scienceflow.research.state.workspace",
)


DEFAULT_RULES = (
    DependencyRule(
        name="contracts-are-leaves",
        source="contracts",
        forbidden=("scienceflow",),
        exclude_parts=(),
    ),
    DependencyRule(
        name="evaluator-only-measures",
        source="quality/evaluator",
        forbidden=tuple(item for item in _DOMAIN_MODULES if item != "scienceflow.research.quality.evaluator"),
    ),
    DependencyRule(
        name="gate-only-decides",
        source="quality/gate",
        forbidden=tuple(
            item
            for item in _DOMAIN_MODULES
            if item != "scienceflow.research.quality.gate"
        ),
    ),
    DependencyRule(
        name="workspace-does-not-own-memory",
        source="state/workspace",
        forbidden=("scienceflow.research.state.knowledge.memory", "scienceflow.research.solver"),
    ),
    DependencyRule(
        name="memory-is-a-projection",
        source="state/memory",
        forbidden=("scienceflow.research.state.workspace", "scienceflow.research.solver"),
    ),
    DependencyRule(
        name="estra-produces-commands",
        source="control/estra",
        forbidden=tuple(
            item for item in _DOMAIN_MODULES if item != "scienceflow.research.control.estra"
        ) + ("scienceflow.research.solver",),
        exclude_parts=(),
    ),
    DependencyRule(
        name="admission-is-pure-policy",
        source="control/admission",
        forbidden=(
            "scienceflow.research.quality.assessment",
            "scienceflow.research.quality.evaluator",
            "scienceflow.research.control.execution_value",
            "scienceflow.research.quality.gate",
            "scienceflow.research.state.knowledge.memory",
            "scienceflow.research.control.estra",
            "scienceflow.research.state.workspace",
            "scienceflow.research.control.resources.policy.effects",
            "scienceflow.research.control.resources.policy.mechanisms",
            "scienceflow.research.control.resources.runtime.service",
            "scienceflow.research.solver",
        ),
        exclude_parts=(),
    ),
    DependencyRule(
        name="resource-management-does-not-judge-value",
        source="control/resources",
        forbidden=(
            "scienceflow.research.quality.assessment",
            "scienceflow.research.control.admission",
            "scienceflow.research.quality.evaluator",
            "scienceflow.research.control.execution_value",
            "scienceflow.research.quality.gate",
            "scienceflow.research.state.knowledge.memory",
            "scienceflow.research.state.workspace",
        ),
    ),
    DependencyRule(
        name="execution-value-does-not-control-resources",
        source="control/execution_value",
        forbidden=(
            "scienceflow.research.control.resources",
            "scienceflow.research.solver.lnr.resources.runtime",
        ),
        exclude_parts=(),
    ),
    DependencyRule(
        name="finalization-does-not-own-runtime-or-policy",
        source="quality/finalization",
        forbidden=(
            "scienceflow.research.control.admission",
            "scienceflow.research.control.execution_value",
            "scienceflow.research.state.knowledge.memory",
            "scienceflow.research.control.resources",
            "scienceflow.research.control.estra",
            "scienceflow.research.state.workspace",
            "scienceflow.research.solver",
        ),
        exclude_parts=(),
    ),
    DependencyRule(
        name="prompt-context-is-pure-projection",
        source="state/prompt",
        forbidden=_DOMAIN_MODULES + ("scienceflow.research.solver", "scienceflow.runtime.core.kernel"),
        exclude_parts=(),
    ),
    DependencyRule(
        name="agent-factory-is-construction-only",
        source="agent/factory",
        forbidden=tuple(
            item for item in _DOMAIN_MODULES if item != "scienceflow.research.state.knowledge.memory"
        )
        + (
            "scienceflow.research.solver",
            "scienceflow.runtime.core.kernel",
            "scienceflow.runtime.observability.monitoring",
        ),
        exclude_parts=(),
    ),
    DependencyRule(
        name="telemetry-is-observation-only",
        source="observability/telemetry",
        forbidden=_DOMAIN_MODULES + ("scienceflow.research.solver", "scienceflow.runtime.safety"),
        exclude_parts=(),
    ),
    DependencyRule(
        name="runtime-has-no-domain-policy",
        source="solver/lnr/runtime",
        forbidden=(
            "scienceflow.research.quality.assessment",
            "scienceflow.research.quality.evaluator",
            "scienceflow.research.control.execution_value",
            "scienceflow.research.quality.gate",
            "scienceflow.research.state.knowledge.memory",
            "scienceflow.research.control.resources",
            "scienceflow.research.control.estra",
            "scienceflow.research.state.workspace",
        ),
        exclude_parts=(),
    ),
    DependencyRule(
        name="runtime-kernel-has-no-domain-policy",
        source="runtime/kernel",
        forbidden=_DOMAIN_MODULES
        + (
            "scienceflow.research.solver",
            "scienceflow.runtime.observability.monitoring",
        ),
        exclude_parts=(),
    ),
    DependencyRule(
        name="process-runtime-has-no-domain-policy",
        source="runtime/process",
        forbidden=_DOMAIN_MODULES
        + (
            "scienceflow.runtime.observability.monitoring",
            "scienceflow.runtime.safety",
            "scienceflow.research.solver",
        ),
    ),
    DependencyRule(
        name="monitoring-only-observes",
        source="observability/monitoring",
        forbidden=_DOMAIN_MODULES
        + (
            "scienceflow.runtime.safety",
            "scienceflow.research.solver",
        ),
        exclude_parts=(),
    ),
)


def _imports(path: Path) -> Iterable[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module


def _matches(imported: str, forbidden: str) -> bool:
    return imported == forbidden or imported.startswith(f"{forbidden}.")


def find_dependency_violations(
    package_root: str | Path,
    *,
    rules: tuple[DependencyRule, ...] = DEFAULT_RULES,
) -> list[DependencyViolation]:
    """Return deterministic violations below the ``scienceflow`` package root."""

    root = Path(package_root).resolve()
    violations: list[DependencyViolation] = []
    for rule in rules:
        source_root = root / rule.source
        source_module = f"scienceflow.{rule.source.replace('/', '.')}"
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*.py")):
            relative_parts = path.relative_to(source_root).parts[:-1]
            if any(part in relative_parts for part in rule.exclude_parts):
                continue
            for line, imported in _imports(path):
                if _matches(imported, source_module):
                    continue
                for forbidden in rule.forbidden:
                    if _matches(imported, forbidden):
                        violations.append(
                            DependencyViolation(
                                rule=rule.name,
                                path=path,
                                line=line,
                                imported=imported,
                            )
                        )
                        break
    return violations


__all__ = [
    "DEFAULT_RULES",
    "DependencyRule",
    "DependencyViolation",
    "find_dependency_violations",
]
