"""Map registered TaskPackage metadata into an onboarding draft."""

from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from scienceflow.runtime.task_package import (
    TaskPackageSpec,
    default_tasks_root,
    find_task_package,
    iter_task_packages,
)

if TYPE_CHECKING:
    from scienceflow.research.onboarding.session import LongResearchDraft


def may_defer_metric_direction(draft: LongResearchDraft) -> bool:
    """Return whether runtime evidence may resolve a registered task's direction."""

    return bool(
        draft.registered_task
        and str(draft.task_profile or "").strip().casefold() == "mlebench"
    )


def resolve_task_package(exp_id: str, task_text: str, *, workspace: Path | None = None) -> TaskPackageSpec | None:
    if workspace is not None:
        from scienceflow.research.onboarding.library.catalog import resolve_local
        local = resolve_local(exp_id or task_text, workspace)
        if local is not None:
            return local
    if exp_id.strip():
        return find_task_package(exp_id.strip())
    lowered = task_text.casefold()
    matches = [
        spec for spec in iter_task_packages() if spec.task_id.casefold() in lowered
    ]
    if not matches:
        normalized = re.sub(r"[^a-z0-9]+", "", lowered)
        matches = [
            spec
            for spec in iter_task_packages()
            if re.sub(r"[^a-z0-9]+", "", spec.task_id.casefold()) in normalized
        ]
    return matches[0] if len(matches) == 1 else None


def apply_task_package(draft: LongResearchDraft, spec: TaskPackageSpec) -> None:
    description_path = spec.source_dir / spec.description_relpath
    if description_path.is_file():
        description = description_path.read_text(encoding="utf-8").strip()
        objective = draft.task_text.strip()
        if objective and objective != spec.task_id and objective != description:
            description += f"\n\nAdditional user requirements:\n{objective}"
        draft.task_text = description
        draft.description_sources = (str(description_path),)
    draft.task_root = str(spec.source_dir)
    draft.exp_id = spec.task_id
    draft.metric_name = spec.metric_name
    if spec.lower_is_better is not None:
        draft.lower_is_better = spec.lower_is_better
    draft.artifact_path = spec.artifact_path
    draft.artifact_kind = spec.artifact_kind
    draft.task_profile = spec.profile
    draft.registered_task = True
    draft.exploratory = not spec.metric_authoritative
    draft.evaluator = {
        "enabled": True,
        "backend": "task_package",
        "task_profile": spec.profile,
        "candidate": {
            "artifact": spec.artifact_path,
            "artifact_kind": spec.artifact_kind,
        },
        "metric": {
            "name": spec.metric_name,
            "type": spec.metric_type,
            "lower_is_better": draft.lower_is_better,
        },
    }
    if not spec.source_dir.is_relative_to(default_tasks_root()):
        draft.evaluator['package_source'] = str(spec.source_dir)
    command = (spec.config.get('evaluator') or {}).get('command')
    if command:
        draft.artifact_command = str(command).replace('{task_dir}', shlex.quote(str(spec.source_dir))).replace('{python}', shlex.quote(sys.executable))
        draft.evaluator.update(backend='artifact_command', command={'evaluator_command': draft.artifact_command})
        draft.evaluator['metric']['json_path'] = 'metric'
    if (spec.config.get('inputs') or {}).get('required') is False:
        draft.input_data_dir = 'none'
    gate_config = spec.config.get("gate")
    if isinstance(gate_config, Mapping):
        draft.gate = {
            "policy": str(gate_config.get("policy") or "default"),
            "params": dict(gate_config.get("params") or {}),
        }
    else:
        draft.gate = {
            "policy": "default",
            "params": {"minimum_metric_validity": "high"},
        }


__all__ = [
    "apply_task_package",
    "may_defer_metric_direction",
    "resolve_task_package",
]
