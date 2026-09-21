"""Materialize onboarding answers as the one existing Parallel manifest."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from scienceflow.research.onboarding.session import LongResearchDraft
from scienceflow.research.onboarding.support.task_defaults import (
    may_defer_metric_direction,
)


@dataclass(frozen=True, slots=True)
class OnboardingFiles:
    manifest_path: Path
    onboarding_path: Path


def build_parallel_manifest(draft: LongResearchDraft) -> dict[str, object]:
    """Build a standard manifest consumed directly by ``ParallelRunner``."""

    _validate_complete_draft(draft)
    exp_id = draft.exp_id.strip() or _generated_exp_id(draft.task_text)
    run_id = draft.run_id.strip() or _generated_run_id(exp_id)
    draft.exp_id = exp_id
    draft.run_id = run_id

    evaluator = dict(draft.evaluator)
    gate = dict(draft.gate)
    if not evaluator:
        evaluator = {
            "enabled": True,
            "backend": "artifact_command",
            "task_profile": "artifact_command",
            "candidate": {
                "artifact": draft.artifact_path,
                "artifact_kind": draft.artifact_kind or "artifact",
            },
            "metric": {
                "name": draft.metric_name,
                "lower_is_better": draft.lower_is_better,
                "type": "exploratory",
            },
            "command": {"evaluator_command": draft.artifact_command},
        }
    if evaluator.get("metric", {}).get("lower_is_better") is None:
        evaluator["metric"] = {**evaluator.get("metric", {}), "lower_is_better": draft.lower_is_better}
    if not gate:
        gate = {"policy": "default", "params": {"minimum_metric_validity": "high"}}

    wall_clock = int(draft.wall_clock_sec or 0)
    outer_limit = wall_clock + min(600, max(60, wall_clock // 10))
    task: dict[str, object] = {
        "exp_id": exp_id,
        "run_id": run_id,
        "task": draft.task_text.strip(),
        "input_data_dir": str(
            (Path(draft.workspace_base).parent / ".scienceflow" / "empty_inputs" / run_id
             if draft.input_data_dir == "none" else Path(draft.input_data_dir)).expanduser().resolve(strict=False)
        ),
        "time_limit": outer_limit,
        "cpu_list": draft.cpu_list,
        "gpu_list": draft.gpu_list,
        "evaluator": evaluator,
        "gate": gate,
        "lnr": {
            "wall_clock_budget_sec": wall_clock,
            "num_workers": int(draft.workers or 0),
        },
    }
    agent: dict[str, object] = {}
    if draft.code_models:
        code_stage = {
            "model_aliases": list(draft.code_models),
            "model_selection": draft.model_selection or "auto",
        }
        if draft.model_config_path:
            code_stage["model_config_path"] = draft.model_config_path
        agent["code"] = code_stage
    if draft.feedback_models:
        feedback_stage = {
            "model_aliases": list(draft.feedback_models),
            "model_selection": draft.model_selection or "auto",
        }
        if draft.model_config_path:
            feedback_stage["model_config_path"] = draft.model_config_path
        agent["feedback"] = feedback_stage
    if agent:
        task["agent"] = agent
    return {
        "max_concurrent": 1,
        "time_limit": outer_limit,
        "resume": False,
        "lnr": {"resource_control_mode": "resource_smart_policy"},
        "defaults": {
            "phase": "run",
            "type": "lnr",
            "workspace_base": draft.workspace_base,
        },
        "tasks": [task],
    }


def write_onboarding_files(
    draft: LongResearchDraft,
    workspace: str | Path,
    *,
    conversation: tuple[tuple[str, str], ...] = (),
) -> OnboardingFiles:
    """Write the canonical manifest and non-authoritative answer provenance."""

    root = Path(workspace).expanduser().resolve(strict=False)
    metadata_dir = root / ".scienceflow"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_parallel_manifest(draft)
    if draft.input_data_dir == "none":
        Path(manifest["tasks"][0]["input_data_dir"]).mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "run_manifest.yaml"
    onboarding_path = metadata_dir / "onboarding.json"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "exploratory" if draft.exploratory else "verified_candidate",
        "registered_task": draft.registered_task,
        "draft": asdict(draft),
        "source": {
            "kind": "detached_conversation",
            "message_count": len(conversation),
            "conversation_sha256": _conversation_hash(conversation),
        },
        "authority": (
            "TaskPackage metadata and preflight determine verification; "
            "this onboarding record is provenance only."
        ),
    }
    onboarding_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return OnboardingFiles(manifest_path, onboarding_path)


def _validate_complete_draft(draft: LongResearchDraft) -> None:
    missing = [
        label
        for label, value in (
            ("task_text", draft.task_text),
            ("input_data_dir", draft.input_data_dir),
            ("metric_name", draft.metric_name),
            ("artifact_path", draft.artifact_path),
            ("gpu_list", draft.gpu_list),
            ("cpu_list", draft.cpu_list),
            ("workers", draft.workers),
            ("wall_clock_sec", draft.wall_clock_sec),
        )
        if value is None or not str(value).strip()
    ]
    if draft.lower_is_better is None and not may_defer_metric_direction(draft):
        missing.append("lower_is_better")
    if not draft.registered_task and not draft.artifact_command.strip():
        missing.append("artifact_command")
    if missing:
        raise ValueError(f"onboarding draft is incomplete: {', '.join(missing)}")


def _generated_exp_id(task_text: str) -> str:
    words = re.findall(r"[a-z0-9]+", task_text.casefold())[:6]
    stem = "-".join(words) or "long-research"
    digest = hashlib.sha256(task_text.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:48].strip('-')}-{digest}"


def _generated_run_id(exp_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{exp_id}-{timestamp}-{uuid4().hex[:8]}"


def _conversation_hash(conversation: tuple[tuple[str, str], ...]) -> str:
    encoded = json.dumps(
        conversation, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["OnboardingFiles", "build_parallel_manifest", "write_onboarding_files"]
