"""Versioned prompt-context input and projection contracts."""

from __future__ import annotations

from dataclasses import dataclass

from scienceflow.foundation.contracts.runtime.base import VersionedContract


@dataclass(frozen=True, slots=True)
class PromptContextRequest(VersionedContract):
    task_description: str
    worker_id: str
    wall_clock_budget_sec: int
    seed: int | None = None
    workspace_facts: str = ""
    memory_projection: str = ""
    parallel_worker_facts: str = ""
    resource_observation: str = ""
    gate_constraints: str = ""
    estra_context: str = ""
    skill_context: str = ""
    tool_output_policy: str = ""
    task_profile: str = "mlebench"


@dataclass(frozen=True, slots=True)
class PromptContextProjection(VersionedContract):
    task_description: str
    worker_id: str
    wall_clock_budget_sec: int
    seed: int | None
    workspace_facts: str
    memory_projection: str
    parallel_worker_facts: str
    resource_observation: str
    gate_constraints: str
    estra_context: str
    skill_context: str
    tool_output_policy: str
    task_profile: str
    projection_hash: str


__all__ = ["PromptContextProjection", "PromptContextRequest"]
