"""Pure prompt-context normalization; no LLM calls or workspace writes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from scienceflow.research.state.knowledge.prompt.contracts import (
    PromptContextProjection,
    PromptContextRequest,
)


def _text(value: object) -> str:
    return str(value or "")


class PromptContextBuilder:
    def project(self, request: PromptContextRequest) -> PromptContextProjection:
        payload = {
            "task_description": _text(request.task_description),
            "worker_id": _text(request.worker_id),
            "wall_clock_budget_sec": max(0, int(request.wall_clock_budget_sec or 0)),
            "seed": request.seed,
            "workspace_facts": _text(request.workspace_facts),
            "memory_projection": _text(request.memory_projection),
            "parallel_worker_facts": _text(request.parallel_worker_facts),
            "resource_observation": _text(request.resource_observation),
            "gate_constraints": _text(request.gate_constraints),
            "estra_context": _text(request.estra_context),
            "skill_context": _text(request.skill_context),
            "tool_output_policy": _text(request.tool_output_policy),
            "task_profile": _text(request.task_profile) or "mlebench",
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return PromptContextProjection(**payload, projection_hash=digest)

    @staticmethod
    def to_dict(projection: PromptContextProjection) -> dict[str, object]:
        return asdict(projection)


__all__ = ["PromptContextBuilder"]
