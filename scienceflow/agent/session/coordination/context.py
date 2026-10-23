"""Conversation projection and request-context ownership."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from inquirycraft.runtime import ContextProjectionExit, RuntimeContext

from scienceflow.agent.core.runtime.run_policy import RoundContext
from scienceflow.research.state.knowledge.memory.context.agent_context_projection import (
    prepare_context_for_round,
)


class SessionProjectionPort(Protocol):
    host: Any
    spec: Any
    control: Any
    runtime: Any


def deadline_capped_llm_timeout(host: Any, policy: Any) -> float:
    configured = max(1.0, float(getattr(host, "_llm_stream_timeout_sec", 300) or 300))
    deadline = getattr(policy, "deadline_monotonic", None)
    if deadline is None:
        return configured
    remaining = max(1.0, float(deadline) - time.monotonic())
    attempts = max(1, int(getattr(host, "_llm_tool_stream_max_attempts", 1) or 1))
    return min(configured, max(1.0, remaining / attempts))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value), ensure_ascii=False, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _canonical_value(value.model_dump(exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


class ScienceFlowContextProjection:
    """Project runtime conversation through ScienceFlow's memory/compact policy."""

    def __init__(self, session: SessionProjectionPort, initial_count: int) -> None:
        self.session = session
        self.synced_count = initial_count
        self.compaction_generation = 0

    async def transform(
        self,
        _messages: Sequence[Mapping[str, Any]],
        context: RuntimeContext,
    ) -> Sequence[Mapping[str, Any]]:
        self.sync_to_host()
        self.session.control.round_index = max(0, context.turn_id - 1)
        policy = getattr(self.session.host, "_run_policy", None)
        context.metadata["request_timeout"] = deadline_capped_llm_timeout(
            self.session.host, policy
        )
        on_round_start = getattr(policy, "on_round_start", None)
        if callable(on_round_start):
            round_context = RoundContext(
                self.session.control.round_index,
                self.session.control.effective_max_steps,
                "",
                self.session.spec.workspace,
            )
            if not bool(on_round_start(round_context)):
                if not bool(
                    getattr(self.session.host, "_lnr_stage_commit_text_pending", False)
                ):
                    stage_id = str(context.metadata.get("stage_id") or "draft")
                    self.session.host._scienceflow_worker_search_outcome = (
                        "finalize_candidate"
                        if stage_id != "draft"
                        else "stop_no_candidate"
                    )
                raise ContextProjectionExit(
                    "[ScienceAgent] Wall-clock deadline reached.",
                    "scienceflow_wall_clock_deadline",
                )
        compacted_before = bool(getattr(self.session.host, "_mid_run_compacted", False))
        prepared = await prepare_context_for_round(
            self.session.host,
            self.session.control,
            self.session.spec.request,
        )
        if (
            bool(getattr(self.session.host, "_mid_run_compacted", False))
            and not compacted_before
        ):
            self.compaction_generation += 1
        context.metadata["compaction_generation"] = self.compaction_generation
        if prepared.early_result is not None:
            raise ContextProjectionExit(
                str(prepared.early_result), "scienceflow_context_early_result"
            )
        projected = list(prepared.messages)
        return tuple(message.to_dict() for message in projected)

    def sync_to_host(self) -> None:
        runtime = getattr(self.session, "runtime", None)
        if runtime is None:
            return
        messages = runtime.conversation.messages
        for message in messages[self.synced_count :]:
            self.session.host.memory.add_message(message)
        self.synced_count = len(messages)
        pending = getattr(self.session.host, "_tool_bundle_deferred_messages", None)
        if isinstance(pending, list):
            for message in pending:
                self.session.host.memory.add_message(message)
            delattr(self.session.host, "_tool_bundle_deferred_messages")


__all__ = [
    "ScienceFlowContextProjection",
    "canonical_hash",
    "deadline_capped_llm_timeout",
]
