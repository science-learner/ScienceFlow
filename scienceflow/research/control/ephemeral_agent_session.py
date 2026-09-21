"""Isolated one-turn sessions that use the same runtime as regular agents."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from inquirycraft.adapters import AskLLMAdapter
from inquirycraft.events import JsonlEventSink
from inquirycraft.llm import (
    CompletionResult,
    LLMClient,
    SameRoundRetryPolicy,
    is_missing_reasoning_replay_error,
    normalize_reasoning_replay,
)
from inquirycraft.memory import Conversation, coerce_message
from inquirycraft.runtime import (
    AgentRuntime,
    LLMRetryRequest,
    RuntimeOptions,
    StreamOutputGuard,
    TurnDecision,
    default_same_round_retry_adapter,
)

from scienceflow.agent.session.execution.tool_result import (
    ScienceFlowLLMAuditHooks,
)
from scienceflow.runtime.observability.telemetry.agent.audit import (
    ScienceFlowProviderAudit,
    runtime_log_dir,
)


class _EphemeralAuditHooks(ScienceFlowLLMAuditHooks):
    def __init__(self, host: Any) -> None:
        super().__init__(host)
        self.result: CompletionResult | None = None

    async def after_llm(self, result: CompletionResult, _context: Any) -> None:
        self.result = result


@dataclass(frozen=True, slots=True)
class EphemeralAgentResult:
    completion: CompletionResult
    correlation: dict[str, Any]

    @property
    def content(self) -> str:
        return self.completion.content

    @property
    def reasoning_content(self) -> str:
        return self.completion.reasoning_content

    @property
    def tool_calls(self) -> tuple[Any, ...]:
        return self.completion.tool_calls


class CorrelatedText(str):
    """String-compatible model output carrying its durable runtime correlation."""

    correlation: dict[str, Any]

    def __new__(cls, value: str, correlation: dict[str, Any]):
        instance = super().__new__(cls, value)
        instance.correlation = dict(correlation)
        return instance


def llm_correlation(value: Any) -> dict[str, Any]:
    return dict(getattr(value, "correlation", {}) or {})


def _identity(host: Any, trigger: str) -> tuple[str, str, str]:
    workspace = host._workspace_dir
    run_id = str(
        getattr(host, "_scienceflow_run_id", "")
        or os.environ.get("SCIENCEFLOW_RUN_ID", "")
        or workspace.parent.name
        or "scienceflow"
    )
    agent_id = str(getattr(host, "_scienceflow_worker_id", "") or "W00")
    route = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(trigger or "route")).strip("-")
    session_id = f"{run_id}-{agent_id}-{route or 'route'}-{uuid4().hex[:12]}"
    return run_id, agent_id, session_id


def configure_resource_agent_audit(owner: Any, agent: Any, role: str) -> None:
    """Assign a collision-free durable audit root to a resource control agent."""
    worker_id = str(owner._worker_uid_prefix() or "W00")
    actor_id = f"{worker_id}-{role}"
    agent._scienceflow_runtime_log_dir = (
        owner.global_log_dir / "resource" / "agent_runtime_audit" / actor_id
    )
    agent._agentic_route_log_dir_override = (
        owner.global_log_dir / "resource" / "agentic_route" / actor_id
    )
    agent._scienceflow_worker_id = actor_id
    agent._scienceflow_run_id = str(
        getattr(owner.cfg, "exp_id", "") or owner.task_root_dir.name
    )
    next_stage = getattr(owner, "_next_stage_id_for_logging", None)
    lineage = getattr(owner, "_lineage_uid_prefix", None)
    node = getattr(owner, "_stage_node_uid", None)
    stage_id = str(next_stage() if callable(next_stage) else "draft") or "draft"
    lineage_id = str(lineage() if callable(lineage) else "")
    agent._scienceflow_stage_id = stage_id
    agent._scienceflow_lineage_id = lineage_id
    agent._scienceflow_node_uid = str(
        node(stage_id, lineage_id=lineage_id) if callable(node) else ""
    )


async def run_ephemeral_science_agent(
    host: Any,
    prompt: str,
    *,
    trigger: str,
    base_messages: Sequence[Any],
    system_messages: Sequence[Any] | None = None,
    timeout: float | None = None,
    llm_override: Any | None = None,
    llm_role: str | None = None,
    stage_id: str | None = None,
    lineage_id: str | None = None,
    node_uid: str | None = None,
) -> EphemeralAgentResult:
    """Run one isolated turn with standard events, operations, and telemetry."""
    workspace = host._workspace_dir
    run_id, agent_id, session_id = _identity(host, trigger)
    log_dir = runtime_log_dir(host, workspace)
    backend = llm_override or host.llm
    audit = ScienceFlowProviderAudit(
        host,
        log_dir,
        llm_override=llm_override,
        llm_role=llm_role,
    )
    llm = backend if isinstance(backend, LLMClient) else AskLLMAdapter(backend)
    hooks = _EphemeralAuditHooks(host)
    runtime_ref: list[AgentRuntime] = []

    def retry_request(request: LLMRetryRequest):
        replacement = default_same_round_retry_adapter(request)
        if replacement is not None and is_missing_reasoning_replay_error(request.error):
            runtime_ref[0].set_reasoning_replay("required")
            backend.reasoning_replay_policy = "required"
        return replacement

    runtime = AgentRuntime(
        llm=llm,
        options=RuntimeOptions(
            model=str(getattr(backend, "model", "") or "scienceflow"),
            workspace=workspace,
            system_prompt="",
            max_turns=1,
            timeout=float(
                timeout if timeout is not None else host._llm_stream_timeout_sec
            ),
            agent_id=agent_id,
            run_id=run_id,
            session_id=session_id,
            operation_log=(
                log_dir / "agent_runtime_operations" / f"{session_id}.jsonl"
            ),
            stream_llm=True,
            max_output_chars_soft=int(host._stream_max_output_chars_soft),
            preserve_empty_reasoning_content=True,
            reasoning_replay=normalize_reasoning_replay(
                getattr(backend, "reasoning_replay_policy", "preserve")
            ),
        ),
        conversation=Conversation(coerce_message(row) for row in base_messages),
        hooks=hooks,
        event_sink=JsonlEventSink(log_dir / "agent_runtime_events.jsonl"),
        tool_choice_policy=lambda _request: "none",
        tool_call_transformer=lambda _calls, _context: (),
        turn_decision_policy=lambda _request: TurnDecision(
            action="terminate",
            reason="scienceflow_ephemeral_route",
            route=str(trigger or "route"),
        ),
        llm_retry_policy=SameRoundRetryPolicy(
            max_attempts=int(host._llm_tool_stream_max_attempts),
            base_delay_sec=float(host._llm_tool_stream_retry_base_delay_sec),
            max_delay_sec=float(host._llm_tool_stream_retry_max_delay_sec),
        ),
        llm_retry_adapter=retry_request,
        provider_observer=audit.observe,
        cancellation=host._agent_runtime_cancellation,
        stream_guard=StreamOutputGuard(
            max_output_chars_soft=int(host._stream_max_output_chars_soft),
            repetition_detection=bool(host._stream_repetition_detection),
            repetition_window_chars=int(host._stream_repetition_window_chars),
            repetition_ngram_len=int(host._stream_repetition_ngram_len),
            repetition_max_repeats=int(host._stream_repetition_max_repeats),
        ),
    )
    runtime_ref.append(runtime)
    selected_stage_id = str(
        stage_id or getattr(host, "_scienceflow_stage_id", "") or "draft"
    )
    selected_lineage_id = str(
        lineage_id or getattr(host, "_scienceflow_lineage_id", "") or ""
    )
    selected_node_uid = str(
        node_uid or getattr(host, "_scienceflow_node_uid", "") or ""
    )
    correlation = {
        "run_id": run_id,
        "worker_id": agent_id,
        "session_id": session_id,
        "stage_id": selected_stage_id,
        **({"lineage_id": selected_lineage_id} if selected_lineage_id else {}),
        **({"node_uid": selected_node_uid} if selected_node_uid else {}),
    }
    runtime.context.metadata.update(
        {
            "worker_id": agent_id,
            "stage_id": selected_stage_id,
            "lineage_id": selected_lineage_id,
            "node_uid": selected_node_uid,
            "correlation": correlation,
            "system_msgs": (
                list(system_messages)
                if system_messages is not None
                else host._host_ports.build_system_messages()
            ),
            "parallel_tool_calls": False,
            "message_objects": True,
            "force_tool_stream": True,
            "route": str(trigger or "route"),
            "call_kind": "ephemeral",
            "turn_kind": str(trigger or "route"),
            "llm_role": llm_role,
        }
    )
    try:
        await runtime.run(str(prompt))
        if hooks.result is None:
            raise RuntimeError("Ephemeral runtime completed without an LLM result")
        return EphemeralAgentResult(
            completion=hooks.result,
            correlation=dict(
                runtime.context.metadata.get("correlation") or correlation
            ),
        )
    finally:
        await runtime.event_sink.aclose()


__all__ = [
    "CorrelatedText",
    "EphemeralAgentResult",
    "configure_resource_agent_audit",
    "llm_correlation",
    "run_ephemeral_science_agent",
]
