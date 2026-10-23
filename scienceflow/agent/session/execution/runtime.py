"""InquiryCraft runtime composition for a ScienceFlow agent session."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from inquirycraft.adapters import AskLLMAdapter
from inquirycraft.events import JsonlEventSink
from inquirycraft.llm import (
    LLMClient,
    SameRoundRetryPolicy,
    is_missing_reasoning_replay_error,
    normalize_reasoning_replay,
)
from inquirycraft.memory import Conversation
from inquirycraft.runtime import (
    AgentRuntime,
    AgentState,
    LLMRetryRequest,
    RuntimeOptions,
    StreamOutputGuard,
    default_same_round_retry_adapter,
)
from inquirycraft.tools import ToolCollection, ToolOutputPolicy

from scienceflow.agent.core.ports.session_callbacks import scienceflow_output_reducers
from scienceflow.agent.session.coordination.context import ScienceFlowContextProjection
from scienceflow.agent.session.coordination.contracts import AgentSessionSpec
from scienceflow.agent.session.execution.tool_result import (
    ScienceFlowAgentHooks,
    ScienceFlowTool,
)
from scienceflow.agent.session.execution.turn import ScienceFlowTurnPolicy
from scienceflow.research.control.agent_session_setup import initialize_run_session
from scienceflow.runtime.observability.agent_io.interaction_log import (
    truncate_for_interaction_log,
)
from scienceflow.runtime.observability.telemetry.agent.audit import (
    ScienceFlowProviderAudit,
)
from scienceflow.runtime.observability.telemetry.agent.audit import (
    runtime_log_dir as resolve_runtime_log_dir,
)


class ScienceFlowAgentSession:
    """Compose one runtime session while delegating each policy to its owner."""

    def __init__(self, host: Any, spec: AgentSessionSpec) -> None:
        self.host, self.spec = host, spec
        run_id = spec.run_id or spec.workspace.parent.name or "scienceflow"
        worker_id = spec.worker_id or "W00"
        session_id = spec.session_id or f"{run_id}-{worker_id}-{uuid4().hex[:12]}"
        self.control, _ = initialize_run_session(host, runtime_manages_lifecycle=True)
        source_messages = list(host.memory.messages)
        source = Conversation(source_messages)
        tools = ToolCollection(
            ScienceFlowTool(host, tool) for tool in host.availableTools
        )
        llm = host.llm if isinstance(host.llm, LLMClient) else AskLLMAdapter(host.llm)
        self._context_projection = ScienceFlowContextProjection(
            self, len(source.messages)
        )
        self._turn_policy = ScienceFlowTurnPolicy(self)
        runtime_log_dir_for_host = resolve_runtime_log_dir(host, spec.workspace)
        self._runtime_log_dir = runtime_log_dir_for_host
        self._provider_audit = ScienceFlowProviderAudit(host, runtime_log_dir_for_host)
        event_sink = JsonlEventSink(
            runtime_log_dir_for_host / "agent_runtime_events.jsonl"
        )
        operation_log = (
            runtime_log_dir_for_host
            / "agent_runtime_operations"
            / f"{session_id}.jsonl"
        )
        self.runtime = AgentRuntime(
            llm=llm,
            options=RuntimeOptions(
                model=spec.model,
                workspace=spec.workspace,
                system_prompt="",
                max_turns=spec.max_turns,
                timeout=float(host._llm_stream_timeout_sec),
                agent_id=worker_id,
                run_id=run_id,
                session_id=session_id,
                operation_log=operation_log,
                stream_llm=True,
                max_output_chars_soft=int(host._stream_max_output_chars_soft),
                preserve_empty_reasoning_content=True,
                reasoning_replay=normalize_reasoning_replay(
                    getattr(host.llm, "reasoning_replay_policy", "preserve")
                ),
            ),
            tools=tools,
            validate_tool_arguments=False,
            conversation=source,
            hooks=ScienceFlowAgentHooks(host),
            event_sink=event_sink,
            context_transformers=(self._context_projection,),
            tool_choice_policy=self._turn_policy.tool_choice,
            turn_decision_policy=self._turn_policy.decide,
            exhaustion_policy=self._turn_policy.exhaustion,
            tool_call_transformer=self._turn_policy.transform_calls,
            tool_bundle_policy=self._turn_policy.bundle_mode,
            llm_retry_policy=SameRoundRetryPolicy(
                max_attempts=int(host._llm_tool_stream_max_attempts),
                base_delay_sec=float(host._llm_tool_stream_retry_base_delay_sec),
                max_delay_sec=float(host._llm_tool_stream_retry_max_delay_sec),
            ),
            llm_retry_adapter=self._retry_request,
            provider_observer=self._provider_audit.observe,
            cancellation=host._agent_runtime_cancellation,
            stream_guard=StreamOutputGuard(
                max_output_chars_soft=int(host._stream_max_output_chars_soft),
                repetition_detection=bool(host._stream_repetition_detection),
                repetition_window_chars=int(host._stream_repetition_window_chars),
                repetition_ngram_len=int(host._stream_repetition_ngram_len),
                repetition_max_repeats=int(host._stream_repetition_max_repeats),
            ),
            tool_output_policy=ToolOutputPolicy(
                max_model_chars=max(1_200, int(host._exec_feedback_max_chars)),
                preserve_raw=False,
                include_artifact_note=False,
                always_reduce=True,
            ),
            tool_output_reducers=scienceflow_output_reducers(host),
        )
        self.runtime.context.metadata.update(
            {
                "worker_id": worker_id,
                "stage_id": spec.stage_id or "draft",
                "lineage_id": str(spec.metadata.get("lineage_id") or ""),
                "node_uid": str(spec.metadata.get("node_uid") or ""),
                "correlation": {
                    "run_id": run_id,
                    "worker_id": worker_id,
                    "session_id": session_id,
                    "stage_id": spec.stage_id or "draft",
                    **(
                        {"lineage_id": str(spec.metadata.get("lineage_id"))}
                        if spec.metadata.get("lineage_id")
                        else {}
                    ),
                    **(
                        {"node_uid": str(spec.metadata.get("node_uid"))}
                        if spec.metadata.get("node_uid")
                        else {}
                    ),
                },
                "system_msgs": host._host_ports.build_system_messages(),
                "parallel_tool_calls": bool(host._parallel_llm_tool_calls),
                "message_objects": True,
                **spec.metadata,
            }
        )

    async def run(self) -> str:
        self.host.state = AgentState.RUNNING
        if self.spec.request is not None:
            self.host._log_info(
                "[user] %s", truncate_for_interaction_log(self.spec.request)
            )
        try:
            return (
                await self.runtime.run(self.spec.request)
                if self.spec.request is not None
                else await self.runtime.resume()
            )
        finally:
            try:
                self._finish_run(write_snapshot=True)
            finally:
                await self.runtime.event_sink.aclose()

    async def resume_pending_tools_only(self) -> bool:
        self.host.state = AgentState.RUNNING
        try:
            return await self.runtime.resume_pending_tools_only()
        finally:
            try:
                self._finish_run(write_snapshot=False)
            finally:
                await self.runtime.event_sink.aclose()

    def _finish_run(self, *, write_snapshot: bool) -> None:
        self._context_projection.sync_to_host()
        self.host._sync_last_run_token_totals()
        self.host.state = AgentState.IDLE
        if write_snapshot:
            self.host._host_ports.write_productivity_snapshot()

    def _retry_request(self, request: LLMRetryRequest):
        replacement = default_same_round_retry_adapter(request)
        if replacement is not None and is_missing_reasoning_replay_error(request.error):
            self.runtime.set_reasoning_replay("required")
            self.host.llm.reasoning_replay_policy = "required"
        return replacement


__all__ = ["ScienceFlowAgentSession"]
