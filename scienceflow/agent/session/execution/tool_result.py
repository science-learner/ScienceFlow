"""Tool execution and post-result callbacks for an agent session."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inquirycraft.memory import coerce_tool_call
from inquirycraft.runtime import AgentHooks, RuntimeContext, ToolCommitObservation
from inquirycraft.tools import (
    BaseTool,
    ToolContext,
    ToolPostprocessDecision,
    ToolResult,
    coerce_tool_result,
)

from scienceflow.agent.core.ports.session_callbacks import ScienceFlowPostCommitAdapter
from scienceflow.runtime.observability.agent_io.interaction_log import (
    format_tool_call_lines_for_interaction_log,
    tool_result_text_for_interaction_log,
)


class ScienceFlowTool(BaseTool):
    """Adapt one ScienceFlow host tool to InquiryCraft's typed tool contract."""

    def __init__(self, host: Any, tool: Any) -> None:
        self.host, self.tool = host, tool
        self.name = str(tool.name)
        self.description = str(tool.description)
        schemas = {
            row["function"]["name"]: row["function"]["parameters"]
            for row in getattr(host, "_tools_with_thought", ())
        }
        self.input_schema = dict(
            schemas.get(self.name) or getattr(tool, "parameters", None) or {}
        )
        self.replay_policy = getattr(tool, "replay_policy", "never")

    async def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolResult:
        args = self.host._host_ports.normalize_tool_input_paths(
            self.name, dict(arguments)
        )
        args.pop("thought", None)
        if self.name == "bash":
            result = await self.host.availableTools.execute(
                name=self.name, tool_input=args
            )
        else:
            result = await self.host._host_ports.execute_tool_with_edit_guard(
                self.name, args
            )
        return coerce_tool_result(
            self.host._host_ports.rewrite_tool_result_paths(result)
        )


class ScienceFlowLLMAuditHooks(AgentHooks):
    """Annotate every ScienceFlow LLM request with stable audit hashes."""

    def __init__(self, host: Any) -> None:
        self.host = host

    async def before_llm(self, request: Any, context: RuntimeContext) -> None:
        from scienceflow.agent.session.coordination.context import canonical_hash

        system_messages = list(request.metadata.get("system_msgs") or ())
        prefix = []
        for message in request.messages:
            if message.get("role") != "system":
                break
            prefix.append(dict(message))
        stable = {
            "system_messages": system_messages,
            "messages": prefix,
            "tools": list(request.tools),
        }
        provider = {
            "system_messages": system_messages,
            "messages": list(request.messages),
            "tools": list(request.tools),
            "tool_choice": request.tool_choice,
            "model": request.model,
        }
        if isinstance(request.metadata, dict):
            request.metadata.update(
                {
                    "stable_prefix_hash": canonical_hash(stable),
                    "provider_context_hash": canonical_hash(provider),
                    "provider_message_count": len(system_messages)
                    + len(request.messages),
                    "compaction_generation": int(
                        context.metadata.get("compaction_generation", 0) or 0
                    ),
                }
            )


class ScienceFlowAgentHooks(ScienceFlowLLMAuditHooks):
    """Preserve ScienceFlow callback timing around InquiryCraft's sole loop."""

    def __init__(self, host: Any) -> None:
        super().__init__(host)
        self.round_tools: list[str] = []
        self.post_commit = ScienceFlowPostCommitAdapter(host)

    async def before_round(self, context: RuntimeContext) -> None:
        round_index = context.turn_id - 1
        self.host._current_round = round_index
        self.host._effective_max_steps = max(
            int(self.host._effective_max_steps), context.turn_id
        )
        self.host._host_ports.log_iteration_header(context.turn_id)
        self.host._host_ports.fresh_workspace_hint(round_index)
        self.host._host_ports.periodic_round_injection(round_index)
        self.host._host_ports.sync_resource_state_summary()

    async def after_llm(self, result: Any, context: RuntimeContext) -> None:
        if result.tool_calls:
            self.host._tool_bundle_deferred_messages = []
        if not result.tool_calls or not bool(
            getattr(self.host, "_parallel_bash_enabled", False)
        ):
            return
        first = coerce_tool_call(result.tool_calls[0]).function.name
        display_round = (
            context.turn_id
            - 1
            + int(getattr(self.host, "_search_round_offset", 0) or 0)
        )
        self.host._log_info(
            "[tool-calls-count] round=%d n=%d first=%s",
            display_round,
            len(result.tool_calls),
            first or "?",
        )

    async def before_tool(self, call: Any, context: RuntimeContext) -> None:
        args = dict(call.arguments)
        self.host._host_ports.pop_and_log_thought(args)
        for line in format_tool_call_lines_for_interaction_log(
            call.name,
            args,
            policy=self.host._interaction_log_policy,
        ):
            self.host._log_info("%s", line)

    async def after_tool(
        self, call: Any, result: Any, context: RuntimeContext
    ) -> ToolPostprocessDecision:
        tool_result = coerce_tool_result(result)
        args = self.host._host_ports.normalize_tool_input_paths(
            call.name, dict(call.arguments)
        )
        self.host._log_info(
            "[tool-result] %s exit=%s %s",
            call.name,
            "err" if tool_result.error else "ok",
            tool_result_text_for_interaction_log(
                call.name,
                str(tool_result),
                self.host._interaction_log_policy,
            ),
        )
        self.host._host_ports.workspace_checkpoint(call.name, tool_result)
        if not tool_result.error and call.name == "read":
            self.host._host_ports.record_read_for_edit_guard(
                str(args.get("path") or "")
            )
        self.round_tools.append(call.name)
        return ToolPostprocessDecision()

    async def after_tool_commit(
        self, observation: ToolCommitObservation, context: RuntimeContext
    ):
        return await self.post_commit(observation, context)

    async def after_round(self, context: RuntimeContext) -> None:
        self.host._host_ports.round_complete(self.round_tools)
        self.round_tools = []

    async def should_continue(self, context: RuntimeContext) -> bool:
        guard = getattr(self.host, "_guard_manager", None)
        return not bool(guard and guard.should_terminate_run())

    async def stop_output(self, context: RuntimeContext) -> str | None:
        guard = getattr(self.host, "_guard_manager", None)
        if not bool(guard and guard.should_terminate_run()):
            return None
        message = "[ScienceAgent] Run stopped early by no-progress hardstop."
        self.host._log_info("[policy] %s", message)
        return message


__all__ = [
    "ScienceFlowAgentHooks",
    "ScienceFlowLLMAuditHooks",
    "ScienceFlowTool",
]
