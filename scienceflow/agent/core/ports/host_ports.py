"""Typed capabilities consumed by the ScienceFlow agent session."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable, Protocol

from inquirycraft.tools import ToolResult

from scienceflow.agent.policies import routing
from scienceflow.agent.prompts import system_prompt
from scienceflow.research.control import agent_runtime_state
from scienceflow.runtime.observability import agent_tool_feedback
from scienceflow.research.quality import embedded_fullrun
from scienceflow.runtime.safety.execution.agent_runtime import edit_guard
from scienceflow.runtime.safety.execution.agent_runtime.tool_bundle_policy import (
    normalize_tool_calls_for_execution,
)
from scienceflow.research.solver.lnr.orchestration import agent_hooks
from scienceflow.research.state.workspace.adapters import path_hygiene


class PrepareToolFeedbackPort(Protocol):
    def __call__(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_result: ToolResult,
        *,
        guard_coaching: str = "",
    ) -> str: ...


WorkspaceCheckpointPort = Callable[[str, ToolResult], None]
NormalizeToolInputPort = Callable[[str, dict[str, Any]], dict[str, Any]]
RewriteToolResultPort = Callable[[ToolResult], ToolResult]
ExecuteToolPort = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]
RecordReadPort = Callable[[str], None]
RoundIndexPort = Callable[[int], None]
RoundCompletePort = Callable[[list[str] | None], None]
NoArgumentPort = Callable[[], None]
BuildSystemMessagesPort = Callable[[], list[Any]]
ToolChoicePort = Callable[[int, str | None], str]
NormalizeCallsPort = Callable[[list[Any]], Any]
EmbeddedToolResultPort = Callable[[dict[str, Any], ToolResult], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class AgentHostPorts:
    """Narrow, explicit host capabilities used by the IQ runtime adapter."""

    workspace_checkpoint: WorkspaceCheckpointPort
    normalize_tool_input_paths: NormalizeToolInputPort
    rewrite_tool_result_paths: RewriteToolResultPort
    execute_tool_with_edit_guard: ExecuteToolPort
    record_read_for_edit_guard: RecordReadPort
    prepare_tool_feedback_for_memory: PrepareToolFeedbackPort
    log_iteration_header: RoundIndexPort
    pop_and_log_thought: Callable[[dict[str, Any]], None]
    fresh_workspace_hint: RoundIndexPort
    periodic_round_injection: RoundIndexPort
    round_complete: RoundCompletePort
    sync_resource_state_summary: NoArgumentPort
    write_productivity_snapshot: NoArgumentPort
    build_system_messages: BuildSystemMessagesPort
    tool_choice_for_main_loop: ToolChoicePort
    normalize_repl_file_change_calls: NormalizeCallsPort
    normalize_tool_calls: NormalizeCallsPort
    maybe_write_bare_run_tail_snapshot: EmbeddedToolResultPort
    maybe_embedded_full_run_after_quick_test: EmbeddedToolResultPort


def compose_agent_host_ports(host: Any) -> AgentHostPorts:
    """Bind real domain owners once at the ScienceAgent composition boundary."""

    return AgentHostPorts(
        workspace_checkpoint=partial(
            path_hygiene._maybe_workspace_git_auto_checkpoint, host
        ),
        normalize_tool_input_paths=partial(
            path_hygiene._maybe_normalize_tool_input_paths, host
        ),
        rewrite_tool_result_paths=partial(
            path_hygiene._maybe_rewrite_tool_result_paths, host
        ),
        execute_tool_with_edit_guard=partial(
            edit_guard._execute_tool_maybe_edit_guard, host
        ),
        record_read_for_edit_guard=partial(
            edit_guard._record_read_for_edit_guard, host
        ),
        prepare_tool_feedback_for_memory=partial(
            agent_tool_feedback._prepare_tool_feedback_for_memory, host
        ),
        log_iteration_header=partial(agent_tool_feedback._log_iteration_header, host),
        pop_and_log_thought=partial(agent_tool_feedback._pop_and_log_thought, host),
        fresh_workspace_hint=partial(
            agent_hooks._lnr_maybe_fresh_workspace_hint_first_round, host
        ),
        periodic_round_injection=partial(
            agent_hooks._lnr_maybe_periodic_inject_at_round_start, host
        ),
        round_complete=partial(agent_hooks._lnr_on_round_complete, host),
        sync_resource_state_summary=partial(
            routing._sync_resource_state_summary_slot, host
        ),
        write_productivity_snapshot=partial(
            agent_runtime_state._write_productivity_snapshot, host
        ),
        build_system_messages=partial(system_prompt._build_system_messages, host),
        tool_choice_for_main_loop=partial(routing._tool_choice_for_main_loop, host),
        normalize_repl_file_change_calls=partial(
            routing._normalize_repl_file_change_tool_calls, host
        ),
        normalize_tool_calls=partial(normalize_tool_calls_for_execution, host),
        maybe_write_bare_run_tail_snapshot=partial(
            embedded_fullrun._maybe_write_bare_run_tail_snapshot, host
        ),
        maybe_embedded_full_run_after_quick_test=partial(
            embedded_fullrun._maybe_embedded_full_run_after_quick_test, host
        ),
    )


__all__ = ["AgentHostPorts", "compose_agent_host_ports"]
