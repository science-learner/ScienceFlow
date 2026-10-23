"""ScienceFlow tool-bundle classification policy for InquiryCraft execution."""

from __future__ import annotations

from typing import Any

from inquirycraft.tools import classify_tool_call_bundle
from inquirycraft.memory import Message

from scienceflow.research.state.knowledge.memory.agent.memory_utils import _tool_call_names_from_list
from scienceflow.runtime.safety.policy.agent_policies.bash_utils import _bash_command_parallel_safe
from scienceflow.foundation.config.runtime.agent_constants import PARALLEL_READONLY_TOOLS

FILE_MUTATION_TOOLS = {"write", "edit"}


def add_message_after_current_tool_bundle(host: Any, message: Message) -> None:
    pending = getattr(host, "_tool_bundle_deferred_messages", None)
    if isinstance(pending, list):
        pending.append(message)
    else:
        host.memory.add_message(message)


def normalize_tool_calls_for_execution(
    host: Any, tool_calls: list[Any]
) -> tuple[str, list[Any]]:
    if len(tool_calls) <= 1:
        return "single", tool_calls
    names = _tool_call_names_from_list(tool_calls)
    force_result_md = getattr(host, "_lnr_should_force_result_md_now", None)
    force_sequential = bool(callable(force_result_md) and force_result_md())
    mode = classify_tool_call_bundle(
        tool_calls,
        readonly_tool_names=PARALLEL_READONLY_TOOLS,
        mutation_tool_names=FILE_MUTATION_TOOLS,
        force_sequential=force_sequential,
        parallel_bash_enabled=host._parallel_bash_enabled,
        bash_parallel_safe=_bash_command_parallel_safe,
    )
    if mode == "blocked_write_edit_bundle":
        host._log_info(
            "[tool-calls] blocked multi-tool bundle containing write/edit: %s",
            ",".join(name or "?" for name in names),
        )
    elif mode == "parallel_bash":
        host._log_info(
            "[tool-calls] multi-bash parallel bundle (%d calls)", len(tool_calls)
        )
    elif mode == "sequential" and not force_sequential:
        host._log_info(
            "[tool-calls] multi-tool response has %d calls; "
            "executing sequentially in one LLM round",
            len(tool_calls),
        )
    return mode, tool_calls


__all__ = [
    "add_message_after_current_tool_bundle",
    "normalize_tool_calls_for_execution",
]
