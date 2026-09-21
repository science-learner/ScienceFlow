"""Lazy public surface for ScienceFlow's domain-specific agent extensions."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentCallbackPorts": ("scienceflow.agent.core.ports.callback_ports", "AgentCallbackPorts"),
    "ScienceAgent": ("scienceflow.agent.core.runtime.agent", "ScienceAgent"),
    "_DEFAULT_SYSTEM": ("scienceflow.agent.prompts.system_prompt", "_DEFAULT_SYSTEM"),
    "_default_system_prompt": (
        "scienceflow.agent.prompts.system_prompt",
        "_default_system_prompt",
    ),
    "_extract_python_code_from_assistant_text": (
        "scienceflow.agent.prompts.write_coaching",
        "_extract_python_code_from_assistant_text",
    ),
    "_bash_command_parallel_safe": (
        "scienceflow.runtime.safety.policy.agent_policies.bash_utils",
        "_bash_command_parallel_safe",
    ),
    "_embedded_failure_user_message": (
        "scienceflow.runtime.safety.policy.agent_policies.bash_utils",
        "_embedded_failure_user_message",
    ),
    "_looks_like_quick_test_solution_run": (
        "scienceflow.runtime.safety.policy.agent_policies.bash_utils",
        "_looks_like_quick_test_solution_run",
    ),
    "_looks_like_write_placeholder_mimicry": (
        "scienceflow.runtime.safety.policy.agent_policies.bash_utils",
        "_looks_like_write_placeholder_mimicry",
    ),
    "_WS_LOG_RECOVERY_CAP": (
        "scienceflow.foundation.config.runtime.agent_constants",
        "_WS_LOG_RECOVERY_CAP",
    ),
    "_WS_LOG_USER_CAP": ("scienceflow.foundation.config.runtime.agent_constants", "_WS_LOG_USER_CAP"),
    "_WS_LOG_BODY_CAP": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "_WS_LOG_BODY_CAP",
    ),
    "_WS_LOG_MAX_LINES": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "_WS_LOG_MAX_LINES",
    ),
    "_WS_LOG_TOOL_CALL_SINGLE_CAP": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "_WS_LOG_TOOL_CALL_SINGLE_CAP",
    ),
    "_WS_LOG_TOOL_RESULT_CAP": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "_WS_LOG_TOOL_RESULT_CAP",
    ),
    "format_tool_call_lines_for_interaction_log": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "format_tool_call_lines_for_interaction_log",
    ),
    "format_tool_result_for_interaction_log": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "format_tool_result_for_interaction_log",
    ),
    "tool_result_text_for_interaction_log": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "tool_result_text_for_interaction_log",
    ),
    "truncate_for_interaction_log": (
        "scienceflow.runtime.observability.agent_io.interaction_log",
        "truncate_for_interaction_log",
    ),
    "InteractionLogPolicy": (
        "scienceflow.runtime.observability.agent_io.interaction_log_policy",
        "InteractionLogPolicy",
    ),
    "resolve_interaction_log_policy": (
        "scienceflow.runtime.observability.agent_io.interaction_log_policy",
        "resolve_interaction_log_policy",
    ),
    "_compress_tool_call_for_memory": (
        "scienceflow.research.state.knowledge.memory.agent.memory_utils",
        "_compress_tool_call_for_memory",
    ),
    "_tool_call_args_from_tc": (
        "scienceflow.research.state.knowledge.memory.agent.memory_utils",
        "_tool_call_args_from_tc",
    ),
    "inject_thought_into_tool_params": (
        "scienceflow.research.state.knowledge.memory.agent.memory_utils",
        "inject_thought_into_tool_params",
    ),
    "create_tool_collection": (
        "scienceflow.runtime.safety.execution.agent_runtime.tool_composition",
        "create_tool_collection",
    ),
    "clear_embedded_full_run_result": (
        "scienceflow.runtime.safety.policy.execution_policy",
        "clear_embedded_full_run_result",
    ),
    "ensure_full_execution": (
        "scienceflow.runtime.safety.policy.execution_policy",
        "ensure_full_execution",
    ),
    "write_embedded_full_run_result": (
        "scienceflow.runtime.safety.policy.execution_policy",
        "write_embedded_full_run_result",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
