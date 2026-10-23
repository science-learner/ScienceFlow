# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""ScienceAgent public composition facade."""

from __future__ import annotations

from typing import Any

from inquirycraft.memory import Memory
from inquirycraft.runtime import AgentState
from inquirycraft.tools import ToolChoice, ToolCollection
from pydantic import BaseModel, ConfigDict, Field

from scienceflow.agent.core.runtime.ephemeral import ask_tool_ephemeral
from scienceflow.agent.core.runtime.registry import register_agent
from scienceflow.agent.factory import construction
from scienceflow.agent.policies import routing
from scienceflow.agent.prompts import system_prompt, write_coaching
from scienceflow.agent.session import run_science_agent
from scienceflow.research.control import agent_runtime_state as runtime_state
from scienceflow.research.quality import embedded_fullrun
from scienceflow.research.solver.lnr.orchestration import agent_hooks
from scienceflow.research.state.knowledge.memory.context import (
    agent_compaction as compaction,
)
from scienceflow.research.state.workspace.adapters import path_hygiene
from scienceflow.research.state.workspace.session import (
    agent_session as workspace_session,
)
from scienceflow.runtime.observability import agent_tool_feedback as tool_feedback
from scienceflow.runtime.observability.telemetry.agent.agent_call import record_llm_call
from scienceflow.runtime.safety.execution.agent_runtime import edit_guard
from scienceflow.runtime.safety.execution.agent_runtime.tool_bundle_policy import (
    add_message_after_current_tool_bundle,
)


@register_agent("science")
class ScienceAgent(BaseModel):
    """Multi-step LLM agent using OpenAI-style function calling and local tools."""

    max_steps: int = 30
    name: str | None = None
    description: str | None = None
    systemPrompt: str | list[dict] | None = None
    llm: Any = Field(...)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    availableTools: ToolCollection | None = None
    toolChoices: Any = ToolChoice.AUTO
    toolCalls: list[Any] | None = None
    ToolConfig: dict | None = None
    exec_mode: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    __init__ = construction.__init__
    configure_callback_ports = construction.configure_callback_ports
    set_repl_session_run_index = path_hygiene.set_repl_session_run_index
    _get_agent_hidden_workspace_filenames = (
        path_hygiene._get_agent_hidden_workspace_filenames
    )
    _normalize_hidden_workspace_path = path_hygiene._normalize_hidden_workspace_path
    _get_agent_hidden_workspace_path_prefixes = (
        path_hygiene._get_agent_hidden_workspace_path_prefixes
    )
    _apply_agent_hidden_path_denials = path_hygiene._apply_agent_hidden_path_denials
    _text_mentions_hidden_workspace_prefix = (
        path_hygiene._text_mentions_hidden_workspace_prefix
    )
    _hide_agent_hidden_workspace_filename_mentions = (
        path_hygiene._hide_agent_hidden_workspace_filename_mentions
    )
    _path_mentions_hidden_workspace_file = (
        path_hygiene._path_mentions_hidden_workspace_file
    )
    _tool_request_mentions_hidden_workspace_file = (
        path_hygiene._tool_request_mentions_hidden_workspace_file
    )
    _hide_agent_hidden_workspace_file_lines = (
        path_hygiene._hide_agent_hidden_workspace_file_lines
    )
    _sanitize_agent_visible_paths = path_hygiene._sanitize_agent_visible_paths
    sanitize_existing_memory_agent_visible_paths = (
        path_hygiene.sanitize_existing_memory_agent_visible_paths
    )
    swap_workspace = workspace_session.swap_workspace
    _make_tool_output_artifact_store = (
        workspace_session._make_tool_output_artifact_store
    )
    _prepare_tool_feedback_for_memory = tool_feedback._prepare_tool_feedback_for_memory
    _dedup_resource_feedback_for_memory = (
        tool_feedback._dedup_resource_feedback_for_memory
    )
    _sanitize_log_format = tool_feedback._sanitize_log_format
    _log_info = tool_feedback._log_info
    _log_warning = tool_feedback._log_warning
    _sync_last_run_token_totals = tool_feedback._sync_last_run_token_totals
    _accumulate_compact_llm_into_run_counters = (
        tool_feedback._accumulate_compact_llm_into_run_counters
    )
    file_state_summary = runtime_state.file_state_summary
    pin_message = runtime_state.pin_message
    _sha256_of_solution = runtime_state._sha256_of_solution
    _sha256_of_workspace_file = runtime_state._sha256_of_workspace_file
    _normalize_valid_run_source_rel_path = (
        runtime_state._normalize_valid_run_source_rel_path
    )
    _lnr_mark_valid_bare_run = runtime_state._lnr_mark_valid_bare_run
    _lnr_restore_valid_bare_run_from_snapshot = (
        runtime_state._lnr_restore_valid_bare_run_from_snapshot
    )
    _lnr_has_current_valid_bare_run = runtime_state._lnr_has_current_valid_bare_run
    _embedded_full_run_metric_token = runtime_state._embedded_full_run_metric_token
    _reset_edit_read_guard_state = edit_guard._reset_edit_read_guard_state
    compact = compaction.compact
    compact_inband = compaction.compact_inband
    _ask_tool_stream_guarded = ask_tool_ephemeral
    _record_llm_call = record_llm_call
    _build_system_prompt = system_prompt._build_system_prompt
    _build_system_messages = system_prompt._build_system_messages
    _format_round_budget = system_prompt._format_round_budget
    _maybe_append_solution_write_nudge = (
        write_coaching._maybe_append_solution_write_nudge
    )
    _maybe_append_write_failure_coaching = (
        write_coaching._maybe_append_write_failure_coaching
    )
    _solution_write_syntax_or_short_failure = (
        write_coaching._solution_write_syntax_or_short_failure
    )
    _try_write_solution_from_assistant_text = (
        write_coaching._try_write_solution_from_assistant_text
    )
    _lnr_maybe_periodic_inject_at_round_start = (
        agent_hooks._lnr_maybe_periodic_inject_at_round_start
    )
    _maybe_write_bare_run_tail_snapshot = (
        embedded_fullrun._maybe_write_bare_run_tail_snapshot
    )
    _inject_run_control_user_message = embedded_fullrun._inject_run_control_user_message
    _append_to_last_tool_message = embedded_fullrun._append_to_last_tool_message
    _inject_run_control_ok_message = embedded_fullrun._inject_run_control_ok_message
    _mirror_embedded_full_run_to_interaction_log = (
        embedded_fullrun._mirror_embedded_full_run_to_interaction_log
    )
    _append_embedded_full_run_to_result_md = (
        embedded_fullrun._append_embedded_full_run_to_result_md
    )
    _append_mlebench_validation_to_result_md = (
        embedded_fullrun._append_mlebench_validation_to_result_md
    )
    _run_mlebench_validation_after_embedded_full_run = (
        embedded_fullrun._run_mlebench_validation_after_embedded_full_run
    )
    run = run_science_agent
    _agentic_route_log_dir = routing._agentic_route_log_dir
    _write_agentic_route_response = routing._write_agentic_route_response
    _build_lnr_stage_commit_compact_messages = (
        routing._build_lnr_stage_commit_compact_messages
    )
    _sync_resource_state_summary_slot = routing._sync_resource_state_summary_slot
    _append_agentic_route_decision_log = routing._append_agentic_route_decision_log
    _accumulate_last_llm_call_tokens_into_run = (
        routing._accumulate_last_llm_call_tokens_into_run
    )
    run_ephemeral_agentic_route_prompt = routing.run_ephemeral_agentic_route_prompt
    _looks_like_repl_small_talk = routing._looks_like_repl_small_talk
    _unavailable_repl_tool_names = routing._unavailable_repl_tool_names
    _tc_function_name = routing._tc_function_name
    _tc_arguments_dict = routing._tc_arguments_dict
    _repl_file_change_command_from_tool = routing._repl_file_change_command_from_tool
    _block_unavailable_repl_tool_calls = routing._block_unavailable_repl_tool_calls
    _add_message_after_current_tool_bundle = add_message_after_current_tool_bundle
