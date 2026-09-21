# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""ScienceAgent responsibility: tool feedback projection, resource deduplication, and accounting."""

from __future__ import annotations

import logging
from typing import Any

from inquirycraft.tools import ToolResult

from scienceflow.research.state.knowledge.context.memory_context import (
    trim_tool_feedback_for_llm_context,
)
from scienceflow.research.state.knowledge.memory.agent.resource_feedback_memory import (
    ResourceFeedbackMemoryDeduper,
)
from scienceflow.research.state.workspace.adapters.path_hygiene import (
    _hide_agent_hidden_workspace_file_lines,
    _hide_agent_hidden_workspace_filename_mentions,
    _sanitize_agent_visible_paths,
    _tool_request_mentions_hidden_workspace_file,
    _workspace_relative_path_mode_enabled,
)
from scienceflow.runtime.safety.policy.agent_policies.artifacts import (
    attach_tool_output_reference,
    raw_tool_result_text,
    reduce_tool_feedback_for_memory,
)

_logger = logging.getLogger("scienceflow")


def _prepare_tool_feedback_for_memory(
    self,
    tool_name: str,
    args: dict[str, Any],
    tool_result: ToolResult,
    *,
    guard_coaching: str = "",
) -> str:
    """Build final tool feedback for chat memory and persist raw output artifact.

    Bash streaming is already complete before this method is called; this only
    touches the final ToolResult text that would otherwise be written to memory.
    """
    raw_text = raw_tool_result_text(tool_result)
    hidden_request = _tool_request_mentions_hidden_workspace_file(self, tool_name, args)
    ref = None
    store = getattr(self, "_tool_output_artifacts", None)
    if store is not None and raw_text:
        ref = store.reserve(tool_name, raw_text)
        if ref is not None and not store.write_raw(ref, raw_text):
            ref = None

    if hidden_request:
        feedback = "[hidden control file output omitted from agent memory]"
    else:
        feedback = self._memory_ctx.record_tool_result(
            tool_name,
            args,
            tool_result,
            raw_id=ref.raw_id if ref is not None else "",
        )
    if guard_coaching:
        feedback = feedback + "\n\n" + guard_coaching

    feedback, reducer = reduce_tool_feedback_for_memory(
        tool_name=tool_name,
        args=args,
        feedback=feedback,
        raw_text=raw_text,
        tool_error=bool(getattr(tool_result, "error", None)),
    )
    feedback, resource_deduped = self._dedup_resource_feedback_for_memory(feedback)
    if resource_deduped:
        reducer = f"{reducer}+resource_feedback_dedup"

    if ref is not None:
        feedback = attach_tool_output_reference(
            feedback,
            raw_id=ref.raw_id,
            raw_chars=ref.raw_chars,
            reducer=reducer,
        )

    feedback = trim_tool_feedback_for_llm_context(
        feedback,
        max_chars=self._exec_feedback_max_chars,
    )
    if not hidden_request:
        feedback = _hide_agent_hidden_workspace_file_lines(self, feedback)
    feedback = _hide_agent_hidden_workspace_filename_mentions(self, feedback)

    if ref is not None:
        store.append_index(
            ref,
            reducer_name=reducer,
            compressed_chars=len(feedback),
        )
        append_stage_index = getattr(store, "append_stage_index", None)
        if callable(append_stage_index):
            append_stage_index(
                ref,
                args=args,
                reducer_name=reducer,
                compressed_chars=len(feedback),
                tool_error=bool(getattr(tool_result, "error", None)),
            )

    return feedback


def _dedup_resource_feedback_for_memory(self, feedback: str) -> tuple[str, bool]:
    deduper = getattr(self, "_resource_feedback_memory_deduper", None)
    if deduper is None:
        deduper = ResourceFeedbackMemoryDeduper()
        self._resource_feedback_memory_deduper = deduper
    return deduper.reduce(feedback)


def _log_iteration_header(self, round_one_based: int) -> None:
    # Teleport mode F5: when ``_search_round_offset > 0`` the search-wide round
    # counter shown to the LLM is ``round_one_based + offset`` / ``max_steps + offset``.
    # Mirror that into the audit log so external readers (interaction.log /
    # scienceflow.log) see the same monotone series rather than a confusing
    # per-node ``Iteration 1/50`` reset.
    _offset = int(getattr(self, "_search_round_offset", 0) or 0)
    _eff_total = self._effective_max_steps + _offset
    _disp_round = round_one_based + _offset
    if self._repl_session_run_index is not None:
        self._log_info(
            "[repl-run %d] ====== Agent Iteration %d/%d =======",
            self._repl_session_run_index,
            _disp_round,
            _eff_total,
        )
    elif _offset > 0:
        self._log_info(
            "====== Agent Iteration %d/%d (search round) =======",
            _disp_round,
            _eff_total,
        )
    else:
        self._log_info(
            "====== Agent Iteration %d/%d =======",
            round_one_based,
            self._effective_max_steps,
        )


def _sanitize_log_format(
    self, msg: str, args: tuple[object, ...]
) -> tuple[str, tuple[object, ...]]:
    if not _workspace_relative_path_mode_enabled(self):
        return msg, args
    try:
        rendered = (msg % args) if args else str(msg)
    except Exception:
        rendered = " ".join([str(msg), *(str(a) for a in args)])
    return "%s", (_sanitize_agent_visible_paths(self, rendered),)


def _log_info(self, msg: str, *args: object) -> None:
    msg, args = self._sanitize_log_format(msg, args)
    _logger.info(msg, *args)
    if self._ws_interaction_log is not None:
        self._ws_interaction_log.info(msg, *args)


def _log_warning(self, msg: str, *args: object) -> None:
    msg, args = self._sanitize_log_format(msg, args)
    _logger.warning(msg, *args)
    if self._ws_interaction_log is not None:
        self._ws_interaction_log.warning(msg, *args)


def _pop_and_log_thought(self, args: dict[str, Any]) -> None:
    """Remove ``thought`` from tool args (not passed to tools) and log it."""

    raw = args.pop("thought", None)
    if raw is None:
        return
    s = str(raw).strip()
    if s:
        self._log_info("[thought] %s", s)


def _next_step(self) -> int:
    self._ui_step += 1
    return self._ui_step


def _sync_last_run_token_totals(self) -> None:
    """Copy cumulative run counters to ``last_run_*`` (incl. compact subset)."""
    self.last_run_tokens_in = self._run_tokens_in
    self.last_run_tokens_out = self._run_tokens_out
    self.last_run_tokens_cached = self._run_tokens_cached
    self.last_run_llm_calls = self._run_llm_calls
    self.last_run_route_tokens_in = self._run_route_tokens_in
    self.last_run_route_tokens_out = self._run_route_tokens_out
    self.last_run_route_tokens_cached = self._run_route_tokens_cached
    self.last_run_route_llm_calls = self._run_route_llm_calls
    self.last_run_compact_tokens_in = self._run_compact_tokens_in
    self.last_run_compact_tokens_out = self._run_compact_tokens_out
    self.last_run_compact_tokens_cached = self._run_compact_tokens_cached
    self.last_run_compact_llm_calls = self._run_compact_llm_calls


def _accumulate_compact_llm_into_run_counters(self) -> None:
    """Compatibility seam; unified provider audit owns compact accounting."""
