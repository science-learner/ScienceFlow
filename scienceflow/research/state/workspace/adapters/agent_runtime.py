"""Workspace-to-agent session adapter for a persistent ``ScienceAgent``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inquirycraft.memory import JsonKeyValueStorage
from inquirycraft.tools import PathGuard

from scienceflow.research.state.workspace.session.agent_contracts import _ON_LLM_CALL_UNSET, _logger
from scienceflow.research.state.knowledge.memory.agent.memory_utils import inject_thought_into_tool_params
from scienceflow.research.state.knowledge.memory.agent.resource_feedback_memory import (
    ResourceFeedbackMemoryDeduper,
)
from scienceflow.runtime.safety.execution.agent_runtime.tool_composition import create_tool_collection
from scienceflow.runtime.observability.interaction_log import (
    attach_workspace_interaction_logger,
    build_interaction_log_context_tag,
    set_interaction_log_context_tag,
)


@dataclass
class WorkspaceSwapContext:
    new_workspace: Path
    old_workspace: Path
    old_round: int
    extra_roots: list[str | Path] | None
    readonly_dirs: list[str] | None
    extra_env: dict[str, str] | None
    skill_registry: Any
    task_type: str | None
    skill_allow_names: tuple[str, ...]
    skill_tool_mode: str
    skill_allow_generic_wildcard: bool
    skill_visible_max: int
    resource_observer: Any
    display_root: Path


def _prepare_workspace_swap(
    agent: Any, options: dict[str, Any]
) -> WorkspaceSwapContext:
    new_workspace = Path(options["workspace_dir"]).resolve()
    old_workspace = agent._workspace_dir
    old_round = int(agent._current_round)
    agent._search_round_offset = (
        int(getattr(agent, "_search_round_offset", 0) or 0) + old_round
    )
    agent._current_round = 0
    agent._call_seq = 0
    if options["on_llm_call"] is not _ON_LLM_CALL_UNSET:
        agent._on_llm_call = options["on_llm_call"]
    if options["max_steps_override"] is not None:
        agent.max_steps = int(options["max_steps_override"])
        agent._effective_max_steps = int(options["max_steps_override"])
    extra_roots = (
        list(options["path_guard_extra_roots"])
        if options["path_guard_extra_roots"] is not None
        else None
    )
    readonly_dirs = (
        list(options["readonly_dirs"]) if options["readonly_dirs"] is not None else None
    )
    extra_env = (
        dict(options["extra_env"])
        if options["extra_env"] is not None
        else agent._extra_env
    )
    skill_registry = options["skill_registry"] or getattr(
        agent, "_skill_registry", None
    )
    task_type = options["task_type"] or getattr(agent, "_task_type", None)
    skill_allow_names = tuple(
        options["skill_allow_names"]
        if options["skill_allow_names"] is not None
        else getattr(agent, "_skill_allow_names", ())
    )
    skill_tool_mode = str(
        options["skill_tool_mode"]
        if options["skill_tool_mode"] is not None
        else getattr(agent, "_skill_tool_mode", "all")
    )
    generic_wildcard = options["skill_allow_generic_wildcard"]
    if generic_wildcard is None:
        generic_wildcard = getattr(agent, "_skill_allow_generic_wildcard", True)
    visible_max = options["skill_visible_max"]
    if visible_max is None:
        visible_max = getattr(agent, "_skill_visible_max", 0)
    observer = options["resource_observer"] or getattr(
        agent, "_resource_observer", None
    )
    if options["bash_timeout_sec"] is not None:
        agent._bash_timeout_sec = float(options["bash_timeout_sec"])
    if options["bash_timeout_slow_sec"] is not None:
        agent._bash_timeout_slow_sec = float(options["bash_timeout_slow_sec"])
    elif not hasattr(agent, "_bash_timeout_slow_sec"):
        agent._bash_timeout_slow_sec = float(agent._bash_timeout_sec)
    old_display_root = Path(
        getattr(agent, "_tool_display_root", old_workspace)
    ).resolve()
    display_root = (
        new_workspace if old_display_root == old_workspace else old_display_root
    )
    return WorkspaceSwapContext(
        new_workspace,
        old_workspace,
        old_round,
        extra_roots,
        readonly_dirs,
        extra_env,
        skill_registry,
        task_type,
        skill_allow_names,
        skill_tool_mode,
        bool(generic_wildcard),
        int(visible_max or 0),
        observer,
        display_root,
    )


def _rebuild_workspace_tools(agent: Any, context: WorkspaceSwapContext) -> None:
    tools = create_tool_collection(
        context.new_workspace,
        sandbox=True,
        path_guard_extra_roots=context.extra_roots,
        max_bash_output_chars=int(getattr(agent, "_bash_max_output_chars", 8000)),
        max_bash_stream_line_chars=int(
            getattr(agent, "_bash_max_stream_line_chars", 2400)
        ),
        bash_observation_summary_enabled=bool(
            getattr(agent, "_bash_observation_summary_enabled", False)
        ),
        bash_timeout_sec=float(agent._bash_timeout_sec),
        bash_timeout_slow_sec=float(agent._bash_timeout_slow_sec),
        extra_env=context.extra_env,
        readonly_dirs=context.readonly_dirs,
        resource_observer=context.resource_observer,
        skill_registry=context.skill_registry,
        task_type=context.task_type,
        skill_allow_names=context.skill_allow_names,
        skill_tool_mode=context.skill_tool_mode or "all",
        skill_allow_generic_wildcard=context.skill_allow_generic_wildcard,
        skill_visible_max=context.skill_visible_max,
        include_write_edit_tools=bool(
            getattr(agent, "_include_write_edit_tools", True)
        ),
        tool_display_paths_relative=bool(
            getattr(agent, "_tool_display_paths_relative", True)
        ),
        tool_display_root=context.display_root,
        grep_max_results_lines=int(getattr(agent, "_grep_max_results_lines", 50)),
    )
    agent.availableTools = tools
    agent._tools_with_thought = inject_thought_into_tool_params(tools.to_params())
    agent._workspace_dir = context.new_workspace
    agent._extra_env = context.extra_env
    agent._skill_registry = context.skill_registry
    agent._task_type = context.task_type
    agent._skill_allow_names = context.skill_allow_names
    agent._skill_tool_mode = context.skill_tool_mode
    agent._skill_allow_generic_wildcard = context.skill_allow_generic_wildcard
    agent._skill_visible_max = context.skill_visible_max
    agent._resource_observer = context.resource_observer
    agent._tool_display_root = context.display_root
    agent._tool_output_artifacts = agent._make_tool_output_artifact_store(
        context.new_workspace
    )
    agent._resource_feedback_memory_deduper = ResourceFeedbackMemoryDeduper()


def _rebind_workspace_memory_context(
    agent: Any,
    context: WorkspaceSwapContext,
    *,
    budget_chars: int | None,
) -> None:
    if not hasattr(agent, "_memory_ctx") or agent._memory_ctx is None:
        return
    agent._memory_ctx._workspace = context.new_workspace
    if budget_chars is not None:
        agent._memory_ctx._budget_chars = max(0, int(budget_chars))
        agent._memory_ctx._last_window_k = 0
    try:
        agent._memory_ctx._guard = PathGuard(
            context.new_workspace,
            extra_roots=context.extra_roots,
        )
    except Exception:
        pass


def _move_workspace_memory_storage(agent: Any, context: WorkspaceSwapContext) -> None:
    if (
        getattr(agent, "memory", None) is None
        or getattr(agent.memory, "chat_history_memory", None) is None
    ):
        return
    try:
        from scienceflow.research.state.knowledge.memory.records.agent_records import (
            select_prefix_safe_agent_memory_records,
            write_agent_memory_record_files,
        )

        storage = agent.memory.chat_history_memory.storage
        old_storage_path = Path(
            str(
                getattr(storage, "json_path", None)
                or getattr(storage, "path", None)
                or ""
            )
        )
        old_agent_dir = old_storage_path.parent if str(old_storage_path) else None
        max_messages = int(getattr(agent.memory, "max_messages", 0) or 0)
        if old_agent_dir is not None and old_agent_dir.is_dir():
            old_records = select_prefix_safe_agent_memory_records(
                old_agent_dir,
                max_messages=max_messages,
            )
        else:
            old_records = list(storage.load() or [])
    except Exception:
        old_records = []
    new_agent_dir = context.new_workspace / ".agent_memory" / "Draft"
    new_short_term = new_agent_dir / "short_term.json"
    new_agent_dir.mkdir(parents=True, exist_ok=True)
    try:
        if old_records:
            write_agent_memory_record_files(
                new_agent_dir,
                old_records,
                write_long_term=True,
            )
        else:
            new_short_term.touch(exist_ok=True)
        agent.memory.chat_history_memory.storage = JsonKeyValueStorage(
            path=str(new_short_term),
            mode="a",
        )
        agent.memory.chat_history_memory._num_records = len(old_records)
    except Exception:
        _logger.warning(
            "[teleport] swap_workspace memory storage swap failed; keeping old storage",
            exc_info=True,
        )


def _rebind_workspace_policy_and_logs(
    agent: Any,
    context: WorkspaceSwapContext,
    options: dict[str, Any],
) -> None:
    if options["run_policy"] is not None:
        agent._run_policy = options["run_policy"]
    try:
        agent._ws_interaction_log = attach_workspace_interaction_logger(
            context.new_workspace,
            color=(
                bool(options["interaction_log_color"])
                if options["interaction_log_color"] is not None
                else False
            ),
            layout=getattr(agent, "_interaction_log_layout", "flat"),
        )
    except Exception:
        _logger.warning("[teleport] interaction-log re-attach failed", exc_info=True)
    if options["interaction_log_session"] is not None:
        agent._lnr_interaction_log_session = (
            options["interaction_log_session"] or "draft"
        ).strip().lower() or "draft"
    if options["interaction_log_phase"] is not None:
        phase = str(options["interaction_log_phase"]).strip().lower()
        agent._lnr_interaction_log_phase = phase or None
    if (
        options["interaction_log_session"] is not None
        or options["interaction_log_phase"] is not None
    ):
        try:
            tag = build_interaction_log_context_tag(
                agent._lnr_interaction_log_session,
                agent._lnr_interaction_log_phase,
            )
            agent._interaction_log_ctx_token = set_interaction_log_context_tag(tag)
        except Exception:
            pass
    if options["lnr_llm_turns_log_path"] is not None:
        value = options["lnr_llm_turns_log_path"]
        agent._lnr_llm_turns_log_path = Path(value).resolve() if value else None
    if options["sft_data_log_path"] is not None:
        value = options["sft_data_log_path"]
        agent._sft_data_log_path = Path(value).resolve() if value else None


def _reset_workspace_node_state(agent: Any) -> None:
    agent._embedded_full_run_done = False
    agent._lnr_snapshot_ok = False
    agent._lnr_snapshot_reason = ""
    agent._lnr_last_valid_solution_sha = ""
    agent._lnr_last_valid_solution_rel_path = "solution.py"
    agent._lnr_last_valid_submission_sha = ""
    agent._lnr_last_valid_bash_cmd = ""
    agent._lnr_last_valid_bare_run_is_local = False
    agent._lnr_committed_ledger_step_count = None
    agent._lnr_ledger_committed = False
    agent._lnr_ledger_step_count = 0
    agent._lnr_ledger_new_entries = 0
    agent._lnr_stage_commit_guard_failed = False
    agent._lnr_stage_journal_pending = False
    agent._run_control_user_injections = 0
    agent._lnr_result_md_after_success_pending = False
    agent._mid_run_compact_enabled = False
    agent._mid_run_compacted = False


def swap_workspace_runtime(agent: Any, options: dict[str, Any]) -> None:
    context = _prepare_workspace_swap(agent, options)
    _rebuild_workspace_tools(agent, context)
    _rebind_workspace_memory_context(
        agent,
        context,
        budget_chars=options["sliding_window_budget_chars"],
    )
    if options["copy_memory_storage_to_new_workspace"]:
        _move_workspace_memory_storage(agent, context)
    _rebind_workspace_policy_and_logs(agent, context, options)
    _reset_workspace_node_state(agent)
    _logger.info(
        "[teleport] swap_workspace: %s -> %s (round_offset=%d -> %d)",
        context.old_workspace,
        context.new_workspace,
        agent._search_round_offset - context.old_round,
        agent._search_round_offset,
    )
