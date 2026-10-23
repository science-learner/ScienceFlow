"""ScienceAgent construction adapter owned by the ScienceFlow domain."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from inquirycraft.memory import Message

from scienceflow.agent.core.runtime.run_policy import AutoContinuePolicy, DefaultPolicy
from scienceflow.foundation.config.schema.settings import Config
from scienceflow.runtime.environment import make_path_env
from scienceflow.research.state.repl_context import (
    _append_repl_bash_file_write_prompt,
    _plain_repl_auto_continue_session,
    _repl_code_organization_prompt,
    _repl_environment_context_prompt,
    _repl_workspace_git_prompt,
)

if TYPE_CHECKING:
    from scienceflow.agent import ScienceAgent
    from scienceflow.agent.core.runtime.run_policy import RunPolicy


def create_science_agent(
    cfg: Config,
    *,
    ui: object | None = None,
    task_description: str | None = None,
    run_policy: RunPolicy | None = None,
    system_prompt: str | None = None,
    system_prompt_hook: Callable[[str], str] | None = None,
    append_repl_system_prompt: bool | None = None,
    pin_task_description: bool = True,
    max_steps_override: int | None = None,
    on_llm_call: Callable[[dict[str, Any]], None] | None = None,
    memory_dir_override: str | Path | None = None,
    load_existing_memory: bool = False,
    memory_agent_name: str = "ScienceAgent",
    teleport_mode: str = "off",
    repl_bash_write_mode: bool = False,
    stable_system_prompt: bool | None = None,
    pin_environment_context: bool | None = None,
    code_organization_hint: str | None = None,
    workspace_git_enabled: bool = False,
    workspace_git_track_globs: list[str] | tuple[str, ...] | None = None,
    workspace_git_auto_review: bool = False,
    workspace_git_auto_checkpoint: bool = False,
    bash_max_output_chars_override: int | None = None,
    bash_max_stream_line_chars_override: int | None = None,
    bash_observation_summary_override: bool | None = None,
    interaction_log_layout: str = "flat",
    extra_env_override: dict[str, str] | None = None,
    skill_registry: object | None = None,
    task_type: str | None = None,
    skill_allow_names: list[str] | tuple[str, ...] | None = None,
    skill_tool_mode: str = "all",
    skill_allow_generic_wildcard: bool = True,
    skill_visible_max: int = 0,
    llm_stage_override: object | None = None,
) -> ScienceAgent:
    """Construct a single ScienceAgent instance (REPL reuses it for persistent memory)."""
    from scienceflow.agent import ScienceAgent
    from scienceflow.foundation.config.llm.llm_factory import build_stage_llm
    from scienceflow.research.state.knowledge.memory.records.agent_records import (
        create_agent_memory,
        load_agent_memory,
    )

    built = build_stage_llm(llm_stage_override or cfg.agent.code)
    memory_dir = (
        Path(memory_dir_override).expanduser().resolve(strict=False)
        if memory_dir_override is not None
        else Path(cfg.log_dir) / "agent_memory"
    )
    memory_dir.mkdir(parents=True, exist_ok=True)
    if load_existing_memory:
        memory = load_agent_memory(memory_dir, memory_agent_name, cfg.max_messages)
    else:
        memory = create_agent_memory(memory_dir, memory_agent_name, cfg.max_messages)
    rp: RunPolicy = (
        run_policy if run_policy is not None else DefaultPolicy(recovery_rounds=3)
    )
    hook: Callable[[str], str] | None = system_prompt_hook
    if repl_bash_write_mode:
        prior_hook = hook

        def _bash_write_hook(base: str) -> str:
            updated = prior_hook(base) if prior_hook is not None else base
            return _append_repl_bash_file_write_prompt(updated)

        hook = _bash_write_hook
    td_stripped = (
        str(task_description).strip()
        if task_description and str(task_description).strip()
        else ""
    )
    append_repl = (
        system_prompt_hook is None
        if append_repl_system_prompt is None
        else bool(append_repl_system_prompt)
    )
    # REPL code-agent guidance now lives in a separate stable system core
    # message. Keep the legacy suffix helper available for tests/docs, but
    # do not append it into the runtime/tool system contract.
    _ = append_repl
    plain_repl = _plain_repl_auto_continue_session(rp, teleport_mode)
    stable_system = (
        isinstance(rp, AutoContinuePolicy)
        if stable_system_prompt is None
        else bool(stable_system_prompt)
    )
    agent_max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else int(cfg.qa_max_steps)
    )
    if agent_max_steps <= 0:
        agent_max_steps = int(cfg.qa_max_steps)
    bash_max_output_chars = (
        int(bash_max_output_chars_override)
        if bash_max_output_chars_override is not None
        else int(getattr(cfg, "bash_max_output_chars", 8000) or 8000)
    )
    if bash_max_output_chars <= 0:
        bash_max_output_chars = 8000
    bash_max_stream_line_chars = (
        int(bash_max_stream_line_chars_override)
        if bash_max_stream_line_chars_override is not None
        else int(getattr(cfg, "bash_max_stream_line_chars", 2400) or 2400)
    )
    if bash_max_stream_line_chars <= 0:
        bash_max_stream_line_chars = 2400
    bash_observation_summary_enabled = (
        bool(bash_observation_summary_override)
        if bash_observation_summary_override is not None
        else False
    )
    pin_env = (
        plain_repl if pin_environment_context is None else bool(pin_environment_context)
    )
    write_auto_snapshot_enabled = bool(
        getattr(cfg, "write_auto_snapshot_enabled", True)
    )
    if plain_repl:
        # The full write/edit payload already lives in assistant tool-call
        # arguments, like shell/patch-style calls. Keep the following tool
        # result short instead of echoing code through an auto-snapshot.
        write_auto_snapshot_enabled = False
    agent = ScienceAgent(
        llm=built,
        memory=memory,
        workspace_dir=cfg.workspace_dir,
        sandbox=cfg.scienceflow_tools_sandbox,
        path_guard_extra_roots=cfg.path_guard_extra_roots,
        ui=ui,
        max_steps=agent_max_steps,
        exec_feedback_max_chars=cfg.exec_feedback_max_chars,
        scienceflow_stdout_max_chars=cfg.scienceflow_stdout_max_chars,
        sliding_window_budget_chars=cfg.sliding_window_budget_chars,
        mid_run_compact_enabled=bool(cfg.mid_run_compact_enabled),
        llm_stream_timeout_sec=cfg.scienceflow_llm_stream_timeout_sec,
        llm_tool_stream_max_attempts=cfg.llm_tool_stream_max_attempts,
        llm_tool_stream_retry_base_delay_sec=cfg.llm_tool_stream_retry_base_delay_sec,
        llm_tool_stream_retry_max_delay_sec=cfg.llm_tool_stream_retry_max_delay_sec,
        stream_repetition_detection=bool(cfg.stream_repetition_detection),
        stream_repetition_window_chars=int(cfg.stream_repetition_window_chars),
        stream_repetition_ngram_len=int(cfg.stream_repetition_ngram_len),
        stream_repetition_max_repeats=int(cfg.stream_repetition_max_repeats),
        stream_max_output_chars_soft=int(cfg.stream_max_output_chars_soft),
        stream_repetition_retry_max=int(cfg.stream_repetition_retry_max),
        bash_timeout_sec=cfg.scienceflow_bash_timeout_sec,
        bash_timeout_slow_sec=cfg.scienceflow_bash_timeout_slow_sec,
        bash_max_output_chars=bash_max_output_chars,
        bash_max_stream_line_chars=bash_max_stream_line_chars,
        bash_observation_summary_enabled=bash_observation_summary_enabled,
        interaction_log_level=cfg.scienceflow_interaction_log_level,
        interaction_log_full=cfg.scienceflow_interaction_log_full,
        interaction_log_color=cfg.scienceflow_interaction_log_color,
        interaction_log_llm_stream=cfg.scienceflow_interaction_log_llm_stream,
        interaction_log_layout=interaction_log_layout,
        tool_memory_compression=cfg.tool_memory_compression,
        include_write_edit_tools=not bool(repl_bash_write_mode),
        msg0_compress_body=bool(getattr(cfg, "msg0_compress_body", False)),
        write_return_full_max_chars=int(cfg.write_return_full_max_chars),
        write_return_full_max_lines=int(cfg.write_return_full_max_lines),
        write_return_head_tail_lines=int(cfg.write_return_head_tail_lines),
        edit_return_full_max_chars=int(cfg.edit_return_full_max_chars),
        edit_return_full_max_lines=int(cfg.edit_return_full_max_lines),
        edit_return_head_tail_lines=int(cfg.edit_return_head_tail_lines),
        edit_return_change_ctx_lines=int(cfg.edit_return_change_ctx_lines),
        edit_failure_top_k_candidates=int(cfg.edit_failure_top_k_candidates),
        edit_failure_diag_max_chars=int(cfg.edit_failure_diag_max_chars),
        file_snapshot_latest_only=bool(cfg.file_snapshot_latest_only),
        grep_max_results_lines=int(getattr(cfg, "grep_max_results_lines", 50)),
        bash_success_tail_lines=cfg.bash_success_tail_lines,
        bash_success_tail_lines_solution=int(
            getattr(cfg, "bash_success_tail_lines_solution", 8),
        ),
        bash_success_tail_lines_test=int(
            getattr(cfg, "bash_success_tail_lines_test", 120),
        ),
        bash_success_tail_lines_readonly=int(
            getattr(cfg, "bash_success_tail_lines_readonly", 30),
        ),
        bash_success_tail_lines_install=int(
            getattr(cfg, "bash_success_tail_lines_install", 5),
        ),
        read_success_max_lines=int(getattr(cfg, "read_success_max_lines", 200) or 200),
        write_auto_snapshot_enabled=write_auto_snapshot_enabled,
        write_auto_snapshot_paths=getattr(cfg, "write_auto_snapshot_paths", None),
        write_auto_snapshot_code_extensions=getattr(
            cfg,
            "write_auto_snapshot_code_extensions",
            None,
        ),
        write_auto_snapshot_max_lines=int(
            getattr(cfg, "write_auto_snapshot_max_lines", 400) or 400,
        ),
        write_auto_snapshot_max_chars=int(
            getattr(cfg, "write_auto_snapshot_max_chars", 8_000) or 8_000,
        ),
        write_auto_snapshot_changed_context_lines=int(
            getattr(cfg, "write_auto_snapshot_changed_context_lines", 10) or 10,
        ),
        write_auto_snapshot_symbol_body_lines=int(
            getattr(cfg, "write_auto_snapshot_symbol_body_lines", 3) or 3,
        ),
        read_overlap_guard_enabled=bool(
            getattr(cfg, "read_overlap_guard_enabled", True)
        ),
        bash_output_dedup_enabled=bool(cfg.bash_output_dedup_enabled),
        bash_output_dedup_min_repeat=int(cfg.bash_output_dedup_min_repeat),
        bash_output_dedup_summary_prefix=str(cfg.bash_output_dedup_summary_prefix),
        bash_output_dedup_apply_to_memory=bool(cfg.bash_output_dedup_apply_to_memory),
        parallel_bash_enabled=cfg.parallel_bash_enabled,
        parallel_llm_tool_calls=cfg.parallel_llm_tool_calls,
        extra_env=make_path_env(extra_env_override or {}),
        skill_registry=skill_registry,
        task_type=task_type,
        skill_allow_names=skill_allow_names,
        skill_tool_mode=skill_tool_mode,
        skill_allow_generic_wildcard=skill_allow_generic_wildcard,
        skill_visible_max=skill_visible_max,
        systemPrompt=system_prompt,
        system_prompt_hook=hook,
        run_policy=rp,
        on_llm_call=on_llm_call,
        teleport_mode=teleport_mode,
        stable_system_prompt=stable_system,
        workspace_git_auto_checkpoint_enabled=bool(
            workspace_git_enabled and workspace_git_auto_checkpoint,
        ),
        workspace_git_track_globs=workspace_git_track_globs,
    )
    if pin_env:
        env_prompt = _repl_environment_context_prompt(cfg.workspace_dir)
        agent.pin_message(Message.user_message(env_prompt))
    code_org = _repl_code_organization_prompt(code_organization_hint)
    if code_org:
        agent.pin_message(Message.user_message(code_org))
    workspace_git_prompt = _repl_workspace_git_prompt(
        workspace_git_enabled,
        workspace_git_track_globs,
        auto_review=workspace_git_auto_review,
        auto_checkpoint=workspace_git_auto_checkpoint,
    )
    if workspace_git_prompt:
        agent.pin_message(Message.user_message(workspace_git_prompt))
    if td_stripped and bool(pin_task_description):
        agent.pin_message(Message.user_message("## Task description\n\n" + td_stripped))
    return agent


__all__ = ["create_science_agent"]
