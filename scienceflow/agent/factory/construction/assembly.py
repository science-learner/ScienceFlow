"""Typed, phased builder for ``ScienceAgent``."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from inquirycraft.runtime import CancellationToken
from pydantic import BaseModel

from scienceflow.agent.core.ports.callback_ports import AgentCallbackPorts
from scienceflow.agent.core.ports.host_ports import compose_agent_host_ports
from scienceflow.agent.core.runtime.run_policy import RunPolicy
from scienceflow.research.state.knowledge.skills.catalog.registry import SkillRegistry
from scienceflow.runtime.observability.agent_io.interaction_log_policy import (
    resolve_interaction_log_policy,
)
from scienceflow.research.state.knowledge.memory.agent.memory_utils import inject_thought_into_tool_params
from scienceflow.research.state.knowledge.memory.agent.resource_feedback_memory import (
    ResourceFeedbackMemoryDeduper,
)
from scienceflow.runtime.safety.policy.agent_policies.guards import (
    BashPythonSourceDumpGuard,
    BashRepeatFailureGuard,
    EditFailureGuard,
    ExploreStreakGuard,
    GuardManager,
    NoProgressHardStopGuard,
    NoSuccessfulSolutionRunGuard,
    RepeatedRuntimeErrorGuard,
    RuntimeErrorGuard,
    SingleReadStreakGuard,
    WriteFailureGuard,
    WriteNudgeGuard,
    WriteRepeatGuard,
)
from scienceflow.agent.prompts.system_prompt import (
    _code_agent_core_prompt,
    _default_system_prompt,
)
from scienceflow.agent.core.runtime.run_policy import DefaultPolicy
from scienceflow.agent.session import ScienceFlowAgentSessionFactory
from scienceflow.research.state.knowledge.context.memory_context import MemoryContextManager
from scienceflow.runtime.safety.execution.agent_runtime.tool_composition import create_tool_collection
from scienceflow.runtime.observability.interaction_log import (
    attach_workspace_interaction_logger,
    build_interaction_log_context_tag,
    set_interaction_log_context_tag,
)

if TYPE_CHECKING:
    from scienceflow.interfaces.ui.console import RichUI


@dataclass(slots=True)
class AgentConstructionOptions:
    """Typed input to the phased ScienceAgent builder."""

    workspace_dir: str | Path
    sandbox: bool
    ui: RichUI | None
    max_steps: int | None
    exec_feedback_max_chars: int
    scienceflow_stdout_max_chars: int
    sliding_window_budget_chars: int
    pinned_budget_ratio: float
    sliding_window_priority_enabled: bool
    mid_run_compact_enabled: bool
    llm_stream_timeout_sec: int
    llm_tool_stream_max_attempts: int
    llm_tool_stream_retry_base_delay_sec: float
    llm_tool_stream_retry_max_delay_sec: float
    stream_repetition_detection: bool
    stream_repetition_window_chars: int
    stream_repetition_ngram_len: int
    stream_repetition_max_repeats: int
    stream_max_output_chars_soft: int
    stream_repetition_retry_max: int
    bash_timeout_sec: float
    bash_timeout_slow_sec: float
    bash_max_output_chars: int
    bash_max_stream_line_chars: int
    bash_observation_summary_enabled: bool
    system_prompt_hook: Callable[[str], str] | None
    system_prompt_core: str | None
    path_guard_extra_roots: Sequence[str | Path] | None
    extra_env: dict[str, str] | None
    readonly_dirs: Sequence[str] | None
    run_policy: RunPolicy | None
    embedded_full_run_enabled: bool
    skip_embedded_duplicate_when_progressive_bare_success: bool
    embedded_full_run_update_result_md: bool
    embedded_full_run_timeout_sec: float | None
    fullrun_output_tail_stdout_lines: int
    fullrun_output_tail_stderr_lines: int
    fullrun_output_tail_max_chars: int
    quick_test_extrapolation_enabled: bool
    quick_test_extrapolation_budget_sec: float | None
    quick_test_extrapolation_rows: int
    fullrun_epoch_watchdog_enabled: bool
    fullrun_epoch_watchdog_budget_sec: float | None
    interaction_log_level: str
    interaction_log_full: bool
    interaction_log_color: bool
    interaction_log_llm_stream: bool
    interaction_log_layout: str
    bash_stream_to_interaction_log: bool
    bash_output_dedup_enabled: bool
    bash_output_dedup_min_repeat: int
    bash_output_dedup_summary_prefix: str
    bash_output_dedup_apply_to_memory: bool
    tool_memory_compression: bool
    include_write_edit_tools: bool
    tool_display_paths_relative: bool
    tool_display_root: str | Path | None
    write_return_full_max_chars: int
    write_return_full_max_lines: int
    write_return_head_tail_lines: int
    edit_return_full_max_chars: int
    edit_return_full_max_lines: int
    edit_return_head_tail_lines: int
    edit_return_change_ctx_lines: int
    edit_failure_top_k_candidates: int
    edit_failure_diag_max_chars: int
    file_snapshot_latest_only: bool
    grep_max_results_lines: int
    bash_success_tail_lines: int
    bash_success_tail_lines_solution: int
    bash_success_tail_lines_test: int
    bash_success_tail_lines_readonly: int
    bash_success_tail_lines_install: int
    read_success_max_lines: int
    write_auto_snapshot_enabled: bool
    write_auto_snapshot_paths: Sequence[str] | None
    write_auto_snapshot_code_extensions: Sequence[str] | None
    write_auto_snapshot_max_lines: int
    write_auto_snapshot_max_chars: int
    write_auto_snapshot_changed_context_lines: int
    write_auto_snapshot_symbol_body_lines: int
    read_overlap_guard_enabled: bool
    msg0_compress_body: bool
    on_llm_call: Callable[[dict[str, Any]], None] | None
    debug_dynamic_steps_enabled: bool
    debug_step_boost: int
    debug_max_steps_cap: int
    round_budget_prompt_cap: int
    parallel_bash_enabled: bool
    parallel_llm_tool_calls: bool
    lnr_explore_streak_inject_after: int
    lnr_single_read_streak_inject_after: int
    no_progress_hard_stop_after: int
    no_progress_hardstop_guard_grace_rounds: int
    repeated_runtime_error_threshold: int
    rotating_runtime_error_threshold: int
    rotating_distinct_types_min: int
    no_success_run_soft_threshold: int
    no_success_run_hard_threshold: int
    edit_short_old_str_chars: int
    lnr_category_skill_text: str
    lnr_phase_skill_registry: dict[str, str] | None
    lnr_phase_skill_descriptions: dict[str, str] | None
    lnr_skill_inject_max_chars: int
    lnr_periodic_no_solution_every: int
    lnr_llm_turns_log_enabled: bool
    lnr_llm_turns_log_path: str | Path | None
    sft_data_log_path: str | Path | None
    mlebench_validate_after_embedded_full_run: bool
    mlebench_data_dir: str | None
    mlebench_exp_id: str | None
    lnr_mlebench_validate_enabled: bool
    lnr_run_control_max_fix_rounds: int
    lnr_stop_after_bare_solution_success: bool
    lnr_superloop_enabled: bool
    teleport_mode: str
    stable_system_prompt: bool
    workspace_git_auto_checkpoint_enabled: bool
    workspace_git_track_globs: Sequence[str] | None
    search_round_offset: int
    skill_registry: SkillRegistry | None
    task_type: str | None
    skill_allow_names: Sequence[str] | None
    skill_tool_mode: str
    skill_allow_generic_wildcard: bool
    skill_visible_max: int
    resource_observer: Any | None
    interaction_log_session: str | None
    interaction_log_phase: str | None
    lnr_exploit_ensemble_mode: bool
    lnr_hide_future_stages: bool
    callback_ports: AgentCallbackPorts | None
    model_fields: dict[str, Any]


@dataclass(slots=True)
class _AgentConstructionState:
    workspace: Path | None = None
    tools: Any = None


def _initialize_agent_base(
    self: Any, options: AgentConstructionOptions, state: _AgentConstructionState
) -> None:
    state.workspace = Path(options.workspace_dir).resolve()
    options.include_write_edit_tools = bool(options.include_write_edit_tools)
    if (
        "systemPrompt" not in options.model_fields
        or options.model_fields["systemPrompt"] is None
    ):
        options.model_fields["systemPrompt"] = _default_system_prompt(
            parallel_bash_enabled=bool(options.parallel_bash_enabled),
            bash_file_write_mode=not options.include_write_edit_tools,
        )
    state.tools = create_tool_collection(
        state.workspace,
        sandbox=options.sandbox,
        path_guard_extra_roots=options.path_guard_extra_roots,
        max_bash_output_chars=int(options.bash_max_output_chars),
        max_bash_stream_line_chars=int(options.bash_max_stream_line_chars),
        bash_observation_summary_enabled=bool(options.bash_observation_summary_enabled),
        bash_timeout_sec=options.bash_timeout_sec,
        bash_timeout_slow_sec=options.bash_timeout_slow_sec,
        extra_env=options.extra_env,
        readonly_dirs=options.readonly_dirs,
        resource_observer=options.resource_observer,
        skill_registry=options.skill_registry,
        task_type=options.task_type,
        skill_allow_names=options.skill_allow_names,
        skill_tool_mode=options.skill_tool_mode,
        skill_allow_generic_wildcard=options.skill_allow_generic_wildcard,
        skill_visible_max=options.skill_visible_max,
        include_write_edit_tools=options.include_write_edit_tools,
        tool_display_paths_relative=options.tool_display_paths_relative,
        tool_display_root=options.tool_display_root,
        write_return_full_max_chars=int(options.write_return_full_max_chars),
        write_return_full_max_lines=int(options.write_return_full_max_lines),
        write_return_head_tail_lines=int(options.write_return_head_tail_lines),
        edit_return_full_max_chars=int(options.edit_return_full_max_chars),
        edit_return_full_max_lines=int(options.edit_return_full_max_lines),
        edit_return_head_tail_lines=int(options.edit_return_head_tail_lines),
        edit_return_change_ctx_lines=int(options.edit_return_change_ctx_lines),
        edit_failure_top_k_candidates=int(options.edit_failure_top_k_candidates),
        edit_failure_diag_max_chars=int(options.edit_failure_diag_max_chars),
        grep_max_results_lines=int(options.grep_max_results_lines),
    )
    options.model_fields.setdefault("availableTools", state.tools)
    BaseModel.__init__(self, **options.model_fields)
    self.callback_ports = options.callback_ports or AgentCallbackPorts()
    self._include_write_edit_tools = options.include_write_edit_tools
    self._tool_display_paths_relative = bool(options.tool_display_paths_relative)
    self._tool_display_root = (
        Path(options.tool_display_root).resolve()
        if options.tool_display_root is not None
        else state.workspace
    )
    self._tools_with_thought = inject_thought_into_tool_params(
        self.availableTools.to_params()
    )
    self._workspace_dir = state.workspace
    self._agent_session_factory = ScienceFlowAgentSessionFactory()
    self._host_ports = compose_agent_host_ports(self)
    self._agent_runtime_cancellation = CancellationToken()


def _initialize_agent_memory(
    self: Any, options: AgentConstructionOptions, state: _AgentConstructionState
) -> None:
    self._ui = options.ui
    self._ui_step = 0
    self._skill_registry = options.skill_registry
    self._task_type = options.task_type
    self._skill_allow_names = tuple(options.skill_allow_names or ())
    self._skill_tool_mode = str(options.skill_tool_mode or "all")
    self._skill_allow_generic_wildcard = bool(options.skill_allow_generic_wildcard)
    self._skill_visible_max = int(options.skill_visible_max or 0)
    self._resource_observer = options.resource_observer
    self._bash_timeout_sec = float(options.bash_timeout_sec)
    self._bash_timeout_slow_sec = float(options.bash_timeout_slow_sec)
    self._bash_max_output_chars = int(options.bash_max_output_chars)
    self._bash_max_stream_line_chars = int(options.bash_max_stream_line_chars)
    self._bash_observation_summary_enabled = bool(
        options.bash_observation_summary_enabled
    )
    self._extra_env: dict[str, str] | None = (
        dict(options.extra_env) if options.extra_env else None
    )
    self._embedded_full_run_enabled = bool(options.embedded_full_run_enabled)
    self._skip_embedded_duplicate_when_progressive_bare_success = bool(
        options.skip_embedded_duplicate_when_progressive_bare_success
    )
    self._embedded_full_run_update_result_md = bool(
        options.embedded_full_run_update_result_md
    )
    self._embedded_full_run_timeout_sec = options.embedded_full_run_timeout_sec
    self._fullrun_output_tail_stdout_lines = int(
        options.fullrun_output_tail_stdout_lines
    )
    self._fullrun_output_tail_stderr_lines = int(
        options.fullrun_output_tail_stderr_lines
    )
    self._fullrun_output_tail_max_chars = int(options.fullrun_output_tail_max_chars)
    self._quick_test_extrapolation_enabled = bool(
        options.quick_test_extrapolation_enabled
    )
    self._quick_test_extrapolation_budget_sec = (
        options.quick_test_extrapolation_budget_sec
    )
    self._quick_test_extrapolation_rows = int(options.quick_test_extrapolation_rows)
    self._fullrun_epoch_watchdog_enabled = bool(options.fullrun_epoch_watchdog_enabled)
    self._fullrun_epoch_watchdog_budget_sec = options.fullrun_epoch_watchdog_budget_sec
    self._embedded_full_run_done = False
    self._consecutive_write_syntax_fails: int = 0
    self._consecutive_infra_errors: int = 0
    self._initial_solution_sha: str | None = None
    if options.max_steps is not None:
        self.max_steps = int(options.max_steps)
    self._exec_feedback_max_chars = int(options.exec_feedback_max_chars)
    self._scienceflow_stdout_max_chars = int(options.scienceflow_stdout_max_chars)
    self._llm_stream_timeout_sec = int(options.llm_stream_timeout_sec)
    self._llm_tool_stream_max_attempts = max(
        1, int(options.llm_tool_stream_max_attempts)
    )
    self._llm_tool_stream_retry_base_delay_sec = float(
        options.llm_tool_stream_retry_base_delay_sec
    )
    self._llm_tool_stream_retry_max_delay_sec = float(
        options.llm_tool_stream_retry_max_delay_sec
    )
    self._stream_repetition_detection = bool(options.stream_repetition_detection)
    self._stream_repetition_window_chars = max(
        256, int(options.stream_repetition_window_chars)
    )
    self._stream_repetition_ngram_len = max(
        32, int(options.stream_repetition_ngram_len)
    )
    self._stream_repetition_max_repeats = max(
        2, int(options.stream_repetition_max_repeats)
    )
    self._stream_max_output_chars_soft = int(options.stream_max_output_chars_soft)
    self._stream_repetition_retry_max = max(0, int(options.stream_repetition_retry_max))
    self._tool_memory_compression = bool(options.tool_memory_compression)
    self._file_snapshot_latest_only = bool(options.file_snapshot_latest_only)
    self._mid_run_compact_enabled = bool(options.mid_run_compact_enabled)
    self._bash_output_dedup_enabled = bool(options.bash_output_dedup_enabled)
    self._bash_output_dedup_min_repeat = int(options.bash_output_dedup_min_repeat)
    self._bash_output_dedup_summary_prefix = str(
        options.bash_output_dedup_summary_prefix
    )
    self._bash_output_dedup_apply_to_memory = bool(
        options.bash_output_dedup_apply_to_memory
    )
    self._grep_max_results_lines = int(options.grep_max_results_lines)
    self._interaction_log_layout = (
        "split"
        if str(options.interaction_log_layout or "flat").strip().lower()
        in {"split", "lhr_split"}
        else "flat"
    )
    self._memory_ctx = MemoryContextManager(
        self.memory,
        self._workspace_dir,
        budget_chars=int(options.sliding_window_budget_chars),
        path_guard_extra_roots=options.path_guard_extra_roots,
        tool_memory_compression=self._tool_memory_compression,
        bash_success_tail_lines=int(options.bash_success_tail_lines),
        bash_success_tail_lines_solution=int(options.bash_success_tail_lines_solution),
        bash_success_tail_lines_test=int(options.bash_success_tail_lines_test),
        bash_success_tail_lines_readonly=int(options.bash_success_tail_lines_readonly),
        bash_success_tail_lines_install=int(options.bash_success_tail_lines_install),
        read_success_max_lines=int(options.read_success_max_lines),
        sliding_window_priority_enabled=bool(options.sliding_window_priority_enabled),
        pinned_budget_ratio=float(options.pinned_budget_ratio),
        bash_output_dedup_apply_to_memory=self._bash_output_dedup_apply_to_memory,
        bash_output_dedup_min_repeat=self._bash_output_dedup_min_repeat,
        bash_output_dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
        write_auto_snapshot_enabled=bool(options.write_auto_snapshot_enabled),
        write_auto_snapshot_paths=options.write_auto_snapshot_paths,
        write_auto_snapshot_code_extensions=options.write_auto_snapshot_code_extensions,
        write_auto_snapshot_max_lines=int(options.write_auto_snapshot_max_lines),
        write_auto_snapshot_max_chars=int(options.write_auto_snapshot_max_chars),
        write_auto_snapshot_changed_context_lines=int(
            options.write_auto_snapshot_changed_context_lines
        ),
        write_auto_snapshot_symbol_body_lines=int(
            options.write_auto_snapshot_symbol_body_lines
        ),
        read_overlap_guard_enabled=bool(options.read_overlap_guard_enabled),
        msg0_compress_body=bool(options.msg0_compress_body),
    )
    self._tool_output_artifacts = self._make_tool_output_artifact_store(
        self._workspace_dir
    )
    self._resource_feedback_memory_deduper = ResourceFeedbackMemoryDeduper()


def _initialize_agent_prompt_and_logs(
    self: Any, options: AgentConstructionOptions, state: _AgentConstructionState
) -> None:
    self._system_prompt_hook = options.system_prompt_hook
    self._system_prompt_core = (
        options.system_prompt_core
        if options.system_prompt_core is not None
        else _code_agent_core_prompt(
            bash_file_write_mode=not self._include_write_edit_tools
        )
        if bool(options.stable_system_prompt)
        and str(options.teleport_mode or "off") in ("", "off")
        else ""
    )
    self._ws_interaction_log = attach_workspace_interaction_logger(
        self._workspace_dir,
        color=bool(options.interaction_log_color),
        layout=self._interaction_log_layout,
    )
    _ilog_sess = (options.interaction_log_session or "draft").strip().lower() or "draft"
    _ilog_tag = build_interaction_log_context_tag(
        _ilog_sess, options.interaction_log_phase
    )
    self._lnr_interaction_log_session = _ilog_sess
    _ilp = (
        str(options.interaction_log_phase).strip().lower()
        if options.interaction_log_phase is not None
        and str(options.interaction_log_phase).strip()
        else ""
    )
    self._lnr_interaction_log_phase: str | None = _ilp or None
    self._lnr_exploit_ensemble_mode = bool(options.lnr_exploit_ensemble_mode)
    self._lnr_hide_future_stages = bool(options.lnr_hide_future_stages)
    self._teleport_mode = str(options.teleport_mode or "off")
    self._stable_system_prompt = bool(options.stable_system_prompt)
    self._workspace_git_auto_checkpoint_enabled = bool(
        options.workspace_git_auto_checkpoint_enabled
    )
    self._workspace_git_track_globs = tuple(
        options.workspace_git_track_globs or ("*.py", "*.md")
    )
    self._lnr_phase_header = ""
    self._interaction_log_ctx_token: Any = set_interaction_log_context_tag(_ilog_tag)
    self._run_policy: RunPolicy = (
        options.run_policy
        if options.run_policy is not None
        else DefaultPolicy(recovery_rounds=3)
    )
    self._interaction_log_policy = resolve_interaction_log_policy(
        level=options.interaction_log_level,
        legacy_full=bool(options.interaction_log_full),
        legacy_llm_stream=bool(options.interaction_log_llm_stream),
        bash_output_dedup_enabled=self._bash_output_dedup_enabled,
        bash_output_dedup_min_repeat=self._bash_output_dedup_min_repeat,
        bash_output_dedup_summary_prefix=self._bash_output_dedup_summary_prefix,
    )
    self._interaction_log_full = self._interaction_log_policy.tool_call_full
    self._interaction_log_llm_stream = self._interaction_log_policy.llm_stream_to_file
    self._bash_stream_to_interaction_log = bool(options.bash_stream_to_interaction_log)


def _initialize_agent_state(
    self: Any, options: AgentConstructionOptions, state: _AgentConstructionState
) -> None:
    self.last_run_tokens_in: int = 0
    self.last_run_tokens_out: int = 0
    self.last_run_tokens_cached: int = 0
    self.last_run_llm_calls: int = 0
    self.last_run_compact_tokens_in: int = 0
    self.last_run_compact_tokens_out: int = 0
    self.last_run_compact_tokens_cached: int = 0
    self.last_run_compact_llm_calls: int = 0
    self.last_run_route_tokens_in: int = 0
    self.last_run_route_tokens_out: int = 0
    self.last_run_route_tokens_cached: int = 0
    self.last_run_route_llm_calls: int = 0
    self._run_tokens_in: int = 0
    self._run_tokens_out: int = 0
    self._run_tokens_cached: int = 0
    self._run_llm_calls: int = 0
    self._run_route_tokens_in: int = 0
    self._run_route_tokens_out: int = 0
    self._run_route_tokens_cached: int = 0
    self._run_route_llm_calls: int = 0
    self._run_compact_tokens_in: int = 0
    self._run_compact_tokens_out: int = 0
    self._run_compact_tokens_cached: int = 0
    self._run_compact_llm_calls: int = 0
    self._on_llm_call: Callable[[dict[str, Any]], None] | None = options.on_llm_call
    self._call_seq: int = 0
    self._current_round: int = 0
    self._round_budget_enabled: bool = True
    self._search_round_offset: int = max(0, int(options.search_round_offset or 0))
    self._repl_session_run_index: int | None = None
    self._debug_dynamic_steps_enabled = bool(options.debug_dynamic_steps_enabled)
    self._debug_step_boost = int(options.debug_step_boost)
    self._debug_max_steps_cap = int(options.debug_max_steps_cap)
    self._round_budget_prompt_cap = int(options.round_budget_prompt_cap)
    self._effective_max_steps: int = int(self.max_steps)
    self._parallel_bash_enabled = bool(options.parallel_bash_enabled)
    self._parallel_llm_tool_calls = bool(options.parallel_llm_tool_calls)
    self._lnr_explore_streak_inject_after = int(options.lnr_explore_streak_inject_after)
    self._lnr_single_read_streak_inject_after = int(
        options.lnr_single_read_streak_inject_after
    )
    self._lnr_category_skill_text = str(options.lnr_category_skill_text or "")
    self._lnr_phase_skill_registry: dict[str, str] = dict(
        options.lnr_phase_skill_registry or {}
    )
    self._lnr_phase_skill_descriptions: dict[str, str] = dict(
        options.lnr_phase_skill_descriptions or {}
    )
    self._lnr_injected_skills: set[str] = set()
    self._lnr_skill_inject_max_chars = int(options.lnr_skill_inject_max_chars)
    self._lnr_periodic_no_solution_every = int(options.lnr_periodic_no_solution_every)
    self._lnr_fresh_hint_injected: bool = False
    self._lnr_llm_turns_log_enabled = bool(options.lnr_llm_turns_log_enabled)
    self._mlebench_validate_after_embedded_full_run = bool(
        options.mlebench_validate_after_embedded_full_run
    )
    _mdd = options.mlebench_data_dir
    self._mlebench_data_dir = (
        str(_mdd).strip() if isinstance(_mdd, str) and _mdd.strip() else None
    )
    _me = options.mlebench_exp_id
    self._mlebench_exp_id = (
        str(_me).strip() if isinstance(_me, str) and _me.strip() else None
    )
    self._lnr_mlebench_validate_enabled = bool(options.lnr_mlebench_validate_enabled)
    self._lnr_run_control_max_fix_rounds = int(options.lnr_run_control_max_fix_rounds)
    self._lnr_stop_after_bare_solution_success = bool(
        options.lnr_stop_after_bare_solution_success
    )
    self._lnr_result_md_after_success_pending = False
    self._lnr_superloop_enabled = bool(options.lnr_superloop_enabled)
    self._run_control_user_injections = 0
    self._lnr_snapshot_ok = False
    self._lnr_snapshot_reason = ""
    self._lnr_last_valid_solution_sha = ""
    self._lnr_last_valid_solution_rel_path = "solution.py"
    self._lnr_last_valid_submission_sha = ""
    self._lnr_last_valid_bash_cmd = ""
    self._lnr_last_valid_bare_run_is_local = False
    self._lnr_stage_journal_pending = False
    self._lnr_committed_ledger_step_count = None
    self._lnr_ledger_committed = False
    self._lnr_ledger_step_count = 0
    self._lnr_ledger_new_entries = 0
    self._lnr_stage_commit_guard_failed = False
    _nlp = options.lnr_llm_turns_log_path
    self._lnr_llm_turns_log_path: Path | None = Path(_nlp).resolve() if _nlp else None
    _sftp = options.sft_data_log_path
    self._sft_data_log_path: Path | None = Path(_sftp).resolve() if _sftp else None
    self._sandbox = bool(options.sandbox)
    self._path_guard_extra_roots = tuple(
        (
            Path(str(p)).expanduser().resolve()
            for p in options.path_guard_extra_roots or ()
            if str(p).strip()
        )
    )
    self._recent_read_history: deque[tuple[str, str]] = deque(maxlen=3)
    self._last_read_sha_by_path: dict[str, str] = {}


def _initialize_agent_guards(
    self: Any, options: AgentConstructionOptions, state: _AgentConstructionState
) -> None:
    def _reset_solution_write_syntax_fail_counter() -> None:
        self._consecutive_write_syntax_fails = 0

    self._guard_manager = GuardManager(
        [
            WriteNudgeGuard(
                get_current_solution_sha=self._sha256_of_solution,
                get_initial_solution_sha=lambda: self._initial_solution_sha,
                on_solution_write_success=_reset_solution_write_syntax_fail_counter,
            ),
            WriteFailureGuard(),
            WriteRepeatGuard(),
            RuntimeErrorGuard(
                rotating_runtime_error_threshold=int(
                    options.rotating_runtime_error_threshold
                ),
                rotating_distinct_types_min=int(options.rotating_distinct_types_min),
            ),
            EditFailureGuard(short_old_str_chars=int(options.edit_short_old_str_chars)),
            NoSuccessfulSolutionRunGuard(
                soft_threshold=int(options.no_success_run_soft_threshold),
                hard_threshold=int(options.no_success_run_hard_threshold),
            ),
            BashRepeatFailureGuard(),
            RepeatedRuntimeErrorGuard(
                same_error_threshold=int(options.repeated_runtime_error_threshold)
            ),
            SingleReadStreakGuard(
                threshold=int(options.lnr_single_read_streak_inject_after),
                parallel_llm_tool_calls_enabled=bool(options.parallel_llm_tool_calls),
            ),
            BashPythonSourceDumpGuard(),
            ExploreStreakGuard(
                threshold=int(options.lnr_explore_streak_inject_after),
                get_current_solution_sha=self._sha256_of_solution,
                get_initial_solution_sha=lambda: self._initial_solution_sha,
                category_skill_text=self._lnr_category_skill_text,
                max_skill_chars=self._lnr_skill_inject_max_chars,
            ),
            NoProgressHardStopGuard(
                threshold=int(options.no_progress_hard_stop_after),
                get_current_solution_sha=self._sha256_of_solution,
                get_initial_solution_sha=lambda: self._initial_solution_sha,
                get_embedded_metric_token=self._embedded_full_run_metric_token,
                grace_rounds_after_peer_inject=int(
                    options.no_progress_hardstop_guard_grace_rounds
                ),
            ),
        ],
        phase_header=self._lnr_phase_header,
    )


__all__ = [
    "AgentConstructionOptions",
    "_AgentConstructionState",
    "_initialize_agent_base",
    "_initialize_agent_guards",
    "_initialize_agent_memory",
    "_initialize_agent_prompt_and_logs",
    "_initialize_agent_state",
]
