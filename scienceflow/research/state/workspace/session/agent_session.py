# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""ScienceAgent responsibility: workspace swapping, tools, memory, and runtime session rebinding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from scienceflow.research.state.workspace.session.agent_contracts import _ON_LLM_CALL_UNSET
from scienceflow.runtime.safety.policy.agent_policies.artifacts import ToolOutputArtifactStore
from scienceflow.agent.core.runtime.run_policy import RunPolicy
from scienceflow.research.state.knowledge.skills.catalog.registry import SkillRegistry

from scienceflow.research.state.workspace.adapters.agent_runtime import swap_workspace_runtime


def swap_workspace(
    self,
    *,
    workspace_dir: str | Path,
    path_guard_extra_roots: Sequence[str | Path] | None = None,
    readonly_dirs: Sequence[str] | None = None,
    run_policy: RunPolicy | None = None,
    max_steps_override: int | None = None,
    bash_timeout_sec: float | None = None,
    bash_timeout_slow_sec: float | None = None,
    extra_env: dict[str, str] | None = None,
    skill_registry: SkillRegistry | None = None,
    task_type: str | None = None,
    skill_allow_names: Sequence[str] | None = None,
    skill_tool_mode: str | None = None,
    skill_allow_generic_wildcard: bool | None = None,
    skill_visible_max: int | None = None,
    resource_observer: Any | None = None,
    interaction_log_color: bool | None = None,
    interaction_log_session: str | None = None,
    interaction_log_phase: str | None = None,
    copy_memory_storage_to_new_workspace: bool = True,
    sliding_window_budget_chars: int | None = None,
    lnr_llm_turns_log_path: str | Path | None = None,
    sft_data_log_path: str | Path | None = None,
    on_llm_call: Callable[[dict[str, Any]], None] | None | object = _ON_LLM_CALL_UNSET,
) -> None:
    """Teleport: swap the underlying workspace **without** reconstructing the agent.

    Used when *teleport_mode* is on for **A-class fork** continuations
    (same-branch / explore↔exploit_explore). The LLM's view (``systemPrompt``,
    ``_system_prompt_hook``, ``memory`` chat history, all ``_consecutive_*``
    counters, pinned messages) is preserved; only orchestrator-side state
    (workspace dir, tool collection, path guard, readonly dirs, run policy,
    interaction log file, optional max_steps / bash_timeout) is rebuilt.

    Round counter behavior: ``_current_round`` is added into
    ``_search_round_offset`` and reset to 0. In teleport mode this counter
    is used for logs and legacy callers, while the dynamic Round Budget is
    deliberately kept out of the system prompt for KV-cache stability.

    Memory storage is hot-swapped so future appends persist under the new
    node's ``.agent_memory/Draft/short_term.json`` instead of the old
    node's. The in-memory record list is preserved by copying records over.

    B/C-class forks should NOT use this method — they want a fresh
    ScienceAgent with deep-copied parent memory + functional first-user msg
    (see :func:`run_agent_session_clone` teleport branch).
    """
    swap_workspace_runtime(self, locals())


def _make_tool_output_artifact_store(
    self,
    workspace_dir: str | Path,
    *,
    log_dir_override: str | Path | None = None,
) -> ToolOutputArtifactStore:
    if getattr(self, "_interaction_log_layout", "flat") == "split":
        return ToolOutputArtifactStore(
            workspace_dir,
            output_parts=("interaction", "tool_outputs"),
            mirror_parts=("traj_interaction", "tool_outputs"),
            log_dir_override=log_dir_override,
        )
    return ToolOutputArtifactStore(workspace_dir, log_dir_override=log_dir_override)
