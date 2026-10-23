"""Shared fixtures and helpers for split science_agent_repl contracts."""

from __future__ import annotations

import json

import logging

import shutil

import subprocess

from pathlib import Path

from types import SimpleNamespace

from typing import Any

from unittest.mock import MagicMock

import pytest

from click import ClickException

from inquirycraft.memory import Memory, Message

from scienceflow.agent.core.runtime.run_policy import AutoContinuePolicy, DefaultPolicy

from scienceflow.agent import ScienceAgent

from scienceflow.runtime.safety.policy.agent_policies.bash_command_classifier import (
    classify_bash_command,
)

from scienceflow.runtime.observability.agent_io.interaction_log import (
    format_tool_call_lines_for_interaction_log,
)

from scienceflow.research.solver.lnr.orchestration.agent_hooks import (
    _adapt_write_tool_hint_for_bash_only,
)

from scienceflow.agent.prompts.system_prompt import _code_agent_core_prompt

from scienceflow.foundation.config.schema.settings import (
    Config,
    apply_profile_overrides,
    apply_repl_manifest_defaults,
)

from scienceflow.interfaces.cli import _repl_resolve_runtime_options

from scienceflow.research.state.repl_context import (
    _plain_repl_auto_continue_session,
    _repl_bash_file_write_prompt,
    _repl_code_agent_append_prompt,
    _repl_code_organization_prompt,
    _repl_environment_context_prompt,
    _repl_workspace_git_prompt,
)

from scienceflow.runtime.safety.execution.agent_runtime.tool_composition import create_tool_collection

from scienceflow.runtime.core.support.node_paths import find_node_log_path

from scienceflow.research.state.workspace.storage.git import (
    archive_workspace_candidate_artifact,
    auto_checkpoint_workspace_source,
    ensure_workspace_source_git,
    render_workspace_gitignore,
    workspace_source_changed,
)

class _FakeLLM:
    """Yields pre-baked assistant messages from ``ask_tool_stream``."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        if not self._replies:
            raise RuntimeError("no fake replies left")
        msg = self._replies.pop(0)
        content = getattr(msg, "content", None) or ""
        if content:
            await handle.put(content)
        handle.finish()
        return msg

class _RecordingFakeLLM(_FakeLLM):
    """Like ``_FakeLLM`` but records ``messages=`` passed to ``ask_tool_stream``."""

    def __init__(self, replies: list[Any]) -> None:
        super().__init__(replies)
        self.message_snapshots: list[list[Any]] = []
        self.kwarg_snapshots: list[dict[str, Any]] = []

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        msgs = kwargs.get("messages")
        self.message_snapshots.append(list(msgs) if msgs is not None else [])
        self.kwarg_snapshots.append({k: v for k, v in kwargs.items() if k != "handle"})
        return await super().ask_tool_stream(handle=handle, **kwargs)

def _tool_msg(name: str, args: dict[str, Any], tc_id: str = "tc1") -> Any:
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id=tc_id,
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args),
                ),
            ),
        ],
        content="",
    )

__all__ = tuple(name for name in globals() if not name.startswith("__"))
