"""Shared fixtures and helpers for split tool_memory_compression contracts."""

from __future__ import annotations

import json

from types import SimpleNamespace

from unittest import mock

from inquirycraft.memory import Message

from inquirycraft.tools import ToolResult

from scienceflow.agent import (
    _DEFAULT_SYSTEM,
    _compress_tool_call_for_memory,
    _looks_like_write_placeholder_mimicry,
)

from scienceflow.foundation.config.runtime.agent_constants import _SCRUBBED_WRITE_PLACEHOLDER_NOTE

from scienceflow.research.state.knowledge.memory.agent.memory_utils import (
    _assistant_message_from_api,
    scrub_last_assistant_write_tool_call_content,
)

from scienceflow.research.state.knowledge.memory.records.agent_records import create_agent_memory

from scienceflow.research.state.knowledge.context.memory_context import (
    MemoryContextManager,
    _bash_is_pure_read_only,
    bash_command_dumps_python_source,
    _message_char_len,
    _READ_OVERLAP_COACHING,
    _AUTO_SNAPSHOT_PREFIX,
    _build_write_auto_snapshot_block,
    compress_bash_tool_output_for_memory,
    compress_edit_success_output_for_memory,
    compress_read_output_for_memory,
)

def _record_successful_bash_with_tail_config(
    tmp_path,
    *,
    command: str,
    line_count: int = 20,
) -> str:
    mem = create_agent_memory(tmp_path / ("m_" + str(abs(hash(command)))), "Draft", 200)
    ws = tmp_path / ("w_" + str(abs(hash(command))))
    ws.mkdir()
    ctx = MemoryContextManager(
        mem,
        ws,
        tool_memory_compression=True,
        bash_success_tail_lines=6,
        bash_success_tail_lines_solution=3,
        bash_success_tail_lines_test=9,
        bash_success_tail_lines_readonly=4,
        bash_success_tail_lines_install=2,
    )
    body = "\n".join(f"L{i:02d}" for i in range(line_count))
    return ctx.record_tool_result(
        "bash",
        {"command": command},
        ToolResult(output="[exit=0, 1.0s]\n" + body),
    )

__all__ = tuple(name for name in globals() if not name.startswith("__"))
