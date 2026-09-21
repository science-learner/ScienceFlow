"""MemoryContextManager responsibility: tool results."""

from __future__ import annotations

from typing import Any

from inquirycraft.tools import ToolResult

from scienceflow.runtime.core.process.commands import looks_like_bare_solution_run
from scienceflow.research.state.knowledge.context.context.results.base import (
    _WRITE_REPEAT_TO_SAME_FILE_COACHING,
)
from scienceflow.research.state.knowledge.context.context.projection.bash_projection import (
    _tool_call_signature,
)
from scienceflow.research.state.knowledge.context.context.compression.clone_compression import (
    _supersede_stale_compressed_records,
)
from scienceflow.research.state.knowledge.context.context.projection.source_projection import (
    _all_messages,
)
from scienceflow.research.state.knowledge.context.context.results.tool_results import (
    _strip_bash_stderr_section_from_block,
)


from scienceflow.research.state.knowledge.context.context.runtime.tool_result_runtime import (
    ToolResultProjection,
    _append_edit_failure_feedback,
    _append_write_snapshot,
    _apply_redundant_read_projection,
    _compress_tool_observation,
    _update_tool_result_state,
)


def record_tool_result(
    self,
    tool_name: str,
    args: dict[str, Any],
    result: ToolResult,
    *,
    raw_id: str = "",
) -> str:
    """Project a tool result into bounded memory while updating snapshot state."""
    rel = str(args.get("path") or "").replace("\\", "/").lstrip("/")
    overlap, overlap_summary = _update_tool_result_state(
        self, tool_name, args, result, rel=rel, raw_id=raw_id
    )
    command = str(args.get("command") or "") if tool_name == "bash" else ""
    bare_solution = bool(
        tool_name == "bash" and not result.error and looks_like_bare_solution_run(command)
    )
    observation = _append_edit_failure_feedback(
        self,
        str(result),
        rel=rel,
        edit_failed=tool_name == "edit" and bool(result.error),
    )
    if bare_solution:
        observation = _strip_bash_stderr_section_from_block(observation)
        self._write_counter_by_path.clear()
    signature = (
        _tool_call_signature(tool_name, args)
        if self._tool_memory_compression and not result.error and tool_name in ("bash", "read")
        else None
    )
    projection = ToolResultProjection(
        rel=rel,
        observation=observation,
        command=command,
        bare_solution_bash=bare_solution,
        signature=signature,
        repeated=bool(signature and signature in self._seen_tool_signatures),
        read_overlap_redundant=overlap,
        read_overlap_summary=overlap_summary,
    )
    if self._released_full_signatures and not projection.repeated:
        _supersede_stale_compressed_records(
            memory=self._memory,
            released_sigs=self._released_full_signatures,
            seen_sigs=self._seen_tool_signatures,
        )
    _compress_tool_observation(self, projection, tool_name, args, result, raw_id=raw_id)
    _apply_redundant_read_projection(self, projection, args)
    _append_write_snapshot(self, projection, tool_name, args, result)
    if tool_name == "write" and not result.error and rel:
        count = self._write_counter_by_path.get(rel, 0) + 1
        self._write_counter_by_path[rel] = count
        if count >= 2:
            projection.observation += _WRITE_REPEAT_TO_SAME_FILE_COACHING.format(
                n=count, rel=rel
            )
    if signature is not None and not projection.repeated:
        self._seen_tool_signatures[signature] = len(_all_messages(self._memory))
    return projection.observation
