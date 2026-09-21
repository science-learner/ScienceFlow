from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scienceflow.runtime.safety.execution.agent_runtime.tool_composition import create_tool_collection

EXPECTED_TOOL_SCHEMA_SHA256 = "cc0aa04ca49d9f72f2b3ceb87073758650a65427136698568e2421ebdc93635b"


def _schema_payload(workspace: Path) -> tuple[list[dict], bytes]:
    params = create_tool_collection(workspace).to_params()
    encoded = json.dumps(
        params,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode()
    return params, encoded


def test_scienceflow_tool_schema_and_order_are_context_stable(tmp_path: Path) -> None:
    params, encoded = _schema_payload(tmp_path)

    assert [item["function"]["name"] for item in params] == [
        "bash",
        "write",
        "edit",
        "read",
        "grep",
        "glob",
        "ls",
    ]
    assert len(encoded) == 5391
    assert hashlib.sha256(encoded).hexdigest() == EXPECTED_TOOL_SCHEMA_SHA256


@pytest.mark.asyncio
async def test_workspace_tool_feedback_is_context_stable(tmp_path: Path) -> None:
    tools = create_tool_collection(tmp_path)

    written = await tools.execute(
        name="write",
        tool_input={"path": "notes/example.txt", "content": "alpha\nbeta\n"},
    )
    read = await tools.execute(
        name="read",
        tool_input={"path": "notes/example.txt", "offset": 1, "limit": 200},
    )
    listed = await tools.execute(name="ls", tool_input={"path": "."})
    globbed = await tools.execute(name="glob", tool_input={"pattern": "**/*.txt"})

    assert (written.output, written.error, written.system) == (
        "Written notes/example.txt (2 lines, 11 bytes, sha256~e49c81e2d2f84e25)\n"
        "     1|alpha\n"
        "     2|beta",
        None,
        None,
    )
    assert (read.output, read.error, read.system) == (
        "[notes/example.txt: 2 lines total]\n     1|alpha\n     2|beta",
        None,
        None,
    )
    assert listed.output == "[ls: .]\nDIR notes/\n[1 entries]"
    assert globbed.output == "[glob: '**/*.txt' under .]\nnotes/example.txt\n[1 matches]"


def test_scienceflow_workspace_shims_export_inquirycraft_objects() -> None:
    from inquirycraft.tools import (
        PathGuard,
        atomic_write,
        looks_like_write_placeholder_mimicry,
    )
    from inquirycraft.tools import PathGuard as CompatibilityPathGuard
    from inquirycraft.tools import atomic_write as compatibility_atomic_write

    assert CompatibilityPathGuard is PathGuard
    assert compatibility_atomic_write is atomic_write
    assert callable(looks_like_write_placeholder_mimicry)


def test_scienceflow_process_and_shell_reducer_shims_export_inquirycraft_objects() -> None:
    from inquirycraft.runtime import live_process_group_ids, spawn_shell
    from inquirycraft.tools import deduplicate_repeated_blocks

    from scienceflow.runtime.core.process import utils as process_compatibility
    from scienceflow.runtime.safety.tooling.workspace.shell_output import (
        _dedup_repeated_blocks as reducer_compatibility,
    )

    assert process_compatibility.live_process_group_ids is live_process_group_ids
    assert process_compatibility.spawn_shell is spawn_shell
    assert reducer_compatibility is deduplicate_repeated_blocks
