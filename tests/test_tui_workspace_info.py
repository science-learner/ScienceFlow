"""Read-only inspection and explicit permission guidance remain available together."""

import json
from types import SimpleNamespace

import pytest

from scienceflow.interfaces.ui.llm_cli import TuiPoolRuntimeFactory


@pytest.mark.asyncio
async def test_inspection_without_shell_and_escape_rejection(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "hello.txt").write_text("hello")
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    factory = TuiPoolRuntimeFactory()
    tool = factory._tools().get("workspace_info")
    context = SimpleNamespace(workspace=root)
    result = await tool.execute({"action": "pwd"}, context)
    assert json.loads(result.output)["workspace"] == str(root)
    result = await tool.execute({"action": "list"}, context)
    assert {"name": "outside", "kind": "symlink"} in json.loads(result.output)[
        "entries"
    ]
    result = await tool.execute({"action": "stat", "path": "hello.txt"}, context)
    assert json.loads(result.output)["bytes"] == 5
    for path in ["../", "outside", str(tmp_path)]:
        result = await tool.execute({"action": "list", "path": path}, context)
        assert result.error
    assert (root / "hello.txt").read_text() == "hello"


@pytest.mark.asyncio
async def test_listing_is_bounded(tmp_path):
    for number in range(110):
        (tmp_path / str(number)).touch()
    tool = TuiPoolRuntimeFactory()._tools().get("workspace_info")
    result = await tool.execute({"action": "list"}, SimpleNamespace(workspace=tmp_path))
    data = json.loads(result.output)
    assert len(data["entries"]) == 100
    assert data["truncated"]
