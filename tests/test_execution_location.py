"""Local placement keeps chat ownership separate from execution and locking."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from scienceflow.interfaces.cli import main
from scienceflow.interfaces.ui.research.control.tasks.records import TaskIndex
from scienceflow.interfaces.ui.research.preparation import location as preparation
from scienceflow.research.onboarding import LongResearchDraft
from scienceflow.research.onboarding.support.preparation import parse_preparation
from scienceflow.runtime.parallel.control.selection import workspace_aliases
from scienceflow.runtime.parallel.location import (
    allocate_local_workspace,
    resolve_local_root,
)


pytestmark = pytest.mark.usefixtures("research_model_registry")


def test_defaults_preserve_both_layouts(tmp_path):
    assert allocate_local_workspace(tmp_path).task_workspace == tmp_path
    entry = TaskIndex(tmp_path).reserve("circle-packing")
    assert Path(entry["workspace"]) == tmp_path / "tasks/task-0001"
    assert entry["execution_location"]["management_workspace"] == str(tmp_path)


def test_external_root_is_unique_across_management_workspaces(tmp_path):
    root = tmp_path / "实验 space"
    indexes = [TaskIndex(tmp_path / f"chat-{i}") for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(
            pool.map(lambda index: index.reserve("math", str(root)), indexes)
        )
    assert {entry["number"] for entry in entries} == {1}
    assert len({entry["workspace"] for entry in entries}) == 8
    assert all(Path(entry["workspace"]).parent == root for entry in entries)
    assert all(Path(entry["workspace"]).is_dir() for entry in entries)


def test_relative_root_and_symlink_are_canonical(tmp_path):
    management = tmp_path / "chat"
    management.mkdir()
    root = tmp_path / "real"
    root.mkdir()
    (management / "alias").symlink_to(root, target_is_directory=True)
    result = allocate_local_workspace(management, requested_root="alias")
    assert result.allocation_root == root
    assert result.task_workspace.parent == root
    assert resolve_local_root(management, "../real") == root


@pytest.mark.parametrize(
    "value", ["ssh://host/path", "user@host:/path", "host:dir", "", "\x00"]
)
def test_invalid_or_remote_root_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        resolve_local_root(tmp_path, value)
    assert not list(tmp_path.iterdir())


def test_no_overwrite_or_escape_through_default_symlink(tmp_path):
    index = TaskIndex(tmp_path)
    existing = tmp_path / "tasks/task-0001"
    existing.mkdir(parents=True)
    marker = existing / "keep"
    marker.write_text("user data")
    with pytest.raises(FileExistsError):
        index.reserve("task")
    assert marker.read_text() == "user data"
    assert index.snapshot() == []
    with pytest.raises(ValueError, match="not a directory"):
        index.reserve("task", str(marker))


def test_failed_index_write_removes_only_new_empty_child(tmp_path, monkeypatch):
    from scienceflow.interfaces.ui.research.control.tasks import records

    root = tmp_path / "external"
    root.mkdir()
    keep = root / "keep"
    keep.write_text("user data")
    monkeypatch.setattr(records, "write_record", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        TaskIndex(tmp_path / "chat").reserve("task", str(root))
    assert list(root.iterdir()) == [keep]


def test_external_run_is_visible_only_to_its_management_index(tmp_path):
    index = TaskIndex(tmp_path / "chat")
    entry = index.reserve("math", str(tmp_path / "external"))
    row = dict(
        run_id="run-one",
        status="stopped",
        control_workspace=entry["workspace"],
        execution_location=entry["execution_location"],
        task_roots=[],
    )
    assert index.workspace in workspace_aliases(row)
    assert index.sync([row])[0]["run_id"] == "run-one"
    assert TaskIndex(tmp_path / "other").sync([row]) == []
    # Existing records keep their paths without requiring new metadata.
    assert Path(entry["workspace"]) in workspace_aliases(
        {"control_workspace": entry["workspace"]}
    )


def test_explicit_location_keeps_other_arguments(tmp_path):
    text, root = preparation.extract_execution_root(
        'circle-packing workers=2 workspace="实验 space"'
    )
    assert text == "circle-packing workers=2"
    assert root == "实验 space"
    for invalid in ("task workspace=", "task workspace=a workspace=b"):
        with pytest.raises(ValueError):
            preparation.extract_execution_root(invalid)


def test_cli_passes_root_and_rejects_manifest_override(tmp_path, monkeypatch):
    from scienceflow.interfaces.cli.commands.run import control

    launch = Mock()
    monkeypatch.setattr(control, "managed_run", launch)
    result = CliRunner().invoke(
        main, ["run", "circle-packing", "--execution-root", "experiments"]
    )
    assert result.exit_code == 0, result.output
    assert launch.call_args.kwargs["execution_root"] == "experiments"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("tasks: []")
    result = CliRunner().invoke(
        main, ["run", "--manifest", str(manifest), "--execution-root", "other"]
    )
    assert result.exit_code == 2
    assert "Cannot combine --manifest with --execution-root" in result.output
    assert launch.call_count == 1


def test_relative_data_stays_relative_to_management(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    draft = LongResearchDraft()
    fields, *_ = parse_preparation(
        json.dumps({"fields": {"input_data_dir": "data"}}), draft, tmp_path
    )
    assert fields["input_data_dir"] == str(data)


@pytest.mark.asyncio
async def test_natural_location_uses_tool_free_extraction_before_allocation(
    tmp_path, monkeypatch
):
    owner = SimpleNamespace(
        workspace=tmp_path / "chat",
        _managed=None,
        _preparation=SimpleNamespace(task=None),
    )
    model = AsyncMock(return_value=json.dumps({"execution_root": "../external"}))
    monkeypatch.setattr(preparation, "run_preparation_task", model)
    root = await preparation.resolve_natural_location(
        owner, object(), "circle packing 放到 ../external"
    )
    _, result = await preparation.allocate_preparation_location(
        owner, object(), "circle packing", requested_root=root
    )
    assert result.task_workspace.parent == tmp_path / "external"
    assert model.call_args.kwargs == {"tool_free": True}
    assert owner._preparation.task is None


@pytest.mark.asyncio
async def test_invented_location_and_cancel_never_allocate(tmp_path, monkeypatch):
    owner = SimpleNamespace(
        workspace=tmp_path / "chat",
        _managed=None,
        _preparation=SimpleNamespace(task=None),
    )
    monkeypatch.setattr(
        preparation,
        "run_preparation_task",
        AsyncMock(return_value='{"execution_root":"invented"}'),
    )
    with pytest.raises(ValueError, match="Cannot determine"):
        await preparation.resolve_natural_location(owner, object(), "math 放到 ./runs")
    assert not list(tmp_path.iterdir())

    async def wait(*args, **kwargs):
        await asyncio.Future()

    monkeypatch.setattr(preparation, "run_preparation_task", wait)
    task = asyncio.create_task(
        preparation.resolve_natural_location(owner, object(), "math 放到 ./runs")
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert owner._preparation.task is None
    assert not list(tmp_path.iterdir())


def test_resume_rejects_missing_or_replaced_directory(tmp_path):
    from scienceflow.runtime.parallel.location.local import validate_record_location

    location = allocate_local_workspace(
        tmp_path / "chat", requested_root=str(tmp_path / "external")
    )
    record = dict(
        execution_location=location.to_record(),
        control_workspace=str(location.task_workspace),
    )
    validate_record_location(record)
    location.task_workspace.rmdir()
    with pytest.raises(ValueError, match="missing or has moved"):
        validate_record_location(record)
    other = tmp_path / "other"
    other.mkdir()
    location.task_workspace.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="missing or has moved"):
        validate_record_location(record)
    location.task_workspace.unlink()
    location.task_workspace.mkdir()
    record["task_roots"] = [str(other)]
    with pytest.raises(ValueError, match="remain inside"):
        validate_record_location(record)


def test_description_loaded_without_giving_tools_access_to_management(tmp_path):
    from scienceflow.research.onboarding import LongResearchSession
    from scienceflow.research.onboarding.support.preparation.agent import (
        preparation_prompt,
    )

    materials = tmp_path / "source materials"
    materials.mkdir()
    (materials / "description.md").write_text("Maximize the packing radius.")
    session = LongResearchSession.start(
        workspace=tmp_path,
        constraints=f'"{materials}" workers=2',
        natural_language=True,
    )
    assert "Maximize the packing radius." in session.draft.task_text
    assert session.draft.description_sources == (str(materials / "description.md"),)
    prompt = preparation_prompt(session.draft, "", "", tmp_path / "external")
    assert "original source files may be outside the tool workspace" in prompt


@pytest.mark.asyncio
async def test_preparation_runtime_has_isolated_files_and_available_shell(
    tmp_path, monkeypatch
):
    from inquirycraft.events import NullEventSink
    from inquirycraft.runtime import RuntimeOptions
    from inquirycraft.runtime.cancellation import CancellationToken
    from inquirycraft.tools.base import ToolContext
    from scienceflow.interfaces.ui.llm_cli import TuiPoolRuntimeFactory
    from scienceflow.interfaces.ui.research.preparation import (
        runtime as preparation_runtime,
    )

    management = tmp_path / "chat"
    execution = tmp_path / "execution"
    management.mkdir()
    execution.mkdir()
    factory = TuiPoolRuntimeFactory()
    created = []

    def create(options, **kwargs):
        tools = kwargs["tools"] if kwargs["tools"] is not None else factory._tools()
        runtime = SimpleNamespace(options=options, tools=tools, aclose=AsyncMock())
        created.append(runtime)
        return runtime

    monkeypatch.setattr(factory, "create_runtime", create)
    registry = SimpleNamespace(
        resolve=lambda *a, **kw: SimpleNamespace(
            models=["fake"], api_keys=["test"], base_urls=["https://example.invalid"]
        )
    )
    monkeypatch.setattr(preparation_runtime, "load_model_registry", lambda _: registry)
    owner = SimpleNamespace(
        preparation_runtime_factory=factory,
        workspace=management,
        task_workspace=execution,
        _model_config_path=tmp_path / "models.json",
        _selected_model_aliases=("fake",),
    )
    chat = SimpleNamespace(
        options=RuntimeOptions(
            model="fake", workspace=management, session_log=management / "session.jsonl"
        )
    )

    async def run(prompt, *, show_response=True, runtime=None):
        context = ToolContext(
            "session",
            "run",
            "agent",
            runtime.options.workspace,
            NullEventSink(),
            CancellationToken(),
        )
        await runtime.tools.get("write_text").execute(
            {"path": "evaluator.py", "content": "# prepared"}, context
        )
        with pytest.raises(ValueError, match="outside|workspace"):
            await runtime.tools.get("write_text").execute(
                {"path": "../chat/blocked", "content": "no"}, context
            )
        assert "shell" in [row["function"]["name"] for row in runtime.tools.schemas()]
        return "prepared"

    host = SimpleNamespace(runtime=chat, run_host_task=run)
    assert (
        await preparation_runtime.run_preparation_task(owner, host, "prepare")
        == "prepared"
    )
    assert (execution / "evaluator.py").read_text() == "# prepared"
    assert not list(management.iterdir())
    assert chat.options.workspace == management
    assert created[0].options.session_log is None
    created[0].aclose.assert_awaited_once()

    async def tool_free(prompt, *, show_response=True, runtime=None):
        assert not runtime.tools.schemas()
        raise ValueError("failed extraction")

    host.run_host_task = tool_free
    with pytest.raises(ValueError, match="failed extraction"):
        await preparation_runtime.run_preparation_task(
            owner, host, "location", tool_free=True
        )
    created[1].aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_tui_allocates_before_registered_preparation_and_rejects_relocation(
    tmp_path, monkeypatch
):
    from scienceflow.interfaces.ui.long_research import LongResearchInteraction
    from scienceflow.interfaces.ui import model_config
    from scienceflow.research.onboarding import write_onboarding_files
    from tests.test_multi_research import Host

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(model_config, "configure_draft_models", lambda *a: None)
    host = Host()
    host.run_host_task = AsyncMock()
    interaction = LongResearchInteraction(
        tmp_path / "chat", managed=True, multi_task=True
    )
    monkeypatch.setattr(interaction, "_preflight", AsyncMock())
    try:
        await interaction.try_handle(
            '/long-research circle-packing workers=2 cpu=0-1 duration=20m workspace="../external space"',
            host,
        )
        await interaction._preparation.task
        session = interaction.session
        assert session is not None
        task_workspace = interaction.task_workspace
        assert task_workspace.parent == tmp_path / "external space"
        assert Path(session.draft.workspace_base) == task_workspace / "runs"
        assert interaction.workspace == tmp_path / "chat"
        host.run_host_task.assert_not_called()
        files = write_onboarding_files(session.draft, task_workspace)
        assert files.manifest_path.parent == task_workspace / ".scienceflow"
        assert not (interaction.workspace / ".scienceflow/run_manifest.yaml").exists()
        await interaction._preparation.begin(host, "workspace=../changed")
        assert any("already allocated" in notice for notice in host.notices)
        assert interaction.task_workspace == task_workspace
        assert not (tmp_path / "changed").exists()
    finally:
        await interaction.aclose()


@pytest.mark.asyncio
async def test_natural_location_selection_runs_in_background_and_continues(
    tmp_path, monkeypatch
):
    from scienceflow.interfaces.ui.long_research import LongResearchInteraction
    from scienceflow.interfaces.ui import model_config
    from tests.test_multi_research import Host

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(model_config, "configure_draft_models", lambda *a: None)
    monkeypatch.setattr(
        preparation, "resolve_natural_location", AsyncMock(return_value="../external")
    )
    interaction = LongResearchInteraction(
        tmp_path / "chat", managed=True, multi_task=True
    )
    host = Host()
    host.run_host_task = AsyncMock()
    monkeypatch.setattr(interaction, "_preflight", AsyncMock())
    try:
        await interaction.try_handle(
            "/long-research circle-packing 放到 ../external workers=2 cpu=0-1 duration=20m",
            host,
        )
        location_task = interaction._preparation.task
        assert interaction.session is None
        await location_task
        assert interaction._preparation.task is not location_task
        await interaction._preparation.task
        assert interaction.task_workspace.parent == tmp_path / "external"
        assert interaction.session.draft.workers == 2
    finally:
        await interaction.aclose()


@pytest.mark.asyncio
async def test_cli_waits_for_location_then_preparation(tmp_path, monkeypatch):
    from scienceflow.interfaces.cli.commands.run.control import guided
    from scienceflow.interfaces.ui import model_config
    from scienceflow.research.onboarding import OnboardingState, write_onboarding_files

    monkeypatch.setattr(model_config, "configure_draft_models", lambda *a: None)
    monkeypatch.setattr(
        preparation, "resolve_natural_location", AsyncMock(return_value="../external")
    )
    launched = Mock(return_value={"run_id": "synthetic"})
    monkeypatch.setattr(guided, "start_run", launched)

    async def preflight(self, host):
        self.files = write_onboarding_files(self.session.draft, self.task_workspace)
        self.session.state = OnboardingState.CONFIRM

    monkeypatch.setattr(guided.LongResearchInteraction, "_preflight", preflight)
    row = await guided.prepare_and_start(
        tmp_path / "chat",
        "circle-packing 放到 ../external workers=2 cpu=0-1 duration=20m",
        None,
    )
    assert row["run_id"] == "synthetic"
    location = launched.call_args.kwargs["draft"].execution_location
    assert Path(location["task_workspace"]).parent == tmp_path / "external"
    assert location["management_workspace"] == str(tmp_path / "chat")


def test_web_uses_indexed_external_task_root_and_does_not_import_other_workspaces(
    tmp_path, monkeypatch
):
    from scienceflow.interfaces.ui.web_monitor import data

    index = TaskIndex(tmp_path / "chat")
    entry = index.reserve("math", str(tmp_path / "external"))
    task_root = Path(entry["workspace"]) / "runs/run-one/math"
    task_root.mkdir(parents=True)
    row = dict(
        run_id="run-one",
        status="stopped",
        alive=False,
        task_roots=[str(task_root)],
        execution_location=entry["execution_location"],
        control_workspace=entry["workspace"],
    )
    index.sync([row])
    monkeypatch.setattr(data, "list_runs", lambda: [row])
    monitor = data.WorkspaceMonitor(index.workspace)
    monkeypatch.setattr(
        monitor.projection,
        "item",
        lambda *a: {"monitor": {"name": "math", "status": "stopped"}},
    )
    monkeypatch.setattr(monitor.resources, "sample", lambda: "")
    assert len(monitor.collect()["tasks"]) == 1
    assert monitor.task_root("1") == task_root
    assert monitor.task_root("../../outside") is None
    other = data.WorkspaceMonitor(tmp_path / "unrelated")
    monkeypatch.setattr(other.resources, "sample", lambda: "")
    assert other.collect()["tasks"] == []


def test_default_math_request_does_not_add_a_location_model_call():
    assert not preparation.needs_location_resolution("circle packing 圆放在单位正方形")
    assert not preparation.needs_location_resolution(
        "circle-packing workspace=/data/experiments"
    )
    assert preparation.needs_location_resolution("circle-packing 放在/data/experiments")


def test_preparation_keeps_bounded_discussion_without_old_host_prompts():
    from scienceflow.interfaces.ui.research.preparation.runtime import (
        conversation_evidence,
    )

    conversation = (
        ("system", "private system policy"),
        ("user", "Pack 26 circles in a unit square."),
        ("assistant", "Maximize the sum of radii."),
        ("user", "Prepare a ScienceFlow research task, old protocol"),
        ("user", "/status"),
    )
    owner = SimpleNamespace(session=SimpleNamespace(conversation=conversation))
    evidence = conversation_evidence(owner)
    assert "26 circles" in evidence and "sum of radii" in evidence
    assert "private system" not in evidence and "old protocol" not in evidence
    assert "/status" not in evidence
    owner.session.conversation = tuple(("user", "a" * 100_000) for _ in range(100))
    assert len(conversation_evidence(owner)) < 25_000


def test_default_tasks_symlink_cannot_relocate_execution_implicitly(tmp_path):
    management = tmp_path / "chat"
    management.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (management / "tasks").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="remain inside"):
        allocate_local_workspace(management, task_number=1)
    assert not list(external.iterdir())


def test_index_notification_failure_keeps_committed_directory(tmp_path, monkeypatch):
    from scienceflow.interfaces.ui.research.control.tasks import records

    write = records.write_record

    def committed_then_failed(directory, payload):
        write(directory, payload)
        raise OSError("notification failed after commit")

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(records, "write_record", committed_then_failed)
    index = TaskIndex(tmp_path / "chat")
    with pytest.raises(OSError, match="notification failed"):
        index.reserve("math", str(tmp_path / "external"))
    (entry,) = index.snapshot()
    assert Path(entry["workspace"]).is_dir()
    assert entry["run_id"] is None


def test_execution_root_symlink_loop_is_a_user_error(tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    with pytest.raises((ValueError, OSError)):
        resolve_local_root(tmp_path, "loop")
