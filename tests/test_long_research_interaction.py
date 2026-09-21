from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scienceflow.interfaces.ui import long_research
from scienceflow.interfaces.ui.long_research import (
    LongResearchInteraction,
    _preflight_summary,
    long_research_argument,
)
from scienceflow.research.onboarding import (
    PreflightCheck,
    PreflightReport,
    PreflightStatus,
)
from scienceflow.runtime.core.support.system_resources import GPUDevice
from scienceflow.runtime.parallel.config.models import TaskResult
from scienceflow.runtime.parallel.service import ParallelRunSummary


class FakeHost:
    def __init__(self, *, busy: bool = False, conversation=()) -> None:
        self._busy = busy
        self._conversation = conversation
        self.notices: list[str] = []
        self.statuses: list[str] = []
        self.snapshot_calls = 0

    @property
    def busy(self) -> bool:
        return self._busy

    async def notice(self, text: str) -> None:
        self.notices.append(text)

    async def cancel_agent(self) -> None:
        self._busy = False

    def conversation_snapshot(self):
        self.snapshot_calls += 1
        return self._conversation

    def set_host_status(self, text: str) -> None:
        self.statuses.append(text)


def _complete_command(data: Path) -> str:
    return (
        "/long-research task='optimize a new dataset' "
        f"data={data} metric=accuracy direction=max "
        "artifact=artifacts/result.json command='python evaluate.py' "
        "gpu=cpu workers=2 cpu=0-7 duration=10m"
    )


def test_long_research_command_matching_is_exact() -> None:
    assert long_research_argument("/long-research") == ""
    assert long_research_argument(" /long-research gpu=cpu ") == "gpu=cpu"
    assert long_research_argument("/long_research") == ""
    assert long_research_argument(" /long_research gpu=cpu ") == "gpu=cpu"
    assert long_research_argument("/long-researcher") is None
    assert long_research_argument("ordinary text") is None


def test_preflight_summary_hides_passes_and_keeps_failure_details() -> None:
    report = PreflightReport(
        status=PreflightStatus.FAILED,
        manifest_path="manifest.yaml",
        checks=(
            PreflightCheck("manifest_shape", True, "valid"),
            PreflightCheck("dataset_scan", False, "dataset directory is missing"),
        ),
    )

    summary = _preflight_summary(report)

    assert "manifest_shape" not in summary
    assert "dataset_scan: dataset directory is missing" in summary


@pytest.mark.asyncio
async def test_plain_text_passes_through_and_busy_command_does_not_snapshot(
    tmp_path: Path,
) -> None:
    interaction = LongResearchInteraction(tmp_path)
    host = FakeHost(busy=True)

    assert await interaction.try_handle("ordinary text", host) is False
    assert await interaction.try_handle("/long-research", host) is True

    assert host.snapshot_calls == 0
    assert "finish or cancel" in host.notices[-1]
    assert interaction.session is None


@pytest.mark.asyncio
async def test_invalid_inline_options_are_reported_without_activating(
    tmp_path: Path,
) -> None:
    interaction = LongResearchInteraction(tmp_path)
    host = FakeHost()

    assert await interaction.try_handle("/long-research invented=value", host) is True

    assert interaction.session is None
    assert "unknown long-research option" in host.notices[-1]


@pytest.mark.asyncio
async def test_global_visual_commands_remain_owned_by_inquirycraft(
    tmp_path: Path,
) -> None:
    interaction = LongResearchInteraction(tmp_path)
    host = FakeHost()
    await interaction.try_handle("/long-research", host)

    assert await interaction.try_handle("/help", host) is False
    assert await interaction.try_handle("/clear", host) is False
    assert await interaction.try_handle("/quit", host) is False
    assert interaction.session is not None


@pytest.mark.asyncio
async def test_onboarding_uses_detached_conversation_and_cancel_returns_to_chat(
    tmp_path: Path,
) -> None:
    host = FakeHost(conversation=({"role": "user", "content": "model a dataset"},))
    interaction = LongResearchInteraction(tmp_path)

    assert await interaction.try_handle("/long-research", host) is True
    assert interaction.session is not None
    assert interaction.session.draft.task_text == "model a dataset"
    assert "dataset directory" in host.notices[-1]

    assert await interaction.try_handle("/cancel", host) is True
    assert interaction.session is None
    assert host.statuses[-1] == ""
    assert await interaction.try_handle("back to chat", host) is False


@pytest.mark.asyncio
async def test_underscore_command_alias_starts_onboarding(tmp_path: Path) -> None:
    host = FakeHost(conversation=({"role": "user", "content": "optimize"},))
    interaction = LongResearchInteraction(tmp_path)

    assert await interaction.try_handle("/long_research", host) is True
    assert interaction.session is not None
    assert host.statuses[-1].startswith("long research")


@pytest.mark.asyncio
async def test_gpu_question_displays_existing_lightweight_probe(
    monkeypatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    host = FakeHost(conversation=({"role": "user", "content": "model a dataset"},))
    interaction = LongResearchInteraction(tmp_path)
    monkeypatch.setattr(
        long_research,
        "query_gpu_devices",
        lambda: (GPUDevice(0, "GPU-test", 81920, 73728),),
    )

    await interaction.try_handle("/long-research", host)
    for answer in (
        str(data),
        "accuracy",
        "max",
        "artifacts/result.json",
        "python evaluate.py",
    ):
        await interaction.try_handle(answer, host)

    assert "GPU 0 (73728 / 81920 MiB free)" in host.notices[-1]


@pytest.mark.asyncio
async def test_complete_flow_preflights_and_runs_directly_with_public_runner(
    monkeypatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    host = FakeHost()
    interaction = LongResearchInteraction(tmp_path / "workspace")
    calls: list[Path] = []

    async def fake_run_manifest(manifest_path):
        calls.append(Path(manifest_path))
        return ParallelRunSummary(
            results=(TaskResult(exp_id="task", run_id="run", status="success"),),
            text="summary",
            task_count=1,
            max_concurrent=1,
        )

    monkeypatch.setattr(long_research, "run_manifest", fake_run_manifest)

    assert await interaction.try_handle(_complete_command(data), host) is True
    assert interaction.preflight is None
    assert interaction._awaiting_research_models
    assert "Default · Code" in host.notices[-1]
    assert await interaction.try_handle("/status", host) is True
    assert host.notices[-1].endswith("Choose research models or enter defaults.")

    assert await interaction.try_handle("defaults", host) is True
    assert interaction.preflight is not None
    assert interaction.preflight.status.value == "exploratory"
    assert interaction.files is not None
    assert interaction.files.manifest_path.is_file()
    assert "Manifest:" not in host.notices[-1]
    assert "Equivalent CLI:" not in host.notices[-1]
    assert "all checks passed" in host.notices[-1]
    assert "manifest_shape=ok" not in host.notices[-1]

    assert await interaction.try_handle("run", host) is True
    while interaction.running:
        await asyncio.sleep(0)

    assert calls == [tmp_path / "workspace" / ".scienceflow" / "run_manifest.yaml"]
    assert interaction.session is None
    assert any("Long research finished" in notice for notice in host.notices)
    assert host.statuses[-1] == ""


@pytest.mark.asyncio
async def test_cancel_and_close_stop_owned_parallel_run(
    monkeypatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    host = FakeHost()
    interaction = LongResearchInteraction(tmp_path / "workspace")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_run_manifest(_manifest_path):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(long_research, "run_manifest", blocked_run_manifest)
    await interaction.try_handle(_complete_command(data), host)
    await interaction.try_handle("defaults", host)
    await interaction.try_handle("confirm", host)
    await interaction.try_handle("run", host)
    await started.wait()

    assert await interaction.try_handle("/cancel", host) is True

    assert cancelled.is_set()
    assert interaction.session is None
    assert host.statuses[-1] == ""
    await interaction.aclose()


@pytest.mark.asyncio
async def test_resource_monitor_is_closed_with_interaction(monkeypatch, tmp_path):
    from scienceflow.interfaces.ui.research.resources import ResourceMonitor

    monkeypatch.setattr(ResourceMonitor, 'sample', lambda self: 'Host · CPU 10%')
    host = FakeHost()
    host.set_resource_status = lambda text: host.statuses.append(text)
    interaction = LongResearchInteraction(tmp_path)
    await interaction.on_mount(host)
    task = interaction._resource_task
    for _ in range(100):
        if host.statuses:
            break
        await asyncio.sleep(0.01)
    assert host.statuses == ['Host · CPU 10%']
    await interaction.aclose()
    assert task.done()
    assert interaction._resource_task is None


@pytest.mark.asyncio
async def test_usage_resources_and_help_commands_without_research(tmp_path):
    interaction = LongResearchInteraction(tmp_path)
    host = FakeHost()
    assert await interaction.try_handle('/research-usage', host)
    assert 'No active research' in host.notices[-1]
    assert await interaction.try_handle('/resources', host)
    assert 'Resources unavailable' in host.notices[-1]
    assert not await interaction.try_handle('/help', host)
    assert '/research-usage' in host.notices[-1]
    assert not await interaction.try_handle('/context', host)
    assert {
        "/long-research",
        "/models",
        "/resources",
        "/research-usage",
        "/web",
    } == {spec.name for spec in interaction.tui_commands}
    await interaction.aclose()


@pytest.mark.asyncio
async def test_models_command_lists_aliases_without_exposing_keys(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "fast": {
                        "model": "provider/model-fast",
                        "endpoints": [
                            {"url": "https://one.example/v1", "key": "secret-one"},
                            {"url": "https://two.example/v1", "key": "secret-two"},
                        ],
                    }
                },
                "defaults": {
                    "code_models": ["fast"],
                    "feedback_models": ["fast"],
                    "selection": "auto",
                },
            }
        ),
        encoding="utf-8",
    )
    interaction = LongResearchInteraction(tmp_path)
    interaction.configure_model_registry(config, ())
    host = FakeHost()

    assert await interaction.try_handle("/models", host)
    assert "fast (2 route(s))" in host.notices[-1]
    assert "Feedback" not in host.notices[-1]
    assert "secret-one" not in host.notices[-1]
    assert "secret-two" not in host.notices[-1]
    assert interaction.tui_completions(
        "/long-research circle code-models=f"
    ) == [("/long-research circle code-models=fast", "Model alias")]


@pytest.mark.asyncio
async def test_models_command_switches_only_the_chat_model(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    alias: {
                        "model": f"provider/{alias}",
                        "endpoints": [
                            {
                                "url": f"https://{alias}.example/v1",
                                "key": f"{alias}-key",
                            }
                        ],
                    }
                    for alias in ("fast", "review")
                },
                "defaults": {
                    "code_models": ["fast", "review"],
                    "feedback_models": ["review"],
                    "selection": "auto",
                },
            }
        ),
        encoding="utf-8",
    )

    class InteractiveHost(FakeHost):
        def open_chat_model_selector(self, path, **kwargs):
            self.popup = (path, kwargs)

        async def switch_chat_models(self, path, **kwargs):
            self.switch = (path, kwargs)
            return kwargs["aliases"]

    interaction = LongResearchInteraction(tmp_path)
    interaction.configure_model_registry(config, ())
    host = InteractiveHost()

    assert await interaction.try_handle("/models", host)
    popup_path, popup = host.popup
    assert popup_path == config
    assert popup["aliases"] == ("fast", "review")
    assert popup["current"] == ("fast", "review")
    assert popup["config_hint"] == (
        f"Config · {config} · edit, then reopen /models"
    )

    assert await interaction.try_handle("/models review policy=fixed", host)
    path, options = host.switch
    assert path == config
    assert options["aliases"] == ("review",)
    assert options["selection"] == "fixed"
    assert interaction._selected_model_aliases == ("review",)
    assert interaction._selected_feedback_aliases == ("review",)
    assert interaction._model_selection == "fixed"


@pytest.mark.asyncio
async def test_valid_model_config_stays_quiet_on_mount(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "fast": {
                        "model": "provider/fast",
                        "endpoints": [
                            {"url": "https://code.example/v1", "key": "code-secret"}
                        ],
                    },
                    "review": {
                        "model": "provider/review",
                        "endpoints": [
                            {"url": "https://review.example/v1", "key": "review-secret"}
                        ],
                    },
                },
                "defaults": {
                    "code_models": ["fast"],
                    "feedback_models": ["review"],
                    "selection": "auto",
                },
            }
        ),
        encoding="utf-8",
    )
    interaction = LongResearchInteraction(tmp_path)
    interaction.configure_model_registry(config, ())
    host = FakeHost()

    await interaction.on_mount(host)

    assert host.notices == []
    await interaction.aclose()


@pytest.mark.asyncio
async def test_missing_model_config_shows_setup_commands(tmp_path):
    interaction = LongResearchInteraction(tmp_path)
    interaction._model_config_path = tmp_path / "missing-models.json"
    host = FakeHost()

    await interaction.on_mount(host)

    assert "scienceflow config init" in host.notices[0]
    assert "scienceflow config path" in host.notices[0]

    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    await interaction.try_handle(_complete_command(data), host)
    assert interaction._awaiting_research_models
    assert "Model configuration unavailable" in host.notices[-1]

    await interaction.try_handle("defaults", host)
    assert interaction._awaiting_research_models
    assert "scienceflow config init" in host.notices[-1]
    await interaction.aclose()


@pytest.mark.asyncio
async def test_managed_long_research_loads_defaults_before_task_is_complete(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    alias: {
                        "model": f"provider/{alias}",
                        "endpoints": [
                            {"url": f"https://{alias}.example/v1", "key": "secret"}
                        ],
                    }
                    for alias in ("fast", "backup", "review")
                },
                "defaults": {
                    "code_models": ["fast", "backup"],
                    "feedback_models": ["review"],
                    "selection": "fixed",
                },
            }
        ),
        encoding="utf-8",
    )
    class Managed:
        async def command(self, text, host):
            return False

    interaction = LongResearchInteraction(tmp_path)
    interaction._managed = Managed()
    interaction.configure_model_registry(config, ())
    host = FakeHost()

    await interaction._start("new optimization", host)

    assert interaction.session.draft.code_models == ["fast"]
    assert interaction.session.draft.feedback_models == ["review"]
    assert interaction.session.draft.model_selection == "auto"
    assert not any("Default · Code" in notice for notice in host.notices)


@pytest.mark.asyncio
async def test_model_question_applies_override_and_inline_models_skip_it(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    alias: {
                        "model": f"provider/{alias}",
                        "endpoints": [
                            {"url": f"https://{alias}.example/v1", "key": "secret"}
                        ],
                    }
                    for alias in ("fast", "review")
                },
                "defaults": {
                    "code_models": ["fast"],
                    "feedback_models": ["review"],
                    "selection": "auto",
                },
            }
        ),
        encoding="utf-8",
    )
    interaction = LongResearchInteraction(tmp_path / "workspace-question")
    interaction.configure_model_registry(config, ())
    host = FakeHost()

    await interaction.try_handle(_complete_command(data), host)

    assert interaction._awaiting_research_models
    assert "Default · Code fast" in host.notices[-1]
    await interaction.try_handle(
        "models=review feedback-models=fast model-policy=fixed",
        host,
    )
    assert interaction.preflight is not None
    assert interaction.session.draft.code_models == ["review"]
    assert interaction.session.draft.feedback_models == ["fast"]
    assert interaction.session.draft.model_selection == "fixed"
    await interaction.aclose()

    interaction = LongResearchInteraction(tmp_path / "workspace-inline")
    interaction.configure_model_registry(config, ())
    host = FakeHost()
    command = (
        f"{_complete_command(data)} models=fast "
        "feedback-models=review model-policy=fixed"
    )
    await interaction.try_handle(command, host)

    assert interaction.preflight is not None
    assert not interaction._awaiting_research_models
    assert interaction.session.draft.code_models == ["fast"]
    assert interaction.session.draft.feedback_models == ["review"]
    assert interaction.session.draft.model_selection == "fixed"
    assert not any("Default · Code" in notice for notice in host.notices)
    await interaction.aclose()


@pytest.mark.asyncio
async def test_managed_long_research_keeps_explicit_multiple_code_models(tmp_path):
    config = tmp_path / "models.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    alias: {
                        "model": f"provider/{alias}",
                        "endpoints": [
                            {"url": f"https://{alias}.example/v1", "key": "secret"}
                        ],
                    }
                    for alias in ("fast", "backup", "review")
                },
                "defaults": {
                    "code_models": ["fast", "backup"],
                    "feedback_models": ["review"],
                    "selection": "auto",
                },
            }
        ),
        encoding="utf-8",
    )
    interaction = LongResearchInteraction(tmp_path)
    interaction.configure_model_registry(config, ())

    await interaction._start(
        "new optimization models=fast,backup model-policy=auto",
        FakeHost(),
    )

    assert interaction.session.draft.code_models == ["fast", "backup"]
    assert interaction.session.draft.model_selection == "auto"


@pytest.mark.asyncio
async def test_configuration_input_echo_precedes_response_without_duplication(tmp_path):
    class EchoHost(FakeHost):
        def __init__(self):
            super().__init__()
            self.entries = []

        async def echo_user(self, text):
            self.entries.append(('user', text))

        async def notice(self, text):
            self.entries.append(('notice', text))
            await super().notice(text)

    host = EchoHost()
    interaction = LongResearchInteraction(tmp_path)
    command = '/long-research optimize a new task'
    await interaction.try_handle(command, host)
    assert host.entries[0] == ('user', command)
    offset = len(host.entries)
    await interaction.try_handle(str(tmp_path), host)
    assert host.entries[offset] == ('user', str(tmp_path))
    assert sum(role == 'user' for role, _ in host.entries) == 2
    offset = len(host.entries)
    assert not await interaction.try_handle('/help', host)
    assert not any(role == 'user' for role, _ in host.entries[offset:])
    await interaction.try_handle('/cancel', host)
    assert host.entries[-2] == ('user', '/cancel')
    await interaction.aclose()


@pytest.mark.asyncio
async def test_running_research_allows_chat_and_chat_cancellation(tmp_path):
    from scienceflow.research.onboarding import LongResearchSession, OnboardingState

    host = FakeHost()
    interaction = LongResearchInteraction(tmp_path)
    interaction.session = LongResearchSession.start(workspace=tmp_path, constraints='circle-packing')
    interaction.session.state = OnboardingState.RUNNING
    interaction._run_task = asyncio.create_task(asyncio.Event().wait())
    assert not await interaction.try_handle('你好，解释一下这个算法', host)
    assert not await interaction.try_handle('/steer 请简短回答', host)
    assert host.notices == []
    assert await interaction.try_handle('/status', host)
    host._busy = True
    assert not await interaction.try_handle('/cancel', host)
    assert not interaction._run_task.done()
    await interaction.aclose()
