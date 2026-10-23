"""Chat preparation and TUI share bounded retry and host-owned budgets."""

from types import SimpleNamespace

import pytest
from inquirycraft.cli.chat_profiles import ChatProfile, ProfileStore, profile_key
from inquirycraft.events import NullEventSink
from inquirycraft.runtime import RuntimeOptions

from scienceflow.interfaces.ui import llm_cli


@pytest.mark.parametrize(
    "factory", [llm_cli.PoolRuntimeFactory, llm_cli.TuiPoolRuntimeFactory]
)
def test_chat_factory_uses_host_profiles_and_bounded_retry(
    tmp_path, monkeypatch, factory
):
    config = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    legacy = ProfileStore()
    legacy.save(
        profile_key("https://offline.invalid", "offline"),
        ChatProfile(window=1000000, output_reserve=32768),
    )
    original = legacy.path.read_bytes()
    backend = SimpleNamespace(max_tokens=4096)
    monkeypatch.setattr(llm_cli, "build_stage_llm", lambda stage: backend)
    runtime = factory().create_runtime(
        RuntimeOptions(model="offline", workspace=tmp_path, max_tokens=4096),
        api_key="fake",
        base_url="https://offline.invalid",
        event_sink=NullEventSink(),
    )
    monitor = runtime.chat_monitor
    assert monitor.store.path == config / "scienceflow" / "chat_profiles.json"
    assert monitor.profiles[0].window == 1000000
    assert runtime.options.max_tokens == backend.max_tokens == 4096
    assert runtime._llm_pipeline.retry_policy.max_attempts == 3
    assert runtime._llm_pipeline.retry_adapter is not None
    monitor.set_profile(0, ChatProfile(output_reserve=8192))
    assert runtime.options.max_tokens == backend.max_tokens == 4096
    assert legacy.path.read_bytes() == original
    assert monitor.store.path.exists()


def test_default_chat_output_is_32768(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(
        llm_cli,
        "build_stage_llm",
        lambda stage: SimpleNamespace(max_tokens=stage.max_tokens),
    )
    runtime = llm_cli.TuiPoolRuntimeFactory().create_runtime(
        RuntimeOptions(model="offline", workspace=tmp_path),
        api_key="fake",
        base_url="https://offline.invalid",
        event_sink=NullEventSink(),
    )
    assert runtime.options.max_tokens == 32768
    assert runtime.llm.backend.max_tokens == 32768
    assert runtime.chat_monitor.output_limit == 32768
