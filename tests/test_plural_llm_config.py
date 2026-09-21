from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from click.testing import CliRunner
from inquirycraft.events import NullEventSink
from inquirycraft.runtime import RuntimeOptions

from scienceflow.foundation.config.runtime.llm_lists import stage_env_values
from scienceflow.foundation.config.schema.models.loading.serialization import (
    dump_resolved_config_yaml,
)
from scienceflow.foundation.config.llm.model_registry import apply_model_registry
from scienceflow.foundation.config.schema.settings import Config, _apply_env, load_cfg
from scienceflow.interfaces.ui.llm_cli import PoolRuntimeFactory, plural_cli_options


def test_only_stage_scoped_variables_are_used():
    env = {"API_KEYS": "a,b", "CODE_API_KEY": "old"}
    assert stage_env_values(env, "code", "api_keys") == ["old"]
    env["CODE_API_KEYS"] = "c,d"
    assert stage_env_values(env, "code", "api_keys") == ["c", "d"]


def test_legacy_yaml_override_and_plural_resolved_output(tmp_path, monkeypatch):
    for name in (
        "CODE_MODEL",
        "CODE_MODELS",
        "FEEDBACK_MODEL",
        "FEEDBACK_MODELS",
        "MODEL",
        "MODELS",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "base.yaml").write_text("agent:\n  code:\n    models: [old]\n")
    path = tmp_path / "config.yaml"
    path.write_text(
        "include: base.yaml\nagent:\n  code:\n    model: new\n    api_key: test\n    base_url: https://example.test/v1\n"
    )
    config = load_cfg(path, cli_args=False)
    assert config.agent.code.models == ["new"]
    assert config.agent.code.api_keys == ["test"]
    out = tmp_path / "resolved.yaml"
    dump_resolved_config_yaml(config, out)
    stage = yaml.safe_load(out.read_text())["agent"]["code"]
    assert stage["models"] == ["new"]
    assert not {"model", "api_key", "base_url"} & stage.keys()


def test_resolved_config_redacts_headers_and_url_credentials(tmp_path):
    config = Config()
    config.agent.code.models = ["model-a"]
    config.agent.code.api_keys = ["sk-secret"]
    config.agent.code.base_urls = [
        "https://credential-user:credential-password@example.test/v1"
        "?credential-token=secret#credential-fragment"
    ]
    config.agent.code.headers = {
        "Authorization": "Bearer secret",
        "X-Gateway-Key": "gateway-secret",
    }
    out = tmp_path / "resolved.yaml"

    dump_resolved_config_yaml(config, out)

    raw = out.read_text(encoding="utf-8")
    stage = yaml.safe_load(raw)["agent"]["code"]
    assert stage["models"] == ["model-a"]
    assert stage["base_urls"] == ["https://example.test/v1"]
    assert "headers" not in stage
    for secret in (
        "sk-secret",
        "credential-user",
        "credential-password",
        "credential-token=secret",
        "credential-fragment",
        "Bearer secret",
        "gateway-secret",
    ):
        assert secret not in raw


def _write_registry(path):
    path.write_text(json.dumps({
        "models": {
            "a": {"model": "m1", "endpoints": [{"url": "https://a/v1", "key": "key-a"}]},
            "b": {"model": "m2", "endpoints": [{"url": "https://b/v1", "key": "key-b"}]},
        },
        "defaults": {"code_models": ["a", "b"], "feedback_models": ["a"]},
    }))


def test_model_registry_projects_models_for_existing_consumers(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)
    cfg = Config()
    apply_model_registry(cfg, path)
    assert cfg.agent.code.model == "m1"
    assert cfg.agent.code.models == ["m1", "m2"]
    assert cfg.agent.code.api_keys == ["key-a", "key-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_factory_fails_over_between_endpoints_of_one_model(
    tmp_path, monkeypatch, stream
):
    from scienceflow.foundation.config.llm import llm_pool

    seen = []

    class Provider:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.api_key = kwargs["api_key"]
            self.base_url = kwargs["base_url"]
            self._last_finish_reason = "stop"

        async def ask_tool(self, **kwargs):
            seen.append((self.model, self.api_key))
            if self.api_key == "a":
                raise httpx.ReadTimeout("unavailable")
            return SimpleNamespace(
                content="recovered", reasoning_content="", tool_calls=[]
            )

        async def ask_tool_stream(self, handle, **kwargs):
            response = await self.ask_tool(**kwargs)
            await handle.queue.put(response.content)
            await handle.queue.put(None)
            return response

    monkeypatch.setattr(llm_pool, "OnlineLLM", Provider)
    monkeypatch.setattr(llm_pool, "_SOFT_RETRY_DELAY_SEC", 0)
    runtime = PoolRuntimeFactory().create_runtime(
        RuntimeOptions(model="model-a,model-a", workspace=tmp_path, stream_llm=stream),
        api_key="a,b",
        base_url="https://a/v1,https://b/v1",
        event_sink=NullEventSink(),
    )
    try:
        assert await runtime.run("hello") == "recovered"
        assert seen[0] == ("model-a", "a")
        assert seen[-1] == ("model-a", "b")
    finally:
        await runtime.aclose()


def test_plural_cli_flags_deliver_full_lists(tmp_path):
    from inquirycraft.cli import create_cli

    captured = {}

    class Factory:
        def create_runtime(self, options, **kwargs):
            captured.update(models=options.model, **kwargs)

            class Runtime:
                async def run(self, task):
                    return "done"

                async def aclose(self):
                    pass

            return Runtime()

    config = tmp_path / "models.json"
    _write_registry(config)
    cli = plural_cli_options(
        create_cli(
            runtime_factory=Factory(), include_tui=False, model_config_default=config
        )
    )
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "hello",
            "--models",
            "a,b",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["models"] == "m1,m2"
    assert captured["api_key"] == "key-a,key-b"
    assert captured["base_url"] == "https://a/v1,https://b/v1"


def test_manifest_worker_retains_pool_and_overrides_inherited_settings():
    from scienceflow.runtime.parallel.config.models import TaskSpec
    from scienceflow.runtime.parallel.state.subprocess_runtime import (
        _apply_endpoint_env,
    )

    env = {
        "CODE_API_KEYS": "stale",
        "CODE_BASE_URLS": "https://stale/v1",
        "FEEDBACK_API_KEYS": "judge-key",
        "FEEDBACK_BASE_URLS": "https://judge/v1",
        "FEEDBACK_MODELS": "judge-model",
    }
    spec = TaskSpec(
        exp_id="x",
        task="x",
        workspace="/tmp/x",
        api_keys="a,b",
        base_urls="https://a/v1,https://b/v1",
        models="m1,m2",
    )
    _apply_endpoint_env(env, spec)
    assert env["CODE_API_KEYS"] == "a,b"
    assert env["FEEDBACK_API_KEYS"] == "judge-key"
    assert env["FEEDBACK_BASE_URLS"] == "https://judge/v1"
    assert env["FEEDBACK_MODELS"] == "judge-model"
    assert env["CODE_BASE_URLS"] == "https://a/v1,https://b/v1"
    assert env["CODE_MODELS"] == "m1,m2"


def test_manifest_rotates_models_with_endpoints(tmp_path):
    from scienceflow.runtime.parallel.execution.runner import ParallelRunner

    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "api_keys": ["a", "b"],
                "base_urls": ["https://a/v1", "https://b/v1"],
                "models": ["m1", "m2"],
                "tasks": [
                    {
                        "exp_id": f"task-{i}",
                        "task": "hello",
                        "workspace": str(tmp_path / str(i)),
                    }
                    for i in range(2)
                ],
            }
        )
    )
    first, second = ParallelRunner(path)._tasks
    assert first.models == "m1,m2"
    assert second.models == "m2,m1"
    assert second.api_keys == "b,a"
    assert second.base_urls == "https://b/v1,https://a/v1"


def test_single_yaml_endpoint_still_creates_list_config(tmp_path):
    path = tmp_path / "single.yaml"
    path.write_text(
        "agent:\n  code:\n    api_keys: [key]\n    base_urls: [https://example.test/v1]\n    models: [model]\n"
    )
    stage = load_cfg(path, cli_args=False).agent.code
    assert stage.api_keys == ["key"]
    assert stage.base_urls == ["https://example.test/v1"]


def test_real_agent_entry_accepts_model_registry(tmp_path, monkeypatch):
    from scienceflow.interfaces.cli import main

    captured = {}

    class Runtime:
        async def run(self, task):
            return "done"

        async def aclose(self):
            pass

    def create(self, options, **kwargs):
        captured.update(models=options.model, **kwargs)
        return Runtime()

    monkeypatch.setattr(PoolRuntimeFactory, "create_runtime", create)
    config = tmp_path / "models.json"
    _write_registry(config)
    result = CliRunner().invoke(
        main,
        [
            "agent",
            "run",
            "hello",
            "--model-config",
            str(config),
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["models"] == "m1,m2"
    assert captured["api_key"] == "key-a,key-b"
    assert captured["base_url"] == "https://a/v1,https://b/v1"


def test_stage_loading_uses_separate_registry_defaults(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)
    cfg = Config()
    apply_model_registry(cfg, path)
    assert cfg.agent.code.models == ["m1", "m2"]
    assert cfg.agent.code.api_keys == ["key-a", "key-b"]
    assert cfg.agent.feedback.models == ["m1"]
    assert cfg.agent.feedback.api_keys == ["key-a"]
    assert cfg.agent.feedback.base_urls == ["https://a/v1"]


def test_feedback_pool_does_not_determine_code_routing():
    from scienceflow.runtime.parallel.config.manifest import _env_sticky_primary_index

    assert (
        _env_sticky_primary_index({"FEEDBACK_API_KEYS": "judge-a,judge-b"}, 1) is None
    )
