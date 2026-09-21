from __future__ import annotations

from typing import Any, ClassVar

import pytest

from scienceflow.foundation.config.llm import llm_factory as agent_runtime
from scienceflow.foundation.config.schema.settings import StageConfig


def test_legacy_single_config_maps_to_one_endpoint_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    class CapturingPool:
        @classmethod
        def from_endpoints(cls, **kwargs):
            captured.append(kwargs)
            return kwargs

    monkeypatch.setattr(agent_runtime, "PooledLLM", CapturingPool)
    monkeypatch.setattr(
        agent_runtime,
        "llm_extra_client_kwargs",
        lambda _stage: {"headers": {"x-contract": "1"}, "http_asyncclient": "client"},
    )
    stage = StageConfig(
        model="model-a",
        api_key="key-a",
        base_url="https://a.example/v1",
        max_tokens=1234,
        frequency_penalty=0.25,
        coalesce_system_messages=True,
    )
    stage.request_seed = 4444

    result = agent_runtime.build_stage_llm(stage)

    assert result is captured[0]
    assert captured[0] == {
        "endpoints": [("https://a.example/v1", "key-a")],
        "endpoint_models": ["model-a"],
        "routing_mode": "round_robin",
        "sticky_id": "",
        "sticky_primary_index": None,
        "rate_limit_cooldown_sec": 60.0,
        "connection_cooldown_sec": 15.0,
        "model": "model-a",
        "max_tokens": 1234,
        "frequency_penalty": 0.25,
        "coalesce_system_messages": True,
        "stream": True,
        "tracker": True,
        "headers": {"x-contract": "1"},
        "http_asyncclient": "client",
        "request_seed": 4444,
    }


def test_stage_config_maps_exact_pool_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingPool:
        captured: ClassVar[dict[str, Any]] = {}

        @classmethod
        def from_endpoints(cls, **kwargs: Any) -> dict[str, Any]:
            cls.captured = kwargs
            return kwargs

    monkeypatch.setattr(agent_runtime, "PooledLLM", CapturingPool)
    monkeypatch.setattr(
        agent_runtime,
        "llm_extra_client_kwargs",
        lambda _stage: {"http_asyncclient": "pooled-client"},
    )
    stage = StageConfig(
        model="fallback",
        models=["model-a", "model-b", "model-c"],
        api_keys=["key-a", "key-b", "key-c"],
        base_urls=["u1", "u2", "u3"],
        max_tokens=2048,
        api_routing_mode="sticky_failover",
        api_sticky_id="run-a",
        api_sticky_primary_index=1,
        api_rate_limit_cooldown_sec=12.5,
        api_connection_cooldown_sec=3.5,
    )

    result = agent_runtime.build_stage_llm(stage)

    assert result is CapturingPool.captured
    assert CapturingPool.captured == {
        "endpoints": [("u1", "key-a"), ("u2", "key-b"), ("u3", "key-c")],
        "endpoint_models": ["model-a", "model-b", "model-c"],
        "routing_mode": "sticky_failover",
        "sticky_id": "run-a",
        "sticky_primary_index": 1,
        "rate_limit_cooldown_sec": 12.5,
        "connection_cooldown_sec": 3.5,
        "model": "model-a",
        "max_tokens": 2048,
        "frequency_penalty": None,
        "coalesce_system_messages": False,
        "stream": True,
        "tracker": True,
        "http_asyncclient": "pooled-client",
    }


def test_stage_reasoning_replay_policy_is_attached_to_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        pass

    client = Client()

    class CapturingPool:
        @classmethod
        def from_endpoints(cls, **kwargs: Any) -> Client:
            return client

    monkeypatch.setattr(agent_runtime, "PooledLLM", CapturingPool)
    result = agent_runtime.build_stage_llm(
        StageConfig(
            model="model-a",
            api_key="key-a",
            base_url="https://a.example/v1",
            reasoning_replay="required",
        )
    )

    assert result is client
    assert client.reasoning_replay_policy == "required"


def test_rejects_ambiguous_model_endpoint_alignment():
    with pytest.raises(ValueError, match="models must contain"):
        agent_runtime.build_stage_llm(
            StageConfig(api_keys=["a", "b", "c"], models=["m1", "m2"])
        )
