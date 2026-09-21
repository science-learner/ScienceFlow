"""Map ScienceFlow stage configuration to InquiryCraft LLM options."""

from __future__ import annotations

import logging
from dataclasses import replace

from inquirycraft.llm import (
    LLMFactoryConfig,
    OnlineLLM,
    build_llm,
    normalize_reasoning_replay,
)

from scienceflow.foundation.config.llm.llm_http import llm_extra_client_kwargs
from scienceflow.foundation.config.llm.llm_pool import PooledLLM
from scienceflow.foundation.config.llm.model_registry import apply_registry_to_stage
from scienceflow.foundation.config.runtime.llm_lists import normalize_stage
from scienceflow.foundation.config.schema.settings import StageConfig

logger = logging.getLogger("scienceflow")


def build_stage_llm(
    stage: StageConfig,
    *,
    role: str = "code",
) -> PooledLLM | OnlineLLM:
    """Create an LLM from a validated ScienceFlow stage configuration."""
    request_seed = getattr(stage, "request_seed", None)
    stage = replace(stage)
    if stage.model_aliases:
        apply_registry_to_stage(stage, role=role)
    normalize_stage(stage)
    if stage.api_keys:
        for name in ("base_urls", "models"):
            count = len(getattr(stage, name))
            if count not in (0, 1, len(stage.api_keys)):
                raise ValueError(
                    f"{name} must contain one value or one per api_keys entry"
                )
    options = LLMFactoryConfig(
        model=stage.model,
        max_tokens=stage.max_tokens,
        frequency_penalty=stage.frequency_penalty,
        coalesce_system_messages=stage.coalesce_system_messages,
        base_url=stage.base_url,
        api_key=stage.api_key,
        api_keys=tuple(stage.api_keys),
        base_urls=tuple(stage.base_urls),
        models=tuple(stage.models),
        routing_mode=stage.api_routing_mode,
        sticky_id=stage.api_sticky_id,
        sticky_primary_index=stage.api_sticky_primary_index,
        rate_limit_cooldown_sec=stage.api_rate_limit_cooldown_sec,
        connection_cooldown_sec=stage.api_connection_cooldown_sec,
        request_seed=int(request_seed) if request_seed is not None else None,
        extra_client_kwargs=llm_extra_client_kwargs(stage),
    )
    client = build_llm(
        options,
        online_llm_factory=OnlineLLM,
        pooled_llm_type=PooledLLM,
        logger=logger,
    )
    try:
        client.reasoning_replay_policy = normalize_reasoning_replay(
            stage.reasoning_replay
        )
    except (AttributeError, TypeError):
        # Contract tests may use immutable or mapping-only constructor doubles.
        pass
    return client


__all__ = ["build_stage_llm"]
