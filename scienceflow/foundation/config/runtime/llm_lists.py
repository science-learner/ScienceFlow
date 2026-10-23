"""Canonical plural LLM configuration; scalar names are compatibility inputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def values(value: Any) -> list[str]:
    if value is None:
        return []
    items = re.split(r"[,\s]+", value) if isinstance(value, str) else value
    return [str(item).strip() for item in items if str(item).strip()]


def stage_env_values(env: Mapping[str, str], stage: str, field: str) -> list[str]:
    plural = field.upper()
    singular = plural[:-1]
    # Each stage is independent; unscoped variables must not fill either pool.
    for name in (
        f"{stage.upper()}_{plural}",
        f"{stage.upper()}_{singular}",
    ):
        result = values(env.get(name, ""))
        if result:
            return result
    return []


def normalize_stage_mapping(stage: dict[str, Any]) -> None:
    for plural, singular in (
        ("models", "model"),
        ("api_keys", "api_key"),
        ("base_urls", "base_url"),
    ):
        if plural in stage or singular in stage:
            items = values(stage.get(plural)) or values(stage.get(singular))
            stage[plural] = items
            stage[singular] = items[0] if items else ""


def normalize_stage(stage: Any) -> None:
    payload = {
        name: getattr(stage, name)
        for name in ("models", "model", "api_keys", "api_key", "base_urls", "base_url")
    }
    normalize_stage_mapping(payload)
    for name, value in payload.items():
        setattr(stage, name, value)
