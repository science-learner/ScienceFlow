"""Configuration responsibility: environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scienceflow.foundation.config.schema.models.contracts.schema import (
    Config,
    StageConfig,
)


def _apply_metadata_env(cfg: Config) -> None:
    mlebench_dir = os.environ.get("MLEBENCH_DATA_ROOT_DIR", "").strip()
    if mlebench_dir and not cfg.mlebench_data_root_dir:
        cfg.mlebench_data_root_dir = mlebench_dir
    exp_id = os.environ.get("SCIENCEFLOW_EXP_ID", "").strip()
    if exp_id:
        cfg.exp_id = exp_id
    metric_name = os.environ.get("CUSTOM_METRIC_NAME", "").strip()
    if metric_name:
        cfg.custom_metric_name = metric_name
        cfg.evaluator.metric.name = metric_name
    lower_is_better = os.environ.get("CUSTOM_IS_LOWER_BETTER", "").strip().lower()
    if lower_is_better in ("1", "true", "yes", "on"):
        cfg.custom_is_lower_better = True
        cfg.evaluator.metric.lower_is_better = True
    elif lower_is_better in ("0", "false", "no", "off"):
        cfg.custom_is_lower_better = False
        cfg.evaluator.metric.lower_is_better = False


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _apply_float_env(stage: StageConfig, raw: str, attr: str) -> None:
    if raw:
        try:
            setattr(stage, attr, float(raw))
        except ValueError:
            pass


def _apply_stage_routing_env(stage: StageConfig, *, name: str) -> None:
    upper = name.upper()
    routing_mode = _first_env(f"{upper}_LLM_ROUTING_MODE", "SCIENCEFLOW_LLM_ROUTING_MODE")
    if routing_mode:
        stage.api_routing_mode = routing_mode
    sticky_id = _first_env(f"{upper}_LLM_STICKY_ID", "SCIENCEFLOW_LLM_STICKY_ID")
    if sticky_id:
        stage.api_sticky_id = sticky_id
    primary = _first_env(
        f"{upper}_LLM_STICKY_PRIMARY_INDEX", "SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX"
    )
    if primary:
        try:
            stage.api_sticky_primary_index = int(primary)
        except ValueError:
            pass
    _apply_float_env(
        stage,
        _first_env(
            f"{upper}_LLM_RATE_LIMIT_COOLDOWN_SEC",
            "SCIENCEFLOW_LLM_RATE_LIMIT_COOLDOWN_SEC",
        ),
        "api_rate_limit_cooldown_sec",
    )
    _apply_float_env(
        stage,
        _first_env(
            f"{upper}_LLM_CONNECTION_COOLDOWN_SEC",
            "SCIENCEFLOW_LLM_CONNECTION_COOLDOWN_SEC",
        ),
        "api_connection_cooldown_sec",
    )
    coalesce = _first_env(
        f"{upper}_COALESCE_SYSTEM_MESSAGES", "SCIENCEFLOW_COALESCE_SYSTEM_MESSAGES"
    ).lower()
    if coalesce in ("1", "true", "yes", "on"):
        stage.coalesce_system_messages = True
    elif coalesce in ("0", "false", "no", "off"):
        stage.coalesce_system_messages = False


def _parse_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _apply_stage_env(cfg: Config) -> None:
    http_timeout = _parse_int_env("HTTP_TIMEOUT")
    for name in ("code", "feedback"):
        stage: StageConfig = getattr(cfg.agent, name)
        if http_timeout is not None:
            stage.http_timeout = http_timeout
        _apply_stage_routing_env(stage, name=name)


def _set_numeric_env(cfg: Config, env_name: str, attr: str, cast: Any) -> None:
    raw = os.environ.get(env_name, "").strip()
    if raw:
        try:
            setattr(cfg, attr, cast(raw))
        except ValueError:
            pass


def _set_bool_env(cfg: Config, env_name: str, attr: str) -> None:
    raw = os.environ.get(env_name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        setattr(cfg, attr, True)
    elif raw in ("0", "false", "no", "off"):
        setattr(cfg, attr, False)


def _apply_runtime_env(cfg: Config) -> None:
    numeric = (
        ("SCIENCEFLOW_LLM_STREAM_TIMEOUT", "scienceflow_llm_stream_timeout_sec", int),
        ("SCIENCEFLOW_BASH_TIMEOUT", "scienceflow_bash_timeout_sec", float),
        ("SCIENCEFLOW_BASH_TIMEOUT_SLOW", "scienceflow_bash_timeout_slow_sec", float),
        ("SCIENCEFLOW_STDOUT_MAX_CHARS", "scienceflow_stdout_max_chars", int),
        ("SCIENCEFLOW_BASH_SUCCESS_TAIL_LINES", "bash_success_tail_lines", int),
    )
    for env_name, attr, cast in numeric:
        _set_numeric_env(cfg, env_name, attr, cast)
    toggles = (
        ("SCIENCEFLOW_INTERACTION_LOG_FULL", "scienceflow_interaction_log_full"),
        ("SCIENCEFLOW_INTERACTION_LOG_COLOR", "scienceflow_interaction_log_color"),
        ("SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM", "scienceflow_interaction_log_llm_stream"),
        ("SCIENCEFLOW_BASH_STREAM_TO_INTERACTION_LOG", "scienceflow_bash_stream_to_interaction_log"),
        ("SCIENCEFLOW_TOOL_MEMORY_COMPRESSION", "tool_memory_compression"),
        ("SCIENCEFLOW_PARALLEL_BASH_ENABLED", "parallel_bash_enabled"),
        ("SCIENCEFLOW_PARALLEL_LLM_TOOL_CALLS", "parallel_llm_tool_calls"),
    )
    for env_name, attr in toggles:
        _set_bool_env(cfg, env_name, attr)
    log_level = os.environ.get("SCIENCEFLOW_INTERACTION_LOG_LEVEL", "").strip().lower()
    levels = {
        "minimal": "minimal", "m": "minimal", "normal": "normal",
        "n": "normal", "verbose": "verbose", "v": "verbose",
    }
    if log_level in levels:
        cfg.scienceflow_interaction_log_level = levels[log_level]


def _apply_path_guard_env(cfg: Config) -> None:
    raw = os.environ.get("PATH_GUARD_EXTRA_ROOTS", "").strip()
    if not raw:
        return
    import re as _re

    seen = {Path(path).resolve() for path in cfg.path_guard_extra_roots}
    for chunk in _re.split(r"[;:,\s]+", raw):
        value = chunk.strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path not in seen:
            cfg.path_guard_extra_roots.append(path)
            seen.add(path)


def _apply_env(cfg: Config) -> None:
    """Merge supported environment overrides into ``cfg`` in place."""
    _apply_metadata_env(cfg)
    _apply_stage_env(cfg)
    _apply_runtime_env(cfg)
    _apply_path_guard_env(cfg)


def _split_env(var: str) -> list[str]:
    """Split an env var by comma or whitespace, return non-empty items."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return []
    import re
    return [s.strip() for s in re.split(r"[,\s]+", raw) if s.strip()]

def _stage_endpoint_pairs(stage: StageConfig, *, stage_name: str) -> set[tuple[str, str]]:
    keys = [str(x).strip() for x in (stage.api_keys or []) if str(x).strip()]
    if not keys and str(stage.api_key or "").strip():
        keys = [str(stage.api_key).strip()]
    if not keys:
        return set()

    urls = [str(x).strip().rstrip("/") for x in (stage.base_urls or []) if str(x).strip()]
    if not urls and str(stage.base_url or "").strip():
        urls = [str(stage.base_url).strip().rstrip("/")]
    if not urls:
        urls = [""] * len(keys)
    elif len(urls) == 1 and len(keys) > 1:
        urls = urls * len(keys)
    elif len(urls) != len(keys):
        raise ValueError(
            f"agent.{stage_name} has {len(keys)} API key(s) but {len(urls)} base URL(s); "
            "provide one base URL to broadcast or the same count as keys.",
        )
    return set(zip(urls, keys))

__all__ = tuple(name for name in globals() if not name.startswith("__"))
