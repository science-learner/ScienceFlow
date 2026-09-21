"""Configuration responsibility: parallel overrides."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import fields, is_dataclass
from typing import Any

from scienceflow.foundation.config.runtime.llm_lists import normalize_stage_mapping
from scienceflow.foundation.config.runtime.resource_modes import (
    expand_lnr_resource_control_mode_payload,
)
from scienceflow.foundation.config.schema.models.contracts.schema import (
    AgentConfig,
    Config,
    LnrConfig,
)


def _expand_parallel_manifest_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_parallel_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_parallel_manifest_value(v) for k, v in value.items()}
    return value

def _apply_parallel_manifest_dataclass_patch(
    target: Any,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    known = {f.name for f in fields(target)}
    for key, value in payload.items():
        if key not in known:
            warnings.warn(
                f"Ignoring unknown {label} key from parallel manifest: {key!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        cur = getattr(target, key)
        value = _expand_parallel_manifest_value(value)
        try:
            if isinstance(cur, bool):
                setattr(target, key, bool(value))
            elif isinstance(cur, int):
                setattr(target, key, int(value))
            elif isinstance(cur, float):
                setattr(target, key, float(value))
            elif isinstance(cur, str):
                setattr(target, key, str(value))
            elif isinstance(cur, list):
                if not isinstance(value, list):
                    raise TypeError(f"expected list for {key!r}")
                setattr(target, key, list(value))
            elif isinstance(cur, dict):
                if not isinstance(value, dict):
                    raise TypeError(f"expected mapping for {key!r}")
                setattr(target, key, dict(value))
            else:
                setattr(target, key, value)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Could not apply {label}.{key}={value!r} from parallel manifest: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

def apply_parallel_manifest_agent_overrides(cfg: Config) -> None:
    """Apply ``agent`` keys from parallel manifest (child env ``SCIENCEFLOW_PARALLEL_AGENT_JSON``).

    Set by :class:`scienceflow.runtime.parallel.execution.runner.ParallelRunner` when the manifest
    defines ``agent:``. Top-level ``AgentConfig`` fields and nested
    ``agent.code`` / ``agent.feedback`` ``StageConfig`` fields are supported.
    """
    raw = os.environ.get("SCIENCEFLOW_PARALLEL_AGENT_JSON", "").strip()
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"SCIENCEFLOW_PARALLEL_AGENT_JSON is invalid JSON: {e}; ignoring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if not isinstance(payload, dict):
        warnings.warn(
            "SCIENCEFLOW_PARALLEL_AGENT_JSON must be a JSON object; ignoring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    ag = cfg.agent
    nested_stage_keys = {"code", "feedback"}
    known = {f.name for f in fields(AgentConfig)} - nested_stage_keys
    for key, value in payload.items():
        if key in nested_stage_keys:
            if not isinstance(value, dict):
                warnings.warn(
                    f"Ignoring agent.{key} from parallel manifest: expected mapping, "
                    f"got {type(value).__name__}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            value = dict(value)
            normalize_stage_mapping(value)
            _apply_parallel_manifest_dataclass_patch(
                getattr(ag, key),
                value,
                label=f"agent.{key}",
            )
            continue
        if key not in known:
            warnings.warn(
                f"Ignoring unknown agent key from parallel manifest: {key!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        cur = getattr(ag, key)
        value = _expand_parallel_manifest_value(value)
        try:
            if isinstance(cur, bool):
                setattr(ag, key, bool(value))
            elif isinstance(cur, int):
                setattr(ag, key, int(value))
            elif isinstance(cur, float):
                setattr(ag, key, float(value))
            elif isinstance(cur, str):
                setattr(ag, key, str(value))
            else:
                setattr(ag, key, value)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Could not apply agent.{key}={value!r} from parallel manifest: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

def _apply_parallel_manifest_lnr_payload(
    lnr: LnrConfig,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    known = {f.name for f in fields(LnrConfig)}
    payload = expand_lnr_resource_control_mode_payload(payload)
    for key, value in payload.items():
        if key not in known:
            warnings.warn(
                f"Ignoring unknown {label} key from parallel manifest: {key!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        cur = getattr(lnr, key)
        try:
            if isinstance(cur, bool):
                setattr(lnr, key, bool(value))
            elif isinstance(cur, int):
                setattr(lnr, key, int(value))
            elif isinstance(cur, float):
                setattr(lnr, key, float(value))
            elif isinstance(cur, str):
                setattr(lnr, key, str(value))
            elif isinstance(cur, list):
                if not isinstance(value, list):
                    raise TypeError(f"expected list for {key!r}")
                setattr(lnr, key, list(value))
            elif is_dataclass(cur) and isinstance(value, dict):
                nested_known = {f.name for f in fields(cur)}
                for nested_key, nested_value in value.items():
                    if nested_key not in nested_known:
                        warnings.warn(
                            f"Ignoring unknown {label}.{key} key from parallel manifest: {nested_key!r}",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    nested_cur = getattr(cur, nested_key)
                    if isinstance(nested_cur, bool):
                        setattr(cur, nested_key, bool(nested_value))
                    elif isinstance(nested_cur, int):
                        setattr(cur, nested_key, int(nested_value))
                    elif isinstance(nested_cur, float):
                        setattr(cur, nested_key, float(nested_value))
                    elif isinstance(nested_cur, str):
                        setattr(cur, nested_key, str(nested_value))
                    else:
                        setattr(cur, nested_key, nested_value)
            else:
                setattr(lnr, key, value)
        except (TypeError, ValueError) as e:
            warnings.warn(
                f"Could not apply {label}.{key}={value!r} from parallel manifest: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

def apply_parallel_manifest_lnr_overrides(cfg: Config) -> None:
    """Apply lnr manifest keys for type: lnr children."""
    raw = os.environ.get("SCIENCEFLOW_PARALLEL_LNR_JSON", "").strip()
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"SCIENCEFLOW_PARALLEL_LNR_JSON is invalid JSON: {e}; ignoring lnr overrides.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if not isinstance(payload, dict):
        warnings.warn(
            "SCIENCEFLOW_PARALLEL_LNR_JSON must be a JSON object; ignoring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _apply_parallel_manifest_lnr_payload(cfg.lnr, payload, label="lnr")

__all__ = tuple(name for name in globals() if not name.startswith("__"))
