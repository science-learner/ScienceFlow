"""Configuration responsibility: profiles."""

from __future__ import annotations

import warnings
from dataclasses import fields
from typing import Any

from omegaconf import OmegaConf

from scienceflow.foundation.config.schema.models.loading.loader import (
    _PARALLEL_MANIFEST_DEFAULTS_SKIP,
    _normalize_yaml_dict,
)
from scienceflow.foundation.config.schema.models.contracts.schema import Config, LnrConfig
from scienceflow.foundation.config.runtime.resource_modes import expand_lnr_resource_control_mode_payload

def merge_parallel_manifest_cfg_patch(
    defaults: dict[str, Any] | None, task: dict[str, Any] | None
) -> dict[str, Any]:
    """Top-level :class:`Config` keys from manifest ``defaults`` and ``task`` (task wins).

    Used by :func:`apply_repl_manifest_defaults` (``task`` empty) and by
    :class:`~scienceflow.runtime.parallel.execution.runner.ParallelRunner` to pass overrides into
    prep/run subprocesses via ``SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON``.
    """
    field_names = {f.name for f in fields(Config)}
    patch_dict: dict[str, Any] = {}
    for source in (defaults or {}, task or {}):
        for k, v in source.items():
            if k in _PARALLEL_MANIFEST_DEFAULTS_SKIP:
                continue
            patch_dict[k] = v
    if not patch_dict:
        return {}
    _normalize_yaml_dict(patch_dict)
    return {k: v for k, v in patch_dict.items() if k in field_names}

def apply_repl_manifest_defaults(
    cfg: Config,
    defaults: dict[str, Any] | None,
    task: dict[str, Any] | None = None,
) -> None:
    """Merge parallel-style manifest ``defaults:`` into *cfg* for ``repl --manifest``.

    Keys like ``repl.repl_max_steps`` / legacy ``repl_max_steps`` or
    ``tool.scienceflow_interaction_log_level`` / legacy ``scienceflow_interaction_log_level``
    live under manifest ``defaults`` or a single REPL task entry.

    Only :class:`Config` fields are merged after legacy top-level keys are moved
    into their canonical blocks; manifest-only keys (``config``,
    ``workspace_base``, ``time_limit``, nested blocks, …) are skipped.
    """
    patch_dict = merge_parallel_manifest_cfg_patch(defaults, task)
    if not patch_dict:
        return
    oc_cfg = OmegaConf.structured(cfg)
    oc_patch = OmegaConf.create(patch_dict)
    merged = OmegaConf.merge(oc_cfg, oc_patch)
    new_obj: Config = OmegaConf.to_object(merged)
    for f in fields(Config):
        setattr(cfg, f.name, getattr(new_obj, f.name))

_BUILTIN_PROFILE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "opt_solver": {
        "lnr": {"code_organization_hint": "opt_solver"},
        "repl": {"repl_code_organization_hint": "opt_solver"},
    },
    "optimization_solver": {
        "lnr": {"code_organization_hint": "opt_solver"},
        "repl": {"repl_code_organization_hint": "opt_solver"},
    },
    "artifact_solver": {
        "lnr": {"code_organization_hint": "opt_solver"},
        "repl": {"repl_code_organization_hint": "opt_solver"},
    },
}

def _profile_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

def _profile_override_for(
    overrides: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    target = _profile_key(profile)
    for key, value in overrides.items():
        if _profile_key(key) == target and isinstance(value, dict):
            return dict(value)
    return {}

def _merge_profile_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        for block_name, block_value in payload.items():
            if (
                isinstance(block_value, dict)
                and isinstance(merged.get(block_name), dict)
            ):
                merged[block_name] = {**merged[block_name], **block_value}
            else:
                merged[block_name] = block_value
    return merged

def _merge_profile_block(
    target: Any,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    known = {f.name for f in fields(target)}
    if isinstance(target, LnrConfig):
        payload = expand_lnr_resource_control_mode_payload(payload)
    patch: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            warnings.warn(
                f"Ignoring unknown profile override key {label}.{key!r}",
                UserWarning,
                stacklevel=3,
            )
            continue
        patch[key] = value
    if not patch:
        return
    merged = OmegaConf.merge(OmegaConf.structured(target), OmegaConf.create(patch))
    new_obj = OmegaConf.to_object(merged)
    for f in fields(target):
        setattr(target, f.name, getattr(new_obj, f.name))

def apply_profile_overrides(cfg: Config, profile: str | None = None) -> None:
    """Apply task-profile defaults for LNR/REPL/evaluator isolation.

    This function is intentionally separate from :func:`load_cfg`: CLI callers
    invoke it after generic manifest defaults and before explicit manifest
    ``lnr``/``agent`` patches, so task-level values remain the highest priority.
    """
    package_profile = ""
    package_spec = None
    if profile is None:
        try:
            from scienceflow.runtime.task_package import find_task_package

            package_spec = find_task_package(str(cfg.exp_id or ""))
            package_profile = str(package_spec.profile or "") if package_spec is not None else ""
        except Exception:
            package_profile = ""
            package_spec = None
    configured_profile = str(cfg.evaluator.task_profile or "").strip()
    raw_profile = profile or package_profile or configured_profile or "auto"
    cfg.evaluator.task_profile = str(raw_profile).strip() or "auto"
    if package_spec is not None:
        if str(cfg.evaluator.backend or "").strip().lower() in {"", "auto"}:
            cfg.evaluator.backend = "task_package"
        if not str(cfg.evaluator.candidate.artifact or "").strip():
            cfg.evaluator.candidate.artifact = package_spec.artifact_path
        if not str(cfg.evaluator.candidate.artifact_kind or "").strip():
            cfg.evaluator.candidate.artifact_kind = package_spec.artifact_kind
        if str(cfg.evaluator.metric.name or "").strip() in {"", "metric"}:
            cfg.evaluator.metric.name = package_spec.metric_name
        if not str(cfg.evaluator.metric.type or "").strip():
            cfg.evaluator.metric.type = package_spec.metric_type
        if cfg.evaluator.metric.lower_is_better is None:
            cfg.evaluator.metric.lower_is_better = package_spec.lower_is_better
    key = _profile_key(raw_profile) or "auto"
    builtin = _profile_override_for(_BUILTIN_PROFILE_OVERRIDES, key)
    custom = _profile_override_for(cfg.profile_overrides or {}, key)
    merged = _merge_profile_payloads(builtin, custom)
    if not merged:
        return
    for block_name, payload in merged.items():
        if not isinstance(payload, dict):
            warnings.warn(
                f"Ignoring profile override block {key}.{block_name!r}: expected mapping",
                UserWarning,
                stacklevel=2,
            )
            continue
        if block_name == "lnr":
            _merge_profile_block(cfg.lnr, payload, label=f"{key}.lnr")
        elif block_name == "repl":
            _merge_profile_block(cfg.repl, payload, label=f"{key}.repl")
        elif block_name == "evaluator":
            _merge_profile_block(cfg.evaluator, payload, label=f"{key}.evaluator")
        else:
            warnings.warn(
                f"Ignoring unsupported profile override block {key}.{block_name!r}",
                UserWarning,
                stacklevel=2,
            )

__all__ = tuple(name for name in globals() if not name.startswith("__"))
