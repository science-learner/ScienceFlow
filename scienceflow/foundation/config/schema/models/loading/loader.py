"""Configuration responsibility: loader."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scienceflow.foundation.config.runtime.llm_lists import normalize_stage_mapping
from scienceflow.foundation.config.runtime.resource_modes import (
    apply_lnr_resource_control_mode_mapping as _apply_lnr_resource_control_mode_mapping,
)
from scienceflow.foundation.config.schema.models.contracts.base import (
    _CONFIG_INCLUDE_KEYS,
)
from scienceflow.foundation.config.schema.models.contracts.schema import (
    _REPL_CONFIG_KEYS,
    _TOOL_CONFIG_KEYS,
    _WORKSPACE_CONFIG_KEYS,
    Config,
)


def _config_alias(block_name: str, field_name: str) -> property:
    def _get(self: Config) -> Any:
        return getattr(getattr(self, block_name), field_name)

    def _set(self: Config, value: Any) -> None:
        setattr(getattr(self, block_name), field_name, value)

    return property(_get, _set)

for _alias_name in _WORKSPACE_CONFIG_KEYS:
    setattr(Config, _alias_name, _config_alias("workspace", _alias_name))

for _alias_name in _REPL_CONFIG_KEYS:
    setattr(Config, _alias_name, _config_alias("repl", _alias_name))

for _alias_name in _TOOL_CONFIG_KEYS:
    setattr(Config, _alias_name, _config_alias("tool", _alias_name))

def _move_legacy_top_level_keys(y: dict[str, Any], block_name: str, keys: tuple[str, ...]) -> None:
    """Move pre-nested top-level config keys into their canonical block."""
    block = y.get(block_name)
    if block is None:
        block = {}
    elif not isinstance(block, dict):
        return
    moved = False
    for key in keys:
        if key not in y:
            continue
        value = y.pop(key)
        if key not in block:
            block[key] = value
            moved = True
    if moved or block:
        y[block_name] = block

def _coerce_include_list(value: Any, *, source: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        includes: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"Config include entries in {source} must be strings, "
                    f"got {type(item).__name__}"
                )
            includes.append(item)
        return includes
    raise TypeError(f"Config include in {source} must be a string or list of strings")

def _load_yaml_with_includes(path: Path, *, seen: set[Path] | None = None) -> Any:
    """Load YAML and recursively merge relative ``include`` files.

    Included files are merged first; keys in *path* override included defaults.
    The include key is loader-only and never reaches the structured config.
    """
    path = path.expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    active = seen or set()
    if resolved in active:
        chain = " -> ".join(str(p) for p in [*active, resolved])
        raise ValueError(f"Recursive config include detected: {chain}")
    active.add(resolved)

    raw = OmegaConf.load(path)
    container = OmegaConf.to_container(raw, resolve=False)
    if not isinstance(container, dict):
        active.remove(resolved)
        return raw

    include_value: Any = None
    for key in _CONFIG_INCLUDE_KEYS:
        if key in container:
            if include_value is not None:
                raise ValueError(f"Config {path} uses both include keys; use only `include`")
            include_value = container.pop(key)
    _normalize_yaml_dict(container)

    merged = OmegaConf.create({})
    for inc in _coerce_include_list(include_value, source=path):
        inc_path = Path(inc).expanduser()
        if not inc_path.is_absolute():
            inc_path = path.parent / inc_path
        merged = OmegaConf.merge(merged, _load_yaml_with_includes(inc_path, seen=active))

    merged = OmegaConf.merge(merged, OmegaConf.create(container))
    active.remove(resolved)
    return merged

def _discard_removed_top_level_config(y: dict) -> None:
    legacy_ds = y.pop("deep_shallow", None)
    if isinstance(legacy_ds, dict):
        warnings.warn(
            "Config block `deep_shallow` was removed and is ignored.",
            DeprecationWarning,
            stacklevel=3,
        )
    legacy_data_split = y.pop("data_split", None)
    if isinstance(legacy_data_split, dict):
        warnings.warn(
            "Config block `data_split` was removed and is ignored; use `prep` "
            "with data-preparation skills instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    removed_top = [
        key
        for key in (
            "enable_deep_shallow",
            "deep_submission_dir",
            "deep_solution_dir",
            "prep_shallow_fraction",
            "hier_cold_start_step",
            "prep_light_validate_enabled",
            "prep_quality_retries",
            "prep_val_fraction",
            "prep_max_steps",
            "prep_split_only",
            "prep_python_timeout_sec",
            "prep_exec_feedback_max_chars",
            "prep_bash_timeout_sec",
        )
        if y.pop(key, None) is not None
    ]
    if removed_top:
        warnings.warn(
            "Config keys were removed and ignored: " + ", ".join(removed_top),
            DeprecationWarning,
            stacklevel=3,
        )


def _discard_removed_nested_config(y: dict) -> None:
    ex = y.get("exec")
    if isinstance(ex, dict):
        removed = [
            key
            for key in ("exec_timeout", "draft_exec_timeout", "deep_exec_timeout")
            if ex.pop(key, None) is not None
        ]
        if removed:
            warnings.warn(
                "Config keys under `exec` were removed and ignored: " + ", ".join(removed),
                DeprecationWarning,
                stacklevel=3,
            )
    agent = y.get("agent")
    if isinstance(agent, dict) and "init_code" in agent:
        agent.pop("init_code", None)
        warnings.warn(
            "Config key `agent.init_code` is no longer supported; ignored.",
            DeprecationWarning,
            stacklevel=3,
        )
    if isinstance(agent, dict):
        if agent.pop("deep_feedback_to_shallow", None) is not None:
            warnings.warn(
                "Config key `agent.deep_feedback_to_shallow` was removed and ignored.",
                DeprecationWarning,
                stacklevel=3,
            )
        for _k in ("check", "prep"):
            if _k in agent:
                agent.pop(_k, None)
                warnings.warn(
                    f"Config key `agent.{_k}` was removed; use `agent.code` (drafts, data_prep) "
                    "and `agent.feedback` (safety summary, insights).",
                    DeprecationWarning,
                    stacklevel=3,
                )
    lnr = y.get("lnr")
    if isinstance(lnr, dict):
        _apply_lnr_resource_control_mode_mapping(lnr)
        removed_lnr = [
            key
            for key in (
                "deep_enabled",
                "ensemble_enabled",
                "stage_commit_max_turns",
                "lnr_skill_debug_log_enabled",
                "state_packet_working_note_max_chars",
                "estra_max_decisions",
                "compact_fallback_max_chars",
                "resource_main_agent_advisory_commit_transcript",
                "resource_main_agent_advisory_tool_calls",
            )
            if lnr.pop(key, None) is not None
        ]
        if removed_lnr:
            warnings.warn(
                "Config keys under `lnr` were removed and ignored: " + ", ".join(removed_lnr),
                DeprecationWarning,
                stacklevel=3,
            )


def _normalize_legacy_data_dir(y: dict) -> None:
    if "data_dir" in y:
        workspace = y.setdefault("workspace", {})
        if isinstance(workspace, dict) and "input_data_dir" not in workspace:
            workspace["input_data_dir"] = y.pop("data_dir")
            warnings.warn(
                "Config key `data_dir` is deprecated; use `workspace.input_data_dir`.",
                DeprecationWarning,
                stacklevel=3,
            )
        else:
            y.pop("data_dir", None)


def _normalize_legacy_optimizer_steps(y: dict) -> None:
    if "optimizer_max_steps" in y:
        repl = y.setdefault("repl", {})
        value = y.pop("optimizer_max_steps", None)
        if isinstance(repl, dict) and "qa_max_steps" not in repl:
            repl["qa_max_steps"] = value
        warnings.warn(
            "Config key `optimizer_max_steps` is deprecated; use `repl.qa_max_steps` for ScienceAgent.",
            DeprecationWarning,
            stacklevel=3,
        )


def _normalize_legacy_workspace_root(y: dict) -> None:
    workspace = y.get("workspace")
    if isinstance(workspace, dict) and "workspace_dir" in workspace:
        if "task_workspace_root_dir" not in workspace:
            wd_raw = workspace.pop("workspace_dir")
            wd = Path(str(wd_raw)).expanduser()
            if wd.name == "workspace":
                parent = wd.parent
                workspace["task_workspace_root_dir"] = (
                    str(parent) if str(parent) not in ("", ".") else "."
                )
            else:
                workspace["task_workspace_root_dir"] = "."
                warnings.warn(
                    "Config key `workspace.workspace_dir` is deprecated; use "
                    "`workspace.task_workspace_root_dir` (task output root). "
                    "Could not migrate automatically (expected a path ending in "
                    f"'workspace'); using task_workspace_root_dir='.'. "
                    f"Ignored legacy workspace_dir={wd_raw!r}.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        else:
            workspace.pop("workspace_dir", None)
            warnings.warn(
                "Ignoring deprecated `workspace.workspace_dir` because "
                "`workspace.task_workspace_root_dir` is set.",
                DeprecationWarning,
                stacklevel=3,
            )


def _normalize_yaml_dict(y: dict) -> None:
    """Map deprecated keys before merging into structured Config (in-place)."""
    agent = y.get("agent")
    if isinstance(agent, dict):
        for name in ("code", "feedback"):
            if isinstance(agent.get(name), dict):
                normalize_stage_mapping(agent[name])
    _move_legacy_top_level_keys(y, "workspace", _WORKSPACE_CONFIG_KEYS)
    _move_legacy_top_level_keys(y, "repl", _REPL_CONFIG_KEYS)
    _move_legacy_top_level_keys(y, "tool", _TOOL_CONFIG_KEYS)
    _discard_removed_top_level_config(y)
    _discard_removed_nested_config(y)
    _normalize_legacy_data_dir(y)
    _normalize_legacy_optimizer_steps(y)
    _normalize_legacy_workspace_root(y)

_PARALLEL_MANIFEST_DEFAULTS_SKIP: frozenset[str] = frozenset(
    {
        "config",
        "workspace_base",
        "tasks",
        "defaults",
        "time_limit",
        "lnr",
        "agent",
        "gpu_list",
        "cpu_list",
    }
)

__all__ = tuple(name for name in globals() if not name.startswith("__"))
