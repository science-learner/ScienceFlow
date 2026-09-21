"""ScienceFlow ownership of model-registry location and role defaults."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from inquirycraft.llm import ModelRegistry

from scienceflow.foundation.config.schema.models.contracts.schema import (
    Config,
    StageConfig,
)


def default_model_config_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return config_home.expanduser() / "scienceflow" / "models.json"


def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    selected = Path(path or default_model_config_path()).expanduser()
    _migrate_role_prefixed_aliases(selected)
    return ModelRegistry.from_json(selected)


def _migrate_role_prefixed_aliases(path: Path) -> bool:
    """Merge legacy ``code-*``/``feedback-*`` aliases into clean role routes."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    models = payload.get("models")
    defaults = payload.get("defaults")
    if not isinstance(models, dict) or not isinstance(defaults, dict):
        return False
    prefixes = (("code-", "code"), ("feedback-", "feedback"))
    renamed: dict[str, str] = {}
    migrated: dict[str, dict] = {}
    changed = False
    for alias, raw_spec in models.items():
        if not isinstance(raw_spec, dict):
            return False
        role = next(
            (name for prefix, name in prefixes if str(alias).startswith(prefix)),
            "",
        )
        clean = (
            str(raw_spec.get("model") or "").strip()
            if role
            else str(alias)
        )
        if not clean:
            return False
        if not role:
            migrated[clean] = raw_spec
            continue
        target = migrated.get(clean)
        if target is not None and target.get("model") != raw_spec.get("model"):
            return False
        target = target or {
            "model": raw_spec.get("model"),
            "pricing": raw_spec.get("pricing"),
            "endpoints": {},
        }
        if target.get("pricing") != raw_spec.get("pricing"):
            return False
        endpoint_map = target.get("endpoints")
        if not isinstance(endpoint_map, dict):
            return False
        raw_endpoints = raw_spec.get("endpoints")
        if isinstance(raw_endpoints, list):
            endpoint_map[role] = raw_endpoints
        elif isinstance(raw_endpoints, dict):
            endpoint_map.update(raw_endpoints)
        else:
            return False
        target["endpoints"] = endpoint_map
        migrated[clean] = target
        renamed[str(alias)] = clean
        changed = True
    if not changed:
        return False
    for name in ("code_models", "feedback_models"):
        values = defaults.get(name)
        if isinstance(values, list):
            defaults[name] = list(dict.fromkeys(renamed.get(str(item), str(item)) for item in values))
    payload["models"] = migrated
    ModelRegistry.from_mapping(payload, source=path)
    backup = path.with_name(path.name + ".legacy-aliases.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        backup.chmod(0o600)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def apply_registry_to_stage(
    stage: StageConfig,
    *,
    role: str,
    registry: ModelRegistry | None = None,
) -> bool:
    path = Path(stage.model_config_path).expanduser() if stage.model_config_path else None
    selected_registry = registry
    if selected_registry is None:
        selected_path = path or default_model_config_path()
        if not selected_path.is_file():
            return False
        selected_registry = load_model_registry(selected_path)
    aliases = [str(item).strip() for item in stage.model_aliases if str(item).strip()]
    if not aliases:
        aliases = list(selected_registry.default_aliases(f"{role}_models"))
    if not aliases:
        raise ValueError(
            f"Model configuration defaults.{role}_models must select at least one alias"
        )
    resolved = selected_registry.resolve(
        aliases,
        role=role,
        reasoning_replay=stage.reasoning_replay or None,
    )
    stage.model_aliases = list(resolved.aliases)
    stage.models = list(resolved.models)
    stage.api_keys = list(resolved.api_keys)
    stage.base_urls = list(resolved.base_urls)
    stage.model = stage.models[0]
    stage.api_key = stage.api_keys[0]
    stage.base_url = stage.base_urls[0]
    stage.reasoning_replay = resolved.reasoning_replay
    stage.model_config_path = str(selected_registry.source or path or default_model_config_path())
    selection = str(
        stage.model_selection or selected_registry.defaults.get("selection") or "auto"
    ).strip().lower()
    if selection not in {"auto", "spread", "fixed"}:
        raise ValueError(f"Unsupported model selection policy: {selection}")
    stage.model_selection = selection
    stage.api_routing_mode = "sticky" if selection == "fixed" else "sticky_failover"
    return True


def apply_model_registry(cfg: Config, path: str | Path | None = None) -> bool:
    explicit_path = Path(path).expanduser() if path else None
    fallback_path = explicit_path or default_model_config_path()
    registries: dict[Path, ModelRegistry] = {}
    applied = False
    for role in ("code", "feedback"):
        stage = getattr(cfg.agent, role)
        has_explicit_endpoint = bool(stage.api_keys or str(stage.api_key or "").strip())
        if has_explicit_endpoint and not stage.model_aliases:
            continue
        selected_path = (
            Path(stage.model_config_path).expanduser()
            if stage.model_config_path and explicit_path is None
            else fallback_path
        ).resolve(strict=False)
        if not selected_path.is_file():
            if stage.model_aliases:
                raise ValueError(f"Model configuration does not exist: {selected_path}")
            continue
        registry = registries.get(selected_path)
        if registry is None:
            registry = load_model_registry(selected_path)
            registries[selected_path] = registry
        if not stage.model_config_path:
            stage.model_config_path = str(selected_path)
        applied = apply_registry_to_stage(stage, role=role, registry=registry) or applied
    return applied


__all__ = [
    "apply_model_registry",
    "apply_registry_to_stage",
    "default_model_config_path",
    "load_model_registry",
]
