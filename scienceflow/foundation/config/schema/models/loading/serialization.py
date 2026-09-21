"""Configuration responsibility: serialization."""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from omegaconf import OmegaConf

from scienceflow.foundation.config.llm.model_registry import apply_model_registry
from scienceflow.foundation.config.runtime.llm_lists import normalize_stage_mapping
from scienceflow.foundation.config.schema.models.contracts.base import _DEFAULT_CFG
from scienceflow.foundation.config.schema.models.contracts.environment import _apply_env
from scienceflow.foundation.config.schema.models.contracts.schema import Config
from scienceflow.foundation.config.schema.models.loading.loader import (
    _load_yaml_with_includes,
    _normalize_yaml_dict,
)


def load_cfg(path: Path | str | None = None, cli_args: bool = True) -> Config:
    """Load config from YAML → apply env → merge CLI overrides → return Config."""
    path = Path(path) if path else _DEFAULT_CFG
    raw = _load_yaml_with_includes(path)
    container = OmegaConf.to_container(raw, resolve=False)
    if isinstance(container, dict):
        _normalize_yaml_dict(container)
        raw = OmegaConf.create(container)
    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(schema, raw)
    if cli_args:
        try:
            cli = OmegaConf.from_cli()
            valid = {f.name for f in Config.__dataclass_fields__.values()}
            raw = OmegaConf.to_container(cli)
            if isinstance(raw, dict):
                _normalize_yaml_dict(raw)
                overrides = {k: v for k, v in raw.items() if k in valid}
            else:
                overrides = {}
            if overrides:
                cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        except Exception:
            pass
    OmegaConf.resolve(cfg)
    config: Config = OmegaConf.to_object(cfg)
    _apply_env(config)
    apply_model_registry(config)
    try:
        config.config_source_path = str(path.expanduser().resolve())
    except OSError:
        config.config_source_path = str(path)
    return config


def _sanitize_config_dict_for_yaml(obj: Any) -> Any:
    """Convert Path / nested structures for ``yaml.safe_dump``."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_config_dict_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_config_dict_for_yaml(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_sanitize_config_dict_for_yaml(v) for v in obj)
    return obj


def _sanitize_endpoint_url(value: Any) -> str:
    """Keep endpoint identity while removing credentials and URL parameters."""
    text = str(value or "").strip()
    if not text:
        return ""

    def fallback() -> str:
        clean = text.split("?", 1)[0].split("#", 1)[0]
        if "@" not in clean:
            return clean
        prefix, endpoint = clean.rsplit("@", 1)
        if "://" in prefix:
            return f"{prefix.split('://', 1)[0]}://{endpoint}"
        return endpoint

    try:
        parsed = urlsplit(text)
    except ValueError:
        return fallback()
    if not parsed.scheme or not parsed.netloc:
        return fallback()
    host = parsed.hostname or parsed.netloc.rsplit("@", 1)[-1]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def dump_resolved_config_yaml(cfg: Config, dest: Path | None = None) -> None:
    """Write the effective :class:`Config` as YAML (paths as strings).

    *dest* defaults to ``<task_workspace_root_dir>/resolved_config.yaml``.
    Skip if env ``SCIENCEFLOW_SKIP_RESOLVED_CONFIG_YAML`` is truthy.
    """
    if os.environ.get("SCIENCEFLOW_SKIP_RESOLVED_CONFIG_YAML", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    root = Path(cfg.task_workspace_root_dir).expanduser().resolve()
    out = dest if dest is not None else root / "resolved_config.yaml"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(cfg)
        for stage in ("code", "feedback"):
            row = data["agent"][stage]
            normalize_stage_mapping(row)
            for secret_or_legacy in (
                "api_key",
                "api_keys",
                "base_url",
                "headers",
                "model",
            ):
                row.pop(secret_or_legacy, None)
            row["base_urls"] = [
                sanitized
                for value in row.get("base_urls", [])
                if (sanitized := _sanitize_endpoint_url(value))
            ]
        data = _sanitize_config_dict_for_yaml(data)
        header = (
            "# scienceflow resolved_config.yaml — effective Config after load, env, CLI, then prep_cfg.\n"
            "# Field config_source_path is the YAML used with --config when applicable.\n"
            "# API keys and headers are omitted; endpoint URLs exclude credentials and parameters.\n"
            "# model_config_path points to the private registry.\n"
        )
        with open(out, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(
                data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
    except Exception as e:
        warnings.warn(
            f"Could not write resolved_config.yaml to {out}: {e}",
            RuntimeWarning,
            stacklevel=2,
        )

def prep_cfg(cfg: Config) -> Config:
    """Derive execution workspace_dir, log_dir, submission paths; create directories."""
    root = Path(cfg.task_workspace_root_dir).expanduser().resolve()
    workspace_override = getattr(cfg, "_workspace_dir_override", None)
    inner = (
        Path(workspace_override).expanduser().resolve()
        if workspace_override is not None and str(workspace_override).strip()
        else root
    )
    cfg.workspace_dir = inner
    log_dir_override = getattr(cfg, "_log_dir_override", None)
    cfg.log_dir = (
        Path(log_dir_override).expanduser().resolve()
        if log_dir_override is not None and str(log_dir_override).strip()
        else root / "logs"
    )
    cfg.submission_dir = inner / "submissions"

    for d in (cfg.log_dir, inner, cfg.submission_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    roots: list[Path] = []
    seen_r: set[Path] = set()
    for p in getattr(cfg, "path_guard_extra_roots", None) or []:
        s = str(p).strip()
        if not s:
            continue
        rp = Path(s).expanduser().resolve()
        if rp not in seen_r:
            roots.append(rp)
            seen_r.add(rp)
    cfg.path_guard_extra_roots = roots

    lnr = getattr(cfg, "lnr", None)
    if lnr is not None and bool(getattr(lnr, "compact_on_context_limit", False)):
        try:
            floor = int(getattr(lnr, "context_limit_min_messages", 0) or 0)
        except (TypeError, ValueError):
            floor = 0
        if floor > 0 and int(getattr(cfg, "max_messages", 0) or 0) < floor:
            cfg.max_messages = floor

    dump_resolved_config_yaml(cfg)
    return cfg

__all__ = tuple(name for name in globals() if not name.startswith("__"))
