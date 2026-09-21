# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Load user and project dotenv files without depending on a source checkout."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from scienceflow.foundation.config.runtime.llm_lists import stage_env_values

logger = logging.getLogger("scienceflow.env_bootstrap")


def user_env_path() -> Path:
    """Return the selected user configuration file, including XDG support."""
    explicit = os.environ.get("SCIENCEFLOW_ENV_FILE", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return config_home.expanduser() / "scienceflow" / ".env"


def _dotenv_override_enabled() -> bool:
    return os.environ.get("SCIENCEFLOW_DOTENV_OVERRIDE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _normalize_endpoint_aliases(source: dict[str, str | None]) -> dict[str, str]:
    """Normalize within each source so lower-priority aliases cannot win later."""
    result = {key: value for key, value in source.items() if value is not None}
    for stage in ("code", "feedback"):
        for field in ("api_keys", "base_urls", "models"):
            items = stage_env_values(result, stage, field)
            if items:
                result[f"{stage.upper()}_{field.upper()}"] = ",".join(items)
    return result


def bootstrap_dotenv(
    *, repo_root: Path | None = None, env_file: Path | None = None
) -> tuple[Path, ...]:
    """Environment > explicit file > project > source checkout > user defaults.

    The old SCIENCEFLOW_DOTENV_OVERRIDE switch explicitly lets files override the
    process environment. YAML endpoint precedence remains owned by the config loader.
    """
    from dotenv import dotenv_values, find_dotenv

    explicit = env_file or os.environ.get("SCIENCEFLOW_ENV_FILE", "").strip()
    paths: list[Path] = []
    if explicit:
        selected = Path(explicit).expanduser().resolve()
        if not selected.is_file():
            raise ValueError(f"Configuration file does not exist: {selected}")
        paths.append(selected)
    project = find_dotenv(usecwd=True)
    if project:
        paths.append(Path(project).resolve())
    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        # A wheel's site-packages directory is not a project configuration location.
        if (root / "pyproject.toml").is_file() and (root / "scienceflow").is_dir():
            paths.append(root / ".env")
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    paths.extend(
        (
            config_home.expanduser() / "scienceflow" / ".env",
            Path.home() / ".scienceflow" / ".env",
        )
    )
    paths = list(dict.fromkeys(path.resolve() for path in paths if path.is_file()))
    override = _dotenv_override_enabled()
    os.environ.update(_normalize_endpoint_aliases(dict(os.environ)))
    for path in reversed(paths) if override else paths:
        values = _normalize_endpoint_aliases(dotenv_values(path))
        for key, value in values.items():
            if override or key not in os.environ:
                os.environ[key] = value
    logger.debug("Loaded %d ScienceFlow configuration file(s)", len(paths))
    return tuple(paths)
