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

from __future__ import annotations

import sysconfig
from pathlib import Path

DEFAULT_SKILL_LIBRARY_REL = Path("skills")
INSTALLED_SKILL_LIBRARY_REL = Path("share") / "scienceflow" / "skills"


def installed_skill_library_dir(*, data_root: Path | None = None) -> Path:
    """Return the wheel-installed Markdown skill library directory."""
    root = data_root or Path(sysconfig.get_path("data"))
    return Path(root).expanduser().resolve(strict=False) / INSTALLED_SKILL_LIBRARY_REL


def default_skill_library_dir(project_root: Path) -> Path:
    """Resolve built-in skills from a source checkout or an installed wheel."""
    source_dir = (
        Path(project_root).expanduser().resolve(strict=False)
        / DEFAULT_SKILL_LIBRARY_REL
    )
    if source_dir.is_dir():
        return source_dir
    installed_dir = installed_skill_library_dir()
    if installed_dir.is_dir():
        return installed_dir
    return source_dir


def resolve_skill_library_dir(
    project_root: Path,
    *,
    configured_dir: str | Path | None = None,
) -> Path:
    """Resolve an optional host-selected library before the built-in default."""
    if configured_dir is not None and str(configured_dir).strip():
        return Path(configured_dir).expanduser().resolve(strict=False)
    return default_skill_library_dir(project_root)
