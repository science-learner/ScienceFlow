"""Publish immutable report bundles through one atomic directory pointer."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

_FILES = ("report.md", "report.html", "report.json")
_VERSIONS = ".report-versions"
_CURRENT = ".report-current"


def _link(target: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.symlink_to(target)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _switch(root: Path, version: Path) -> None:
    """The only commit point: all public links now select this full bundle."""
    _link(str(version.relative_to(root)), root / _CURRENT)


def _copy_bundle(source: Path, versions: Path) -> Path:
    version = versions / uuid4().hex
    # Unpublished directories are harmless if a process dies during copying.
    version.mkdir()
    for name in _FILES:
        if (source / name).is_file():
            shutil.copyfile(source / name, version / name)
    assets = version / "report_assets"
    assets.mkdir()
    source_assets = source / "report_assets"
    if source_assets.is_dir():
        for path in source_assets.glob("*.svg"):
            if path.is_symlink():
                raise ValueError("Legacy report assets must not be symlinks")
            shutil.copyfile(path, assets / path.name)
    return version


def _prepare_links(root: Path, version: Path) -> None:
    # Existing directories stay in place. Individual asset aliases, like the
    # document aliases, all traverse the same .report-current pointer.
    assets = root / "report_assets"
    if assets.is_symlink():
        raise ValueError("Report assets directory must not be a symlink")
    assets.mkdir(exist_ok=True)
    for name in _FILES:
        path = root / name
        target = f"{_CURRENT}/{name}"
        if not path.is_symlink() or os.readlink(path) != target:
            _link(target, path)
    for path in (version / "report_assets").glob("*.svg"):
        target = f"../{_CURRENT}/report_assets/{path.name}"
        alias = assets / path.name
        if not alias.is_symlink() or os.readlink(alias) != target:
            _link(target, alias)


def publish_bundle(source: Path, root: Path) -> None:
    """Retain old bundles; a failed publication leaves the old pointer intact."""
    if root.resolve() != root:
        raise ValueError("Report publication directory has moved")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "report_assets").is_symlink():
        raise ValueError("Report assets directory must not be a symlink")
    versions = root / _VERSIONS
    if versions.is_symlink():
        raise ValueError("Report versions directory must not be a symlink")
    versions.mkdir(exist_ok=True)
    current = root / _CURRENT
    if current.exists() or current.is_symlink():
        if (
            not current.is_symlink()
            or not current.is_dir()
            or current.resolve().parent != versions
        ):
            raise ValueError("Invalid report version pointer")
    else:
        # Bootstrap legacy regular files before replacing them with aliases.
        # Each replacement still reads identical old bytes until final commit.
        if any((root / name).is_symlink() for name in _FILES):
            raise ValueError("Unexpected legacy report symlink")
        legacy = _copy_bundle(root, versions)
        _switch(root, legacy)
    version = _copy_bundle(source, versions)
    _prepare_links(root, version)
    _switch(root, version)
