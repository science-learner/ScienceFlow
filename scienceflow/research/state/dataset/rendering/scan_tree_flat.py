# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Large flat-directory decision and summary rendering."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from scienceflow.research.state.dataset.discovery.scan_metadata import (
    EXPAND_LIST_MAX_ITEMS,
    FLAT_DIR_META_EXTS,
    SCAN_EXCLUDE_DIRS,
    _ext_comment_with_hint,
    _format_meta_sample_lines,
    _hide_from_preview,
    _meta_file_comment,
)


def _is_directory(path: Path) -> bool:
    if path.is_dir():
        return True
    if not path.is_symlink():
        return False
    try:
        return path.resolve().is_dir()
    except OSError:
        return False


def count_direct_entries(path: Path) -> tuple[int, int, Counter[str]]:
    """Return visible-shape counts without retaining a large filename list."""

    directory_count = 0
    file_count = 0
    extensions: Counter[str] = Counter()
    for child in path.iterdir():
        if child.name in SCAN_EXCLUDE_DIRS:
            continue
        if _is_directory(child):
            directory_count += 1
            continue
        file_count += 1
        extensions[(child.suffix.lower() or "no_ext").lstrip(".")] += 1
    return directory_count, file_count, extensions


def _load_metadata_files(
    path: Path,
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    *,
    include_val_outputs: bool,
) -> list[tuple[str, dict[str, Any]]]:
    metadata_extensions = {extension.lstrip(".") for extension in FLAT_DIR_META_EXTS}
    result: list[tuple[str, dict[str, Any]]] = []
    for child in sorted(path.iterdir()):
        if _is_directory(child):
            continue
        extension = (child.suffix.lower() or "no_ext").lstrip(".")
        if extension not in metadata_extensions or _hide_from_preview(
            child.name,
            include_val_outputs,
        ):
            continue
        meta_key = f"{meta_prefix}/{child.name}"
        value = meta.get(meta_key)
        if value is None and child.suffix in meta_readers:
            value = meta_readers[child.suffix](child) or {}
            if value:
                meta[meta_key] = value
        result.append((child.name, value or {}))
    return result


def _summary_parts(
    path: Path,
    extensions: Counter[str],
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    *,
    include_val_outputs: bool,
) -> list[str]:
    metadata_files = _load_metadata_files(
        path,
        meta,
        meta_prefix,
        meta_readers,
        include_val_outputs=include_val_outputs,
    )
    max_metadata = EXPAND_LIST_MAX_ITEMS * 2
    parts: list[str] = []
    for name, value in metadata_files[:max_metadata]:
        parts.append(name + _meta_file_comment(value))
        if "columns" in value or value.get("sample_items"):
            parts.extend(_format_meta_sample_lines(value, indent=""))
    if len(metadata_files) > max_metadata:
        parts.append(
            f"… ({len(metadata_files)} meta files total, "
            f"showing first {max_metadata})"
        )

    metadata_extensions = {extension.lstrip(".") for extension in FLAT_DIR_META_EXTS}
    parts.extend(
        _ext_comment_with_hint(extension, count)
        for extension, count in extensions.most_common()
        if extension != "no_ext" and extension not in metadata_extensions
    )
    return parts


def render_large_flat_directory(
    path: Path,
    extensions: Counter[str],
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    *,
    include_val_outputs: bool,
) -> list[str]:
    """Summarize payload extensions while preserving named metadata files."""

    parts = _summary_parts(
        path,
        extensions,
        meta,
        meta_prefix,
        meta_readers,
        include_val_outputs=include_val_outputs,
    )
    lines: list[str] = []
    for index, part in enumerate(parts):
        if part.startswith("  |"):
            lines.append("│   " + part)
            continue
        prefix = "└── " if index == len(parts) - 1 else "├── "
        lines.append(prefix + part)
    return lines


__all__ = ["count_direct_entries", "render_large_flat_directory"]
