# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Dataset subtree composition and compatibility exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scienceflow.research.state.dataset.discovery.scan_metadata import FLAT_DIR_FILE_LIST_THRESHOLD
from scienceflow.research.state.dataset.rendering.scan_tree_flat import (
    count_direct_entries,
    render_large_flat_directory,
)
from scienceflow.research.state.dataset.rendering.scan_tree_nested import (
    _append_raw_preview_for_capped_or_nested as _append_raw_preview_for_capped_or_nested,
    _append_raw_preview_for_dir as _append_raw_preview_for_dir,
    _append_raw_preview_for_subdir_with_children as _append_raw_preview_for_subdir_with_children,
    _try_resolve_isdir as _try_resolve_isdir,
    _walk_count_files_and_exts as _walk_count_files_and_exts,
    render_nested_directory,
)


def _build_subdir_tree_lines(
    subdir_path: Path,
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    include_val_outputs: bool = False,
    preview_raw_files: bool = False,
    probe_binary_files: bool = False,
) -> list[str]:
    """Route a subtree to its bounded flat or nested rendering owner."""

    if not subdir_path.exists() or not subdir_path.is_dir():
        return []
    directory_count, file_count, extensions = count_direct_entries(subdir_path)
    if directory_count == 0 and file_count > FLAT_DIR_FILE_LIST_THRESHOLD:
        return render_large_flat_directory(
            subdir_path,
            extensions,
            meta,
            meta_prefix,
            meta_readers,
            include_val_outputs=include_val_outputs,
        )
    return render_nested_directory(
        subdir_path,
        meta,
        meta_prefix,
        meta_readers,
        include_val_outputs=include_val_outputs,
        preview_raw_files=preview_raw_files,
        probe_binary_files=probe_binary_files,
    )


__all__ = [
    "_append_raw_preview_for_capped_or_nested",
    "_append_raw_preview_for_dir",
    "_append_raw_preview_for_subdir_with_children",
    "_build_subdir_tree_lines",
    "_try_resolve_isdir",
    "_walk_count_files_and_exts",
]
