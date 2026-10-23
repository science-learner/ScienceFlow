# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Public dataset scan orchestration and tree traversal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scienceflow.research.state.dataset.discovery.eval_signature import (
    extract_eval_signature as extract_eval_signature,
)
from scienceflow.research.state.dataset.rendering.scan_tree import (
    _walk_count_files_and_exts as _walk_count_files_and_exts,
)
from scienceflow.research.state.dataset.rendering.scan_render import (
    build_top_level_index,
    project_scan_result,
    render_dataset_tree,
)
from scienceflow.research.state.dataset.discovery.scan_traversal import (
    aggregate_directory_stats,
    walk_dataset,
)
from scienceflow.research.state.dataset.discovery.scan_metadata import (
    BINARY_PROBERS as BINARY_PROBERS,
    META_READERS as META_READERS,
    SCAN_EXCLUDE_DIRS,
    _ScanMetaCsvScope,
    _read_csv as _read_csv,
    _read_json as _read_json,
)


def scan_data_dir(
    data_dir: str | Path,
    expand_subdirs: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
    include_val_outputs: bool = False,
    preview_raw_files: bool = False,
    probe_binary_files: bool = False,
    *,
    walk_budget_dirs: int | None = 200_000,
    walk_budget_files: int | None = 500_000,
    probe_binary_dirs_budget: int = 8,
    preview_raw_dirs_budget: int = 12,
    meta_sample_max_bytes: int = 50_000,
    csv_max_rows_to_scan: int = 200_000,
) -> dict[str, Any]:
    """
    Scan data dir: traverse + read meta + stats. Output fixed schema.

    Args:
        data_dir: path to data dir (e.g. public/)
        expand_subdirs: top-level dir names to expand (e.g. ["Deep", "Shallow"]). Default None.
        exclude_dirs: dir names to exclude from scanning (e.g. ["working", "cache"]). Default None.
        include_val_outputs: when True, do not hide ``validation*`` basenames in the tree
            (split_prep holdout files at dataset root).
        probe_binary_files: when True, run optional Python readers (.mat/.npy/...) on one
            sample file per extension under each listed directory (best-effort).
        walk_budget_dirs: max ``os.walk`` directory visits (None = unlimited).
        walk_budget_files: max files counted across the walk (None = unlimited).
        probe_binary_dirs_budget: max top-level dir entries that run binary probes.
        preview_raw_dirs_budget: max top-level dir entries that append raw text previews.
        meta_sample_max_bytes: cap JSON read size for meta parsing.
        csv_max_rows_to_scan: max CSV rows to scan for row_count / samples.

    Returns:
        {
            "path": str,
            "dir_structure": str,
            "meta": {...},
            "stats": {"total_files": N, "by_extension": {...}},
            "scan_budget": {...},
        }
    """
    if expand_subdirs is None:
        expand_subdirs = []
    if exclude_dirs is None:
        exclude_dirs = []
    exclude_set = set(exclude_dirs) | SCAN_EXCLUDE_DIRS
    root = Path(data_dir).resolve()
    if not root.exists() or not root.is_dir():
        return {"error": "path_not_found", "path": str(root)}

    with _ScanMetaCsvScope(meta_sample_max_bytes, csv_max_rows_to_scan):
        return _scan_data_dir_body(
            root,
            expand_subdirs,
            exclude_set,
            include_val_outputs,
            preview_raw_files,
            probe_binary_files,
            walk_budget_dirs,
            walk_budget_files,
            probe_binary_dirs_budget,
            preview_raw_dirs_budget,
        )


def _scan_data_dir_body(
    root: Path,
    expand_subdirs: list[str],
    exclude_set: set[str],
    include_val_outputs: bool,
    preview_raw_files: bool,
    probe_binary_files: bool,
    walk_budget_dirs: int | None,
    walk_budget_files: int | None,
    probe_binary_dirs_budget: int,
    preview_raw_dirs_budget: int,
) -> dict[str, Any]:
    """Compose traversal, metadata projection, and bounded rendering owners."""

    walk = walk_dataset(
        root,
        exclude_dirs=exclude_set,
        budget_dirs=walk_budget_dirs,
        budget_files=walk_budget_files,
    )
    dir_stats = aggregate_directory_stats(walk)
    tree, meta = build_top_level_index(
        root,
        walk,
        dir_stats,
        exclude_dirs=exclude_set,
        include_val_outputs=include_val_outputs,
    )
    structure, preview_budget = render_dataset_tree(
        root,
        tree,
        meta,
        walk,
        dir_stats,
        expand_subdirs=expand_subdirs,
        include_val_outputs=include_val_outputs,
        preview_raw_files=preview_raw_files,
        probe_binary_files=probe_binary_files,
        probe_binary_dirs_budget=probe_binary_dirs_budget,
        preview_raw_dirs_budget=preview_raw_dirs_budget,
    )
    return project_scan_result(
        root,
        meta,
        walk,
        structure,
        preview_budget,
    )
