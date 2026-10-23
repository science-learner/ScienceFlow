# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Single-pass dataset traversal and bottom-up directory statistics."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WALK_MAX_DIRECTORIES = 80


@dataclass(slots=True)
class DatasetWalk:
    """Bounded traversal facts consumed by metadata and rendering owners."""

    entries: dict[str, tuple[list[str], list[str]]]
    capped_dirs: set[str]
    ext_counter: Counter[str]
    total_files: int
    dirs_seen: int
    dirs_exceeded: bool
    files_exceeded: bool


def walk_dataset(
    root: Path,
    *,
    exclude_dirs: set[str],
    budget_dirs: int | None,
    budget_files: int | None,
) -> DatasetWalk:
    """Traverse once while preserving names at recursion cut points."""

    entries: dict[str, tuple[list[str], list[str]]] = {}
    capped_dirs: set[str] = set()
    ext_counter: Counter[str] = Counter()
    total_files = 0
    dirs_seen = 0
    dirs_exceeded = False
    files_exceeded = False
    root_text = str(root)

    for dirpath, dirnames, filenames in os.walk(
        root_text,
        topdown=True,
        followlinks=True,
    ):
        dirs_seen += 1
        dirnames[:] = [name for name in dirnames if name not in exclude_dirs]
        rel = os.path.relpath(dirpath, root_text) if dirpath != root_text else "."
        rel = Path(rel).as_posix()
        if rel != "." and any(part in exclude_dirs for part in Path(rel).parts):
            continue
        if budget_dirs is not None and dirs_seen > budget_dirs:
            dirs_exceeded = True
            dirnames[:] = []

        accepted_files: list[str] = []
        for filename in filenames:
            if budget_files is not None and total_files >= budget_files:
                files_exceeded = True
                break
            accepted_files.append(filename)
            suffix = Path(filename).suffix.lstrip(".").lower() or "no_ext"
            ext_counter[suffix] += 1
            total_files += 1

        entries[rel] = (list(dirnames), accepted_files)
        if len(dirnames) > WALK_MAX_DIRECTORIES:
            capped_dirs.add(rel)
            dirnames[:] = []
        if files_exceeded:
            dirnames[:] = []

    return DatasetWalk(
        entries=entries,
        capped_dirs=capped_dirs,
        ext_counter=ext_counter,
        total_files=total_files,
        dirs_seen=dirs_seen,
        dirs_exceeded=dirs_exceeded,
        files_exceeded=files_exceeded,
    )


def aggregate_directory_stats(walk: DatasetWalk) -> dict[str, dict[str, Any]]:
    """Aggregate recursive file and extension counts from deepest paths upward."""

    stats: dict[str, dict[str, Any]] = {}
    for rel in sorted(walk.entries, key=lambda path: (-path.count("/"), path)):
        dirs_here, files_here = walk.entries[rel]
        file_count = len(files_here)
        extensions: Counter[str] = Counter(
            Path(filename).suffix.lstrip(".").lower() or "no_ext"
            for filename in files_here
        )
        capped = rel in walk.capped_dirs
        for dirname in dirs_here:
            child_rel = f"{rel}/{dirname}" if rel != "." else dirname
            capped = capped or child_rel in walk.capped_dirs
            child = stats.get(child_rel)
            if child is None:
                continue
            file_count += int(child.get("file_count") or 0)
            extensions += child.get("ext_counter") or Counter()
            capped = capped or bool(child.get("capped", False))
        stats[rel] = {
            "file_count": file_count,
            "ext_counter": extensions,
            "capped": capped,
        }
    return stats


__all__ = ["DatasetWalk", "aggregate_directory_stats", "walk_dataset"]
