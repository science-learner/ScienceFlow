# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Top-level dataset index projection and bounded tree rendering."""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scienceflow.research.state.dataset.discovery.scan_metadata import (
    JSON_KEY_SHOW_LIMIT,
    META_READERS,
    SUBDIRS_SHOW_LIMIT,
    _append_binary_probe_samples,
    _binary_hint,
    _ext_counter_from_dir_files,
    _ext_list_with_hints,
    _format_meta_sample_lines,
    _hide_from_preview,
    _meta_file_comment,
)
from scienceflow.research.state.dataset.discovery.scan_traversal import DatasetWalk
from scienceflow.research.state.dataset.rendering.scan_tree import (
    _append_raw_preview_for_capped_or_nested,
    _append_raw_preview_for_dir,
    _build_subdir_tree_lines,
)

Tree = dict[str, dict[str, Any]]
TreeItem = tuple[str, str, dict[str, Any]]

DIR_GROUP_THRESHOLD = 20
FILE_GROUP_THRESHOLD = 15


@dataclass(slots=True)
class PreviewBudget:
    """Mutable rendering-only budget; traversal facts remain immutable."""

    binary_dirs: int
    raw_dirs: int


def build_top_level_index(
    root: Path,
    walk: DatasetWalk,
    dir_stats: dict[str, dict[str, Any]],
    *,
    exclude_dirs: set[str],
    include_val_outputs: bool,
) -> tuple[Tree, dict[str, Any]]:
    """Project direct entries and bounded metadata samples from traversal facts."""

    tree: Tree = {}
    meta: dict[str, Any] = {}
    top_dirs, top_files = walk.entries.get(".", ([], []))
    for rel in sorted(top_dirs + top_files):
        if rel in exclude_dirs or _hide_from_preview(rel, include_val_outputs):
            continue
        if rel in top_dirs:
            sub_dirs, _ = walk.entries.get(rel, ([], []))
            tree[rel] = {
                "type": "dir",
                "file_count": dir_stats.get(rel, {}).get("file_count", 0),
                "sub_sample": [f"{name}/" for name in sub_dirs][
                    :SUBDIRS_SHOW_LIMIT
                ],
                "subdir_count": len(sub_dirs),
            }
            continue
        path = root / rel
        tree[rel] = {
            "type": "file",
            "ext": (path.suffix.lower() or "no_ext").lstrip("."),
        }
        reader = META_READERS.get(path.suffix)
        if reader is not None and (value := reader(path)) is not None:
            meta[rel] = value

    _load_subdir_metadata(
        root,
        tree,
        meta,
        include_val_outputs=include_val_outputs,
    )
    return tree, meta


def _load_subdir_metadata(
    root: Path,
    tree: Tree,
    meta: dict[str, Any],
    *,
    include_val_outputs: bool,
) -> None:
    for dirname, entry in tree.items():
        if entry.get("type") != "dir":
            continue
        directory = root / dirname
        for extension, reader in META_READERS.items():
            for path in list(directory.glob(f"*{extension}"))[:5]:
                if not path.is_file() or _hide_from_preview(
                    path.name,
                    include_val_outputs,
                ):
                    continue
                value = reader(path)
                if value is not None:
                    meta[f"{dirname}/{path.name}"] = value


def _dir_prefix(name: str) -> str:
    match = re.match(r"^([^\d]+)", name)
    return (match.group(1).rstrip("_-") or name) if match else name


def _directory_items(
    tree: Tree,
    dir_stats: dict[str, dict[str, Any]],
    *,
    include_val_outputs: bool,
) -> list[TreeItem]:
    grouped: dict[str, list[str]] = {}
    for dirname in sorted(
        name for name, entry in tree.items() if entry.get("type") == "dir"
    ):
        grouped.setdefault(_dir_prefix(dirname), []).append(dirname)

    items: list[TreeItem] = []
    for prefix in sorted(
        grouped,
        key=lambda value: grouped[value][0] if grouped[value] else "",
    ):
        names = grouped[prefix]
        if len(names) < DIR_GROUP_THRESHOLD:
            items.extend(
                (name, "dir", tree[name])
                for name in sorted(names)
                if not _hide_from_preview(name, include_val_outputs)
            )
            continue
        extensions: Counter[str] = Counter()
        for name in names:
            extensions += dir_stats.get(name, {}).get("ext_counter") or Counter()
        dominant = "." + extensions.most_common(1)[0][0] if extensions else ""
        items.append(
            (
                f"{prefix}*/",
                "dir_group",
                {
                    "dir_count": len(names),
                    "file_count": sum(tree[name]["file_count"] for name in names),
                    "ext": dominant,
                    "_first_dir": names[0] if names else None,
                    "_ext_counter": extensions,
                },
            )
        )
    return items


def _file_items(tree: Tree, *, include_val_outputs: bool) -> list[TreeItem]:
    groups: dict[str, list[str]] = {}
    for name, entry in tree.items():
        if entry.get("type") != "file":
            continue
        extension = entry.get("ext", "no_ext")
        extension = "" if extension == "no_ext" else extension
        groups.setdefault(f".{extension}" if extension else "no_ext", []).append(name)

    items: list[TreeItem] = []
    def ordering(row: tuple[str, list[str]]) -> tuple[bool, bool, str]:
        return row[0] != ".csv", row[0] != ".json", row[0]

    for extension, names in sorted(groups.items(), key=ordering):
        if len(names) > FILE_GROUP_THRESHOLD:
            suffix = extension if extension not in (".no_ext", "no_ext") else ""
            items.append(
                (
                    f"*{suffix}({len(names):,})",
                    "group",
                    {"count": len(names), "ext": suffix},
                )
            )
            continue
        items.extend(
            (name, "file", tree[name])
            for name in sorted(names)
            if not _hide_from_preview(name, include_val_outputs)
        )
    return items


def _dominant_extensions(
    tree: Tree,
    dir_stats: dict[str, dict[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, entry in tree.items():
        if entry.get("type") != "dir":
            continue
        extensions = dir_stats.get(name, {}).get("ext_counter") or Counter()
        if extensions:
            result[name] = "." + extensions.most_common(1)[0][0]
    return result


def _identifier_column(meta: dict[str, Any], dirname: str) -> str:
    for name, value in meta.items():
        if not name.endswith(".csv") or not name.startswith(dirname + "."):
            continue
        columns = value.get("columns", [])
        for column in (
            "image_name",
            "image_id",
            "StudyInstanceUID",
            "Id",
            "id",
            "fname",
        ):
            if column in columns:
                return f"{{{column}}}"
        break
    return "{id}"


def _directory_comment(
    name: str,
    info: dict[str, Any],
    *,
    meta: dict[str, Any],
    dir_stats: dict[str, dict[str, Any]],
    dominant_extensions: dict[str, str],
    root_capped: bool,
) -> str:
    stats = dir_stats.get(name, {})
    if stats.get("capped"):
        return " # (many, scan skipped)"
    if root_capped:
        return " # (not expanded)"

    file_count = info.get("file_count", 0)
    extensions = stats.get("ext_counter") or Counter()
    if len(extensions) > 1:
        comment = f" # ({file_count:,} files: {_ext_list_with_hints(extensions)})"
    else:
        extension = dominant_extensions.get(name, "")
        suffix = f"{_identifier_column(meta, name)}{extension}" if extension else ""
        comment = f" # {suffix}({file_count:,})" if suffix else f" # {file_count:,} files"
        hint = _binary_hint(extension.lstrip(".")) if extension else ""
        if hint:
            comment += f"  [{hint}]"

    samples = info.get("sub_sample") or []
    subdir_count = info.get("subdir_count", len(samples))
    if samples or subdir_count:
        comment += " subdirs: " + ", ".join(samples[:SUBDIRS_SHOW_LIMIT])
        if subdir_count > SUBDIRS_SHOW_LIMIT:
            comment += f" ... ({subdir_count} total)"
    return comment


def _directory_group_comment(
    root: Path,
    info: dict[str, Any],
    *,
    root_capped: bool,
) -> str:
    dir_count = info.get("dir_count", 0)
    file_count = info.get("file_count", 0)
    if root_capped:
        comment = f" # (~{dir_count:,} subdirs, not expanded)"
        first_dir = info.get("_first_dir")
        if first_dir:
            extensions = _ext_counter_from_dir_files(root / first_dir)
            if extensions:
                comment += f"; e.g. {first_dir}/: {_ext_list_with_hints(extensions)}"
        return comment

    extensions = info.get("_ext_counter") or Counter()
    if len(extensions) > 1:
        return (
            f" # ({dir_count:,} subdirs, {file_count:,} files: "
            f"{_ext_list_with_hints(extensions)})"
        )
    extension = info.get("ext", "")
    suffix = f"{{id}}{extension}" if extension else ""
    comment = (
        f" # {suffix}({dir_count:,} subdirs, {file_count:,} files)"
        if suffix
        else f" # ({dir_count:,} subdirs, {file_count:,} files)"
    )
    hint = _binary_hint(extension.lstrip(".")) if extension else ""
    return comment + (f"  [{hint}]" if hint else "")


def _item_label(
    root: Path,
    name: str,
    kind: str,
    info: dict[str, Any],
    *,
    meta: dict[str, Any],
    dir_stats: dict[str, dict[str, Any]],
    dominant_extensions: dict[str, str],
    root_capped: bool,
) -> str:
    if kind == "dir":
        return name + "/" + _directory_comment(
            name,
            info,
            meta=meta,
            dir_stats=dir_stats,
            dominant_extensions=dominant_extensions,
            root_capped=root_capped,
        )
    if kind == "dir_group":
        return name + _directory_group_comment(root, info, root_capped=root_capped)
    if kind == "file":
        return name + _meta_file_comment(meta.get(name, {}))
    return name


def _append_file_meta(name: str, meta: dict[str, Any], lines: list[str]) -> None:
    if name not in meta:
        return
    lines.extend(f"│   {sample}" for sample in _format_meta_sample_lines(meta[name]))


def _append_unexpanded_meta(
    name: str,
    meta: dict[str, Any],
    lines: list[str],
) -> None:
    for extension in (".json", ".csv"):
        sample_key = next(
            (
                key
                for key in sorted(meta)
                if key.startswith(name + "/") and key.endswith(extension)
            ),
            None,
        )
        if sample_key is None:
            continue
        sample_meta = meta.get(sample_key)
        if not sample_meta or not (
            "columns" in sample_meta
            or sample_meta.get("sample_items") is not None
            or sample_meta.get("type")
        ):
            continue
        lines.append("│   └── [sample] " + os.path.basename(sample_key))
        lines.extend(
            f"│   {sample}" for sample in _format_meta_sample_lines(sample_meta)
        )


def _append_directory_previews(
    root: Path,
    name: str,
    info: dict[str, Any],
    *,
    dir_stats: dict[str, dict[str, Any]],
    preview_raw_files: bool,
    probe_binary_files: bool,
    budget: PreviewBudget,
    lines: list[str],
) -> None:
    directory = root / name
    stats = dir_stats.get(name, {})
    extensions = stats.get("ext_counter") or Counter()
    if preview_raw_files and budget.raw_dirs > 0:
        budget.raw_dirs -= 1
        if stats.get("capped", False) or info.get("sub_sample"):
            _append_raw_preview_for_capped_or_nested(directory, lines)
        else:
            _append_raw_preview_for_dir(directory, extensions, lines, nested=False)
    if probe_binary_files and not stats.get("capped") and budget.binary_dirs > 0:
        budget.binary_dirs -= 1
        _append_binary_probe_samples(
            directory,
            extensions,
            lines,
            probe_binary_files=True,
        )


def _append_group_probe(
    root: Path,
    info: dict[str, Any],
    *,
    probe_binary_files: bool,
    budget: PreviewBudget,
    lines: list[str],
) -> None:
    first_dir = info.get("_first_dir")
    if not probe_binary_files or not first_dir or budget.binary_dirs <= 0:
        return
    sample_path = root / first_dir
    if not sample_path.is_dir():
        return
    extensions = Counter(info.get("_ext_counter") or Counter())
    extensions.update(_ext_counter_from_dir_files(sample_path))
    budget.binary_dirs -= 1
    lines.append(
        f"│   └── [dir_group sample: {first_dir}/] "
        f"(binary probe from 1 of ~{info.get('dir_count', 0):,} dirs)"
    )
    _append_binary_probe_samples(
        sample_path,
        extensions,
        lines,
        probe_binary_files=True,
    )


def _append_expanded_subtree(
    root: Path,
    name: str,
    meta: dict[str, Any],
    *,
    include_val_outputs: bool,
    preview_raw_files: bool,
    probe_binary_files: bool,
    lines: list[str],
) -> None:
    subtree = _build_subdir_tree_lines(
        root / name,
        meta,
        name,
        META_READERS,
        include_val_outputs,
        preview_raw_files=preview_raw_files,
        probe_binary_files=probe_binary_files,
    )
    lines.extend(f"│   {line}" for line in subtree)


def render_dataset_tree(
    root: Path,
    tree: Tree,
    meta: dict[str, Any],
    walk: DatasetWalk,
    dir_stats: dict[str, dict[str, Any]],
    *,
    expand_subdirs: list[str],
    include_val_outputs: bool,
    preview_raw_files: bool,
    probe_binary_files: bool,
    probe_binary_dirs_budget: int,
    preview_raw_dirs_budget: int,
) -> tuple[str, PreviewBudget]:
    """Render bounded facts without mutating traversal state."""

    items = _directory_items(
        tree,
        dir_stats,
        include_val_outputs=include_val_outputs,
    ) + _file_items(tree, include_val_outputs=include_val_outputs)
    dominant = _dominant_extensions(tree, dir_stats)
    budget = PreviewBudget(
        binary_dirs=max(0, int(probe_binary_dirs_budget)),
        raw_dirs=max(0, int(preview_raw_dirs_budget)),
    )
    lines = [root.name + "/"]
    if walk.dirs_exceeded or walk.files_exceeded:
        note = (
            f"# ... scan budget: dirs_visited={walk.dirs_seen:,} "
            f"files_counted={walk.total_files:,}"
        )
        if walk.dirs_exceeded:
            note += " [walk_dirs_cap]"
        if walk.files_exceeded:
            note += " [walk_files_cap]"
        lines.append(note)

    root_capped = "." in walk.capped_dirs
    for index, (name, kind, info) in enumerate(items):
        prefix = "└── " if index == len(items) - 1 else "├── "
        lines.append(
            prefix
            + _item_label(
                root,
                name,
                kind,
                info,
                meta=meta,
                dir_stats=dir_stats,
                dominant_extensions=dominant,
                root_capped=root_capped,
            )
        )
        if kind == "file":
            _append_file_meta(name, meta, lines)
        if kind == "dir" and name not in expand_subdirs and not dir_stats.get(
            name,
            {},
        ).get("capped"):
            _append_unexpanded_meta(name, meta, lines)
        if kind == "dir" and name not in expand_subdirs:
            _append_directory_previews(
                root,
                name,
                info,
                dir_stats=dir_stats,
                preview_raw_files=preview_raw_files,
                probe_binary_files=probe_binary_files,
                budget=budget,
                lines=lines,
            )
        if kind == "dir_group":
            _append_group_probe(
                root,
                info,
                probe_binary_files=probe_binary_files,
                budget=budget,
                lines=lines,
            )
        if kind == "dir" and name in expand_subdirs:
            _append_expanded_subtree(
                root,
                name,
                meta,
                include_val_outputs=include_val_outputs,
                preview_raw_files=preview_raw_files,
                probe_binary_files=probe_binary_files,
                lines=lines,
            )
    return "\n".join(lines), budget


def project_scan_result(
    root: Path,
    meta: dict[str, Any],
    walk: DatasetWalk,
    dir_structure: str,
    preview_budget: PreviewBudget,
) -> dict[str, Any]:
    """Project the stable public scan schema from typed owner outputs."""

    return {
        "path": str(root),
        "dir_structure": dir_structure,
        "meta": {
            key: (
                {"columns": value["columns"]}
                if "columns" in value
                else {
                    "type": value.get("type"),
                    "keys": value.get("keys", [])[: JSON_KEY_SHOW_LIMIT + 3],
                }
            )
            for key, value in meta.items()
        },
        "stats": {
            "total_files": walk.total_files,
            "by_extension": dict(walk.ext_counter.most_common(15)),
        },
        "scan_budget": {
            "walk_dirs_exceeded": walk.dirs_exceeded,
            "walk_files_exceeded": walk.files_exceeded,
            "dirs_visited": walk.dirs_seen,
            "files_counted": walk.total_files,
            "probe_binary_remaining": preview_budget.binary_dirs,
            "preview_raw_remaining": preview_budget.raw_dirs,
        },
    }


__all__ = [
    "PreviewBudget",
    "build_top_level_index",
    "project_scan_result",
    "render_dataset_tree",
]
