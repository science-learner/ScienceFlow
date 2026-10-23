# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Nested dataset-directory facts, grouping, samples, and rendering."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scienceflow.research.state.dataset.discovery.scan_metadata import (
    EXPAND_LIST_MAX_ITEMS,
    FLAT_DIR_META_EXTS,
    RAW_PREVIEW_EXTS,
    RAW_PREVIEW_MAX_LINES,
    SCAN_EXCLUDE_DIRS,
    SUBDIRS_SHOW_LIMIT,
    _append_binary_probe_samples,
    _binary_hint,
    _ext_list_with_hints,
    _format_meta_sample_lines,
    _format_raw_preview_lines,
    _hide_from_preview,
    _meta_file_comment,
    _read_raw_file_preview,
)


@dataclass(slots=True)
class DirectoryFact:
    name: str
    file_count: int
    extension: str
    children: list[str]
    extensions: Counter[str]


@dataclass(slots=True)
class FileFact:
    name: str
    extension: str
    meta: dict[str, Any] | None


@dataclass(slots=True)
class RenderItem:
    name: str
    kind: str
    value: DirectoryFact | FileFact | None


def _try_resolve_isdir(path: Path) -> bool:
    try:
        return path.resolve().is_dir()
    except OSError:
        return False


def _is_directory(path: Path) -> bool:
    return path.is_dir() or (path.is_symlink() and _try_resolve_isdir(path))


def _walk_count_files_and_exts(child_path: Path) -> tuple[int, Counter[str]]:
    """Count recursive files and extensions in a single pass."""

    file_count = 0
    extensions: Counter[str] = Counter()
    try:
        for _dirpath, _dirnames, filenames in os.walk(
            str(child_path),
            topdown=True,
            followlinks=True,
        ):
            for filename in filenames:
                file_count += 1
                suffix = Path(filename).suffix.lstrip(".").lower() or "no_ext"
                extensions[suffix] += 1
    except OSError:
        pass
    return file_count, extensions


def _append_raw_file(path: Path, display_name: str, lines: list[str]) -> None:
    preview = _read_raw_file_preview(path)
    if not preview:
        return
    lines.append(
        f"│   └── [sample] {display_name} (first {RAW_PREVIEW_MAX_LINES} lines)"
    )
    lines.extend(
        f"│   {sample}"
        for sample in _format_raw_preview_lines(preview, indent="")
    )


def _append_raw_preview_for_dir(
    dir_path: Path,
    exts_counter: Counter[str],
    lines_out: list[str],
    nested: bool = False,
) -> None:
    """Append one direct raw preview per matching extension."""

    _ = nested
    extensions = {
        extension
        for extension in exts_counter
        if f".{extension}" in RAW_PREVIEW_EXTS
    }
    for extension in sorted(extensions):
        try:
            sample = next(dir_path.glob(f"*.{extension}"), None)
            if sample is not None and sample.is_file():
                _append_raw_file(sample, sample.name, lines_out)
        except OSError:
            pass


def _append_raw_preview_for_subdir_with_children(
    dir_path: Path,
    sub_sample: list[str],
    lines_out: list[str],
) -> None:
    """Preview the first raw file under the first child directory."""

    if not sub_sample:
        return
    child_name = sub_sample[0].rstrip("/")
    child_path = dir_path / child_name
    if not child_path.is_dir():
        return
    try:
        sample = next(
            (
                path
                for path in sorted(child_path.iterdir())
                if path.is_file() and path.suffix.lower() in RAW_PREVIEW_EXTS
            ),
            None,
        )
        if sample is not None:
            _append_raw_file(sample, f"{child_name}/{sample.name}", lines_out)
    except OSError:
        pass


def _first_nested_raw_file(dir_path: Path) -> tuple[Path, str] | None:
    for entry in sorted(dir_path.iterdir()):
        if entry.is_dir():
            sample = next(
                (
                    path
                    for path in sorted(entry.iterdir())
                    if path.is_file() and path.suffix.lower() in RAW_PREVIEW_EXTS
                ),
                None,
            )
            if sample is not None:
                return sample, f"{entry.name}/{sample.name}"
        elif entry.is_file() and entry.suffix.lower() in RAW_PREVIEW_EXTS:
            return entry, entry.name
    return None


def _append_raw_preview_for_capped_or_nested(
    dir_path: Path,
    lines_out: list[str],
) -> None:
    """Preview the first raw file from a capped or nested directory."""

    try:
        found = _first_nested_raw_file(dir_path)
        if found is not None:
            _append_raw_file(found[0], found[1], lines_out)
    except OSError:
        pass


def _collect_entries(
    subdir_path: Path,
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    *,
    include_val_outputs: bool,
) -> tuple[list[DirectoryFact], list[FileFact]]:
    directories: list[DirectoryFact] = []
    files: list[FileFact] = []
    for path in sorted(subdir_path.iterdir()):
        name = path.name
        if name in SCAN_EXCLUDE_DIRS or _hide_from_preview(
            name,
            include_val_outputs,
        ):
            continue
        if _is_directory(path):
            file_count, extensions = _walk_count_files_and_exts(path)
            dominant = (
                "." + extensions.most_common(1)[0][0] if extensions else ""
            )
            try:
                children = sorted(
                    f"{child.name}/"
                    for child in path.iterdir()
                    if _is_directory(child) and child.name not in SCAN_EXCLUDE_DIRS
                )
            except OSError:
                children = []
            directories.append(
                DirectoryFact(name, file_count, dominant, children, extensions)
            )
            continue

        extension = (path.suffix.lower() or "no_ext").lstrip(".")
        meta_key = f"{meta_prefix}/{name}"
        value = meta.get(meta_key)
        if value is None and path.suffix in meta_readers:
            value = meta_readers[path.suffix](path)
            if value is not None:
                meta[meta_key] = value
        files.append(FileFact(name, extension, value))
    return directories, files


def _build_items(
    directories: list[DirectoryFact],
    files: list[FileFact],
) -> list[RenderItem]:
    metadata_extensions = {
        extension.lstrip(".").lower() for extension in FLAT_DIR_META_EXTS
    }
    metadata_files = sorted(
        (item for item in files if item.extension.lower() in metadata_extensions),
        key=lambda item: item.name,
    )
    other_files = sorted(
        (item for item in files if item.extension.lower() not in metadata_extensions),
        key=lambda item: item.name,
    )
    items = [RenderItem(item.name, "file", item) for item in metadata_files]

    signature_groups: dict[tuple[int, tuple[tuple[str, int], ...]], list[DirectoryFact]] = {}
    for entry in sorted(directories, key=lambda item: item.name):
        signature = (entry.file_count, tuple(sorted(entry.extensions.items())))
        signature_groups.setdefault(signature, []).append(entry)
    for group in signature_groups.values():
        first = group[0]
        items.append(RenderItem(first.name, "dir", first))
        if len(group) > 1:
            items.append(
                RenderItem(
                    f"... ({len(group) - 1} more similar dirs, "
                    f"{first.name} ~ {group[-1].name})",
                    "collapsed",
                    None,
                )
            )
    items.extend(RenderItem(item.name, "file", item) for item in other_files)
    return items


def _identifier_column(
    meta: dict[str, Any],
    meta_prefix: str,
    name: str,
) -> str:
    csv_meta = meta.get(f"{meta_prefix}/{name}.csv")
    if not csv_meta:
        return "{id}"
    for column in (
        "image_name",
        "image_id",
        "StudyInstanceUID",
        "Id",
        "id",
        "fname",
    ):
        if column in csv_meta.get("columns", []):
            return f"{{{column}}}"
    return "{id}"


def _directory_comment(
    fact: DirectoryFact,
    meta: dict[str, Any],
    meta_prefix: str,
) -> str:
    if len(fact.extensions) > 1:
        comment = (
            f" # ({fact.file_count:,} files: "
            f"{_ext_list_with_hints(fact.extensions)})"
        )
    else:
        identifier = _identifier_column(meta, meta_prefix, fact.name)
        suffix = f"{identifier}{fact.extension}" if fact.extension else ""
        comment = (
            f" # {suffix}({fact.file_count:,})"
            if suffix
            else f" # {fact.file_count:,} files"
        )
        hint = _binary_hint(fact.extension.lstrip(".")) if fact.extension else ""
        if hint:
            comment += f"  [{hint}]"
    if fact.children:
        comment += " subdirs: " + ", ".join(fact.children[:SUBDIRS_SHOW_LIMIT])
        if len(fact.children) > SUBDIRS_SHOW_LIMIT:
            comment += f" ... ({len(fact.children)} total)"
    return comment


def _append_leaf_samples(
    child_path: Path,
    fact: DirectoryFact,
    meta_readers: dict[str, Any],
    *,
    preview_raw_files: bool,
    probe_binary_files: bool,
    lines: list[str],
) -> None:
    for extension in ("json", "csv"):
        if extension not in fact.extensions or f".{extension}" not in meta_readers:
            continue
        try:
            sample = next(child_path.glob(f"*.{extension}"), None)
            reader = meta_readers.get(f".{extension}")
            value = reader(sample) if reader and sample is not None else None
            if value and (
                "columns" in value
                or value.get("sample_items") is not None
                or value.get("type")
            ):
                lines.append("│   └── [sample] " + sample.name)
                lines.extend(
                    f"│   {row}"
                    for row in _format_meta_sample_lines(value, indent="")
                )
        except OSError:
            pass
    if preview_raw_files:
        _append_raw_preview_for_dir(child_path, fact.extensions, lines, nested=False)
    if probe_binary_files:
        _append_binary_probe_samples(
            child_path,
            fact.extensions,
            lines,
            probe_binary_files=True,
        )


def _append_nested_samples(
    child_path: Path,
    fact: DirectoryFact,
    *,
    preview_raw_files: bool,
    probe_binary_files: bool,
    lines: list[str],
) -> None:
    if preview_raw_files:
        _append_raw_preview_for_subdir_with_children(
            child_path,
            fact.children,
            lines,
        )
    if not probe_binary_files or not fact.children:
        return
    first_child = child_path / fact.children[0].rstrip("/")
    if not first_child.is_dir():
        return
    extensions: Counter[str] = Counter()
    try:
        for path in first_child.iterdir():
            if path.is_file():
                extensions[(path.suffix.lower() or "no_ext").lstrip(".")] += 1
    except OSError:
        extensions = Counter()
    _append_binary_probe_samples(
        first_child,
        extensions,
        lines,
        probe_binary_files=True,
    )


def _append_directory_item(
    prefix: str,
    subdir_path: Path,
    fact: DirectoryFact,
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    *,
    preview_raw_files: bool,
    probe_binary_files: bool,
    lines: list[str],
) -> None:
    lines.append(prefix + fact.name + "/" + _directory_comment(fact, meta, meta_prefix))
    child_path = subdir_path / fact.name
    if not fact.children and child_path.is_dir():
        _append_leaf_samples(
            child_path,
            fact,
            meta_readers,
            preview_raw_files=preview_raw_files,
            probe_binary_files=probe_binary_files,
            lines=lines,
        )
    elif fact.children:
        _append_nested_samples(
            child_path,
            fact,
            preview_raw_files=preview_raw_files,
            probe_binary_files=probe_binary_files,
            lines=lines,
        )


def render_nested_directory(
    subdir_path: Path,
    meta: dict[str, Any],
    meta_prefix: str,
    meta_readers: dict[str, Any],
    *,
    include_val_outputs: bool,
    preview_raw_files: bool,
    probe_binary_files: bool,
) -> list[str]:
    """Render grouped nested facts with bounded item expansion."""

    directories, files = _collect_entries(
        subdir_path,
        meta,
        meta_prefix,
        meta_readers,
        include_val_outputs=include_val_outputs,
    )
    all_items = _build_items(directories, files)
    items = all_items[:EXPAND_LIST_MAX_ITEMS]
    lines: list[str] = []
    for index, item in enumerate(items):
        is_last = index == len(items) - 1 and len(all_items) <= EXPAND_LIST_MAX_ITEMS
        prefix = "└── " if is_last else "├── "
        if item.kind == "collapsed":
            lines.append(prefix + item.name)
        elif item.kind == "dir":
            assert isinstance(item.value, DirectoryFact)
            _append_directory_item(
                prefix,
                subdir_path,
                item.value,
                meta,
                meta_prefix,
                meta_readers,
                preview_raw_files=preview_raw_files,
                probe_binary_files=probe_binary_files,
                lines=lines,
            )
        else:
            assert isinstance(item.value, FileFact)
            value = item.value.meta or {}
            lines.append(prefix + item.name + _meta_file_comment(value))
            lines.extend(
                f"│   {row}" for row in _format_meta_sample_lines(value, indent="")
            )
    if len(all_items) > EXPAND_LIST_MAX_ITEMS:
        lines.append(
            f"└── … ({len(all_items):,} items total, "
            f"showing first {EXPAND_LIST_MAX_ITEMS})"
        )
    return lines


__all__ = [
    "_append_raw_preview_for_capped_or_nested",
    "_append_raw_preview_for_dir",
    "_append_raw_preview_for_subdir_with_children",
    "_try_resolve_isdir",
    "_walk_count_files_and_exts",
    "render_nested_directory",
]
