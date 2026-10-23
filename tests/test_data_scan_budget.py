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

"""Budget / fast-path behavior for :func:`scienceflow.research.state.dataset.discovery.scan.scan_data_dir`."""

from __future__ import annotations

import json
from pathlib import Path

from scienceflow.research.state.dataset.discovery.scan import scan_data_dir, _walk_count_files_and_exts
from scienceflow.research.state.dataset.discovery.scan_metadata import FLAT_DIR_FILE_LIST_THRESHOLD


def test_walk_count_files_and_exts_single_pass(tmp_path: Path) -> None:
    d = tmp_path / "sub"
    d.mkdir()
    (d / "a.txt").write_text("x")
    (d / "b.txt").write_text("y")
    (d / "nest").mkdir()
    (d / "nest" / "c.bin").write_bytes(b"\x00")
    fc, exts = _walk_count_files_and_exts(d)
    assert fc == 3
    assert exts.get("txt") == 2
    assert exts.get("bin") == 1


def test_scan_data_dir_walk_budget_emits_flag(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    for i in range(5):
        (root / f"d{i}").mkdir()
        (root / f"d{i}" / "f.txt").write_text("ok")
    r = scan_data_dir(
        root,
        walk_budget_dirs=3,
        walk_budget_files=10_000,
        probe_binary_dirs_budget=0,
        preview_raw_dirs_budget=0,
    )
    assert r.get("scan_budget", {}).get("walk_dirs_exceeded") is True
    ds = r.get("dir_structure", "")
    assert "walk_dirs_cap" in ds or "[walk_dirs_cap]" in ds


def test_scan_data_dir_returns_scan_budget_keys(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    r = scan_data_dir(tmp_path)
    sb = r.get("scan_budget")
    assert isinstance(sb, dict)
    assert "files_counted" in sb
    assert "probe_binary_remaining" in sb


def test_scan_data_dir_preserves_metadata_stats_and_validation_visibility(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    images = root / "images"
    images.mkdir(parents=True)
    (root / "train.csv").write_text("image_id,target\na,1\n", encoding="utf-8")
    (root / "validation_holdout.csv").write_text(
        "image_id,target\nb,0\n",
        encoding="utf-8",
    )
    (images / "a.jpg").write_bytes(b"jpg")
    (images / "b.png").write_bytes(b"png")

    hidden = scan_data_dir(root)
    visible = scan_data_dir(root, include_val_outputs=True)

    assert hidden["stats"] == {
        "total_files": 4,
        "by_extension": {"csv": 2, "jpg": 1, "png": 1},
    }
    assert hidden["meta"]["train.csv"] == {"columns": ["image_id", "target"]}
    assert "validation_holdout.csv" not in hidden["dir_structure"]
    assert "validation_holdout.csv" in visible["dir_structure"]
    assert "images/" in hidden["dir_structure"]
    assert ".jpg" in hidden["dir_structure"]
    assert ".png" in hidden["dir_structure"]


def test_scan_data_dir_groups_repeated_top_level_directories(tmp_path: Path) -> None:
    root = tmp_path / "grouped"
    root.mkdir()
    for index in range(20):
        sample = root / f"Sample{index:02d}"
        sample.mkdir()
        (sample / "record.mat").write_bytes(b"mat")

    report = scan_data_dir(root)

    assert report["stats"]["total_files"] == 20
    assert report["stats"]["by_extension"] == {"mat": 20}
    assert "Sample*/" in report["dir_structure"]
    assert "20 subdirs" in report["dir_structure"]


def test_scan_data_dir_groups_root_payload_files_without_label_state_leak(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root_files"
    root.mkdir()
    for index in range(16):
        (root / f"payload_{index:02d}.txt").write_text("payload", encoding="utf-8")

    report = scan_data_dir(root)

    assert "*.txt(16)" in report["dir_structure"]
    assert report["stats"]["by_extension"] == {"txt": 16}


def test_scan_data_dir_expands_large_flat_directory_without_itemizing_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "flat"
    payload = root / "audio"
    payload.mkdir(parents=True)
    (payload / "labels.csv").write_text("id,label\na,1\n", encoding="utf-8")
    for index in range(FLAT_DIR_FILE_LIST_THRESHOLD + 1):
        (payload / f"clip_{index:03d}.wav").write_bytes(b"wav")

    report = scan_data_dir(root, expand_subdirs=["audio"])
    structure = report["dir_structure"]

    assert "labels.csv" in structure
    assert "*.wav" in structure
    assert "clip_000.wav" not in structure
    assert report["meta"]["audio/labels.csv"] == {"columns": ["id", "label"]}


def test_scan_data_dir_collapses_identical_nested_directory_signatures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nested"
    train = root / "train"
    for name in ("class_a", "class_b", "class_c"):
        child = train / name
        child.mkdir(parents=True)
        (child / "image.jpg").write_bytes(b"jpg")

    structure = scan_data_dir(root, expand_subdirs=["train"])["dir_structure"]

    assert "class_a/" in structure
    assert "2 more similar dirs, class_a ~ class_c" in structure
    assert "\n│   ├── class_b/" not in structure
    assert "\n│   └── class_b/" not in structure


def test_scan_data_dir_renders_expanded_payload_file_once(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "notes.txt").write_text("one", encoding="utf-8")

    structure = scan_data_dir(tmp_path, expand_subdirs=["payload"])[
        "dir_structure"
    ]

    assert structure.count("notes.txt") == 1


def test_scan_data_dir_renders_nested_json_sample_values(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "sample-a",
                    "tags": ["train", "verified"],
                    "metrics": {"auc": 0.91},
                }
            ]
        ),
        encoding="utf-8",
    )

    report = scan_data_dir(tmp_path)
    structure = report["dir_structure"]

    assert "id: sample-a" in structure
    assert 'tags: ["train", …] (2 items)' in structure
    assert 'metrics: {"auc": 0.91}' in structure


def test_scan_data_dir_renders_json_object_shape_and_counts(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "records": [{"id": 1}, {"id": 2}],
                "schema": {"target": "float"},
                "name": "demo",
            }
        ),
        encoding="utf-8",
    )

    structure = scan_data_dir(tmp_path)["dir_structure"]

    assert 'records: [{"id": 1}, …] (2 items)' in structure
    assert "schema: {…1 keys}" in structure
    assert "name: demo" in structure


def test_scan_data_dir_preserves_bounded_json_recovery(tmp_path: Path) -> None:
    path = tmp_path / "recovered.json"
    path.write_text(
        'non-json prefix\n{"folds": [1, 2, 3]}\nnon-json suffix',
        encoding="utf-8",
    )

    report = scan_data_dir(tmp_path, meta_sample_max_bytes=1_000)

    assert report["meta"]["recovered.json"] == {
        "type": "object",
        "keys": ["folds"],
    }
    assert "folds: [1, …] (3 items)" in report["dir_structure"]
