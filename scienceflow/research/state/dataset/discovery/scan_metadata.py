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

"""
Data directory scan (e.g. MLEBench prepared/public).

Used by LNR workspace preparation and dataset preview helpers.
related tooling. Unified flow: traverse, read meta by extension, optional binary
probes, output fixed schema. No task-specific logic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter
from typing import Any, Callable

# Meta file ext -> reader (unified: Path -> dict|str|None)
META_READERS: dict[str, Any] = {}

# Per-call overrides (set by :func:`scan_data_dir` for balanced budgets).
_META_JSON_MAX_BYTES: int = 50_000
_CSV_MAX_ROWS_TO_COUNT: int = 200_000


class _ScanMetaCsvScope:
    """Temporarily override JSON/CSV scan caps for :func:`scan_data_dir`."""

    __slots__ = ("_meta", "_csv", "_old")

    def __init__(self, meta_sample_max_bytes: int, csv_max_rows_to_scan: int) -> None:
        self._meta = int(meta_sample_max_bytes)
        self._csv = int(csv_max_rows_to_scan)
        self._old: tuple[int, int] = (50_000, 200_000)

    def __enter__(self) -> None:
        global _META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT
        self._old = (_META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT)
        _META_JSON_MAX_BYTES = self._meta
        _CSV_MAX_ROWS_TO_COUNT = self._csv

    def __exit__(self, *exc: object) -> None:
        global _META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT
        _META_JSON_MAX_BYTES, _CSV_MAX_ROWS_TO_COUNT = self._old
        return None

# When a single directory holds more files than this, stop listing them one by one and show only a summary (e.g. *.wav (28,150))
FLAT_DIR_FILE_LIST_THRESHOLD = 500
# When a metadata file has at most this many lines/entries, append the first few sample records to dir_structure to help the LLM understand the actual data format
META_SAMPLE_COUNT = 1
# Maximum number of child items (subdirectories + files) listed when expanding a subdirectory; beyond that, show only the first N items with an omission note
EXPAND_LIST_MAX_ITEMS = 20
# Extensions treated as "metadata" in large flat directories; listed individually (with file names) so prep can see csv/json/md etc.
FLAT_DIR_META_EXTS = (".csv", ".json", ".md", ".yml", ".yaml", ".txt")
# Directory names always excluded during scanning (never walked, never shown)
SCAN_EXCLUDE_DIRS = frozenset({"__pycache__"})
# Upper limit for displayed JSON array/object keys
JSON_KEY_SHOW_LIMIT = 12
# subdirs under a directory: maximum number of subdirectories shown; beyond that, show the first N plus "... (M total)"
SUBDIRS_SHOW_LIMIT = 5
# preview_raw_files: read the first few lines of common non-CSV/JSON data files (.xyz, .txt, .dat, etc.) in subdirectories as a preview
RAW_PREVIEW_EXTS = frozenset({".xyz", ".txt", ".dat", ".sdf", ".mol", ".pdb", ".cif", ".mol2"})
RAW_PREVIEW_MAX_LINES = 10
RAW_PREVIEW_MAX_LINE_LEN = 120
# Read hints for binary/media files, helping the LLM know which library to open them with
BINARY_FILE_HINTS: dict[str, str] = {
    ".dicom": "DICOM medical image; read with pydicom",
    ".dcm":   "DICOM medical image; read with pydicom",
    ".tiff":  "large TIFF image; read with tifffile or openslide",
    ".tif":   "TIFF image; read with tifffile or PIL",
    ".jpg":   "JPEG image; read with PIL or cv2",
    ".jpeg":  "JPEG image; read with PIL or cv2",
    ".png":   "PNG image; read with PIL or cv2",
    ".webp":  "WebP image; read with PIL",
    ".bmp":   "BMP image; read with PIL or cv2",
    ".gif":   "GIF image; read with PIL",
    ".mat":   "Matlab data; read with scipy.io.loadmat",
    ".bin":   "binary data; read with numpy.fromfile or struct",
    ".wav":   "audio waveform; read with scipy.io.wavfile or librosa",
    ".mp3":   "audio; read with librosa",
    ".mp4":   "video; read with cv2.VideoCapture or decord",
    ".avi":   "video; read with cv2.VideoCapture",
    ".npy":   "numpy array; read with numpy.load",
    ".npz":   "numpy archive; read with numpy.load",
    ".h5":    "HDF5; read with h5py",
    ".hdf5":  "HDF5; read with h5py",
    ".parquet": "columnar data; read with pandas.read_parquet",
    ".feather": "columnar data; read with pandas.read_feather",
    ".tfrecord": "TensorFlow record; read with tf.data.TFRecordDataset",
    ".tfrec":   "TensorFlow record; read with tf.data.TFRecordDataset",
    ".nii":   "NIfTI neuroimaging; read with nibabel",
    ".nii.gz": "NIfTI neuroimaging; read with nibabel",
}
# Runtime artifact file extensions; not raw data, hidden from preview
RUNTIME_ARTIFACT_EXTS = frozenset({".pth", ".pt", ".ckpt", ".pkl", ".pickle"})
# Single-line output cap for binary probing (to avoid blowing up the prompt)
BINARY_PROBE_MAX_LINE = 900


def _probe_mat(path: Path) -> str | None:
    try:
        import scipy.io

        d = scipy.io.loadmat(str(path))
        items: list[str] = []
        for k, v in sorted(d.items()):
            if k.startswith("__"):
                continue
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                items.append(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                items.append(f"{k}: type={type(v).__name__}")
        return "; ".join(items) if items else None
    except Exception:
        return None


def _probe_npy(path: Path) -> str | None:
    try:
        import numpy as np

        arr = np.load(str(path), allow_pickle=False)
        return f"shape={tuple(arr.shape)}, dtype={arr.dtype}"
    except Exception:
        return None


def _probe_npz(path: Path) -> str | None:
    try:
        import numpy as np

        with np.load(str(path), allow_pickle=False) as data:
            parts = [
                f"{k}: shape={tuple(data[k].shape)}, dtype={data[k].dtype}"
                for k in sorted(data.files)[:24]
            ]
        return "; ".join(parts) if parts else None
    except Exception:
        return None


def _probe_h5(path: Path) -> str | None:
    try:
        import h5py

        items: list[str] = []
        with h5py.File(str(path), "r") as f:

            def visit(name: str, obj: Any) -> None:
                if len(items) >= 24:
                    return
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    items.append(f"{name}: shape={tuple(obj.shape)}, dtype={obj.dtype}")

            f.visititems(visit)
        return "; ".join(sorted(items)) if items else None
    except Exception:
        return None


def _probe_parquet(path: Path) -> str | None:
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        cols = ", ".join(f"{c}({df[c].dtype})" for c in df.columns[:20])
        more = f", ... ({len(df.columns)} cols)" if len(df.columns) > 20 else ""
        return f"{len(df):,} rows; {cols}{more}"
    except Exception:
        return None


def _probe_image(path: Path) -> str | None:
    """Best-effort image dimension probe using PIL (header only, no pixel decode)."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
        return f"size={w}x{h}, mode={mode}"
    except Exception:
        return None


# ext key matches Counter keys (suffix without dot, lowercased)
BINARY_PROBERS: dict[str, Callable[[Path], str | None]] = {
    "mat": _probe_mat,
    "npy": _probe_npy,
    "npz": _probe_npz,
    "h5": _probe_h5,
    "hdf5": _probe_h5,
    "parquet": _probe_parquet,
    "jpg": _probe_image,
    "jpeg": _probe_image,
    "png": _probe_image,
    "webp": _probe_image,
    "bmp": _probe_image,
    "gif": _probe_image,
    "tif": _probe_image,
    "tiff": _probe_image,
}


def _append_binary_probe_samples(
    dir_path: Path,
    exts_counter: Counter,
    lines_out: list[str],
    *,
    probe_binary_files: bool,
) -> None:
    """Append one probed line per supported binary ext present in *dir_path*."""
    if not probe_binary_files:
        return
    for ext_no_dot in sorted(k for k in BINARY_PROBERS if exts_counter.get(k, 0) > 0):
        suf = "." + ext_no_dot
        try:
            one = next(dir_path.glob(f"*{suf}"), None)
            if one is None or not one.is_file():
                continue
            fn = BINARY_PROBERS[ext_no_dot]
            txt = fn(one)
            if not txt:
                continue
            if len(txt) > BINARY_PROBE_MAX_LINE:
                txt = txt[:BINARY_PROBE_MAX_LINE] + "..."
            lines_out.append(f"│   └── [binary_probe] {one.name}")
            lines_out.append(f"│     | {txt}")
        except OSError:
            pass


def _ext_counter_from_dir_files(dir_path: Path) -> Counter:
    """Non-recursive extension counts under *dir_path* (one sample dir).

    When the parent data root hits WALK_MAX_DIRS and os.walk does not recurse
    into Sample*/ children, aggregated ``dir_stats`` ext counters are empty;
    probing still needs to see .mat/.npy/etc. from a real subdirectory.
    """
    c: Counter = Counter()
    try:
        for p in dir_path.iterdir():
            if p.is_file():
                c[(p.suffix.lstrip(".") or "no_ext").lower()] += 1
    except OSError:
        pass
    return c


def _binary_hint(ext_no_dot: str) -> str:
    """Return a read-hint string for binary/media extensions, or '' if none."""
    key = "." + ext_no_dot.lower()
    return BINARY_FILE_HINTS.get(key, "")


def _ext_comment_with_hint(ext_no_dot: str, count: int) -> str:
    """Build '*.ext # {id}.ext(N)' with optional read-hint appended."""
    ext_dot = "." + ext_no_dot
    hint = _binary_hint(ext_no_dot)
    base = f"*{ext_dot} # {{id}}{ext_dot}({count:,})"
    if hint:
        base += f"  [{hint}]"
    return base


def _ext_list_with_hints(exts_counter: Counter, top_n: int = 5) -> str:
    """Build comma-separated ext list with binary hints for dir comments.
    E.g. '.mp4, .mat [read with scipy.io.loadmat], .wav [read with librosa]'
    Skips runtime artifact extensions (.pth, .pt, etc.)."""
    parts: list[str] = []
    for e, _ in exts_counter.most_common(top_n + len(RUNTIME_ARTIFACT_EXTS)):
        if not e or e == "no_ext":
            continue
        if ("." + e) in RUNTIME_ARTIFACT_EXTS:
            continue
        hint = _binary_hint(e)
        if hint:
            parts.append(f".{e} [{hint}]")
        else:
            parts.append(f".{e}")
        if len(parts) >= top_n:
            break
    return ", ".join(parts) or "mixed"


def _is_numeric_str(s: str) -> bool:
    """True if s looks numeric, empty, or null-like (NaN/NA/None)."""
    s = s.strip()
    if not s or s.lower() in ("nan", "na", "null", "none", "inf", "-inf"):
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def _json_type_comment(m: dict) -> str:
    """Concise type-aware comment for JSON metadata."""
    typ = m.get("type", "")
    if typ == "array":
        return _json_array_comment(m)
    if typ == "object":
        return _json_object_comment(m)
    if typ == "truncated":
        return " # (truncated, could not parse)"
    return ""


def _json_array_comment(m: dict) -> str:
    count = m.get("count", 0)
    keys = m.get("keys", [])
    if keys:
        shown = ", ".join(keys[:8])
        if len(keys) > 8:
            shown += ", ..."
        return f" # [array of {count:,} objects, each: {{{shown}}}]"
    sample = m.get("sample_items", [])
    if sample:
        return f" # [array of {count:,} {type(sample[0]).__name__}s]"
    return f" # [array of {count:,} items]"


def _json_object_comment(m: dict) -> str:
    keys = m.get("keys", [])
    sample = m.get("sample_items", {})
    key_count = m.get("key_count", len(keys))
    if not keys or not sample:
        return f" # (dict, {key_count} keys)"

    value_counts = m.get("value_counts", {})
    parts = [
        _json_object_key_comment(key, sample.get(key), value_counts.get(key))
        for key in keys[:JSON_KEY_SHOW_LIMIT]
    ]
    tail = f", ... ({key_count} total)" if key_count > JSON_KEY_SHOW_LIMIT else ""
    return f" # {{dict: {', '.join(parts)}{tail}}}"


def _json_object_key_comment(key: str, value: Any, count: int | None) -> str:
    if isinstance(value, list):
        return f"{key}(list[{count or len(value):,}])"
    if isinstance(value, dict):
        return f"{key}(dict[{count or len(value):,}])"
    return f"{key}({type(value).__name__})"


def _meta_file_comment(m: dict) -> str:
    """One-line comment for any metadata file (CSV or JSON)."""
    if "columns" in m:
        cols = m["columns"]
        row_count = m.get("row_count", 0)
        return f" # ({len(cols)} cols, {row_count:,} rows)"
    return _json_type_comment(m)


def _hide_from_preview(name: str, include_val_outputs: bool = False) -> bool:
    """Return True if this file/dir should be hidden from data_preview.
    When include_val_outputs is False, hide basenames starting with ``validation`` (aligned with
    dataset preview helpers). ``val_metadata*`` is always hidden."""
    stem = Path(name).stem.lower()
    ext = Path(name).suffix.lower()
    base = Path(name).name.lower()
    if stem == "description":
        return True
    # Runtime artifacts (model weights/checkpoints) are not raw data
    if ext in RUNTIME_ARTIFACT_EXTS:
        return True
    # submission.csv is the reference answer; only sample_submission.csv is kept
    if stem == "submission" and ext == ".csv":
        return True
    # Always hide val_metadata*.json so the agent cannot see validation metadata
    if stem == "val_metadata" or stem.startswith("val_metadata_"):
        return True
    # Hide intermediate/temporary and merged artifacts: _subset, tmp, _combined
    name_lower = name.lower()
    if "_subset" in name_lower or "_combined" in name_lower:
        return True
    if "tmp" in name_lower:
        return True
    if include_val_outputs:
        return False
    if base.startswith("validation"):
        return True
    return False


def _register(ext: str):
    def _inner(fn):
        META_READERS[ext] = fn
        return fn

    return _inner


@_register(".csv")
def _read_csv(path: Path, max_rows: int = 3, max_rows_to_count: int | None = None) -> dict[str, Any] | None:
    try:
        old_limit = csv.field_size_limit()
        cap = max_rows_to_count if max_rows_to_count is not None else _CSV_MAX_ROWS_TO_COUNT
        try:
            csv.field_size_limit(100 * 1024 * 1024)  # 100MB for large fields
            with open(path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                rows, row_count = [], 0
                for i, row in enumerate(reader):
                    row_count = i + 1
                    if i < max_rows:
                        rows.append(dict(row))
                    if row_count >= cap:
                        break
                # detect columns whose sample values are non-numeric strings
                str_value_cols: dict[str, str] = {}
                for col in columns:
                    for row in rows:
                        val = (row.get(col) or "").strip()
                        if val and not _is_numeric_str(val):
                            str_value_cols[col] = val
                            break
                result: dict[str, Any] = {
                    "columns": columns,
                    "sample_rows": rows,
                    "row_count": row_count,
                }
                if row_count >= cap:
                    result["row_count_truncated"] = True
                if str_value_cols:
                    result["string_value_cols"] = str_value_cols
                return result
        finally:
            csv.field_size_limit(old_limit)
    except Exception:
        return None


@_register(".json")
def _read_json(path: Path, max_bytes: int | None = None) -> dict[str, Any] | None:
    eff = max_bytes if max_bytes is not None else _META_JSON_MAX_BYTES
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        data = _decode_json_content(content, eff)
        if data is _TRUNCATED_JSON:
            return {"type": "truncated"}
        return _json_metadata(data)
    except Exception:
        return None


_TRUNCATED_JSON = object()


def _decode_json_content(content: str, max_bytes: int) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        bounded = content[:max_bytes]
        object_start = bounded.find("{")
        array_start = bounded.find("[")
        starts = [position for position in (object_start, array_start) if position >= 0]
        if not starts:
            return _TRUNCATED_JSON
        start = min(starts)
        closing = "}" if start == object_start else "]"
        end = bounded.rfind(closing)
        if end <= start:
            return _TRUNCATED_JSON
        try:
            return json.loads(bounded[start : end + 1])
        except json.JSONDecodeError:
            return _TRUNCATED_JSON


def _json_metadata(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
        return {
            "type": "array",
            "count": len(data),
            "keys": keys,
            "sample_items": data[:META_SAMPLE_COUNT],
        }
    if isinstance(data, dict):
        sample, value_counts = _json_object_sample(data)
        keys = list(data.keys())
        return {
            "type": "object",
            "keys": keys,
            "key_count": len(keys),
            "sample_items": sample,
            "value_counts": value_counts,
        }
    return {"type": type(data).__name__}


def _json_object_sample(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    sample: dict[str, Any] = {}
    value_counts: dict[str, int] = {}
    for key, value in data.items():
        if isinstance(value, list):
            value_counts[key] = len(value)
            sample[key] = value[:1]
        elif isinstance(value, dict):
            value_counts[key] = len(value)
            sample[key] = value
        else:
            sample[key] = value
    return sample, value_counts


MAX_SAMPLE_LINE_LEN = 120
# Detection/box columns (PredictionString etc.) need more characters, otherwise the class name (e.g. car) is invisible in the preview and the column is easily misjudged as all-numeric
MAX_CELL_LEN_DEFAULT = 60
MAX_CELL_LEN_PREDICTION_STRING = 200
# Per-key value display length cap in JSON preview, so all keys stay fully visible and are not squeezed out by long values
JSON_KEY_VAL_MAX_LEN = 100


def _read_raw_file_preview(path: Path) -> str | None:
    """Read the first few lines of a raw data file for LLM preview."""
    try:
        lines: list[str] = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= RAW_PREVIEW_MAX_LINES:
                    break
                line = line.rstrip("\n\r")
                if len(line) > RAW_PREVIEW_MAX_LINE_LEN:
                    line = line[:RAW_PREVIEW_MAX_LINE_LEN] + "..."
                lines.append(line)
        if not lines:
            return None
        return "\n".join(lines)
    except Exception:
        return None


def _format_raw_preview_lines(preview_text: str, indent: str = "    ") -> list[str]:
    """Format raw file preview text into tree-display lines."""
    lines: list[str] = []
    for line in preview_text.splitlines():
        lines.append(f"{indent}  | {line}")
    return lines


def _format_meta_sample_lines(m: dict, indent: str = "    ") -> list[str]:
    """Format sample data from CSV or JSON meta for display in dir_structure tree.
    First line = header (column names or keys), next line(s) = sample data.
    """
    sample_rows = m.get("sample_rows", [])
    cols = m.get("columns", [])
    if sample_rows and cols:
        return _format_csv_sample_lines(sample_rows, cols, indent)

    sample_items = m.get("sample_items")
    if not sample_items:
        return []
    typ = m.get("type", "")
    if typ == "array" and isinstance(sample_items, list):
        return _format_json_array_sample_lines(sample_items, m.get("keys", []), indent)
    if typ == "object" and isinstance(sample_items, dict):
        return _format_json_object_sample_lines(
            sample_items, m.get("value_counts", {}), indent
        )
    return []


def _is_prediction_column(column: str) -> bool:
    lowered = column.lower()
    return "prediction" in lowered or lowered in (
        "predictionstring",
        "label",
        "encoding",
    )


def _format_csv_sample_lines(
    sample_rows: list[dict[str, Any]], cols: list[str], indent: str
) -> list[str]:
    lines = [f"{indent}  | {', '.join(cols)}"]
    has_long_column = any(_is_prediction_column(column) for column in cols)
    line_limit = MAX_SAMPLE_LINE_LEN
    if has_long_column:
        line_limit += MAX_CELL_LEN_PREDICTION_STRING - MAX_CELL_LEN_DEFAULT

    for row in sample_rows[:META_SAMPLE_COUNT]:
        values = [_format_csv_cell(row.get(column, ""), column) for column in cols]
        line = f"{indent}  | {', '.join(values)}"
        if len(line) > line_limit:
            line = line[:line_limit] + "..."
        lines.append(line)
    return lines


def _format_csv_cell(value: Any, column: str) -> str:
    max_length = (
        MAX_CELL_LEN_PREDICTION_STRING
        if _is_prediction_column(column)
        else MAX_CELL_LEN_DEFAULT
    )
    return str(value)[:max_length]


def _bounded_json(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) > JSON_KEY_VAL_MAX_LEN:
        return rendered[:JSON_KEY_VAL_MAX_LEN] + "…"
    return rendered


def _format_json_array_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "[]"
        first = _bounded_json(value[0])
        return f"[{first}, …] ({len(value)} items)" if len(value) > 1 else f"[{first}]"
    if isinstance(value, dict):
        return _bounded_json(value)
    rendered = str(value)
    if len(rendered) > JSON_KEY_VAL_MAX_LEN:
        return rendered[:JSON_KEY_VAL_MAX_LEN] + "…"
    return rendered


def _format_json_array_sample_lines(
    sample_items: list[Any], keys: list[str], indent: str
) -> list[str]:
    lines: list[str] = []
    for item in sample_items[:META_SAMPLE_COUNT]:
        if isinstance(item, dict):
            for key in keys or list(item.keys()):
                value = _format_json_array_value(item.get(key, ""))
                lines.append(f"{indent}  | {key}: {value}")
            continue
        line = f"{indent}  | {json.dumps(item, ensure_ascii=False)}"
        if len(line) > MAX_SAMPLE_LINE_LEN:
            line = line[:MAX_SAMPLE_LINE_LEN] + "..."
        lines.append(line)
    return lines


def _format_json_object_value(value: Any, count: int | None) -> str:
    if isinstance(value, list):
        item_count = count if count is not None else len(value)
        if value:
            return f"[{_bounded_json(value[0])}, …] ({item_count:,} items)"
        return "[] (0 items)"
    if isinstance(value, dict):
        item_count = count if count is not None else len(value)
        return f"{{…{item_count:,} keys}}"
    if isinstance(value, str):
        if len(value) > JSON_KEY_VAL_MAX_LEN:
            return value[:JSON_KEY_VAL_MAX_LEN] + "…"
        return value
    return _bounded_json(value)


def _format_json_object_sample_lines(
    sample_items: dict[str, Any], value_counts: dict[str, int], indent: str
) -> list[str]:
    return [
        f"{indent}  | {key}: {_format_json_object_value(value, value_counts.get(key))}"
        for key, value in sample_items.items()
    ]

