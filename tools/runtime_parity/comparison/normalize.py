from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RULES_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "runtime_parity"
    / "normalization.yaml"
)


def load_normalization_rules() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(_RULES_PATH.read_text(encoding="utf-8"))
    rows = document.get("rules") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Invalid normalization registry: {_RULES_PATH}")
    rules: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("field"):
            raise ValueError(f"Invalid normalization rule: {row!r}")
        field = str(row["field"])
        if field in rules:
            raise ValueError(f"Duplicate normalization rule: {field}")
        rules[field] = dict(row)
    return rules


NORMALIZATION_RULES = load_normalization_rules()
_VALUE_KEY_RULES = {
    field: str(rule["replacement"])
    for field, rule in NORMALIZATION_RULES.items()
    if field
    in {"duration_sec", "elapsed_sec", "pid", "pgid", "ttft_sec", "tpot_ms"}
}


def rule_registered(field: str) -> bool:
    return field in NORMALIZATION_RULES


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): json_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_value(child) for child in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): json_value(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }
    return repr(value)


def normalize_value(value: Any, *, workspace: Path | None = None) -> Any:
    """Normalize only registered runtime timestamps, ANSI, and the temporary root."""
    converted = json_value(value)
    workspace_text = str(workspace.resolve()) if workspace is not None else ""

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: _VALUE_KEY_RULES[key] if key in _VALUE_KEY_RULES else visit(child)
                for key, child in sorted(item.items())
            }
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, str):
            return item
        text = _ANSI_RE.sub("", item) if rule_registered("ansi_escape") else item
        if workspace_text and rule_registered("workspace_root"):
            text = text.replace(workspace_text, "<workspace>")
        if rule_registered("timestamp"):
            text = _TIMESTAMP_RE.sub("<timestamp>", text)
        return text

    return visit(converted)


__all__ = [
    "NORMALIZATION_RULES",
    "json_value",
    "load_normalization_rules",
    "normalize_value",
    "rule_registered",
]
