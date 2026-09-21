"""Deterministic parsing for long-research onboarding answers."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

FIELD_ALIASES = {
    "task": "task_text",
    "description": "task_text",
    "data": "input_data_dir",
    "dataset": "input_data_dir",
    "input": "input_data_dir",
    "metric": "metric_name",
    "direction": "lower_is_better",
    "artifact": "artifact_path",
    "command": "artifact_command",
    "evaluator_command": "artifact_command",
    "gpu": "gpu_list",
    "gpus": "gpu_list",
    "worker": "workers",
    "num_workers": "workers",
    "cpu": "cpu_list",
    "cpus": "cpu_list",
    "duration": "wall_clock_sec",
    "time": "wall_clock_sec",
    "task_id": "exp_id",
    "models": "code_models",
    "code_models": "code_models",
    "code-models": "code_models",
    "feedback_models": "feedback_models",
    "feedback-models": "feedback_models",
    "model_policy": "model_selection",
    "model-policy": "model_selection",
    "selection": "model_selection",
}

SUPPORTED_FIELDS = {
    "task_text",
    "input_data_dir",
    "metric_name",
    "lower_is_better",
    "artifact_path",
    "artifact_command",
    "gpu_list",
    "workers",
    "cpu_list",
    "wall_clock_sec",
    "exp_id",
    "code_models",
    "feedback_models",
    "model_selection",
}

MIN_RESEARCH_DURATION_SEC = 10 * 60


def detach_conversation(conversation: Sequence[object]) -> tuple[tuple[str, str], ...]:
    """Copy supported message shapes into immutable role/content pairs."""

    detached: list[tuple[str, str]] = []
    for message in conversation:
        role: object = ""
        content: object = ""
        if isinstance(message, Mapping):
            role, content = message.get("role", ""), message.get("content", "")
        elif isinstance(message, (tuple, list)) and len(message) >= 2:
            role, content = message[0], message[1]
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
        role_text = str(getattr(role, "value", role) or "").strip().casefold()
        content_text = str(content or "").strip()
        if role_text and content_text:
            detached.append((role_text, content_text))
    return tuple(detached)


def extract_task_text(conversation: Sequence[object]) -> str:
    """Extract the latest user objective without retaining live conversation objects."""

    for role, content in reversed(detach_conversation(conversation)):
        if role in {"user", "human"} and not content.lstrip().startswith("/"):
            return content.strip()
    return ""


def parse_constraints(text: str) -> dict[str, object]:
    """Parse optional ``key=value`` command arguments without guessing free text."""

    source = str(text or "").strip()
    if not source:
        return {}
    try:
        lexer = shlex.shlex(source, posix=True)
        lexer.whitespace += "，；"
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError(f"invalid long-research arguments: {exc}") from exc
    values: dict[str, object] = {}
    free: list[str] = []
    for token in tokens:
        if "=" not in token:
            if re.fullmatch(r"\d+(?:\.\d+)?(?:min|m|h|s|d|分钟|小时)", token, re.IGNORECASE):
                values["wall_clock_sec"] = parse_duration(token)
                continue
            free.append(token)
            continue
        key, raw = token.split("=", 1)
        field_name = FIELD_ALIASES.get(key.strip().casefold(), key.strip().casefold())
        if field_name not in SUPPORTED_FIELDS:
            raise ValueError(f"unknown long-research option: {key}")
        values[field_name] = parse_field(field_name, raw)
    if free:
        values.setdefault("task_text", " ".join(free))
    return values


def parse_gpu_selection(value: str) -> str:
    text = str(value or "").strip().casefold().replace(" ", "")
    if text in {"cpu", "none", "off", "disabled", "-1"}:
        return "cpu"
    if text == "auto":
        return "auto"
    match = re.fullmatch(r"auto:(\d+)", text)
    if match:
        count = int(match.group(1))
        if count < 1:
            raise ValueError("automatic GPU count must be positive")
        return f"auto:{count}"
    if not re.fullmatch(r"\d+(?:,\d+)*", text):
        raise ValueError(
            "GPU must be cpu, auto, auto:N, or comma-separated non-negative IDs"
        )
    ids = [int(item) for item in text.split(",")]
    if len(ids) != len(set(ids)):
        raise ValueError("GPU IDs must not contain duplicates")
    return ",".join(str(item) for item in ids)


def parse_cpu_list(value: str) -> str:
    text = str(value or "").strip().replace(" ", "")
    if not text or not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", text):
        raise ValueError("CPU pool must use IDs/ranges such as 0-15,20-23")
    ids: set[int] = set()
    for part in text.split(","):
        bounds = [int(item) for item in part.split("-", 1)]
        start, end = (bounds[0], bounds[0]) if len(bounds) == 1 else bounds
        if end < start:
            raise ValueError(f"CPU range must be ascending: {part}")
        if ids.intersection(range(start, end + 1)):
            raise ValueError("CPU pool must not contain overlapping IDs")
        ids.update(range(start, end + 1))
    return _compact_ids(ids)


def parse_duration(value: str | int) -> int:
    text = str(value).strip().casefold()
    text = re.sub(r"(?:minutes?|min|分钟)$", "m", text)
    text = re.sub(r"(?:hours?|hr|小时)$", "h", text)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", text)
    if not match:
        raise ValueError("duration must look like 7200s, 90m, 2h, or 1d")
    amount = float(match.group(1))
    multiplier = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    seconds = int(amount * multiplier)
    if seconds < 1:
        raise ValueError("duration must be positive")
    if seconds < MIN_RESEARCH_DURATION_SEC:
        raise ValueError(
            "research duration must be at least 10 minutes "
            "(for example 10m or 600s)"
        )
    return seconds


def parse_workers(value: str | int) -> int:
    try:
        workers = int(str(value).strip())
    except ValueError as exc:
        raise ValueError("worker count must be a positive integer") from exc
    if workers < 1:
        raise ValueError("worker count must be a positive integer")
    return workers


def parse_field(name: str, value: object) -> object:
    if name == "lower_is_better" and isinstance(value, bool):
        return value
    if name in {"code_models", "feedback_models"} and isinstance(value, (list, tuple)):
        aliases = [str(item).strip() for item in value if str(item).strip()]
        if not aliases:
            return []
        return aliases
    text = str(value or "").strip()
    if name == "gpu_list":
        return parse_gpu_selection(text)
    if name == "cpu_list":
        return parse_cpu_list(text)
    if name == "workers":
        return parse_workers(text)
    if name == "wall_clock_sec":
        return parse_duration(text)
    if name in {"code_models", "feedback_models"}:
        if text.casefold() == "auto":
            return []
        aliases = [item.strip() for item in text.split(",") if item.strip()]
        if not aliases:
            raise ValueError(f"{name} must contain model aliases or auto")
        return aliases
    if name == "model_selection":
        policy = text.casefold()
        if policy not in {"auto", "spread", "fixed"}:
            raise ValueError("model selection must be auto, spread, or fixed")
        return policy
    if name == "lower_is_better":
        normalized = text.casefold()
        if normalized in {
            "lower",
            "min",
            "minimize",
            "minimized",
            "minimised",
            "minimise",
            "yes",
            "true",
            "低",
            "最小化",
        }:
            return True
        if normalized in {
            "higher",
            "max",
            "maximize",
            "maximized",
            "maximised",
            "maximise",
            "no",
            "false",
            "高",
            "最大化",
        }:
            return False
        raise ValueError("metric direction must be lower/minimize or higher/maximize")
    if name == "input_data_dir" and text.casefold() in {"none", "no data", "无", "无需数据", "不需要数据"}:
        return "none"
    if name == "artifact_path":
        path = Path(text)
        if not text or path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "artifact path must be a safe path relative to the worker workspace"
            )
        return path.as_posix()
    if name in {
        "task_text",
        "input_data_dir",
        "metric_name",
        "artifact_command",
        "exp_id",
    }:
        if not text:
            raise ValueError(f"{name} must not be empty")
        return text
    raise ValueError(f"unsupported onboarding field: {name}")


def _compact_ids(ids: set[int]) -> str:
    values = sorted(ids)
    ranges: list[str] = []
    start = previous = values[0]
    for current in values[1:]:
        if current == previous + 1:
            previous = current
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = current
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


__all__ = [
    "detach_conversation",
    "extract_task_text",
    "parse_constraints",
    "parse_cpu_list",
    "parse_duration",
    "parse_field",
    "parse_gpu_selection",
    "parse_workers",
]
