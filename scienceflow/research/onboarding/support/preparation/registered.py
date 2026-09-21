"""Deterministic preparation for trusted registered task packages."""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from tempfile import NamedTemporaryFile

from scienceflow.research.onboarding.support.answers import (
    parse_cpu_list,
    parse_duration,
    parse_gpu_selection,
    parse_workers,
)
from scienceflow.runtime.task_package import TaskPackageSpec

_DURATION_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?\s*(?:seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|hr|h|days?|d|分钟|小时))(?!\w)",
    re.IGNORECASE,
)


def registered_preparation_fields(
    spec: TaskPackageSpec,
    text: str,
    *,
    workspace: Path,
    reserved_cpu_ids: frozenset[int] = frozenset(),
) -> dict[str, object]:
    """Parse runtime overrides without asking an LLM to rediscover the task."""

    source = str(text or "").strip()
    fields: dict[str, object] = {}

    workers = _match_value(source, r"(?:workers?|worker数量|工作进程)\s*=?\s*(\d+)")
    if workers is None:
        workers = _match_value(source, r"(\d+)\s*个?\s*workers?")
    if workers is not None:
        fields["workers"] = parse_workers(workers)

    duration = _match_value(source, r"(?:duration|time|时长|时间)\s*=?\s*([^\s,，;；]+)")
    if duration is None:
        match = _DURATION_RE.search(source)
        duration = match.group(1) if match else None
    if duration is not None:
        fields["wall_clock_sec"] = parse_duration(duration)

    gpu = _match_value(source, r"(?:gpus?|显卡)\s*=\s*([^\s，;；]+)")
    if gpu is not None:
        fields["gpu_list"] = parse_gpu_selection(gpu)

    cpu = _match_value(source, r"(?:cpus?|cpu核)\s*=?\s*([^\s，;；]+)")
    if cpu is not None:
        fields["cpu_list"] = _cpu_pool(cpu, reserved_cpu_ids=reserved_cpu_ids)
        fields.setdefault("gpu_list", "cpu")
    elif re.search(r"(?:使用|用)?\s*cpu(?:运行)?", source, re.IGNORECASE):
        fields["gpu_list"] = "cpu"

    data = _explicit_data_path(source)
    if data is not None:
        resolved = _validated_data_path(data, workspace)
        if resolved == "none" and _inputs_required(spec):
            raise ValueError(f"Task {spec.task_id} requires an input dataset")
        fields["input_data_dir"] = resolved
        if resolved != "none":
            remember_registered_input(spec, workspace=workspace, path=Path(resolved))
    elif _inputs_required(spec):
        discovered = discover_registered_input(spec, workspace=workspace)
        if discovered is not None:
            fields["input_data_dir"] = str(discovered)
    else:
        fields["input_data_dir"] = "none"

    return fields


def registered_missing_prompt(draft) -> str:
    """Return one compact prompt for all unresolved registered-task settings."""

    missing: list[str] = []
    if not draft.input_data_dir:
        missing.append("data=/path/to/dataset")
    if draft.workers is None:
        missing.append("workers=2")
    if not draft.cpu_list:
        missing.append("cpu=8")
    if not draft.gpu_list:
        missing.append("gpu=cpu")
    if draft.wall_clock_sec is None:
        missing.append("duration=20min")
    if not missing:
        return ""
    return "请补充运行配置：" + " ".join(missing) + " 可在一行内填写"


def discover_registered_input(spec: TaskPackageSpec, *, workspace: Path) -> Path | None:
    """Resolve a dataset from workspace state or an explicitly configured root."""

    inputs = spec.config.get("inputs") or {}
    candidates: list[Path] = []
    remembered = _read_dataset_registry(workspace).get(spec.task_id)
    if remembered:
        candidates.append(Path(remembered).expanduser())
    configured = inputs.get("path") if isinstance(inputs, dict) else None
    if configured:
        candidates.append(Path(str(configured)).expanduser())

    if spec.profile == "mlebench":
        roots: list[Path] = []
        env_root = os.environ.get("MLEBENCH_DATA_ROOT_DIR", "").strip()
        if env_root:
            roots.append(Path(env_root).expanduser())
        roots.extend(
            [
                workspace / "data" / "mlebench_all_data",
            ]
        )
        for root in roots:
            task_root = root / spec.task_id
            candidates.extend(
                [
                    task_root / "prepared" / "dataset_split" / "Deep",
                    task_root / "prepared" / "public",
                    task_root / "public",
                ]
            )

    for candidate in candidates:
        path = candidate if candidate.is_absolute() else workspace / candidate
        if path.is_dir():
            return path.resolve()
    return None


def remember_registered_input(
    spec: TaskPackageSpec,
    *,
    workspace: Path,
    path: Path,
) -> None:
    """Persist an explicit task dataset selection in the current workspace."""

    registry_path = _dataset_registry_path(workspace)
    registry = _read_dataset_registry(workspace)
    workspace_root = Path(workspace).resolve()
    resolved = path.expanduser().resolve()
    try:
        stored_path = resolved.relative_to(workspace_root).as_posix()
    except ValueError:
        stored_path = str(resolved)
    registry[spec.task_id] = stored_path
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=registry_path.parent,
        delete=False,
    ) as stream:
        json.dump({"version": 1, "tasks": registry}, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        temporary.replace(registry_path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_registry_path(workspace: Path) -> Path:
    return Path(workspace).resolve() / ".scienceflow" / "datasets.json"


def _read_dataset_registry(workspace: Path) -> dict[str, str]:
    path = _dataset_registry_path(workspace)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, dict):
        return {}
    return {
        str(task_id): str(value)
        for task_id, value in tasks.items()
        if str(task_id).strip() and str(value).strip()
    }


def _inputs_required(spec: TaskPackageSpec) -> bool:
    inputs = spec.config.get("inputs") or {}
    return not (isinstance(inputs, dict) and inputs.get("required") is False)


def _match_value(source: str, pattern: str) -> str | None:
    match = re.search(pattern, source, re.IGNORECASE)
    return match.group(1).strip().rstrip("。.!！,，;；") if match else None


def _cpu_pool(raw: str, *, reserved_cpu_ids: frozenset[int]) -> str:
    value = str(raw).strip()
    if re.fullmatch(r"\d+", value):
        count = int(value)
        if count < 1:
            raise ValueError("CPU count must be positive")
        affinity = (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else list(range(os.cpu_count() or 1))
        )
        available = [cpu_id for cpu_id in affinity if cpu_id not in reserved_cpu_ids]
        if count > len(available):
            raise ValueError(
                f"CPU count {count} exceeds the {len(available)} unreserved CPUs"
            )
        return _compact_ids(available[:count])
    parsed = parse_cpu_list(value)
    overlap = _expand_cpu_ids(parsed) & reserved_cpu_ids
    if overlap:
        raise ValueError(f"CPU IDs already reserved by another task: {sorted(overlap)}")
    return parsed


def _compact_ids(values: list[int]) -> str:
    if not values:
        raise ValueError("No CPUs are available")
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


def _expand_cpu_ids(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        bounds = [int(item) for item in part.split("-", 1)]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _explicit_data_path(source: str) -> str | None:
    try:
        tokens = shlex.split(source.replace("，", " ").replace("；", " "))
    except ValueError:
        tokens = source.replace("，", " ").replace("；", " ").split()
    for index, token in enumerate(tokens):
        match = re.fullmatch(r"(?:data|dataset|input)=(.+)", token, re.IGNORECASE)
        if match:
            return match.group(1)
        if token.casefold() in {"data", "dataset", "input"} and index + 1 < len(tokens):
            return tokens[index + 1]
    if len(tokens) == 1:
        candidate = Path(tokens[0]).expanduser()
        if candidate.is_absolute() or tokens[0].startswith(("./", "../", "~/")):
            return tokens[0]
    return None


def _validated_data_path(value: str, workspace: Path) -> str:
    if value.strip().casefold() in {"none", "no-data", "无", "不需要数据"}:
        return "none"
    candidate = Path(value).expanduser()
    candidate = candidate if candidate.is_absolute() else workspace / candidate
    if not candidate.is_dir():
        raise ValueError(f"Dataset directory does not exist: {candidate}")
    return str(candidate.resolve())


__all__ = [
    "discover_registered_input",
    "registered_missing_prompt",
    "registered_preparation_fields",
    "remember_registered_input",
]
