"""Parallel runner responsibility: manifest."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scienceflow.foundation.config.schema.settings import (
    expand_lnr_resource_control_mode_payload,
)
from scienceflow.runtime.parallel.execution.base import _public_override
from scienceflow.runtime.task_package import description_path_for_task

if TYPE_CHECKING:
    from scienceflow.runtime.parallel.config.models import TaskSpec

def _format_cpu_set_compact(cpu_ids: list[int]) -> str:
    """Format CPU ids as a compact taskset-compatible range string."""
    vals = sorted({int(c) for c in cpu_ids if int(c) >= 0})
    if not vals:
        return ""
    ranges: list[str] = []
    start = prev = vals[0]
    for cur in vals[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cur
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)

def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None

_VALID_PHASES = frozenset({"run", "prep"})

_BUDGET_DONE_STATUS = "budget_done"

def _scienceflow_repo_root() -> Path:
    """Project root (parent of the installed ``scienceflow`` package directory)."""
    override = _public_override("_scienceflow_repo_root", _scienceflow_repo_root)
    if override is not None:
        return Path(override())
    # scienceflow/runtime/parallel/config/manifest.py -> parents[4] is the repo.
    return Path(__file__).resolve().parents[4]

def _resolve_parallel_log_dir_override(log_dir: str | Path | None) -> Path | None:
    """Resolve optional legacy subprocess log dir and reject repo-root parallel_logs."""
    if log_dir is None or not str(log_dir).strip():
        return None
    raw = Path(log_dir).expanduser()
    if raw.parts and raw.parts[0] == "parallel_logs":
        raise ValueError(
            "--log-dir under repo-root parallel_logs is no longer supported; "
            "parallel subprocess logs are stored in each task workspace task_logs/.",
        )
    policy_path = raw if raw.is_absolute() else _scienceflow_repo_root() / raw
    resolved = policy_path.resolve(strict=False)
    forbidden = (_scienceflow_repo_root() / "parallel_logs").resolve(strict=False)
    try:
        inside_forbidden = resolved == forbidden or resolved.is_relative_to(forbidden)
    except ValueError:
        inside_forbidden = False
    if inside_forbidden:
        raise ValueError(
            "--log-dir under repo-root parallel_logs is no longer supported; "
            "parallel subprocess logs are stored in each task workspace task_logs/.",
        )
    return raw

def _resolve_parallel_task_text(exp_id: str, raw_task: Any) -> str:
    """Inline manifest ``task`` wins; otherwise load categorized task text."""
    if raw_task is not None:
        text = str(raw_task).strip()
        if text:
            return text
    default_path = description_path_for_task(exp_id, tasks_root=_scienceflow_repo_root() / "tasks")
    if default_path is None:
        raise ValueError(
            f"Task {exp_id!r}: empty or missing inline `task` and no default description file "
            f"registered by tasks/**/{exp_id}/task.yaml "
            f"(add `task: |` in the manifest or create that file).",
        )
    return default_path.read_text(encoding="utf-8")

def _safe_filename(s: str, max_len: int = 200) -> str:
    """Sanitize a string for use as a single path component (run_id / exp_id segments)."""
    t = re.sub(r"[^a-zA-Z0-9._-]", "_", s)[:max_len]
    return t if t else "_"

def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}

def _manifest_string_list(raw: Any) -> list[str]:
    """Normalize a manifest scalar/list CSV-ish value into a list of non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    text = str(raw).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"[,\s]+", text) if s.strip()]

def _manifest_string_csv(raw: Any) -> str:
    return ",".join(_manifest_string_list(raw))

def _env_key_pool_size(env: dict[str, str]) -> int:
    """Return the inherited key pool size used by code LLM routing."""
    for name in ("CODE_API_KEYS", "CODE_API_KEY"):
        size = len(_manifest_string_list(env.get(name, "")))
        if size > 0:
            return size
    return 0

def _env_sticky_primary_index(env: dict[str, str], task_index: int) -> int | None:
    pool_size = _env_key_pool_size(env)
    if pool_size <= 0:
        return None
    return max(0, int(task_index)) % pool_size

def _manifest_endpoint_pairs(
    keys: list[str],
    *,
    base_url: str = "",
    base_urls: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Build ``(base_url, api_key)`` pairs with the same broadcast semantics as parse_key_env."""
    if not keys:
        return []
    urls = list(base_urls or [])
    fallback_url = str(base_url or "").strip()
    if not urls and fallback_url:
        urls = [fallback_url]
    if not urls:
        urls = [""] * len(keys)
    elif len(urls) == 1 and len(keys) > 1:
        urls = urls * len(keys)
    elif len(urls) != len(keys):
        raise ValueError(
            f"Mismatch: {len(keys)} api_keys but {len(urls)} base_urls. "
            "Provide one base_url/base_urls entry to broadcast or the same count as keys.",
        )
    return list(zip(urls, keys))

def _rotate_endpoint_pairs_for_primary(
    pairs: list[tuple[str, str]], primary_index: int,
) -> list[tuple[str, str]]:
    if not pairs:
        return []
    idx = primary_index % len(pairs)
    return pairs[idx:] + pairs[:idx]

def _manifest_task_run_id(task: dict[str, Any], idx: int, exp_id: str) -> str:
    """Resolve per-task run id. Optional key ``run_id``; not inherited from defaults.

    When omitted or blank, defaults to ``exp_id`` (backward compatible).
    """
    if "run_id" not in task:
        return exp_id
    raw = task["run_id"]
    s = str(raw).strip() if raw is not None else ""
    return s if s else exp_id

def _resolve_task_workspace(
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    run_id: str,
    exp_id: str,
) -> str:
    """Resolve workspace path: explicit ``workspace`` wins, else ``workspace_base/run_id/exp_id``."""
    explicit_ws = str(task.get("workspace", "") or "").strip()
    if explicit_ws:
        return explicit_ws
    task_base = str(task.get("workspace_base", "") or "").strip()
    default_base = str(defaults.get("workspace_base", "") or "").strip()
    workspace_base = task_base or default_base
    if workspace_base:
        base = Path(workspace_base).expanduser()
        return str(base / _safe_filename(run_id) / _safe_filename(exp_id))
    raise ValueError(
        f"tasks[{idx}]: need non-empty 'workspace' or 'workspace_base' "
        f"(defaults or task) for run_id={run_id!r}, exp_id={exp_id!r}",
    )

def resolve_manifest_task_workspace(
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> str:
    """Public helper: same workspace resolution as :class:`ParallelRunner` (for monitor, tests)."""
    run_id = _manifest_task_run_id(task, idx, exp_id)
    return _resolve_task_workspace(defaults, task, idx, run_id, exp_id)

def _manifest_task_exp_id(task: dict[str, Any], idx: int) -> str:
    """Resolve per-task experiment / competition id (manifest key ``exp_id``).

    Legacy key ``name`` is still accepted but deprecated.
    """
    if "exp_id" in task:
        raw = task["exp_id"]
    elif "name" in task:
        warnings.warn(
            "Parallel manifest key 'name' is deprecated; use 'exp_id' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        raw = task["name"]
    else:
        raise ValueError(
            f"tasks[{idx}]: missing required key 'exp_id' "
            "(replace legacy 'name' with 'exp_id').",
        )
    s = str(raw).strip() if raw is not None else ""
    if not s:
        raise ValueError(f"tasks[{idx}]: 'exp_id' must be non-empty")
    return s

def _manifest_input_data_dir(defaults: dict[str, Any], task: dict[str, Any]) -> str:
    """Resolve read-only input root: same as Config.input_data_dir / ``prep --input-data-dir`` / ``-d``.

    Prefer ``input_data_dir``; accept legacy manifest key ``data_dir`` (task overrides
    defaults, first non-empty wins).
    """
    for src in (
        task.get("input_data_dir"),
        task.get("data_dir"),
        defaults.get("input_data_dir"),
        defaults.get("data_dir"),
    ):
        if src is not None and str(src).strip():
            return str(src).strip()
    return ""

def _manifest_scienceflow_interaction_log_full(
    defaults: dict[str, Any], task: dict[str, Any]
) -> bool | None:
    """Resolve ``scienceflow_interaction_log_full`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_full" in task:
        return bool(task["scienceflow_interaction_log_full"])
    if "scienceflow_interaction_log_full" in defaults:
        return bool(defaults["scienceflow_interaction_log_full"])
    return None

def _manifest_scienceflow_interaction_log_color(
    defaults: dict[str, Any], task: dict[str, Any]
) -> bool | None:
    """Resolve ``scienceflow_interaction_log_color`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_color" in task:
        return bool(task["scienceflow_interaction_log_color"])
    if "scienceflow_interaction_log_color" in defaults:
        return bool(defaults["scienceflow_interaction_log_color"])
    return None

def _manifest_scienceflow_interaction_log_llm_stream(
    defaults: dict[str, Any], task: dict[str, Any]
) -> bool | None:
    """Resolve ``scienceflow_interaction_log_llm_stream`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_llm_stream" in task:
        return bool(task["scienceflow_interaction_log_llm_stream"])
    if "scienceflow_interaction_log_llm_stream" in defaults:
        return bool(defaults["scienceflow_interaction_log_llm_stream"])
    return None

def _manifest_scienceflow_interaction_log_level(
    defaults: dict[str, Any], task: dict[str, Any]
) -> str | None:
    """Resolve ``scienceflow_interaction_log_level`` (task overrides defaults). None = do not touch env."""
    if "scienceflow_interaction_log_level" in task:
        raw = task["scienceflow_interaction_log_level"]
        return str(raw).strip().lower() if raw is not None else None
    if "scienceflow_interaction_log_level" in defaults:
        raw = defaults["scienceflow_interaction_log_level"]
        return str(raw).strip().lower() if raw is not None else None
    return None

def _manifest_run_type(
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> str:
    """Resolve phase=run child CLI type. REPL-only builds accept one solver type."""
    raw_type = task.get("type", defaults.get("type", "lnr"))
    run_type = str(raw_type).strip().lower() if raw_type is not None else "lnr"
    if run_type != "lnr":
        raise ValueError(
            f"tasks[{idx}] ({exp_id!r}): invalid type {run_type!r} (use 'lnr')",
        )
    return run_type

def _manifest_merge_lnr(
    manifest: dict[str, Any],
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> dict[str, Any]:
    """Merge lnr blocks: manifest root < defaults < task."""

    def _as_block(raw: Any, where: str) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        raise TypeError(
            f"tasks[{idx}] ({exp_id!r}): lnr at {where} must be a mapping, "
            f"got {type(raw).__name__}",
        )

    root = expand_lnr_resource_control_mode_payload(_as_block(manifest.get("lnr"), "manifest root"))
    dflt = expand_lnr_resource_control_mode_payload(_as_block(defaults.get("lnr"), "defaults"))
    tblk = expand_lnr_resource_control_mode_payload(_as_block(task.get("lnr"), "task"))
    return {**root, **dflt, **tblk}

def _manifest_merge_agent(
    manifest: dict[str, Any],
    defaults: dict[str, Any],
    task: dict[str, Any],
    idx: int,
    exp_id: str,
) -> dict[str, Any]:
    """Merge ``agent`` blocks (top-level AgentConfig scalars): manifest root < defaults < task."""

    def _as_block(raw: Any, where: str) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        raise TypeError(
            f"tasks[{idx}] ({exp_id!r}): agent at {where} must be a mapping, "
            f"got {type(raw).__name__}",
        )

    root = _as_block(manifest.get("agent"), "manifest root")
    dflt = _as_block(defaults.get("agent"), "defaults")
    tblk = _as_block(task.get("agent"), "task")
    return {**root, **dflt, **tblk}

def _validate_parallel_manifest(tasks: list[TaskSpec]) -> None:
    """Fail fast before spawning subprocesses (prep needs input_data_dir)."""
    errors: list[str] = []
    seen_run: dict[str, int] = {}
    for i, spec in enumerate(tasks):
        rid = spec.run_id or spec.exp_id
        if rid in seen_run:
            errors.append(
                f"Duplicate run_id {rid!r} at tasks[{seen_run[rid]}] and tasks[{i}]. "
                "Add explicit distinct `run_id` values per task.",
            )
        else:
            seen_run[rid] = i
    for spec in tasks:
        if spec.phase not in _VALID_PHASES:
            errors.append(
                f"{spec.exp_id}: invalid phase {spec.phase!r} (use 'run' or 'prep')",
            )
            continue
        if spec.phase == "prep":
            if not spec.input_data_dir.strip():
                errors.append(
                    f"{spec.exp_id}: phase=prep requires non-empty input_data_dir "
                    f"(manifest key input_data_dir, or legacy data_dir)",
                )
                continue
            p = Path(spec.input_data_dir).expanduser()
            if not p.is_dir():
                errors.append(
                    f"{spec.exp_id}: input_data_dir is not an existing directory: {p}",
                )
        elif spec.input_data_dir.strip():
            p = Path(spec.input_data_dir).expanduser()
            if not p.is_dir():
                errors.append(
                    f"{spec.exp_id}: input_data_dir is not an existing directory: {p}",
                )
    if errors:
        raise ValueError(
            "Parallel manifest validation failed:\n- " + "\n- ".join(errors),
        )

__all__ = tuple(name for name in globals() if not name.startswith("__"))


def _auto_worker_gpu_count(raw_gpu, run_type, is_auto, auto_count, lnr_patch):
    """Scale bare automatic GPU selection to the requested worker count."""
    count = max(1, auto_count)
    if is_auto and run_type == "lnr" and raw_gpu.strip().lower() == "auto":
        try:
            workers = int((lnr_patch or {}).get("num_workers", 2) or 2)
        except (TypeError, ValueError):
            workers = 2
        if workers > 1:
            count = max(count, workers)
    return count
