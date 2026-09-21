"""ParallelRunner responsibility: construction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml

from scienceflow.foundation.config.schema.settings import (
    merge_parallel_manifest_cfg_patch,
)
from scienceflow.runtime.core.support.system_resources import parse_gpu_list_auto
from scienceflow.runtime.parallel.config.manifest import (
    _auto_worker_gpu_count,
    _manifest_endpoint_pairs,
    _manifest_input_data_dir,
    _manifest_merge_agent,
    _manifest_merge_lnr,
    _manifest_run_type,
    _manifest_scienceflow_interaction_log_color,
    _manifest_scienceflow_interaction_log_full,
    _manifest_scienceflow_interaction_log_level,
    _manifest_scienceflow_interaction_log_llm_stream,
    _manifest_string_csv,
    _manifest_string_list,
    _manifest_task_exp_id,
    _manifest_task_run_id,
    _resolve_parallel_log_dir_override,
    _resolve_parallel_task_text,
    _resolve_task_workspace,
    _rotate_endpoint_pairs_for_primary,
    _validate_parallel_manifest,
)
from scienceflow.runtime.parallel.config.models import TaskSpec


def __init__(
    self,
    manifest_path: str | Path,
    max_concurrent: int | None = None,
    log_dir: str | Path | None = None,
):
    self.manifest_path = Path(manifest_path)
    self.log_dir = _resolve_parallel_log_dir_override(log_dir)
    self._manifest: dict[str, Any] = {}
    self._tasks: list[TaskSpec] = []
    self._max_concurrent = max_concurrent
    self._resume_budget_policy = "remaining"
    self._gpu_auto_lock = asyncio.Lock()
    self._gpu_auto_assigned: set[int] = set()
    self._load_manifest()


def _load_manifest(self) -> None:
    with open(self.manifest_path) as f:
        self._manifest = yaml.safe_load(f) or {}

    defaults = self._manifest.get("defaults") or {}

    global_time_limit = self._manifest.get("time_limit", 3600)
    self._resume_budget_policy = str(
        self._manifest.get("resume_budget_policy", "remaining") or "remaining"
    ).strip().lower()
    if self._resume_budget_policy not in {"remaining", "fresh"}:
        raise ValueError("resume_budget_policy must be one of: remaining, fresh")
    if self._max_concurrent is None:
        self._max_concurrent = self._manifest.get("max_concurrent", 4)

    global_api_keys = _manifest_string_list(self._manifest.get("api_keys", defaults.get("api_keys", self._manifest.get("api_key", ""))))
    global_base_url = self._manifest.get("base_url", defaults.get("base_url", ""))
    global_base_urls = _manifest_string_list(self._manifest.get("base_urls", defaults.get("base_urls", [])))
    global_models = _manifest_string_list(self._manifest.get("models", defaults.get("models", [])))
    global_endpoint_pairs = _manifest_endpoint_pairs(
        global_api_keys,
        base_url=str(global_base_url or ""),
        base_urls=global_base_urls,
    )

    default_phase = str(defaults.get("phase", "run")).strip().lower() or "run"

    raw_tasks = self._manifest.get("tasks", [])
    if raw_tasks is None:
        raw_tasks = []
    for idx, t in enumerate(raw_tasks):
        if not isinstance(t, dict):
            raise TypeError(f"tasks[{idx}] must be a mapping, got {type(t).__name__}")

        phase = str(t.get("phase", default_phase)).strip().lower() or "run"
        input_data_dir = _manifest_input_data_dir(defaults, t)
        exp_id = _manifest_task_exp_id(t, idx)
        run_id = _manifest_task_run_id(t, idx, exp_id)
        workspace = _resolve_task_workspace(defaults, t, idx, run_id, exp_id)
        task_api_key = str(t.get("api_key", "") or "").strip()
        task_api_keys = _manifest_string_csv(t.get("api_keys", task_api_key))
        task_models = _manifest_string_csv(t.get("models", global_models))
        task_base_url = str(t.get("base_url", global_base_url) or "").strip()
        task_base_urls = _manifest_string_csv(
            t.get("base_urls", t.get("base_url", global_base_urls or task_base_url)),
        )
        llm_routing_mode = str(
            t.get("llm_routing_mode", self._manifest.get("llm_routing_mode", ""))
            or "",
        ).strip()
        llm_sticky_id = str(
            t.get("llm_sticky_id", self._manifest.get("llm_sticky_id", ""))
            or "",
        ).strip()
        llm_sticky_primary_index: int | None = None
        if not task_api_key and not task_api_keys and global_endpoint_pairs:
            primary_index = idx % len(global_endpoint_pairs)
            ordered_pairs = _rotate_endpoint_pairs_for_primary(
                global_endpoint_pairs,
                primary_index,
            )
            if "models" not in t and len(global_models) > 1:
                if len(global_models) != len(global_endpoint_pairs):
                    raise ValueError("models must contain one entry or one per api_keys entry")
                task_models = ",".join(global_models[primary_index:] + global_models[:primary_index])
            task_api_key = ordered_pairs[0][1]
            task_base_url = ordered_pairs[0][0]
            if len(ordered_pairs) > 1:
                task_api_keys = ",".join(key for _, key in ordered_pairs)
                urls = [url for url, _ in ordered_pairs]
                if len(set(urls)) == 1:
                    task_base_url = urls[0]
                    task_base_urls = ""
                else:
                    task_base_urls = ",".join(urls)
                llm_routing_mode = llm_routing_mode or "sticky_failover"
                llm_sticky_id = llm_sticky_id or run_id or exp_id
                llm_sticky_primary_index = 0

        # Keep TaskSpec's canonical lists even for a single endpoint.
        task_api_keys = task_api_keys or task_api_key
        task_base_urls = task_base_urls or task_base_url
        task_api_key = task_api_keys.split(",", 1)[0] if task_api_keys else ""
        task_base_url = task_base_urls.split(",", 1)[0] if task_base_urls else ""

        if phase == "run":
            run_type = _manifest_run_type(defaults, t, idx, exp_id)
        else:
            run_type = "lnr"

        lnr_patch = _manifest_merge_lnr(self._manifest, defaults, t, idx, exp_id)
        lnr_patch_arg: dict[str, Any] | None = lnr_patch if lnr_patch else None
        agent_patch = _manifest_merge_agent(self._manifest, defaults, t, idx, exp_id)
        agent_patch_arg: dict[str, Any] | None = agent_patch if agent_patch else None
        mcfg = merge_parallel_manifest_cfg_patch(defaults, t)
        manifest_cfg_patch_arg: dict[str, Any] | None = mcfg if mcfg else None
        interaction_log_full = _manifest_scienceflow_interaction_log_full(defaults, t)
        interaction_log_color = _manifest_scienceflow_interaction_log_color(defaults, t)
        interaction_log_llm_stream = _manifest_scienceflow_interaction_log_llm_stream(defaults, t)
        interaction_log_level = _manifest_scienceflow_interaction_log_level(defaults, t)

        raw_gpu = str(t.get("gpu_list", "") or "")
        gpu_disabled = raw_gpu.strip().lower() in {"none", "cpu", "off", "disabled", "-1"}
        is_auto, auto_count, auto_candidates = (
            (False, 0, None) if gpu_disabled else parse_gpu_list_auto(raw_gpu)
        )
        auto_count_eff = _auto_worker_gpu_count(raw_gpu, run_type, is_auto, auto_count, lnr_patch_arg)
        self._tasks.append(
            TaskSpec(
                exp_id=exp_id,
                task=_resolve_parallel_task_text(exp_id, t.get("task")),
                workspace=workspace,
                manifest_index=idx,
                run_id=run_id,
                cpu_list=t.get("cpu_list", ""),
                gpu_list="" if (is_auto or gpu_disabled) else raw_gpu,
                config=t.get("config", defaults.get("config", "")),
                type=run_type,
                time_limit=t.get("time_limit", global_time_limit),
                api_key=task_api_key,
                api_keys=task_api_keys,
                base_url=task_base_url,
                base_urls=task_base_urls,
                models=task_models,
                llm_routing_mode=llm_routing_mode,
                llm_sticky_id=llm_sticky_id,
                llm_sticky_primary_index=llm_sticky_primary_index,
                phase=phase,
                input_data_dir=input_data_dir,
                num_drafts=int(t.get("num_drafts", defaults.get("num_drafts", 1)) or 1),
                draft_parallel=int(
                    t.get("draft_parallel", defaults.get("draft_parallel", 1)) or 1,
                ),
                lnr_patch=lnr_patch_arg,
                agent_patch=agent_patch_arg,
                manifest_cfg_patch=manifest_cfg_patch_arg,
                scienceflow_interaction_log_full=interaction_log_full,
                scienceflow_interaction_log_color=interaction_log_color,
                scienceflow_interaction_log_llm_stream=interaction_log_llm_stream,
                scienceflow_interaction_log_level=interaction_log_level,
                gpu_auto=is_auto,
                gpu_auto_count=auto_count_eff,
                gpu_auto_candidates=auto_candidates,
                gpu_list_raw=raw_gpu,
            )
        )

    _validate_parallel_manifest(self._tasks)
