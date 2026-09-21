"""Subprocess command, environment, GPU assignment, and lifecycle helpers."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from scienceflow.research.solver.lnr.resources.runtime.control.gpu.workspace_gpu_guard import (
    cleanup_workspace_gpu_processes,
)
from scienceflow.runtime.core.process.utils import (
    spawn_exec,
    unregister_process_group,
)
from scienceflow.runtime.core.process.utils import (
    terminate_process_tree as terminate_tree,
)
from scienceflow.runtime.core.support.system_resources import (
    parse_cpu_list,
    select_least_used_gpu,
)
from scienceflow.runtime.observability.telemetry.agent.llm_config_summary import (
    format_llm_config_summary,
    merge_llm_config_summary,
    summarize_llm_config_from_agent_patch,
    summarize_llm_config_from_env,
)
from scienceflow.runtime.parallel.config.manifest import (
    _env_sticky_primary_index,
    _format_cpu_set_compact,
)
from scienceflow.runtime.parallel.config.models import TaskSpec
from scienceflow.runtime.parallel.config.preparation import _write_gpu_assignment_json
from scienceflow.runtime.parallel.execution.base import logger, resolve_project_python


def _build_parallel_command(spec: TaskSpec, child_argv: list[str]) -> tuple[list[str], str]:
    command: list[str] = []
    effective_cpu_list = spec.cpu_list or ""
    if effective_cpu_list:
        host_cores = os.cpu_count() or 0
        parsed_ids = parse_cpu_list(effective_cpu_list)
        valid_ids = [cpu for cpu in parsed_ids if cpu < host_cores]
        out_of_range = [cpu for cpu in parsed_ids if cpu >= host_cores]
        if out_of_range:
            logger.warning(
                "[parallel-runner] task=%s: dropping %d out-of-range CPU(s) %s "
                "from cpu_list %r (host has %d cores)",
                spec.run_id,
                len(out_of_range),
                out_of_range[:8],
                effective_cpu_list,
                host_cores,
            )
        if valid_ids:
            effective_cpu_list = _format_cpu_set_compact(valid_ids)
            command.extend(["taskset", "-c", effective_cpu_list])
        else:
            logger.warning(
                "[parallel-runner] task=%s: cpu_list %r entirely out-of-range "
                "for this host (%d cores); skipping taskset",
                spec.run_id,
                spec.cpu_list,
                host_cores,
            )
            effective_cpu_list = ""
    command.extend(child_argv)
    return command, effective_cpu_list


def _apply_interaction_env(env: dict[str, str], spec: TaskSpec) -> None:
    toggles = (
        ("SCIENCEFLOW_INTERACTION_LOG_FULL", spec.scienceflow_interaction_log_full, "0"),
        ("SCIENCEFLOW_INTERACTION_LOG_COLOR", spec.scienceflow_interaction_log_color, "0"),
        ("SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM", spec.scienceflow_interaction_log_llm_stream, "0"),
    )
    for name, value, false_value in toggles:
        if value is True:
            env[name] = "1"
        elif value is False:
            if name == "SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM":
                env[name] = false_value
            else:
                env.pop(name, None)
    if spec.scienceflow_interaction_log_level:
        env["SCIENCEFLOW_INTERACTION_LOG_LEVEL"] = spec.scienceflow_interaction_log_level


def _build_child_env(spec: TaskSpec, effective_cpu_list: str) -> dict[str, str]:
    env = os.environ.copy()
    child_python, child_bin = resolve_project_python()
    if child_bin:
        env["PATH"] = child_bin + ":" + env.get("PATH", "")
    child_py = Path(child_python)
    if child_py.parent.name == "bin" and child_py.parent.parent.name == ".venv":
        env["VIRTUAL_ENV"] = str(child_py.parent.parent)
    if effective_cpu_list:
        env["SCIENCEFLOW_CPU_LIST"] = effective_cpu_list
    if spec.manifest_cfg_patch:
        env["SCIENCEFLOW_PARALLEL_MANIFEST_CFG_JSON"] = json.dumps(spec.manifest_cfg_patch)
    if spec.lnr_patch and spec.phase == "run" and spec.type == "lnr":
        env["SCIENCEFLOW_PARALLEL_LNR_JSON"] = json.dumps(spec.lnr_patch)
    _apply_interaction_env(env, spec)
    return env


async def _resolve_gpu_assignment(
    runner: Any,
    spec: TaskSpec,
    env: dict[str, str],
) -> tuple[str, list[int], bool]:
    resolved_gpu = spec.gpu_list
    assigned_auto_gpus: list[int] = []
    gpu_disabled = (spec.gpu_list_raw or "").strip().lower() in {
        "none", "cpu", "off", "disabled", "-1",
    }
    if gpu_disabled:
        env["SCIENCEFLOW_DISABLE_GPU"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if not spec.gpu_auto:
        return resolved_gpu, assigned_auto_gpus, gpu_disabled
    async with runner._gpu_auto_lock:
        chosen = select_least_used_gpu(
            candidates=spec.gpu_auto_candidates or None,
            exclude=runner._gpu_auto_assigned,
            count=spec.gpu_auto_count,
        )
        if chosen is None:
            logger.warning(
                "[%s] gpu_list=auto but nvidia-smi unavailable; clearing "
                "CUDA_VISIBLE_DEVICES to let CUDA auto-select",
                spec.run_id,
            )
            env.pop("CUDA_VISIBLE_DEVICES", None)
            return resolved_gpu, assigned_auto_gpus, gpu_disabled
        resolved_gpu = chosen
        chosen_ids = [int(g.strip()) for g in chosen.split(",") if g.strip()]
        assigned_auto_gpus = [
            gpu for gpu in chosen_ids if gpu not in runner._gpu_auto_assigned
        ]
        runner._gpu_auto_assigned.update(assigned_auto_gpus)
        if len(assigned_auto_gpus) < len(chosen_ids):
            logger.info(
                "[%s] GPU stacked: %s (some already assigned to other tasks)",
                spec.run_id,
                chosen,
            )
    return resolved_gpu, assigned_auto_gpus, gpu_disabled


def _apply_resource_env(
    env: dict[str, str],
    spec: TaskSpec,
    *,
    resolved_gpu: str,
    gpu_disabled: bool,
    effective_cpu_list: str,
) -> None:
    if resolved_gpu and not gpu_disabled:
        env["CUDA_VISIBLE_DEVICES"] = resolved_gpu
        env["SCIENCEFLOW_RESOLVED_CUDA_DEVICES"] = resolved_gpu
        env["SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL"] = resolved_gpu
    if effective_cpu_list:
        env["SCIENCEFLOW_TASK_CPU_LIST"] = effective_cpu_list
    if spec.lnr_patch and spec.phase == "run" and spec.type == "lnr":
        lnr_patch = dict(spec.lnr_patch)
        if resolved_gpu and not gpu_disabled and not lnr_patch.get("resource_gpu_pool"):
            lnr_patch["resource_gpu_pool"] = [
                gpu.strip() for gpu in resolved_gpu.split(",") if gpu.strip()
            ]
        env["SCIENCEFLOW_PARALLEL_LNR_JSON"] = json.dumps(lnr_patch)


def _apply_endpoint_env(env: dict[str, str], spec: TaskSpec) -> bool:
    # Legacy unscoped manifest endpoint assignments apply only to code.
    for plural, singular in (("api_keys", "api_key"), ("base_urls", "base_url")):
        value = getattr(spec, plural) or getattr(spec, singular)
        if value:
            env[plural.upper()] = value
            env[f"CODE_{plural.upper()}"] = value
            # Compatibility projection; never discard the full list.
            env[singular.upper()] = value.split(",", 1)[0]
    if spec.models:
        env["MODELS"] = spec.models
        env["CODE_MODELS"] = spec.models
    inherited_pool = bool(
        not spec.api_key
        and not spec.api_keys
        and any(
            env.get(name, "").strip()
            for name in ("CODE_API_KEYS", "CODE_API_KEY")
        )
    )
    primary = _env_sticky_primary_index(env, spec.manifest_index) if inherited_pool else None
    if spec.llm_routing_mode:
        env["SCIENCEFLOW_LLM_ROUTING_MODE"] = spec.llm_routing_mode
    elif inherited_pool:
        env.setdefault("SCIENCEFLOW_LLM_ROUTING_MODE", "sticky_failover")
    if spec.llm_sticky_id:
        env["SCIENCEFLOW_LLM_STICKY_ID"] = spec.llm_sticky_id
    elif inherited_pool:
        env.setdefault("SCIENCEFLOW_LLM_STICKY_ID", spec.run_id or spec.exp_id)
    if spec.llm_sticky_primary_index is not None:
        env["SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX"] = str(spec.llm_sticky_primary_index)
    elif primary is not None:
        env.setdefault("SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX", str(primary))
    return inherited_pool


def _finalize_child_env(env: dict[str, str], spec: TaskSpec) -> None:
    env["SCIENCEFLOW_EXP_ID"] = spec.exp_id
    if spec.run_id != spec.exp_id:
        env["SCIENCEFLOW_RUN_ID"] = spec.run_id
    else:
        env.pop("SCIENCEFLOW_RUN_ID", None)
    if spec.agent_patch and spec.phase in {"run", "prep"}:
        env["SCIENCEFLOW_PARALLEL_AGENT_JSON"] = json.dumps(spec.agent_patch)


def _describe_key_mode(spec: TaskSpec, env: dict[str, str], inherited_pool: bool) -> str:
    if spec.api_keys and spec.llm_routing_mode:
        return f"pool:{spec.llm_routing_mode}"
    if spec.api_key:
        return "assigned"
    if inherited_pool:
        primary = env.get("SCIENCEFLOW_LLM_STICKY_PRIMARY_INDEX", "").strip()
        return "env:sticky_failover" + (f":primary={primary}" if primary else "")
    return "pool" if spec.api_keys else "env"


def _record_subprocess_start(
    runner: Any,
    spec: TaskSpec,
    *,
    env: dict[str, str],
    resolved_gpu: str,
    inherited_pool: bool,
    log_file: str,
) -> None:
    key_mode = _describe_key_mode(spec, env, inherited_pool)
    gpu_info = f"gpu={resolved_gpu}" + (" (auto)" if spec.gpu_auto else "")
    llm_config = summarize_llm_config_from_env(env, source="parallel_runner_env")
    agent_config = summarize_llm_config_from_agent_patch(spec.agent_patch)
    if agent_config:
        llm_config = merge_llm_config_summary(
            llm_config,
            agent_config,
            source="parallel_runner_env+agent_patch",
        )
    llm_info = format_llm_config_summary(llm_config).replace("\n", "; ") or "llm=unknown"
    logger.info(
        f"[{spec.run_id}] ({spec.exp_id}) Starting phase={spec.phase} "
        f"cpu={spec.cpu_list} {gpu_info} keys={key_mode} llm={llm_info}"
    )
    _write_gpu_assignment_json(spec, resolved_gpu)
    runner._write_running_state(
        spec,
        resolved_gpu=resolved_gpu,
        llm_config=llm_config,
        log_file=log_file,
    )


async def _run_child(command: list[str], env: dict[str, str], spec: TaskSpec, log_file: str) -> int:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as output:
        proc = await spawn_exec(
            *command,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=spec.time_limit)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await terminate_tree(proc)
            raise
        finally:
            unregister_process_group(proc.pid)
    return proc.returncode


async def _cleanup_subprocess_resources(
    runner: Any,
    spec: TaskSpec,
    *,
    resolved_gpu: str,
    gpu_disabled: bool,
    assigned_auto_gpus: list[int],
    task_started_at: float,
    log_file: str,
) -> None:
    if resolved_gpu and not gpu_disabled:
        try:
            cleanup = await asyncio.to_thread(
                cleanup_workspace_gpu_processes,
                workspace_dir=spec.workspace,
                allowed_gpu_ids=[gpu.strip() for gpu in resolved_gpu.split(",") if gpu.strip()],
                task_started_at=task_started_at,
                reason="parallel_task_finish",
                dry_run=False,
                sigterm_grace_sec=0.5,
            )
            relevant_skips = any(
                str((row or {}).get("skip_reason") or "")
                not in {"workspace_mismatch", "process_started_before_task"}
                for row in cleanup.get("skipped") or []
            )
            if cleanup.get("killed_count") or cleanup.get("violation_count") or relevant_skips:
                with open(log_file, "a") as output:
                    output.write("\n[parallel-runner] workspace_gpu_cleanup ")
                    output.write(json.dumps(cleanup, ensure_ascii=False, sort_keys=True)[:4000])
                    output.write("\n")
        except Exception:
            logger.debug("[parallel-runner] workspace gpu cleanup failed", exc_info=True)
    if assigned_auto_gpus:
        async with runner._gpu_auto_lock:
            for gpu in assigned_auto_gpus:
                runner._gpu_auto_assigned.discard(gpu)
