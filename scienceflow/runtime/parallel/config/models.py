"""Parallel runner responsibility: models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskSpec:
    exp_id: str
    task: str
    workspace: str
    manifest_index: int = 0
    run_id: str = ""
    cpu_list: str = ""
    gpu_list: str = ""
    config: str = ""
    type: str = "lnr"
    time_limit: int = 3600
    api_key: str = ""
    api_keys: str = ""
    base_url: str = ""
    base_urls: str = ""
    models: str = ""
    llm_routing_mode: str = ""
    llm_sticky_id: str = ""
    llm_sticky_primary_index: int | None = None
    # phase "run" -> `cli run`; phase "prep" -> `cli prep`.
    phase: str = "run"
    input_data_dir: str = ""
    num_drafts: int = 1
    draft_parallel: int = 1
    # Merged from manifest lnr; passed via SCIENCEFLOW_PARALLEL_LNR_JSON.
    lnr_patch: dict[str, Any] | None = None
    # Merged from manifest ``agent`` (root / defaults / task); top-level AgentConfig scalars via SCIENCEFLOW_PARALLEL_AGENT_JSON.
    agent_patch: dict[str, Any] | None = None
    # Top-level :class:`Config` keys from manifest ``defaults`` + ``task`` (task wins); passed via env to prep/run.
    manifest_cfg_patch: dict[str, Any] | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_FULL`` is forced on/off (see ``_exec_subprocess``).
    scienceflow_interaction_log_full: bool | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_COLOR`` is forced on/off.
    scienceflow_interaction_log_color: bool | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_LLM_STREAM`` is forced 1/0.
    scienceflow_interaction_log_llm_stream: bool | None = None
    # When set, child env ``SCIENCEFLOW_INTERACTION_LOG_LEVEL`` is set (minimal|normal|verbose).
    scienceflow_interaction_log_level: str | None = None
    # Auto GPU selection: when True, gpu_list is resolved at subprocess launch
    # by picking the GPU(s) with the most free memory from gpu_auto_candidates.
    gpu_auto: bool = False
    gpu_auto_count: int = 1
    gpu_auto_candidates: list[int] = field(default_factory=list)
    # Original manifest ``gpu_list`` string (e.g. ``auto:2``) for gpu_assignment.json.
    gpu_list_raw: str = ""

@dataclass
class TaskResult:
    exp_id: str
    run_id: str = ""
    status: str = "pending"
    exit_code: int | None = None
    elapsed_sec: float = 0.0
    error: str = ""
    log_file: str = ""

__all__ = tuple(name for name in globals() if not name.startswith("__"))
