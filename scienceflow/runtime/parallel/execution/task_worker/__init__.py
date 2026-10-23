"""Private subprocess entry point for one supervised parallel worker."""

from __future__ import annotations

import asyncio
from pathlib import Path

from scienceflow.foundation.config.schema.settings import (
    apply_parallel_manifest_agent_overrides,
    apply_parallel_manifest_lnr_overrides,
    apply_profile_overrides,
    load_cfg,
    prep_cfg,
)
from scienceflow.interfaces.cli import (
    _apply_parallel_manifest_cfg_env,
    _apply_task_workspace,
)
from scienceflow.runtime.observability.logging import setup_logging


def run_worker(
    task, *, config=None, workspace=".", input_data_dir=None, task_type="lnr"
):
    """Run one worker in the foreground under its parent supervisor."""
    cfg = load_cfg(config)
    _apply_parallel_manifest_cfg_env(cfg)
    _apply_task_workspace(cfg, workspace)
    if input_data_dir is not None:
        cfg.input_data_dir = Path(input_data_dir).expanduser().resolve(strict=False)
    if task_type == "lnr":
        setattr(
            cfg,
            "_log_dir_override",
            Path(cfg.task_workspace_root_dir).expanduser().resolve(strict=False)
            / "task_logs",
        )
    apply_profile_overrides(cfg)
    apply_parallel_manifest_lnr_overrides(cfg)
    apply_parallel_manifest_agent_overrides(cfg)
    prep_cfg(cfg)
    setup_logging(
        cfg.log_dir, interaction_log_color=cfg.scienceflow_interaction_log_color
    )

    from scienceflow.runtime.workflow import Orchestrator

    result = asyncio.run(Orchestrator(cfg).run(task, task_type))
    print(f"Result: {result}")
    if isinstance(result, dict) and str(result.get("status") or "").lower() not in {
        "",
        "success",
    }:
        return 1
    return 0
