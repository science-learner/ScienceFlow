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

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from scienceflow.agent.factory.composition import create_science_agent as build_science_agent
from scienceflow.foundation.config.llm.llm_http import aclose_llm_clients as _aclose_llm_clients
from scienceflow.foundation.config.schema.settings import Config, prep_cfg
from scienceflow.runtime.observability.telemetry.agent.llm_call_adapter import (
    make_llm_call_tracer as build_llm_call_tracer,
)

if TYPE_CHECKING:
    from scienceflow.agent import ScienceAgent
    from scienceflow.agent.core.runtime.run_policy import RunPolicy

class Orchestrator:
    """Main entry point for REPL-native long-horizon tasks."""

    def __init__(self, cfg: Config):
        self.cfg = prep_cfg(cfg)

    def make_llm_call_tracer(
        self,
        *,
        node_id: str = "repl",
        process_id: int | str | None = None,
        fork_class: str | None = None,
        detail_prefix: str = "mode=repl",
    ) -> Callable[[dict[str, Any]], None] | None:
        return build_llm_call_tracer(
            self.cfg,
            node_id=node_id,
            process_id=process_id,
            fork_class=fork_class,
            detail_prefix=detail_prefix,
        )

    def create_science_agent(
        self,
        *,
        ui: object | None = None,
        task_description: str | None = None,
        run_policy: RunPolicy | None = None,
        system_prompt: str | None = None,
        system_prompt_hook: Callable[[str], str] | None = None,
        append_repl_system_prompt: bool | None = None,
        pin_task_description: bool = True,
        max_steps_override: int | None = None,
        on_llm_call: Callable[[dict[str, Any]], None] | None = None,
        memory_dir_override: str | Path | None = None,
        load_existing_memory: bool = False,
        memory_agent_name: str = "ScienceAgent",
        teleport_mode: str = "off",
        repl_bash_write_mode: bool = False,
        stable_system_prompt: bool | None = None,
        pin_environment_context: bool | None = None,
        code_organization_hint: str | None = None,
        workspace_git_enabled: bool = False,
        workspace_git_track_globs: list[str] | tuple[str, ...] | None = None,
        workspace_git_auto_review: bool = False,
        workspace_git_auto_checkpoint: bool = False,
        bash_max_output_chars_override: int | None = None,
        bash_max_stream_line_chars_override: int | None = None,
        bash_observation_summary_override: bool | None = None,
        interaction_log_layout: str = "flat",
        extra_env_override: dict[str, str] | None = None,
        skill_registry: object | None = None,
        task_type: str | None = None,
        skill_allow_names: list[str] | tuple[str, ...] | None = None,
        skill_tool_mode: str = "all",
        skill_allow_generic_wildcard: bool = True,
        skill_visible_max: int = 0,
        llm_stage_override: object | None = None,
    ) -> ScienceAgent:
        """Construct a ScienceAgent through the domain composition adapter."""
        options = dict(locals())
        options.pop("self")
        return build_science_agent(self.cfg, **options)

    async def run_science_task(self, request: str, *, ui: object | None = None) -> str:
        """One-shot ScienceAgent run (new agent + memory each call)."""
        agent = self.create_science_agent(ui=ui)
        try:
            return await agent.run(request) or "(no text output)"
        finally:
            await _aclose_llm_clients(agent.llm)

    async def run_lnr_task(
        self,
        task_desc: str,
        resume: bool = False,
        resume_step: int | None = None,
    ) -> dict:
        """Run the REPL-native long-horizon stage-capture solver."""
        _ = resume, resume_step
        from scienceflow.research.solver.lnr import LnrSolver

        solver = LnrSolver(
            task_desc=task_desc,
            cfg=self.cfg,
            orchestrator=self,
        )
        return await solver.run()

    async def run(
        self,
        task_desc: str,
        task_type: str = "lnr",
        *,
        ui: "object | None" = None,
        resume: bool = False,
        resume_step: int | None = None,
    ) -> dict | str:
        """Main entry: REPL-native long-horizon solver only."""
        _ = ui, resume, resume_step
        tt = (task_type or "lnr").strip().lower()
        if tt != "lnr":
            raise ValueError(
                f"Unknown task_type {task_type!r} (expected 'lnr')",
            )
        return await self.run_lnr_task(task_desc)
