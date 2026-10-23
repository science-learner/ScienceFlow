# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Single-worker LNR lifecycle loop."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from scienceflow.foundation.config.llm.llm_http import aclose_llm_clients
from scienceflow.research.state.knowledge.prompt import PromptContextRequest
from scienceflow.research.solver.lnr.support.prompts import build_first_user_prompt
from scienceflow.research.solver.lnr.transitions.resume import resume_loaded_agent_from_memory
from scienceflow.research.solver.lnr.orchestration.runtime.services.worker_services import WorkerRuntimeServices
from scienceflow.research.solver.lnr.orchestration.runtime.services.worker_services import WorkerSearchOutcome

logger = logging.getLogger("scienceflow")


async def run_single_worker(
    services: WorkerRuntimeServices, *, keep_agent_open: bool = False
) -> dict[str, Any]:
    """Run one worker using only its declared capability ports."""

    spec = services.spec
    services.prepare_workspace()
    services.load_existing_stage_snapshots()
    if not services.initialize_snapshot_baseline():
        logger.warning(
            "[lnr] snapshot baseline initialization failed; falling back to allowlist snapshots"
        )
    services.lifecycle.mark_running(
        {"mode": "single_worker", "worker_id": spec.worker_id}
    )
    prompt_context = services.build_prompt_context(
        PromptContextRequest(
            task_description=spec.task_description,
            worker_id=spec.worker_id,
            wall_clock_budget_sec=spec.wall_clock_budget_sec,
            seed=spec.seed,
            workspace_facts=services.initial_workspace_state(),
            parallel_worker_facts=services.parallel_worker_snapshot_for_prompt(),
            resource_observation=services.resource_context_for_prompt(),
            gate_constraints=services.evaluator_prompt_contract(),
            skill_context=services.skill_hint(),
            tool_output_policy=services.task_runtime_prompt_contract(),
            task_profile=services.evaluator_task_profile(),
        )
    )
    first_request = build_first_user_prompt(
        prompt_context.task_description,
        wall_clock_budget_sec=prompt_context.wall_clock_budget_sec,
        seed=prompt_context.seed,
        worker_id=prompt_context.worker_id,
        parallel_worker_snapshot=prompt_context.parallel_worker_facts,
        resource_context=prompt_context.resource_observation,
        skill_hint=prompt_context.skill_context,
        evaluator_contract=prompt_context.gate_constraints,
        task_profile=prompt_context.task_profile,
        task_runtime_contract=prompt_context.tool_output_policy,
        initial_workspace_state=prompt_context.workspace_facts,
    )
    load_existing = (spec.memory_dir / "ScienceAgent").is_dir()
    agent = services.make_agent(load_existing)
    stop_reason = "budget_expired"
    retain_agent = False
    try:
        request: str | None = None if load_existing else first_request
        resume_early_out: str | None = None
        post_budget_stage_commit_runs = 0
        if load_existing:
            resume_result = await resume_loaded_agent_from_memory(
                agent,
                effective_max=spec.max_steps,
                initial_max=spec.max_steps,
            )
            services.append_event(
                "lhr_resume_events.jsonl",
                {"event": "agent_memory_resume", **resume_result.to_event()},
            )
            resume_early_out = resume_result.early_out
        while (
            time.monotonic() < services.deadline()
            or services.pending_stage_commit_active()
        ):
            if (
                time.monotonic() >= services.deadline()
                and services.pending_stage_commit_active()
            ):
                post_budget_stage_commit_runs += 1
                if post_budget_stage_commit_runs > 3:
                    pending = services.pending_stage_commit()
                    pending_stage_id = (
                        pending.get("stage_id") if isinstance(pending, dict) else ""
                    )
                    pending_attempts = (
                        pending.get("attempts") if isinstance(pending, dict) else ""
                    )
                    services.append_event(
                        "lhr_stage_commit_events.jsonl",
                        {
                            "event": "stage_commit_text_post_budget_cap_reached",
                            "stage_id": pending_stage_id,
                            "attempts": pending_attempts,
                        },
                    )
                    break
                services.extend_stage_commit_deadline(agent)
            if resume_early_out is not None:
                out = resume_early_out
                resume_early_out = None
            else:
                out = await agent.run(request)
                services.accumulate_main_run_tokens(agent)
            _ = out
            request = None
            session_outcome = str(
                getattr(agent, "_scienceflow_worker_search_outcome", "") or ""
            )
            agent._scienceflow_worker_search_outcome = ""
            if session_outcome in {
                WorkerSearchOutcome.FINALIZE_CANDIDATE.value,
                WorkerSearchOutcome.STOP_NO_CANDIDATE.value,
            }:
                stop_reason = f"text_only_{session_outcome}"
                services.append_event(
                    "lhr_coordinator_events.jsonl",
                    {"event": "worker_outcome", "outcome": session_outcome},
                )
                break
            if services.evaluator_stop_requested():
                stop_reason = services.evaluator_stop_reason()
                stop_reason = stop_reason or "evaluator_query_budget_exhausted"
                break
            if services.pending_estra():
                services.append_event(
                    "lhr_coordinator_events.jsonl",
                    {
                        "event": "worker_outcome",
                        "outcome": WorkerSearchOutcome.STAGE_SWITCH.value,
                    },
                )
                await aclose_llm_clients(agent.llm)
                await services.restore_pending_estra()
                agent = services.make_agent(True)
                continue
            if (
                time.monotonic() >= services.deadline()
                and not services.pending_stage_commit_active()
            ):
                break
            services.append_event(
                "lhr_coordinator_events.jsonl",
                {
                    "event": "worker_outcome",
                    "outcome": WorkerSearchOutcome.CONTINUE_SEARCH.value,
                },
            )
            await asyncio.sleep(0.1)
        result = services.build_result(stop_reason)
        terminal_payload = {
            "best_metric": result.get("best_metric"),
            "outcome": result.get("outcome"),
            "stage_count": result.get("stage_count", 0),
            "stop_reason": result.get("stop_reason", stop_reason),
        }
        if result.get("status") == "success":
            services.lifecycle.mark_finished(terminal_payload)
        else:
            services.lifecycle.mark_failed(terminal_payload)
        if keep_agent_open:
            services.retain_live_agent(agent)
            retain_agent = True
        return result
    except BaseException as exc:
        try:
            services.lifecycle.mark_failed({"error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            logger.debug(
                "[lnr] state-machine failure status write failed", exc_info=True
            )
        raise
    finally:
        services.abandon_pending_stage_commit(agent)
        if not retain_agent:
            try:
                await aclose_llm_clients(agent.llm)
            except Exception:
                logger.debug("[lnr] llm close skipped", exc_info=True)
        metric_llm = services.metric_feedback_llm()
        if metric_llm is not None:
            try:
                await aclose_llm_clients(metric_llm)
            except Exception:
                logger.debug(
                    "[lnr] metric-validity feedback llm close skipped",
                    exc_info=True,
                )


__all__ = ["run_single_worker"]
