# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""LNR coordinator responsibility: resource, admission, and observer bridge construction.

Function bodies are mechanically moved from the frozen V3 baseline.
They keep the coordinator instance as their explicit state/port argument and
do not retain a host back-reference or duplicate domain state.
"""

from __future__ import annotations

from scienceflow.research.control.ephemeral_agent_session import (
    configure_resource_agent_audit,
    llm_correlation,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.shared import (
    INLINE_RESOURCE_ADVISORY_MODE,
    Any,
    Message,
    _WallClockAutoContinuePolicy,
    build_inline_resource_advisory_prompt,
    build_resource_admission_prompt,
    build_resource_arbiter_prompt,
    callback_ports_for,
    fallback_policy_decision,
    install_callback_ports,
    normalize_arbiter_decision,
    normalize_inline_resource_advisory,
    parse_arbiter_decision_text,
    parse_inline_resource_advisory_response,
    safe_inline_resource_advisory_boundary,
    time,
)


def _make_resource_arbiter_decider(self):
    if not bool(getattr(self.lhr, "resource_arbiter_enabled", False)):
        return None
    mode = (
        str(getattr(self.lhr, "resource_arbiter_mode", "policy") or "policy")
        .strip()
        .lower()
    )
    if mode != "llm":
        return None

    async def _decide(proposal: dict[str, Any]) -> dict[str, Any]:
        try:
            agent = self._resource_arbiter_agent
            if agent is None:
                hook = self.orchestrator.make_llm_call_tracer(
                    node_id="resource_arbiter",
                    process_id=f"{getattr(self.cfg, 'exp_id', '') or 'lhr'}:resource_arbiter",
                    detail_prefix="mode=lnr;role=resource_arbiter;llm_role=feedback",
                )
                agent = self._agent_factory_service().create(
                    "resource_arbiter",
                    correlation={"worker_id": getattr(self, "worker_id", "") or "W00"},
                    task_description=None,
                    run_policy=_WallClockAutoContinuePolicy(
                        deadline_monotonic=self.deadline, max_text_only_retries=0
                    ),
                    system_prompt=(
                        "You are Resource Arbiter. You are isolated from the main science agent. "
                        "Return JSON only. Do not call tools. Decide resource kill proposals using evidence. "
                        "Recent metric, checkpoint, submission, or training-progress stdout supports DENY_KILL; CPU busy, log churn, or artifact mtime alone does not, especially when declared GPU work is observed CPU-only while assigned GPU is idle."
                    ),
                    append_repl_system_prompt=False,
                    pin_task_description=False,
                    max_steps_override=1,
                    on_llm_call=hook,
                    memory_dir_override=self.global_log_dir
                    / "resource"
                    / "arbiter_memory",
                    load_existing_memory=False,
                    memory_agent_name="ResourceArbiter",
                    teleport_mode="off",
                    repl_bash_write_mode=False,
                    stable_system_prompt=True,
                    pin_environment_context=False,
                    workspace_git_enabled=False,
                    workspace_git_auto_checkpoint=False,
                    bash_observation_summary_override=True,
                    llm_stage_override=(
                        self._worker_llm_stage_override("feedback")
                        or self.cfg.agent.feedback
                    ),
                )
                self._resource_arbiter_agent = agent
            configure_resource_agent_audit(self, agent, "resource-arbiter")
            prompt = build_resource_arbiter_prompt(proposal)
            text = await agent.run_ephemeral_agentic_route_prompt(
                prompt,
                trigger="resource_arbiter",
                base_messages=[],
                llm_role="feedback",
            )
            parsed = parse_arbiter_decision_text(text)
            if not parsed:
                parsed = fallback_policy_decision(proposal)
                parsed["reason"] = (
                    "arbiter LLM returned unparsable output; used conservative fallback"
                )
            decision = normalize_arbiter_decision(
                parsed, proposal=proposal, source="resource_arbiter_llm"
            )
            decision["llm_correlation"] = llm_correlation(text)
            return decision
        except Exception as exc:
            fallback = fallback_policy_decision(proposal)
            fallback["action"] = "OBSERVE_MORE"
            fallback["reason"] = f"arbiter LLM failed: {type(exc).__name__}"
            fallback["confidence"] = "low"
            return normalize_arbiter_decision(
                fallback, proposal=proposal, source="llm_error_fallback"
            )

    return _decide


@staticmethod
def _resource_advisory_prompt(proposal: dict[str, Any]) -> str:
    return build_inline_resource_advisory_prompt(proposal)


@staticmethod
def _drop_dangling_tool_call_tail(messages: list[Any]) -> list[Any]:
    out = list(messages or [])
    if not out:
        return out
    tail = out[-1]
    role = str(getattr(tail, "role", "") or "").lower()
    tool_calls = getattr(tail, "tool_calls", None) or []
    if role == "assistant" and tool_calls:
        return out[:-1]
    return out


async def _direct_resource_main_agent_advisory(
    self,
    *,
    agent: Any,
    proposal: dict[str, Any],
    prompt: str,
    safe: bool,
    unsafe_reason: str,
) -> dict[str, Any]:
    t0 = time.time()
    blocked_state = not safe
    try:
        base_messages = list(
            getattr(agent, "_resource_advisory_base_messages", []) or []
        )
        if not base_messages and hasattr(agent, "_memory_ctx"):
            base_messages = self._drop_dangling_tool_call_tail(
                agent._memory_ctx.build_messages_for_llm()
            )
        else:
            base_messages = self._drop_dangling_tool_call_tail(base_messages)
        system_msgs = list(
            getattr(agent, "_resource_advisory_base_system_msgs", []) or []
        )
        if not system_msgs and hasattr(agent, "_build_system_messages"):
            system_msgs = list(agent._build_system_messages())
        system_msgs.append(
            Message.system_message(
                "You are answering a resource advisory for your own current command. "
                "Return exactly one RESOURCE_ADVISORY_RESPONSE block. "
                "Do not call tools, write files, or continue experiments."
            )
        )
        text = await agent.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="resource_main_agent_advisory",
            base_messages=base_messages,
            system_messages=system_msgs,
            timeout=float(
                getattr(self.lhr, "resource_main_agent_advisory_timeout_sec", 60.0)
                or 60.0
            ),
            llm_role="code",
        )
        correlation = llm_correlation(text)
        text = str(text or "").strip()
        advisory, block_text, parse_reason = parse_inline_resource_advisory_response(
            text
        )
        if parse_reason:
            advisory = normalize_inline_resource_advisory(
                {
                    "preference": "unknown",
                    "confidence": "low",
                    "reason": parse_reason,
                }
            )
        advisory.update(
            {
                "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                "advisory_status": "captured"
                if not parse_reason
                else "captured_with_parse_issue",
                "blocked_state": blocked_state,
                "memory_edit_applied": True,
                "memory_edit_removed_tokens": 0,
                "advisory_tool_call_rejected": False,
            }
        )
        advisory["_audit"] = {
            "event_version": 1,
            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
            "status": advisory["advisory_status"],
            "blocked_state": blocked_state,
            "defer_reason": unsafe_reason
            if blocked_state
            else "cache_friendly_direct_advisory",
            "proposal_id": proposal.get("proposal_id"),
            "proposal_type": proposal.get("proposal_type"),
            "reason_code": proposal.get("reason_code"),
            "request": prompt[:6000],
            "raw_response": str(text or "")[:6000],
            "parsed_response": {
                k: v for k, v in advisory.items() if not str(k).startswith("_")
            },
            "block_text": block_text[:2000],
            "parse_reason": parse_reason,
            "tool_call_rejected": False,
            "memory_edit_applied": True,
            "memory_edit_removed_tokens": 0,
            "final_feedback_emitted": False,
            "duration_sec": round(time.time() - t0, 3),
            "llm_correlation": correlation,
        }
        return advisory
    except Exception as exc:
        return {
            "preference": "unknown",
            "confidence": "low",
            "reason": f"main_agent_resource_advisory_error:{type(exc).__name__}",
            "ttl_sec": 120,
            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
            "advisory_status": "error",
            "blocked_state": blocked_state,
            "_audit": {
                "event_version": 1,
                "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                "status": "error",
                "blocked_state": blocked_state,
                "defer_reason": unsafe_reason
                if blocked_state
                else "cache_friendly_direct_advisory",
                "proposal_id": proposal.get("proposal_id"),
                "proposal_type": proposal.get("proposal_type"),
                "request": prompt[:6000],
                "error": type(exc).__name__,
                "duration_sec": round(time.time() - t0, 3),
            },
        }


async def _capture_inline_resource_advisory(
    self,
    *,
    proposal: dict[str, Any],
    prompt: str,
    result_box: dict[str, Any],
    agent: Any,
    assistant_text: str,
    round_idx: int,
    max_steps: int,
) -> str:
    _ = round_idx, max_steps
    tool_rejected = bool(
        getattr(agent, "_lnr_resource_advisory_tool_call_rejected", False)
    )
    if tool_rejected:
        advisory = normalize_inline_resource_advisory(
            {
                "preference": "unknown",
                "confidence": "low",
                "reason": "advisory_tool_call_rejected",
            }
        )
        block_text = ""
        parse_reason = "tool_call_rejected"
    else:
        advisory, block_text, parse_reason = parse_inline_resource_advisory_response(
            assistant_text
        )
        if parse_reason:
            advisory = normalize_inline_resource_advisory(
                {
                    "preference": "unknown",
                    "confidence": "low",
                    "reason": parse_reason,
                }
            )
    advisory.update(
        {
            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
            "memory_edit_applied": True,
            "memory_edit_removed_tokens": 0,
            "advisory_tool_call_rejected": tool_rejected,
        }
    )
    advisory["_audit"] = {
        "event_version": 1,
        "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
        "status": "captured" if not parse_reason else "captured_with_parse_issue",
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type"),
        "reason_code": proposal.get("reason_code"),
        "request": prompt[:6000],
        "raw_response": str(assistant_text or "")[:6000],
        "parsed_response": {
            k: v for k, v in advisory.items() if not str(k).startswith("_")
        },
        "block_text": block_text[:2000],
        "parse_reason": parse_reason,
        "tool_call_rejected": tool_rejected,
        "memory_edit_applied": True,
        "memory_edit_removed_tokens": 0,
        "final_feedback_emitted": False,
    }
    result_box.clear()
    result_box.update(advisory)
    setattr(
        agent,
        "_lnr_suppress_current_text_only_memory",
        "resource_advisory_memory_edit",
    )
    setattr(agent, "_lnr_resource_advisory_text_handled", True)
    return "RESOURCE_ADVISORY_CAPTURED"


async def _decide_resource_main_agent_advisory(
    self, proposal: dict[str, Any]
) -> dict[str, Any]:
    agent = self._resource_main_agent_ref
    mode = str(
        getattr(self.lhr, "resource_advisory_mode", INLINE_RESOURCE_ADVISORY_MODE)
        or INLINE_RESOURCE_ADVISORY_MODE
    )
    if mode != INLINE_RESOURCE_ADVISORY_MODE:
        return {
            "preference": "unknown",
            "confidence": "low",
            "reason": f"unsupported_resource_advisory_mode:{mode}",
            "ttl_sec": 300,
            "advisory_mode": mode,
        }
    if agent is None or getattr(agent, "llm", None) is None:
        return {
            "preference": "unknown",
            "confidence": "low",
            "reason": "main_agent_context_unavailable",
            "ttl_sec": 300,
            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
        }
    safe, unsafe_reason = safe_inline_resource_advisory_boundary(agent)
    prompt = self._resource_advisory_prompt(proposal)
    if hasattr(getattr(agent, "llm", None), "ask_tool_stream"):
        return await _direct_resource_main_agent_advisory(
            self,
            agent=agent,
            proposal=proposal,
            prompt=prompt,
            safe=safe,
            unsafe_reason=unsafe_reason,
        )
    if not safe:
        return {
            "preference": "unknown",
            "confidence": "low",
            "reason": f"inline_resource_advisory_deferred:{unsafe_reason}",
            "ttl_sec": 120,
            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
            "advisory_status": "deferred",
            "_audit": {
                "event_version": 1,
                "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                "status": "deferred",
                "defer_reason": unsafe_reason,
                "proposal_id": proposal.get("proposal_id"),
                "proposal_type": proposal.get("proposal_type"),
            },
        }

    t0 = time.time()
    previous_ports = callback_ports_for(agent)
    result_box: dict[str, Any] = {}

    async def _inline_resource_advisory_callback(
        *, agent: Any, assistant_text: str, round_idx: int, max_steps: int
    ) -> str:
        return await _capture_inline_resource_advisory(
            self,
            proposal=proposal,
            prompt=prompt,
            result_box=result_box,
            agent=agent,
            assistant_text=assistant_text,
            round_idx=round_idx,
            max_steps=max_steps,
        )

    try:
        install_callback_ports(
            agent,
            previous_ports.with_overrides(
                text_only_decision=_inline_resource_advisory_callback,
            ),
        )
        setattr(agent, "_lnr_transient_user_prompt", prompt)
        setattr(agent, "_lnr_transient_tool_choice_none", False)
        setattr(agent, "_lnr_transient_turn_kind", "inline_resource_advisory")
        setattr(agent, "_lnr_resource_advisory_text_pending", True)
        setattr(agent, "_lnr_resource_advisory_text_handled", False)
        setattr(agent, "_lnr_resource_advisory_tool_call_rejected", False)
        await agent.run(None)
        if result_box:
            return dict(result_box)
        return {
            "preference": "unknown",
            "confidence": "low",
            "reason": "inline_resource_advisory_empty_response",
            "ttl_sec": 300,
            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
            "_audit": {
                "event_version": 1,
                "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                "status": "empty_response",
                "proposal_id": proposal.get("proposal_id"),
                "proposal_type": proposal.get("proposal_type"),
                "request": prompt[:6000],
                "memory_edit_applied": True,
            },
        }
    except Exception as exc:
        return {
            "preference": "unknown",
            "confidence": "low",
            "reason": f"main_agent_advisory_error:{type(exc).__name__}",
            "ttl_sec": 300,
            "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
            "_audit": {
                "event_version": 1,
                "advisory_mode": INLINE_RESOURCE_ADVISORY_MODE,
                "status": "error",
                "proposal_id": proposal.get("proposal_id"),
                "proposal_type": proposal.get("proposal_type"),
                "request": prompt[:6000],
                "error": type(exc).__name__,
                "duration_sec": round(time.time() - t0, 3),
            },
        }
    finally:
        try:
            install_callback_ports(agent, previous_ports)
        except Exception:
            pass
        for name, value in (
            ("_lnr_transient_user_prompt", ""),
            ("_lnr_transient_tool_choice_none", False),
            ("_lnr_transient_user_prompt_active", False),
            ("_lnr_transient_turn_kind", ""),
            ("_lnr_resource_advisory_text_pending", False),
            ("_lnr_resource_advisory_text_handled", False),
            ("_lnr_resource_advisory_tool_call_rejected", False),
            ("_lnr_suppress_current_text_only_memory", ""),
        ):
            try:
                setattr(agent, name, value)
            except Exception:
                pass


def _make_resource_main_agent_advisory_decider(self):
    if not bool(getattr(self.lhr, "resource_main_agent_advisory_enabled", False)):
        return None

    async def _decide(proposal: dict[str, Any]) -> dict[str, Any]:
        return await _decide_resource_main_agent_advisory(self, proposal)

    return _decide


def _make_resource_admission_decider(self):
    if not bool(getattr(self.lhr, "resource_admission_llm_enabled", False)):
        return None

    async def _decide(task_card: dict[str, Any], rule_result: dict[str, Any]) -> str:
        agent = self._resource_admission_agent
        if agent is None:
            hook = self.orchestrator.make_llm_call_tracer(
                node_id="resource_admission_arbiter",
                process_id=f"{getattr(self.cfg, 'exp_id', '') or 'lhr'}:resource_admission_arbiter",
                detail_prefix="mode=lnr;role=resource_admission_arbiter;llm_role=feedback",
            )
            agent = self._agent_factory_service().create(
                "resource_admission_arbiter",
                correlation={"worker_id": getattr(self, "worker_id", "") or "W00"},
                task_description=None,
                run_policy=_WallClockAutoContinuePolicy(
                    deadline_monotonic=self.deadline, max_text_only_retries=0
                ),
                system_prompt=(
                    "You are ResourceAdmissionArbiter. You are isolated from the main science agent. "
                    "Return JSON only. Do not call tools. Decide startup admission for GPU tasks using only "
                    "the provided ResourceTaskCard and rule result. You may approve RUN_NOW or OBSERVE_THEN_RUN "
                    "only when lease_grantable_by_llm is true; the runtime will then atomically acquire the lease."
                ),
                append_repl_system_prompt=False,
                pin_task_description=False,
                max_steps_override=1,
                on_llm_call=hook,
                memory_dir_override=self.global_log_dir
                / "resource"
                / "admission_memory",
                load_existing_memory=False,
                memory_agent_name="ResourceAdmissionArbiter",
                teleport_mode="off",
                repl_bash_write_mode=False,
                stable_system_prompt=True,
                pin_environment_context=False,
                workspace_git_enabled=False,
                workspace_git_auto_checkpoint=False,
                bash_observation_summary_override=True,
                llm_stage_override=(
                    self._worker_llm_stage_override("feedback")
                    or self.cfg.agent.feedback
                ),
            )
            self._resource_admission_agent = agent
        configure_resource_agent_audit(self, agent, "resource-admission-arbiter")
        prompt = build_resource_admission_prompt(task_card, rule_result)
        return await agent.run_ephemeral_agentic_route_prompt(
            prompt,
            trigger="resource_admission_arbiter",
            base_messages=[],
            llm_role="feedback",
        )

    return _decide
