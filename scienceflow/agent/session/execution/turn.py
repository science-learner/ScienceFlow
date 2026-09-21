"""Turn routing, tool-bundle and exhaustion policy ownership."""

from __future__ import annotations

import inspect
import re
from typing import Any

from inquirycraft.memory import Message, coerce_tool_call
from inquirycraft.runtime import (
    ExhaustionDecision,
    ExhaustionRequest,
    ToolChoiceRequest,
    TurnDecision,
    TurnDecisionRequest,
)

from scienceflow.agent.core.ports.callback_ports import resolve_agent_callback
from scienceflow.agent.core.runtime.run_policy import RoundContext
from scienceflow.runtime.observability.agent_io.interaction_log import (
    truncate_for_interaction_log,
)


def _looks_like_unparsed_tool_call(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    markers = (
        "</tool_call>",
        "<tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    )
    if any(marker in lowered for marker in markers):
        return True
    return bool(re.match(r"^(?:→|->)\s*[a-z_][\w.-]*\s*\{", raw.lstrip(), re.I))


class ScienceFlowTurnPolicy:
    """Own decisions made after an LLM turn without owning runtime lifecycle."""

    def __init__(self, session: Any) -> None:
        self.session = session

    @property
    def host(self) -> Any:
        return self.session.host

    def tool_choice(self, request: ToolChoiceRequest):
        request.context.metadata["system_msgs"] = (
            self.host._host_ports.build_system_messages()
        )
        if request.turn == 1 and self.session.spec.first_round_tool_choice is not None:
            return self.session.spec.first_round_tool_choice
        if bool(getattr(self.host, "_lnr_transient_tool_choice_none", False)):
            return "none"
        return self.host._host_ports.tool_choice_for_main_loop(
            request.turn - 1, self.session.spec.request
        )

    async def decide(self, request: TurnDecisionRequest) -> TurnDecision:
        callback = resolve_agent_callback(self.host, "text_only_decision")
        stage_commit_pending = bool(
            getattr(self.host, "_lnr_stage_commit_text_pending", False)
        )
        if stage_commit_pending and callable(callback):
            text = request.result.content.strip() or "(empty assistant message)"
            callback_output = callback(
                agent=self.host,
                assistant_text=text,
                round_idx=request.context.turn_id - 1,
                max_steps=self.session.control.effective_max_steps,
            )
            if inspect.isawaitable(callback_output):
                callback_output = await callback_output
            if callback_output:
                return TurnDecision(
                    action="terminate",
                    output=str(callback_output),
                    reason="scienceflow_text_only_callback",
                    route="text_only_callback",
                )
            return TurnDecision(
                action="continue",
                reason=(
                    "scienceflow_stage_commit_retry"
                    if str(
                        getattr(self.host, "_lnr_transient_user_prompt", "") or ""
                    ).strip()
                    else "scienceflow_stage_commit_handled"
                ),
                route="stage_commit_retry",
                suppress_assistant=True,
            )
        if request.result.tool_calls:
            self.host._scienceflow_unparsed_tool_call_retries = 0
            mode, calls = self.host._host_ports.normalize_tool_calls(
                [coerce_tool_call(call) for call in request.result.tool_calls]
            )
            if mode == "blocked_write_edit_bundle":
                names = ", ".join(call.function.name or "?" for call in calls)
                self.host._log_info(
                    "[tool-calls] write/edit bundle blocked; "
                    "no tools executed; names=%s",
                    names,
                )
                return TurnDecision(
                    action="continue",
                    reason="blocked_write_edit_bundle",
                    suppress_assistant=True,
                    inject_messages=(
                        Message.user_message(
                            "[Guard] Multiple tool calls were returned in one turn "
                            "and at least one was `write` or `edit`. No tools were "
                            "executed. Retry with exactly one `write` or exactly one "
                            "`edit` as the sole tool call, then wait for its result "
                            "before the next file operation."
                        ),
                    ),
                )
            return TurnDecision()
        text = request.result.content.strip() or "(empty assistant message)"
        if _looks_like_unparsed_tool_call(text):
            retries = (
                int(
                    getattr(self.host, "_scienceflow_unparsed_tool_call_retries", 0)
                    or 0
                )
                + 1
            )
            self.host._scienceflow_unparsed_tool_call_retries = retries
            if retries <= 2:
                return TurnDecision(
                    action="continue",
                    reason="scienceflow_unparsed_tool_call_retry",
                    route="unparsed_tool_call_retry",
                    suppress_assistant=True,
                    inject_messages=(
                        Message.user_message(
                            "[Tool-call recovery] Your previous response contained "
                            "unparsed tool-call markup, so no tool ran. Retry the next "
                            "action as one valid structured tool call. Do not print XML "
                            "or tool-call tags as assistant text."
                        ),
                    ),
                )
            self.host._scienceflow_worker_search_outcome = (
                "finalize_candidate"
                if str(request.context.metadata.get("stage_id") or "draft") != "draft"
                else "stop_no_candidate"
            )
            return TurnDecision(
                action="terminate",
                output=text,
                reason="scienceflow_unparsed_tool_call_exhausted",
                route="unparsed_tool_call_exhausted",
            )
        if callable(callback):
            callback_output = callback(
                agent=self.host,
                assistant_text=text,
                round_idx=request.context.turn_id - 1,
                max_steps=self.session.control.effective_max_steps,
            )
            if inspect.isawaitable(callback_output):
                callback_output = await callback_output
            if callback_output:
                return TurnDecision(
                    action="terminate",
                    output=str(callback_output),
                    reason="scienceflow_text_only_callback",
                    route="text_only_callback",
                )
            if (
                bool(getattr(self.host, "_lnr_stage_commit_text_handled", False))
                and str(
                    getattr(self.host, "_lnr_transient_user_prompt", "") or ""
                ).strip()
            ):
                return TurnDecision(
                    action="continue",
                    reason="scienceflow_stage_commit_retry",
                    route="stage_commit_retry",
                    suppress_assistant=True,
                )
        route_prompt = str(
            getattr(self.host, "_lnr_agentic_text_only_route_prompt", "") or ""
        ).strip()
        route_allowed = not bool(getattr(self.host, "_lnr_stage_commit_enabled", False))
        if not route_allowed:
            route_allowed = bool(
                getattr(self.host, "_lnr_last_valid_bare_run_is_local", False)
            ) and not bool(getattr(self.host, "_lnr_stage_commit_guard_failed", False))
        if route_prompt and route_allowed:
            self.host._log_info(
                "[assistant-text-only] %s (agentic route prompt; ephemeral)",
                truncate_for_interaction_log(text),
            )
            base_messages = list(self.session.runtime.conversation.messages)
            base_messages.append(
                Message.assistant_message(
                    text,
                    reasoning_content=request.result.reasoning_content or None,
                )
            )
            route_text = await self.host.run_ephemeral_agentic_route_prompt(
                route_prompt,
                trigger="text_only",
                base_messages=base_messages,
            )
            self.host._lnr_agentic_text_only_route_prompt_injected = True
            self.host._lnr_agentic_text_only_route_without_result_md = not (
                self.session.spec.workspace / "result.md"
            ).exists()
            self.host._log_info(
                "[agentic-route] %s", truncate_for_interaction_log(route_prompt)
            )
            output = (text + ("\n" + route_text if route_text else "")).strip()
            return TurnDecision(
                action="terminate",
                reason="scienceflow_agentic_text_route",
                route="agentic_text_only",
                suppress_assistant=True,
                output=output,
            )
        round_context = RoundContext(
            request.context.turn_id - 1,
            self.session.control.effective_max_steps,
            text,
            self.session.spec.workspace,
        )
        should_continue, injection = self.host._run_policy.on_text_only(round_context)
        if not should_continue:
            self.host._log_info("[assistant] %s", truncate_for_interaction_log(text))
            self.host._scienceflow_worker_search_outcome = (
                "finalize_candidate"
                if str(request.context.metadata.get("stage_id") or "draft") != "draft"
                else "stop_no_candidate"
            )
            return TurnDecision(action="terminate", reason="scienceflow_text_policy")
        messages = (Message.user_message(injection),) if injection else ()
        return TurnDecision(
            action="continue",
            reason="scienceflow_text_policy",
            inject_messages=messages,
        )

    def transform_calls(self, calls, _context):
        legacy = [coerce_tool_call(call) for call in calls]
        return tuple(
            coerce_tool_call(call).model_dump(exclude_none=True)
            for call in self.host._host_ports.normalize_repl_file_change_calls(legacy)
        )

    def bundle_mode(self, calls, _context):
        mode, _ = self.host._host_ports.normalize_tool_calls(
            [coerce_tool_call(call) for call in calls]
        )
        return mode

    def exhaustion(self, request: ExhaustionRequest) -> ExhaustionDecision:
        context = RoundContext(
            request.context.turn_id,
            self.session.control.effective_max_steps,
            request.final_content,
            self.session.spec.workspace,
        )
        count, prompt = self.host._run_policy.on_loop_exhausted(context)
        if count > 0 and prompt:
            self.session.control.effective_max_steps += int(count)
            return ExhaustionDecision(
                additional_turns=int(count),
                inject_messages=(Message.user_message(prompt),),
                reason="scienceflow_result_recovery",
            )
        if request.final_content:
            return ExhaustionDecision()
        message = (
            "[ScienceAgent] Stopped after "
            f"{self.session.control.effective_max_steps} LLM rounds."
        )
        self.host._log_info(
            "[limit] Stopped after %d rounds",
            self.session.control.effective_max_steps,
        )
        return ExhaustionDecision(output=message, reason="scienceflow_round_limit")


__all__ = ["ScienceFlowTurnPolicy"]
