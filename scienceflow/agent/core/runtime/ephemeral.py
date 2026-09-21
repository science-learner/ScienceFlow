"""Thin ScienceFlow adapter for InquiryCraft ephemeral provider calls."""

from __future__ import annotations

from typing import Any, Mapping

from inquirycraft.adapters import AskLLMAdapter
from inquirycraft.llm import CompletionRequest, LLMClient, SameRoundRetryPolicy
from inquirycraft.memory import coerce_message
from inquirycraft.runtime import AgentRuntime, RuntimeOptions, StreamOutputGuard


def _message_row(message: Any) -> Mapping[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    return coerce_message(message).to_dict()


async def ask_tool_ephemeral(
    self: Any,
    *,
    llm: Any | None = None,
    render: bool = False,
    interaction_logger: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Run a non-conversation provider turn through InquiryCraft's LLM pipeline."""
    _ = render, interaction_logger
    target = llm or self.llm
    client = target if isinstance(target, LLMClient) else AskLLMAdapter(target)
    retry_max = max(0, int(getattr(self, "_stream_repetition_retry_max", 0) or 0))
    runtime = AgentRuntime(
        llm=client,
        options=RuntimeOptions(
            model=str(getattr(target, "model", "") or "scienceflow"),
            workspace=self._workspace_dir,
            timeout=float(kwargs.get("timeout") or self._llm_stream_timeout_sec),
            stream_llm=True,
        ),
        llm_retry_policy=SameRoundRetryPolicy(
            max_attempts=retry_max + 1,
            base_delay_sec=float(self._llm_tool_stream_retry_base_delay_sec),
            max_delay_sec=float(self._llm_tool_stream_retry_max_delay_sec),
        ),
        stream_guard=StreamOutputGuard(
            max_output_chars_soft=int(self._stream_max_output_chars_soft),
            repetition_detection=bool(self._stream_repetition_detection),
            repetition_window_chars=int(self._stream_repetition_window_chars),
            repetition_ngram_len=int(self._stream_repetition_ngram_len),
            repetition_max_repeats=int(self._stream_repetition_max_repeats),
        ),
    )
    request = CompletionRequest(
        messages=tuple(
            _message_row(message) for message in kwargs.get("messages") or ()
        ),
        model=runtime.options.model,
        tools=tuple(dict(tool) for tool in kwargs.get("tools") or ()),
        tool_choice=kwargs.get("tool_choice"),
        temperature=kwargs.get("temperature"),
        timeout=runtime.options.timeout,
        metadata={
            "system_msgs": kwargs.get("system_msgs"),
            "parallel_tool_calls": kwargs.get("parallel_tool_calls"),
            "message_objects": True,
            "force_tool_stream": True,
            "route": "ephemeral",
        },
    )
    return await runtime.complete_ephemeral(request)


__all__ = ["ask_tool_ephemeral"]
