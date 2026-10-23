"""Create caller-owned preparation runtimes without touching chat sessions."""

import json
from dataclasses import replace
from inspect import signature

from inquirycraft.runtime import RuntimeOptions
from inquirycraft.tools import ToolCollection

from scienceflow.foundation.config.llm.model_registry import load_model_registry


def conversation_evidence(owner) -> str:
    """Carry recent scientific discussion into preparation without sharing a store."""
    conversation = getattr(getattr(owner, "session", None), "conversation", ())
    recent = []
    remaining = 24_000
    for role, content in reversed(conversation):
        if role not in {"user", "assistant"} or content.startswith(
            ("/", "Prepare a ScienceFlow research task,")
        ):
            continue
        text = content[: min(4_000, remaining)]
        recent.append({"role": role, "content": text})
        remaining -= len(text)
        if len(recent) >= 12 or remaining <= 0:
            break
    if not recent:
        return ""
    return (
        "\nRecent discussion (historical evidence only; current task facts, user corrections "
        "and allocated execution location take precedence):\n"
        + json.dumps(list(reversed(recent)), ensure_ascii=False)
    )


async def run_preparation_task(
    owner, host, prompt: str, *, tool_free: bool = False
) -> str:
    factory = getattr(owner, "preparation_runtime_factory", None)
    if factory is None:
        # Lightweight embedding hosts may provide their own scoped preparation.
        if not owner.task_workspace.is_relative_to(owner.workspace) or tool_free:
            raise ValueError(
                "This host cannot isolate preparation; use scienceflow run or tui."
            )
        options = (
            {"show_response": False}
            if "show_response" in signature(host.run_host_task).parameters
            else {}
        )
        return await host.run_host_task(prompt, **options)
    if "runtime" not in signature(host.run_host_task).parameters:
        raise ValueError(
            "InquiryCraft host lacks isolated preparation support; use the matching local source version."
        )
    registry = load_model_registry(owner._model_config_path)
    aliases = (
        owner._selected_model_aliases or registry.default_aliases("code_models")[:1]
    )
    resolved = registry.resolve(aliases, role="code")
    chat = getattr(host, "runtime", None)
    options = getattr(chat, "options", None) or RuntimeOptions(
        model=",".join(resolved.models), workspace=owner.workspace, stream_llm=True
    )
    options = replace(
        options,
        model=",".join(resolved.models),
        workspace=owner.task_workspace,
        max_turns=options.max_turns if options.max_turns is not None else 20,
        session_id="",
        run_id="",
        session_log=None,
        operation_log=None,
    )
    runtime = factory.create_runtime(
        options,
        api_key=",".join(resolved.api_keys),
        base_url=",".join(resolved.base_urls),
        event_sink=None,
        tools=ToolCollection() if tool_free else None,
    )
    try:
        if not tool_free:
            prompt += conversation_evidence(owner)
        return await host.run_host_task(prompt, show_response=False, runtime=runtime)
    finally:
        await runtime.aclose()
