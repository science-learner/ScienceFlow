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

"""ScienceAgent LLM trace: compact token totals + ask_tool_stream turn_kind / first_tool_name."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from inquirycraft.memory import Memory

from scienceflow.agent import ScienceAgent


class _FakeLLMStream:
    """ask_tool_stream with synthetic token stats on the LLM object."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self._last_call_input_tokens: int | None = None
        self._last_call_output_tokens: int | None = None
        self._last_call_input_cached_tokens: int | None = None
        self.model = "fake-model"

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        if not self._replies:
            raise RuntimeError("no fake replies left")
        msg = self._replies.pop(0)
        self._last_call_input_tokens = 100
        self._last_call_output_tokens = 7
        self._last_call_input_cached_tokens = 80
        content = getattr(msg, "content", None) or ""
        if content:
            await handle.put(content)
        handle.finish()
        return msg


class _CompactLLM:
    """Plain backend adapted by the unified runtime during compaction."""

    def __init__(self) -> None:
        self._last_call_input_tokens: int | None = None
        self._last_call_output_tokens: int | None = None
        self._last_call_input_cached_tokens: int | None = None
        self.model = "fake-compact"

    async def ask(self, **kwargs: Any) -> str:
        self._last_call_input_tokens = 42
        self._last_call_output_tokens = 18
        self._last_call_input_cached_tokens = 21
        return "ok summary body for handoff"


def _tool_msg(name: str, args: dict[str, Any], tc_id: str = "tc1") -> Any:
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id=tc_id,
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args),
                ),
            ),
        ],
        content="",
    )


@pytest.mark.asyncio
async def test_on_llm_call_ask_tool_stream_turn_qa(tmp_path: Path) -> None:
    payloads: list[dict[str, Any]] = []

    def hook(p: dict[str, Any]) -> None:
        payloads.append(dict(p))

    llm = _FakeLLMStream([SimpleNamespace(tool_calls=[], content="final answer")])
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=3,
        on_llm_call=hook,
    )
    agent._scienceflow_stage_id = "S02"
    agent._scienceflow_lineage_id = "L03"
    agent._scienceflow_node_uid = "W00:L03:S02"
    await agent.run("hi")
    ask_rows = [p for p in payloads if p.get("operation") == "ask_tool_stream"]
    assert len(ask_rows) == 1
    assert ask_rows[0].get("tokens_input") == 100
    assert ask_rows[0].get("tokens_output") == 7
    assert ask_rows[0].get("tokens_cached") == 80
    assert agent.last_run_tokens_cached == 80
    assert ask_rows[0].get("turn_kind") == "qa"
    assert ask_rows[0].get("first_tool_name") is None
    provider_rows = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_provider_calls.jsonl")
        .read_text()
        .splitlines()
    ]
    assert provider_rows[0]["usage_status"] == "known"
    assert provider_rows[0]["cache_rate"] == 0.8
    assert len(provider_rows[0]["stable_prefix_hash"]) == 64
    assert len(provider_rows[0]["provider_context_hash"]) == 64
    assert provider_rows[0]["provider_message_count"] > 0
    correlation = provider_rows[0]["correlation"]
    assert provider_rows[0]["call_id"] == correlation["call_id"]
    assert provider_rows[0]["stage_id"] == correlation["stage_id"] == "S02"
    assert provider_rows[0]["lineage_id"] == correlation["lineage_id"] == "L03"
    assert provider_rows[0]["node_uid"] == correlation["node_uid"] == "W00:L03:S02"

    journal_path = next(
        (tmp_path / ".logs" / "agent_runtime_operations").glob("*.jsonl")
    )
    operations = [json.loads(line) for line in journal_path.read_text().splitlines()]
    llm_intent = next(
        row for row in operations if row["kind"] == "llm" and row["phase"] == "intent"
    )
    assert llm_intent["payload"]["correlation"] == correlation

    events = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "agent_runtime_events.jsonl")
        .read_text()
        .splitlines()
    ]
    requested = next(row for row in events if row["type"] == "llm.requested")
    assert requested["payload"]["correlation"] == correlation


@pytest.mark.asyncio
async def test_on_llm_call_ask_tool_stream_turn_tool_and_name(tmp_path: Path) -> None:
    payloads: list[dict[str, Any]] = []

    def hook(p: dict[str, Any]) -> None:
        payloads.append(dict(p))

    llm = _FakeLLMStream(
        [
            _tool_msg("bash", {"command": "echo hi"}),
            SimpleNamespace(tool_calls=[], content="done"),
        ],
    )
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=5,
        on_llm_call=hook,
    )
    await agent.run("go")
    ask_rows = [p for p in payloads if p.get("operation") == "ask_tool_stream"]
    assert len(ask_rows) >= 1
    assert ask_rows[0].get("turn_kind") == "tool"
    assert ask_rows[0].get("first_tool_name") == "bash"


@pytest.mark.asyncio
async def test_compact_accumulates_last_run_tokens(tmp_path: Path) -> None:
    """After main run + compact(), last_run_* includes compact LLM tokens."""
    stream = _FakeLLMStream([SimpleNamespace(tool_calls=[], content="done")])
    agent = ScienceAgent(
        llm=stream,
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=2,
    )
    await agent.run("x")
    assert agent.last_run_compact_tokens_in == 0
    in0 = agent.last_run_tokens_in
    out0 = agent.last_run_tokens_out
    cached0 = agent.last_run_tokens_cached
    calls0 = agent.last_run_llm_calls
    agent.llm = _CompactLLM()
    out = await agent.compact()
    assert "Compacted" in out
    assert agent.last_run_tokens_in == in0 + 42
    assert agent.last_run_tokens_out == out0 + 18
    assert agent.last_run_tokens_cached == cached0 + 21
    assert agent.last_run_llm_calls == calls0 + 1
    assert agent.last_run_compact_tokens_in == 42
    assert agent.last_run_compact_tokens_out == 18
    assert agent.last_run_compact_tokens_cached == 21
    assert agent.last_run_compact_llm_calls == 1
