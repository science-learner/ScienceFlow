from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from inquirycraft.memory import Memory

from scienceflow.agent import ScienceAgent


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class _RecordedLLM:
    model = "recorded-model"

    def __init__(self, replies: list[Any] | None = None) -> None:
        self._replies = replies or [
            SimpleNamespace(
                content="",
                tool_calls=[_tool_call("call_write", "write", {"path": "answer.txt", "content": "42\n"})],
            ),
            SimpleNamespace(content="finished", tool_calls=[]),
        ]
        self.requests: list[dict[str, Any]] = []
        self._last_call_input_tokens = 10
        self._last_call_output_tokens = 2
        self._last_call_input_cached_tokens = 0

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        self.requests.append({key: value for key, value in kwargs.items() if key != "handle"})
        reply = self._replies.pop(0)
        if reply.content:
            await handle.put(reply.content)
        handle.finish()
        return reply


def _message_dump(memory: Memory) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json") for message in memory.messages]


def _stable_telemetry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in {"duration_sec", "ttft_sec", "tpot_ms"}}
        for row in rows
    ]


async def _run_session(root: Path, label: str) -> tuple[Any, ...]:
    workspace = root / label
    llm = _RecordedLLM()
    memory = Memory(max_messages=50)
    agent = ScienceAgent(
        llm=llm,
        memory=memory,
        workspace_dir=workspace,
        max_steps=3,
    )
    output = await agent.run("write the answer")
    files = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file() and ".logs" not in path.parts
    }
    return output, llm.requests, _message_dump(memory), files


async def _run_scripted_session(
    root: Path,
    label: str,
    replies: list[Any],
    *,
    initial_files: dict[str, str] | None = None,
) -> tuple[Any, ...]:
    workspace = root / label
    workspace.mkdir(parents=True)
    for relative, content in (initial_files or {}).items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    llm = _RecordedLLM(replies)
    telemetry: list[dict[str, Any]] = []
    memory = Memory(max_messages=50)
    agent = ScienceAgent(
        llm=llm,
        memory=memory,
        workspace_dir=workspace,
        max_steps=3,
        on_llm_call=lambda row: telemetry.append(dict(row)),
    )
    output = await agent.run("execute the recorded turn")
    files = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file() and ".logs" not in path.parts
    }
    mechanism = {
        "telemetry": _stable_telemetry(telemetry),
        "roles": [message.role for message in memory.messages],
        "state": agent.state.value,
        "round": agent._current_round,
    }
    return output, llm.requests, _message_dump(memory), mechanism, files


@pytest.mark.asyncio
async def test_inquirycraft_runtime_is_deterministic_for_provider_memory_and_workspace(
    tmp_path: Path,
) -> None:
    first = await _run_session(tmp_path, "first")
    second = await _run_session(tmp_path, "second")
    assert second == first


@pytest.mark.asyncio
async def test_runtime_replay_preserves_parallel_readonly_bundle(tmp_path: Path) -> None:
    replies = [
        SimpleNamespace(
            content="",
            tool_calls=[
                _tool_call("call_a", "read", {"path": "a.txt"}),
                _tool_call("call_b", "read", {"path": "b.txt"}),
            ],
        ),
        SimpleNamespace(content="both read", tool_calls=[]),
    ]
    first = await _run_scripted_session(
        tmp_path,
        "first",
        list(replies),
        initial_files={"a.txt": "alpha\n", "b.txt": "beta\n"},
    )
    second = await _run_scripted_session(
        tmp_path,
        "second",
        list(replies),
        initial_files={"a.txt": "alpha\n", "b.txt": "beta\n"},
    )
    assert second == first


class _RetryRecordedLLM(_RecordedLLM):
    def __init__(self) -> None:
        super().__init__([SimpleNamespace(content="retry complete", tool_calls=[])])
        self.attempts = 0

    async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        if self.attempts == 1:
            self.requests.append({key: value for key, value in kwargs.items() if key != "handle"})
            handle.finish()
            raise ValueError("Empty response from streaming tool LLM")
        return await super().ask_tool_stream(handle=handle, **kwargs)


async def _run_retry_session(root: Path, label: str) -> tuple[Any, ...]:
    workspace = root / label
    llm = _RetryRecordedLLM()
    telemetry: list[dict[str, Any]] = []
    memory = Memory(max_messages=20)
    agent = ScienceAgent(
        llm=llm,
        memory=memory,
        workspace_dir=workspace,
        max_steps=2,
        llm_tool_stream_max_attempts=2,
        llm_tool_stream_retry_base_delay_sec=0.0,
        llm_tool_stream_retry_max_delay_sec=0.0,
        on_llm_call=lambda row: telemetry.append(dict(row)),
    )
    output = await agent.run("retry once")
    return output, llm.requests, _message_dump(memory), _stable_telemetry(telemetry), agent.state.value


@pytest.mark.asyncio
async def test_runtime_replay_preserves_same_round_retry(tmp_path: Path) -> None:
    first = await _run_retry_session(tmp_path, "first")
    second = await _run_retry_session(tmp_path, "second")
    assert second == first


@pytest.mark.asyncio
async def test_inquirycraft_cancellation_stops_before_provider_call(tmp_path: Path) -> None:
    llm = _RecordedLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=10),
        workspace_dir=tmp_path,
    )
    agent._agent_runtime_cancellation.cancel()
    with pytest.raises(BaseException) as caught:
        await agent.run("must not reach provider")
    assert type(caught.value).__name__ == "CancelledError"
    assert llm.requests == []
    assert agent.state.value == "IDLE"


def test_runtime_backend_option_is_removed() -> None:
    assert "runtime_backend" not in inspect.signature(ScienceAgent.__init__).parameters


@pytest.mark.asyncio
async def test_long_research_host_keeps_its_own_context_projection(tmp_path, monkeypatch):
    from scienceflow.agent.session.execution import runtime as bridge

    original = bridge.AgentRuntime
    policies = []

    def capture(**kwargs):
        runtime = original(**kwargs)
        policies.append((runtime._context_pipeline.compaction.enabled,
                         bool(kwargs.get('context_transformers'))))
        return runtime

    monkeypatch.setattr(bridge, 'AgentRuntime', capture)
    await _run_session(tmp_path, 'context-policy')
    assert policies
    assert all(not compacted and projected for compacted, projected in policies)
