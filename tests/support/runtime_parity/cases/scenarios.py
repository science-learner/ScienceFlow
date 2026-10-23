from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.support.runtime_parity.execution.capture import capture_case as _capture
from .full_scenarios import FULL_SCENARIOS


async def _scienceflow_tool_turn(workspace: Path) -> dict[str, Any]:
    from inquirycraft.memory import Memory
    from scienceflow.agent import ScienceAgent

    def tool_message(name: str, arguments: dict[str, Any], call_id: str) -> Any:
        return SimpleNamespace(
            content="",
            reasoning_content=f"call {name}",
            tool_calls=[
                SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(
                            arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    ),
                )
            ],
        )

    class ScriptedScienceLLM:
        model = "runtime-parity-model"
        _last_call_input_tokens = 11
        _last_call_output_tokens = 3
        _last_call_input_cached_tokens = 2

        def __init__(self) -> None:
            self.replies = [
                tool_message(
                    "write",
                    {"path": "parity.txt", "content": "scienceflow-parity\n"},
                    "call-write",
                ),
                tool_message("read", {"path": "parity.txt"}, "call-read"),
                SimpleNamespace(
                    content="SCIENCEFLOW_PARITY_OK",
                    reasoning_content="verified",
                    tool_calls=[],
                ),
            ]
            self.requests: list[dict[str, Any]] = []

        async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
            self.requests.append(
                {key: value for key, value in kwargs.items() if key != "handle"}
            )
            reply = self.replies.pop(0)
            if reply.content:
                await handle.put(reply.content)
            handle.finish()
            return reply

    llm = ScriptedScienceLLM()
    telemetry: list[dict[str, Any]] = []
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=50),
        workspace_dir=workspace,
        max_steps=5,
        parallel_bash_enabled=False,
        parallel_llm_tool_calls=False,
        interaction_log_color=False,
        on_llm_call=lambda row: telemetry.append(dict(row)),
    )
    answer = await agent.run(
        "Write parity.txt with exact content scienceflow-parity followed by a newline, "
        "read it, then answer SCIENCEFLOW_PARITY_OK."
    )
    messages = [message.model_dump() for message in agent.memory.messages]
    context = {
        "answer": answer,
        "requests": llm.requests,
        "messages": messages,
        "tool_schema": agent.availableTools.to_params(),
    }
    mechanism = {
        "telemetry": telemetry,
        "roles": [message.role for message in agent.memory.messages],
        "tool_names": [
            call.function.name
            for message in agent.memory.messages
            for call in (message.tool_calls or [])
        ],
        "state": str(agent.state.value),
        "round": int(agent._current_round),
    }
    return _capture("scienceflow_tool_turn", context, mechanism, workspace)


async def _scienceflow_retry_stop(workspace: Path) -> dict[str, Any]:
    from inquirycraft.memory import Memory
    from scienceflow.agent import ScienceAgent

    class RetryLLM:
        model = "runtime-parity-retry"
        _last_call_input_tokens = 1
        _last_call_output_tokens = 2
        _last_call_input_cached_tokens = 0

        def __init__(self) -> None:
            self.calls = 0
            self.requests: list[dict[str, Any]] = []

        async def ask_tool_stream(self, **kwargs: Any) -> Any:
            self.calls += 1
            self.requests.append(
                {key: value for key, value in kwargs.items() if key != "handle"}
            )
            handle = kwargs["handle"]
            handle.finish()
            if self.calls == 1:
                raise ValueError("Empty response from streaming tool LLM")
            return SimpleNamespace(
                content="RETRY_OK", reasoning_content="", tool_calls=[]
            )

    llm = RetryLLM()
    telemetry: list[dict[str, Any]] = []
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=20),
        workspace_dir=workspace,
        max_steps=2,
        llm_tool_stream_max_attempts=2,
        llm_tool_stream_retry_base_delay_sec=0.0,
        llm_tool_stream_retry_max_delay_sec=0.0,
        interaction_log_color=False,
        on_llm_call=lambda row: telemetry.append(dict(row)),
    )
    answer = await agent.run("Return RETRY_OK after retrying the empty stream.")
    return _capture(
        "scienceflow_retry_stop",
        {
            "answer": answer,
            "requests": llm.requests,
            "messages": [message.model_dump() for message in agent.memory.messages],
        },
        {
            "llm_calls": llm.calls,
            "telemetry": telemetry,
            "state": str(agent.state.value),
            "round": int(agent._current_round),
        },
        workspace,
    )


async def _scienceflow_text_turn(workspace: Path) -> dict[str, Any]:
    from inquirycraft.memory import Memory
    from scienceflow.agent import ScienceAgent

    class TextLLM:
        model = "runtime-parity-text"
        _last_call_input_tokens = 3
        _last_call_output_tokens = 1
        _last_call_input_cached_tokens = 0

        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        async def ask_tool_stream(self, *, handle: Any, **kwargs: Any) -> Any:
            self.requests.append(
                {key: value for key, value in kwargs.items() if key != "handle"}
            )
            await handle.put("TEXT_PARITY_OK")
            handle.finish()
            return SimpleNamespace(
                content="TEXT_PARITY_OK", reasoning_content="direct", tool_calls=[]
            )

    llm = TextLLM()
    agent = ScienceAgent(
        llm=llm,
        memory=Memory(max_messages=10),
        workspace_dir=workspace,
        max_steps=1,
        interaction_log_color=False,
    )
    answer = await agent.run("Return TEXT_PARITY_OK without calling a tool.")
    return _capture(
        "scienceflow_text_turn",
        {
            "answer": answer,
            "requests": llm.requests,
            "messages": [message.model_dump() for message in agent.memory.messages],
            "tool_schema": agent.availableTools.to_params(),
        },
        {
            "state": str(agent.state.value),
            "round": int(agent._current_round),
            "roles": [message.role for message in agent.memory.messages],
        },
        workspace,
    )


async def _deepcraft_runtime_tool_turn(workspace: Path) -> dict[str, Any]:
    from inquirycraft.events import JsonlEventSink, load_events
    from inquirycraft.events.replay import canonicalize_runtime_events, request_to_dict
    from inquirycraft.llm import CompletionRequest, CompletionResult, StreamChunk, Usage
    from inquirycraft.memory import Conversation, JsonlConversationStore
    from inquirycraft.runtime import AgentRuntime, RuntimeOptions
    from inquirycraft.tools import BaseTool, ToolCollection, ToolContext, ToolResult

    class ScriptedLLM:
        def __init__(self) -> None:
            self.calls = 0
            self.requests: list[CompletionRequest] = []

        async def complete(self, request: CompletionRequest) -> CompletionResult:
            self.calls += 1
            self.requests.append(request)
            if self.calls == 1:
                return CompletionResult(
                    reasoning_content="write deterministic output",
                    tool_calls=(
                        {
                            "id": "call-fixed",
                            "type": "function",
                            "function": {
                                "name": "write_fixture",
                                "arguments": '{"text":"stable"}',
                            },
                        },
                    ),
                    finish_reason="tool_calls",
                    usage=Usage(
                        input_tokens=11, output_tokens=4, cached_input_tokens=2
                    ),
                )
            return CompletionResult(
                content="DEEPCRAFT_PARITY_OK",
                finish_reason="stop",
                usage=Usage(input_tokens=17, output_tokens=1),
            )

        def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
            async def generate() -> AsyncIterator[StreamChunk]:
                yield StreamChunk(content="unused")

            return generate()

        async def aclose(self) -> None:
            return None

    class WriteFixture(BaseTool):
        name = "write_fixture"
        description = "write deterministic fixture"
        input_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        async def execute(
            self, arguments: Mapping[str, Any], context: ToolContext
        ) -> ToolResult:
            (context.workspace / "fixture.txt").write_text(
                str(arguments["text"]), encoding="utf-8"
            )
            return ToolResult(output="fixture.txt")

    events_path = workspace / "events.jsonl"
    session_path = workspace / "session.jsonl"
    llm = ScriptedLLM()
    runtime = AgentRuntime(
        llm=llm,
        options=RuntimeOptions(
            model="recorded-model",
            workspace=workspace,
            system_prompt="runtime parity system",
            max_turns=3,
            session_id="session-fixed",
            run_id="run-fixed",
            agent_id="agent-fixed",
        ),
        tools=ToolCollection((WriteFixture(),)),
        conversation=Conversation(),
        conversation_store=JsonlConversationStore(session_path),
        event_sink=JsonlEventSink(events_path),
    )
    answer = await runtime.run("write stable fixture")
    await runtime.aclose()
    return _capture(
        "deepcraft_runtime_tool_turn",
        {
            "answer": answer,
            "requests": [request_to_dict(request) for request in llm.requests],
            "conversation": runtime.conversation.to_openai(),
        },
        canonicalize_runtime_events(load_events(events_path)),
        workspace,
    )


async def _deepcraft_runtime_resume(workspace: Path) -> dict[str, Any]:
    import shutil

    from inquirycraft.events import RecordingLLMClient, ReplayLLMClient
    from inquirycraft.llm import CompletionRequest, CompletionResult, StreamChunk
    from inquirycraft.memory import Conversation
    from inquirycraft.runtime import AgentRuntime, RuntimeOptions
    from inquirycraft.tools import BaseTool, ToolCollection, ToolContext, ToolResult

    class ResumeLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: CompletionRequest) -> CompletionResult:
            self.calls += 1
            if self.calls == 1:
                return CompletionResult(
                    reasoning_content="write before interruption",
                    tool_calls=(
                        {
                            "id": "resume-call-fixed",
                            "type": "function",
                            "function": {
                                "name": "write_resume_fixture",
                                "arguments": '{"text":"resume-stable"}',
                            },
                        },
                    ),
                    finish_reason="tool_calls",
                )
            return CompletionResult(content="RESUME_PARITY_OK", finish_reason="stop")

        def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
            async def generate() -> AsyncIterator[StreamChunk]:
                yield StreamChunk(content="unused")

            return generate()

        async def aclose(self) -> None:
            return None

    class WriteResumeFixture(BaseTool):
        name = "write_resume_fixture"
        description = "write a resume fixture"
        input_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        async def execute(
            self, arguments: Mapping[str, Any], context: ToolContext
        ) -> ToolResult:
            (context.workspace / "resume.txt").write_text(
                str(arguments["text"]), encoding="utf-8"
            )
            return ToolResult(output="resume.txt")

    def options(root: Path) -> RuntimeOptions:
        return RuntimeOptions(
            model="resume-model",
            workspace=root,
            max_turns=3,
            system_prompt="resume parity system",
            session_id="resume-session",
            run_id="resume-run",
            agent_id="resume-agent",
        )

    recording = workspace / "recorded_llm.jsonl"
    initial_workspace = workspace / "initial_workspace"
    resumed_workspace = workspace / "resumed_workspace"
    tools = ToolCollection((WriteResumeFixture(),))
    baseline = AgentRuntime(
        llm=RecordingLLMClient(ResumeLLM(), recording),
        options=options(initial_workspace),
        tools=tools,
        conversation=Conversation(),
    )
    baseline_answer = await baseline.run("write fixture then finish")
    await baseline.aclose()
    resumed_workspace.mkdir(parents=True)
    shutil.copy2(initial_workspace / "resume.txt", resumed_workspace / "resume.txt")
    interrupted = Conversation(baseline.conversation.messages[:3])
    replay = ReplayLLMClient(recording, start_sequence=2)
    resumed = AgentRuntime(
        llm=replay,
        options=options(resumed_workspace),
        tools=tools,
        conversation=interrupted,
    )
    resumed_answer = await resumed.resume()
    replay.assert_exhausted()
    await resumed.aclose()
    return _capture(
        "deepcraft_runtime_resume",
        {
            "baseline_answer": baseline_answer,
            "resumed_answer": resumed_answer,
            "interrupted": interrupted.to_openai(),
            "baseline_conversation": baseline.conversation.to_openai(),
            "resumed_conversation": resumed.conversation.to_openai(),
        },
        {
            "recorded_calls": 2,
            "replay_consumed": replay.consumed,
            "replay_remaining": replay.remaining,
            "artifact_equal": (initial_workspace / "resume.txt").read_bytes()
            == (resumed_workspace / "resume.txt").read_bytes(),
        },
        workspace,
    )


async def _tool_dispatch_and_interrupt(workspace: Path) -> dict[str, Any]:
    from inquirycraft.tools import classify_tool_call_bundle, execute_concurrently_in_order

    def call(name: str, arguments: str = "{}") -> Any:
        return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))

    readonly = [call("read"), call("grep"), call("ls")]
    mutation = [call("read"), call("write")]
    trace: list[str] = []

    async def run(item: int) -> str:
        trace.append(f"start:{item}")
        await asyncio.sleep((4 - item) * 0.005)
        trace.append(f"finish:{item}")
        return f"result:{item}"

    ordered_results = await execute_concurrently_in_order([1, 2, 3], run)

    stream_trace: list[str] = []

    async def stream() -> AsyncIterator[str]:
        try:
            stream_trace.append("open")
            yield "first"
            stream_trace.append("second_requested")
            yield "second"
        finally:
            stream_trace.append("closed")

    generator = stream()
    first_chunk = await anext(generator)
    await generator.aclose()
    error_result = {
        "output": "partial",
        "error": "fixture failure",
        "system": "raw diagnostics",
    }
    return _capture(
        "tool_dispatch_and_interrupt",
        {
            "readonly_calls": readonly,
            "mutation_calls": mutation,
            "first_chunk": first_chunk,
            "error_result": error_result,
        },
        {
            "readonly_mode": classify_tool_call_bundle(
                readonly,
                readonly_tool_names={"read", "grep", "ls"},
                mutation_tool_names={"write", "edit"},
            ),
            "mutation_mode": classify_tool_call_bundle(
                mutation,
                readonly_tool_names={"read", "grep", "ls"},
                mutation_tool_names={"write", "edit"},
            ),
            "forced_mode": classify_tool_call_bundle(
                readonly,
                readonly_tool_names={"read", "grep", "ls"},
                mutation_tool_names={"write", "edit"},
                force_sequential=True,
            ),
            "execution_trace": trace,
            "ordered_results": ordered_results,
            "stream_trace": stream_trace,
            "error_render": "Error: fixture failure\npartial",
        },
        workspace,
    )


async def _performance_microbench(workspace: Path) -> dict[str, Any]:
    from inquirycraft.events import RuntimeEvent
    from inquirycraft.llm import CompletionRequest
    from inquirycraft.memory import Conversation
    from inquirycraft.memory.message import Message as RuntimeMessage
    from inquirycraft.tools import classify_tool_call_bundle

    readonly_calls = [
        SimpleNamespace(function=SimpleNamespace(name=name, arguments="{}"))
        for name in ("read", "grep", "ls")
    ]

    def context_build() -> None:
        CompletionRequest(
            messages=(
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ),
            model="performance-model",
            tools=({"type": "function", "function": {"name": "read"}},),
        )

    def memory_append_retrieve() -> None:
        conversation = Conversation()
        for index in range(20):
            conversation.append(RuntimeMessage(role="user", content=f"message-{index}"))
        conversation.to_openai()

    def tool_dispatch() -> None:
        classify_tool_call_bundle(
            readonly_calls,
            readonly_tool_names={"read", "grep", "ls"},
            mutation_tool_names={"write", "edit"},
        )

    def event_build() -> None:
        RuntimeEvent(
            type="benchmark.event",
            session_id="session-fixed",
            run_id="run-fixed",
            agent_id="agent-fixed",
            event_id="event-fixed",
            timestamp="2026-01-01T00:00:00Z",
            payload={"value": 1},
        ).to_dict()

    def measure(
        operation: Any, *, inner: int = 100, repeats: int = 11
    ) -> dict[str, float]:
        for _ in range(inner):
            operation()
        samples: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            for _ in range(inner):
                operation()
            samples.append((time.perf_counter_ns() - started) / inner / 1_000_000)
        ordered = sorted(samples)
        return {
            "median_ms": round(statistics.median(ordered), 6),
            "p95_ms": round(ordered[-1], 6),
        }

    metrics = {
        "context_build": measure(context_build),
        "event_build": measure(event_build),
        "memory_append_retrieve": measure(memory_append_retrieve, inner=30),
        "tool_dispatch": measure(tool_dispatch),
    }
    return _capture(
        "performance_microbench",
        {"warmup": 100, "repeats": 11, "operations": sorted(metrics)},
        {"metrics": metrics},
        workspace,
    )


async def _legacy_message_memory(workspace: Path) -> dict[str, Any]:
    from inquirycraft.memory import (
        Function,
        Memory,
        Message,
        ToolCall,
        dedupe_message_records,
        repair_message_record_window,
    )

    call = ToolCall(
        id="call-fixed",
        function=Function(name="read", arguments='{"path":"parity.txt"}'),
    )
    messages = [
        Message.system_message("system"),
        Message.user_message("用户输入"),
        Message(
            role="assistant",
            content="",
            reasoning_content="reasoning",
            tool_calls=[call],
        ),
        Message.tool_message("content", "read", "call-fixed"),
    ]
    memory = Memory(max_messages=10)
    for message in messages:
        memory.add_message(message)
    records = [
        {"uuid": "u1", "memory_record": {"message": messages[0].model_dump()}},
        {"uuid": "u1", "memory_record": {"message": messages[0].model_dump()}},
        {"uuid": "u2", "memory_record": {"message": messages[1].model_dump()}},
        {"uuid": "u3", "memory_record": {"message": messages[2].model_dump()}},
        {"uuid": "u4", "memory_record": {"message": messages[3].model_dump()}},
    ]
    deduped = dedupe_message_records(records)
    repaired = repair_message_record_window(deduped[2:], max_messages=10)
    return _capture(
        "legacy_message_memory",
        {
            "messages": [message.model_dump() for message in messages],
            "to_dict": [message.to_dict() for message in messages],
            "memory_messages": [message.model_dump() for message in memory.messages],
        },
        {"deduped": deduped, "repaired": repaired},
        workspace,
    )


async def _stage_gate_resource_decision(workspace: Path) -> dict[str, Any]:
    from scienceflow.foundation.contracts import EvalContext, MetricEvent
    from scienceflow.research.quality.gate import GateManager
    from scienceflow.runtime.safety.resource.lifecycle.gpu_sublease import plan_gpu_sublease
    from scienceflow.research.solver.lnr.lifecycle.stage.records.stage_ledger import (
        parse_stage_cards,
        validate_append_only_stage_commit,
    )

    stage_text = (
        "### S01 deterministic candidate\n"
        "metric: 1.25\n"
        "lower_is_better: false\n"
        "BRIEF: parity candidate\n"
        "WHY: deterministic runtime fixture\n"
        "FILES: artifacts/best_solution.json\n"
    )
    stage_path = workspace / "stage_ledger.md"
    stage_path.write_text(stage_text, encoding="utf-8")
    stage_valid, stage_reason = validate_append_only_stage_commit("", stage_text, "S01")
    stage_cards = parse_stage_cards(stage_text)

    context = EvalContext(
        task_profile="opt_solver",
        task_id="runtime-parity-task",
        task_root=workspace,
        workspace=workspace,
        worker_id="W00",
        stage_id="S01",
        cfg={
            "gate": {
                "policy": "optimization_feasibility",
                "params": {"max_constraint_violation": 1e-6},
            }
        },
    )

    def event(violation: float) -> MetricEvent:
        return MetricEvent(
            candidate_id="candidate-fixed",
            worker_id="W00",
            stage_id="S01",
            metric_value=1.25,
            metric_name="score",
            lower_is_better=False,
            validation_ok=True,
            candidate_ready=True,
            selection_eligible=True,
            metric_validity="high",
            metric_validity_reason_code="authoritative",
            artifact_path="artifacts/best_solution.json",
            artifact_sha="abc",
            evaluator_backend="artifact_command",
            evaluator_status="success",
            metric_type="optimization",
            task_profile="opt_solver",
            extra={"constraint_violation": violation},
        )

    manager = GateManager.default()
    decisions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for violation in (0.0, 1e-3):
        decision, trace = manager.decide_with_trace(
            context, event(violation), trigger="stage_end"
        )
        decisions.append(decision.to_dict())
        traces.append(trace)
    resource_plan = plan_gpu_sublease(
        candidate_gpu_ids=["0", "1"],
        default_request=1,
        max_request=2,
        command="torchrun --nproc_per_node=2 train.py",
    )
    return _capture(
        "stage_gate_resource_decision",
        {
            "context": context,
            "events": [event(0.0), event(1e-3)],
            "stage_text": stage_text,
            "resource_input": {
                "candidate_gpu_ids": ["0", "1"],
                "default_request": 1,
                "max_request": 2,
                "command": "torchrun --nproc_per_node=2 train.py",
            },
        },
        {
            "stage": {
                "valid": stage_valid,
                "reason": stage_reason,
                "cards": stage_cards,
            },
            "gate": {"decisions": decisions, "traces": traces},
            "resource": resource_plan.to_json(),
            "sequence": [
                "stage_validate",
                "gate_accept",
                "gate_reject",
                "resource_plan",
            ],
        },
        workspace,
    )


SCENARIOS = {
    "deepcraft_runtime_resume": _deepcraft_runtime_resume,
    "deepcraft_runtime_tool_turn": _deepcraft_runtime_tool_turn,
    "stage_gate_resource_decision": _stage_gate_resource_decision,
    "legacy_message_memory": _legacy_message_memory,
    "performance_microbench": _performance_microbench,
    "scienceflow_retry_stop": _scienceflow_retry_stop,
    "scienceflow_text_turn": _scienceflow_text_turn,
    "scienceflow_tool_turn": _scienceflow_tool_turn,
    "tool_dispatch_and_interrupt": _tool_dispatch_and_interrupt,
}

SCENARIOS.update(FULL_SCENARIOS)


def run_scenario(case_id: str, workspace: Path) -> dict[str, Any]:
    try:
        scenario = SCENARIOS[case_id]
    except KeyError as exc:
        raise ValueError(f"Unknown runtime parity case: {case_id}") from exc
    workspace.mkdir(parents=True, exist_ok=False)
    return asyncio.run(scenario(workspace))


__all__ = ["SCENARIOS", "run_scenario"]
